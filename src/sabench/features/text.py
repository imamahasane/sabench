"""Transcript-based features: CHAT transcript parsing, Whisper ASR, and frozen language-model
embeddings (layer-wise, mean-pooled), so the E3 probe machinery and its search-aware null apply to
text exactly as they do to speech embeddings.

ADReSS ships manual CHAT transcripts; ADReSSo does not, so ASR transcripts are produced for both
datasets to keep the comparison like-for-like. ASR transcribes the whole recording (participant and
interviewer, no diarisation); manual transcripts keep participant (``*PAR``) turns only.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from sabench.audio.prepare import load_audio, resample, to_mono
from sabench.exceptions import FeatureExtractionError, ModelUnavailableError
from sabench.logging_config import get_logger

logger = get_logger(__name__)

_TARGET_SR = 16_000
_BULLET = re.compile(r"\x15[^\x15]*\x15")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_ANGLE = re.compile(r"[<>]")
_KEEP = re.compile(r"[a-z']+")


def resolve_device(device: str) -> str:
    """``auto`` -> CUDA, else Apple MPS, else CPU."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError as exc:
        raise ModelUnavailableError("torch is not installed. Install requirements-extra.txt.") from exc
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chat_participant_text(cha_path: str | Path) -> str:
    """Participant (``*PAR``) speech from a CHAT transcript as plain lower-case words.

    Drops timing bullets and bracketed annotation codes; keeps fillers (``&uh`` -> ``uh``) and
    retracings, since disfluency is part of the clinical signal text models are credited with.
    """
    turns: list[str] = []
    current: list[str] | None = None
    for line in Path(cha_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("*"):
            current = [line.split(":", 1)[1]] if line.startswith("*PAR:") else None
            if current is not None:
                turns.append("")
        elif line.startswith("\t") and current is not None:
            current.append(line)
        elif line.startswith(("%", "@")):
            current = None
        if current is not None:
            turns[-1] = " ".join(current)
    words: list[str] = []
    for turn in turns:
        cleaned = _ANGLE.sub(" ", _BRACKETS.sub(" ", _BULLET.sub(" ", turn.lower())))
        cleaned = cleaned.replace("(", "").replace(")", "")  # CHAT omitted sounds: sayin(g) -> saying
        words.extend(w for w in _KEEP.findall(cleaned.replace("&", " ")) if w != "'")
    return " ".join(words)


# Whisper's long-form decoding safeguards (the reference implementation's defaults, with the
# compression threshold on transformers' zlib scale): a segment whose text compresses too well
# (a repetition loop) or decodes with low confidence is re-decoded at the next temperature, and no
# segment is conditioned on the previous one's text, which is how loops propagate.
_ASR_GENERATE_KWARGS = {
    "task": "transcribe",
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_prev_tokens": False,
}


_TOKEN_KEY = re.compile(r"[^\w']")


def collapse_repeats(text: str, max_repeats: int = 3, max_n: int = 2) -> str:
    """Trim runs of an identical word or word pair repeated more than ``max_repeats`` times in a row
    down to ``max_repeats`` (compared case- and punctuation-insensitively). Whisper still emits short
    bursts like "okay okay okay ..." on near-silent stretches even with long-form safeguards; genuine
    stuttering or perseveration ("the the the") stays within the cap and is kept."""
    tokens = text.split()
    for n in range(1, max_n + 1):
        keys = [_TOKEN_KEY.sub("", t.lower()) for t in tokens]
        out: list[str] = []
        i = 0
        while i < len(tokens):
            gram = keys[i : i + n]
            reps = 1
            while len(gram) == n and keys[i + reps * n : i + (reps + 1) * n] == gram:
                reps += 1
            if reps == 1:
                out.append(tokens[i])
                i += 1
            else:
                out.extend(tokens[i : i + min(reps, max_repeats) * n])
                i += reps * n
        tokens = out
    return " ".join(tokens)


@lru_cache(maxsize=2)
def _asr_pipeline(model_name: str, device: str):
    try:
        from transformers import pipeline
        from transformers.generation import utils as generation_utils
    except ImportError as exc:
        raise ModelUnavailableError("transformers is not installed. Install requirements-extra.txt.") from exc
    if hasattr(generation_utils, "DeferredStopCheck"):
        # transformers 5.x: the MPS-only DeferredStopCheck calls `cache.layers`, which Whisper's
        # EncoderDecoderCache lacks; the regular StopCheck yields identical output.
        generation_utils.DeferredStopCheck.is_supported = staticmethod(lambda *args, **kwargs: False)  # type: ignore[method-assign]
    logger.info("loading ASR model", extra={"model_name": model_name, "device": device})
    return pipeline("automatic-speech-recognition", model=model_name, device=device)


def transcribe(
    paths_by_id: Mapping[str, str | Path],
    model_name: str,
    *,
    device: str = "auto",
    languages: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Whisper transcripts for every recording, decoded sequentially over the whole file with
    loop-prevention fallbacks. ``languages`` maps a recording id to a Whisper language code (e.g.
    ``en``, ``zh``); recordings without one are decoded as English. A failed recording becomes an
    empty string, logged, rather than aborting the batch."""
    asr = _asr_pipeline(model_name, resolve_device(device))
    out: dict[str, str] = {}
    for rec_id, path in paths_by_id.items():
        try:
            samples, sr, _ = load_audio(path)
            audio = resample(to_mono(samples), sr, _TARGET_SR)
            kwargs = {**_ASR_GENERATE_KWARGS, "language": (languages or {}).get(rec_id, "en")}
            result = asr({"raw": audio, "sampling_rate": _TARGET_SR}, return_timestamps=True, generate_kwargs=kwargs)
            out[rec_id] = collapse_repeats(result["text"].strip())
        except Exception as exc:
            logger.warning("ASR failed, recording as empty transcript", extra={"id": rec_id, "reason": str(exc)})
            out[rec_id] = ""
    return out


@lru_cache(maxsize=2)
def _text_backend(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ModelUnavailableError("torch/transformers are not installed. Install requirements-extra.txt.") from exc
    logger.info("loading text model", extra={"model_name": model_name, "device": device})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    return tokenizer, model, torch


def _embed_text(text: str, tokenizer, model, torch_module, device: str, max_tokens: int) -> np.ndarray:
    """``(n_layers, hidden_dim)``: every hidden layer mean-pooled over content tokens. Texts longer
    than the model window are split into consecutive windows, pooled per window, and combined
    weighted by each window's token count."""
    ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]  # windowed below
    if not ids:
        raise FeatureExtractionError("empty transcript")
    window = max_tokens - 2
    layer_sum, total = None, 0
    for start in range(0, len(ids), window):
        chunk = ids[start : start + window]
        wrapped = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]  # [CLS] ... [SEP] / <s> ... </s>
        input_ids = torch_module.tensor([wrapped], device=device)
        with torch_module.no_grad():
            hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states
        pooled = np.stack([h[0, 1:-1].mean(dim=0).float().cpu().numpy() for h in hidden])
        layer_sum = pooled * len(chunk) if layer_sum is None else layer_sum + pooled * len(chunk)
        total += len(chunk)
    assert layer_sum is not None  # ids is non-empty, so the loop ran at least once
    return layer_sum / total


def extract_text_embeddings(
    texts_by_id: Mapping[str, str], model_name: str, *, device: str = "auto", max_tokens: int = 512
) -> tuple[list[str], np.ndarray]:
    """Layer-wise mean-pooled embeddings, same ``(ids, X[n, n_layers, dim])`` contract as
    :func:`sabench.features.embeddings.extract_embeddings`; empty or failed texts become NaN."""
    resolved = resolve_device(device)
    tokenizer, model, torch_module = _text_backend(model_name, resolved)
    ids = list(texts_by_id)
    vectors: dict[str, np.ndarray] = {}
    for rec_id in ids:
        try:
            vectors[rec_id] = _embed_text(texts_by_id[rec_id], tokenizer, model, torch_module, resolved, max_tokens)
        except Exception as exc:
            logger.warning("text embedding failed, recording as NaN", extra={"id": rec_id, "reason": str(exc)})
    if not vectors:
        raise FeatureExtractionError(f"text embedding failed for every transcript (model={model_name})")
    shape = next(iter(vectors.values())).shape
    X = np.stack([vectors.get(i, np.full(shape, np.nan, dtype=np.float32)) for i in ids])
    return ids, X
