"""Frozen speech-embedding extraction: WavLM, wav2vec 2.0, Whisper encoder.

All hidden states (every transformer layer), mean-pooled over time per chunk (audio is processed in
chunks of at most ``chunk_seconds`` to bound memory) and then duration-weighted-averaged across
chunks, so a recording's final representation is one vector per layer regardless of its length.

Models are frozen: no gradient is ever computed (``torch.no_grad()``), and nothing here fine-tunes a
backbone -- only a downstream linear probe (``sabench.modeling.probes``) is ever trained.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from sabench.audio.prepare import load_audio, resample, to_mono
from sabench.exceptions import FeatureExtractionError, ModelUnavailableError
from sabench.logging_config import get_logger

logger = get_logger(__name__)

ModelKey = Literal["wavlm", "wav2vec2", "whisper"]
_TARGET_SR = 16_000


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError as exc:
        raise ModelUnavailableError(
            "torch is not installed. Install requirements-extra.txt to extract embeddings."
        ) from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load_backend(model_key: ModelKey, model_name: str, device: str):
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise ModelUnavailableError(
            "torch/transformers are not installed. Install requirements-extra.txt to extract embeddings."
        ) from exc

    logger.info("loading embedding model", extra={"model_key": model_key, "model_name": model_name})
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    encoder = model.get_encoder() if model_key == "whisper" else model
    return processor, encoder, torch


def _chunk_audio(audio: np.ndarray, sample_rate: int, chunk_seconds: int) -> list[np.ndarray]:
    chunk_len = chunk_seconds * sample_rate
    if len(audio) <= chunk_len:
        return [audio]
    return [audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len)]


def _embed_recording(
    audio: np.ndarray,
    sample_rate: int,
    *,
    processor,
    encoder,
    torch_module,
    device: str,
    chunk_seconds: int,
) -> np.ndarray:
    """Returns an ``(n_layers, hidden_dim)`` array: per-layer, duration-weighted mean over time."""
    chunks = _chunk_audio(audio, sample_rate, chunk_seconds)
    layer_sums: np.ndarray | None = None
    total_weight = 0.0

    for chunk in chunks:
        if len(chunk) == 0:
            continue
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch_module.no_grad():
            outputs = encoder(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (1, seq_len, hidden_dim), length = n_layers (+ embeddings layer)
        hidden_states = outputs.hidden_states
        pooled = np.stack(
            [hs.squeeze(0).mean(dim=0).cpu().numpy() for hs in hidden_states], axis=0
        )  # (n_layers, hidden_dim)
        weight = len(chunk) / sample_rate
        layer_sums = pooled * weight if layer_sums is None else layer_sums + pooled * weight
        total_weight += weight

    if layer_sums is None or total_weight == 0:
        raise FeatureExtractionError("recording produced no non-empty chunks to embed")
    return layer_sums / total_weight


def extract_embeddings(
    paths_by_id: Mapping[str, str | Path],
    model_key: ModelKey,
    model_name: str,
    *,
    chunk_seconds: int = 30,
    device: str = "auto",
) -> tuple[list[str], np.ndarray]:
    """Extract layer-wise, mean-pooled embeddings for every recording in ``paths_by_id``.

    :returns: ``(ids, X)`` where ``ids`` preserves the input order and ``X`` has shape
        ``(n_recordings, n_layers, hidden_dim)``. A recording that fails to load or embed becomes an
        all-NaN slice (logged as a warning) rather than aborting the batch, once the embedding shape
        is known from at least one successful recording.
    :raises ModelUnavailableError: ``torch``/``transformers`` are not installed.
    :raises FeatureExtractionError: every recording in the batch failed.
    """
    resolved_device = _resolve_device(device)
    processor, encoder, torch_module = _load_backend(model_key, model_name, resolved_device)

    ids = list(paths_by_id.keys())
    vectors: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    failures: list[str] = []

    for rec_id in ids:
        path = paths_by_id[rec_id]
        try:
            samples, orig_sr, _ = load_audio(path)
            mono = to_mono(samples)
            audio = resample(mono, orig_sr, _TARGET_SR)
            vec = _embed_recording(
                audio,
                _TARGET_SR,
                processor=processor,
                encoder=encoder,
                torch_module=torch_module,
                device=resolved_device,
                chunk_seconds=chunk_seconds,
            )
            if shape is None:
                shape = vec.shape
            vectors[rec_id] = vec
        except Exception as exc:
            failures.append(rec_id)
            logger.warning(
                "embedding extraction failed, recording as NaN", extra={"id": rec_id, "reason": str(exc)}
            )

    if shape is None:
        raise FeatureExtractionError(f"embedding extraction failed for every recording (model={model_name})")

    for rec_id in failures:
        vectors[rec_id] = np.full(shape, np.nan, dtype=np.float32)

    X = np.stack([vectors[i] for i in ids], axis=0)
    return ids, X
