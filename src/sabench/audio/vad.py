"""Voice-activity detection: split a recording into speech and non-speech (silence) segments.

Two backends:

* ``pyannote`` -- the gated ``pyannote/segmentation-3.0`` model (requires ``pyannote.audio`` from
  ``requirements-extra.txt`` and a Hugging Face token with the model's terms accepted). Used when
  available, since it is what the original paper's silence/speech split was built on.
* ``energy`` -- a dependency-free short-time energy threshold, always available. Used automatically
  when ``pyannote.audio`` is not installed or ``HF_TOKEN`` is not set, and recommended as a second,
  independent VAD for the VAD sensitivity check (``docs/EXPERIMENTS.md``, E4).

Both return the same :class:`Segment` list so every downstream consumer (``extract_variant``,
``channel.py``) is backend-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sabench.exceptions import ModelUnavailableError, VADError
from sabench.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Segment:
    start: float  # seconds
    end: float    # seconds

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"segment end {self.end} precedes start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start


class VADBackend(Protocol):
    def __call__(self, audio: np.ndarray, sample_rate: int) -> list[Segment]: ...


def _energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    threshold_db: float = -35.0,
) -> list[Segment]:
    """Short-time-energy VAD: frames above ``threshold_db`` (relative to the recording's peak) are
    speech. Deterministic, dependency-free, and fast -- a reasonable fallback and a useful
    second opinion against a learned VAD.
    """
    if audio.ndim != 1:
        raise VADError("energy VAD expects mono audio")
    if len(audio) == 0:
        return []

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    peak = float(np.max(np.abs(audio))) or 1e-12

    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len) if len(audio) >= frame_len else 1
    is_speech = np.zeros(n_frames, dtype=bool)
    frame_times = np.zeros(n_frames, dtype=float)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame))))
        db = 20 * np.log10(max(rms, 1e-12) / peak)
        is_speech[i] = db >= threshold_db
        frame_times[i] = start / sample_rate

    segments: list[Segment] = []
    in_speech = False
    seg_start = 0.0
    frame_duration = frame_len / sample_rate
    for t, speech in zip(frame_times, is_speech, strict=True):
        if speech and not in_speech:
            seg_start = t
            in_speech = True
        elif not speech and in_speech:
            segments.append(Segment(seg_start, t))
            in_speech = False
    if in_speech:
        segments.append(Segment(seg_start, frame_times[-1] + frame_duration))
    return segments


@lru_cache(maxsize=4)
def _load_pyannote_pipeline(model_name: str, min_duration_on: float, min_duration_off: float):
    """``pyannote/segmentation-3.0`` is a segmentation *model*, not a ready pipeline: it is wrapped
    in pyannote's ``VoiceActivityDetection`` pipeline, as its model card prescribes."""
    try:
        from huggingface_hub import get_token
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
    except ImportError as exc:
        raise ModelUnavailableError(
            "pyannote.audio is not installed. Install requirements-extra.txt, or rely on the "
            "'energy' VAD fallback (configs/default.yaml: vad.fallback)."
        ) from exc

    token = get_token()  # HF_TOKEN env var, else the credential saved by `hf auth login`
    if not token:
        raise ModelUnavailableError(
            "no Hugging Face credential found. Accept the pyannote/segmentation-3.0 terms on Hugging "
            "Face, then run `hf auth login` (or export HF_TOKEN), or use the 'energy' VAD."
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(OSError),
    )
    def _load():
        return Model.from_pretrained(model_name, token=token)

    model = _load()
    if model is None:
        raise ModelUnavailableError(f"could not load {model_name}; are its Hugging Face terms accepted?")
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate({"min_duration_on": min_duration_on, "min_duration_off": min_duration_off})
    return pipeline


def _pyannote_vad(
    audio: np.ndarray, sample_rate: int, *, model_name: str, min_duration_on: float, min_duration_off: float
) -> list[Segment]:
    import torch

    pipeline = _load_pyannote_pipeline(model_name, min_duration_on, min_duration_off)
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    return [Segment(float(seg.start), float(seg.end)) for seg in annotation.get_timeline().support()]


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model_name: str = "pyannote/segmentation-3.0",
    backend: str = "auto",
    fallback: str = "energy",
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
    energy_frame_ms: float = 30.0,
    energy_hop_ms: float = 10.0,
    energy_threshold_db: float = -35.0,
) -> list[Segment]:
    """Detect speech segments.

    :param backend: ``auto`` tries pyannote and falls back to the energy VAD; ``pyannote`` requires
        pyannote and raises instead of falling back (use it for runs whose results claim pyannote);
        ``energy`` uses the energy VAD only.
    :raises VADError: the required backend fails, or both backends fail under ``auto``.
    """
    if backend not in ("auto", "pyannote", "energy"):
        raise VADError(f"unknown VAD backend {backend!r}; expected auto, pyannote, or energy")
    if backend in ("auto", "pyannote"):
        try:
            segments = _pyannote_vad(
                audio, sample_rate, model_name=model_name,
                min_duration_on=min_duration_on, min_duration_off=min_duration_off,
            )
            logger.info("VAD backend=pyannote", extra={"n_segments": len(segments)})
            return segments
        except Exception as exc:
            if backend == "pyannote":
                raise VADError(f"pyannote VAD required but failed: {exc}") from exc
            reason = "unavailable" if isinstance(exc, ModelUnavailableError) else "failed"
            logger.warning(f"pyannote VAD {reason}, falling back to energy VAD", extra={"reason": str(exc)})

    if fallback != "energy":
        raise VADError(f"unknown VAD fallback {fallback!r}")
    try:
        segments = _energy_vad(
            audio,
            sample_rate,
            frame_ms=energy_frame_ms,
            hop_ms=energy_hop_ms,
            threshold_db=energy_threshold_db,
        )
    except Exception as exc:
        raise VADError(f"energy VAD failed: {exc}") from exc
    logger.info("VAD backend=energy", extra={"n_segments": len(segments)})
    return segments
