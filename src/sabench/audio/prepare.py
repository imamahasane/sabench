"""Audio ingestion: load, mono-mix, resample, and cut into the ``full`` / ``speech`` / ``silence``
variants that every downstream feature extractor operates on. Uses ``soundfile`` + ``scipy`` only, so
it works without the heavy ``requirements-extra.txt`` stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from sabench.audio.vad import Segment
from sabench.exceptions import AudioProcessingError
from sabench.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioInfo:
    recording_id: str
    source_path: Path
    original_sample_rate: int
    original_channels: int
    duration_seconds: float


def load_audio(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Read an audio file as float32. Returns ``(samples, sample_rate, n_channels)``.

    ``samples`` has shape ``(n_frames,)`` for mono or ``(n_frames, n_channels)`` for multi-channel.
    """
    p = Path(path)
    if not p.exists():
        raise AudioProcessingError(f"audio file not found: {p}")
    try:
        samples, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioProcessingError(f"could not read audio file {p}: {exc}") from exc
    if samples.size == 0:
        raise AudioProcessingError(f"audio file {p} contains zero samples")
    n_channels = 1 if samples.ndim == 1 else samples.shape[1]
    return samples, sr, n_channels


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average all channels down to one. A no-op for already-mono input."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1).astype(np.float32)


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono signal with a polyphase filter (good anti-aliasing, no external dependency).

    A no-op (returns ``samples`` unchanged) if ``orig_sr == target_sr``.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise AudioProcessingError(f"sample rates must be positive, got orig={orig_sr} target={target_sr}")
    if orig_sr == target_sr:
        return samples
    frac = Fraction(target_sr, orig_sr).limit_denominator(10_000)
    out = resample_poly(samples, frac.numerator, frac.denominator)
    return out.astype(np.float32)


def prepare_recording(
    path: str | Path, recording_id: str, *, target_sample_rate: int
) -> tuple[np.ndarray, AudioInfo]:
    """Load a raw recording, mono-mix, and resample to ``target_sample_rate``.

    :raises AudioProcessingError: the file cannot be read, is empty, or resampling fails.
    """
    samples, orig_sr, n_channels = load_audio(path)
    duration = len(samples) / orig_sr if samples.ndim == 1 else samples.shape[0] / orig_sr
    mono = to_mono(samples)
    resampled = resample(mono, orig_sr, target_sample_rate)
    info = AudioInfo(
        recording_id=recording_id,
        source_path=Path(path),
        original_sample_rate=orig_sr,
        original_channels=n_channels,
        duration_seconds=duration,
    )
    logger.debug(
        "prepared recording",
        extra={"id": recording_id, "orig_sr": orig_sr, "channels": n_channels, "duration_s": duration},
    )
    return resampled, info


def extract_variant(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[Segment],
    *,
    mode: str,
    min_variant_seconds: float = 1.0,
) -> np.ndarray | None:
    """Build the ``speech`` or ``silence`` variant by concatenating VAD segments (or their complement).

    :param mode: ``"speech"`` keeps only the given segments; ``"silence"`` keeps everything else.
    :returns: the concatenated audio, or ``None`` if the result is shorter than
        ``min_variant_seconds`` (callers should record this as a flagged/NaN variant, not a silent
        zero-length file).
    """
    if mode not in ("speech", "silence"):
        raise AudioProcessingError(f"mode must be 'speech' or 'silence', got {mode!r}")

    n_samples = len(audio)
    total_seconds = n_samples / sample_rate
    keep_mask = np.zeros(n_samples, dtype=bool)
    for seg in segments:
        start = max(0, int(math.floor(seg.start * sample_rate)))
        end = min(n_samples, int(math.ceil(seg.end * sample_rate)))
        if end > start:
            keep_mask[start:end] = True

    if mode == "silence":
        keep_mask = ~keep_mask

    out = audio[keep_mask]
    if len(out) / sample_rate < min_variant_seconds:
        logger.warning(
            "variant below minimum duration, flagging as unavailable",
            extra={"mode": mode, "kept_seconds": len(out) / sample_rate, "total_seconds": total_seconds},
        )
        return None
    return out


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), audio, sample_rate, subtype="PCM_16")
