"""Recording-channel descriptors: cheap, content-agnostic statistics (duration, VAD segment count,
noise-floor level, long-term spectrum of the non-speech portion) used by E4's channel-only baseline
to measure how much of the label is predictable from recording conditions rather than speech.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sabench.audio.vad import Segment
from sabench.exceptions import AudioProcessingError

_EPS = 1e-12


@dataclass(frozen=True)
class ChannelDescriptors:
    duration_seconds: float
    n_segments: int
    noise_floor_rms_db: float
    ltas_db: tuple[float, ...]  # one value per log-spaced band

    def as_feature_dict(self) -> dict[str, float]:
        d = {
            "duration_seconds": self.duration_seconds,
            "n_segments": float(self.n_segments),
            "noise_floor_rms_db": self.noise_floor_rms_db,
        }
        for i, v in enumerate(self.ltas_db):
            d[f"ltas_band_{i}"] = v
        return d


def _frame_rms(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(audio) < frame_len:
        return np.array([float(np.sqrt(np.mean(np.square(audio)) + _EPS))])
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        out[i] = np.sqrt(np.mean(np.square(frame)) + _EPS)
    return out


def _long_term_average_spectrum(audio: np.ndarray, sample_rate: int, n_bands: int) -> tuple[float, ...]:
    """Average power spectrum (Welch-style, Hann window, 50% overlap) binned into ``n_bands``
    log-spaced bands between 20 Hz and Nyquist, expressed in dB."""
    if n_bands < 1:
        raise AudioProcessingError(f"n_bands must be >= 1, got {n_bands}")
    nyquist = sample_rate / 2
    frame_len = min(len(audio), 2048) if len(audio) > 0 else 0
    if frame_len < 8:
        return tuple(float("nan") for _ in range(n_bands))

    hop_len = max(1, frame_len // 2)
    window = np.hanning(frame_len)
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
    freqs = np.fft.rfftfreq(frame_len, d=1 / sample_rate)
    power_sum = np.zeros_like(freqs)
    for i in range(n_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        spectrum = np.fft.rfft(frame * window)
        power_sum += np.abs(spectrum) ** 2
    power_avg = power_sum / n_frames

    low = max(20.0, freqs[1] if len(freqs) > 1 else 20.0)
    edges = np.geomspace(low, nyquist, n_bands + 1)
    bands_db = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (freqs >= lo) & (freqs < hi)
        band_power = power_avg[mask].mean() if mask.any() else _EPS
        bands_db.append(float(10 * np.log10(band_power + _EPS)))
    return tuple(bands_db)


def compute_channel_descriptors(
    full_audio: np.ndarray,
    sample_rate: int,
    segments: list[Segment],
    silence_audio: np.ndarray | None,
    *,
    ltas_bands: int = 8,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    noise_floor_percentile: float = 10.0,
) -> ChannelDescriptors:
    """Compute duration, VAD segment count, a noise-floor estimate (a low percentile of frame RMS,
    in dB relative to the recording's peak), and the silence-only long-term average spectrum.

    ``ltas_db`` is all-NaN when ``silence_audio`` is ``None`` (variant unavailable / below the
    minimum duration) -- callers persist this as missing, not zero, per the project's leakage and
    honesty rules.
    """
    duration = len(full_audio) / sample_rate
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    peak = float(np.max(np.abs(full_audio))) or 1.0
    rms = _frame_rms(full_audio, frame_len, hop_len)
    floor_rms = float(np.percentile(rms, noise_floor_percentile))
    noise_floor_db = float(20 * np.log10(max(floor_rms, _EPS) / peak))

    if silence_audio is not None and len(silence_audio) > 0:
        ltas = _long_term_average_spectrum(silence_audio, sample_rate, ltas_bands)
    else:
        ltas = tuple(float("nan") for _ in range(ltas_bands))

    return ChannelDescriptors(
        duration_seconds=duration,
        n_segments=len(segments),
        noise_floor_rms_db=noise_floor_db,
        ltas_db=ltas,
    )
