"""Audio interventions for E5: loudness normalisation, band-limiting, and noise-floor matching.

These exist to test whether the channel cues E4 identifies are *causing* part of the benchmark
score: if normalising loudness, limiting bandwidth, or equalising the noise floor collapses a
feature's or embedding's macro-F1 while leaving others unaffected, that is direct evidence the
affected feature was encoding a channel artefact rather than speech content.

Deliberately dependency-free (no ``pyloudnorm``/``pydub``): loudness normalisation here is an
RMS-based approximation, not ITU-R BS.1770-compliant LUFS. That is an acceptable simplification for
a relative, within-study comparison (same approximation applied uniformly to every recording) but
should not be cited as a broadcast-standard loudness value.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from sabench.exceptions import AudioProcessingError
from sabench.utils.seeding import perm_seed

_EPS = 1e-12


def normalize_loudness(audio: np.ndarray, *, target_rms_dbfs: float = -23.0) -> np.ndarray:
    """Scale ``audio`` so its RMS level equals ``target_rms_dbfs`` (dB relative to full scale).

    An RMS-based approximation of loudness normalisation (see module docstring), applied uniformly
    across recordings so it is still a valid relative comparison within this study.
    """
    if len(audio) == 0:
        raise AudioProcessingError("cannot normalise loudness of an empty signal")
    rms = float(np.sqrt(np.mean(np.square(audio)) + _EPS))
    current_dbfs = 20 * np.log10(max(rms, _EPS))
    gain_db = target_rms_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20)
    out = audio * gain_linear
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak  # avoid clipping; rare, since recordings are rarely already near-silent
    return out.astype(np.float32)


def band_limit(
    audio: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (``sosfiltfilt``: no phase distortion, stable at
    higher orders than a transfer-function ``butter`` + ``filtfilt`` pair)."""
    nyquist = sample_rate / 2
    if not (0 < low_hz < high_hz < nyquist):
        raise AudioProcessingError(
            f"require 0 < low_hz < high_hz < nyquist ({nyquist}); got low={low_hz}, high={high_hz}"
        )
    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def _estimate_noise_floor_db(audio: np.ndarray, *, percentile: float = 10.0, frame_len: int = 480) -> float:
    if len(audio) < frame_len:
        return float(20 * np.log10(np.sqrt(np.mean(np.square(audio)) + _EPS)))
    n_frames = len(audio) // frame_len
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + _EPS)
    floor_rms = np.percentile(rms, percentile)
    return float(20 * np.log10(max(floor_rms, _EPS)))


def match_noise_floor(
    audio: np.ndarray, *, target_floor_db: float, seed: int = 0, percentile: float = 10.0
) -> np.ndarray:
    """Add calibrated white Gaussian noise so the recording's estimated noise floor reaches
    ``target_floor_db``. A no-op if the recording's floor is already at or above the target: this
    intervention can only raise a noise floor, not denoise a recording below it.
    """
    current_db = _estimate_noise_floor_db(audio, percentile=percentile)
    if current_db >= target_floor_db:
        return audio.astype(np.float32)
    target_rms = 10 ** (target_floor_db / 20)
    current_rms = 10 ** (current_db / 20)
    added_power = max(target_rms**2 - current_rms**2, 0.0)
    rng = np.random.default_rng(perm_seed(seed, 0))
    noise = rng.normal(0.0, np.sqrt(added_power), size=len(audio))
    out = audio + noise
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)
