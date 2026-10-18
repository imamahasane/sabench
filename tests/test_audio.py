from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from sabench.audio.channel import compute_channel_descriptors
from sabench.audio.interventions import band_limit, match_noise_floor, normalize_loudness
from sabench.audio.prepare import extract_variant, load_audio, resample, to_mono
from sabench.audio.vad import Segment, detect_speech_segments
from sabench.exceptions import AudioProcessingError, VADError


def _tone(duration_s: float, sr: int, freq: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_load_audio_roundtrip(tmp_path):
    sr = 16000
    signal = _tone(0.5, sr)
    path = tmp_path / "tone.wav"
    sf.write(str(path), signal, sr)
    samples, read_sr, channels = load_audio(path)
    assert read_sr == sr
    assert channels == 1
    assert len(samples) == len(signal)


def test_load_audio_missing_file_raises(tmp_path):
    with pytest.raises(AudioProcessingError):
        load_audio(tmp_path / "nope.wav")


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    mono = to_mono(stereo)
    assert np.allclose(mono, [2.0, 3.0])


def test_to_mono_is_noop_for_mono():
    mono = np.array([1.0, 2.0], dtype=np.float32)
    assert np.allclose(to_mono(mono), mono)


def test_resample_noop_when_rates_match():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(resample(x, 16000, 16000), x)


def test_resample_changes_length_proportionally():
    x = _tone(1.0, 8000)
    out = resample(x, 8000, 16000)
    assert abs(len(out) - 16000) <= 2


def test_resample_rejects_nonpositive_rate():
    with pytest.raises(AudioProcessingError):
        resample(np.zeros(10), 0, 16000)


def test_energy_vad_detects_tone_surrounded_by_silence():
    sr = 16000
    silence = np.zeros(int(0.3 * sr), dtype=np.float32)
    tone = _tone(0.5, sr, amplitude=0.8)
    audio = np.concatenate([silence, tone, silence])
    segments = detect_speech_segments(
        audio, sr, fallback="energy", backend="energy",
        energy_frame_ms=30, energy_hop_ms=10, energy_threshold_db=-35.0,
    )
    assert len(segments) >= 1
    # the detected segment should roughly cover the tone region, not the whole recording
    total_speech = sum(s.duration for s in segments)
    assert 0.3 < total_speech < 0.7


def test_energy_vad_rejects_stereo():
    with pytest.raises(VADError):
        detect_speech_segments(np.zeros((10, 2)), 16000, backend="energy")


def test_segment_rejects_end_before_start():
    with pytest.raises(ValueError):
        Segment(start=1.0, end=0.5)


def test_extract_variant_speech_keeps_only_segments():
    sr = 1000
    audio = np.arange(2000, dtype=np.float32)  # 2 seconds
    segments = [Segment(start=0.5, end=1.0)]
    speech = extract_variant(audio, sr, segments, mode="speech", min_variant_seconds=0.1)
    assert speech is not None
    assert len(speech) == pytest.approx(500, abs=1)


def test_extract_variant_silence_is_complement():
    sr = 1000
    audio = np.ones(2000, dtype=np.float32)
    segments = [Segment(start=0.5, end=1.0)]
    silence = extract_variant(audio, sr, segments, mode="silence", min_variant_seconds=0.1)
    assert silence is not None
    assert len(silence) == pytest.approx(1500, abs=1)


def test_extract_variant_returns_none_below_minimum_duration():
    sr = 1000
    audio = np.ones(2000, dtype=np.float32)
    segments = [Segment(start=0.0, end=0.05)]
    speech = extract_variant(audio, sr, segments, mode="speech", min_variant_seconds=1.0)
    assert speech is None


def test_extract_variant_rejects_bad_mode():
    with pytest.raises(AudioProcessingError):
        extract_variant(np.zeros(10), 1000, [], mode="bogus")


def test_channel_descriptors_nan_ltas_without_silence():
    sr = 16000
    audio = _tone(1.0, sr)
    descriptors = compute_channel_descriptors(audio, sr, [Segment(0.0, 1.0)], None, ltas_bands=4)
    assert descriptors.n_segments == 1
    assert all(np.isnan(v) for v in descriptors.ltas_db)


def test_channel_descriptors_with_silence_has_finite_ltas():
    sr = 16000
    audio = _tone(1.0, sr)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    descriptors = compute_channel_descriptors(audio, sr, [], silence, ltas_bands=4)
    assert len(descriptors.ltas_db) == 4
    assert all(np.isfinite(v) for v in descriptors.ltas_db)


def test_normalize_loudness_moves_rms_toward_target():
    rng = np.random.default_rng(0)
    quiet = (0.001 * rng.normal(size=16000)).astype(np.float32)
    out = normalize_loudness(quiet, target_rms_dbfs=-20.0)
    out_db = 20 * np.log10(np.sqrt(np.mean(out**2)) + 1e-12)
    assert abs(out_db - (-20.0)) < 1.0


def test_band_limit_attenuates_out_of_band_energy():
    sr = 16000
    low_tone = _tone(1.0, sr, freq=50.0, amplitude=0.5)
    filtered = band_limit(low_tone, sr, low_hz=300, high_hz=3400)
    assert np.sqrt(np.mean(filtered**2)) < 0.5 * np.sqrt(np.mean(low_tone**2))


def test_band_limit_rejects_invalid_range():
    with pytest.raises(AudioProcessingError):
        band_limit(np.zeros(1000), 16000, low_hz=5000, high_hz=100)


def test_match_noise_floor_raises_floor_when_below_target():
    quiet = np.zeros(16000, dtype=np.float32)
    out = match_noise_floor(quiet, target_floor_db=-40.0, seed=0)
    assert np.std(out) > 0


def test_match_noise_floor_is_noop_when_already_above_target():
    rng = np.random.default_rng(0)
    loud_noise = rng.normal(0, 0.5, size=16000).astype(np.float32)
    out = match_noise_floor(loud_noise, target_floor_db=-80.0, seed=0)
    assert np.array_equal(out, loud_noise)
