"""E0's audio-preparation stage: for every manifest row, load the raw recording, resample, run VAD,
write the ``full``/``speech``/``silence`` variants, and compute channel descriptors. One failed
recording is logged and skipped (flagged as missing downstream), never aborts the whole dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sabench.audio.channel import compute_channel_descriptors
from sabench.audio.prepare import extract_variant, prepare_recording, write_wav
from sabench.audio.vad import detect_speech_segments
from sabench.config import AudioConfig, ChannelConfig, VADConfig
from sabench.exceptions import AudioProcessingError
from sabench.logging_config import get_logger
from sabench.utils.io import safe_join

logger = get_logger(__name__)


def prepare_dataset_audio(
    manifest: pd.DataFrame,
    raw_dir: str | Path,
    interim_dir: str | Path,
    *,
    audio_cfg: AudioConfig,
    vad_cfg: VADConfig,
    channel_cfg: ChannelConfig,
) -> pd.DataFrame:
    """Prepare every recording in ``manifest`` and return a channel-descriptor table indexed by id.

    Writes ``<interim_dir>/<full|speech|silence>/<id>.wav`` and returns one row per recording id
    (even ones that failed, as all-NaN) so the output aligns one-to-one with the manifest.
    """
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    rows: dict[str, dict] = {}

    for _, record in manifest.iterrows():
        rec_id = str(record["id"])
        try:
            source_path = safe_join(raw_dir, record["path"])
            audio, info = prepare_recording(
                source_path, rec_id, target_sample_rate=audio_cfg.sample_rate
            )
            write_wav(interim_dir / "full" / f"{rec_id}.wav", audio, audio_cfg.sample_rate)

            segments = detect_speech_segments(
                audio, audio_cfg.sample_rate,
                model_name=vad_cfg.model, backend=vad_cfg.backend, fallback=vad_cfg.fallback,
                min_duration_on=vad_cfg.min_duration_on, min_duration_off=vad_cfg.min_duration_off,
                energy_frame_ms=vad_cfg.energy_frame_ms, energy_hop_ms=vad_cfg.energy_hop_ms,
                energy_threshold_db=vad_cfg.energy_threshold_db,
            )

            speech = extract_variant(
                audio, audio_cfg.sample_rate, segments, mode="speech",
                min_variant_seconds=audio_cfg.min_variant_seconds,
            )
            silence = extract_variant(
                audio, audio_cfg.sample_rate, segments, mode="silence",
                min_variant_seconds=audio_cfg.min_variant_seconds,
            )
            if speech is not None:
                write_wav(interim_dir / "speech" / f"{rec_id}.wav", speech, audio_cfg.sample_rate)
            if silence is not None:
                write_wav(interim_dir / "silence" / f"{rec_id}.wav", silence, audio_cfg.sample_rate)

            descriptors = compute_channel_descriptors(
                audio, audio_cfg.sample_rate, segments, silence, ltas_bands=channel_cfg.ltas_bands
            )
            rows[rec_id] = descriptors.as_feature_dict()
        except AudioProcessingError as exc:
            logger.warning("audio preparation failed for recording", extra={"id": rec_id, "reason": str(exc)})
            rows[rec_id] = {}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(manifest["id"].astype(str).tolist())
    df.index.name = "id"
    return df
