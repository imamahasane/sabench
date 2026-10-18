"""Add a ``language`` column to a dataset manifest using Whisper language identification.

TAUKADIAL releases no language field, but its English and Mandarin halves differ in recording
setup, picture set, diagnostic procedure, and MCI prevalence, so language must be known to test
for shortcuts. Each recording's first 30 s are classified; each speaker gets the majority language
of their recordings (one TAUKADIAL recording is misidentified on its own).

Usage (from the repository root):  python scripts/detect_language.py configs/full.yaml taukadial
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from sabench.audio.prepare import load_audio, resample, to_mono
from sabench.config import ResolvedPaths, load_config
from sabench.utils.io import safe_join

MODEL = "openai/whisper-small"
CANDIDATES = ("en", "zh")


def main() -> None:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cfg = load_config(Path(sys.argv[1]))
    dataset = sys.argv[2]
    paths = ResolvedPaths(cfg, ".")
    manifest_path = paths.manifest_path(dataset)
    manifest = pd.read_csv(manifest_path, dtype={"id": str})

    processor = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL).eval()
    token_to_lang = {processor.tokenizer.convert_tokens_to_ids(f"<|{c}|>"): c for c in CANDIDATES}

    detected = []
    for rel in manifest["path"]:
        samples, sr, _ = load_audio(safe_join(paths.raw / dataset, rel))
        audio = resample(to_mono(samples), sr, 16_000)[: 16_000 * 30]
        features = processor(audio, sampling_rate=16_000, return_tensors="pt").input_features
        with torch.no_grad():
            token = int(model.detect_language(features).flatten()[0])
        detected.append(token_to_lang.get(token, "other"))

    manifest["language_detected"] = detected
    majority = manifest.groupby("speaker")["language_detected"].agg(lambda s: s.mode()[0])
    manifest["language"] = manifest["speaker"].map(majority)
    disagree = int((manifest["language"] != manifest["language_detected"]).sum())
    manifest = manifest.drop(columns="language_detected")
    manifest.to_csv(manifest_path, index=False)
    speakers = manifest.drop_duplicates("speaker")
    print(pd.crosstab(speakers["split"], speakers["language"]).to_string())
    print(f"{disagree} recording(s) whose own detection disagreed with their speaker's majority; wrote {manifest_path}")


if __name__ == "__main__":
    main()
