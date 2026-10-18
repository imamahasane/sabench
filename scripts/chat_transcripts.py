"""Manual transcripts (participant turns only) -> ``<features>/<dataset>/transcripts_manual.csv``
for every dataset that ships CHAT transcripts: ADReSS and the Pitt-wide benchmark. ADReSSo ships
none, so it only gets ASR transcripts (``sabench transcribe``).

Usage (from the repository root):  python scripts/chat_transcripts.py configs/full.yaml ../Data
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from sabench.config import ResolvedPaths, load_config
from sabench.features.text import chat_participant_text
from sabench.pipeline.io import save_lld_features


def adress_cha(data: Path, row) -> Path:
    root = data / "ADReSS2020" / "ADReSS-IS2020-data"
    sub = f"train/transcription/{'cc' if row.label == 0 else 'cd'}" if row.split == "train" else "test/transcription"
    return root / sub / f"{row.id}.cha"


def pitt_cha(data: Path, row) -> Path:
    group = Path(row.path).parts[0]  # manifest path: <Control|Dementia>/cookie/<id>.mp3
    return data / "Pitt" / "transcripts" / group / "cookie" / f"{row.id}.cha"


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    data = Path(sys.argv[2])
    paths = ResolvedPaths(cfg, ".")
    for dataset, locate in (("adress", adress_cha), ("pitt", pitt_cha)):
        if dataset not in paths.manifests:
            continue
        manifest = pd.read_csv(paths.manifest_path(dataset), dtype={"id": str})
        texts = {r.id: chat_participant_text(locate(data, r)) for r in manifest.itertuples()}
        df = pd.DataFrame({"text": pd.Series(texts)}).rename_axis("id")
        out = paths.features / dataset / "transcripts_manual.csv"
        save_lld_features(df, out)
        n_words = df["text"].str.split().str.len()
        print(f"{dataset}: wrote {out.name}, {len(df)} transcripts, words median {int(n_words.median())} "
              f"(min {n_words.min()})")


if __name__ == "__main__":
    main()
