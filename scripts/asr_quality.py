"""ASR transcript quality against the manual transcripts (ADReSS and Pitt).

WER is not meaningful here: ASR transcribes the whole recording (participant and interviewer) while
the manual transcripts keep participant turns only. Instead this reports participant-word recall,
the share of the participant's manually transcribed words (as a multiset) that also occur in the ASR
transcript, plus a repetition-loop check (largest count of any repeated 4-gram). Chinese text is counted
per character, since it has no word spaces.
Writes ``<results>/tables/asr_quality.csv`` (per recording) and prints a summary.

Usage (from the repository root):  python scripts/asr_quality.py configs/full.yaml
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from sabench.config import ResolvedPaths, load_config

_WORD = re.compile(r"[a-z']+|[\u4e00-\u9fff]")  # Latin words; each CJK character is one unit


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text).lower()) if w != "'"]


def participant_recall(manual: str, asr: str) -> float:
    ref, hyp = Counter(words(manual)), Counter(words(asr))
    total = sum(ref.values())
    return sum(min(c, hyp[w]) for w, c in ref.items()) / total if total else float("nan")


def max_4gram_repeat(text: str) -> int:
    w = words(text)
    grams = Counter(" ".join(w[i : i + 4]) for i in range(len(w) - 3))
    return max(grams.values(), default=0)


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    paths = ResolvedPaths(cfg, ".")
    rows = []
    for ds in cfg.datasets:
        asr_path = paths.features / ds / "transcripts_asr.csv"
        if not asr_path.exists():
            continue
        asr = pd.read_csv(asr_path, dtype={"id": str}).set_index("id")["text"].fillna("")
        manual_path = paths.features / ds / "transcripts_manual.csv"
        manual = (pd.read_csv(manual_path, dtype={"id": str}).set_index("id")["text"].fillna("")
                  if manual_path.exists() else None)
        for rec_id, text in asr.items():
            rows.append({
                "dataset": ds, "id": rec_id, "asr_words": len(words(text)),
                "max_4gram_repeat": max_4gram_repeat(text),
                "participant_recall": participant_recall(manual[rec_id], text) if manual is not None else float("nan"),
            })
    df = pd.DataFrame(rows)
    out = paths.tables / "asr_quality.csv"
    df.to_csv(out, index=False)
    summary = df.groupby("dataset").agg(
        n=("id", "size"), asr_words_median=("asr_words", "median"),
        loops_gt5=("max_4gram_repeat", lambda s: int((s > 5).sum())),
        recall_median=("participant_recall", "median"),
        recall_q25=("participant_recall", lambda s: s.quantile(0.25)),
        recall_below_half=("participant_recall", lambda s: int((s < 0.5).sum())),
    ).round(3)
    print(summary.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
