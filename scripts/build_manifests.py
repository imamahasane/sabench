"""Build ``manifest_adress.csv`` / ``manifest_adresso.csv`` from the challenge data releases.

Labels: ADReSS train from the cc/cd folders, ADReSS test from ``2020Labels.txt``; ADReSSo train from
the cn/ad folders, ADReSSo test from ``ground truth of test sets/task1.csv``. Paths are written
relative to ``<run_dir>/<dataset>``, which should be a symlink to the dataset's release folder.
Speaker keys are added afterwards by ``scripts/match_pitt.py``.

Also builds the Pitt-wide and TAUKADIAL manifests when their data are present.

Usage (from the repository root):  python scripts/build_manifests.py ../Data ../run_full
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

DX_TO_LABEL = {"Control": 0, "ProbableAD": 1}


def read_semicolon_table(path: Path) -> dict[str, dict[str, str]]:
    """``ID ; age; gender ;mmse``-style files -> ``{id: {column: value}}``."""
    lines = [line.rstrip("\n") for line in path.open() if line.strip()]
    header = [h.strip().lower() for h in lines[0].split(";")]
    rows = {}
    for line in lines[1:]:
        row = dict(zip(header, (p.strip() for p in line.split(";")), strict=False))
        if row.get("id"):
            rows[row["id"]] = row
    return rows


def read_csv_by(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        return {r[key]: r for r in csv.DictReader(fh) if r.get(key)}


def write(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def adress_rows(data: Path) -> list[dict]:
    root = data / "ADReSS2020" / "ADReSS-IS2020-data"
    rows = []
    for cat, label in (("cc", 0), ("cd", 1)):
        meta = read_semicolon_table(root / "train" / f"{cat}_meta_data.txt")
        for wav in sorted((root / "train" / "Full_wave_enhanced_audio" / cat).glob("*.wav")):
            m = meta.get(wav.stem, {})
            rows.append({"id": wav.stem, "split": "train", "label": label,
                         "path": f"train/Full_wave_enhanced_audio/{cat}/{wav.name}",
                         "age": m.get("age", ""), "gender": m.get("gender", ""), "mmse": m.get("mmse", "")})
    labels = read_semicolon_table(data / "ADReSS2020" / "2020Labels.txt")
    for wav in sorted((root / "test" / "Full_wave_enhanced_audio").glob("*.wav")):
        m = labels[wav.stem]
        rows.append({"id": wav.stem, "split": "test", "label": int(m["label"]),
                     "path": f"test/Full_wave_enhanced_audio/{wav.name}",
                     "age": m.get("age", ""), "gender": m.get("gender", ""), "mmse": m.get("mmse", "")})
    return rows


def adresso_rows(data: Path) -> list[dict]:
    root = data / "AddReSSo2021" / "diagnosis"
    mmse = read_csv_by(root / "train" / "adresso-train-mmse-scores.csv", "adressfname")
    rows = []
    for cat, label in (("cn", 0), ("ad", 1)):
        for wav in sorted((root / "train" / "audio" / cat).glob("*.wav")):
            rows.append({"id": wav.stem, "split": "train", "label": label,
                         "path": f"train/audio/{cat}/{wav.name}", "mmse": mmse.get(wav.stem, {}).get("mmse", "")})
    gt = data / "AddReSSo2021" / "ground truth of test sets"
    dx, mmse_test = read_csv_by(gt / "task1.csv", "ID"), read_csv_by(gt / "task2.csv", "ID")
    for wav in sorted((root / "test-dist" / "audio").glob("*.wav")):
        rows.append({"id": wav.stem, "split": "test", "label": DX_TO_LABEL[dx[wav.stem]["Dx"]],
                     "path": f"test-dist/audio/{wav.name}", "mmse": mmse_test.get(wav.stem, {}).get("MMSE", "")})
    return rows


PITT_LABELS = {"Control": 0, "ProbableAD": 1}
PITT_SPLIT_SEED = 1234
PITT_TEST_SIZE = 0.30


def _pitt_participant_id(cha: Path) -> dict[str, str]:
    """The participant ``@ID`` tier: ``eng|Pitt|PAR|age;|gender|Dx|...|Participant|MMSE|``."""
    for line in cha.open(encoding="utf-8", errors="ignore"):
        if line.startswith("@ID:") and "|PAR|" in line:
            f = line.split("\t", 1)[1].strip().split("|")
            return {"age": f[3].rstrip(";"), "gender": f[4], "dx": f[5], "mmse": f[8]}
    return {}


def pitt_rows(data: Path) -> list[dict]:
    """Pitt-wide benchmark: ProbableAD vs Control Cookie Theft recordings, one per speaker (the
    earliest visit carrying one of those two diagnoses), split 70/30 stratified by label with a
    fixed seed. One recording per speaker keeps the split speaker-disjoint and every
    bootstrap/resampling unit independent. Unlike ADReSS/ADReSSo, groups are NOT age/gender
    matched -- report a demographics-only baseline alongside any result on this set."""
    from sklearn.model_selection import train_test_split

    root = data / "Pitt"
    sessions = []
    for cha in sorted((root / "transcripts").glob("*/cookie/*.cha")):
        speaker, visit = cha.stem.split("-")
        meta = _pitt_participant_id(cha)
        if meta.get("dx") in PITT_LABELS:
            sessions.append({"id": cha.stem, "speaker": f"pitt_{speaker}", "visit": int(visit),
                             "group": cha.parts[-3], **meta})
    first = {}
    for s in sorted(sessions, key=lambda s: (s["speaker"], s["visit"])):
        first.setdefault(s["speaker"], s)
    chosen = list(first.values())
    labels = [PITT_LABELS[s["dx"]] for s in chosen]
    train_idx, _ = train_test_split(range(len(chosen)), test_size=PITT_TEST_SIZE,
                                    stratify=labels, random_state=PITT_SPLIT_SEED)
    train_set = set(train_idx)
    return [{
        "id": s["id"], "split": "train" if i in train_set else "test", "label": PITT_LABELS[s["dx"]],
        "path": f"{s['group']}/cookie/{s['id']}.mp3", "age": s["age"], "gender": s["gender"],
        "mmse": s["mmse"], "speaker": s["speaker"], "pitt_session": s["id"],
    } for i, s in enumerate(chosen)]


TAUKADIAL_LABELS = {"NC": 0, "MCI": 1}
TAUKADIAL_GENDER = {"F": "female", "M": "male"}


def taukadial_rows(data: Path) -> list[dict]:
    """TAUKADIAL (Interspeech 2024): MCI vs NC, three picture descriptions per speaker, official
    train/test split with released test ground truth. ``speaker`` groups a speaker's three
    recordings; spoken language is added afterwards by ``scripts/detect_language.py``."""
    root = data / "TAUKADIAL"
    train = pd.read_csv(root / "train" / "groundtruth.csv")
    test = pd.read_csv(root / "testgroundtruth.csv", sep=";").merge(
        pd.read_csv(root / "meta_test.csv", sep=";"), on="tkdname"
    )
    rows = []
    for split, frame in (("train", train), ("test", test)):
        for r in frame.itertuples():
            stem = Path(r.tkdname).stem  # taukdial-002-1
            _, speaker, picture = stem.split("-")
            rows.append({
                "id": stem, "split": split, "label": TAUKADIAL_LABELS[r.dx], "path": f"{split}/{r.tkdname}",
                "age": r.age, "gender": TAUKADIAL_GENDER[r.sex], "mmse": r.mmse,
                "speaker": f"tkd_{speaker}", "picture": int(picture),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    a, o = adress_rows(args.data_dir), adresso_rows(args.data_dir)
    write(a, args.run_dir / "manifest_adress.csv", ["id", "split", "label", "path", "age", "gender", "mmse"])
    write(o, args.run_dir / "manifest_adresso.csv", ["id", "split", "label", "path", "mmse"])
    print(f"adress: {len(a)} rows, adresso: {len(o)} rows -> {args.run_dir}")
    if (args.data_dir / "Pitt" / "transcripts").exists():
        p = pitt_rows(args.data_dir)
        write(p, args.run_dir / "manifest_pitt.csv",
              ["id", "split", "label", "path", "age", "gender", "mmse", "speaker", "pitt_session"])
        print(f"pitt: {len(p)} rows")
    if (args.data_dir / "TAUKADIAL" / "train" / "groundtruth.csv").exists():
        t = taukadial_rows(args.data_dir)
        write(t, args.run_dir / "manifest_taukadial.csv",
              ["id", "split", "label", "path", "age", "gender", "mmse", "speaker", "picture"])
        print(f"taukadial: {len(t)} rows")


if __name__ == "__main__":
    main()
