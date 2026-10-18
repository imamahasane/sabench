"""Link every ADReSS / ADReSSo recording to its source Pitt Cookie Theft session (speaker + visit).

Both challenge sets are re-releases of Pitt corpus recordings under new ids (ADReSS ``S001``,
ADReSSo ``adrso001`` / ``adrsdt1``). The challenge files carry no Pitt id, and ADReSS transcripts
use a different TalkBank PID namespace than the Pitt ones, so recordings are matched on audio:
candidates by duration, confirmed by the correlation of their loudness envelopes (robust to the
enhancement/normalisation each challenge applied).

Writes ``<run_dir>/pitt_matches.csv`` and rewrites both manifests with ``pitt_session`` (e.g.
``003-1``) and a speaker-level ``speaker`` key (``pitt_003``; unmatched recordings keep a
dataset-prefixed id), so ``sabench cross-dataset``'s speaker-overlap check works at speaker level.

Usage (from the repository root):  python scripts/match_pitt.py ../run_full ../Data/Pitt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import correlate

ENV_HOP_S = 0.05
DUR_TOL_S = 3.0
MAX_LAG_HOPS = 100
MIN_CORR = 0.8
GROUP_TO_LABEL = {"Control": 0, "Dementia": 1}


def envelope(path: Path) -> tuple[float, np.ndarray]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    hop = int(sr * ENV_HOP_S)
    n = len(audio) // hop
    frames = audio[: n * hop].reshape(n, hop)
    env = np.log(np.sqrt(np.mean(frames**2, axis=1)) + 1e-6)
    return len(audio) / sr, (env - env.mean()) / (env.std() + 1e-9)


def best_xcorr(a: np.ndarray, b: np.ndarray) -> float:
    full = correlate(a, b, mode="full") / min(len(a), len(b))
    mid = len(b) - 1
    return float(full[max(0, mid - MAX_LAG_HOPS) : mid + MAX_LAG_HOPS + 1].max())


def load_pitt(pitt_root: Path) -> pd.DataFrame:
    rows = []
    for group in GROUP_TO_LABEL:
        for p in sorted((pitt_root / "media" / group / "cookie").glob("*.mp3")):
            m = re.fullmatch(r"(\d+)-(\d+)", p.stem)
            if m:
                rows.append({"pitt_session": p.stem, "pitt_speaker": m.group(1), "visit": int(m.group(2)),
                             "pitt_group": group, "path": p})
    df = pd.DataFrame(rows)
    envs = Parallel(n_jobs=-1)(delayed(envelope)(p) for p in df["path"])
    df["dur"] = [d for d, _ in envs]
    df["env"] = [e for _, e in envs]
    return df


def match_one(dur: float, env: np.ndarray, pitt: pd.DataFrame) -> tuple[str | None, float, float]:
    cand = pitt[(pitt["dur"] - dur).abs() <= DUR_TOL_S]
    best = (None, -1.0, np.nan)
    for r in cand.itertuples():
        c = best_xcorr(env, r.env)
        if c > best[1]:
            best = (r.pitt_session, c, r.dur)
    return best


def match_dataset(name: str, manifest: pd.DataFrame, root: Path, pitt: pd.DataFrame) -> pd.DataFrame:
    envs = Parallel(n_jobs=-1)(delayed(envelope)(root / p) for p in manifest["path"])
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(match_one)(d, e, pitt) for d, e in envs
    )
    out = manifest[["id", "split", "label"]].copy()
    out.insert(0, "dataset", name)
    out["dur"] = [d for d, _ in envs]
    out["pitt_session"] = [r[0] for r in results]
    out["env_corr"] = [r[1] for r in results]
    out["pitt_dur"] = [r[2] for r in results]
    out["_env"] = [e for _, e in envs]
    return out


def resolve_adresso(matches: pd.DataFrame, dx: pd.DataFrame, pitt: pd.DataFrame) -> None:
    """Use ADReSSo's own metadata, which names the Pitt session behind every ``adrso*`` id.

    Train files keep their ``adrso*`` names, so their session comes straight from the metadata
    (audio matches are checked against it). Test files were renamed ``adrsdt*``, but the metadata
    still lists which Pitt sessions went into the test set; sessions not already claimed by a
    confident audio match are assigned to the remaining unmatched test files by best envelope
    correlation, one-to-one, accepting only pairs that are each other's best choice.
    """
    pitt_dx = dx[dx["dataset"] == "Pitt"]
    by_fname = pitt_dx.set_index("adressfname")["id"]
    is_o = matches["dataset"] == "adresso"

    train = is_o & (matches["split"] == "train")
    meta_session = matches.loc[train, "id"].map(by_fname)
    audio = matches.loc[train, "pitt_session"]
    conflicts = (audio.notna() & meta_session.notna() & (audio != meta_session)).sum()
    if conflicts:
        raise SystemExit(f"{conflicts} ADReSSo train audio matches contradict dx-mmse.csv")
    fill = train & matches["pitt_session"].isna() & matches["id"].map(by_fname).notna()
    matches.loc[fill, "pitt_session"] = matches.loc[fill, "id"].map(by_fname)
    matches.loc[fill, "method"] = "metadata"

    test = is_o & (matches["split"] == "test")
    remaining = set(pitt_dx.loc[pitt_dx["test"], "id"]) - set(matches.loc[test, "pitt_session"].dropna())
    open_rows = matches.index[test & matches["pitt_session"].isna()]
    if not remaining or len(open_rows) == 0:
        return
    sessions = pitt.set_index("pitt_session").loc[sorted(remaining)]
    root_envs = {i: matches.at[i, "_env"] for i in open_rows}
    corr = pd.DataFrame(
        [[best_xcorr(root_envs[i], e) for e in sessions["env"]] for i in open_rows],
        index=open_rows, columns=sessions.index,
    )
    for i in open_rows:
        s = corr.loc[i].idxmax()
        if corr[s].idxmax() == i:
            matches.at[i, "pitt_session"] = s
            matches.at[i, "env_corr"] = corr.at[i, s]
            matches.at[i, "method"] = "elimination"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("pitt_root", type=Path)
    ap.add_argument("--adresso-meta", type=Path, default=None,
                    help="ADReSSo 'MMSE scores/dx-mmse.csv' (maps adrso ids to Pitt sessions)")
    args = ap.parse_args()
    run, pitt_root = args.run_dir.resolve(), args.pitt_root.resolve()

    pitt = load_pitt(pitt_root)
    print(f"Pitt cookie sessions: {len(pitt)} ({pitt.pitt_speaker.nunique()} speakers)")

    manifests = {d: pd.read_csv(run / f"manifest_{d}.csv", dtype={"id": str}) for d in ("adress", "adresso")}
    matches = pd.concat(
        [match_dataset(d, m, run / d, pitt) for d, m in manifests.items()], ignore_index=True
    )
    matches["matched"] = matches["env_corr"] >= MIN_CORR
    matches["method"] = np.where(matches["matched"], "audio", "none")
    matches.loc[~matches["matched"], "pitt_session"] = None
    if args.adresso_meta is not None:
        resolve_adresso(matches, pd.read_csv(args.adresso_meta), pitt)
    matches["matched"] = matches["pitt_session"].notna()
    meta = pitt.set_index("pitt_session")[["pitt_speaker", "visit", "pitt_group"]]
    matches = matches.drop(columns="_env").join(meta, on="pitt_session")
    matches["label_agrees"] = matches["pitt_group"].map(GROUP_TO_LABEL) == matches["label"]
    matches.to_csv(run / "pitt_matches.csv", index=False)
    print("assignment method by dataset:")
    print(pd.crosstab(matches["dataset"], matches["method"]).to_string())

    for d, m in manifests.items():
        sub = matches[matches["dataset"] == d].set_index("id")
        m["pitt_session"] = m["id"].map(sub["pitt_session"])
        spk = m["id"].map(sub["pitt_speaker"])
        m["speaker"] = np.where(spk.notna(), "pitt_" + spk.astype(str), f"{d}_" + m["id"])
        m.to_csv(run / f"manifest_{d}.csv", index=False)

    print(matches.groupby("dataset").agg(
        n=("id", "size"), matched=("matched", "sum"),
        label_agreement=("label_agrees", lambda s: s[matches.loc[s.index, "matched"]].mean()),
        corr_median=("env_corr", "median"),
    ).round(4).to_string())
    dupes = matches[matches.matched].groupby("dataset")["pitt_session"].apply(lambda s: s.duplicated().sum())
    print("duplicate session claims within a dataset:", dupes.to_dict())
    print(f"wrote {run / 'pitt_matches.csv'}")


if __name__ == "__main__":
    main()
