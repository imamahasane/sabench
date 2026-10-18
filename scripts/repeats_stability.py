"""How many inner-CV repeats make train_only pair selection seed-stable?

For each dataset x featureset, run the train_only search at the configured K under 10 seeds for
several ``cv_repeats`` values, and report how many distinct pairs are chosen and how much the
honest test F1 moves. Writes ``results/tables/repeats_stability.csv``.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from sabench.config import ResolvedPaths, load_config
from sabench.modeling.search import pair_search
from sabench.pipeline.io import align_features_to_manifest, load_lld_features, split_train_test_frame

REPEATS = (1, 5, 10, 20)
SEEDS = tuple(range(10))


def one(split, cols, ds, fs, repeats, seed, cfg):
    *xy, groups = split
    r = pair_search(*xy, top_k=cfg.search.top_k, regime="train_only",
                    rank_method=cfg.search.rank_method, cv_folds=cfg.search.cv_folds,
                    cv_repeats=repeats, seed=seed, groups_train=groups)
    return {"dataset": ds, "featureset": fs, "cv_repeats": repeats, "seed": seed,
            "test_f1": r.test_f1, "criterion": r.criterion,
            "pair": f"{cols[r.best_pair[0]]} | {cols[r.best_pair[1]]}"}


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    paths = ResolvedPaths(cfg, base_dir=".")
    jobs = []
    for ds, fs in product(cfg.datasets, ("eGeMAPSv02", "ComParE_2016")):
        m = pd.read_csv(paths.manifest_path(ds), dtype={"id": str})
        f = load_lld_features(paths.features / ds / "full" / f"{fs}.csv")
        cols = list(f.columns)
        groups = m.loc[m["split"] == "train", "speaker"].astype(str).to_numpy() if "speaker" in m else None
        split = (*split_train_test_frame(align_features_to_manifest(m, f), cols), groups)
        jobs.extend((split, cols, ds, fs, r, s) for r in REPEATS for s in SEEDS)
    df = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(*j, cfg) for j in jobs))
    out = paths.tables / "repeats_stability.csv"
    df.to_csv(out, index=False)

    def modal_share(s: pd.Series) -> float:
        return s.value_counts().iloc[0] / len(s)

    summary = df.groupby(["dataset", "featureset", "cv_repeats"]).agg(
        n_distinct_pairs=("pair", "nunique"), modal_pair_share=("pair", modal_share),
        test_f1_mean=("test_f1", "mean"), test_f1_sd=("test_f1", "std"),
        test_f1_min=("test_f1", "min"), test_f1_max=("test_f1", "max"),
    ).round(3)
    print(summary.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
