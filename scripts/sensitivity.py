"""K / seed sensitivity of the observed pair search (no permutation null).

``top_k`` is not stated in the original paper and ``train_only`` selection depends on the inner-CV
fold seed, so both are swept here: test_informed is seed-free (no CV); train_only is run for every
seed. Writes ``results/tables/sensitivity_k_seed.csv``.
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

KS = (10, 25, 50, 100)
SEEDS = tuple(range(5))


def load_split(paths, dataset, featureset):
    m = pd.read_csv(paths.manifest_path(dataset), dtype={"id": str})
    f = load_lld_features(paths.features / dataset / "full" / f"{featureset}.csv")
    aligned = align_features_to_manifest(m, f)
    cols = list(f.columns)
    groups = m.loc[m["split"] == "train", "speaker"].astype(str).to_numpy() if "speaker" in m else None
    return (*split_train_test_frame(aligned, cols), groups, cols)


def one(split, cols, dataset, featureset, k, regime, seed, cfg):
    Xtr, ytr, Xte, yte, groups = split
    r = pair_search(Xtr, ytr, Xte, yte, top_k=k, regime=regime,
                    rank_method=cfg.search.rank_method, cv_folds=cfg.search.cv_folds,
                    cv_repeats=cfg.search.cv_repeats, seed=seed, groups_train=groups)
    return {"dataset": dataset, "featureset": featureset, "top_k": k, "regime": regime, "seed": seed,
            "test_f1": r.test_f1, "criterion": r.criterion, "max_test_f1": r.max_test_f1,
            "pair": f"{cols[r.best_pair[0]]} | {cols[r.best_pair[1]]}", "config_hash": cfg.config_hash()}


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    paths = ResolvedPaths(cfg, base_dir=".")
    jobs = []
    for ds, fs in product(cfg.datasets, ("eGeMAPSv02", "ComParE_2016")):
        *split, cols = load_split(paths, ds, fs)
        for k in KS:
            jobs.append((split, cols, ds, fs, k, "test_informed", 0))
            jobs.extend((split, cols, ds, fs, k, "train_only", s) for s in SEEDS)
    rows = Parallel(n_jobs=-1)(delayed(one)(*j, cfg) for j in jobs)
    df = pd.DataFrame(rows)
    out = paths.tables / "sensitivity_k_seed.csv"
    df.to_csv(out, index=False)

    summary = (df.groupby(["dataset", "featureset", "regime", "top_k"])
                 .agg(test_f1_mean=("test_f1", "mean"), test_f1_min=("test_f1", "min"),
                      test_f1_max=("test_f1", "max"), n_distinct_pairs=("pair", "nunique"))
                 .round(3))
    print(summary.to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
