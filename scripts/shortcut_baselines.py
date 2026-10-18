"""Shortcut baselines: how well do non-speech variables alone separate the classes?

* ``age+gender`` -- ADReSS/ADReSSo and TAUKADIAL's training set were age/gender matched by design;
  the Pitt-wide benchmark and TAUKADIAL's test set are not.
* ``language`` and ``age+gender+language`` (TAUKADIAL) -- its English and Mandarin halves differ in
  MCI prevalence, recording setup, picture set, and diagnostic procedure.

MMSE is deliberately excluded: it is a cognitive test and part of the diagnosis itself. Each
baseline is a liblinear logistic regression fit on the official training split and scored on the
official test split (bootstrap CI over speakers), plus a speaker-grouped repeated CV mean inside the
training split only (what a re-split / cross-validated evaluation of the training data would see) and a
speaker-disjoint Monte Carlo mean over pooled train+test. Reports
macro-F1 and UAR (balanced accuracy; TAUKADIAL's official metric). Writes
``<results>/tables/shortcut_baselines.csv``.

Usage (from the repository root):  python scripts/shortcut_baselines.py configs/full.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sabench.config import ResolvedPaths, load_config
from sabench.modeling.bootstrap import bootstrap_ci
from sabench.modeling.metrics import liblinear_classifier, macro_f1
from sabench.modeling.preprocessing import fit_transform
from sabench.modeling.resample import monte_carlo_resample, repeated_stratified_cv


def uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]))


def columns(m: pd.DataFrame, spec: str) -> np.ndarray:
    parts = set(spec.split("+"))  # exact names: "language" must not match "age"
    cols = []
    if "age" in parts:
        cols.append(pd.to_numeric(m["age"], errors="coerce").to_numpy(float))
    if "gender" in parts:
        cols.append((m["gender"].astype(str).str.lower() == "female").to_numpy(float))
    if "language" in parts:
        cols.append((m["language"] == "en").to_numpy(float))
    return np.column_stack(cols)


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    paths = ResolvedPaths(cfg, ".")
    rows = []
    for ds in cfg.datasets:
        m = pd.read_csv(paths.manifest_path(ds), dtype={"id": str})
        specs = []
        if {"age", "gender"} <= set(m.columns) and m["age"].notna().any():
            specs.append("age+gender")
        if "language" in m.columns:
            specs += ["language", "age+gender+language"]
        groups = m["speaker"].astype(str).to_numpy() if "speaker" in m.columns else None
        tr, te = (m["split"] == "train").to_numpy(), (m["split"] == "test").to_numpy()
        y = m["label"].to_numpy()
        for spec in specs:
            X = columns(m, spec)
            Xtr, Xte = fit_transform(X[tr], X[te])
            y_pred = liblinear_classifier().fit(Xtr, y[tr]).predict(Xte)
            ci = bootstrap_ci(y[te], y_pred, n_bootstrap=cfg.uncertainty.n_bootstrap, alpha=cfg.uncertainty.alpha,
                              seed=cfg.seed, groups=None if groups is None else groups[te])
            cv_train = repeated_stratified_cv(
                X[tr], y[tr], liblinear_classifier, n_splits=cfg.uncertainty.cv_folds,
                n_repeats=cfg.uncertainty.cv_repeats, seed=cfg.seed, groups=None if groups is None else groups[tr],
            )
            mc = monte_carlo_resample(X, y, liblinear_classifier, n_splits=cfg.uncertainty.n_monte_carlo,
                                      test_size=cfg.uncertainty.test_size, seed=cfg.seed, groups=groups)
            rows.append({
                "dataset": ds, "features": spec, "test_f1": macro_f1(y[te], y_pred), "test_uar": uar(y[te], y_pred),
                "ci_low": ci.ci_low, "ci_high": ci.ci_high, "train_cv_mean": float(cv_train.mean()),
                "mc_mean": float(mc.mean()), "mc_sd": float(mc.std(ddof=1)),
                "seed": cfg.seed, "config_hash": cfg.config_hash(), "regime": "train_only",
            })
    df = pd.DataFrame(rows)
    out = paths.tables / "shortcut_baselines.csv"
    df.to_csv(out, index=False)
    print(df.drop(columns=["seed", "config_hash", "regime"]).round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
