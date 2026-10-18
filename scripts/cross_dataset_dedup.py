"""E6 with shared Pitt material removed from the evaluation set, at three levels.

ADReSS and ADReSSo are both cut from the Pitt Cookie Theft corpus. After ``scripts/match_pitt.py``
the manifests carry ``pitt_session`` (speaker + visit) and a speaker-level ``speaker`` key, so the
test side of each direction can be filtered by:

* ``none``     -- every recording (the contaminated number a naive cross-dataset study reports);
* ``session``  -- drop test recordings that are the *same recording* as a training one;
* ``speaker``  -- drop test recordings from *any speaker* on the training side (any visit).

``sabench cross-dataset`` refuses to run on overlapping speakers, correctly; this script reports
all three levels side by side, each with a bootstrap CI on the honestly-selected pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from sabench.config import ResolvedPaths, load_config
from sabench.modeling.bootstrap import bootstrap_ci
from sabench.modeling.metrics import liblinear_classifier
from sabench.modeling.preprocessing import fit_transform
from sabench.pipeline.io import align_features_to_manifest, load_lld_features
from sabench.pipeline.stages import stage_cross_dataset

LEVELS = ("none", "session", "speaker")


def run(cfg, paths, train_on, test_on, featureset, level):
    m_tr = pd.read_csv(paths.manifest_path(train_on), dtype={"id": str})
    m_te = pd.read_csv(paths.manifest_path(test_on), dtype={"id": str})
    n_before = len(m_te)
    if level == "session":
        m_te = m_te[~m_te["pitt_session"].isin(set(m_tr["pitt_session"].dropna()))]
    elif level == "speaker":
        m_te = m_te[~m_te["speaker"].isin(set(m_tr["speaker"]))]
    m_te = m_te.reset_index(drop=True)

    f_tr = load_lld_features(paths.features / train_on / "full" / f"{featureset}.csv")
    f_te = load_lld_features(paths.features / test_on / "full" / f"{featureset}.csv")
    cols = [c for c in f_tr.columns if c in f_te.columns]
    a_tr = align_features_to_manifest(m_tr, f_tr)
    a_te = align_features_to_manifest(m_te, f_te)
    X_tr, y_tr = a_tr[cols].to_numpy(float), a_tr["label"].to_numpy()
    X_te, y_te = a_te[cols].to_numpy(float), a_te["label"].to_numpy()
    res = stage_cross_dataset(X_tr, y_tr, X_te, y_te, config=cfg, seed=cfg.seed)

    pair = list(res.best_pair)
    Xtr_p, Xte_p = fit_transform(X_tr[:, pair], X_te[:, pair])
    y_pred = liblinear_classifier().fit(Xtr_p, y_tr).predict(Xte_p)
    ci = bootstrap_ci(y_te, y_pred, n_bootstrap=cfg.uncertainty.n_bootstrap,
                      alpha=cfg.uncertainty.alpha, seed=cfg.seed)
    return {
        "train_on": train_on, "test_on": test_on, "featureset": featureset, "exclusion": level,
        "n_test": len(m_te), "n_test_removed": n_before - len(m_te),
        "n_test_hc": int((y_te == 0).sum()), "n_test_ad": int((y_te == 1).sum()),
        "test_f1": res.test_f1, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
        "max_test_f1": res.max_test_f1,
        "pair_i": cols[pair[0]], "pair_j": cols[pair[1]],
        "seed": cfg.seed, "config_hash": cfg.config_hash(), "regime": "train_only",
    }


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    paths = ResolvedPaths(cfg, base_dir=".")
    rows = []
    for fs in ("eGeMAPSv02", "ComParE_2016"):
        for tr, te in (("adress", "adresso"), ("adresso", "adress")):
            for level in LEVELS:
                r = run(cfg, paths, tr, te, fs, level)
                rows.append(r)
                print(f"{fs:13s} {tr:>7s} -> {te:7s} {level:8s} n={r['n_test']:3d} "
                      f"(HC {r['n_test_hc']}, AD {r['n_test_ad']})  "
                      f"F1={r['test_f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    out = paths.tables / "e6_cross_dedup.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
