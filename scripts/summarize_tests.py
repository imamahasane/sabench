"""Collect every search-aware significance test into families and add Holm-Bonferroni adjusted
p-values within each family:

* ``E1_lld_pair``    -- two-feature LLD search (``e1_observed_<ds>_<featureset>_<regime>``);
* ``E3_probe``       -- layer-wise probes with CV-selected layer, audio and text
                        (``e3_probe_observed_<ds>_<model>[_<variant>]``);
* ``E3_embed_pair``  -- two-dimension search inside the CV-selected embedding layer
                        (``e3_pairsearch_observed_<ds>_<model>[_<variant>]_<regime>``).

The permutation p-value floor is 1/(B+1) (~0.001 at B=1000). Writes
``<tables>/tests_summary_holm.csv``.

Usage (from the repository root):  python scripts/summarize_tests.py ../run_full/results/tables
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = r"(adress|adresso|pitt|taukadial)"
FAMILIES = {
    "E1_lld_pair": re.compile(rf"e1_observed_{DATASETS}_(eGeMAPSv02|ComParE_2016)_(test_informed|train_only)\.csv"),
    "E3_probe": re.compile(rf"e3_probe_observed_{DATASETS}_(\w+?)\.csv"),
    "E3_embed_pair": re.compile(rf"e3_pairsearch_observed_{DATASETS}_(\w+?)_(test_informed|train_only)\.csv"),
}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main() -> None:
    tables = Path(sys.argv[1])
    rows = []
    for family, pattern in FAMILIES.items():
        for f in sorted(tables.glob("*.csv")):
            m = pattern.fullmatch(f.name)
            if not m:
                continue
            obs = pd.read_csv(f).iloc[0]
            regime = m.group(3) if family != "E3_probe" else "train_only"
            rows.append({
                "family": family, "dataset": m.group(1), "features": m.group(2), "regime": regime,
                "observed_test_f1": obs["test_f1"], "pvalue": obs["pvalue"],
                "layer": obs.get("layer", np.nan),
            })
    df = pd.DataFrame(rows).sort_values(["family", "regime", "features", "dataset"]).reset_index(drop=True)
    df["pvalue_holm"] = np.nan
    for _, idx in df.groupby("family").groups.items():
        df.loc[idx, "pvalue_holm"] = holm(df.loc[idx, "pvalue"].to_numpy())
    df["significant_holm_0.05"] = df["pvalue_holm"] < 0.05
    out = tables / "tests_summary_holm.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
