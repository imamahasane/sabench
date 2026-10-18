"""Replicate the base paper's Table 4: a fixed two-feature pair evaluated on silence-only and
speech-only versions of the recordings (pyannote VAD, as in the paper).

Two pairs per dataset: the paper's published Table 3 pair, and this project's own E0 (ComParE,
test-informed) pair. Each is trained on the training split of a variant and scored on the test
split of the same variant, with a bootstrap CI. Writes ``<results>/tables/table4_replication.csv``.

Usage (from the repository root):  python scripts/table4_replication.py configs/full_pyannote.yaml configs/full.yaml
(the first config points at the pyannote variant features, the second at the E0 result tables).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from sabench.config import ResolvedPaths, load_config
from sabench.modeling.bootstrap import bootstrap_ci
from sabench.modeling.metrics import liblinear_classifier, macro_f1
from sabench.modeling.preprocessing import fit_transform
from sabench.pipeline.io import align_features_to_manifest, load_lld_features, split_train_test_frame

FEATURESET = "ComParE_2016"
PAPER_PAIRS = {
    "adress": ("audSpec_Rfilt_sma[22]_quartile1", "pcm_fftMag_spectralCentroid_sma_de_quartile2"),
    "adresso": ("mfcc_sma[6]_centroid", "mfcc_sma[11]_flatness"),
}
PAPER_TABLE4 = {"adress": {"silence": 0.702, "speech": 0.643}, "adresso": {"silence": 0.631, "speech": 0.576}}


def main() -> None:
    vad_cfg, main_cfg = load_config(Path(sys.argv[1])), load_config(Path(sys.argv[2]))
    vad_paths, main_paths = ResolvedPaths(vad_cfg, "."), ResolvedPaths(main_cfg, ".")
    rows = []
    for ds in ("adress", "adresso"):
        manifest = pd.read_csv(vad_paths.manifest_path(ds), dtype={"id": str})
        cols = list(load_lld_features(vad_paths.features / ds / "full" / f"{FEATURESET}.csv").columns)
        e0 = pd.read_csv(main_paths.tables / f"e0_{ds}_{FEATURESET}.csv").iloc[0]
        pairs = {"paper_table3": PAPER_PAIRS[ds], "our_e0": (cols[int(e0.pair_i)], cols[int(e0.pair_j)])}
        for pair_name, pair in pairs.items():
            for variant in ("full", "silence", "speech"):
                feats = load_lld_features(vad_paths.features / ds / variant / f"{FEATURESET}.csv")
                Xtr, ytr, Xte, yte = split_train_test_frame(align_features_to_manifest(manifest, feats), list(pair))
                Xtr, Xte = fit_transform(Xtr, Xte)
                y_pred = liblinear_classifier().fit(Xtr, ytr).predict(Xte)
                ci = bootstrap_ci(yte, y_pred, n_bootstrap=vad_cfg.uncertainty.n_bootstrap,
                                  alpha=vad_cfg.uncertainty.alpha, seed=vad_cfg.seed)
                rows.append({
                    "dataset": ds, "pair_source": pair_name, "feature_1": pair[0], "feature_2": pair[1],
                    "variant": variant, "test_f1": macro_f1(yte, y_pred),
                    "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                    "paper_table4_f1": PAPER_TABLE4[ds].get(variant),
                    "vad": "pyannote", "seed": vad_cfg.seed, "config_hash": vad_cfg.config_hash(),
                    "regime": "test_informed",
                })
    df = pd.DataFrame(rows)
    out = vad_paths.tables / "table4_replication.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.pivot_table(index=["dataset", "pair_source"], columns="variant", values="test_f1").round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
