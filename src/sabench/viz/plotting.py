"""Figure generation for the paper: Figure 1 (null distributions) and Figure 2 (layer-wise probes).

Single-column legible, colour-blind-safe, exported as vector PDF for camera-ready figures. Uses the
``Agg`` backend explicitly so this module works in headless CI and batch jobs without a display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito colour-blind-safe palette.
COLORBLIND_SAFE_PALETTE = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "blue": "#0072B2",
    "yellow": "#F0E442",
    "black": "#000000",
}

_FIGSIZE_SINGLE_COLUMN = (3.4, 2.6)


def _new_single_column_figure():
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE_COLUMN, dpi=300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_null_distribution(
    null_scores_by_regime: dict[str, np.ndarray],
    observed_by_regime: dict[str, float],
    *,
    title: str = "",
    out_path: str | Path,
) -> Path:
    """Figure 1: histogram of the max-over-search permutation null per regime, with the observed
    (non-permuted) score marked as a vertical line."""
    fig, ax = _new_single_column_figure()
    colors = [COLORBLIND_SAFE_PALETTE["sky_blue"], COLORBLIND_SAFE_PALETTE["vermillion"]]
    for color, (regime, scores) in zip(colors, null_scores_by_regime.items(), strict=False):
        if len(scores) == 0:
            continue  # nothing to histogram; density=True would divide by zero internally
        ax.hist(scores, bins=30, alpha=0.5, color=color, label=f"null ({regime})", density=True)
        if regime in observed_by_regime:
            ax.axvline(observed_by_regime[regime], color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel("test macro-F1")
    ax.set_ylabel("density")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_layerwise_probe(
    probe_table: pd.DataFrame,
    resampled_mean: float,
    resampled_ci: tuple[float, float] | None = None,
    *,
    title: str = "",
    out_path: str | Path,
) -> Path:
    """Figure 2: layer-wise embedding macro-F1 (with a 95% CI band if ``ci_low``/``ci_high`` columns
    are present) against the resampled (Monte Carlo) mean."""
    fig, ax = _new_single_column_figure()
    layers = probe_table["layer"].to_numpy()
    scores = probe_table["test_f1"].to_numpy()
    ax.plot(layers, scores, color=COLORBLIND_SAFE_PALETTE["blue"], marker="o", markersize=3, linewidth=1.2)
    if {"ci_low", "ci_high"} <= set(probe_table.columns):
        ax.fill_between(
            layers, probe_table["ci_low"], probe_table["ci_high"],
            color=COLORBLIND_SAFE_PALETTE["blue"], alpha=0.2,
        )
    ax.axhline(resampled_mean, color=COLORBLIND_SAFE_PALETTE["vermillion"], linestyle="--", linewidth=1.2,
               label="resampled mean")
    if resampled_ci is not None:
        ax.axhspan(resampled_ci[0], resampled_ci[1], color=COLORBLIND_SAFE_PALETTE["vermillion"], alpha=0.1)
    ax.set_xlabel("layer")
    ax.set_ylabel("macro-F1")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p
