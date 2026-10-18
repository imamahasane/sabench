"""Bootstrap confidence intervals for a fixed-split macro-F1 score (E2).

Resamples *predictions*, not raw features: given the true/predicted labels a model already produced
on the official test split, it resamples test participants with replacement (stratified by true
label, so class balance is preserved in every resample) and recomputes macro-F1. No refitting, so
this is cheap enough to run at B = 2000 for every reported number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sabench.constants import LABELS
from sabench.exceptions import SearchError
from sabench.modeling.groups import stratified_bootstrap_indices
from sabench.modeling.metrics import macro_f1
from sabench.utils.seeding import perm_seed


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_bootstrap: int
    alpha: float


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI for macro-F1 over stratified resamples of ``(y_true, y_pred)``;
    with ``groups`` (one speaker key per test row) whole speakers are resampled (cluster bootstrap).

    :raises SearchError: ``y_true``/``y_pred`` have mismatched length, or a class is entirely absent
        (macro-F1 and stratified resampling are both undefined in that case).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise SearchError(f"y_true and y_pred have mismatched length: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise SearchError("cannot bootstrap an empty test set")

    class_indices = {label: np.flatnonzero(y_true == label) for label in LABELS}
    empty = [label for label, idx in class_indices.items() if len(idx) == 0]
    if empty:
        raise SearchError(f"class(es) {empty} have zero test examples; stratified bootstrap is undefined")

    point_estimate = macro_f1(y_true, y_pred)
    scores = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        rng = np.random.default_rng(perm_seed(seed, b))
        sample_idx = stratified_bootstrap_indices(y_true, rng, groups)
        scores[b] = macro_f1(y_true[sample_idx], y_pred[sample_idx])

    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point_estimate=point_estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        se=float(np.std(scores, ddof=1)),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
