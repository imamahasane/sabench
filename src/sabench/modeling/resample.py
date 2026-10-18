"""Uncertainty via resampling (E2): stratified Monte Carlo 70/30 splits and repeated stratified
k-fold CV, for a *fixed* feature pair or probe configuration (the search itself is not rerun here --
that is what :mod:`sabench.modeling.nulls` does). Reports how far a fixed-split score sits from the
distribution of scores a different split of the same data would have produced.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from sabench.exceptions import SearchError
from sabench.modeling.groups import cv_splits, effective_groups, stratified_holdout
from sabench.modeling.metrics import ClassifierFactory, prepped_macro_f1
from sabench.utils.seeding import perm_seed


def monte_carlo_resample(
    X: np.ndarray,
    y: np.ndarray,
    classifier_factory: ClassifierFactory,
    *,
    n_splits: int = 100,
    test_size: float = 0.30,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Macro-F1 over ``n_splits`` independent stratified random 70/30 splits of pooled ``(X, y)``
    (speaker-disjoint when ``groups`` has repeated speakers).

    Preprocessing is re-fit on each split's training rows only (via ``prepped_macro_f1``), so no
    split can leak test-row statistics into its own score.
    """
    y = np.asarray(y)
    scores = np.empty(n_splits)
    for b in range(n_splits):
        split_seed = perm_seed(seed, b)
        tr, te = stratified_holdout(y, test_size, split_seed, groups)
        scores[b] = prepped_macro_f1(classifier_factory, X[tr], y[tr], X[te], y[te])
    return scores


def repeated_stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifier_factory: ClassifierFactory,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Macro-F1 over every fold of ``n_repeats`` independent stratified k-fold partitions
    (speaker-disjoint folds when ``groups`` has repeated speakers)."""
    y = np.asarray(y)
    if n_splits < 2:
        raise SearchError(f"n_splits must be >= 2, got {n_splits}")
    if effective_groups(groups) is None:
        folds = list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X, y))
    else:
        folds = [f for r in range(n_repeats) for f in cv_splits(y, n_splits, perm_seed(seed, r) % (2**31 - 1), groups)]
    scores = [prepped_macro_f1(classifier_factory, X[tr], y[tr], X[va], y[va]) for tr, va in folds]
    return np.array(scores)


def summarize(scores: np.ndarray, fixed_split_score: float) -> dict:
    """Mean, SD, and gap to a fixed-split score -- the three numbers E2's tables report for every
    resampling distribution."""
    return {
        "mean": float(np.mean(scores)),
        "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "n": int(len(scores)),
        "fixed_split_score": float(fixed_split_score),
        "gap_fixed_minus_mean": float(fixed_split_score - np.mean(scores)),
    }
