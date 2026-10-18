"""Single-feature ranking and pairwise feature search -- the mechanism the search-aware null (E1)
re-runs end to end on every permutation, so its correctness is the single most load-bearing piece of
code in this project.

Two selection regimes (see ``docs/EXPERIMENTS.md``):

* ``test_informed``: features are ranked using the test split and the pair is chosen by test
  macro-F1. Replicates the original paper's analysis and exposes how much selection inflates scores.
* ``train_only``: features are ranked using the training split and the pair is chosen by stratified
  CV macro-F1 computed *inside* the training split. The test split is touched only to report the
  final score of the already-chosen pair.

Preprocessing (impute + z-score) is always fit on the training split only, including inside the
``train_only`` regime's inner CV -- each fold re-fits its own statistics from that fold's training
rows (see ``sabench.modeling.preprocessing``). An earlier version of this code fit preprocessing once
on the whole training split before the inner CV split, which let each validation fold leak into its
own normalisation statistics; see ``docs/DECISIONS.md`` for the fix.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from sabench.constants import REGIMES
from sabench.exceptions import SearchError
from sabench.modeling.groups import cv_splits
from sabench.modeling.metrics import fit_predict_macro_f1, liblinear_classifier, prepped_macro_f1
from sabench.modeling.preprocessing import fit_transform
from sabench.utils.seeding import perm_seed

RankMethod = Literal["smd", "lr"]
Regime = Literal["test_informed", "train_only"]


def rank_single_smd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised per-column absolute standardised mean difference (pooled SD). Fast and monotone
    with univariate separability; the default ranking method for the pair search."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    a, b = X[y == 1], X[y == 0]
    if len(a) < 2 or len(b) < 2:
        return np.zeros(X.shape[1])
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1))
        / (len(a) + len(b) - 2)
    )
    pooled = np.where(pooled < 1e-12, np.inf, pooled)
    return np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled


def rank_single_lr(
    X: np.ndarray, y: np.ndarray, *, cv_folds: int, seed: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Per-column stratified-CV macro-F1 of a single-feature logistic regression. Slower than SMD
    (one CV loop per column) but ranks by the same objective the pair search ultimately optimises;
    used to validate that SMD ranking is a reasonable fast proxy for it."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scores = np.empty(X.shape[1])
    folds = cv_splits(y, cv_folds, seed, groups)
    for j in range(X.shape[1]):
        col = X[:, [j]]
        fold_scores = []
        for tr, va in folds:
            fold_scores.append(
                prepped_macro_f1(liblinear_classifier, col[tr], y[tr], col[va], y[va])
            )
        scores[j] = float(np.mean(fold_scores))
    return scores


def rank_single(
    X: np.ndarray, y: np.ndarray, *, method: RankMethod = "smd", cv_folds: int = 5, seed: int = 0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    if method == "smd":
        return rank_single_smd(X, y)
    if method == "lr":
        return rank_single_lr(X, y, cv_folds=cv_folds, seed=seed, groups=groups)
    raise SearchError(f"unknown rank method {method!r}; expected 'smd' or 'lr'")


def rank_overlap(scores_a: np.ndarray, scores_b: np.ndarray, top_k: int) -> float:
    """Jaccard overlap between the top-``top_k`` columns of two ranking score vectors. Used to check
    that the fast SMD ranking selects nearly the same columns as the LR ranking on real labels."""
    k = min(top_k, len(scores_a))
    set_a = set(np.argsort(-scores_a, kind="stable")[:k].tolist())
    set_b = set(np.argsort(-scores_b, kind="stable")[:k].tolist())
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


@dataclass(frozen=True)
class _Fold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def _inner_cv_folds(
    X_raw: np.ndarray, y: np.ndarray, folds: int, seed: int, repeats: int, groups: np.ndarray | None = None
) -> list[_Fold]:
    """Stratified k-fold splits over ``repeats`` independent shufflings, with impute/z-score fit on
    each fold's training rows only. Preprocessing is per-column, so fitting it once on every
    candidate column and slicing a pair out afterwards equals fitting it on that pair alone.

    Repeat 0 uses ``seed`` itself, so ``repeats=1`` reproduces single-shuffle CV exactly; averaging
    over more repeats makes the selected pair far less dependent on one arbitrary fold assignment.
    """
    out = []
    for r in range(repeats):
        fold_seed = seed if r == 0 else perm_seed(seed, r) % (2**31 - 1)
        for tr, va in cv_splits(y, folds, fold_seed, groups):
            Xtr, Xva = fit_transform(X_raw[tr], X_raw[va])
            out.append(_Fold(Xtr, y[tr], Xva, y[va]))
    return out


def _cv_f1(cv_folds: list[_Fold], cols: list[int], classifier) -> float:
    return float(np.mean([
        fit_predict_macro_f1(classifier, f.X_train[:, cols], f.y_train, f.X_val[:, cols], f.y_val)
        for f in cv_folds
    ]))


@dataclass(frozen=True)
class PairSearchResult:
    best_pair: tuple[int, int]
    criterion: float
    test_f1: float
    max_test_f1: float
    n_pairs: int
    regime: Regime


def pair_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    top_k: int = 50,
    regime: Regime = "train_only",
    rank_method: RankMethod = "smd",
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: int = 0,
    groups_train: np.ndarray | None = None,
) -> PairSearchResult:
    """Run the full two-feature search and return the chosen pair and its scores.

    ``max_test_f1`` is an oracle quantity (the best test macro-F1 over every pair considered) kept
    alongside the honestly-chosen ``test_f1`` so the gap between them can be reported directly; the
    two coincide under ``test_informed``, where the pair is chosen by test performance.

    ``cv_repeats`` (``train_only`` only) averages the selection criterion over that many independent
    k-fold shufflings. ``groups_train`` (one speaker key per training row) makes the inner CV
    speaker-disjoint; see :mod:`sabench.modeling.groups`.
    """
    if regime not in REGIMES:
        raise SearchError(f"regime must be one of {REGIMES}, got {regime!r}")
    if cv_repeats < 1:
        raise SearchError(f"cv_repeats must be >= 1, got {cv_repeats}")
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    X_train_raw = np.asarray(X_train, dtype=float)
    Xtr, Xte = fit_transform(X_train, X_test)

    rank_scores = (
        rank_single(Xte, y_test, method=rank_method, cv_folds=cv_folds, seed=seed)
        if regime == "test_informed"
        else rank_single(Xtr, y_train, method=rank_method, cv_folds=cv_folds, seed=seed, groups=groups_train)
    )
    k = min(top_k, Xtr.shape[1])
    if k < 2:
        raise SearchError(f"need at least 2 features to search pairs, have {Xtr.shape[1]}")
    top = np.argsort(-rank_scores, kind="stable")[:k]
    top_pos = {col: pos for pos, col in enumerate(top.tolist())}
    inner = (
        _inner_cv_folds(X_train_raw[:, top], y_train, cv_folds, seed, cv_repeats, groups_train)
        if regime == "train_only"
        else []
    )

    best_pair: tuple[int, int] | None = None
    best_criterion = -np.inf
    best_test_f1 = -np.inf
    max_test_f1 = -np.inf
    n_pairs = 0

    classifier = liblinear_classifier()
    for i, j in combinations(top.tolist(), 2):
        cols = [i, j]
        test_f1 = fit_predict_macro_f1(classifier, Xtr[:, cols], y_train, Xte[:, cols], y_test)
        max_test_f1 = max(max_test_f1, test_f1)
        criterion = (
            test_f1 if regime == "test_informed"
            else _cv_f1(inner, [top_pos[i], top_pos[j]], classifier)
        )
        n_pairs += 1
        if criterion > best_criterion:
            best_criterion, best_pair, best_test_f1 = criterion, (i, j), test_f1

    if best_pair is None:
        raise SearchError("pair search considered zero pairs; top_k must be >= 2")

    return PairSearchResult(
        best_pair=best_pair,
        criterion=float(best_criterion),
        test_f1=float(best_test_f1),
        max_test_f1=float(max_test_f1),
        n_pairs=n_pairs,
        regime=regime,
    )
