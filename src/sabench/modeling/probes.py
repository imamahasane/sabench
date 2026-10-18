"""Layer-wise regularised logistic-regression probes on frozen embeddings (E3).

``C`` and the reported CV score are always chosen by cross-validation *inside the training split
only* -- the test split is touched exactly once, to report the final number. Uses an explicit CV loop
rather than ``LogisticRegressionCV(scoring="f1_macro")``, which raises on recent scikit-learn
versions.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from sabench.exceptions import SearchError
from sabench.modeling.groups import cv_splits
from sabench.modeling.metrics import prepped_macro_f1
from sabench.utils.seeding import perm_seed

_DEFAULT_MAX_ITER = 1000


def _classifier_factory(c: float) -> Callable[[], LogisticRegression]:
    def factory() -> LogisticRegression:
        return LogisticRegression(C=c, solver="lbfgs", max_iter=_DEFAULT_MAX_ITER)

    return factory


@dataclass(frozen=True)
class ProbeResult:
    layer: int
    best_c: float
    cv_score: float
    test_f1: float


def select_c(
    X_train: np.ndarray, y_train: np.ndarray, cs: list[float], *, cv_folds: int, seed: int,
    cv_repeats: int = 1, groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick the regularisation strength with the best stratified-CV macro-F1 inside the training
    split, re-fitting impute/z-score per fold (no test data involved). ``cv_repeats`` averages over
    that many independent shufflings; repeat 0 uses ``seed`` itself."""
    y_train = np.asarray(y_train)
    if not cs:
        raise SearchError("cs must be a non-empty list of candidate regularisation strengths")
    if cv_repeats < 1:
        raise SearchError(f"cv_repeats must be >= 1, got {cv_repeats}")
    folds = []
    for r in range(cv_repeats):
        fold_seed = seed if r == 0 else perm_seed(seed, r) % (2**31 - 1)
        folds.extend(cv_splits(y_train, cv_folds, fold_seed, groups))

    best_c, best_score = cs[0], -np.inf
    for c in cs:
        factory = _classifier_factory(c)
        scores = [
            prepped_macro_f1(factory, X_train[tr], y_train[tr], X_train[va], y_train[va])
            for tr, va in folds
        ]
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_c, best_score = c, mean_score
    return best_c, best_score


def run_layer_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    layer: int,
    cs: list[float],
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: int = 0,
    groups_train: np.ndarray | None = None,
) -> ProbeResult:
    """Fit and score a single layer's probe: select ``C`` by inner CV, then one fixed-split fit."""
    best_c, cv_score = select_c(
        X_train, y_train, cs, cv_folds=cv_folds, seed=seed, cv_repeats=cv_repeats, groups=groups_train
    )
    test_f1 = prepped_macro_f1(_classifier_factory(best_c), X_train, y_train, X_test, y_test)
    return ProbeResult(layer=layer, best_c=best_c, cv_score=cv_score, test_f1=test_f1)


def run_all_layers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cs: list[float],
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: int = 0,
    groups_train: np.ndarray | None = None,
) -> list[ProbeResult]:
    """Run :func:`run_layer_probe` on every layer of a ``(n, n_layers, dim)`` embedding tensor."""
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise SearchError(
            f"expected (n, n_layers, dim) embeddings, got shapes {X_train.shape}/{X_test.shape}"
        )
    n_layers = X_train.shape[1]
    return [
        run_layer_probe(
            X_train[:, layer, :], y_train, X_test[:, layer, :], y_test,
            layer=layer, cs=cs, cv_folds=cv_folds, cv_repeats=cv_repeats, seed=seed, groups_train=groups_train,
        )
        for layer in range(n_layers)
    ]


def select_layer(results: list[ProbeResult]) -> ProbeResult:
    """The honestly-selected layer: best inner-CV score, never test F1 (ties -> earliest layer).
    Picking the layer by test F1 is exactly the test-informed selection E1 quantifies for LLDs."""
    return max(results, key=lambda r: (r.cv_score, -r.layer))
