"""Thin, reusable fit-predict-score helpers shared by search, resampling, bootstrap, and probes.

Centralising "preprocess on train, fit, predict, score macro-F1" in one place means every consumer
gets the same leakage-safe behaviour for free, instead of each module reimplementing (and risking
subtly breaking) the same four lines.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Protocol

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import _liblinear

from sabench.modeling.preprocessing import fit_transform


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


ClassifierFactory = Callable[[], Classifier]


class FastLiblinearLR:
    """``LogisticRegression(solver="liblinear")`` with default parameters, minus sklearn's per-call
    Python overhead (input validation, dataframe detection), which is ~90% of a fit on the tiny
    2-feature problems the pair search solves hundreds of thousands of times.

    Calls the same ``liblinear.train_wrap`` routine with the same arguments sklearn passes (primal
    L2 logistic regression, solver type 0, tol=1e-4, C=1, intercept_scaling=1, max_iter=100), so
    coefficients and predictions are bit-identical; ``tests/test_search.py`` pins that equality, so
    an incompatible sklearn upgrade fails loudly instead of silently changing results.
    """

    _SOLVER_L2R_LR_PRIMAL = 0
    _TOL = 1e-4
    _C = 1.0
    _INTERCEPT_SCALING = 1.0
    _MAX_ITER = 100

    def fit(self, X: np.ndarray, y: np.ndarray) -> FastLiblinearLR:
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            # Defer anything but the binary case to sklearn itself (and its error messages).
            ref = LogisticRegression(solver="liblinear").fit(X, y)
            self.coef_, self.intercept_, self.classes_ = ref.coef_, ref.intercept_, ref.classes_
            return self
        y_ind = (y == self.classes_[1]).astype(np.float64)
        _liblinear.set_verbosity_wrap(0)  # process-global flag; sklearn resets it before every fit
        raw_coef, n_iter = _liblinear.train_wrap(
            X, y_ind, False, self._SOLVER_L2R_LR_PRIMAL, self._TOL, self._INTERCEPT_SCALING,
            self._C, np.ones(2), self._MAX_ITER, 0, 0.1, np.ones(X.shape[0]),
        )
        if max(n_iter) >= self._MAX_ITER:
            warnings.warn("Liblinear failed to converge, increase the number of iterations.",
                          ConvergenceWarning, stacklevel=2)
        self.coef_ = raw_coef[:, :-1]
        self.intercept_ = self._INTERCEPT_SCALING * raw_coef[:, -1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.asarray(X, dtype=np.float64) @ self.coef_.T + self.intercept_
        if scores.shape[1] == 1:
            return self.classes_[(scores.ravel() > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]


def liblinear_classifier() -> FastLiblinearLR:
    """``liblinear`` logistic regression -- the classifier the original paper's pair search used,
    and this project's default for anything that isn't an embedding probe."""
    return FastLiblinearLR()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the labels present in ``y_true`` or ``y_pred``, with 0 for a label whose F1 is
    undefined -- numerically identical to ``f1_score(average="macro")`` (which treats 0/0 as 0), but
    without sklearn's per-call input validation, which dominated pair-search runtime."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.union1d(y_true, y_pred)
    scores = []
    for label in labels:
        t = y_true == label
        p = y_pred == label
        tp = np.count_nonzero(t & p)
        denom = np.count_nonzero(t) + np.count_nonzero(p)
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def fit_predict_macro_f1(
    classifier: Classifier, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Fit ``classifier`` on already-preprocessed training data and score macro-F1 on the test data."""
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return macro_f1(y_test, y_pred)


def prepped_macro_f1(
    classifier_factory: ClassifierFactory,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Impute/z-score (fit on ``X_train_raw`` only), fit a fresh classifier, and score macro-F1.

    This is the standard "scorer" callback passed to :mod:`sabench.modeling.resample` and
    :mod:`sabench.modeling.bootstrap`: every call re-fits preprocessing on whatever subset of rows
    the caller currently considers "training," so resampled or cross-validated scores never leak
    statistics from rows outside that subset.
    """
    Xtr, Xte = fit_transform(X_train_raw, X_test_raw)
    return fit_predict_macro_f1(classifier_factory(), Xtr, y_train, Xte, y_test)
