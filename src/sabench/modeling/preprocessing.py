"""Mean imputation + z-scoring, fit on the training split only -- everywhere, including inside
inner cross-validation. This is the project's single most important hard rule: violating it is the
exact category of leakage this whole study exists to catch in someone else's benchmark.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from sabench.exceptions import SearchError

_MIN_STD = 1e-12


@dataclass(frozen=True)
class Preprocessor:
    """Per-column mean (for imputation) and standard deviation (for z-scoring), fit on one split."""

    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray) -> Preprocessor:
        X_train = np.asarray(X_train, dtype=float)
        if X_train.ndim != 2:
            raise SearchError(f"expected a 2D feature matrix, got shape {X_train.shape}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns handled below
            mean = np.nanmean(X_train, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        imputed = np.where(np.isnan(X_train), mean, X_train)
        std = imputed.std(axis=0)
        std = np.where(std < _MIN_STD, 1.0, std)
        return cls(mean_=mean, std_=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.mean_.shape[0]:
            raise SearchError(
                f"expected {self.mean_.shape[0]} columns, got {X.shape[1]}"
            )
        imputed = np.where(np.isnan(X), self.mean_, X)
        return (imputed - self.mean_) / self.std_


def fit_transform(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit impute/z-score statistics on ``X_train`` only, then transform both splits.

    This is the only preprocessing entry point library code should call -- never ``fit`` the
    training split and a held-out split independently, and never fit on data that includes rows the
    caller considers "held out" (a test split, or a validation fold inside CV).
    """
    pre = Preprocessor.fit(X_train)
    return pre.transform(X_train), pre.transform(X_test)
