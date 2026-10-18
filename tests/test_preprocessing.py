from __future__ import annotations

import numpy as np
import pytest

from sabench.exceptions import SearchError
from sabench.modeling.preprocessing import Preprocessor, fit_transform


def test_fit_transform_uses_train_statistics_only():
    X_train = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    X_test = np.array([[100.0, np.nan]])
    Xtr, Xte = fit_transform(X_train, X_test)
    assert not np.isnan(Xtr).any()
    assert not np.isnan(Xte).any()
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-9)
    assert Xte[0, 0] > 10  # far outside train's range once scaled by train stats


def test_all_nan_column_becomes_zero_not_nan():
    X = np.array([[np.nan, 1.0], [np.nan, 1.0], [np.nan, 1.0]])
    Xtr, Xte = fit_transform(X, X)
    assert np.isfinite(Xtr).all()
    assert np.isfinite(Xte).all()


def test_constant_column_does_not_divide_by_zero():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Xtr, _ = fit_transform(X, X)
    assert np.isfinite(Xtr).all()
    assert np.allclose(Xtr[:, 0], 0.0)  # constant column maps to 0 after (x - mean)


def test_transform_rejects_wrong_column_count():
    pre = Preprocessor.fit(np.ones((5, 3)))
    with pytest.raises(SearchError):
        pre.transform(np.ones((5, 2)))


def test_test_split_never_influences_train_statistics():
    """The defining leakage test: an extreme value in the test split must not shift the training
    split's z-scored mean away from zero."""
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_test = np.array([[1e6]])
    Xtr, _ = fit_transform(X_train, X_test)
    assert abs(Xtr.mean()) < 1e-9
