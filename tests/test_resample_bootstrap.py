from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from sabench.exceptions import SearchError
from sabench.modeling.bootstrap import bootstrap_ci
from sabench.modeling.metrics import liblinear_classifier
from sabench.modeling.resample import monte_carlo_resample, repeated_stratified_cv, summarize


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_pred = y_true.copy()
    flip = rng.choice(100, size=15, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    result = bootstrap_ci(y_true, y_pred, n_bootstrap=500, seed=1)
    assert result.ci_low <= result.point_estimate <= result.ci_high
    assert result.point_estimate == pytest.approx(f1_score(y_true, y_pred, average="macro"))


def test_bootstrap_ci_perfect_predictions_is_degenerate():
    y_true = np.array([0, 1] * 20)
    result = bootstrap_ci(y_true, y_true.copy(), n_bootstrap=200, seed=0)
    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)


def test_bootstrap_ci_rejects_mismatched_length():
    with pytest.raises(SearchError):
        bootstrap_ci(np.array([0, 1]), np.array([0, 1, 1]))


def test_bootstrap_ci_rejects_missing_class():
    with pytest.raises(SearchError):
        bootstrap_ci(np.array([0, 0, 0]), np.array([0, 0, 0]))


def test_bootstrap_ci_reproducible_with_seed():
    y_true = np.array([0, 1] * 30)
    y_pred = np.roll(y_true, 3)
    a = bootstrap_ci(y_true, y_pred, n_bootstrap=300, seed=5)
    b = bootstrap_ci(y_true, y_pred, n_bootstrap=300, seed=5)
    assert a.ci_low == b.ci_low and a.ci_high == b.ci_high


def test_monte_carlo_resample_shape_and_reproducibility(synth):
    X = np.vstack([synth.X_train, synth.X_test])
    y = np.concatenate([synth.y_train, synth.y_test])
    a = monte_carlo_resample(X, y, liblinear_classifier, n_splits=10, test_size=0.3, seed=0)
    b = monte_carlo_resample(X, y, liblinear_classifier, n_splits=10, test_size=0.3, seed=0)
    assert a.shape == (10,)
    assert np.allclose(a, b)


def test_repeated_stratified_cv_rejects_bad_n_splits(synth):
    X = np.vstack([synth.X_train, synth.X_test])
    y = np.concatenate([synth.y_train, synth.y_test])
    with pytest.raises(SearchError):
        repeated_stratified_cv(X, y, liblinear_classifier, n_splits=1, n_repeats=2, seed=0)


def test_repeated_stratified_cv_runs(synth):
    X = np.vstack([synth.X_train, synth.X_test])
    y = np.concatenate([synth.y_train, synth.y_test])
    scores = repeated_stratified_cv(X, y, liblinear_classifier, n_splits=3, n_repeats=2, seed=0)
    assert scores.shape == (6,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_summarize_reports_gap_to_fixed_split():
    scores = np.array([0.5, 0.6, 0.7])
    summary = summarize(scores, fixed_split_score=0.9)
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["gap_fixed_minus_mean"] == pytest.approx(0.3)
    assert summary["n"] == 3
