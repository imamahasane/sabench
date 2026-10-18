from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sabench.exceptions import SearchError
from sabench.modeling.nulls import run_probe_null, search_aware_pvalue
from sabench.modeling.probes import ProbeResult, run_all_layers, run_layer_probe, select_c, select_layer
from sabench.pipeline.stages import Split


def test_select_c_prefers_better_regularisation(synth):
    best_c, score = select_c(synth.X_train, synth.y_train, [0.0001, 1.0, 100.0], cv_folds=3, seed=0)
    assert best_c in (0.0001, 1.0, 100.0)
    assert 0.0 <= score <= 1.0


def test_select_c_rejects_empty_candidates(synth):
    with pytest.raises(SearchError):
        select_c(synth.X_train, synth.y_train, [], cv_folds=3, seed=0)


def test_run_layer_probe_returns_fields(synth):
    result = run_layer_probe(*synth, layer=3, cs=[0.01, 1.0], cv_folds=3, seed=0)
    assert result.layer == 3
    assert result.best_c in (0.01, 1.0)
    assert 0.0 <= result.test_f1 <= 1.0


def test_run_all_layers_matches_tensor_layer_count():
    rng = np.random.default_rng(0)
    n_layers, dim = 4, 6
    X_train = rng.normal(size=(30, n_layers, dim))
    X_test = rng.normal(size=(15, n_layers, dim))
    y_train = rng.integers(0, 2, size=30)
    y_test = rng.integers(0, 2, size=15)
    results = run_all_layers(X_train, y_train, X_test, y_test, cs=[0.1, 1.0], cv_folds=3, seed=0)
    assert len(results) == n_layers
    assert [r.layer for r in results] == list(range(n_layers))


def test_run_all_layers_rejects_non_3d_input(synth):
    with pytest.raises(SearchError):
        run_all_layers(synth.X_train, synth.y_train, synth.X_test, synth.y_test, cs=[1.0], cv_folds=3)


def test_select_layer_uses_cv_score_never_test_f1():
    results = [
        ProbeResult(layer=0, best_c=1.0, cv_score=0.60, test_f1=0.95),  # best on test: must not win
        ProbeResult(layer=1, best_c=1.0, cv_score=0.80, test_f1=0.55),
        ProbeResult(layer=2, best_c=1.0, cv_score=0.80, test_f1=0.70),  # tie on CV: earliest layer wins
    ]
    assert select_layer(results).layer == 1


def test_select_c_with_one_repeat_matches_default(synth):
    assert select_c(synth.X_train, synth.y_train, [0.01, 1.0], cv_folds=3, seed=5) == select_c(
        synth.X_train, synth.y_train, [0.01, 1.0], cv_folds=3, seed=5, cv_repeats=1
    )


def _layered(split, n_layers=3):
    """Stack a 2D split into a (n, n_layers, dim) tensor whose layers differ only by noise."""
    rng = np.random.default_rng(0)
    def stack(X):
        return np.stack([X + 0.1 * rng.normal(size=X.shape) for _ in range(n_layers)], axis=1)
    return stack(split.X_train), split.y_train, stack(split.X_test), split.y_test


def test_probe_null_is_deterministic_and_detects_signal(pure_noise_synth):
    kw = {"cs": [0.1, 1.0], "n_permutations": 30, "cv_folds": 3, "seed": 4, "n_jobs": 1}
    rng = np.random.default_rng(11)
    y_tr, y_te = np.array([0, 1] * 30), np.array([0, 1] * 15)
    X_tr, X_te = rng.normal(size=(60, 20)), rng.normal(size=(30, 20))
    X_tr[:, :5] += 1.5 * y_tr[:, None]
    X_te[:, :5] += 1.5 * y_te[:, None]
    sig = _layered(Split(X_tr, y_tr, X_te, y_te))
    a, b = run_probe_null(*sig, **kw), run_probe_null(*sig, **kw)
    pd.testing.assert_frame_equal(a, b)
    observed = select_layer(run_all_layers(*sig, cs=[0.1, 1.0], cv_folds=3, seed=4)).test_f1
    assert search_aware_pvalue(a["test_f1"], observed) < 0.1

    noise = _layered(pure_noise_synth)
    null = run_probe_null(*noise, **kw)
    observed_noise = select_layer(run_all_layers(*noise, cs=[0.1, 1.0], cv_folds=3, seed=4)).test_f1
    assert search_aware_pvalue(null["test_f1"], observed_noise) > 0.05
