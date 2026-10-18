from __future__ import annotations

import numpy as np
import pytest

from sabench.exceptions import SearchError
from sabench.modeling.nulls import run_null, search_aware_pvalue
from sabench.modeling.search import pair_search

_SMALL_NULL_KWARGS = {"top_k": 6, "regime": "train_only", "cv_folds": 3, "n_jobs": 1}


def test_run_null_reproducible_with_same_seed(pure_noise_synth):
    df1 = run_null(*pure_noise_synth, n_permutations=10, seed=1, **_SMALL_NULL_KWARGS)
    df2 = run_null(*pure_noise_synth, n_permutations=10, seed=1, **_SMALL_NULL_KWARGS)
    assert np.allclose(df1["test_f1"].to_numpy(), df2["test_f1"].to_numpy())


def test_run_null_different_seeds_differ(pure_noise_synth):
    df1 = run_null(*pure_noise_synth, n_permutations=10, seed=1, **_SMALL_NULL_KWARGS)
    df2 = run_null(*pure_noise_synth, n_permutations=10, seed=2, **_SMALL_NULL_KWARGS)
    assert not np.allclose(df1["test_f1"].to_numpy(), df2["test_f1"].to_numpy())


def test_run_null_max_test_f1_never_below_test_f1(synth):
    df = run_null(*synth, n_permutations=8, seed=0, **_SMALL_NULL_KWARGS)
    assert (df["max_test_f1"] >= df["test_f1"] - 1e-9).all()


def test_search_aware_pvalue_extremes():
    null_scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert search_aware_pvalue(null_scores, observed=1.0) == pytest.approx(1 / 5)
    assert search_aware_pvalue(null_scores, observed=0.0) == pytest.approx(1.0)


def test_search_aware_pvalue_rejects_empty_null():
    with pytest.raises(SearchError):
        search_aware_pvalue(np.array([]), observed=0.5)


def test_observed_score_against_its_own_null_is_plausible_under_noise(pure_noise_synth):
    """Sanity check from docs/EXPERIMENTS.md: on pure noise, the observed train_only score should not
    be an extreme outlier against its own permutation null."""
    observed = pair_search(*pure_noise_synth, top_k=6, regime="train_only", cv_folds=3, seed=0)
    null_df = run_null(*pure_noise_synth, n_permutations=30, seed=0, **_SMALL_NULL_KWARGS)
    pvalue = search_aware_pvalue(null_df["test_f1"], observed.test_f1)
    assert pvalue > 0.01  # not formally guaranteed, but should hold for genuine noise almost always
