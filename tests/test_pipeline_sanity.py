"""The two sanity checks docs/EXPERIMENTS.md requires to "always" run, plus an end-to-end stage
smoke test. If either sanity check ever fails, assume leakage before anything else -- this mirrors
the project's `/check-leak` discipline.
"""
from __future__ import annotations

import numpy as np
import pytest

from sabench.config import load_config
from sabench.modeling.nulls import run_null, search_aware_pvalue
from sabench.modeling.search import pair_search
from sabench.pipeline.stages import (
    Split,
    stage_embedding_pair_search,
    stage_null,
    stage_replicate,
    stage_uncertainty,
)


def test_label_leak_gives_near_perfect_score():
    """A column equal to the label must drive macro-F1 to ~1.0 under either regime."""
    rng = np.random.default_rng(0)
    n_train, n_test = 60, 30
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    noise_train = rng.normal(size=(n_train, 5))
    noise_test = rng.normal(size=(n_test, 5))
    X_train = np.column_stack([y_train.astype(float), noise_train])
    X_test = np.column_stack([y_test.astype(float), noise_test])

    for regime in ("test_informed", "train_only"):
        result = pair_search(X_train, y_train, X_test, y_test, top_k=6, regime=regime, cv_folds=3)
        assert result.test_f1 > 0.95, f"leak test failed under regime={regime}: {result.test_f1}"


def test_gaussian_noise_gives_near_chance_score_and_similar_null():
    """Pure Gaussian noise features should score near chance, and the permutation null should look
    like the observed score's own distribution (neither systematically higher nor lower)."""
    rng = np.random.default_rng(1)
    n_train, n_test, n_features = 60, 30, 20
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)

    observed = pair_search(X_train, y_train, X_test, y_test, top_k=10, regime="train_only", cv_folds=3)
    assert observed.test_f1 < 0.85  # loose bound: noise + feature search can still drift from exact 0.5

    null_df = run_null(
        X_train, y_train, X_test, y_test,
        n_permutations=40, top_k=10, regime="train_only", cv_folds=3, seed=0, n_jobs=1,
    )
    pvalue = search_aware_pvalue(null_df["test_f1"], observed.test_f1)
    assert pvalue > 0.02  # observed should not look like an extreme outlier against its own null


def test_end_to_end_stage_smoke_test():
    """E0 -> E1 -> E2 wired together on synthetic data, exactly what `sabench demo` runs."""
    cfg = load_config(overrides=["permutation_null.n_permutations=20"])
    rng = np.random.default_rng(2)
    n_train, n_test, n_features = 50, 25, 15
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    X_train[:, 0] += 1.8 * y_train
    X_test[:, 0] += 1.8 * y_test
    split = Split(X_train, y_train, X_test, y_test)

    e0 = stage_replicate(split, config=cfg, seed=cfg.seed)
    assert {"test_f1", "max_test_f1", "pair_i", "pair_j"} <= set(e0.columns)

    observed, null_df, pvalue = stage_null(split, config=cfg, regime="train_only", seed=cfg.seed)
    assert len(null_df) == 20
    assert 0.0 <= pvalue <= 1.0

    cols = list(observed.best_pair)
    restricted = Split(X_train[:, cols], y_train, X_test[:, cols], y_test)
    summary = stage_uncertainty(restricted, config=cfg, seed=cfg.seed)
    assert 0.0 <= summary["fixed_split_f1"] <= 1.0
    assert summary["bootstrap"]["ci_low"] <= summary["bootstrap"]["ci_high"]


def test_stage_embedding_pair_search_matches_stage_null():
    """E3's 'pair search over embedding dimensions, same K and null as E1' is a thin, identically-
    named wrapper over stage_null -- same base seed must give identical results."""
    cfg = load_config(overrides=["permutation_null.n_permutations=5"])
    rng = np.random.default_rng(3)
    n_train, n_test, n_dims = 40, 20, 12
    X_train = rng.normal(size=(n_train, n_dims))
    X_test = rng.normal(size=(n_test, n_dims))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    split = Split(X_train, y_train, X_test, y_test)

    expected = stage_null(split, config=cfg, regime="train_only", seed=cfg.seed)
    got = stage_embedding_pair_search(split, config=cfg, regime="train_only", seed=cfg.seed)

    assert got[0] == expected[0]
    assert np.allclose(got[1]["test_f1"].to_numpy(), expected[1]["test_f1"].to_numpy())
    assert got[2] == pytest.approx(expected[2])
