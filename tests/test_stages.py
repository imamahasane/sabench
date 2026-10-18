"""Direct unit tests for the E4/E5/E6 pipeline stages, which (unlike E0-E2) weren't exercised by any
other test file. All operate on in-memory arrays, per pipeline/stages.py's design -- no disk I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sabench.config import load_config
from sabench.pipeline.stages import (
    Split,
    speaker_overlap,
    stage_cross_dataset,
    stage_interventions,
    stage_segment_channel,
)


@pytest.fixture
def cfg():
    return load_config(overrides=["uncertainty.n_bootstrap=100"])


def _toy_split(seed: int, n_train: int = 30, n_test: int = 15, n_features: int = 6) -> Split:
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    X_train[:, 0] += 1.5 * y_train
    X_test[:, 0] += 1.5 * y_test
    return Split(X_train, y_train, X_test, y_test)


def test_stage_segment_channel_returns_one_row_per_variant(cfg):
    variant_splits = {
        "full": _toy_split(0),
        "speech": _toy_split(1),
        "silence": _toy_split(2),
        "channel": _toy_split(3),
    }
    df = stage_segment_channel(variant_splits, config=cfg, seed=cfg.seed)
    assert set(df["variant"]) == set(variant_splits)
    assert {"test_f1", "max_test_f1", "ci_low", "ci_high"} <= set(df.columns)
    assert (df["ci_low"] <= df["ci_high"]).all()
    assert (df["max_test_f1"] >= df["test_f1"] - 1e-9).all()


def test_stage_interventions_zero_delta_for_identical_splits(cfg):
    split = _toy_split(5)
    result = stage_interventions(split, split, config=cfg, seed=cfg.seed)
    assert result["delta_f1"] == pytest.approx(0.0)
    assert result["baseline_f1"] == pytest.approx(result["intervened_f1"])
    assert result["delta_ci_low"] <= 0.0 <= result["delta_ci_high"]


def test_stage_interventions_detects_degraded_intervention(cfg):
    rng = np.random.default_rng(6)
    n_train, n_test = 40, 20
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    X_train = np.column_stack([y_train.astype(float) * 3, rng.normal(size=(n_train, 4))])
    X_test = np.column_stack([y_test.astype(float) * 3, rng.normal(size=(n_test, 4))])
    baseline = Split(X_train, y_train, X_test, y_test)
    # the "intervention" destroys the informative column, imitating a successful channel-cue ablation
    X_train_bad, X_test_bad = X_train.copy(), X_test.copy()
    X_train_bad[:, 0] = rng.normal(size=n_train)
    X_test_bad[:, 0] = rng.normal(size=n_test)
    intervened = Split(X_train_bad, y_train, X_test_bad, y_test)

    result = stage_interventions(baseline, intervened, config=cfg, seed=cfg.seed)
    assert result["delta_f1"] < 0
    assert result["baseline_f1"] > result["intervened_f1"]


def test_stage_cross_dataset_runs_and_respects_top_k(cfg):
    train_split = _toy_split(7, n_train=30, n_test=15, n_features=8)
    test_split = _toy_split(8, n_train=30, n_test=15, n_features=8)
    result = stage_cross_dataset(
        train_split.X_train, train_split.y_train, test_split.X_test, test_split.y_test,
        config=cfg, seed=cfg.seed,
    )
    assert result.regime == "train_only"
    assert result.max_test_f1 >= result.test_f1 - 1e-9


def test_speaker_overlap_detects_shared_ids():
    a = pd.Series(["s1", "s2", "s3", None])
    b = pd.Series(["s3", "s4"])
    assert speaker_overlap(a, b) == {"s3"}


def test_speaker_overlap_empty_when_disjoint():
    a = pd.Series(["s1", "s2"])
    b = pd.Series(["s3", "s4"])
    assert speaker_overlap(a, b) == set()
