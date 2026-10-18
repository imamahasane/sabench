from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sabench.config import load_config
from sabench.pipeline.stages import Split


@pytest.fixture
def synth():
    """Small synthetic classification problem: one informative column, the rest noise."""
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 60, 30, 20
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    X_train[:, 0] += 2.0 * y_train
    X_test[:, 0] += 2.0 * y_test
    return Split(X_train, y_train, X_test, y_test)


@pytest.fixture
def pure_noise_synth():
    """Same shape as `synth`, but no column carries any signal."""
    rng = np.random.default_rng(7)
    n_train, n_test, n_features = 60, 30, 20
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    return Split(X_train, y_train, X_test, y_test)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append({
            "id": f"train_{i}", "split": "train", "label": i % 2,
            "path": f"train_{i}.wav", "speaker": f"spk_{i % 4}",
        })
    for i in range(4):
        rows.append({
            "id": f"test_{i}", "split": "test", "label": i % 2,
            "path": f"test_{i}.wav", "speaker": f"spk_{i % 4}",
        })
    return pd.DataFrame(rows)
