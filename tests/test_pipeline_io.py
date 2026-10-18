from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sabench.exceptions import DataAlignmentError, ResultIntegrityError
from sabench.pipeline.io import (
    align_embeddings_to_manifest,
    check_result_integrity,
    split_train_test_array,
    split_train_test_frame,
    with_provenance_columns,
)


def test_align_embeddings_to_manifest_reorders():
    manifest = pd.DataFrame({"id": ["b", "a", "c"]})
    ids = ["a", "b", "c"]
    X = np.array([[1.0], [2.0], [3.0]])  # a=1, b=2, c=3
    out = align_embeddings_to_manifest(manifest, ids, X)
    assert np.allclose(out.ravel(), [2.0, 1.0, 3.0])


def test_align_embeddings_to_manifest_raises_on_missing_id():
    manifest = pd.DataFrame({"id": ["a", "missing"]})
    with pytest.raises(DataAlignmentError):
        align_embeddings_to_manifest(manifest, ["a"], np.array([[1.0]]))


def test_split_train_test_frame():
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "split": ["train", "train", "test", "test"],
        "label": [0, 1, 0, 1],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    Xtr, ytr, Xte, yte = split_train_test_frame(df, ["f1"])
    assert Xtr.tolist() == [[1.0], [2.0]]
    assert ytr.tolist() == [0, 1]
    assert Xte.tolist() == [[3.0], [4.0]]
    assert yte.tolist() == [0, 1]


def test_split_train_test_frame_requires_both_splits():
    df = pd.DataFrame({"id": ["a"], "split": ["train"], "label": [0], "f1": [1.0]})
    with pytest.raises(DataAlignmentError):
        split_train_test_frame(df, ["f1"])


def test_split_train_test_array_matches_manifest_length():
    manifest = pd.DataFrame({
        "id": ["a", "b", "c"], "split": ["train", "train", "test"], "label": [0, 1, 0],
    })
    X = np.zeros((2, 3))
    with pytest.raises(DataAlignmentError):
        split_train_test_array(manifest, X)


def test_with_provenance_columns():
    df = pd.DataFrame({"x": [1]})
    out = with_provenance_columns(df, regime="train_only", seed=7, config_hash="abc123")
    assert out["regime"].iloc[0] == "train_only"
    assert out["seed"].iloc[0] == 7
    assert out["config_hash"].iloc[0] == "abc123"
    assert "x" in out.columns  # original data untouched


def test_check_result_integrity_raises_on_missing_column():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ResultIntegrityError):
        check_result_integrity(df, ("regime", "seed", "config_hash"))
