from __future__ import annotations

import pandas as pd
import pytest

from sabench.config import DatasetCounts
from sabench.data.manifest import align_on_id, load_manifest, validate_counts
from sabench.exceptions import DataAlignmentError, ManifestError


def _write(tmp_path, rows, name="manifest.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_manifest_happy_path(tmp_path):
    path = _write(tmp_path, [
        {"id": "a", "split": "train", "label": 0, "path": "a.wav"},
        {"id": "b", "split": "test", "label": 1, "path": "b.wav"},
    ])
    df = load_manifest(path)
    assert list(df["id"]) == ["a", "b"]
    assert df["label"].tolist() == [0, 1]


def test_missing_file_raises():
    with pytest.raises(ManifestError):
        load_manifest("/does/not/exist.csv")


def test_missing_required_column_raises(tmp_path):
    path = _write(tmp_path, [{"id": "a", "split": "train", "path": "a.wav"}])  # no label
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_duplicate_id_raises(tmp_path):
    path = _write(tmp_path, [
        {"id": "a", "split": "train", "label": 0, "path": "a.wav"},
        {"id": "a", "split": "test", "label": 1, "path": "a2.wav"},
    ])
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_unknown_split_raises(tmp_path):
    path = _write(tmp_path, [{"id": "a", "split": "validation", "label": 0, "path": "a.wav"}])
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_unknown_label_raises(tmp_path):
    path = _write(tmp_path, [{"id": "a", "split": "train", "label": 2, "path": "a.wav"}])
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_validate_counts_ok():
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "split": ["train", "train", "test", "test"],
        "label": [0, 1, 0, 1],
    })
    expected = DatasetCounts(train={"0": 1, "1": 1}, test={"0": 1, "1": 1})
    report = validate_counts(df, expected, dataset="toy")
    assert report.ok
    assert not report.mismatches


def test_validate_counts_mismatch():
    df = pd.DataFrame({"id": ["a"], "split": ["train"], "label": [0]})
    expected = DatasetCounts(train={"0": 5, "1": 5}, test={"0": 1, "1": 1})
    report = validate_counts(df, expected, dataset="toy")
    assert not report.ok
    assert report.mismatches


def test_align_on_id_detects_missing_feature_row():
    manifest = pd.DataFrame({"id": ["a", "b", "c"]})
    features = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
    with pytest.raises(DataAlignmentError):
        align_on_id(manifest, features)


def test_align_on_id_preserves_all_nan_rows_as_present():
    """An id present in features but with all-NaN values (e.g. a too-short audio variant) must NOT
    be reported as missing -- only ids absent from `features` entirely are missing."""
    manifest = pd.DataFrame({"id": ["a", "b"]})
    features = pd.DataFrame({"id": ["a", "b"], "x": [1.0, float("nan")]})
    merged = align_on_id(manifest, features)
    assert len(merged) == 2
    assert merged.loc[merged["id"] == "b", "x"].isna().all()


def test_align_on_id_reorders_to_manifest_order():
    manifest = pd.DataFrame({"id": ["b", "a"]})
    features = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
    merged = align_on_id(manifest, features)
    assert merged["id"].tolist() == ["b", "a"]
    assert merged["x"].tolist() == [2.0, 1.0]
