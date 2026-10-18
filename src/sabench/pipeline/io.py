"""Feature-file I/O and manifest-aligned train/test splitting.

Keeps disk formats (LLD CSVs, embedding ``.npz`` files) out of :mod:`sabench.modeling`, whose
functions only ever see plain NumPy arrays -- this module is the one place that knows how those
arrays get to and from disk, and it is the one place that enforces "align by id, never row position."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sabench.constants import SPLIT_TEST, SPLIT_TRAIN
from sabench.exceptions import DataAlignmentError, ResultIntegrityError
from sabench.utils.io import atomic_write_csv


def load_lld_features(path: str | Path) -> pd.DataFrame:
    """Load an LLD feature CSV written by ``sabench features lld`` (index = ``id``)."""
    df = pd.read_csv(path, index_col="id")
    df.index = df.index.astype(str)
    # pandas 3 read_csv yields one block per column; copy() consolidates (avoids fragmentation warnings).
    return df.copy()


def save_lld_features(df: pd.DataFrame, path: str | Path) -> None:
    atomic_write_csv(df.rename_axis("id"), path, index=True)


def load_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load an embedding ``.npz`` written by ``sabench features embeddings``: ``ids`` (n,), ``X``
    (n, n_layers, dim)."""
    with np.load(path, allow_pickle=False) as data:
        ids = [str(i) for i in data["ids"]]
        X = data["X"]
    return ids, X


def save_embeddings(ids: list[str], X: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A plain np.array(ids) of Python strings gets a fixed-width unicode dtype ("<U..."), not
    # `object` -- it round-trips through `allow_pickle=False`. Forcing dtype=object here would save
    # fine but make load_embeddings's allow_pickle=False unable to ever read it back.
    np.savez(p, ids=np.array(ids), X=X)


def align_features_to_manifest(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join an LLD feature table onto a manifest by id (manifest row order is preserved)."""
    from sabench.data.manifest import align_on_id

    return align_on_id(manifest, features, context="LLD features")


def align_embeddings_to_manifest(
    manifest: pd.DataFrame, ids: list[str], X: np.ndarray
) -> np.ndarray:
    """Reorder an embedding array so row ``i`` corresponds to ``manifest.iloc[i]``.

    :raises DataAlignmentError: any manifest id is missing from ``ids``.
    """
    position = {rec_id: i for i, rec_id in enumerate(ids)}
    manifest_ids = manifest["id"].tolist()
    missing = [i for i in manifest_ids if i not in position]
    if missing:
        raise DataAlignmentError(f"embeddings: {len(missing)} manifest id(s) not found, e.g. {missing[:5]}")
    reorder = [position[i] for i in manifest_ids]
    return X[reorder]


def split_train_test_frame(
    manifest_with_features: pd.DataFrame, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split an id-aligned, manifest-joined feature table into ``X_train, y_train, X_test, y_test``
    using the manifest's official ``split`` column."""
    train = manifest_with_features.loc[manifest_with_features["split"] == SPLIT_TRAIN]
    test = manifest_with_features.loc[manifest_with_features["split"] == SPLIT_TEST]
    if len(train) == 0 or len(test) == 0:
        raise DataAlignmentError(
            f"expected both a non-empty '{SPLIT_TRAIN}' and '{SPLIT_TEST}' split, got "
            f"{len(train)} train / {len(test)} test rows"
        )
    return (
        train[feature_columns].to_numpy(dtype=float),
        train["label"].to_numpy(),
        test[feature_columns].to_numpy(dtype=float),
        test["label"].to_numpy(),
    )


def split_train_test_array(
    manifest: pd.DataFrame, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a manifest-row-aligned array (e.g. embeddings) into train/test using ``manifest.split``."""
    if len(manifest) != len(X):
        raise DataAlignmentError(f"manifest has {len(manifest)} rows but array has {len(X)}")
    train_mask = (manifest["split"] == SPLIT_TRAIN).to_numpy()
    test_mask = (manifest["split"] == SPLIT_TEST).to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise DataAlignmentError("expected both non-empty train and test rows in the manifest")
    y = manifest["label"].to_numpy()
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def with_provenance_columns(df: pd.DataFrame, *, regime: str, seed: int, config_hash: str) -> pd.DataFrame:
    """Attach the three columns every result CSV must carry, per this project's hard rules."""
    out = df.copy()
    out["regime"] = regime
    out["seed"] = seed
    out["config_hash"] = config_hash
    return out


def check_result_integrity(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ResultIntegrityError(f"result table is missing required column(s): {missing}")
