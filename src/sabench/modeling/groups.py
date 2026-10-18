"""Speaker-aware resampling. When a dataset has several recordings per speaker (TAUKADIAL: three
picture descriptions each), treating recordings as independent is wrong in three places: inner CV
puts one speaker in both training and validation folds (so selection can reward speaker identity),
a permutation null that shuffles recording labels breaks a speaker's shared diagnosis, and a
recording-level bootstrap understates uncertainty. Every split, permutation, and resample in
:mod:`sabench.modeling` therefore goes through these helpers.

``groups`` is one speaker key per row. When every speaker has a single row, grouping is a no-op and
:func:`effective_groups` returns ``None``, so the ungrouped code paths -- and every result computed
with them -- stay exactly as they were.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from sabench.exceptions import SearchError


class SplitGroups(NamedTuple):
    train: np.ndarray | None
    test: np.ndarray | None


def effective_groups(groups: np.ndarray | None) -> np.ndarray | None:
    """``groups`` if any speaker has more than one row, else ``None`` (grouping would be a no-op)."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    return groups if len(np.unique(groups)) < len(groups) else None


def _group_labels(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique groups and the single label each carries; a group spanning both labels is an error."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    labels = np.empty(len(uniq), dtype=np.asarray(y).dtype)
    for g in range(len(uniq)):
        vals = np.unique(np.asarray(y)[inverse == g])
        if len(vals) != 1:
            raise SearchError(f"group {uniq[g]!r} carries more than one label {vals.tolist()}")
        labels[g] = vals[0]
    return labels, inverse


def cv_splits(y: np.ndarray, n_splits: int, seed: int, groups: np.ndarray | None = None) -> list:
    """Stratified k-fold splits; speaker-disjoint (StratifiedGroupKFold) when groups are non-trivial."""
    groups = effective_groups(groups)
    if groups is None:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def permute_labels(y: np.ndarray, rng: np.random.Generator, groups: np.ndarray | None = None) -> np.ndarray:
    """A label permutation for the null: shuffles rows when ungrouped, otherwise shuffles the
    per-speaker labels so every speaker's recordings keep one shared (permuted) label."""
    groups = effective_groups(groups)
    if groups is None:
        return rng.permutation(y)
    labels, inverse = _group_labels(y, groups)
    return rng.permutation(labels)[inverse]


def stratified_bootstrap_indices(
    y: np.ndarray, rng: np.random.Generator, groups: np.ndarray | None = None
) -> np.ndarray:
    """Row indices of one bootstrap resample, stratified by label; resamples whole speakers when
    grouped (cluster bootstrap)."""
    y = np.asarray(y)
    groups = effective_groups(groups)
    if groups is None:
        return np.concatenate([rng.choice(np.flatnonzero(y == c), size=int((y == c).sum()), replace=True)
                               for c in np.unique(y)])
    labels, inverse = _group_labels(y, groups)
    rows_of = [np.flatnonzero(inverse == g) for g in range(len(labels))]
    picked = np.concatenate([rng.choice(np.flatnonzero(labels == c), size=int((labels == c).sum()), replace=True)
                             for c in np.unique(labels)])
    return np.concatenate([rows_of[g] for g in picked])


def stratified_holdout(
    y: np.ndarray, test_size: float, seed: int, groups: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One stratified train/test split of row indices; speaker-disjoint when grouped."""
    y = np.asarray(y)
    groups = effective_groups(groups)
    if groups is None:
        return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    labels, inverse = _group_labels(y, groups)
    g_train, g_test = train_test_split(np.arange(len(labels)), test_size=test_size, stratify=labels,
                                       random_state=seed)
    return np.flatnonzero(np.isin(inverse, g_train)), np.flatnonzero(np.isin(inverse, g_test))
