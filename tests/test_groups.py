from __future__ import annotations

import numpy as np
import pytest

from sabench.exceptions import SearchError
from sabench.modeling.bootstrap import bootstrap_ci
from sabench.modeling.groups import (
    cv_splits,
    effective_groups,
    permute_labels,
    stratified_bootstrap_indices,
    stratified_holdout,
)
from sabench.modeling.search import pair_search


def _clustered(n_speakers=40, per_speaker=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n_speakers // 2))
    groups = np.repeat([f"spk{i:03d}" for i in range(n_speakers)], per_speaker)
    y = np.repeat(labels, per_speaker)
    return y, groups, rng


def test_effective_groups_is_none_when_every_speaker_has_one_row():
    assert effective_groups(np.array(["a", "b", "c"])) is None
    assert effective_groups(None) is None
    assert effective_groups(np.array(["a", "a", "b"])) is not None


def test_ungrouped_splits_match_plain_stratified_kfold():
    from sklearn.model_selection import StratifiedKFold
    y = np.array([0, 1] * 15)
    expected = list(StratifiedKFold(5, shuffle=True, random_state=3).split(np.zeros(30), y))
    got = cv_splits(y, 5, 3, groups=np.arange(30).astype(str))  # unique speakers -> ungrouped path
    for (a, b), (c, d) in zip(expected, got, strict=True):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


def test_grouped_cv_never_splits_a_speaker():
    y, groups, _ = _clustered()
    for tr, va in cv_splits(y, 5, 1, groups):
        assert not set(groups[tr]) & set(groups[va])


def test_grouped_permutation_keeps_one_label_per_speaker():
    y, groups, rng = _clustered()
    perm = permute_labels(y, rng, groups)
    for g in np.unique(groups):
        assert len(np.unique(perm[groups == g])) == 1
    assert perm.sum() == y.sum()
    assert not np.array_equal(perm, y)


def test_grouped_bootstrap_resamples_whole_speakers():
    y, groups, rng = _clustered()
    idx = stratified_bootstrap_indices(y, rng, groups)
    counts = {g: int((groups[idx] == g).sum()) for g in np.unique(groups[idx])}
    assert all(c % 3 == 0 for c in counts.values())
    assert len(idx) == len(y)


def test_grouped_holdout_is_speaker_disjoint():
    y, groups, _ = _clustered()
    tr, te = stratified_holdout(y, 0.3, 5, groups)
    assert not set(groups[tr]) & set(groups[te])
    assert len(tr) + len(te) == len(y)


def test_group_spanning_two_labels_is_rejected():
    with pytest.raises(SearchError):
        permute_labels(np.array([0, 1, 1, 1]), np.random.default_rng(0), np.array(["a", "a", "b", "b"]))


def test_cluster_bootstrap_ci_is_wider_than_row_bootstrap_for_clustered_predictions():
    y, groups, rng = _clustered(n_speakers=40)
    speaker_correct = rng.random(40) < 0.7
    y_pred = np.where(np.repeat(speaker_correct, 3), y, 1 - y)
    row = bootstrap_ci(y, y_pred, n_bootstrap=500, seed=0)
    cluster = bootstrap_ci(y, y_pred, n_bootstrap=500, seed=0, groups=groups)
    assert (cluster.ci_high - cluster.ci_low) > (row.ci_high - row.ci_low)


def test_pair_search_accepts_groups(synth):
    groups = np.repeat(np.arange(len(synth.y_train) // 2), 2).astype(str)
    y_tr = np.repeat(synth.y_train[::2], 2)
    r = pair_search(synth.X_train, y_tr, synth.X_test, synth.y_test, top_k=6, regime="train_only",
                    cv_folds=3, groups_train=groups)
    assert r.n_pairs == 15
