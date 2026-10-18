from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from sabench.exceptions import SearchError
from sabench.modeling.metrics import fit_predict_macro_f1, liblinear_classifier, macro_f1
from sabench.modeling.preprocessing import fit_transform
from sabench.modeling.search import (
    _cv_f1,
    _inner_cv_folds,
    pair_search,
    rank_overlap,
    rank_single_smd,
)


def test_rank_single_smd_finds_informative_column():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 50)
    X = rng.normal(size=(100, 10))
    X[:, 3] += 2.0 * y
    assert int(np.argmax(rank_single_smd(X, y))) == 3


def test_rank_overlap_identical_is_one():
    scores = np.array([3.0, 1.0, 2.0, 0.5])
    assert rank_overlap(scores, scores, top_k=2) == 1.0


def test_rank_overlap_disjoint_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert rank_overlap(a, b, top_k=2) == 0.0


def test_cv_f1_preprocesses_within_each_fold_only():
    """Regression test for the preprocessing leak this project treats as a hard rule
    violation: the inner CV must re-fit impute/z-score per fold, never once on the whole array. Also
    pins that preprocessing every candidate column per fold and slicing a pair out afterwards equals
    preprocessing that pair alone (the fold cache's correctness premise)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = rng.integers(0, 2, size=40)
    folds, seed, pair = 5, 3, [1, 3]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    expected = []
    for tr, va in skf.split(X, y):
        Xtr, Xva = fit_transform(X[tr][:, pair], X[va][:, pair])
        expected.append(fit_predict_macro_f1(liblinear_classifier(), Xtr, y[tr], Xva, y[va]))

    cache = _inner_cv_folds(X, y, folds, seed, repeats=1)
    got = _cv_f1(cache, pair, liblinear_classifier())
    assert got == pytest.approx(float(np.mean(expected)), abs=1e-12)


def test_inner_cv_repeats_extend_the_first_shuffle():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = np.array([0, 1] * 15)
    one = _inner_cv_folds(X, y, 3, 7, repeats=1)
    many = _inner_cv_folds(X, y, 3, 7, repeats=4)
    assert len(many) == 12
    for a, b in zip(one, many[:3], strict=True):
        np.testing.assert_array_equal(a.y_val, b.y_val)
        np.testing.assert_array_equal(a.X_val, b.X_val)
    assert not np.array_equal(many[0].X_val, many[3].X_val)  # repeat 1 is a different shuffle


def test_pair_search_rejects_nonpositive_cv_repeats(synth):
    with pytest.raises(SearchError):
        pair_search(*synth, regime="train_only", cv_repeats=0)


def test_fast_liblinear_is_bit_identical_to_sklearn():
    rng = np.random.default_rng(3)
    cases = []
    for _ in range(300):
        n, d = int(rng.integers(10, 200)), int(rng.integers(1, 4))
        X = rng.normal(size=(n, d)) * rng.uniform(0.1, 10)
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        X[:, 0] += rng.uniform(0, 4) * y
        cases.append((X, y))
    sep = np.r_[np.zeros(20), np.ones(20)]
    cases.append((np.c_[sep * 10 - 5, np.zeros(40)], sep.astype(int)))  # separable + constant column
    cases.append((rng.normal(size=(30, 2)), np.r_[np.zeros(27), np.ones(3)].astype(int)))  # unbalanced
    cases.append((rng.normal(size=(30, 2)), np.r_[np.full(15, 3), np.full(15, 7)]))  # non-0/1 labels
    for X, y in cases:
        Xv = np.random.default_rng(0).normal(size=(25, X.shape[1])) * X.std()
        ref = LogisticRegression(solver="liblinear").fit(X, y)
        fast = liblinear_classifier().fit(X, y)
        np.testing.assert_array_equal(fast.coef_, ref.coef_)
        np.testing.assert_array_equal(fast.intercept_, ref.intercept_)
        np.testing.assert_array_equal(fast.predict(Xv), ref.predict(Xv))
        np.testing.assert_array_equal(fast.predict(X), ref.predict(X))


def test_fast_liblinear_rejects_single_class_like_sklearn():
    with pytest.raises(ValueError):
        liblinear_classifier().fit(np.zeros((5, 2)), np.ones(5, dtype=int))


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(2)
    cases = [(rng.integers(0, 2, 37), rng.integers(0, 2, 37)) for _ in range(200)]
    cases += [
        (np.array([0, 1, 1, 0]), np.array([1, 1, 1, 1])),  # one class never predicted
        (np.array([1, 1, 1]), np.array([1, 1, 1])),  # single label present
        (np.array([1, 1, 1]), np.array([0, 0, 0])),  # disjoint labels
    ]
    for y_true, y_pred in cases:
        assert macro_f1(y_true, y_pred) == pytest.approx(
            f1_score(y_true, y_pred, average="macro", zero_division=0), abs=1e-12
        )


def test_pair_search_rejects_unknown_regime(synth):
    with pytest.raises(SearchError):
        pair_search(*synth, regime="nope")


def test_pair_search_rejects_top_k_below_two(synth):
    with pytest.raises(SearchError):
        pair_search(*synth, top_k=1)


def test_pair_search_counts_pairs_and_fields(synth):
    r = pair_search(*synth, top_k=8, regime="train_only", cv_folds=3)
    assert r.n_pairs == 28  # C(8, 2)
    assert r.max_test_f1 >= r.test_f1 - 1e-9
    assert r.regime == "train_only"
    assert len(r.best_pair) == 2


def test_test_informed_is_optimistic_on_pure_noise(pure_noise_synth):
    """On pure noise, selecting with test labels must score higher than honest train-only selection,
    on average over repeated noise draws -- the search-aware null's entire premise."""
    Xtr, ytr, Xte, yte = pure_noise_synth
    ti, to = [], []
    for s in range(6):
        rng = np.random.default_rng(100 + s)
        Xtr2, Xte2 = rng.normal(size=Xtr.shape), rng.normal(size=Xte.shape)
        ti.append(pair_search(Xtr2, ytr, Xte2, yte, top_k=15, regime="test_informed").test_f1)
        to.append(pair_search(Xtr2, ytr, Xte2, yte, top_k=15, regime="train_only", cv_folds=3).test_f1)
    assert np.mean(ti) > np.mean(to) + 0.03


def test_lr_rank_method_runs_and_agrees_reasonably_with_smd():
    rng = np.random.default_rng(5)
    y = np.array([0, 1] * 25)
    X = rng.normal(size=(50, 6))
    X[:, 2] += 2.5 * y
    r = pair_search(X, y, X, y, top_k=4, regime="train_only", rank_method="lr", cv_folds=3)
    assert 2 in r.best_pair
