"""Search-aware label-permutation null (E1).

For each permutation ``b`` the *training* labels are shuffled and the *entire* feature search
(:func:`sabench.modeling.search.pair_search`) is rerun from scratch -- ranking, top-K selection, and
every pairwise fit. This is the project's central methodological contribution: the original paper's
permutation control only retrains the already-selected pair, which cannot detect how much of the
published score search itself can manufacture.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sabench.exceptions import SearchError
from sabench.logging_config import get_logger
from sabench.modeling.groups import permute_labels
from sabench.modeling.probes import run_all_layers, select_layer
from sabench.modeling.search import RankMethod, Regime, pair_search
from sabench.utils.seeding import perm_seed

logger = get_logger(__name__)


def _one_permutation(
    b: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    top_k: int,
    regime: Regime,
    rank_method: RankMethod,
    cv_folds: int,
    cv_repeats: int,
    base_seed: int,
    groups_train: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(perm_seed(base_seed, b))
    y_perm = permute_labels(y_train, rng, groups_train)
    inner_seed = perm_seed(base_seed, b) % (2**31 - 1)
    result = pair_search(
        X_train,
        y_perm,
        X_test,
        y_test,
        top_k=top_k,
        regime=regime,
        rank_method=rank_method,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        seed=inner_seed,
        groups_train=groups_train,
    )
    return {
        "b": b,
        "test_f1": result.test_f1,
        "max_test_f1": result.max_test_f1,
        "criterion": result.criterion,
        "pair_i": result.best_pair[0],
        "pair_j": result.best_pair[1],
        "n_pairs": result.n_pairs,
    }


def run_null(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_permutations: int = 1000,
    top_k: int = 50,
    regime: Regime = "train_only",
    rank_method: RankMethod = "smd",
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: int = 0,
    n_jobs: int = -1,
    time_budget_probe: int = 0,
    groups_train: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the search-aware permutation null; one row per permutation.

    :param time_budget_probe: if > 0, time this many permutations serially first and log a wall-clock
        estimate for the full ``n_permutations`` run before launching the parallel job -- the
        "time 10 permutations before launching 1,000" rule, enforced in code rather
        than left to the operator to remember.
    """
    if time_budget_probe > 0:
        probe_n = min(time_budget_probe, n_permutations)
        start = time.perf_counter()
        for b in range(probe_n):
            _one_permutation(
                b, X_train, y_train, X_test, y_test,
                top_k=top_k, regime=regime, rank_method=rank_method, cv_folds=cv_folds, cv_repeats=cv_repeats,
                base_seed=seed, groups_train=groups_train,
            )
        elapsed = time.perf_counter() - start
        per_perm = elapsed / probe_n
        logger.info(
            "permutation timing probe",
            extra={
                "probed": probe_n,
                "seconds_per_permutation": round(per_perm, 4),
                "estimated_total_seconds_serial": round(per_perm * n_permutations, 1),
            },
        )

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one_permutation)(
            b, X_train, y_train, X_test, y_test,
            top_k=top_k, regime=regime, rank_method=rank_method, cv_folds=cv_folds, cv_repeats=cv_repeats,
                base_seed=seed, groups_train=groups_train,
        )
        for b in range(n_permutations)
    )
    df = pd.DataFrame(rows)
    if (df["max_test_f1"] < df["test_f1"] - 1e-9).any():
        raise SearchError("null distribution violates max_test_f1 >= test_f1 for at least one permutation")
    return df


def _one_probe_permutation(
    b: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cs: list[float],
    cv_folds: int,
    cv_repeats: int,
    base_seed: int,
    groups_train: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(perm_seed(base_seed, b))
    y_perm = permute_labels(y_train, rng, groups_train)
    inner_seed = perm_seed(base_seed, b) % (2**31 - 1)
    chosen = select_layer(run_all_layers(
        X_train, y_perm, X_test, y_test, cs=cs, cv_folds=cv_folds, cv_repeats=cv_repeats, seed=inner_seed,
        groups_train=groups_train,
    ))
    return {"b": b, "test_f1": chosen.test_f1, "layer": chosen.layer,
            "cv_score": chosen.cv_score, "best_c": chosen.best_c}


def run_probe_null(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cs: list[float],
    n_permutations: int = 1000,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: int = 0,
    n_jobs: int = -1,
    groups_train: np.ndarray | None = None,
) -> pd.DataFrame:
    """Search-aware null for E3's layer-wise probes: each permutation shuffles the training labels
    and reruns the whole probe procedure -- ``C`` selection on every layer, then layer selection by
    inner CV -- before scoring the chosen layer on the test split. The probe analogue of
    :func:`run_null`; one row per permutation."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one_probe_permutation)(
            b, X_train, y_train, X_test, y_test,
            cs=cs, cv_folds=cv_folds, cv_repeats=cv_repeats, base_seed=seed, groups_train=groups_train,
        )
        for b in range(n_permutations)
    )
    return pd.DataFrame(rows)


def search_aware_pvalue(null_scores: np.ndarray | pd.Series, observed: float) -> float:
    """``p = (1 + #{null >= observed}) / (B + 1)``. The add-one correction keeps p strictly positive
    and is the standard, conservative estimator for a Monte Carlo permutation p-value."""
    null_scores = np.asarray(null_scores, dtype=float)
    if len(null_scores) == 0:
        raise SearchError("cannot compute a p-value against an empty null distribution")
    return float((1 + np.sum(null_scores >= observed - 1e-12)) / (len(null_scores) + 1))
