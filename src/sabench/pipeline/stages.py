"""Experiment stages E0-E6: thin orchestration over :mod:`sabench.modeling`, operating on in-memory
arrays so every stage is unit-testable on synthetic data without touching disk. CLI commands
(``sabench.cli.app``) are the only code that loads features from disk and writes result CSVs.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from sabench.config import Config
from sabench.constants import REGIME_TEST_INFORMED, REGIME_TRAIN_ONLY
from sabench.modeling.bootstrap import bootstrap_ci
from sabench.modeling.groups import SplitGroups
from sabench.modeling.metrics import liblinear_classifier
from sabench.modeling.nulls import run_null, run_probe_null, search_aware_pvalue
from sabench.modeling.preprocessing import fit_transform
from sabench.modeling.probes import run_all_layers, select_layer
from sabench.modeling.resample import monte_carlo_resample, repeated_stratified_cv, summarize
from sabench.modeling.search import PairSearchResult, Regime, pair_search
from sabench.utils.seeding import perm_seed


def _train(groups: SplitGroups | None):
    return None if groups is None else groups.train


def _test(groups: SplitGroups | None):
    return None if groups is None else groups.test


class Split(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


# ---------------------------------------------------------------------------
# E0: replication
# ---------------------------------------------------------------------------


def stage_replicate(split: Split, *, config: Config, seed: int, groups: SplitGroups | None = None) -> pd.DataFrame:
    """Replicate the original paper's search-and-select analysis (``test_informed``), reporting
    against the published targets in ``config.replication``."""
    result = pair_search(
        *split,
        top_k=config.search.top_k,
        regime=REGIME_TEST_INFORMED,
        rank_method=config.search.rank_method,
        cv_folds=config.search.cv_folds, cv_repeats=config.search.cv_repeats,
        seed=seed, groups_train=_train(groups),
    )
    return pd.DataFrame([_pair_result_row(result)])


def check_replication_gate(observed_test_f1: float, target_test_f1: float, tolerance: float) -> bool:
    """Gate G1: is the replicated score within ``tolerance`` macro-F1 of the published target?"""
    return abs(observed_test_f1 - target_test_f1) <= tolerance


# ---------------------------------------------------------------------------
# E1: search-aware permutation null
# ---------------------------------------------------------------------------


def stage_null(
    split: Split, *, config: Config, regime: Regime, seed: int, time_budget_probe: int = 0,
    groups: SplitGroups | None = None,
) -> tuple[PairSearchResult, pd.DataFrame, float]:
    """Observed (non-permuted) pair search, its permutation null distribution, and its p-value."""
    observed = pair_search(
        *split,
        top_k=config.search.top_k,
        regime=regime,
        rank_method=config.search.rank_method,
        cv_folds=config.search.cv_folds, cv_repeats=config.search.cv_repeats,
        seed=seed, groups_train=_train(groups),
    )
    null_df = run_null(
        *split,
        n_permutations=config.permutation_null.n_permutations,
        top_k=config.search.top_k,
        regime=regime,
        rank_method=config.search.rank_method,
        cv_folds=config.search.cv_folds, cv_repeats=config.search.cv_repeats,
        seed=seed,
        n_jobs=config.permutation_null.n_jobs,
        time_budget_probe=time_budget_probe,
        groups_train=_train(groups),
    )
    pvalue = search_aware_pvalue(null_df["test_f1"], observed.test_f1)
    return observed, null_df, pvalue


# ---------------------------------------------------------------------------
# E2: uncertainty (bootstrap CI, Monte Carlo, repeated CV) for a fixed pair/probe
# ---------------------------------------------------------------------------


def stage_uncertainty(split: Split, *, config: Config, seed: int, groups: SplitGroups | None = None) -> dict:
    """Bootstrap CI for the fixed-split score, plus Monte Carlo and repeated-CV resampling means,
    for already-selected features (``split`` should already be restricted to the chosen columns)."""
    Xtr, Xte = fit_transform(split.X_train, split.X_test)
    classifier = liblinear_classifier()
    classifier.fit(Xtr, split.y_train)
    y_pred = classifier.predict(Xte)
    fixed_split_f1 = float(f1_score(split.y_test, y_pred, average="macro"))

    ci = bootstrap_ci(
        split.y_test, y_pred, n_bootstrap=config.uncertainty.n_bootstrap,
        alpha=config.uncertainty.alpha, seed=seed, groups=_test(groups),
    )

    X_pooled = np.vstack([split.X_train, split.X_test])
    y_pooled = np.concatenate([split.y_train, split.y_test])
    g_pooled = None if groups is None or groups.train is None else np.concatenate([groups.train, groups.test])
    mc_scores = monte_carlo_resample(
        X_pooled, y_pooled, liblinear_classifier,
        n_splits=config.uncertainty.n_monte_carlo, test_size=config.uncertainty.test_size, seed=seed,
        groups=g_pooled,
    )
    cv_scores = repeated_stratified_cv(
        X_pooled, y_pooled, liblinear_classifier,
        n_splits=config.uncertainty.cv_folds, n_repeats=config.uncertainty.cv_repeats, seed=seed,
        groups=g_pooled,
    )

    return {
        "fixed_split_f1": fixed_split_f1,
        "bootstrap": asdict(ci),
        "monte_carlo": summarize(mc_scores, fixed_split_f1),
        "repeated_cv": summarize(cv_scores, fixed_split_f1),
    }


# ---------------------------------------------------------------------------
# E3: embedding probes (layer-wise + pair search over embedding dimensions)
# ---------------------------------------------------------------------------


def stage_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: Config,
    seed: int,
    groups_train: np.ndarray | None = None,
) -> pd.DataFrame:
    """Layer-wise regularised-LR probes over a ``(n, n_layers, dim)`` embedding tensor. The
    ``selected`` column marks the honestly-chosen layer (best inner-CV score, never test F1)."""
    probe = config.embeddings.probe
    results = run_all_layers(
        X_train, y_train, X_test, y_test,
        cs=probe.cs, cv_folds=probe.cv_folds, cv_repeats=probe.cv_repeats, seed=seed, groups_train=groups_train,
    )
    chosen = select_layer(results)
    df = pd.DataFrame([asdict(r) for r in results])
    df["selected"] = df["layer"] == chosen.layer
    return df


def stage_probe_null(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    observed_test_f1: float,
    config: Config,
    seed: int,
    groups_train: np.ndarray | None = None,
) -> tuple[pd.DataFrame, float]:
    """E3's search-aware probe null (C and layer re-selected on every permutation) and the p-value
    of the honestly-selected layer's test F1 against it."""
    probe = config.embeddings.probe
    null_df = run_probe_null(
        X_train, y_train, X_test, y_test,
        cs=probe.cs, n_permutations=config.permutation_null.n_permutations,
        cv_folds=probe.cv_folds, cv_repeats=probe.cv_repeats, seed=seed,
        n_jobs=config.permutation_null.n_jobs, groups_train=groups_train,
    )
    return null_df, search_aware_pvalue(null_df["test_f1"], observed_test_f1)


def stage_embedding_pair_search(
    split: Split, *, config: Config, regime: Regime, seed: int, time_budget_probe: int = 0,
    groups: SplitGroups | None = None,
) -> tuple[PairSearchResult, pd.DataFrame, float]:
    """E3's "pair search over embedding dimensions with the same K and null as E1"
    (``docs/EXPERIMENTS.md``): identical to :func:`stage_null`, applied to one layer's embedding
    dimensions (``split.X_train[:, layer, :]``) instead of LLD features. Kept as a distinctly-named
    wrapper so call sites read as "the E3 requirement", even though the underlying mechanism -- an
    observed pair search plus its permutation null -- is exactly :func:`stage_null`."""
    return stage_null(
        split, config=config, regime=regime, seed=seed, time_budget_probe=time_budget_probe, groups=groups
    )


# ---------------------------------------------------------------------------
# E4: segment (full/speech/silence) and channel-only baselines
# ---------------------------------------------------------------------------


def stage_segment_channel(
    variant_splits: dict[str, Split], *, config: Config, seed: int, groups: SplitGroups | None = None
) -> pd.DataFrame:
    """Best-pair macro-F1 (with bootstrap CI) for each of the ``full`` / ``speech`` / ``silence`` /
    ``channel`` variants supplied in ``variant_splits`` (Table 2)."""
    rows = []
    for variant, split in variant_splits.items():
        result = pair_search(
            *split,
            top_k=config.search.top_k,
            regime=REGIME_TRAIN_ONLY,
            rank_method=config.search.rank_method,
            cv_folds=config.search.cv_folds, cv_repeats=config.search.cv_repeats,
            seed=seed, groups_train=_train(groups),
        )
        Xtr, Xte = fit_transform(split.X_train, split.X_test)
        cols = list(result.best_pair)
        classifier = liblinear_classifier()
        classifier.fit(Xtr[:, cols], split.y_train)
        y_pred = classifier.predict(Xte[:, cols])
        ci = bootstrap_ci(split.y_test, y_pred, n_bootstrap=config.uncertainty.n_bootstrap, seed=seed,
                          groups=_test(groups))
        row = _pair_result_row(result)
        row.update({"variant": variant, "ci_low": ci.ci_low, "ci_high": ci.ci_high})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E5: audio interventions (delta macro-F1 under normalisation/band-limiting/noise-floor matching)
# ---------------------------------------------------------------------------


def stage_interventions(
    baseline_split: Split, intervened_split: Split, *, config: Config, seed: int
) -> dict:
    """Delta macro-F1 (with bootstrap CI on the difference) between a baseline feature/label split
    and the same recordings after an audio intervention (``sabench.audio.interventions``)."""
    baseline_f1, baseline_pred = _fit_predict_with_preds(baseline_split)
    intervened_f1, intervened_pred = _fit_predict_with_preds(intervened_split)

    rng_scores = []
    for b in range(config.uncertainty.n_bootstrap):
        rng = np.random.default_rng(perm_seed(seed, b))
        idx = rng.integers(0, len(baseline_split.y_test), size=len(baseline_split.y_test))
        f1_b = f1_score(baseline_split.y_test[idx], baseline_pred[idx], average="macro")
        f1_i = f1_score(intervened_split.y_test[idx], intervened_pred[idx], average="macro")
        rng_scores.append(f1_i - f1_b)
    rng_scores_arr = np.array(rng_scores)
    alpha = config.uncertainty.alpha
    lo, hi = np.percentile(rng_scores_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "baseline_f1": baseline_f1,
        "intervened_f1": intervened_f1,
        "delta_f1": intervened_f1 - baseline_f1,
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
    }


def _fit_predict_with_preds(split: Split) -> tuple[float, np.ndarray]:
    Xtr, Xte = fit_transform(split.X_train, split.X_test)
    classifier = liblinear_classifier()
    classifier.fit(Xtr, split.y_train)
    y_pred = classifier.predict(Xte)
    return float(f1_score(split.y_test, y_pred, average="macro")), y_pred


# ---------------------------------------------------------------------------
# E6: cross-dataset generalisation
# ---------------------------------------------------------------------------


def speaker_overlap(speakers_a: pd.Series, speakers_b: pd.Series) -> set[str]:
    """Speaker ids present in both datasets -- must be checked and reported before any cross-dataset
    experiment, since ADReSS and ADReSSo both derive from the Pitt corpus."""
    a = set(speakers_a.dropna().astype(str))
    b = set(speakers_b.dropna().astype(str))
    return a & b


def stage_cross_dataset(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    *, config: Config, seed: int, groups_train: np.ndarray | None = None,
) -> PairSearchResult:
    """Train (search + fit) on one dataset's pooled data, evaluate on the other dataset's pooled
    data. Feature columns must already be aligned (same feature set, same column order) by the
    caller."""
    return pair_search(
        X_train, y_train, X_test, y_test,
        top_k=config.search.top_k, regime=REGIME_TRAIN_ONLY,
        rank_method=config.search.rank_method, cv_folds=config.search.cv_folds,
        cv_repeats=config.search.cv_repeats, seed=seed, groups_train=groups_train,
    )


def _pair_result_row(result: PairSearchResult) -> dict:
    row = asdict(result)
    row["pair_i"], row["pair_j"] = row.pop("best_pair")
    return row
