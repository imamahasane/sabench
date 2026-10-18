"""The ``sabench`` command-line interface. One subcommand per pipeline stage (``docs/EXPERIMENTS.md``
E0-E6), plus ``validate``, ``figures``, and ``demo``. Every command shares the same
``--config``/``--set`` override pattern and exits non-zero with a clear message on any
:class:`sabench.exceptions.SabenchError`, instead of a raw traceback.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sabench import __version__
from sabench.constants import REGIMES
from sabench.data.manifest import align_on_id
from sabench.exceptions import SabenchError
from sabench.features.embeddings import ModelKey
from sabench.features.lld import SUPPORTED_FEATURESETS, FeatureSetName
from sabench.modeling.groups import SplitGroups
from sabench.modeling.search import Regime
from sabench.pipeline.io import (
    align_embeddings_to_manifest,
    align_features_to_manifest,
    load_embeddings,
    load_lld_features,
    save_embeddings,
    save_lld_features,
    split_train_test_array,
    split_train_test_frame,
)
from sabench.pipeline.runner import ExperimentRunner
from sabench.pipeline.stages import (
    Split,
    check_replication_gate,
    speaker_overlap,
    stage_cross_dataset,
    stage_embedding_pair_search,
    stage_null,
    stage_probe_null,
    stage_probes,
    stage_replicate,
    stage_segment_channel,
    stage_uncertainty,
)

app = typer.Typer(
    name="sabench",
    help="Search-aware evaluation of acoustic and embedding features on ADReSS/ADReSSo.",
    no_args_is_help=True,
)
console = Console()

ConfigOption = typer.Option(None, "--config", help="Path to a config YAML (default: packaged configs/default.yaml).")
SetOption = typer.Option([], "--set", help="Override 'dotted.key=value'; repeatable.")
BaseDirOption = typer.Option(Path("."), help="Project root the config's relative paths resolve against.")


def _runner(config: Path | None, overrides: list[str], base_dir: Path) -> ExperimentRunner:
    try:
        return ExperimentRunner.from_file(config, overrides, base_dir=base_dir)
    except SabenchError as exc:
        console.print(f"[red]configuration error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _fail(exc: Exception) -> None:
    console.print(f"[red]error:[/red] {exc}")
    raise typer.Exit(code=1) from exc


_MODEL_KEYS = ("wavlm", "wav2vec2", "whisper")


def _parse_regime(value: str) -> Regime:
    if value not in REGIMES:
        raise typer.BadParameter(f"regime must be one of {REGIMES}, got {value!r}")
    return cast(Regime, value)


def _parse_featureset(value: str) -> FeatureSetName:
    if value not in SUPPORTED_FEATURESETS:
        raise typer.BadParameter(f"featureset must be one of {SUPPORTED_FEATURESETS}, got {value!r}")
    return cast(FeatureSetName, value)


def _parse_model_key(value: str) -> ModelKey:
    if value not in _MODEL_KEYS:
        raise typer.BadParameter(f"model must be one of {_MODEL_KEYS}, got {value!r}")
    return cast(ModelKey, value)


@app.callback()
def _main() -> None:
    pass


@app.command()
def version() -> None:
    """Print the installed sabench version."""
    console.print(__version__)


@app.command()
def validate(
    dataset: str = typer.Option("all", help="'all' or one dataset name from the config."),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Check a manifest's per-split class counts against the published challenge counts."""
    runner = _runner(config, set_, base_dir)
    datasets = runner.config.datasets if dataset == "all" else [dataset]
    all_ok = True
    for d in datasets:
        try:
            report = runner.validate_manifest(d)
        except SabenchError as exc:
            console.print(f"[red]{d}:[/red] {exc}")
            all_ok = False
            continue
        style = "green" if report.ok else "red"
        console.print(f"[{style}]{report.summary()}[/{style}]")
        all_ok = all_ok and report.ok
    raise typer.Exit(code=0 if all_ok else 1)


def _split_groups(manifest: pd.DataFrame) -> SplitGroups | None:
    """Speaker keys for the train and test rows, in manifest order (the order every split helper
    preserves); ``None`` when the manifest has no ``speaker`` column."""
    if "speaker" not in manifest.columns:
        return None
    speakers = manifest["speaker"].astype(str)
    return SplitGroups(speakers[manifest["split"] == "train"].to_numpy(), speakers[manifest["split"] == "test"].to_numpy())


def _load_lld_split(runner: ExperimentRunner, dataset: str, featureset: str, variant: str) -> tuple[Split, list[str]]:
    manifest = runner.load_manifest(dataset)
    feat_path = runner.paths.features / dataset / variant / f"{featureset}.csv"
    features = load_lld_features(feat_path)
    aligned = align_features_to_manifest(manifest, features)
    feature_cols = list(features.columns)
    X_train, y_train, X_test, y_test = split_train_test_frame(aligned, feature_cols)
    return Split(X_train, y_train, X_test, y_test), feature_cols


@app.command(name="replicate")
def replicate_cmd(
    dataset: str = typer.Argument(..., help="adress or adresso"),
    featureset: str = typer.Option("eGeMAPSv02"),
    variant: str = typer.Option("full"),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """E0: replicate the original paper's test-informed search-and-select analysis."""
    runner = _runner(config, set_, base_dir)
    try:
        split, feature_cols = _load_lld_split(runner, dataset, featureset, variant)
        groups = _split_groups(runner.load_manifest(dataset))
        df = stage_replicate(split, config=runner.config, seed=runner.config.seed, groups=groups)
        runner.save_table(df, f"e0_{dataset}_{featureset}", regime="test_informed", seed=runner.config.seed)
        observed = float(df.iloc[0]["test_f1"])
        targets = runner.config.replication.targets
        if dataset in targets:
            target = targets[dataset].test_f1
            passed = check_replication_gate(observed, target, runner.config.replication.tolerance)
            style = "green" if passed else "yellow"
            console.print(
                f"[{style}]Gate G1 {'PASSED' if passed else 'FAILED'}[/{style}]: "
                f"observed={observed:.3f} target={target:.3f} tolerance={runner.config.replication.tolerance}"
            )
        else:
            console.print(f"test_informed test_f1={observed:.3f} (no published replication target for {dataset})")
        chosen = (feature_cols[int(df.iloc[0]['pair_i'])], feature_cols[int(df.iloc[0]['pair_j'])])
        console.print(f"chosen pair: {chosen}")
    except SabenchError as exc:
        _fail(exc)


@app.command(name="null")
def null_cmd(
    dataset: str = typer.Argument(...),
    featureset: str = typer.Option("eGeMAPSv02"),
    variant: str = typer.Option("full"),
    regime: str = typer.Option("train_only"),
    n_permutations: int | None = typer.Option(None, help="Override configs/default.yaml permutation_null.n_permutations."),
    time_probe: int = typer.Option(10, help="Time this many permutations serially before the full parallel run."),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """E1: search-aware label-permutation null and its p-value against the observed score."""
    runner = _runner(config, set_, base_dir)
    try:
        split, _ = _load_lld_split(runner, dataset, featureset, variant)
        groups = _split_groups(runner.load_manifest(dataset))
        cfg = runner.config
        if n_permutations is not None:
            cfg = cfg.model_copy(update={
                "permutation_null": cfg.permutation_null.model_copy(update={"n_permutations": n_permutations})
            })
        observed, null_df, pvalue = stage_null(
            split, config=cfg, regime=_parse_regime(regime), seed=cfg.seed, time_budget_probe=time_probe,
            groups=groups,
        )
        runner.save_table(null_df, f"e1_null_{dataset}_{featureset}_{regime}", regime=regime, seed=cfg.seed)
        observed_df = pd.DataFrame([{
            "pair_i": observed.best_pair[0], "pair_j": observed.best_pair[1],
            "test_f1": observed.test_f1, "max_test_f1": observed.max_test_f1,
            "n_pairs": observed.n_pairs, "pvalue": pvalue,
        }])
        runner.save_table(
            observed_df, f"e1_observed_{dataset}_{featureset}_{regime}", regime=regime, seed=cfg.seed
        )
        console.print(
            f"observed test_f1={observed.test_f1:.3f}  "
            f"null mean={null_df['test_f1'].mean():.3f}  "
            f"p-value={pvalue:.4f}  (B={len(null_df)})"
        )
    except SabenchError as exc:
        _fail(exc)


@app.command(name="uncertainty")
def uncertainty_cmd(
    dataset: str = typer.Argument(...),
    featureset: str = typer.Option("eGeMAPSv02"),
    variant: str = typer.Option("full"),
    pair: str | None = typer.Option(None, help="Comma-separated feature column names; default: rerun E0's search."),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """E2: bootstrap CI and resampled (Monte Carlo / repeated CV) means for a fixed pair."""
    runner = _runner(config, set_, base_dir)
    try:
        split, feature_cols = _load_lld_split(runner, dataset, featureset, variant)
        groups = _split_groups(runner.load_manifest(dataset))
        if pair is not None:
            names = [p.strip() for p in pair.split(",")]
            cols = [feature_cols.index(n) for n in names]
        else:
            result = stage_replicate(split, config=runner.config, seed=runner.config.seed, groups=groups)
            cols = [int(result.iloc[0]["pair_i"]), int(result.iloc[0]["pair_j"])]
        restricted = Split(split.X_train[:, cols], split.y_train, split.X_test[:, cols], split.y_test)
        summary = stage_uncertainty(restricted, config=runner.config, seed=runner.config.seed, groups=groups)
        table = Table(show_header=False)
        table.add_row("fixed-split F1", f"{summary['fixed_split_f1']:.3f}")
        boot = summary["bootstrap"]
        table.add_row("bootstrap 95% CI", f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")
        mc = summary["monte_carlo"]
        table.add_row("Monte Carlo mean (SD)", f"{mc['mean']:.3f} ({mc['sd']:.3f})")
        cv = summary["repeated_cv"]
        table.add_row("repeated-CV mean (SD)", f"{cv['mean']:.3f} ({cv['sd']:.3f})")
        console.print(table)
        flat = pd.DataFrame([{
            "fixed_split_f1": summary["fixed_split_f1"],
            "bootstrap_ci_low": boot["ci_low"], "bootstrap_ci_high": boot["ci_high"],
            "mc_mean": mc["mean"], "mc_sd": mc["sd"],
            "cv_mean": cv["mean"], "cv_sd": cv["sd"],
        }])
        runner.save_table(flat, f"e2_uncertainty_{dataset}_{featureset}", regime="train_only", seed=runner.config.seed)
    except SabenchError as exc:
        _fail(exc)


@app.command(name="probes")
def probes_cmd(
    dataset: str = typer.Argument(...),
    model: str = typer.Option("wavlm"),
    variant: str = typer.Option("full"),
    probe_null: bool = typer.Option(
        False,
        "--probe-null/--no-probe-null",
        help="Also run the search-aware probe null: permute training labels and redo C and layer "
        "selection per permutation, giving a p-value for the CV-selected layer's test F1.",
    ),
    pair_search_null: bool = typer.Option(
        False,
        "--pair-search-null/--no-pair-search-null",
        help="Also run E3's 'pair search over embedding dimensions, same K and null as E1' on the "
        "CV-selected layer. Expensive: reruns the full search per permutation.",
    ),
    regime: str = typer.Option("train_only"),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """E3: layer-wise regularised-LR probes on frozen embeddings."""
    runner = _runner(config, set_, base_dir)
    try:
        manifest = runner.load_manifest(dataset)
        emb_path = runner.paths.features / dataset / variant / f"emb_{model}.npz"
        tag = model if variant == "full" else f"{model}_{variant}"
        ids, X = load_embeddings(emb_path)
        X = align_embeddings_to_manifest(manifest, ids, X)
        X_train, y_train, X_test, y_test = split_train_test_array(manifest, X)
        groups = _split_groups(manifest)
        g_train = None if groups is None else groups.train
        seed = runner.config.seed
        df = stage_probes(X_train, y_train, X_test, y_test, config=runner.config, seed=seed, groups_train=g_train)
        runner.save_table(df, f"e3_probe_{dataset}_{tag}", regime="train_only", seed=seed)
        chosen = df.loc[df["selected"]].iloc[0]
        best_layer = int(chosen["layer"])
        oracle = df.loc[df["test_f1"].idxmax()]
        console.print(
            f"CV-selected layer={best_layer}  test_f1={chosen['test_f1']:.3f}  C={chosen['best_c']}  "
            f"(test-picked oracle, optimistic: layer={int(oracle['layer'])} test_f1={oracle['test_f1']:.3f})"
        )

        if probe_null:
            null_df, pvalue = stage_probe_null(
                X_train, y_train, X_test, y_test,
                observed_test_f1=float(chosen["test_f1"]), config=runner.config, seed=seed, groups_train=g_train,
            )
            runner.save_table(null_df, f"e3_probe_null_{dataset}_{tag}", regime="train_only", seed=seed)
            observed_df = pd.DataFrame([{
                "layer": best_layer, "best_c": chosen["best_c"], "cv_score": chosen["cv_score"],
                "test_f1": chosen["test_f1"], "oracle_layer": int(oracle["layer"]),
                "oracle_test_f1": oracle["test_f1"], "pvalue": pvalue,
            }])
            runner.save_table(observed_df, f"e3_probe_observed_{dataset}_{tag}", regime="train_only", seed=seed)
            console.print(
                f"probe null: observed={chosen['test_f1']:.3f}  null_mean={null_df['test_f1'].mean():.3f}  "
                f"p-value={pvalue:.4f}  (B={len(null_df)})"
            )

        if pair_search_null:
            regime_typed = _parse_regime(regime)
            layer_split = Split(
                X_train[:, best_layer, :], y_train, X_test[:, best_layer, :], y_test
            )
            observed, null_df, pvalue = stage_embedding_pair_search(
                layer_split, config=runner.config, regime=regime_typed, seed=runner.config.seed, time_budget_probe=10,
                groups=groups,
            )
            runner.save_table(
                null_df, f"e3_pairsearch_null_{dataset}_{tag}_layer{best_layer}_{regime}",
                regime=regime, seed=runner.config.seed,
            )
            runner.save_table(
                pd.DataFrame([{
                    "layer": best_layer, "pair_i": observed.best_pair[0], "pair_j": observed.best_pair[1],
                    "test_f1": observed.test_f1, "max_test_f1": observed.max_test_f1, "pvalue": pvalue,
                }]),
                f"e3_pairsearch_observed_{dataset}_{tag}_{regime}", regime=regime, seed=runner.config.seed,
            )
            console.print(
                f"layer {best_layer} pair-search null: observed={observed.test_f1:.3f} "
                f"null_mean={null_df['test_f1'].mean():.3f}  p-value={pvalue:.4f}  (B={len(null_df)})"
            )
    except SabenchError as exc:
        _fail(exc)


@app.command(name="segment-channel")
def segment_channel_cmd(
    dataset: str = typer.Argument(...),
    featureset: str = typer.Option("eGeMAPSv02"),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """E4: best-pair macro-F1 (with CI) on full/speech/silence variants plus a channel-only baseline."""
    runner = _runner(config, set_, base_dir)
    try:
        variant_splits: dict[str, Split] = {}
        for variant in ("full", "speech", "silence"):
            split, _ = _load_lld_split(runner, dataset, featureset, variant)
            variant_splits[variant] = split
        manifest = runner.load_manifest(dataset)
        channel_path = runner.paths.interim / dataset / "channel.csv"
        channel = load_lld_features(channel_path)
        aligned = align_on_id(manifest, channel, context="channel descriptors")
        channel_cols = list(channel.columns)
        variant_splits["channel"] = Split(*split_train_test_frame(aligned, channel_cols))
        df = stage_segment_channel(variant_splits, config=runner.config, seed=runner.config.seed,
                                   groups=_split_groups(manifest))
        runner.save_table(df, f"e4_segment_channel_{dataset}_{featureset}", regime="train_only", seed=runner.config.seed)
        console.print(df.to_string(index=False))
    except SabenchError as exc:
        _fail(exc)


@app.command(name="cross-dataset")
def cross_dataset_cmd(
    train_on: str = typer.Argument(..., help="Dataset to train+rank on, e.g. adress."),
    test_on: str = typer.Argument(..., help="Dataset to evaluate on, e.g. adresso."),
    featureset: str = typer.Option("eGeMAPSv02"),
    variant: str = typer.Option("full"),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """E6: speaker-overlap check, then train on one dataset's pooled data and test on the other's."""
    runner = _runner(config, set_, base_dir)
    try:
        m_train = runner.load_manifest(train_on)
        m_test = runner.load_manifest(test_on)
        if runner.config.cross_dataset.check_speaker_overlap and "speaker" in m_train and "speaker" in m_test:
            overlap = speaker_overlap(m_train["speaker"], m_test["speaker"])
            if overlap:
                console.print(f"[red]speaker overlap detected ({len(overlap)} ids); results would be optimistic[/red]")
                raise typer.Exit(code=1)
            console.print("[green]no speaker overlap detected[/green]")

        feat_train = load_lld_features(runner.paths.features / train_on / variant / f"{featureset}.csv")
        feat_test = load_lld_features(runner.paths.features / test_on / variant / f"{featureset}.csv")
        common_cols = [c for c in feat_train.columns if c in feat_test.columns]
        if len(common_cols) < len(feat_train.columns):
            console.print(
                f"[yellow]using {len(common_cols)}/{len(feat_train.columns)} feature columns common to both datasets[/yellow]"
            )
        aligned_train = align_features_to_manifest(m_train, feat_train)
        aligned_test = align_features_to_manifest(m_test, feat_test)
        X_train = aligned_train[common_cols].to_numpy(dtype=float)
        y_train = aligned_train["label"].to_numpy()
        X_test = aligned_test[common_cols].to_numpy(dtype=float)
        y_test = aligned_test["label"].to_numpy()

        g_train = aligned_train["speaker"].astype(str).to_numpy() if "speaker" in aligned_train else None
        result = stage_cross_dataset(
            X_train, y_train, X_test, y_test, config=runner.config, seed=runner.config.seed, groups_train=g_train
        )
        df = pd.DataFrame([{
            "train_on": train_on, "test_on": test_on,
            "pair_i": result.best_pair[0], "pair_j": result.best_pair[1],
            "test_f1": result.test_f1, "max_test_f1": result.max_test_f1,
        }])
        runner.save_table(df, f"e6_cross_{train_on}_to_{test_on}_{featureset}", regime="train_only", seed=runner.config.seed)
        console.print(f"test_f1={result.test_f1:.3f}  max_test_f1={result.max_test_f1:.3f}")
    except SabenchError as exc:
        _fail(exc)


@app.command(name="prepare-audio")
def prepare_audio_cmd(
    dataset: str = typer.Argument(...),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Load raw audio, resample, run VAD, write full/speech/silence variants and channel descriptors."""
    from sabench.pipeline.audio_pipeline import prepare_dataset_audio

    runner = _runner(config, set_, base_dir)
    try:
        manifest = runner.load_manifest(dataset)
        runner.paths.ensure_output_dirs()
        channel_df = prepare_dataset_audio(
            manifest, runner.paths.raw / dataset, runner.paths.interim / dataset,
            audio_cfg=runner.config.audio, vad_cfg=runner.config.vad, channel_cfg=runner.config.channel,
        )
        save_lld_features(channel_df, runner.paths.interim / dataset / "channel.csv")
        n_ok = int(channel_df.notna().any(axis=1).sum())
        console.print(f"prepared {n_ok}/{len(manifest)} recordings for {dataset}")
    except SabenchError as exc:
        _fail(exc)


@app.command(name="extract-lld")
def extract_lld_cmd(
    dataset: str = typer.Argument(...),
    featureset: str = typer.Option("eGeMAPSv02"),
    variant: str = typer.Option("full"),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Extract openSMILE LLD functionals for one dataset/variant."""
    from sabench.features.lld import extract_lld_batch

    runner = _runner(config, set_, base_dir)
    try:
        manifest = runner.load_manifest(dataset)
        wav_dir = runner.paths.interim / dataset / variant
        paths_by_id = {
            str(rec_id): wav_dir / f"{rec_id}.wav"
            for rec_id in manifest["id"].astype(str)
            if (wav_dir / f"{rec_id}.wav").exists()
        }
        missing = len(manifest) - len(paths_by_id)
        if missing:
            console.print(f"[yellow]{missing} recording(s) missing a '{variant}' wav; will be NaN[/yellow]")
        df = extract_lld_batch(paths_by_id, _parse_featureset(featureset))
        df = df.reindex(manifest["id"].astype(str).tolist())
        out_path = runner.paths.features / dataset / variant / f"{featureset}.csv"
        save_lld_features(df, out_path)
        console.print(f"wrote {out_path} ({df.shape[0]} rows x {df.shape[1]} cols)")
    except SabenchError as exc:
        _fail(exc)


@app.command(name="extract-embeddings")
def extract_embeddings_cmd(
    dataset: str = typer.Argument(...),
    model: str = typer.Option("wavlm"),
    variant: str = typer.Option("full"),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Extract frozen, layer-wise, mean-pooled embeddings for one dataset/variant/model."""
    from sabench.features.embeddings import extract_embeddings

    runner = _runner(config, set_, base_dir)
    try:
        model_name = runner.config.embeddings.models[model]
        manifest = runner.load_manifest(dataset)
        wav_dir = runner.paths.interim / dataset / variant
        paths_by_id = {
            str(rec_id): wav_dir / f"{rec_id}.wav"
            for rec_id in manifest["id"].astype(str)
            if (wav_dir / f"{rec_id}.wav").exists()
        }
        missing = len(manifest) - len(paths_by_id)
        if missing:
            console.print(f"[yellow]{missing} recording(s) missing a '{variant}' wav; will be NaN[/yellow]")
        ids, X = extract_embeddings(
            paths_by_id, _parse_model_key(model), model_name,
            chunk_seconds=runner.config.embeddings.chunk_seconds, device=runner.config.embeddings.device,
        )
        out_path = runner.paths.features / dataset / variant / f"emb_{model}.npz"
        save_embeddings(ids, X, out_path)
        console.print(f"wrote {out_path} (shape {X.shape})")
    except SabenchError as exc:
        _fail(exc)


@app.command()
def transcribe(
    dataset: str = typer.Argument(...),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Whisper ASR transcripts of every prepared ``full`` recording -> ``<features>/<dataset>/transcripts_asr.csv``."""
    from sabench.features.text import transcribe as run_asr

    runner = _runner(config, set_, base_dir)
    try:
        manifest = runner.load_manifest(dataset)
        wav_dir = runner.paths.interim / dataset / "full"
        paths_by_id = {str(i): wav_dir / f"{i}.wav" for i in manifest["id"].astype(str)}
        languages = (dict(zip(manifest["id"].astype(str), manifest["language"].astype(str), strict=True))
                     if "language" in manifest.columns else None)
        texts = run_asr(paths_by_id, runner.config.text.asr_model, device=runner.config.text.device, languages=languages)
        df = pd.DataFrame({"text": pd.Series(texts)}).rename_axis("id")
        out = runner.paths.features / dataset / "transcripts_asr.csv"
        save_lld_features(df, out)
        empty = int((df["text"].str.len() == 0).sum())
        console.print(f"wrote {out} ({len(df)} transcripts, {empty} empty)")
    except SabenchError as exc:
        _fail(exc)


@app.command(name="extract-text-embeddings")
def extract_text_embeddings_cmd(
    dataset: str = typer.Argument(...),
    source: str = typer.Option("asr", help="'asr' (transcripts_asr.csv) or 'manual' (transcripts_manual.csv)."),
    model: str = typer.Option("bert", help="Key into config text.models."),
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Layer-wise frozen LM embeddings of transcripts -> ``<features>/<dataset>/text_<source>/emb_<model>.npz``
    (probe them with ``sabench probes <dataset> --model <model> --variant text_<source>``)."""
    from sabench.features.text import extract_text_embeddings

    runner = _runner(config, set_, base_dir)
    try:
        cfg = runner.config.text
        if model not in cfg.models:
            raise typer.BadParameter(f"model must be one of {sorted(cfg.models)}, got {model!r}")
        manifest = runner.load_manifest(dataset)
        transcripts = load_lld_features(runner.paths.features / dataset / f"transcripts_{source}.csv")
        texts = transcripts["text"].fillna("").astype(str)
        texts_by_id = {i: texts.get(i, "") for i in manifest["id"].astype(str)}
        ids, X = extract_text_embeddings(texts_by_id, cfg.models[model], device=cfg.device, max_tokens=cfg.max_tokens)
        out = runner.paths.features / dataset / f"text_{source}" / f"emb_{model}.npz"
        save_embeddings(ids, X, out)
        console.print(f"wrote {out} (shape {X.shape})")
    except SabenchError as exc:
        _fail(exc)


@app.command()
def figures(
    config: Path | None = ConfigOption,
    set_: list[str] = SetOption,
    base_dir: Path = BaseDirOption,
) -> None:
    """Rebuild Figure 1 (null distributions) and Figure 2 (layer-wise probes) from result tables
    already present in ``results/tables``."""
    from sabench.viz.plotting import plot_layerwise_probe, plot_null_distribution

    runner = _runner(config, set_, base_dir)
    made = 0
    for dataset in runner.config.datasets:
        for featureset in runner.config.lld.featuresets:
            # One combined figure per dataset/featureset, overlaying every regime that has a saved
            # null, with the observed score of each regime marked (docs/EXPERIMENTS.md).
            null_scores_by_regime: dict[str, np.ndarray] = {}
            observed_by_regime: dict[str, float] = {}
            for regime in runner.config.search.regimes:
                null_path = runner.paths.tables / f"e1_null_{dataset}_{featureset}_{regime}.csv"
                if not null_path.exists():
                    continue
                null_scores_by_regime[regime] = pd.read_csv(null_path)["test_f1"].to_numpy()
                observed_path = runner.paths.tables / f"e1_observed_{dataset}_{featureset}_{regime}.csv"
                if observed_path.exists():
                    observed_by_regime[regime] = float(pd.read_csv(observed_path)["test_f1"].iloc[0])
            if null_scores_by_regime:
                fig_path = runner.paths.figures / f"figure1_{dataset}_{featureset}.pdf"
                plot_null_distribution(
                    null_scores_by_regime, observed_by_regime, out_path=fig_path,
                    title=f"{dataset} / {featureset}",
                )
                made += 1
        for model in runner.config.embeddings.models:
            probe_path = runner.paths.tables / f"e3_probe_{dataset}_{model}.csv"
            if not probe_path.exists():
                continue
            probe_df = pd.read_csv(probe_path)
            fig_path = runner.paths.figures / f"figure2_{dataset}_{model}.pdf"
            plot_layerwise_probe(
                probe_df, resampled_mean=float(probe_df["cv_score"].mean()), out_path=fig_path,
                title=f"{dataset} / {model}",
            )
            made += 1
    console.print(f"wrote {made} figure(s) to {runner.paths.figures}")


@app.command()
def demo(
    n_train: int = typer.Option(60, help="Synthetic training participants."),
    n_test: int = typer.Option(30, help="Synthetic test participants."),
    n_features: int = typer.Option(40, help="Synthetic feature columns."),
    n_permutations: int = typer.Option(50, help="Permutations for the demo null (kept small intentionally)."),
    seed: int = typer.Option(1234),
) -> None:
    """Synthetic end-to-end smoke test: E0 -> E1 -> E2 on fabricated data, no audio or config file
    needed. Run this before launching anything on real data."""
    from sabench.config import load_config

    cfg = load_config(overrides=[f"seed={seed}", f"permutation_null.n_permutations={n_permutations}"])
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))
    y_train = rng.integers(0, 2, size=n_train)
    y_test = rng.integers(0, 2, size=n_test)
    X_train[:, 0] += 1.5 * y_train  # one genuinely informative column
    X_test[:, 0] += 1.5 * y_test
    split = Split(X_train, y_train, X_test, y_test)

    console.print("[bold]sabench demo[/bold]: synthetic data, not a real result")
    df0 = stage_replicate(split, config=cfg, seed=seed)
    console.print(f"E0 (test_informed search): test_f1={df0.iloc[0]['test_f1']:.3f}")

    observed, null_df, pvalue = stage_null(
        split, config=cfg, regime="train_only", seed=seed, time_budget_probe=5
    )
    console.print(
        f"E1 (train_only null, B={len(null_df)}): observed={observed.test_f1:.3f} "
        f"null_mean={null_df['test_f1'].mean():.3f} p={pvalue:.3f}"
    )

    cols = list(observed.best_pair)
    restricted = Split(X_train[:, cols], y_train, X_test[:, cols], y_test)
    summary = stage_uncertainty(restricted, config=cfg, seed=seed)
    console.print(
        f"E2 (uncertainty): fixed_split={summary['fixed_split_f1']:.3f} "
        f"bootstrap_ci=[{summary['bootstrap']['ci_low']:.3f}, {summary['bootstrap']['ci_high']:.3f}] "
        f"mc_mean={summary['monte_carlo']['mean']:.3f}"
    )
    console.print("[green]demo completed without error[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
