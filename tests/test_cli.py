from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from sabench import __version__
from sabench.cli.app import app
from sabench.config import PACKAGE_DEFAULT_CONFIG
from sabench.pipeline.io import save_embeddings, save_lld_features

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_demo_command_runs_clean():
    result = runner.invoke(app, ["demo", "--n-permutations", "10"])
    assert result.exit_code == 0, result.stdout
    assert "demo completed without error" in result.stdout


def _make_project(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(PACKAGE_DEFAULT_CONFIG.read_text())
    cfg["paths"]["manifests"]["adress"] = "data/manifest_adress.csv"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n_train, n_test = 108, 48  # 54/54 and 24/24 per docs/DATA_ACCESS.md published counts
    manifest = pd.DataFrame({
        "id": [f"t{i}" for i in range(n_train + n_test)],
        "split": ["train"] * n_train + ["test"] * n_test,
        "label": ([0] * 54 + [1] * 54) + ([0] * 24 + [1] * 24),
        "path": [f"t{i}.wav" for i in range(n_train + n_test)],
    })
    manifest.to_csv(data_dir / "manifest_adress.csv", index=False)
    return cfg_path


def _validate_args(cfg_path: Path, tmp_path: Path) -> list[str]:
    return ["validate", "--dataset", "adress", "--config", str(cfg_path), "--base-dir", str(tmp_path)]


def test_validate_command_passes_on_matching_counts(tmp_path):
    cfg_path = _make_project(tmp_path)
    result = runner.invoke(app, _validate_args(cfg_path, tmp_path))
    assert result.exit_code == 0, result.stdout
    assert "counts match" in result.stdout


def test_validate_command_fails_on_missing_manifest(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    shutil.copy(PACKAGE_DEFAULT_CONFIG, cfg_path)
    result = runner.invoke(app, _validate_args(cfg_path, tmp_path))
    assert result.exit_code == 1


def test_invalid_config_override_exits_nonzero():
    result = runner.invoke(app, ["validate", "--set", "search.top_k=-5"])
    assert result.exit_code == 2


def _make_feature_project(tmp_path: Path) -> Path:
    """A small, fast-running project with LLD features and embeddings already on disk, for testing
    the E0/E1/E3/figures command chain end to end without real audio."""
    cfg = yaml.safe_load(PACKAGE_DEFAULT_CONFIG.read_text())
    cfg["paths"]["manifests"]["adress"] = "data/manifest_adress.csv"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    n_train, n_test, n_features = 24, 12, 10
    ids = [f"t{i}" for i in range(n_train + n_test)]
    manifest = pd.DataFrame({
        "id": ids,
        "split": ["train"] * n_train + ["test"] * n_test,
        "label": ([0] * (n_train // 2) + [1] * (n_train // 2)) + ([0] * (n_test // 2) + [1] * (n_test // 2)),
        "path": [f"{i}.wav" for i in ids],
    })
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest.to_csv(data_dir / "manifest_adress.csv", index=False)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(ids), n_features))
    X[:, 0] += 2.0 * manifest["label"].to_numpy()  # one informative column
    features = pd.DataFrame(X, index=ids, columns=[f"feat_{j}" for j in range(n_features)])
    save_lld_features(features, tmp_path / "features" / "adress" / "full" / "eGeMAPSv02.csv")

    n_layers, dim = 3, 4
    emb = rng.normal(size=(len(ids), n_layers, dim)).astype(np.float32)
    save_embeddings(ids, emb, tmp_path / "features" / "adress" / "full" / "emb_wavlm.npz")

    return cfg_path


def _base(cfg_path: Path, tmp_path: Path, *extra: str) -> list[str]:
    return ["--config", str(cfg_path), "--base-dir", str(tmp_path), *extra]


def test_replicate_command_writes_table_and_reports_gate(tmp_path):
    cfg_path = _make_feature_project(tmp_path)
    result = runner.invoke(app, ["replicate", "adress", *_base(cfg_path, tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "Gate G1" in result.stdout
    assert (tmp_path / "results" / "tables" / "e0_adress_eGeMAPSv02.csv").exists()


def test_null_command_writes_null_and_observed_tables(tmp_path):
    cfg_path = _make_feature_project(tmp_path)
    args = [
        "null", "adress", "--set", "permutation_null.n_permutations=5",
        *_base(cfg_path, tmp_path, "--time-probe", "0"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    assert "p-value=" in result.stdout
    tables = tmp_path / "results" / "tables"
    assert (tables / "e1_null_adress_eGeMAPSv02_train_only.csv").exists()
    assert (tables / "e1_observed_adress_eGeMAPSv02_train_only.csv").exists()


def test_figures_command_combines_both_regimes_into_one_figure(tmp_path):
    cfg_path = _make_feature_project(tmp_path)
    for regime in ("train_only", "test_informed"):
        args = [
            "null", "adress", "--regime", regime, "--set", "permutation_null.n_permutations=5",
            *_base(cfg_path, tmp_path, "--time-probe", "0"),
        ]
        assert runner.invoke(app, args).exit_code == 0

    result = runner.invoke(app, ["figures", *_base(cfg_path, tmp_path)])
    assert result.exit_code == 0, result.stdout
    fig_path = tmp_path / "results" / "figures" / "figure1_adress_eGeMAPSv02.pdf"
    assert fig_path.exists()
    assert fig_path.read_bytes().startswith(b"%PDF")
    # one combined figure, not one per regime
    assert not (tmp_path / "results" / "figures" / "figure1_adress_eGeMAPSv02_train_only.pdf").exists()


def test_probes_command_with_pair_search_null(tmp_path):
    cfg_path = _make_feature_project(tmp_path)
    args = [
        "probes", "adress", "--model", "wavlm", "--pair-search-null",
        "--set", "permutation_null.n_permutations=5",
        *_base(cfg_path, tmp_path),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    assert "CV-selected layer=" in result.stdout
    assert "pair-search null" in result.stdout
    tables = tmp_path / "results" / "tables"
    assert (tables / "e3_probe_adress_wavlm.csv").exists()
    assert any(tables.glob("e3_pairsearch_null_adress_wavlm_layer*_train_only.csv"))
