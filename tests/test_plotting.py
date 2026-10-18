from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sabench.viz.plotting import COLORBLIND_SAFE_PALETTE, plot_layerwise_probe, plot_null_distribution


def test_palette_values_are_valid_hex_colors():
    for name, value in COLORBLIND_SAFE_PALETTE.items():
        assert value.startswith("#"), name
        assert len(value) == 7, name
        int(value[1:], 16)  # raises ValueError if not valid hex


def test_plot_null_distribution_writes_a_nonempty_pdf(tmp_path):
    rng = np.random.default_rng(0)
    scores = {
        "train_only": rng.normal(0.5, 0.05, size=100),
        "test_informed": rng.normal(0.6, 0.05, size=100),
    }
    observed = {"train_only": 0.55, "test_informed": 0.62}
    out = tmp_path / "figure1.pdf"
    result_path = plot_null_distribution(scores, observed, title="toy", out_path=out)
    assert result_path == out
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes().startswith(b"%PDF")


def test_plot_null_distribution_handles_missing_observed(tmp_path):
    scores = {"train_only": np.array([0.4, 0.5, 0.6])}
    out = tmp_path / "figure1_partial.pdf"
    plot_null_distribution(scores, {}, out_path=out)
    assert out.exists()


def test_plot_layerwise_probe_writes_a_nonempty_pdf(tmp_path):
    df = pd.DataFrame({
        "layer": range(6),
        "test_f1": [0.5, 0.55, 0.6, 0.62, 0.58, 0.57],
    })
    out = tmp_path / "figure2.pdf"
    result_path = plot_layerwise_probe(df, resampled_mean=0.5, out_path=out, title="toy")
    assert result_path == out
    assert out.stat().st_size > 0


def test_plot_layerwise_probe_with_ci_band_and_resampled_ci(tmp_path):
    df = pd.DataFrame({
        "layer": range(4),
        "test_f1": [0.5, 0.55, 0.6, 0.58],
        "ci_low": [0.4, 0.45, 0.5, 0.48],
        "ci_high": [0.6, 0.65, 0.7, 0.68],
    })
    out = tmp_path / "figure2_ci.pdf"
    plot_layerwise_probe(df, resampled_mean=0.5, resampled_ci=(0.45, 0.55), out_path=out)
    assert out.exists()


def test_plot_functions_create_parent_directories(tmp_path):
    out = tmp_path / "nested" / "dir" / "figure1.pdf"
    plot_null_distribution({"train_only": np.array([0.5])}, {}, out_path=out)
    assert out.exists()


@pytest.mark.parametrize("n_points", [0, 1])
def test_plot_null_distribution_handles_degenerate_sizes(tmp_path, n_points):
    scores = {"train_only": np.full(max(n_points, 1), 0.5)[:n_points] if n_points else np.array([])}
    out = tmp_path / f"figure1_{n_points}.pdf"
    plot_null_distribution(scores, {}, out_path=out)
    assert out.exists()
