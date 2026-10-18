from __future__ import annotations

import pytest
from pydantic import ValidationError

from sabench.config import ResolvedPaths, load_config
from sabench.exceptions import ConfigError


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert cfg.seed == 1234
    assert cfg.search.top_k == 50
    assert set(cfg.datasets) == {"adress", "adresso"}


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.seed = 999  # type: ignore[misc]


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\nnot_a_real_key: true\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_override_dotted_path():
    cfg = load_config(overrides=["search.top_k=99", "seed=7"])
    assert cfg.search.top_k == 99
    assert cfg.seed == 7


def test_malformed_override_raises():
    with pytest.raises(ConfigError):
        load_config(overrides=["no_equals_sign_here"])


def test_unknown_regime_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["search.regimes=[bogus_regime]"])


def test_config_hash_changes_with_overrides():
    a = load_config()
    b = load_config(overrides=["seed=9999"])
    assert a.config_hash() != b.config_hash()
    assert a.config_hash() == load_config().config_hash()


def test_resolved_paths_never_creates_raw(tmp_path):
    cfg = load_config()
    paths = ResolvedPaths(cfg, base_dir=tmp_path)
    paths.ensure_output_dirs()
    assert not paths.raw.exists()
    assert paths.tables.exists()
    assert paths.figures.exists()
