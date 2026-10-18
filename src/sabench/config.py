"""Typed, validated configuration.

``configs/default.yaml`` is the single source of truth for every experiment parameter (seeds, K,
model names, bootstrap counts, ...). This module loads it into a validated :class:`Config` object so
a typo or an out-of-range value fails fast, at startup, with a precise error -- not three hours into a
permutation run. Every result CSV also needs a ``config_hash`` column (a hard rule of this project);
:meth:`Config.config_hash` produces it from the resolved, post-override configuration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sabench.exceptions import ConfigError

PACKAGE_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManifestPaths(StrictModel):
    adress: str
    adresso: str
    pitt: str | None = None
    taukadial: str | None = None


class PathsConfig(StrictModel):
    raw: str
    interim: str
    features: str
    splits: str
    results: str
    manifests: ManifestPaths


class DatasetCounts(StrictModel):
    train: dict[str, int]
    test: dict[str, int]


class AudioConfig(StrictModel):
    sample_rate: int = Field(gt=0)
    variants: list[str]
    min_variant_seconds: float = Field(ge=0)


class VADConfig(StrictModel):
    model: str
    backend: Literal["auto", "pyannote", "energy"] = "auto"
    min_duration_on: float = Field(ge=0)
    min_duration_off: float = Field(ge=0)
    fallback: Literal["energy"] = "energy"
    energy_frame_ms: float = Field(gt=0)
    energy_hop_ms: float = Field(gt=0)
    energy_threshold_db: float


class LLDConfig(StrictModel):
    featuresets: list[str]


class SearchConfig(StrictModel):
    top_k: int = Field(gt=1)
    rank_method: Literal["smd", "lr"]
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(default=1, ge=1)
    regimes: list[str]

    @field_validator("regimes")
    @classmethod
    def _known_regimes(cls, v: list[str]) -> list[str]:
        allowed = {"test_informed", "train_only"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"unknown regime(s) {bad}; expected subset of {allowed}")
        return v


class PermutationNullConfig(StrictModel):
    n_permutations: int = Field(gt=0)
    n_jobs: int


class UncertaintyConfig(StrictModel):
    n_bootstrap: int = Field(gt=0)
    alpha: float = Field(gt=0, lt=1)
    n_monte_carlo: int = Field(gt=0)
    test_size: float = Field(gt=0, lt=1)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(gt=0)


class EmbeddingsProbeConfig(StrictModel):
    cs: list[float] = Field(min_length=1)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(default=1, ge=1)


class EmbeddingsConfig(StrictModel):
    chunk_seconds: int = Field(gt=0)
    device: Literal["auto", "cpu", "cuda"]
    models: dict[str, str]
    probe: EmbeddingsProbeConfig


class TextConfig(StrictModel):
    asr_model: str = "openai/whisper-small"
    models: dict[str, str] = Field(
        default_factory=lambda: {"bert": "bert-base-uncased", "roberta": "roberta-base", "xlmr": "xlm-roberta-base"}
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    max_tokens: int = Field(default=512, gt=2)


class ChannelConfig(StrictModel):
    ltas_bands: int = Field(gt=0)


class InterventionsConfig(StrictModel):
    loudness_target_lufs: float
    band_limit_hz: list[float] = Field(min_length=2, max_length=2)
    noise_floor_match_db: float


class CrossDatasetConfig(StrictModel):
    check_speaker_overlap: bool


class ReplicationTarget(StrictModel):
    test_f1: float = Field(ge=0, le=1)
    mc_mean: float = Field(ge=0, le=1)


class ReplicationConfig(StrictModel):
    tolerance: float = Field(gt=0, lt=1)
    targets: dict[str, ReplicationTarget]


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False
    file: str | None = None


class Config(StrictModel):
    """Root configuration object. Immutable once loaded (``frozen=True``): derive a new config via
    :func:`load_config` with overrides rather than mutating one in place, so a `config_hash` always
    reflects exactly what ran.
    """

    seed: int
    paths: PathsConfig
    datasets: list[str]
    expected_counts: dict[str, DatasetCounts]
    audio: AudioConfig
    vad: VADConfig
    lld: LLDConfig
    search: SearchConfig
    permutation_null: PermutationNullConfig
    uncertainty: UncertaintyConfig
    embeddings: EmbeddingsConfig
    text: TextConfig = Field(default_factory=TextConfig)
    channel: ChannelConfig
    interventions: InterventionsConfig
    cross_dataset: CrossDatasetConfig
    replication: ReplicationConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def _datasets_have_counts_and_manifests(self) -> Config:
        missing_counts = [d for d in self.datasets if d not in self.expected_counts]
        if missing_counts:
            raise ValueError(f"datasets missing expected_counts entries: {missing_counts}")
        known_manifests = {k for k, v in self.paths.manifests.model_dump().items() if v is not None}
        missing_manifests = [d for d in self.datasets if d not in known_manifests]
        if missing_manifests:
            raise ValueError(f"datasets missing a manifest path: {missing_manifests}")
        return self

    def config_hash(self, length: int = 10) -> str:
        """Short, stable hash of the fully-resolved config. Stored in every result CSV so a number can
        always be traced back to the exact settings that produced it."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _deep_set(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load, override, and validate the experiment configuration.

    :param path: YAML file to load. Defaults to the packaged ``configs/default.yaml``.
    :param overrides: ``"dotted.key.path=value"`` strings, applied in order after the file is parsed
        (as used by ``--set`` on the CLI). Values are parsed as YAML scalars, so ``true``, ``1.5`` and
        ``[1, 2]`` all come through as their native Python types.
    :raises ConfigError: the file is missing/malformed, an override is malformed, or the result fails
        schema validation.
    """
    cfg_path = Path(path) if path is not None else PACKAGE_DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"top level of {cfg_path} must be a mapping, got {type(raw).__name__}")

    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"malformed override {item!r}; expected 'dotted.key=value'")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"malformed override {item!r}; empty key")
        _deep_set(raw, key, _parse_override_value(value.strip()))

    try:
        return Config(**raw)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigError(f"invalid configuration in {cfg_path}: {exc}") from exc


class ResolvedPaths:
    """Absolute, created-on-demand paths derived from a :class:`Config` and a base directory
    (the directory the config file was loaded from, or the current working directory)."""

    def __init__(self, config: Config, base_dir: str | Path = "."):
        self.base_dir = Path(base_dir).resolve()
        self.raw = self.base_dir / config.paths.raw
        self.interim = self.base_dir / config.paths.interim
        self.features = self.base_dir / config.paths.features
        self.splits = self.base_dir / config.paths.splits
        self.results = self.base_dir / config.paths.results
        self.tables = self.results / "tables"
        self.figures = self.results / "figures"
        self.logs = self.results / "logs"
        self.manifests = {
            dataset: self.base_dir / path
            for dataset, path in config.paths.manifests.model_dump().items()
            if path is not None
        }
        self.log_file = (
            self.base_dir / config.logging.file if config.logging.file else None
        )

    def ensure_output_dirs(self) -> None:
        """Create every directory this run will write to. Never creates ``raw`` (read-only source data)."""
        for d in (self.interim, self.features, self.splits, self.tables, self.figures, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, dataset: str) -> Path:
        try:
            return self.manifests[dataset]
        except KeyError as exc:
            raise ConfigError(f"no manifest path configured for dataset {dataset!r}") from exc
