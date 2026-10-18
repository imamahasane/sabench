"""Glue between configuration, resolved filesystem paths, and result persistence -- the one object
every CLI command constructs first.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sabench.config import Config, ResolvedPaths, load_config
from sabench.data.manifest import ValidationReport, load_manifest, validate_counts
from sabench.logging_config import configure_logging, get_logger
from sabench.pipeline.io import check_result_integrity, with_provenance_columns
from sabench.utils.io import atomic_write_csv

logger = get_logger(__name__)


class ExperimentRunner:
    """Owns the resolved config/paths for one invocation and provides the few operations every
    stage needs: loading a manifest, validating it, and persisting a result table with the
    provenance columns (`regime`, `seed`, `config_hash`) this project's hard rules require.
    """

    def __init__(self, config: Config, base_dir: str | Path = "."):
        self.config = config
        self.paths = ResolvedPaths(config, base_dir=base_dir)

    @classmethod
    def from_file(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
        *,
        base_dir: str | Path = ".",
    ) -> ExperimentRunner:
        config = load_config(config_path, overrides)
        runner = cls(config, base_dir=base_dir)
        configure_logging(
            level=config.logging.level, json_format=config.logging.json_format, log_file=runner.paths.log_file
        )
        return runner

    def load_manifest(self, dataset: str) -> pd.DataFrame:
        return load_manifest(self.paths.manifest_path(dataset))

    def validate_manifest(self, dataset: str) -> ValidationReport:
        manifest = self.load_manifest(dataset)
        return validate_counts(manifest, self.config.expected_counts[dataset], dataset=dataset)

    def save_table(
        self, df: pd.DataFrame, name: str, *, regime: str | None = None, seed: int | None = None
    ) -> Path:
        """Write ``df`` to ``results/tables/<name>.csv`` with provenance columns, atomically."""
        out = df
        if regime is not None or seed is not None:
            out = with_provenance_columns(
                df, regime=regime or "", seed=seed if seed is not None else self.config.seed,
                config_hash=self.config.config_hash(),
            )
        else:
            out = df.copy()
            out["config_hash"] = self.config.config_hash()
        check_result_integrity(out, ("config_hash",))
        path = self.paths.tables / f"{name}.csv"
        atomic_write_csv(out, path)
        logger.info("wrote result table", extra={"path": str(path), "n_rows": len(out)})
        return path
