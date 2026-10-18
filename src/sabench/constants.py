"""Fixed values shared across the codebase. Nothing here should ever need a CLI override —
things users can tune live in ``configs/default.yaml`` via :mod:`sabench.config` instead.
"""
from __future__ import annotations

from typing import Final

HEALTHY_CONTROL: Final[int] = 0
ALZHEIMERS_DEMENTIA: Final[int] = 1
LABELS: Final[tuple[int, int]] = (HEALTHY_CONTROL, ALZHEIMERS_DEMENTIA)

REGIME_TEST_INFORMED: Final = "test_informed"
REGIME_TRAIN_ONLY: Final = "train_only"
REGIMES: Final = (REGIME_TEST_INFORMED, REGIME_TRAIN_ONLY)

SPLIT_TRAIN: Final[str] = "train"
SPLIT_TEST: Final[str] = "test"
SPLITS: Final[tuple[str, str]] = (SPLIT_TRAIN, SPLIT_TEST)

AUDIO_VARIANT_FULL: Final[str] = "full"
AUDIO_VARIANT_SPEECH: Final[str] = "speech"
AUDIO_VARIANT_SILENCE: Final[str] = "silence"
AUDIO_VARIANTS: Final[tuple[str, str, str]] = (
    AUDIO_VARIANT_FULL, AUDIO_VARIANT_SPEECH, AUDIO_VARIANT_SILENCE,
)

RESULT_COLUMNS_REQUIRED: Final[tuple[str, ...]] = ("regime", "seed", "config_hash")

# Result CSVs are the only source of truth for numbers in the paper; never hand-edit them.
RESULTS_TABLES_DIRNAME: Final[str] = "tables"
RESULTS_FIGURES_DIRNAME: Final[str] = "figures"
RESULTS_LOGS_DIRNAME: Final[str] = "logs"
