"""Low-level descriptor (LLD) functional extraction via openSMILE: eGeMAPSv02 (88 features) and
ComParE 2016 (6,373 features), matching the feature sets used by the original paper's Table 3.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sabench.exceptions import FeatureExtractionError, ModelUnavailableError
from sabench.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_FEATURESETS: tuple[str, ...] = ("eGeMAPSv02", "ComParE_2016")
FeatureSetName = Literal["eGeMAPSv02", "ComParE_2016"]


def _build_smile(featureset: FeatureSetName):
    try:
        import opensmile
    except ImportError as exc:
        raise ModelUnavailableError(
            "opensmile is not installed. Install requirements-extra.txt to extract LLD features."
        ) from exc
    if featureset not in SUPPORTED_FEATURESETS:
        raise FeatureExtractionError(f"unknown feature set {featureset!r}; expected {SUPPORTED_FEATURESETS}")
    feature_set = getattr(opensmile.FeatureSet, featureset)
    return opensmile.Smile(feature_set=feature_set, feature_level=opensmile.FeatureLevel.Functionals)


def extract_lld_batch(
    paths_by_id: Mapping[str, str | Path], featureset: FeatureSetName
) -> pd.DataFrame:
    """Extract functionals for every recording in ``paths_by_id``.

    Returns a DataFrame indexed by ``id`` with one column per openSMILE feature (names preserved
    exactly, since the paper's Table 3 cites them by name). A recording whose audio is missing,
    unreadable, or below openSMILE's minimum duration gets an all-NaN row instead of aborting the
    whole batch -- consistent with this project's "missing is NaN, never dropped" rule.
    """
    smile = _build_smile(featureset)
    rows: dict[str, pd.Series] = {}
    columns: list[str] | None = None
    failures: list[str] = []

    for rec_id, path in paths_by_id.items():
        p = Path(path)
        try:
            if not p.exists():
                raise FeatureExtractionError(f"audio file not found: {p}")
            result = smile.process_file(str(p))
            series = result.iloc[0]
            if columns is None:
                columns = list(series.index)
            rows[rec_id] = series
        except Exception as exc:
            failures.append(rec_id)
            logger.warning(
                "LLD extraction failed, recording as NaN", extra={"id": rec_id, "reason": str(exc)}
            )

    if columns is None:
        raise FeatureExtractionError(
            f"LLD extraction failed for every recording in this batch (featureset={featureset})"
        )

    for rec_id in failures:
        rows[rec_id] = pd.Series(np.nan, index=columns)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.loc[list(paths_by_id.keys())]  # preserve caller's id order, not dict/failure order
    df.index.name = "id"

    constant_cols = df.columns[df.nunique(dropna=True) <= 1]
    if len(constant_cols):
        logger.info(
            "constant-valued LLD columns detected (kept, not dropped)",
            extra={"featureset": featureset, "n_constant": len(constant_cols)},
        )
    return df
