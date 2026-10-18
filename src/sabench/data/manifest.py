"""Dataset manifests: the single join key (``id``) that everything else -- audio variants, LLD
features, embeddings, channel descriptors -- aligns to. See ``docs/DATA_ACCESS.md`` for the column
contract this module enforces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sabench.config import DatasetCounts
from sabench.constants import LABELS, SPLITS
from sabench.exceptions import DataAlignmentError, ManifestError
from sabench.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("id", "split", "label", "path")
OPTIONAL_COLUMNS = ("age", "gender", "mmse", "speaker")


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a dataset manifest CSV.

    :raises ManifestError: the file is missing, missing required columns, has duplicate ids, or
        contains out-of-contract values (unknown split, non-binary label, empty path).
    """
    p = Path(path)
    if not p.exists():
        raise ManifestError(
            f"manifest not found: {p}. See docs/DATA_ACCESS.md for the expected format and how to "
            "build it from the challenge data release."
        )
    try:
        df = pd.read_csv(p, dtype={"id": str})
    except Exception as exc:
        raise ManifestError(f"could not parse manifest {p} as CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"manifest {p} is missing required column(s): {missing}")

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ManifestError(f"manifest {p} has duplicate id(s): {dupes[:10]}")

    bad_split = sorted(set(df["split"]) - set(SPLITS))
    if bad_split:
        raise ManifestError(f"manifest {p} has unknown split value(s) {bad_split}; expected {SPLITS}")

    try:
        labels = df["label"].astype(int)
    except (ValueError, TypeError) as exc:
        raise ManifestError(f"manifest {p} has non-integer label values: {exc}") from exc
    bad_labels = sorted(set(labels) - set(LABELS))
    if bad_labels:
        raise ManifestError(f"manifest {p} has unknown label value(s) {bad_labels}; expected {LABELS}")
    df["label"] = labels

    if (df["path"].astype(str).str.len() == 0).any():
        raise ManifestError(f"manifest {p} has empty path value(s)")

    df = df.reset_index(drop=True)
    logger.info("loaded manifest", extra={"path": str(p), "n_rows": len(df)})
    return df


@dataclass
class ValidationReport:
    dataset: str
    ok: bool
    actual: dict[str, dict[str, int]]
    expected: dict[str, dict[str, int]]
    mismatches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"{self.dataset}: counts match published splits ({self.actual})"
        lines = [f"{self.dataset}: count mismatch"]
        lines.extend(f"  - {m}" for m in self.mismatches)
        return "\n".join(lines)


def validate_counts(manifest: pd.DataFrame, expected: DatasetCounts, *, dataset: str) -> ValidationReport:
    """Compare a manifest's per-split class counts against the published challenge counts
    (``make validate`` / ``sabench validate``)."""
    actual: dict[str, dict[str, int]] = {}
    mismatches: list[str] = []
    for split in SPLITS:
        sub = manifest.loc[manifest["split"] == split, "label"]
        counts = {str(label): int((sub == label).sum()) for label in LABELS}
        actual[split] = counts
        expected_counts = getattr(expected, split)
        for label, exp_n in expected_counts.items():
            got = counts.get(label, 0)
            if got != exp_n:
                mismatches.append(f"{split}/label={label}: expected {exp_n}, got {got}")
    return ValidationReport(
        dataset=dataset,
        ok=not mismatches,
        actual=actual,
        expected={s: getattr(expected, s) for s in SPLITS},
        mismatches=mismatches,
    )


def align_on_id(
    manifest: pd.DataFrame, features: pd.DataFrame, *, id_col: str = "id", context: str = "features"
) -> pd.DataFrame:
    """Join ``features`` (indexed or columned by ``id``) onto ``manifest`` by id -- never by row
    position, per this project's hard rule.

    :raises DataAlignmentError: any manifest id is missing from ``features``, guarding against a
        silently dropped or reordered recording.
    """
    feats = features.reset_index() if features.index.name == id_col else features.copy()
    if id_col not in feats.columns:
        raise DataAlignmentError(f"{context}: expected an {id_col!r} column or index, found neither")

    merged = manifest.merge(
        feats, on=id_col, how="left", validate="one_to_one", suffixes=("", "_dup"), indicator=True
    )
    missing_ids = merged.loc[merged["_merge"] == "left_only", id_col].tolist()
    if missing_ids:
        raise DataAlignmentError(
            f"{context}: {len(missing_ids)} manifest id(s) have no matching row, e.g. {missing_ids[:5]}"
        )
    return merged.drop(columns="_merge")
