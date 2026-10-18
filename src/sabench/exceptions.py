"""Domain exception hierarchy.

Every error that can reasonably occur while running the pipeline is raised as one of these,
never as a bare ``Exception``/``ValueError``, so callers (CLI, tests, notebooks) can catch at the
right granularity and so error messages always carry enough context to act on.
"""
from __future__ import annotations


class SabenchError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(SabenchError):
    """The configuration file or an override is missing, malformed, or fails validation."""


class ManifestError(SabenchError):
    """A dataset manifest is missing, malformed, or does not match the expected published counts."""


class DataAlignmentError(SabenchError):
    """Two arrays or tables that should be joined on ``id`` are misaligned or have mismatched ids."""


class AudioProcessingError(SabenchError):
    """Loading, resampling, or segmenting an audio file failed."""


class VADError(AudioProcessingError):
    """Voice-activity detection failed (model unavailable, gated access, or malformed audio)."""


class FeatureExtractionError(SabenchError):
    """Low-level feature or embedding extraction failed for one or more recordings."""


class ModelUnavailableError(SabenchError):
    """An optional heavy dependency (opensmile, torch, transformers, pyannote.audio) is not installed,
    or a gated Hugging Face model could not be downloaded (missing/invalid ``HF_TOKEN``)."""


class SearchError(SabenchError):
    """The feature search (ranking or pairwise search) received invalid inputs or an invalid regime."""


class ResultIntegrityError(SabenchError):
    """A generated result table fails a post-hoc sanity check (e.g. ``max_test_f1 < test_f1``)."""
