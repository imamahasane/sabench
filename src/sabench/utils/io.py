"""Filesystem helpers: atomic writes and path-traversal-safe joins.

Manifests list audio paths as untrusted, user-supplied relative strings (the DementiaBank release
layout differs by year). :func:`safe_join` is the only sanctioned way to turn one of those into a
filesystem path, so a malformed or malicious manifest entry (``../../etc/passwd``, an absolute path,
a symlink escape) cannot make the pipeline read or write outside the intended data root.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from sabench.exceptions import SabenchError


class PathSecurityError(SabenchError):
    """A relative path, once resolved, would escape its intended root directory."""


def safe_join(root: str | Path, relative: str) -> Path:
    """Join ``relative`` onto ``root`` and guarantee the result stays inside ``root``.

    :raises PathSecurityError: ``relative`` is absolute, or resolves outside ``root`` (e.g. via ``..``
        segments or a symlink).
    """
    root_resolved = Path(root).resolve()
    if Path(relative).is_absolute():
        raise PathSecurityError(f"manifest path must be relative, got absolute path: {relative!r}")
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise PathSecurityError(
            f"path {relative!r} resolves outside root {root_resolved}: {candidate}"
        ) from exc
    return candidate


def atomic_write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    """Write ``df`` to ``path`` atomically: a crash or concurrent read never sees a partial file.

    Writes to a temp file in the same directory (so the final ``os.replace`` is on the same
    filesystem and therefore atomic) and renames it into place.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            df.to_csv(fh, index=index)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
