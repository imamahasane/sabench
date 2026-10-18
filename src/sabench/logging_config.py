"""Structured logging setup shared by every CLI command and library entry point.

Call :func:`configure_logging` once, as early as possible (the CLI does this before dispatching to
any subcommand). Library modules only ever call ``logging.getLogger(__name__)`` and never configure
handlers themselves, so importing ``sabench`` has no side effects on the host application's logging.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Renders one JSON object per line: timestamp, level, logger name, message, and any ``extra=``."""

    _RESERVED = set(logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None,
    ).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and key != "message":
                payload[key] = value
        return json.dumps(payload, default=str)


_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(
    level: str = "INFO",
    *,
    json_format: bool = False,
    log_file: str | Path | None = None,
) -> None:
    """Configure the root ``sabench`` logger exactly once per process.

    Safe to call more than once (e.g. once from the CLI, once from a test fixture): subsequent calls
    update the level/handlers in place instead of stacking duplicate handlers.
    """
    global _configured
    root = logging.getLogger("sabench")
    root.setLevel(level.upper())
    root.handlers.clear()

    formatter: logging.Formatter
    formatter = JsonFormatter() if json_format else logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Configures a sane default (INFO, console only) if nobody has yet."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
