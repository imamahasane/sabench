"""sabench: search-aware evaluation of acoustic and embedding features on ADReSS/ADReSSo.

Public API surface is intentionally small -- most work happens through the CLI
(``sabench --help``) or by importing specific submodules (``sabench.modeling.search``,
``sabench.modeling.nulls``, ...). This module only exposes version metadata and the two things
almost every caller needs: configuration loading and logger setup.
"""
from __future__ import annotations

from sabench.config import Config, load_config
from sabench.logging_config import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = ["Config", "load_config", "configure_logging", "get_logger", "__version__"]
