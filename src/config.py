"""Configuration governance for reproducible ML operations.

This module standardizes how project settings are loaded and validated so
experiments are comparable, auditable, and aligned with business reporting
requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration and return a plain dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML object at root level.")

    return config
