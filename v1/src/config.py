"""Tiny config loader shared by all modules. Single source of truth: config/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config.yaml as a dict."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open() as f:
        return yaml.safe_load(f)


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path against REPO_ROOT."""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)
