"""Resolve project paths from configs/config.yaml so nothing is hardcoded.

Every other module should get its paths through load_config()/resolve_path()
instead of building path strings directly.
"""
from __future__ import annotations

from pathlib import Path
import yaml

# src/utils/paths.py -> src/utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Run this project from the nids-project root, or check configs/config.yaml exists."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a path from config.yaml (relative to project root) into an absolute Path."""
    return PROJECT_ROOT / relative_path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
