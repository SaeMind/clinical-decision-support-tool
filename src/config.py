"""Configuration utilities for the clinical triage CDS repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


def load_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML settings from disk.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Dictionary containing project settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        settings = yaml.safe_load(file_obj)
    return settings


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path.

    Args:
        relative_path: Relative or absolute path.

    Returns:
        Absolute path.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(settings: dict[str, Any]) -> Path:
    """Create and return the configured output directory.

    Args:
        settings: Loaded settings dictionary.

    Returns:
        Output directory path.
    """
    output_dir = resolve_path(settings["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
