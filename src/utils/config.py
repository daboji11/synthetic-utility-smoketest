"""Small YAML defaults merger, with paths relative to the referring file."""
from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    defaults = config.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError("defaults must be a list")
    merged: dict = {}
    for item in defaults:
        merged = _merge(merged, load_config(path.parent / item))
    return _merge(merged, config)


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
