"""Configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config. Paths inside it are relative to the current working directory.

    A config may declare `extends: <other config>`; it is then deep-merged over that parent,
    so experiment configs only need to list what differs. A value of `null` unsets a parameter.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(parent), cfg)
    return cfg
