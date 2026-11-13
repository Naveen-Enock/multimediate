"""Config loading + lightweight CLI overrides."""

from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides, e.g. {'train.max_steps': 5}."""
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg
