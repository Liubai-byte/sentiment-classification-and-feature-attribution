"""YAML config loading."""

import yaml


def load_config(path):
    """Read a YAML config file and return a dict."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config
