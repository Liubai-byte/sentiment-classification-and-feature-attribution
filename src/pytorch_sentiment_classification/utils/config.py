"""YAML 配置加载。"""

import yaml


def load_config(path):
    """读取 YAML 配置文件，返回 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config
