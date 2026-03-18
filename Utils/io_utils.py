"""IO and config utilities"""
import os
import sys
import yaml
import json
import importlib


def load_yaml_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.full_load(f)


def instantiate_from_config(config):
    if config is None:
        return None
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))
