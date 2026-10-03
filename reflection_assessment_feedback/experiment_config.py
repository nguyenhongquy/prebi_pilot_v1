from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.toml"


def load_experiment_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)
    if config.get("config_version") != 1:
        raise ValueError("Unsupported experiment config version.")
    return config


def experiment_config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
