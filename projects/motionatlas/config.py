from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_value(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return ""

    return _ENV_PATTERN.sub(replace, value)


def expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env_value(obj)
    if isinstance(obj, list):
        return [expand_env(item) for item in obj]
    if isinstance(obj, dict):
        return {key: expand_env(value) for key, value in obj.items()}
    return obj


def load_yaml_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    cfg = expand_env(cfg)
    for override in overrides or []:
        apply_override(cfg, override)
    return cfg


def apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must have KEY=VALUE form: {override}")
    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override contains empty key: {override}")
    try:
        value = yaml.safe_load(raw_value)
    except Exception:
        value = raw_value

    cursor: dict[str, Any] = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def extract_cfg_options(argv: list[str]) -> tuple[list[str], list[str]]:
    """Remove GAR/MMEngine-style --cfg-options from argv and return overrides."""
    cleaned: list[str] = []
    overrides: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token != "--cfg-options":
            cleaned.append(token)
            idx += 1
            continue
        idx += 1
        while idx < len(argv) and not argv[idx].startswith("--"):
            overrides.append(argv[idx])
            idx += 1
    return cleaned, overrides


def deep_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def with_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    for override in overrides or []:
        apply_override(result, override)
    return result

