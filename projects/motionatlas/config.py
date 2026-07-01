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


def dataset_kwargs_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    ann_cfg = cfg.get("annotation", {})

    return {
        "model_path": model_cfg.get("name_or_path", "Qwen/Qwen3-VL-4B-Instruct"),
        "hf_dataset": data_cfg.get("hf_dataset", "maxLWSv2/motionatlas-data"),
        "data_root": data_cfg.get("local_dir", "data/motionatlas-data"),
        "split": data_cfg.get("split", "train"),
        "datasets": data_cfg.get("datasets"),
        "data_paths": data_cfg.get("data_paths"),
        "source_roots": data_cfg.get("source_roots", {}),
        "source_sample_limits": data_cfg.get("source_sample_limits"),
        "source_sample_seed": int(data_cfg.get("source_sample_seed", 42)),
        "recipe_media_root": data_cfg.get("recipe_media_root", ""),
        "image_source_dir": data_cfg.get("image_source_dir", ""),
        "video_source_dir": data_cfg.get("video_source_dir", ""),
        "skip_video": bool(data_cfg.get("skip_video", False)),
        "strict_single_modality": bool(data_cfg.get("strict_single_modality", True)),
        "max_frames": int(data_cfg.get("max_frames", 16)),
        "per_frame_tokens": int(data_cfg.get("per_frame_tokens", 256)),
        "max_seq_length": int(data_cfg.get("max_seq_length", 16384)),
        "max_samples": int(data_cfg.get("max_samples", 0) or 0),
        "annotation_routing": bool(data_cfg.get("annotation_routing", True)),
        "annotation_drop_if_invalid": bool(data_cfg.get("annotation_drop_if_invalid", True)),
        "annotation_mode": ann_cfg.get("mode", "highlight"),
        "annotation_prompt": ann_cfg.get("prompt", "Describe the highlighted object in detail."),
        "annotation_contour_color": ann_cfg.get("contour_color", [0, 255, 0]),
        "annotation_contour_thickness": int(ann_cfg.get("contour_thickness", 2)),
    }
