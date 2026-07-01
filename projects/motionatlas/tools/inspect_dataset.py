#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from projects.motionatlas.config import extract_cfg_options, load_yaml_config
from projects.motionatlas.datasets import MotionAtlasQwen3VLDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MotionAtlas Qwen3-VL training data.")
    parser.add_argument("config", type=Path, help="Path to MotionAtlas training YAML.")
    parser.add_argument("--limit", type=int, default=16, help="Number of samples to materialize.")
    parser.add_argument("--cfg-options", nargs="+", default=[], help="Override YAML values, e.g. data.max_samples=8")
    return parser.parse_args()


def build_dataset_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    ann_cfg = cfg.get("annotation", {})
    return {
        "model_path": model_cfg.get("name_or_path", "Qwen/Qwen3-VL-4B-Instruct"),
        "hf_dataset": data_cfg.get("hf_dataset", "maxLWSv2/motionatlas-data"),
        "data_root": data_cfg.get("local_dir", "data/motionatlas-data"),
        "split": data_cfg.get("split", "train"),
        "source_roots": data_cfg.get("source_roots", {}),
        "max_frames": int(data_cfg.get("max_frames", 16)),
        "per_frame_tokens": int(data_cfg.get("per_frame_tokens", 256)),
        "max_seq_length": int(data_cfg.get("max_seq_length", 16384)),
        "max_samples": int(data_cfg.get("max_samples", 0) or 0),
        "annotation_mode": ann_cfg.get("mode", "highlight"),
        "annotation_prompt": ann_cfg.get("prompt", "Describe the highlighted object in detail."),
        "annotation_contour_color": ann_cfg.get("contour_color", [0, 255, 0]),
        "annotation_contour_thickness": int(ann_cfg.get("contour_thickness", 2)),
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, overrides=args.cfg_options)
    dataset = MotionAtlasQwen3VLDataset(**build_dataset_kwargs(cfg))

    failures = []
    rows = []
    limit = min(args.limit, len(dataset))
    for idx in range(limit):
        try:
            item = dataset[idx]
        except Exception as exc:
            failures.append({"index": idx, "error": repr(exc)})
            continue
        rows.append(
            {
                "index": idx,
                "seq_len": int(item["input_ids"].shape[-1]),
                "has_images": "pixel_values" in item,
                "image_grid_shape": list(item["image_grid_thw"].shape) if "image_grid_thw" in item else None,
            }
        )

    report = {
        "status": "ok" if not failures else "failed",
        "dataset_len": len(dataset),
        "checked": limit,
        "materialized": len(rows),
        "failures": failures[:10],
        "samples": rows[:10],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

