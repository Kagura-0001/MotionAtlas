#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.motionatlas.config import dataset_kwargs_from_cfg, load_yaml_config
from projects.motionatlas.datasets import MotionAtlasQwen3VLDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MotionAtlas Qwen3-VL training data.")
    parser.add_argument("config", type=Path, help="Path to MotionAtlas training YAML.")
    parser.add_argument("--limit", type=int, default=16, help="Number of samples to materialize.")
    parser.add_argument("--cfg-options", nargs="+", default=[], help="Override YAML values, e.g. data.max_samples=8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, overrides=args.cfg_options)
    dataset = MotionAtlasQwen3VLDataset(**dataset_kwargs_from_cfg(cfg))

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
