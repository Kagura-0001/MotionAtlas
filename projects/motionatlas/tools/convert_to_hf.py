#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoProcessor

from projects.motionatlas.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a MotionAtlas Qwen3-VL checkpoint to HuggingFace format.")
    parser.add_argument("--config", required=True, type=Path, help="Training YAML.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="MMEngine/XTuner checkpoint.")
    parser.add_argument("--output-dir", required=True, type=Path, help="HF output directory.")
    parser.add_argument("--base-model", default=None, help="Override base model path from YAML.")
    parser.add_argument("--prefix", default="mllm.", help="Prefix to strip from checkpoint keys.")
    parser.add_argument("--no-merge-base", action="store_true", help="Do not overlay checkpoint on base model weights.")
    return parser.parse_args()


def load_checkpoint(path: Path) -> OrderedDict:
    if path.suffix == ".safetensors":
        return OrderedDict(load_file(str(path)))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return OrderedDict(checkpoint["state_dict"])
        if "model" in checkpoint:
            return OrderedDict(checkpoint["model"])
    return OrderedDict(checkpoint)


def strip_prefix(state_dict: OrderedDict, prefix: str) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if "lora_" in new_key:
            continue
        out[new_key] = value
    return out


def load_base_state_dict(base_model: str) -> OrderedDict:
    base_dir = Path(base_model)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"--no-merge-base is required when base model is not a local directory: {base_model}"
        )
    index = base_dir / "model.safetensors.index.json"
    if index.exists():
        with index.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        merged = OrderedDict()
        for shard in sorted(set(weight_map.values())):
            merged.update(load_file(str(base_dir / shard)))
        return merged
    single = base_dir / "model.safetensors"
    if single.exists():
        return OrderedDict(load_file(str(single)))
    bin_index = base_dir / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with bin_index.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        merged = OrderedDict()
        for shard in sorted(set(weight_map.values())):
            merged.update(torch.load(base_dir / shard, map_location="cpu", weights_only=False))
        return merged
    single_bin = base_dir / "pytorch_model.bin"
    if single_bin.exists():
        return OrderedDict(torch.load(single_bin, map_location="cpu", weights_only=False))
    raise FileNotFoundError(f"No HF weights found under {base_model}")


def overlay(base: OrderedDict, delta: OrderedDict) -> OrderedDict:
    updated = 0
    for key, value in delta.items():
        if key in base and tuple(base[key].shape) == tuple(value.shape):
            base[key] = value
            updated += 1
    if updated == 0:
        raise RuntimeError("No checkpoint tensors matched the base model")
    print(f"Updated {updated} tensors from checkpoint", flush=True)
    return base


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    base_model = args.base_model or cfg.get("model", {}).get("name_or_path")
    if not base_model:
        raise ValueError("Base model path is missing; set model.name_or_path or --base-model")

    state_dict = strip_prefix(load_checkpoint(args.checkpoint), args.prefix)
    if not args.no_merge_base:
        state_dict = overlay(load_base_state_dict(base_model), state_dict)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    AutoConfig.from_pretrained(base_model, trust_remote_code=True).save_pretrained(args.output_dir)
    AutoProcessor.from_pretrained(base_model, trust_remote_code=True).save_pretrained(args.output_dir)
    save_file(state_dict, str(args.output_dir / "model.safetensors"))
    print(f"Saved HuggingFace model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

