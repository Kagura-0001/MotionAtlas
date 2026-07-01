from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

IGNORE_INDEX = -100


def _pad_position_ids(position_ids: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[2] for t in position_ids)
    padded = []
    for tensor in position_ids:
        pad_len = max_len - tensor.shape[2]
        padded.append(torch.nn.functional.pad(tensor, (0, pad_len), "constant", 1))
    return torch.cat(padded, dim=1)


@dataclass
class Qwen3VLMotionAtlasCollator:
    tokenizer_cfg: dict[str, Any]

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(**self.tokenizer_cfg)

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = [item["input_ids"].squeeze(0) for item in instances]
        labels = [item["labels"].squeeze(0) for item in instances]
        position_ids = [item["position_ids"] for item in instances]

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
            "position_ids": _pad_position_ids(position_ids),
        }
        batch["attention_mask"] = batch["input_ids"].ne(pad_token_id)

        if any("pixel_values" in item for item in instances):
            batch["pixel_values"] = torch.cat([item["pixel_values"] for item in instances if "pixel_values" in item], dim=0)
            batch["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in instances if "image_grid_thw" in item], dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if any("pixel_values_videos" in item for item in instances):
            batch["pixel_values_videos"] = torch.cat([item["pixel_values_videos"] for item in instances if "pixel_values_videos" in item], dim=0)
            batch["video_grid_thw"] = torch.cat([item["video_grid_thw"] for item in instances if "video_grid_thw" in item], dim=0)
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        second_per_grid_ts = []
        for item in instances:
            value = item.get("second_per_grid_ts")
            if isinstance(value, list):
                second_per_grid_ts.extend(value)
            elif value is not None:
                second_per_grid_ts.append(value)
        batch["second_per_grid_ts"] = second_per_grid_ts or None
        return {"data": batch, "data_samples": None}


def qwen3vl_motionatlas_collect(instances: list[dict[str, Any]], tokenizer_cfg: dict[str, Any]) -> dict[str, Any]:
    collator = getattr(qwen3vl_motionatlas_collect, "_collator", None)
    if collator is None:
        collator = Qwen3VLMotionAtlasCollator(tokenizer_cfg=tokenizer_cfg)
        qwen3vl_motionatlas_collect._collator = collator
    return collator(instances)

