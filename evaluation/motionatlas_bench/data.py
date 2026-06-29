from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(record)
    return records


def load_mcqs(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    required = {
        "id",
        "sample_id",
        "event_id",
        "video_path",
        "video_type",
        "question",
        "options",
        "answer_index",
    }
    for idx, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"{path}:{idx}: missing fields: {sorted(missing)}")
        if not isinstance(record["options"], list) or not record["options"]:
            raise ValueError(f"{path}:{idx}: options must be a non-empty list")
        record["sample_id"] = int(record["sample_id"])
        record["answer_index"] = int(record["answer_index"])
    return records


def load_target_masks(path: Path) -> dict[int, dict[str, Any]]:
    records = load_jsonl(path)
    masks_by_sample: dict[int, dict[str, Any]] = {}
    for line_no, record in enumerate(records, start=1):
        if set(record) != {"sample_id", "masks"}:
            raise ValueError(
                f"{path}:{line_no}: expected exactly fields ['sample_id', 'masks'], "
                f"got {sorted(record)}"
            )
        sample_id = int(record["sample_id"])
        masks = record["masks"]
        if not isinstance(masks, dict) or not masks:
            raise ValueError(f"{path}:{line_no}: masks must be a non-empty object")
        if sample_id in masks_by_sample:
            raise ValueError(f"{path}:{line_no}: duplicate sample_id={sample_id}")
        masks_by_sample[sample_id] = masks
    return masks_by_sample


def group_mcqs_by_sample(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[int(record["sample_id"])].append(record)
    return dict(groups)


def representative_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must be non-empty")
    first = records[0]
    for record in records[1:]:
        for key in ("sample_id", "video_path", "video_type"):
            if record.get(key) != first.get(key):
                raise ValueError(
                    f"sample_id={first.get('sample_id')} has inconsistent {key}: "
                    f"{first.get(key)!r} vs {record.get(key)!r}"
                )
    return first
