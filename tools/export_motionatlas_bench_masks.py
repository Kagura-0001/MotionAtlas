#!/usr/bin/env python3
"""Export public MotionAtlas-Bench target masks from internal annotations.

The output is gzip-compressed JSONL. Each line has exactly:

    {"sample_id": <int>, "masks": {"0": <COCO RLE>, ...}}

Frame indices are normalized to the public media order: 0-based decoded video
frames for mp4 media, and 0-based indices into sorted image filenames for image
directory media.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(record)
    return records


def group_public_samples(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for record in records:
        sample_id = int(record["sample_id"])
        entry = grouped.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "video_path": record["video_path"],
                "video_type": record["video_type"],
                "num_mcqs": 0,
            },
        )
        for key in ("video_path", "video_type"):
            if entry[key] != record[key]:
                raise ValueError(
                    f"sample_id={sample_id} has inconsistent {key}: "
                    f"{entry[key]!r} vs {record[key]!r}"
                )
        entry["num_mcqs"] += 1
    return grouped


def get_frame_count(data_root: Path, public_sample: dict[str, Any]) -> int:
    media_path = data_root / public_sample["video_path"]
    video_type = public_sample["video_type"]
    if video_type == "images":
        if not media_path.is_dir():
            raise ValueError(f"{media_path} is not an image directory")
        return sum(
            1
            for path in sorted(media_path.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if video_type == "video":
        if not media_path.is_file():
            raise ValueError(f"{media_path} is not a video file")
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to validate video frame counts") from exc
        capture = cv2.VideoCapture(str(media_path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if frame_count <= 0:
            raise ValueError(f"could not read frame count from {media_path}")
        return frame_count

    raise ValueError(f"unsupported video_type={video_type!r}")


def extract_mask_map(raw_sample: dict[str, Any], sample_id: int) -> dict[int, Any]:
    masks = raw_sample.get("masks")
    if not isinstance(masks, list) or not masks:
        raise ValueError(f"sample_id={sample_id} has no mask list in raw annotations")
    if len(masks) != 1:
        raise ValueError(f"sample_id={sample_id} expected one target mask, got {len(masks)}")
    mask_map = masks[0]
    if not isinstance(mask_map, dict) or not mask_map:
        raise ValueError(f"sample_id={sample_id} has invalid mask map")
    out: dict[int, Any] = {}
    for frame_idx, rle in mask_map.items():
        if not str(frame_idx).isdigit():
            continue
        if not isinstance(rle, dict) or "size" not in rle or "counts" not in rle:
            raise ValueError(f"sample_id={sample_id} frame={frame_idx} has invalid COCO RLE")
        out[int(frame_idx)] = {"size": rle["size"], "counts": rle["counts"]}
    if not out:
        raise ValueError(f"sample_id={sample_id} has no numeric mask frames")
    return out


def normalize_mask_map(mask_map: dict[int, Any], frame_count: int, sample_id: int) -> tuple[dict[str, Any], int]:
    keys = sorted(mask_map)
    if keys[0] >= 0 and keys[-1] < frame_count:
        offset = 0
    elif keys[0] >= 1 and keys[-1] <= frame_count:
        offset = -1
    else:
        raise ValueError(
            f"sample_id={sample_id} mask keys [{keys[0]}, {keys[-1]}] "
            f"do not fit frame_count={frame_count}"
        )

    normalized: dict[str, Any] = {}
    for old_idx in keys:
        new_idx = old_idx + offset
        if new_idx < 0 or new_idx >= frame_count:
            raise ValueError(
                f"sample_id={sample_id} normalized frame index {new_idx} "
                f"outside frame_count={frame_count}"
            )
        key = str(new_idx)
        if key in normalized:
            raise ValueError(f"sample_id={sample_id} duplicate normalized frame index {new_idx}")
        normalized[key] = mask_map[old_idx]
    return normalized, offset


def build_records(mcq_path: Path, raw_path: Path, data_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    public_records = load_jsonl(mcq_path)
    public_samples = group_public_samples(public_records)
    raw_data = load_json(raw_path)
    if isinstance(raw_data, dict) and isinstance(raw_data.get("samples"), list):
        raw_samples = raw_data["samples"]
    elif isinstance(raw_data, list):
        raw_samples = raw_data
    else:
        raise ValueError(f"{raw_path} must contain a sample list")

    raw_by_id = {int(sample["id"]): sample for sample in raw_samples if isinstance(sample, dict) and "id" in sample}

    records: list[dict[str, Any]] = []
    total_mask_frames = 0
    shifted_samples = 0
    for sample_id in sorted(public_samples):
        if sample_id not in raw_by_id:
            raise ValueError(f"sample_id={sample_id} is missing from raw annotations")
        public_sample = public_samples[sample_id]
        frame_count = get_frame_count(data_root, public_sample)
        mask_map, offset = normalize_mask_map(extract_mask_map(raw_by_id[sample_id], sample_id), frame_count, sample_id)
        if offset:
            shifted_samples += 1
        total_mask_frames += len(mask_map)
        records.append({"sample_id": sample_id, "masks": mask_map})

    stats = {
        "num_samples": len(records),
        "num_mcqs": len(public_records),
        "total_mask_frames": total_mask_frames,
        "shifted_samples": shifted_samples,
    }
    return records, stats


def dump_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".gz":
        with output_path.open("wb") as raw_f:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_f, compresslevel=9, mtime=0) as gz_f:
                gz_f.write(data)
    else:
        output_path.write_bytes(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcqs", required=True, type=Path, help="Public mcqs.jsonl")
    parser.add_argument("--raw", required=True, type=Path, help="Internal raw_merged.json with target masks")
    parser.add_argument("--data-root", required=True, type=Path, help="Public release root containing media paths")
    parser.add_argument("--output", required=True, type=Path, help="Output target_masks.jsonl or .jsonl.gz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, stats = build_records(args.mcqs, args.raw, args.data_root)
    dump_jsonl(records, args.output)
    print(
        f"wrote {args.output} with {stats['num_samples']} samples, "
        f"{stats['num_mcqs']} MCQs, {stats['total_mask_frames']} mask frames, "
        f"{stats['shifted_samples']} shifted samples"
    )


if __name__ == "__main__":
    main()
