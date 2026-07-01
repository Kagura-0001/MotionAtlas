from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def uniform_indices(total: int, target: int) -> list[int]:
    if total <= 0:
        return []
    if target <= 0 or total <= target:
        return list(range(total))
    return np.linspace(0, total - 1, target, dtype=int).tolist()


def list_frame_dir(path: Path) -> list[Path]:
    return sorted(
        [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda item: natural_key(item.name),
    )


def media_frame_count(path: str | Path) -> int:
    path = Path(path)
    if path.is_dir():
        return len(list_frame_dir(path))
    if path.suffix.lower() == ".gif":
        with Image.open(path) as img:
            return int(getattr(img, "n_frames", 1))
    try:
        from decord import VideoReader, cpu

        return int(len(VideoReader(str(path), ctx=cpu(0))))
    except Exception:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total


def _read_frame_dir(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    files = list_frame_dir(path)
    frames = {}
    for idx in indices:
        if 0 <= idx < len(files):
            with Image.open(files[idx]) as img:
                frames[idx] = np.asarray(img.convert("RGB"))
    return frames


def _read_gif(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    with Image.open(path) as img:
        all_frames = [np.asarray(frame.convert("RGB")) for frame in ImageSequence.Iterator(img)]
    return {idx: all_frames[idx] for idx in indices if 0 <= idx < len(all_frames)}


def _read_video_decord(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0))
    valid = [idx for idx in dict.fromkeys(indices) if 0 <= idx < len(vr)]
    if not valid:
        return {}
    return {idx: frame.astype(np.uint8) for idx, frame in zip(valid, vr.get_batch(valid).asnumpy())}


def _read_video_cv2(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = {idx for idx in indices if idx >= 0}
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(frames) == len(wanted):
                break
        frame_idx += 1
    cap.release()
    return frames


def read_media_frames(path: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    path = Path(path)
    if path.is_dir():
        return _read_frame_dir(path, indices)
    if path.suffix.lower() == ".gif":
        return _read_gif(path, indices)
    try:
        return _read_video_decord(path, indices)
    except Exception:
        return _read_video_cv2(path, indices)


def sampled_indices_with_annotation(total: int, max_frames: int, annotation_frame: int | None) -> list[int]:
    sampled = uniform_indices(total, max_frames)
    if annotation_frame is None or annotation_frame < 0 or annotation_frame >= total:
        return sampled
    merged = sorted(set(sampled + [annotation_frame]))
    if len(merged) <= max_frames + 1:
        return merged
    return merged
