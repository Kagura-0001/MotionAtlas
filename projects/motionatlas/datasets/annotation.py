from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def decode_mask(rle: dict[str, Any], height: int | None = None, width: int | None = None) -> np.ndarray:
    mask = mask_utils.decode(rle).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if height is not None and width is not None and mask.shape[:2] != (height, width):
        pil = Image.fromarray(mask * 255)
        pil = pil.resize((width, height), Image.NEAREST)
        mask = (np.asarray(pil) > 127).astype(np.uint8)
    return mask


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]


def normalize_bbox(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    out = [
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
        int(round(x2 / width * 1000)),
        int(round(y2 / height * 1000)),
    ]
    out = [min(max(v, 0), 1000) for v in out]
    if out[2] <= out[0]:
        out[2] = min(out[0] + 1, 1000)
    if out[3] <= out[1]:
        out[3] = min(out[1] + 1, 1000)
    return out


def first_valid_annotation_frame(annotation: Any) -> dict[str, Any] | None:
    if not isinstance(annotation, dict):
        return None
    frames = annotation.get("frames")
    if not isinstance(frames, list):
        return None
    normalized = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            frame_idx = int(frame.get("frame_idx"))
        except Exception:
            continue
        if isinstance(frame.get("mask"), dict) or isinstance(frame.get("bbox"), list):
            item = dict(frame)
            item["frame_idx"] = frame_idx
            normalized.append(item)
    if not normalized:
        return None
    return sorted(normalized, key=lambda item: int(item["frame_idx"]))[0]


def draw_highlight(frame_rgb: np.ndarray, ann_frame: dict[str, Any], color: tuple[int, int, int], thickness: int) -> np.ndarray | None:
    canvas = frame_rgb.copy()
    h, w = canvas.shape[:2]
    if isinstance(ann_frame.get("mask"), dict):
        mask = decode_mask(ann_frame["mask"], height=h, width=w)
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.max() == 1:
            mask_u8 *= 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cv2.drawContours(canvas, contours, -1, color, thickness)
        return canvas
    if isinstance(ann_frame.get("bbox"), list):
        bbox = ann_frame["bbox"]
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        return canvas
    return None


def pixel_bbox_for_frame(frame_rgb: np.ndarray, ann_frame: dict[str, Any]) -> list[float] | None:
    h, w = frame_rgb.shape[:2]
    if isinstance(ann_frame.get("mask"), dict):
        mask = decode_mask(ann_frame["mask"], height=h, width=w)
        return bbox_from_mask(mask)
    if isinstance(ann_frame.get("bbox"), list):
        return [float(v) for v in ann_frame["bbox"]]
    return None


def text_bbox_prompt(qwen_bbox: list[int], frame_index: int) -> str:
    payload = json.dumps(
        {"frame_index": int(frame_index), "bbox_2d": [int(v) for v in qwen_bbox]},
        separators=(",", ":"),
    )
    return (
        f"The target object is specified by this region in the provided frame sequence: {payload}. "
        "The bbox uses normalized 0-1000 coordinates in [x1,y1,x2,y2] order. "
        "Describe the target object's appearance and actions throughout the video."
    )

