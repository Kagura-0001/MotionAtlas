from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from .media import FrameReader, uniform_sample_indices


@dataclass
class RenderedFrame:
    frame_index: int
    image: Image.Image
    highlighted: bool = False


@dataclass
class RenderedSample:
    frames: list[RenderedFrame]
    metadata: dict[str, Any]


def parse_mask_map(masks: dict[str, Any]) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for key, value in masks.items():
        if str(key).isdigit() and value:
            out[int(key)] = value
    return dict(sorted(out.items()))


def decode_rle_mask(rle_data: Any) -> Optional[np.ndarray]:
    try:
        from pycocotools import mask as mask_util
    except ImportError as exc:
        raise RuntimeError("pycocotools is required to render target masks") from exc

    try:
        if isinstance(rle_data, list):
            merged = None
            for item in rle_data:
                decoded = decode_rle_mask(item)
                if decoded is None:
                    continue
                merged = decoded if merged is None else np.maximum(merged, decoded)
            return merged
        if isinstance(rle_data, dict) and "size" in rle_data and "counts" in rle_data:
            rle_obj = rle_data
            if isinstance(rle_obj.get("counts"), list):
                height, width = rle_obj["size"]
                rle_obj = mask_util.frPyObjects(rle_obj, height, width)
            decoded = mask_util.decode(rle_obj)
            if decoded is None:
                return None
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            return decoded.astype(np.uint8)
    except Exception:
        return None
    return None


def draw_mask_contour(
    frame_bgr: np.ndarray,
    rle_data: Any,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    mask = decode_rle_mask(rle_data)
    if mask is None:
        return frame_bgr.copy()
    if mask.shape[:2] != frame_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = mask.astype(np.uint8)
    if mask.size > 0 and mask.max() == 1:
        mask = mask * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame_bgr.copy()
    cv2.drawContours(output, contours, -1, color_bgr, thickness)
    return output


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).convert("RGB")


def read_rendered_frame(
    reader: FrameReader,
    frame_index: int,
    mask_rle: Any = None,
    highlighted: bool = False,
    outline_color: tuple[int, int, int] = (0, 255, 0),
    outline_thickness: int = 3,
) -> RenderedFrame:
    frame = reader.read_frame(frame_index)
    if frame is None:
        raise RuntimeError(f"failed to read frame index {frame_index} from {reader.media_path}")
    if highlighted and mask_rle is not None:
        frame = draw_mask_contour(frame, mask_rle, color_bgr=outline_color, thickness=outline_thickness)
    return RenderedFrame(frame_index=int(frame_index), image=bgr_to_pil(frame), highlighted=highlighted)


def insert_highlight_frame(
    sampled_frames: list[RenderedFrame],
    highlight_frame: Optional[RenderedFrame],
) -> list[RenderedFrame]:
    if highlight_frame is None:
        return sampled_frames
    output: list[RenderedFrame] = []
    inserted = False
    for frame in sampled_frames:
        if not inserted and highlight_frame.frame_index <= frame.frame_index:
            output.append(highlight_frame)
            inserted = True
            if highlight_frame.frame_index == frame.frame_index:
                continue
        output.append(frame)
    if not inserted:
        output.append(highlight_frame)
    return output


def include_frame_in_sample_budget(sampled_indices: list[int], required_index: int) -> list[int]:
    if not sampled_indices:
        return [int(required_index)]
    if required_index in sampled_indices:
        return sampled_indices
    replace_pos = min(
        range(len(sampled_indices)),
        key=lambda i: (abs(sampled_indices[i] - required_index), i),
    )
    output = list(sampled_indices)
    output[replace_pos] = int(required_index)
    return sorted(output)


def render_sample(
    data_root: Path,
    sample_record: dict[str, Any],
    masks: dict[str, Any],
    setting: str,
    num_frames: int,
    overlay_first_within_frame_budget: bool = False,
    outline_color: tuple[int, int, int] = (0, 255, 0),
    outline_thickness: int = 3,
) -> RenderedSample:
    if setting not in {"first_mask", "overlay_all"}:
        raise ValueError(f"unsupported setting: {setting}")

    media_path = data_root / sample_record["video_path"]
    mask_map = parse_mask_map(masks)
    if not mask_map:
        raise ValueError(f"sample_id={sample_record['sample_id']} has no target masks")

    with FrameReader(media_path) as reader:
        base_indices = uniform_sample_indices(reader.total_frames, num_frames)
        if not base_indices:
            raise ValueError(f"sample_id={sample_record['sample_id']} has no readable frames")

        frames: list[RenderedFrame] = []
        highlighted_indices: list[int] = []
        timeline_indices = list(base_indices)

        if setting == "first_mask":
            first_mask_index = min(mask_map)
            if first_mask_index < 0 or first_mask_index >= reader.total_frames:
                raise ValueError(
                    f"sample_id={sample_record['sample_id']} first mask frame {first_mask_index} "
                    f"outside total_frames={reader.total_frames}"
                )
            if overlay_first_within_frame_budget:
                timeline_indices = include_frame_in_sample_budget(timeline_indices, first_mask_index)
            highlight_frame = read_rendered_frame(
                reader,
                first_mask_index,
                mask_map[first_mask_index],
                highlighted=True,
                outline_color=outline_color,
                outline_thickness=outline_thickness,
            )
            highlighted_indices.append(first_mask_index)

            sampled_frames: list[RenderedFrame] = []
            for frame_index in timeline_indices:
                if overlay_first_within_frame_budget and frame_index == first_mask_index:
                    sampled_frames.append(highlight_frame)
                else:
                    sampled_frames.append(read_rendered_frame(reader, frame_index))
            frames = sampled_frames if overlay_first_within_frame_budget else insert_highlight_frame(sampled_frames, highlight_frame)

        else:
            for frame_index in timeline_indices:
                if frame_index in mask_map:
                    frames.append(
                        read_rendered_frame(
                            reader,
                            frame_index,
                            mask_map[frame_index],
                            highlighted=True,
                            outline_color=outline_color,
                            outline_thickness=outline_thickness,
                        )
                    )
                    highlighted_indices.append(frame_index)
                else:
                    frames.append(read_rendered_frame(reader, frame_index))

        metadata = {
            "sample_id": int(sample_record["sample_id"]),
            "video_path": sample_record["video_path"],
            "video_type": sample_record["video_type"],
            "setting": setting,
            "num_frames_requested": int(num_frames),
            "output_frame_count": len(frames),
            "total_frames": int(reader.total_frames),
            "fps": reader.fps,
            "base_sampled_frame_indices": [int(idx) for idx in base_indices],
            "timeline_sampled_frame_indices": [int(idx) for idx in timeline_indices],
            "sampled_frame_indices": [int(frame.frame_index) for frame in frames],
            "highlighted_frame_indices": [int(idx) for idx in highlighted_indices],
            "overlay_first_within_frame_budget": bool(overlay_first_within_frame_budget),
        }
        return RenderedSample(frames=frames, metadata=metadata)
