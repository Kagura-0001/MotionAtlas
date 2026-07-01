from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .annotation import (
    draw_highlight,
    first_valid_annotation_frame,
    normalize_bbox,
    pixel_bbox_for_frame,
    text_bbox_prompt,
)
from .rope2d import get_rope_index_3
from .video_io import media_frame_count, read_media_frames, sampled_indices_with_annotation, uniform_indices

IGNORE_INDEX = -100


def preprocess_qwen_visual(messages: list[dict[str, Any]], processor: Any) -> dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")

    input_ids = inputs["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    ids = input_ids[0].tolist()
    pos = 0
    while pos < len(ids):
        if ids[pos] == 77091:  # assistant token in Qwen3-VL chat template
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < len(ids) and ids[ans_end] != 151645:
                ans_end += 1
            if ans_end < len(ids):
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    inputs["input_ids"] = input_ids
    inputs["labels"] = labels
    return inputs


class MotionAtlasQwen3VLDataset(Dataset):
    """Qwen3-VL SFT dataset for public MotionAtlas-Data records."""

    def __init__(
        self,
        model_path: str,
        hf_dataset: str = "maxLWSv2/motionatlas-data",
        data_root: str = "data/motionatlas-data",
        split: str = "train",
        source_roots: dict[str, str] | None = None,
        max_frames: int = 16,
        per_frame_tokens: int = 256,
        max_seq_length: int = 16384,
        max_samples: int = 0,
        annotation_mode: str = "highlight",
        annotation_prompt: str = "Describe the highlighted object in detail.",
        annotation_contour_color: list[int] | tuple[int, int, int] = (0, 255, 0),
        annotation_contour_thickness: int = 2,
        max_refetch: int = 100,
        **_: Any,
    ):
        self.model_path = model_path
        self.hf_dataset = hf_dataset
        self.data_root = Path(data_root)
        self.split = split
        self.source_roots = {str(k): str(v) for k, v in (source_roots or {}).items() if str(v)}
        self.max_frames = int(max_frames)
        self.per_frame_tokens = int(per_frame_tokens)
        self.max_seq_length = int(max_seq_length)
        self.max_samples = int(max_samples or 0)
        self.annotation_mode = str(annotation_mode)
        if self.annotation_mode not in {"highlight", "text_bbox"}:
            raise ValueError("annotation_mode must be one of: highlight, text_bbox")
        self.annotation_prompt = str(annotation_prompt)
        self.annotation_contour_color = tuple(int(v) for v in annotation_contour_color)
        self.annotation_contour_thickness = int(annotation_contour_thickness)
        self._max_refetch = int(max_refetch)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        self.data = self._load_data()
        if self.max_samples > 0:
            self.data = self.data.select(range(min(self.max_samples, len(self.data))))
        if len(self.data) == 0:
            raise ValueError("MotionAtlas training dataset is empty")
        print(f"MotionAtlasQwen3VLDataset loaded {len(self.data)} records from {self.data_root or self.hf_dataset}")

    @property
    def modality_length(self) -> list[int]:
        return [100] * len(self)

    def __len__(self) -> int:
        return len(self.data)

    def _load_data(self) -> HFDataset:
        if self.data_root.exists():
            if self.data_root.is_file():
                return self._load_file(self.data_root)
            if (self.data_root / "dataset_info.json").exists() or (self.data_root / "state.json").exists():
                ds = load_from_disk(str(self.data_root))
                if isinstance(ds, dict):
                    return ds[self.split]
                return ds
            split_file_candidates = [
                self.data_root / f"{self.split}.jsonl",
                self.data_root / f"{self.split}.json",
                self.data_root / f"{self.split}.parquet",
            ]
            for candidate in split_file_candidates:
                if candidate.exists():
                    return self._load_file(candidate)
            parquet_files = sorted(self.data_root.glob("*.parquet"))
            jsonl_files = sorted(self.data_root.glob("*.jsonl"))
            if parquet_files:
                return load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
            if jsonl_files:
                return load_dataset("json", data_files=[str(p) for p in jsonl_files], split="train")
            raise FileNotFoundError(f"No supported dataset files found under {self.data_root}")
        return load_dataset(self.hf_dataset, split=self.split)

    def _load_file(self, path: Path) -> HFDataset:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset file: {path}")

    def _resolve_media_path(self, sample: dict[str, Any]) -> Path | None:
        source = str(sample.get("source") or "")
        rel = str(sample.get("video") or "")
        root = self.source_roots.get(source)
        if not source or not rel or not root:
            return None
        return Path(root) / rel

    def _normalize_role(self, role: Any) -> str | None:
        value = str(role or "").lower().strip()
        if value in {"human", "user"}:
            return "user"
        if value in {"gpt", "assistant"}:
            return "assistant"
        if value == "system":
            return "system"
        return None

    def _extract_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            return "\n".join(chunks).strip()
        if isinstance(value, dict):
            return str(value.get("text") or value.get("value") or value.get("content") or "")
        return str(value)

    def _normalize_conversations(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        raw = sample.get("conversations") or sample.get("messages") or []
        if isinstance(raw, dict):
            raw = [raw]
        convs = []
        if isinstance(raw, list):
            for turn in raw:
                if not isinstance(turn, dict):
                    continue
                if "from" in turn or "value" in turn:
                    role = self._normalize_role(turn.get("from", "human"))
                    text = self._extract_text(turn.get("value", ""))
                else:
                    role = self._normalize_role(turn.get("role", "user"))
                    text = self._extract_text(turn.get("content", ""))
                if role and text:
                    convs.append({"role": role, "text": text})
        if convs:
            return convs
        caption = sample.get("caption")
        if isinstance(caption, str) and caption.strip():
            return [{"role": "user", "text": self.annotation_prompt}, {"role": "assistant", "text": caption.strip()}]
        return []

    def _assistant_caption(self, sample: dict[str, Any]) -> str:
        for turn in self._normalize_conversations(sample):
            if turn["role"] == "assistant" and turn["text"].strip():
                return turn["text"].strip()
        return "The target object is visible in the video."

    def _build_messages(self, frames: list[Image.Image], prompt: str, answer: str) -> list[dict[str, Any]]:
        content = [
            {"type": "image", "image": frame, "max_pixels": self.per_frame_tokens * 32 * 32}
            for frame in frames
        ]
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

    def _prepare_visual_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        media_path = self._resolve_media_path(sample)
        ann_frame = first_valid_annotation_frame(sample.get("annotation"))
        if media_path is None or ann_frame is None or not media_path.exists():
            return None
        total = media_frame_count(media_path)
        if total <= 0:
            return None
        ann_fid = int(ann_frame["frame_idx"])
        frame_indices = sampled_indices_with_annotation(total, self.max_frames, ann_fid)
        raw_frames = read_media_frames(media_path, frame_indices)
        if not raw_frames:
            return None
        fid_to_frame = {fid: frame for fid, frame in zip(frame_indices, raw_frames)}
        ann_rgb = fid_to_frame.get(ann_fid)
        if ann_rgb is None:
            return None
        caption = self._assistant_caption(sample)

        final_frames = []
        frame_index = -1
        if self.annotation_mode == "highlight":
            highlighted = draw_highlight(
                ann_rgb,
                ann_frame,
                color=self.annotation_contour_color,
                thickness=self.annotation_contour_thickness,
            )
            if highlighted is None:
                return None
            for fid in frame_indices:
                frame = highlighted if fid == ann_fid else fid_to_frame.get(fid)
                if frame is not None:
                    final_frames.append(Image.fromarray(frame).convert("RGB"))
            prompt = self.annotation_prompt
        else:
            pixel_bbox = pixel_bbox_for_frame(ann_rgb, ann_frame)
            if pixel_bbox is None:
                return None
            h, w = ann_rgb.shape[:2]
            qwen_bbox = normalize_bbox(pixel_bbox, width=w, height=h)
            if qwen_bbox is None:
                return None
            for fid in frame_indices:
                frame = fid_to_frame.get(fid)
                if frame is not None:
                    if fid == ann_fid and frame_index < 0:
                        frame_index = len(final_frames)
                    final_frames.append(Image.fromarray(frame).convert("RGB"))
            if frame_index < 0:
                return None
            prompt = text_bbox_prompt(qwen_bbox, frame_index)

        if not final_frames:
            return None
        return self._prepare_with_processor(self._build_messages(final_frames, prompt, caption))

    def _prepare_with_processor(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        try:
            out = preprocess_qwen_visual(messages, self.processor)
        except Exception as exc:
            print(f"[MotionAtlas] tokenization failed: {exc}", flush=True)
            return None
        seq_len = int(out["input_ids"][0].size(0))
        if self.max_seq_length and seq_len > self.max_seq_length:
            print(f"[MotionAtlas] skip seq_len={seq_len} > max_seq_length={self.max_seq_length}", flush=True)
            return None

        image_grid = out.get("image_grid_thw")
        video_grid = out.get("video_grid_thw")
        image_grid_list = image_grid if isinstance(image_grid, list) else ([image_grid] if image_grid is not None else None)
        video_grid_list = video_grid if isinstance(video_grid, list) else ([video_grid] if video_grid is not None else None)
        second_per_grid_ts = None
        if video_grid_list:
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size / self.processor.video_processor.fps
            ] * len(video_grid_list)

        position_ids, _ = get_rope_index_3(
            self.merge_size,
            out["input_ids"],
            image_grid_thw=torch.cat(image_grid_list, dim=0) if image_grid_list else None,
            video_grid_thw=torch.cat(video_grid_list, dim=0) if video_grid_list else None,
            second_per_grid_ts=second_per_grid_ts,
        )
        out["position_ids"] = position_ids
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        base = index % len(self.data)
        for offset in range(self._max_refetch + 1):
            sample = copy.deepcopy(self.data[(base + offset) % len(self.data)])
            result = self._prepare_visual_sample(sample)
            if result is not None:
                return result
        raise RuntimeError(f"Failed to load a valid MotionAtlas sample near index {index}")

