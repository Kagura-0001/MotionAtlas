from __future__ import annotations

import copy
import random
from io import BytesIO
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
FALLBACK_ASSISTANT_START_IDS = [151644, 77091, 198]
FALLBACK_IM_END_IDS = [151645]


def _encode_template_ids(processor: Any, text: str, fallback: list[int]) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return fallback
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids) if ids else fallback


def _find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    end = len(values) - len(pattern) + 1
    for pos in range(start, max(start, end)):
        if values[pos : pos + len(pattern)] == pattern:
            return pos
    return -1


def _assistant_labels(input_ids: torch.Tensor, processor: Any) -> torch.Tensor | None:
    ids = input_ids[0].tolist()
    assistant_start_ids = _encode_template_ids(
        processor,
        "<|im_start|>assistant\n",
        FALLBACK_ASSISTANT_START_IDS,
    )
    im_end_ids = _encode_template_ids(processor, "<|im_end|>", FALLBACK_IM_END_IDS)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    pos = 0
    found = False
    while pos < len(ids):
        start = _find_subsequence(ids, assistant_start_ids, pos)
        if start < 0:
            break
        answer_start = start + len(assistant_start_ids)
        answer_end = _find_subsequence(ids, im_end_ids, answer_start)
        if answer_end < 0:
            return None
        label_end = answer_end + len(im_end_ids)
        labels[0, answer_start:label_end] = input_ids[0, answer_start:label_end]
        found = True
        pos = label_end
    return labels if found else None


def preprocess_qwen_visual(messages: list[dict[str, Any]], processor: Any) -> dict[str, Any] | None:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    images, videos = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")

    input_ids = inputs["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = _assistant_labels(input_ids, processor)
    if labels is None:
        return None

    inputs["input_ids"] = input_ids
    inputs["labels"] = labels
    return inputs


class MotionAtlasQwen3VLDataset(Dataset):
    """Qwen3-VL SFT dataset for public MotionAtlas region and recipe records."""

    def __init__(
        self,
        model_path: str,
        hf_dataset: str = "maxLWSv2/motionatlas-data",
        data_root: str = "data/motionatlas-data",
        split: str = "train",
        datasets: list[dict[str, Any] | str] | None = None,
        data_paths: list[dict[str, Any] | str] | str | None = None,
        source_roots: dict[str, str] | None = None,
        source_sample_limits: dict[str, int] | None = None,
        source_sample_seed: int = 42,
        recipe_media_root: str = "",
        image_source_dir: str = "",
        video_source_dir: str = "",
        skip_video: bool = False,
        strict_single_modality: bool = True,
        max_frames: int = 16,
        per_frame_tokens: int = 256,
        max_seq_length: int = 16384,
        max_samples: int = 0,
        annotation_routing: bool = True,
        annotation_drop_if_invalid: bool = True,
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
        self.source_sample_limits = self._normalize_source_sample_limits(source_sample_limits)
        self.source_sample_seed = int(source_sample_seed)
        self.recipe_media_root = str(recipe_media_root or "")
        self.image_source_dir = str(image_source_dir or "")
        self.video_source_dir = str(video_source_dir or "")
        self.skip_video = bool(skip_video)
        self.strict_single_modality = bool(strict_single_modality)
        self.max_frames = int(max_frames)
        self.per_frame_tokens = int(per_frame_tokens)
        self.max_seq_length = int(max_seq_length)
        self.max_samples = int(max_samples or 0)
        self.annotation_routing = bool(annotation_routing)
        self.annotation_drop_if_invalid = bool(annotation_drop_if_invalid)
        self.annotation_mode = str(annotation_mode)
        if self.annotation_mode not in {"highlight", "text_bbox"}:
            raise ValueError("annotation_mode must be one of: highlight, text_bbox")
        self.annotation_prompt = str(annotation_prompt)
        self.annotation_contour_color = tuple(int(v) for v in annotation_contour_color)
        self.annotation_contour_thickness = int(annotation_contour_thickness)
        self._max_refetch = int(max_refetch)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        entries = self._normalize_dataset_entries(datasets=datasets, data_paths=data_paths)
        self.sources = self._build_sources(entries)
        self.sample_pairs = self._build_sample_pairs()
        if self.max_samples > 0:
            self.sample_pairs = self.sample_pairs[: self.max_samples]
        if len(self.sample_pairs) == 0:
            raise ValueError("MotionAtlas training dataset is empty")
        print(
            "MotionAtlasQwen3VLDataset loaded",
            f"sources={len(self.sources)}",
            f"items={len(self.sample_pairs)}",
            flush=True,
        )

    @property
    def modality_length(self) -> list[int]:
        return [100] * len(self)

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def _normalize_source_sample_limits(self, source_sample_limits: dict[str, int] | None) -> dict[str, int]:
        if not source_sample_limits:
            return {}
        return {str(key).rstrip("/"): int(value) for key, value in source_sample_limits.items()}

    def _normalize_dataset_entries(
        self,
        datasets: list[dict[str, Any] | str] | None,
        data_paths: list[dict[str, Any] | str] | str | None,
    ) -> list[dict[str, Any]]:
        raw_entries: list[dict[str, Any] | str]
        if datasets:
            raw_entries = list(datasets)
        elif data_paths:
            raw_entries = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        else:
            raw_entries = [
                {
                    "name": "motionatlas-region",
                    "kind": "region",
                    "path": str(self.data_root),
                    "hf_dataset": self.hf_dataset,
                    "split": self.split,
                }
            ]

        entries = []
        for idx, item in enumerate(raw_entries):
            if isinstance(item, str):
                entry = {"name": Path(item).stem or f"source_{idx}", "path": item}
            elif isinstance(item, dict):
                entry = copy.deepcopy(item)
            else:
                raise TypeError(f"Unsupported dataset entry: {item!r}")
            entry.setdefault("name", entry.get("path") or entry.get("hf_dataset") or f"source_{idx}")
            entry.setdefault("kind", "auto")
            entry.setdefault("split", self.split)
            entry.setdefault("source_roots", self.source_roots)
            entry.setdefault("media_root", self.recipe_media_root if entry.get("kind") == "recipe" else "")
            entries.append(entry)
        return entries

    def _build_sources(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        for entry in entries:
            for source in self._build_sources_for_entry(entry):
                if source["length"] > 0:
                    sources.append(source)
                    print(
                        f"=> Loaded {source['name']} ({source['kind']}) with {source['length']} items.",
                        flush=True,
                    )
        if not sources:
            raise ValueError("No MotionAtlas data source loaded")
        return sources

    def _build_sources_for_entry(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        path_value = entry.get("path") or entry.get("local_dir")
        split = str(entry.get("split") or self.split)
        if path_value:
            path = Path(str(path_value))
            if path.exists():
                return self._load_path_sources(path, entry, split)
        hf_dataset = entry.get("hf_dataset")
        if hf_dataset:
            ds = load_dataset(str(hf_dataset), split=split)
            return [self._source_from_dataset(entry, ds, path=str(hf_dataset))]
        if path_value:
            raise FileNotFoundError(f"Dataset path does not exist: {path_value}")
        raise ValueError(f"Dataset entry must define path/local_dir or hf_dataset: {entry}")

    def _load_path_sources(self, path: Path, entry: dict[str, Any], split: str) -> list[dict[str, Any]]:
        if path.is_file():
            return [self._source_from_dataset(entry, self._load_file(path), path=str(path))]
        if (path / "dataset_info.json").exists() or (path / "state.json").exists():
            ds = load_from_disk(str(path))
            if isinstance(ds, dict):
                ds = ds[split]
            return [self._source_from_dataset(entry, ds, path=str(path))]

        split_candidates = [
            path / f"{split}.jsonl",
            path / f"{split}.json",
            path / f"{split}.parquet",
        ]
        for candidate in split_candidates:
            if candidate.exists():
                return [self._source_from_dataset(entry, self._load_file(candidate), path=str(candidate))]

        parquet_files = sorted(path.rglob("*.parquet"))
        jsonl_files = sorted(path.rglob("*.jsonl"))
        json_files = sorted(path.rglob("*.json"))
        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")
            return [self._source_from_dataset(entry, ds, path=str(path))]
        if jsonl_files:
            ds = load_dataset("json", data_files=[str(p) for p in jsonl_files], split="train")
            return [self._source_from_dataset(entry, ds, path=str(path))]
        if json_files:
            ds = load_dataset("json", data_files=[str(p) for p in json_files], split="train")
            return [self._source_from_dataset(entry, ds, path=str(path))]
        raise FileNotFoundError(f"No supported dataset files found under {path}")

    def _source_from_dataset(self, entry: dict[str, Any], dataset: HFDataset, path: str) -> dict[str, Any]:
        source_roots = self.source_roots.copy()
        source_roots.update({str(k): str(v) for k, v in (entry.get("source_roots") or {}).items() if str(v)})
        return {
            "name": str(entry.get("name") or path),
            "kind": str(entry.get("kind") or "auto"),
            "path": path,
            "data": dataset,
            "length": len(dataset),
            "source_roots": source_roots,
            "media_root": str(entry.get("media_root") or ""),
            "image_source_dir": str(entry.get("image_source_dir") or self.image_source_dir or ""),
            "video_source_dir": str(entry.get("video_source_dir") or self.video_source_dir or ""),
            "skip_video": bool(entry.get("skip_video", self.skip_video)),
            "max_samples": int(entry.get("max_samples", 0) or 0),
        }

    def _build_sample_pairs(self) -> list[tuple[int, int]]:
        selected: list[tuple[int, int]] = []
        for src_idx, source in enumerate(self.sources):
            pairs = [(src_idx, local_idx) for local_idx in range(source["length"])]
            limit_key = self._match_source_limit(source)
            limit = self.source_sample_limits.get(limit_key) if limit_key else None
            if limit is None and source["max_samples"] > 0:
                limit = source["max_samples"]
            if limit is not None and limit < len(pairs):
                rng = random.Random(self.source_sample_seed + src_idx)
                picked = sorted(rng.sample(range(len(pairs)), int(limit)))
                pairs = [pairs[i] for i in picked]
            selected.extend(pairs)
        if len(self.sources) > 1:
            random.Random(self.source_sample_seed).shuffle(selected)
        return selected

    def _match_source_limit(self, source: dict[str, Any]) -> str | None:
        if not self.source_sample_limits:
            return None
        candidates = [str(source.get("name") or ""), str(source.get("path") or "").rstrip("/")]
        for key in sorted(self.source_sample_limits, key=len, reverse=True):
            for candidate in candidates:
                if candidate == key or candidate.startswith(key + "/"):
                    return key
        return None

    def _get_sample_and_source(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        src_idx, local_idx = self.sample_pairs[index]
        source = self.sources[src_idx]
        return copy.deepcopy(source["data"][local_idx]), source

    def _load_file(self, path: Path) -> HFDataset:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset file: {path}")

    def _resolve_media_path(self, sample: dict[str, Any], source_cfg: dict[str, Any] | None = None) -> Path | None:
        value = sample.get("video") or sample.get("video_path")
        if isinstance(value, dict):
            value = value.get("path")
        rel = str(value or "").strip()
        if not rel:
            return None
        path = Path(rel)
        if path.is_absolute() and path.exists():
            return path

        source = str(sample.get("source") or "")
        source_roots = (source_cfg or {}).get("source_roots") or self.source_roots
        root = source_roots.get(source) if source else None
        if root:
            return Path(root) / rel

        media_root = str((source_cfg or {}).get("media_root") or (source_cfg or {}).get("video_source_dir") or "")
        if media_root:
            return Path(media_root) / rel
        return path if path.exists() else None

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

    def _clean_text(self, text: Any, is_vision: bool) -> str:
        value = str(text or "")
        if is_vision:
            value = value.replace("<image>", "").replace("<video>", "")
        return value.replace("<region>", "").strip()

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
        for q_key, a_key in [
            ("question", "answer"),
            ("prompt", "response"),
            ("instruction", "output"),
            ("query", "response"),
        ]:
            question = sample.get(q_key)
            answer = sample.get(a_key)
            if isinstance(question, str) and question.strip():
                convs = [{"role": "user", "text": question.strip()}]
                if isinstance(answer, str) and answer.strip():
                    convs.append({"role": "assistant", "text": answer.strip()})
                return convs
        caption = sample.get("caption")
        if isinstance(caption, str) and caption.strip():
            return [{"role": "user", "text": self.annotation_prompt}, {"role": "assistant", "text": caption.strip()}]
        text = sample.get("text")
        if isinstance(text, str) and text.strip():
            return [{"role": "user", "text": text.strip()}]
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

    def _sample_has_annotation(self, sample: dict[str, Any]) -> bool:
        annotation = sample.get("annotation")
        if isinstance(annotation, list):
            return len(annotation) > 0
        if isinstance(annotation, dict):
            return len(annotation) > 0
        return False

    def _sample_has_video(self, sample: dict[str, Any]) -> bool:
        value = sample.get("video") or sample.get("video_path")
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return bool(str(value.get("path") or "").strip())
        return True

    def _sample_has_image(self, sample: dict[str, Any]) -> bool:
        value = sample.get("image") or sample.get("image_path")
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value) > 0
        if isinstance(value, dict):
            raw_bytes = value.get("bytes")
            raw_path = value.get("path")
            return (raw_bytes is not None and len(raw_bytes) > 0) or bool(str(raw_path or "").strip())
        if isinstance(value, list):
            return len(value) > 0
        return True

    def _sample_has_text(self, sample: dict[str, Any]) -> bool:
        raw = sample.get("messages") or sample.get("conversations")
        if isinstance(raw, list) and raw:
            return True
        for key in ("text", "question", "prompt", "instruction", "query", "caption"):
            if isinstance(sample.get(key), str) and sample[key].strip():
                return True
        return False

    def _detect_modality(self, sample: dict[str, Any]) -> str | None:
        explicit = str(sample.get("modality") or "").strip().lower()
        has_video = self._sample_has_video(sample)
        has_image = self._sample_has_image(sample)
        has_text = self._sample_has_text(sample)

        if explicit in {"video", "image", "text"}:
            return explicit
        if self.strict_single_modality and has_video and has_image:
            return None
        if has_video:
            return "video"
        if has_image:
            return "image"
        if has_text:
            return "text"
        return None

    def _prepare_visual_sample(
        self,
        sample: dict[str, Any],
        source_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        media_path = self._resolve_media_path(sample, source_cfg)
        ann_frame = first_valid_annotation_frame(sample.get("annotation"))
        if media_path is None or ann_frame is None or not media_path.exists():
            return None
        total = media_frame_count(media_path)
        if total <= 0:
            return None
        ann_fid = int(ann_frame["frame_idx"])
        frame_indices = sampled_indices_with_annotation(total, self.max_frames, ann_fid)
        fid_to_frame = read_media_frames(media_path, frame_indices)
        if not fid_to_frame:
            return None
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

    def _resolve_image_path(self, value: str, source_cfg: dict[str, Any]) -> Path | None:
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        for root in [
            source_cfg.get("image_source_dir"),
            source_cfg.get("media_root"),
            self.image_source_dir,
        ]:
            if root:
                candidate = Path(str(root)) / value
                if candidate.exists():
                    return candidate
        return path if path.exists() else None

    def _decode_image_payload(self, payload: Any, source_cfg: dict[str, Any]) -> Image.Image | None:
        if payload is None:
            return None
        if isinstance(payload, Image.Image):
            return payload.convert("RGB")
        if isinstance(payload, str):
            image_path = self._resolve_image_path(payload.strip(), source_cfg)
            if image_path is None:
                return None
            return Image.open(image_path).convert("RGB")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            data = bytes(payload)
            if not data:
                return None
            return Image.open(BytesIO(data)).convert("RGB")
        if isinstance(payload, dict):
            raw_bytes = payload.get("bytes")
            if raw_bytes is not None:
                try:
                    return self._decode_image_payload(raw_bytes, source_cfg)
                except Exception:
                    pass
            raw_path = payload.get("path")
            if raw_path:
                return self._decode_image_payload(str(raw_path), source_cfg)
        return None

    def _load_image_sample(self, sample: dict[str, Any], source_cfg: dict[str, Any]) -> list[Image.Image] | None:
        value = sample.get("image") or sample.get("image_path")
        items = value if isinstance(value, list) else [value]
        images = []
        for item in items:
            image = self._decode_image_payload(item, source_cfg)
            if image is not None:
                images.append(image)
        return images or None

    def _load_video_sample(self, sample: dict[str, Any], source_cfg: dict[str, Any]) -> list[Image.Image] | None:
        media_path = self._resolve_media_path(sample, source_cfg)
        if media_path is None or not media_path.exists():
            return None
        total = media_frame_count(media_path)
        if total <= 0:
            return None
        frame_indices = uniform_indices(total, self.max_frames)
        fid_to_frame = read_media_frames(media_path, frame_indices)
        frames = [
            Image.fromarray(fid_to_frame[fid]).convert("RGB")
            for fid in frame_indices
            if fid in fid_to_frame
        ]
        return frames or None

    def _build_general_messages(
        self,
        conversations: list[dict[str, str]],
        modality: str,
        images: list[Image.Image] | None = None,
    ) -> list[dict[str, Any]]:
        is_vision = modality in {"image", "video"}
        messages = []
        first_user_done = False
        for turn in conversations:
            role = turn["role"]
            text = self._clean_text(turn["text"], is_vision=is_vision)
            if is_vision and role == "user" and not first_user_done:
                content = [
                    {"type": "image", "image": image, "max_pixels": self.per_frame_tokens * 32 * 32}
                    for image in (images or [])
                ]
                content.append({"type": "text", "text": text})
                messages.append({"role": role, "content": content})
                first_user_done = True
            else:
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})

        if is_vision and not first_user_done:
            prompt = "Describe the video in detail." if modality == "video" else "Describe the image in detail."
            content = [
                {"type": "image", "image": image, "max_pixels": self.per_frame_tokens * 32 * 32}
                for image in (images or [])
            ]
            content.append({"type": "text", "text": prompt})
            messages.insert(0, {"role": "user", "content": content})
        return messages

    def _prepare_general_sample(self, sample: dict[str, Any], source_cfg: dict[str, Any]) -> dict[str, Any] | None:
        modality = self._detect_modality(sample)
        if modality is None:
            return None

        conversations = self._normalize_conversations(sample)
        if not conversations:
            return None

        visuals = None
        if modality == "video":
            if source_cfg.get("skip_video"):
                return None
            visuals = self._load_video_sample(sample, source_cfg)
            if not visuals:
                return None
        elif modality == "image":
            visuals = self._load_image_sample(sample, source_cfg)
            if not visuals:
                return None

        messages = self._build_general_messages(conversations, modality=modality, images=visuals)
        return self._prepare_with_processor(messages)

    def _prepare_sample(self, sample: dict[str, Any], source_cfg: dict[str, Any]) -> dict[str, Any] | None:
        if self.annotation_routing and self._sample_has_annotation(sample):
            result = self._prepare_visual_sample(sample, source_cfg)
            if result is not None:
                return result
            if self.annotation_drop_if_invalid:
                return None
        return self._prepare_general_sample(sample, source_cfg)

    def _prepare_with_processor(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        try:
            out = preprocess_qwen_visual(messages, self.processor)
        except Exception as exc:
            print(f"[MotionAtlas] tokenization failed: {exc}", flush=True)
            return None
        if out is None:
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
        base = index % len(self)
        for offset in range(self._max_refetch + 1):
            sample, source = self._get_sample_and_source((base + offset) % len(self))
            result = self._prepare_sample(sample, source)
            if result is not None:
                return result
        raise RuntimeError(f"Failed to load a valid MotionAtlas sample near index {index}")
