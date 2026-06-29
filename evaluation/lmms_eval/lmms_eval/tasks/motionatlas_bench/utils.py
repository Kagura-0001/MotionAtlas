import gzip
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import cv2
import datasets
import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

DATASET_REPO = "maxLWSv2/motionatlas-bench"
TARGET_MASKS_FILE = "target_masks.jsonl.gz"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MISS_OPTION_ROLES = {"neg_not_shown", "neg_shown_no_value"}
METRIC_KEYS = (
    "motionatlas_weighted_score",
    "motionatlas_accuracy",
    "motionatlas_recall",
    "motionatlas_precision",
    "motionatlas_answered_rate",
)

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "A single uppercase letter matching the selected option.",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


class TextJudgeClient(Protocol):
    model: str

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        ...


class GeminiOfficialJudgeClient:
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for MotionAtlas Gemini judging. Install lmms_eval[gemini].") from exc

        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not resolved_key:
            raise RuntimeError("MotionAtlas Gemini judge requires GEMINI_API_KEY or GOOGLE_API_KEY.")

        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.client = genai.Client(api_key=resolved_key)

    @staticmethod
    def _extract_text(response: Any) -> str:
        for attr in ("output_text", "text"):
            value = getattr(response, attr, None)
            if value:
                return str(value)
        output = getattr(response, "output", None)
        if output:
            parts: List[str] = []
            for item in output if isinstance(output, list) else [output]:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            if parts:
                return "\n".join(parts)
        return str(response)

    def _generate_content(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("installed google-genai package does not expose google.genai.types") from exc

        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": JUDGE_RESPONSE_SCHEMA,
            "max_output_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        try:
            config = types.GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("response_schema", None)
            config = types.GenerateContentConfig(**config_kwargs)
        return self.client.models.generate_content(model=self.model, contents=prompt, config=config)

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._generate_content(prompt, max_tokens, temperature, top_p)
                return self._extract_text(response)
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError(f"Gemini judge failed after {self.max_retries} attempts: {last_error}")


class JudgeCache:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.lock = threading.Lock()
        self.data: Dict[str, str] = {}
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    self.data = {str(key): str(value) for key, value in loaded.items()}
            except Exception as exc:
                eval_logger.warning("[motionatlas] failed loading judge cache {}: {}", self.path, exc)

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        with self.lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(self.path.parent), delete=False) as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
                tmp_name = handle.name
            os.replace(tmp_name, self.path)


@lru_cache(maxsize=1)
def create_judge_client() -> TextJudgeClient:
    model = os.getenv("MOTIONATLAS_JUDGE_MODEL", "gemini-2.5-pro")
    api_key = os.getenv("MOTIONATLAS_JUDGE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    max_retries = int(os.getenv("MOTIONATLAS_JUDGE_MAX_RETRIES", "3"))
    return GeminiOfficialJudgeClient(model=model, api_key=api_key, max_retries=max_retries)


@lru_cache(maxsize=1)
def _judge_cache() -> Optional[JudgeCache]:
    cache_path = os.getenv("MOTIONATLAS_JUDGE_CACHE", "").strip()
    return JudgeCache(cache_path) if cache_path else None


def _repo_id() -> str:
    return os.getenv("MOTIONATLAS_BENCH_REPO", DATASET_REPO)


def _local_root() -> Optional[Path]:
    root = os.getenv("MOTIONATLAS_BENCH_ROOT", "").strip()
    return Path(root).expanduser() if root else None


def _download_file(filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download MotionAtlas-Bench files") from exc

    return Path(hf_hub_download(repo_id=_repo_id(), repo_type="dataset", filename=filename))


def _resolve_bench_file(filename: str) -> Path:
    local_root = _local_root()
    if local_root is not None:
        path = local_root / filename
        if not path.exists():
            raise FileNotFoundError(f"MotionAtlas-Bench file not found: {path}")
        return path
    try:
        return _download_file(filename)
    except Exception as exc:
        raise RuntimeError(f"failed to resolve {filename!r} from HF dataset {_repo_id()!r}. If using local data, set MOTIONATLAS_BENCH_ROOT.") from exc


def _resolve_media_path(video_path: str, video_type: str) -> Path:
    local_root = _local_root()
    if local_root is not None:
        path = local_root / video_path
        if not path.exists():
            raise FileNotFoundError(f"MotionAtlas-Bench media not found: {path}")
        return path

    if video_type == "images":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to download MotionAtlas-Bench image-frame directories") from exc
        snapshot_root = Path(snapshot_download(repo_id=_repo_id(), repo_type="dataset", allow_patterns=[f"{video_path.rstrip('/')}/*"]))
        path = snapshot_root / video_path
        if not path.exists():
            raise FileNotFoundError(f"MotionAtlas-Bench image directory not found after download: {path}")
        return path
    return _download_file(video_path)


def _jsonl_records(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(record)
    return records


@lru_cache(maxsize=2)
def load_target_masks() -> Dict[int, Dict[str, Any]]:
    path = _resolve_bench_file(TARGET_MASKS_FILE)
    out: Dict[int, Dict[str, Any]] = {}
    for line_no, record in enumerate(_jsonl_records(path), start=1):
        if set(record) != {"sample_id", "masks"}:
            raise ValueError(f"{path}:{line_no}: expected exactly sample_id and masks fields")
        sample_id = int(record["sample_id"])
        masks = record["masks"]
        if not isinstance(masks, dict) or not masks:
            raise ValueError(f"{path}:{line_no}: masks must be a non-empty object")
        out[sample_id] = masks
    return out


def _letter_for_index(index: int) -> str:
    return chr(ord("A") + int(index))


def _normalize_kwargs(lmms_eval_specific_kwargs: Optional[dict]) -> Dict[str, Any]:
    kwargs = dict(lmms_eval_specific_kwargs or {})
    if isinstance(kwargs.get("default"), dict):
        merged = dict(kwargs["default"])
        for key, value in kwargs.items():
            if key != "default":
                merged[key] = value
        return merged
    return kwargs


def motionatlas_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for raw in dataset:
        record = dict(raw)
        record["sample_id"] = int(record["sample_id"])
        record["answer_index"] = int(record["answer_index"])
        grouped[record["sample_id"]].append(record)

    docs: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        mcqs = grouped[sample_id]
        first = mcqs[0]
        for record in mcqs[1:]:
            for key in ("video_path", "video_type"):
                if record.get(key) != first.get(key):
                    raise ValueError(f"sample_id={sample_id} has inconsistent {key}")
        docs.append(
            {
                "id": f"sample_{sample_id}",
                "sample_id": sample_id,
                "video_path": first["video_path"],
                "video_type": first["video_type"],
                "target_entity": first.get("target_entity", {}),
                "mcqs": mcqs,
                "caption_target": "",
            }
        )

    eval_logger.info("[motionatlas] Loaded {} sample-level docs from {} MCQs", len(docs), len(dataset))
    return datasets.Dataset.from_list(docs)


class FrameReader:
    def __init__(self, media_path: Path):
        self.media_path = Path(media_path)
        self.capture: Optional[cv2.VideoCapture] = None
        self.image_paths: List[Path] = []
        self.total_frames = 0
        self.fps: Optional[float] = None

        if self.media_path.is_dir():
            self.image_paths = sorted([path for path in self.media_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda path: path.name)
            self.total_frames = len(self.image_paths)
            return

        if self.media_path.is_file():
            self.capture = cv2.VideoCapture(str(self.media_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"failed to open video: {self.media_path}")
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self.fps = fps if fps > 0 else None
            return

        raise FileNotFoundError(f"media path does not exist: {self.media_path}")

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self.total_frames:
            return None
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.capture.read()
            return frame if ok else None
        return cv2.imread(str(self.image_paths[frame_index]))

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def uniform_sample_indices(total_frames: int, num_frames: int) -> List[int]:
    total_frames = int(total_frames)
    num_frames = int(num_frames)
    if total_frames <= 0 or num_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    if num_frames == 1:
        return [0]
    return [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]


def parse_mask_map(masks: Dict[str, Any]) -> Dict[int, Any]:
    parsed: Dict[int, Any] = {}
    for key, value in masks.items():
        if str(key).isdigit() and value:
            parsed[int(key)] = value
    return dict(sorted(parsed.items()))


def _decode_uncompressed_rle(rle_data: Dict[str, Any]) -> Optional[np.ndarray]:
    size = rle_data.get("size")
    counts = rle_data.get("counts")
    if not isinstance(size, list) or len(size) != 2 or not isinstance(counts, list):
        return None
    height, width = int(size[0]), int(size[1])
    values: List[int] = []
    value = 0
    for count in counts:
        values.extend([value] * int(count))
        value = 1 - value
    total = height * width
    if len(values) < total:
        values.extend([0] * (total - len(values)))
    return np.array(values[:total], dtype=np.uint8).reshape((height, width), order="F")


def decode_rle_mask(rle_data: Any) -> Optional[np.ndarray]:
    if isinstance(rle_data, list):
        merged = None
        for item in rle_data:
            decoded = decode_rle_mask(item)
            if decoded is None:
                continue
            merged = decoded if merged is None else np.maximum(merged, decoded)
        return merged

    if not isinstance(rle_data, dict) or "size" not in rle_data or "counts" not in rle_data:
        return None

    if isinstance(rle_data.get("counts"), list):
        return _decode_uncompressed_rle(rle_data)

    try:
        from pycocotools import mask as mask_util
    except ImportError as exc:
        raise RuntimeError("pycocotools is required to decode compressed COCO RLE masks") from exc

    try:
        decoded = mask_util.decode(rle_data)
        if decoded is None:
            return None
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded.astype(np.uint8)
    except Exception:
        return None


def draw_mask_contour(frame_bgr: np.ndarray, rle_data: Any, thickness: int = 3) -> np.ndarray:
    mask = decode_rle_mask(rle_data)
    if mask is None:
        return frame_bgr.copy()
    if mask.shape[:2] != frame_bgr.shape[:2]:
        mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)
    if mask.size > 0 and mask.max() == 1:
        mask = mask * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame_bgr.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), int(thickness))
    return output


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")


def _read_rendered_frame(reader: FrameReader, frame_index: int, mask_rle: Any = None, highlighted: bool = False, outline_thickness: int = 3) -> Image.Image:
    frame = reader.read_frame(frame_index)
    if frame is None:
        raise RuntimeError(f"failed to read frame index {frame_index} from {reader.media_path}")
    if highlighted and mask_rle is not None:
        frame = draw_mask_contour(frame, mask_rle, thickness=outline_thickness)
    return _bgr_to_pil(frame)


def _insert_highlight_frame(sampled: List[tuple[int, Image.Image]], highlighted: tuple[int, Image.Image]) -> List[tuple[int, Image.Image]]:
    highlighted_index, highlighted_image = highlighted
    output: List[tuple[int, Image.Image]] = []
    inserted = False
    for frame_index, image in sampled:
        if not inserted and highlighted_index <= frame_index:
            output.append((highlighted_index, highlighted_image))
            inserted = True
            if highlighted_index == frame_index:
                continue
        output.append((frame_index, image))
    if not inserted:
        output.append((highlighted_index, highlighted_image))
    return output


def render_motionatlas_sample(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None) -> List[Image.Image]:
    kwargs = _normalize_kwargs(lmms_eval_specific_kwargs)
    setting = kwargs.get("setting", "first_mask")
    if setting != "first_mask":
        raise ValueError(f"unsupported MotionAtlas setting for lmms-eval task: {setting}")
    num_frames = int(kwargs.get("num_frames", 16))
    outline_thickness = int(kwargs.get("outline_thickness", 3))

    sample_id = int(doc["sample_id"])
    masks_by_sample = load_target_masks()
    if sample_id not in masks_by_sample:
        raise ValueError(f"sample_id={sample_id} has no target masks")
    mask_map = parse_mask_map(masks_by_sample[sample_id])
    if not mask_map:
        raise ValueError(f"sample_id={sample_id} has no valid target masks")

    media_path = _resolve_media_path(str(doc["video_path"]), str(doc["video_type"]))
    with FrameReader(media_path) as reader:
        base_indices = uniform_sample_indices(reader.total_frames, num_frames)
        if not base_indices:
            raise ValueError(f"sample_id={sample_id} has no readable frames")

        first_mask_index = min(mask_map)
        if first_mask_index < 0 or first_mask_index >= reader.total_frames:
            raise ValueError(f"sample_id={sample_id} first mask frame {first_mask_index} outside total_frames={reader.total_frames}")

        sampled = [(frame_index, _read_rendered_frame(reader, frame_index)) for frame_index in base_indices]
        highlighted = (first_mask_index, _read_rendered_frame(reader, first_mask_index, mask_map[first_mask_index], highlighted=True, outline_thickness=outline_thickness))
        return [image for _, image in _insert_highlight_frame(sampled, highlighted)]


def build_caption_prompt(setting: str = "first_mask") -> str:
    if setting != "first_mask":
        raise ValueError(f"unsupported MotionAtlas setting for lmms-eval task: {setting}")
    return (
        "You are given sampled video frames in chronological order. "
        "Each image is labeled as Frame 1, Frame 2, and so on in the input sequence. "
        "One frame contains a green contour highlighting the target object. "
        "Use it to identify the target object, then describe that same object throughout the full video.\n"
        "Write a detailed motion caption for the target object. Cover its appearance, pose, actions, motion, state changes, and interactions with nearby objects. "
        "Focus strictly on the target object and directly relevant interacted objects. "
        "Do not answer any multiple-choice question."
    )


def motionatlas_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None):
    return render_motionatlas_sample(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)


def motionatlas_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    kwargs = _normalize_kwargs(lmms_eval_specific_kwargs)
    return build_caption_prompt(str(kwargs.get("setting", "first_mask")))


def motionatlas_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None):
    images = motionatlas_doc_to_visual(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    prompt = motionatlas_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    content: List[Dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        content.append({"type": "text", "text": f"Frame {index}:"})
        content.append({"type": "image", "url": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def format_options(options: List[Any]) -> str:
    return "\n".join(f"{_letter_for_index(i)}. {option}" for i, option in enumerate(options))


def build_judge_prompt(record: Dict[str, Any], caption: str) -> str:
    allowed = ", ".join(_letter_for_index(i) for i in range(len(record["options"])))
    return (
        "Based on the following motion caption, answer the multiple-choice question.\n\n"
        f"Motion Caption:\n{caption}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{format_options(record['options'])}\n\n"
        "Instructions:\n"
        "- Read the motion caption carefully.\n"
        "- Select the option best supported by the caption.\n"
        "- If the caption does not contain enough information, choose the matching not-mentioned or unclear/not-visible option when present.\n\n"
        "Return JSON only with exactly one field in this format:\n"
        '{"answer":"<LETTER>"}\n'
        f"The answer letter must be one of: {allowed}."
    )


def parse_answer_letter(raw_response: str, num_options: int) -> tuple[Optional[str], Optional[int]]:
    if num_options <= 0:
        return None, None
    allowed = [_letter_for_index(i) for i in range(num_options)]
    text = (raw_response or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    candidates = [text]
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        candidates.append(text[first_obj : last_obj + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            value = parsed.get("answer", parsed.get("choice", parsed.get("pred_answer")))
            if value is not None:
                letter = str(value).strip().upper()
                if letter in allowed:
                    return letter, ord(letter) - ord("A")

    letter = extract_mcq_answer(text.upper(), choices=allowed)
    if letter in allowed:
        return letter, ord(letter) - ord("A")
    return None, None


def option_role(option_text: Any) -> str:
    text = " ".join(str(option_text).lower().split())
    if "does not mention" in text or "not mention" in text:
        return "neg_not_shown"
    if "unclear" in text or "not visible" in text or "obscured" in text or "indistinct" in text:
        return "neg_shown_no_value"
    if "value differs" in text or "differs from all listed options" in text:
        return "neg_shown_diff_value"
    return "other"


def option_roles(options: List[Any]) -> List[str]:
    return [option_role(option) for option in options]


def is_correct(record: Dict[str, Any], pred_index: Optional[int]) -> bool:
    return pred_index is not None and int(record["answer_index"]) == int(pred_index)


def classify_judge_answer(record: Dict[str, Any], judge_index: Optional[int], correct: bool) -> str:
    if judge_index is None:
        return "wrong"
    if correct:
        return "correct"
    roles = option_roles(record["options"])
    answer_index = int(record["answer_index"])
    if 0 <= answer_index < len(roles) and roles[answer_index] in MISS_OPTION_ROLES:
        return "wrong"
    if 0 <= judge_index < len(roles) and roles[judge_index] in MISS_OPTION_ROLES:
        return "miss"
    return "wrong"


def _cache_key(record: Dict[str, Any], caption: str, judge_model: str) -> str:
    payload = json.dumps({"id": record.get("id"), "caption": caption, "judge_model": judge_model}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def judge_one_mcq(record: Dict[str, Any], caption: str, client: TextJudgeClient) -> Dict[str, Any]:
    prompt = build_judge_prompt(record, caption)
    cache = _judge_cache()
    key = _cache_key(record, caption, client.model)
    raw_response = cache.get(key) if cache is not None else None
    error = ""
    if raw_response is None:
        try:
            raw_response = client.generate(
                prompt=prompt,
                max_tokens=int(os.getenv("MOTIONATLAS_JUDGE_MAX_TOKENS", "256")),
                temperature=float(os.getenv("MOTIONATLAS_JUDGE_TEMPERATURE", "0")),
                top_p=float(os.getenv("MOTIONATLAS_JUDGE_TOP_P", "1")),
            )
            if cache is not None:
                cache.set(key, raw_response)
        except Exception as exc:
            raw_response = ""
            error = str(exc)

    judge_letter, judge_index = parse_answer_letter(raw_response, len(record["options"])) if raw_response and not error else (None, None)
    correct = is_correct(record, judge_index)
    classification = classify_judge_answer(record, judge_index, correct)
    return {
        "id": record["id"],
        "sample_id": int(record["sample_id"]),
        "event_id": record.get("event_id"),
        "video_path": record["video_path"],
        "video_type": record["video_type"],
        "target_entity": record.get("target_entity", {}),
        "question": record["question"],
        "options": record["options"],
        "answer": record.get("answer", ""),
        "answer_index": int(record["answer_index"]),
        "option_roles": option_roles(record["options"]),
        "caption": caption,
        "judge_model": client.model,
        "judge_answer": judge_letter or "",
        "judge_index": judge_index,
        "is_correct": correct,
        "classification": classification,
        "raw_judge_response": raw_response,
        "judge_error": error,
    }


def _failed_judge_record(record: Dict[str, Any], caption: str, error: str) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "sample_id": int(record["sample_id"]),
        "event_id": record.get("event_id"),
        "video_path": record["video_path"],
        "video_type": record["video_type"],
        "target_entity": record.get("target_entity", {}),
        "question": record["question"],
        "options": record["options"],
        "answer": record.get("answer", ""),
        "answer_index": int(record["answer_index"]),
        "option_roles": option_roles(record["options"]),
        "caption": caption,
        "judge_model": os.getenv("MOTIONATLAS_JUDGE_MODEL", "gemini-2.5-pro"),
        "judge_answer": "",
        "judge_index": None,
        "is_correct": False,
        "classification": "wrong",
        "raw_judge_response": "",
        "judge_error": error,
    }


def _judge_sample(doc: Dict[str, Any], caption: str) -> Dict[str, Any]:
    mcqs = list(doc.get("mcqs", []) or [])
    if not caption.strip():
        mcq_results = [_failed_judge_record(record, caption, "empty caption") for record in mcqs]
    else:
        client = create_judge_client()
        max_workers = max(1, int(os.getenv("MOTIONATLAS_JUDGE_WORKERS", "4")))
        mcq_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(judge_one_mcq, record, caption, client) for record in mcqs]
            for future in as_completed(futures):
                mcq_results.append(future.result())
        mcq_results.sort(key=lambda item: str(item.get("id", "")))

    return {
        "sample_id": int(doc["sample_id"]),
        "video_path": doc["video_path"],
        "video_type": doc["video_type"],
        "target_entity": doc.get("target_entity", {}),
        "caption": caption,
        "mcq_results": mcq_results,
    }


def motionatlas_process_results(doc: Dict[str, Any], results):
    caption = str(results[0] if results else "")
    payload = _judge_sample(doc, caption)
    return {metric: payload for metric in METRIC_KEYS}


def _flatten_results(results: List[Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for result in results:
        if isinstance(result, dict) and isinstance(result.get("mcq_results"), list):
            flat.extend(result["mcq_results"])
        elif isinstance(result, list):
            flat.extend([item for item in result if isinstance(item, dict)])
    return flat


def _compute_counts(results: List[Any]) -> Dict[str, float]:
    flat = _flatten_results(results)
    total = len(flat)
    correct = sum(1 for item in flat if item.get("classification") == "correct")
    miss = sum(1 for item in flat if item.get("classification") == "miss")
    wrong = sum(1 for item in flat if item.get("classification") == "wrong")
    answered = sum(1 for item in flat if item.get("judge_index") is not None)
    return {
        "total": float(total),
        "answered": float(answered),
        "correct": float(correct),
        "miss": float(miss),
        "wrong": float(wrong),
    }


def motionatlas_aggregate_weighted_score(results):
    counts = _compute_counts(results)
    total = counts["total"]
    return 0.0 if total == 0 else (counts["correct"] - 0.5 * counts["wrong"]) / total


def motionatlas_aggregate_accuracy(results):
    counts = _compute_counts(results)
    return 0.0 if counts["total"] == 0 else counts["correct"] / counts["total"]


def motionatlas_aggregate_recall(results):
    counts = _compute_counts(results)
    return 0.0 if counts["total"] == 0 else (counts["correct"] + counts["wrong"]) / counts["total"]


def motionatlas_aggregate_precision(results):
    counts = _compute_counts(results)
    denom = counts["correct"] + counts["wrong"]
    return 0.0 if denom == 0 else counts["correct"] / denom


def motionatlas_aggregate_answered_rate(results):
    counts = _compute_counts(results)
    return 0.0 if counts["total"] == 0 else counts["answered"] / counts["total"]
