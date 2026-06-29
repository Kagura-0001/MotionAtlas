from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .data import group_mcqs_by_sample, load_mcqs, load_target_masks, representative_record
from .io_utils import append_jsonl, read_jsonl, reset_jsonl, utc_now

if TYPE_CHECKING:
    from .clients.openai_compatible import OpenAICompatibleClient

QWEN_DEFAULT_MAX_PIXELS = 224 * 32 * 32


def build_caption_prompt(setting: str) -> str:
    if setting == "first_mask":
        grounding = (
            "One frame contains a green contour highlighting the target object. "
            "Use it to identify the target object, then describe that same object "
            "throughout the full video."
        )
    elif setting == "overlay_all":
        grounding = (
            "Some frames contain green contours highlighting the target object. "
            "Use the highlighted frames to identify the target object, then describe "
            "that same object throughout the full video."
        )
    else:
        raise ValueError(f"unsupported setting: {setting}")

    return (
        "You are given sampled video frames in chronological order. "
        "Each image is labeled as Frame 1, Frame 2, and so on in the input sequence. "
        f"{grounding}\n"
        "Write a detailed motion caption for the target object. Cover its appearance, "
        "pose, actions, motion, state changes, and interactions with nearby objects. "
        "Focus strictly on the target object and directly relevant interacted objects. "
        "Do not answer any multiple-choice question."
    )


def build_mm_processor_kwargs(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if args.min_pixels > 0:
        kwargs["min_pixels"] = int(args.min_pixels)
    if args.max_pixels > 0:
        kwargs["max_pixels"] = int(args.max_pixels)
    if args.total_pixels > 0:
        kwargs["total_pixels"] = int(args.total_pixels)
    return kwargs or None


def build_caption_record(
    sample_record: dict[str, Any],
    pred_caption: str,
    render_metadata: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    error: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": int(sample_record["sample_id"]),
        "video_path": sample_record["video_path"],
        "video_type": sample_record["video_type"],
        "target_entity": sample_record.get("target_entity", {}),
        "setting": args.setting,
        "num_frames": int(args.num_frames),
        "caption_model": args.caption_model,
        "pred_caption": pred_caption,
        "error": error,
        "prompt": prompt,
        "render_metadata": render_metadata,
        "dry_run": bool(args.dry_run),
        "generated_at": utc_now(),
    }


def load_existing_captions(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def run_caption_sample(
    sample_id: int,
    records: list[dict[str, Any]],
    masks_by_sample: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    client: Optional["OpenAICompatibleClient"],
    processed_sample_ids: set[int],
) -> Optional[dict[str, Any]]:
    from .render import render_sample

    if sample_id in processed_sample_ids:
        return None

    sample_record = representative_record(records)
    prompt = build_caption_prompt(args.setting)

    if sample_id not in masks_by_sample:
        return build_caption_record(
            sample_record,
            pred_caption="",
            render_metadata={"sample_id": sample_id, "setting": args.setting},
            prompt=prompt,
            args=args,
            error=f"missing target masks for sample_id={sample_id}",
        )

    try:
        rendered = render_sample(
            data_root=args.data_root,
            sample_record=sample_record,
            masks=masks_by_sample[sample_id],
            setting=args.setting,
            num_frames=args.num_frames,
            overlay_first_within_frame_budget=args.first_mask_within_frame_budget,
            outline_thickness=args.outline_thickness,
        )
    except Exception as exc:
        return build_caption_record(
            sample_record,
            pred_caption="",
            render_metadata={"sample_id": sample_id, "setting": args.setting},
            prompt=prompt,
            args=args,
            error=str(exc),
        )

    if args.dry_run:
        return build_caption_record(sample_record, "", rendered.metadata, prompt, args)

    if client is None:
        return build_caption_record(sample_record, "", rendered.metadata, prompt, args, error="client is not initialized")

    try:
        caption = client.chat_with_images(
            prompt=prompt,
            images=[frame.image for frame in rendered.frames],
            max_tokens=args.caption_max_tokens,
            temperature=args.caption_temperature,
            top_p=args.caption_top_p,
            system_prompt=args.caption_system_prompt or None,
            image_detail=args.image_detail,
            image_format=args.image_format,
            mm_processor_kwargs=build_mm_processor_kwargs(args),
        )
        return build_caption_record(sample_record, caption, rendered.metadata, prompt, args)
    except Exception as exc:
        return build_caption_record(sample_record, "", rendered.metadata, prompt, args, error=str(exc))


def load_eval_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    records = load_mcqs(args.mcqs)
    if args.limit > 0:
        records = records[: args.limit]
    groups = group_mcqs_by_sample(records)
    if args.limit_samples > 0:
        keep_sample_ids = sorted(groups)[: args.limit_samples]
        groups = {sample_id: groups[sample_id] for sample_id in keep_sample_ids}
        records = [record for sample_id in keep_sample_ids for record in groups[sample_id]]
    return records, groups


def run_caption_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    captions_path = args.captions_path
    existing = load_existing_captions(captions_path) if args.resume else []
    if not args.resume and captions_path.exists():
        if not args.overwrite:
            raise RuntimeError(f"{captions_path} already exists; pass --resume or --overwrite")
        reset_jsonl(captions_path)

    _, groups = load_eval_records(args)
    masks_by_sample = load_target_masks(args.target_masks)
    processed_sample_ids = {int(item["sample_id"]) for item in existing if "sample_id" in item}

    client = None
    if not args.dry_run:
        from .clients.openai_compatible import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            model=args.caption_model,
            base_url=args.caption_base_url,
            api_key=args.caption_api_key,
            timeout=args.caption_timeout,
            max_retries=args.caption_max_retries,
        )

    sample_items = [(sample_id, records) for sample_id, records in sorted(groups.items()) if sample_id not in processed_sample_ids]
    iterator = sample_items
    if tqdm is not None:
        iterator = tqdm(sample_items, desc="MotionAtlas caption samples")  # type: ignore[assignment]

    new_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.caption_workers)) as executor:
        futures = [
            executor.submit(run_caption_sample, sample_id, records, masks_by_sample, args, client, processed_sample_ids)
            for sample_id, records in iterator
        ]
        done_iter = as_completed(futures)
        if tqdm is not None:
            done_iter = tqdm(done_iter, total=len(futures), desc="Completed caption samples")  # type: ignore[assignment]
        for future in done_iter:
            record = future.result()
            if record is None:
                continue
            append_jsonl(captions_path, [record])
            new_records.append(record)

    return existing + new_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MotionAtlas-Bench target-object motion captions.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mcqs", type=Path, default=None)
    parser.add_argument("--target-masks", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", choices=["first_mask", "overlay_all"], default="first_mask")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--first-mask-within-frame-budget", action="store_true")
    parser.add_argument("--outline-thickness", type=int, default=3)
    parser.add_argument("--caption-model", default=os.getenv("MOTIONATLAS_CAPTION_MODEL", "qwen3-vl-4b"))
    parser.add_argument("--caption-base-url", default=os.getenv("MOTIONATLAS_CAPTION_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--caption-api-key", default=os.getenv("MOTIONATLAS_CAPTION_API_KEY", "EMPTY"))
    parser.add_argument("--caption-workers", type=int, default=16)
    parser.add_argument("--caption-timeout", type=float, default=300.0)
    parser.add_argument("--caption-max-retries", type=int, default=3)
    parser.add_argument("--caption-max-tokens", type=int, default=10240)
    parser.add_argument("--caption-temperature", type=float, default=0.0)
    parser.add_argument("--caption-top-p", type=float, default=1.0)
    parser.add_argument("--caption-system-prompt", default="")
    parser.add_argument("--image-detail", default="auto")
    parser.add_argument("--image-format", choices=["JPEG", "PNG"], default="JPEG")
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=QWEN_DEFAULT_MAX_PIXELS)
    parser.add_argument("--total-pixels", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.mcqs = (args.mcqs or args.data_root / "mcqs.jsonl").resolve()
    args.target_masks = (args.target_masks or args.data_root / "target_masks.jsonl.gz").resolve()
    args.output = args.output.resolve()
    args.captions_path = args.output / "captions.jsonl"
    return args


def main() -> None:
    args = parse_args()
    records = run_caption_inference(args)
    print(f"wrote {len(records)} total captions to {args.captions_path}")


if __name__ == "__main__":
    main()
