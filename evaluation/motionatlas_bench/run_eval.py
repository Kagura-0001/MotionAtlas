from __future__ import annotations

import argparse
import os
from pathlib import Path

from .infer_captions import QWEN_DEFAULT_MAX_PIXELS, run_caption_inference
from .io_utils import dump_json, utc_now
from .judge_captions import run_caption_judging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MotionAtlas-Bench caption-to-judge evaluation.")
    parser.add_argument("--data-root", type=Path, required=True, help="MotionAtlas-Bench release directory")
    parser.add_argument("--mcqs", type=Path, default=None, help="Path to mcqs.jsonl")
    parser.add_argument("--target-masks", type=Path, default=None, help="Path to target_masks.jsonl.gz")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--setting", choices=["first_mask", "overlay_all"], default="first_mask")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--first-mask-within-frame-budget",
        action="store_true",
        help="Replace a sampled frame with the first mask frame instead of inserting an extra frame.",
    )
    parser.add_argument("--outline-thickness", type=int, default=3)

    parser.add_argument(
        "--caption-model",
        default=os.getenv(
            "MOTIONATLAS_CAPTION_MODEL",
            os.getenv("MOTIONATLAS_MODEL", os.getenv("OPENAI_MODEL", "qwen3-vl-4b")),
        ),
    )
    parser.add_argument(
        "--caption-base-url",
        default=os.getenv(
            "MOTIONATLAS_CAPTION_BASE_URL",
            os.getenv("MOTIONATLAS_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")),
        ),
    )
    parser.add_argument(
        "--caption-api-key",
        default=os.getenv(
            "MOTIONATLAS_CAPTION_API_KEY",
            os.getenv("MOTIONATLAS_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        ),
    )
    parser.add_argument("--caption-workers", type=int, default=16)
    parser.add_argument("--caption-timeout", type=float, default=300.0)
    parser.add_argument("--caption-max-retries", type=int, default=3)
    parser.add_argument("--caption-max-tokens", type=int, default=10240)
    parser.add_argument("--caption-temperature", type=float, default=0.0)
    parser.add_argument("--caption-top-p", type=float, default=1.0)
    parser.add_argument("--caption-system-prompt", default="")
    parser.add_argument("--image-detail", default="auto")
    parser.add_argument("--image-format", choices=["JPEG", "PNG"], default="JPEG")
    parser.add_argument("--min-pixels", type=int, default=0, help="Optional vLLM/Qwen mm_processor_kwargs min_pixels")
    parser.add_argument("--max-pixels", type=int, default=QWEN_DEFAULT_MAX_PIXELS, help="Optional vLLM/Qwen mm_processor_kwargs max_pixels")
    parser.add_argument("--total-pixels", type=int, default=0, help="Optional vLLM/Qwen mm_processor_kwargs total_pixels")

    parser.add_argument("--judge-provider", choices=["gemini", "openai-compatible"], default="gemini")
    parser.add_argument("--judge-model", default=os.getenv("MOTIONATLAS_JUDGE_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--judge-base-url", default=os.getenv("MOTIONATLAS_JUDGE_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("MOTIONATLAS_JUDGE_API_KEY", ""))
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)

    parser.add_argument("--limit", type=int, default=0, help="Limit number of MCQs after loading")
    parser.add_argument("--limit-samples", type=int, default=0, help="Limit number of sample groups after loading")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="Render prompts and captions metadata without calling APIs")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.resolve()
    args.mcqs = (args.mcqs or args.data_root / "mcqs.jsonl").resolve()
    args.target_masks = (args.target_masks or args.data_root / "target_masks.jsonl.gz").resolve()
    args.output = args.output.resolve()
    args.captions_path = args.output / "captions.jsonl"
    args.judge_predictions_path = args.output / "judge_predictions.jsonl"
    args.metrics_path = args.output / "metrics.json"
    args.config_path = args.output / "run_config.json"

    if not args.judge_api_key and args.judge_provider == "gemini":
        args.judge_api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not args.judge_api_key and args.judge_provider == "openai-compatible":
        args.judge_api_key = "EMPTY"
    return args


def dump_run_config(args: argparse.Namespace) -> None:
    dump_json(
        args.config_path,
        {
            "data_root": str(args.data_root),
            "mcqs": str(args.mcqs),
            "target_masks": str(args.target_masks),
            "output": str(args.output),
            "captions_path": str(args.captions_path),
            "judge_predictions_path": str(args.judge_predictions_path),
            "metrics_path": str(args.metrics_path),
            "setting": args.setting,
            "num_frames": args.num_frames,
            "first_mask_within_frame_budget": args.first_mask_within_frame_budget,
            "caption_model": args.caption_model,
            "caption_base_url": args.caption_base_url,
            "caption_workers": args.caption_workers,
            "caption_max_tokens": args.caption_max_tokens,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "judge_workers": args.judge_workers,
            "judge_max_tokens": args.judge_max_tokens,
            "resume": args.resume,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "generated_at": utc_now(),
        },
    )


def main() -> None:
    args = normalize_args(parse_args())
    dump_run_config(args)

    caption_records = run_caption_inference(args)
    if args.dry_run:
        print(f"dry run wrote {len(caption_records)} caption records to {args.captions_path}; skipped judge")
        return

    _, metrics = run_caption_judging(args)
    print(
        f"wrote captions to {args.captions_path}; "
        f"wrote judge results to {args.judge_predictions_path}; "
        f"weighted_score={metrics['weighted_score']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})"
    )


if __name__ == "__main__":
    main()
