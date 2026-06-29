from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Protocol

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .data import load_mcqs
from .io_utils import append_jsonl, dump_json, read_jsonl, reset_jsonl, utc_now
from .prompt import is_correct, letter_for_index, parse_answer_letter
from .score import compute_judge_metrics

MISS_OPTION_ROLES = {"neg_not_shown", "neg_shown_no_value"}

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

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        ...


class GeminiOfficialJudgeClient:
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai is required for --judge-provider gemini") from exc

        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not resolved_key:
            raise RuntimeError("Gemini judge requires --judge-api-key, GEMINI_API_KEY, or GOOGLE_API_KEY")

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
            parts: list[str] = []
            for item in output if isinstance(output, list) else [output]:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            if parts:
                return "\n".join(parts)
        return str(response)

    def _create_interaction(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Any:
        interactions = getattr(self.client, "interactions", None)
        if interactions is None or not hasattr(interactions, "create"):
            return self._create_generate_content(prompt, max_tokens, temperature, top_p)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "motionatlas_judge_answer",
                    "schema": JUDGE_RESPONSE_SCHEMA,
                },
            },
            "generation_config": {
                "max_output_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
            "store": False,
        }
        try:
            return self.client.interactions.create(**kwargs)
        except TypeError:
            kwargs.pop("generation_config", None)
            kwargs.pop("store", None)
            return self.client.interactions.create(**kwargs)

    def _create_generate_content(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Any:
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
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                try:
                    response = self._create_interaction(prompt, max_tokens, temperature, top_p)
                except Exception as exc:
                    message = str(exc).lower()
                    if "response_format" not in message and "generation_config" not in message:
                        raise
                    response = self._create_generate_content(prompt, max_tokens, temperature, top_p)
                return self._extract_text(response)
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError(f"Gemini judge failed after {self.max_retries} attempts: {last_error}")


class OpenAICompatibleJudgeClient:
    def __init__(self, model: str, base_url: str, api_key: str, timeout: float, max_retries: int):
        if not base_url:
            raise RuntimeError("--judge-base-url is required for --judge-provider openai-compatible")
        from .clients.openai_compatible import OpenAICompatibleClient

        self.model = model
        self.client = OpenAICompatibleClient(
            model=model,
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        return self.client.chat(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def create_judge_client(args: argparse.Namespace) -> TextJudgeClient:
    if args.judge_provider == "gemini":
        return GeminiOfficialJudgeClient(
            model=args.judge_model,
            api_key=args.judge_api_key,
            max_retries=args.judge_max_retries,
        )
    if args.judge_provider == "openai-compatible":
        return OpenAICompatibleJudgeClient(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
        )
    raise ValueError(f"unsupported judge provider: {args.judge_provider}")


def format_options(options: list[Any]) -> str:
    return "\n".join(f"{letter_for_index(i)}. {option}" for i, option in enumerate(options))


def build_judge_prompt(record: dict[str, Any], caption: str) -> str:
    allowed = ", ".join(letter_for_index(i) for i in range(len(record["options"])))
    return (
        "Based on the following motion caption, answer the multiple-choice question.\n\n"
        f"Motion Caption:\n{caption}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{format_options(record['options'])}\n\n"
        "Instructions:\n"
        "- Read the motion caption carefully.\n"
        "- Select the option best supported by the caption.\n"
        "- If the caption does not contain enough information, choose the matching "
        "not-mentioned or unclear/not-visible option when present.\n\n"
        "Return JSON only with exactly one field in this format:\n"
        '{"answer":"<LETTER>"}\n'
        f"The answer letter must be one of: {allowed}."
    )


def option_role(option_text: Any) -> str:
    text = " ".join(str(option_text).lower().split())
    if "does not mention" in text or "not mention" in text:
        return "neg_not_shown"
    if "unclear" in text or "not visible" in text or "obscured" in text or "indistinct" in text:
        return "neg_shown_no_value"
    if "value differs" in text or "differs from all listed options" in text:
        return "neg_shown_diff_value"
    return "other"


def option_roles(options: list[Any]) -> list[str]:
    return [option_role(option) for option in options]


def classify_judge_answer(record: dict[str, Any], judge_index: Optional[int], correct: bool) -> str:
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


def load_captions_by_sample(path: Path) -> dict[int, dict[str, Any]]:
    captions: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if "sample_id" not in record:
            continue
        captions[int(record["sample_id"])] = record
    return captions


def build_judge_record(
    record: dict[str, Any],
    caption_record: Optional[dict[str, Any]],
    raw_response: str,
    prompt: str,
    args: argparse.Namespace,
    error: str = "",
) -> dict[str, Any]:
    pred_letter, pred_index = parse_answer_letter(raw_response, len(record["options"])) if raw_response and not error else (None, None)
    correct = is_correct(record, pred_index)
    classification = classify_judge_answer(record, pred_index, correct)
    caption_text = str(caption_record.get("pred_caption", "")) if caption_record else ""
    return {
        "id": record["id"],
        "sample_id": int(record["sample_id"]),
        "event_id": record.get("event_id"),
        "video_path": record["video_path"],
        "video_type": record["video_type"],
        "target_entity": record.get("target_entity", {}),
        "setting": caption_record.get("setting") if caption_record else args.setting,
        "question": record["question"],
        "options": record["options"],
        "answer": record.get("answer", ""),
        "answer_index": int(record["answer_index"]),
        "option_roles": option_roles(record["options"]),
        "caption": caption_text,
        "caption_error": str(caption_record.get("error", "")) if caption_record else "missing caption",
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_answer": pred_letter or "",
        "judge_index": pred_index,
        "is_correct": correct,
        "classification": classification,
        "raw_judge_response": raw_response,
        "judge_error": error,
        "judge_prompt": prompt,
        "generated_at": utc_now(),
    }


def judge_one_record(
    record: dict[str, Any],
    captions_by_sample: dict[int, dict[str, Any]],
    client: Optional[TextJudgeClient],
    args: argparse.Namespace,
) -> dict[str, Any]:
    caption_record = captions_by_sample.get(int(record["sample_id"]))
    if caption_record is None:
        return build_judge_record(record, None, "", "", args, error=f"missing caption for sample_id={record['sample_id']}")

    caption = str(caption_record.get("pred_caption", "")).strip()
    caption_error = str(caption_record.get("error", "")).strip()
    if not caption:
        error = caption_error or f"empty caption for sample_id={record['sample_id']}"
        return build_judge_record(record, caption_record, "", "", args, error=error)

    prompt = build_judge_prompt(record, caption)
    if args.dry_run:
        return build_judge_record(record, caption_record, "", prompt, args)
    if client is None:
        return build_judge_record(record, caption_record, "", prompt, args, error="judge client is not initialized")

    try:
        raw_response = client.generate(
            prompt=prompt,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
        )
        return build_judge_record(record, caption_record, raw_response, prompt, args)
    except Exception as exc:
        return build_judge_record(record, caption_record, "", prompt, args, error=str(exc))


def prepare_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_mcqs(args.mcqs)
    if args.limit > 0:
        records = records[: args.limit]
    if args.limit_samples > 0:
        keep_sample_ids = set(sorted({int(record["sample_id"]) for record in records})[: args.limit_samples])
        records = [record for record in records if int(record["sample_id"]) in keep_sample_ids]
    return records


def run_caption_judging(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_path = args.judge_predictions_path
    metrics_path = args.metrics_path

    existing = read_jsonl(predictions_path) if args.resume else []
    if not args.resume and predictions_path.exists():
        if not args.overwrite:
            raise RuntimeError(f"{predictions_path} already exists; pass --resume or --overwrite")
        reset_jsonl(predictions_path)
    processed_ids = {str(item.get("id")) for item in existing}

    records = [record for record in prepare_records(args) if str(record["id"]) not in processed_ids]
    captions_by_sample = load_captions_by_sample(args.captions_path)
    client = None if args.dry_run else create_judge_client(args)

    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="MotionAtlas judge MCQs")  # type: ignore[assignment]

    new_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as executor:
        futures = [
            executor.submit(judge_one_record, record, captions_by_sample, client, args)
            for record in iterator
        ]
        done_iter = as_completed(futures)
        if tqdm is not None:
            done_iter = tqdm(done_iter, total=len(futures), desc="Completed judge MCQs")  # type: ignore[assignment]
        for future in done_iter:
            result = future.result()
            append_jsonl(predictions_path, [result])
            new_results.append(result)

    final_results = existing + new_results
    metrics = compute_judge_metrics(final_results)
    metrics.update(
        {
            "num_existing_predictions": len(existing),
            "num_new_predictions": len(new_results),
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "captions_path": str(args.captions_path),
            "generated_at": utc_now(),
        }
    )
    dump_json(metrics_path, metrics)
    return final_results, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge MotionAtlas-Bench MCQs from generated motion captions.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mcqs", type=Path, default=None)
    parser.add_argument("--captions", dest="captions_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", choices=["first_mask", "overlay_all"], default="first_mask")
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.mcqs = (args.mcqs or args.data_root / "mcqs.jsonl").resolve()
    args.captions_path = args.captions_path.resolve()
    args.output = args.output.resolve()
    args.judge_predictions_path = args.output / "judge_predictions.jsonl"
    args.metrics_path = args.output / "metrics.json"
    if not args.judge_api_key and args.judge_provider == "gemini":
        args.judge_api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not args.judge_api_key and args.judge_provider == "openai-compatible":
        args.judge_api_key = "EMPTY"
    return args


def main() -> None:
    args = parse_args()
    _, metrics = run_caption_judging(args)
    print(
        f"wrote judge results to {args.judge_predictions_path}; "
        f"weighted_score={metrics['weighted_score']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})"
    )


if __name__ == "__main__":
    main()
