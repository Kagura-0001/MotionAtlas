from __future__ import annotations

import json
import re
from typing import Any, Optional


def letter_for_index(index: int) -> str:
    return chr(ord("A") + int(index))


def build_mcq_prompt(record: dict[str, Any], setting: str) -> str:
    options = [str(option) for option in record["options"]]
    options_lines = "\n".join(f"{letter_for_index(i)}. {option}" for i, option in enumerate(options))
    allowed = ", ".join(letter_for_index(i) for i in range(len(options)))
    if setting == "first_mask":
        grounding = "One provided frame contains a green contour highlighting the target object."
    else:
        grounding = "Some provided frames may contain a green contour highlighting the target object."

    return (
        "You are given sampled video frames in chronological order. "
        "Each image is labeled as Frame 1, Frame 2, and so on in the input sequence. "
        f"{grounding}\n"
        "Answer the multiple-choice question based on the full video evidence and the highlighted target.\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{options_lines}\n\n"
        "Return JSON only with exactly one field in this format:\n"
        '{"answer":"<LETTER>"}\n'
        f"The answer letter must be one of: {allowed}."
    )


def _strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", value).strip()
        value = re.sub(r"\s*```$", "", value).strip()
    return value


def parse_answer_letter(raw_response: str, num_options: int) -> tuple[Optional[str], Optional[int]]:
    if num_options <= 0:
        return None, None
    allowed = {letter_for_index(i) for i in range(num_options)}
    text = _strip_code_fence(raw_response)

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

    patterns = [
        r'"answer"\s*:\s*"([A-Za-z])"',
        r"'answer'\s*:\s*'([A-Za-z])'",
        r"(?:answer|choice|option)\s*[:：]?\s*\(?([A-Za-z])\)?",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            letter = match.group(1).upper()
            if letter in allowed:
                return letter, ord(letter) - ord("A")
    return None, None


def is_correct(record: dict[str, Any], pred_index: Optional[int]) -> bool:
    if pred_index is None:
        return False
    if pred_index < 0 or pred_index >= len(record["options"]):
        return False
    return int(record["answer_index"]) == int(pred_index)
