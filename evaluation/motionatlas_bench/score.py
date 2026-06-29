from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    answered = sum(1 for item in results if item.get("pred_index") is not None)
    correct = sum(1 for item in results if item.get("is_correct") is True)

    metrics: dict[str, Any] = {
        "total": total,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "answered_accuracy": correct / answered if answered else 0.0,
    }

    by_video_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "answered": 0, "correct": 0})
    for item in results:
        key = str(item.get("video_type", "unknown"))
        by_video_type[key]["total"] += 1
        if item.get("pred_index") is not None:
            by_video_type[key]["answered"] += 1
        if item.get("is_correct") is True:
            by_video_type[key]["correct"] += 1

    metrics["by_video_type"] = {
        key: {
            **value,
            "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            "answered_accuracy": value["correct"] / value["answered"] if value["answered"] else 0.0,
        }
        for key, value in sorted(by_video_type.items())
    }
    return metrics


def compute_judge_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for item in results if item.get("classification") == "correct")
    miss = sum(1 for item in results if item.get("classification") == "miss")
    wrong = sum(1 for item in results if item.get("classification") == "wrong")
    answered = sum(1 for item in results if item.get("judge_index") is not None)

    metrics: dict[str, Any] = {
        "total": total,
        "answered": answered,
        "correct": correct,
        "miss": miss,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "recall": (correct + wrong) / total if total else 0.0,
        "precision": correct / (correct + wrong) if (correct + wrong) else 0.0,
        "weighted_score": (correct - 0.5 * wrong) / total if total else 0.0,
    }

    by_video_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "answered": 0, "correct": 0, "miss": 0, "wrong": 0}
    )
    for item in results:
        key = str(item.get("video_type", "unknown"))
        classification = item.get("classification")
        if classification not in {"correct", "miss", "wrong"}:
            classification = "wrong"
        by_video_type[key]["total"] += 1
        if item.get("judge_index") is not None:
            by_video_type[key]["answered"] += 1
        by_video_type[key][classification] += 1

    metrics["by_video_type"] = {}
    for key, value in sorted(by_video_type.items()):
        total_type = value["total"]
        correct_type = value["correct"]
        wrong_type = value["wrong"]
        metrics["by_video_type"][key] = {
            **value,
            "accuracy": correct_type / total_type if total_type else 0.0,
            "recall": (correct_type + wrong_type) / total_type if total_type else 0.0,
            "precision": correct_type / (correct_type + wrong_type) if (correct_type + wrong_type) else 0.0,
            "weighted_score": (correct_type - 0.5 * wrong_type) / total_type if total_type else 0.0,
        }
    return metrics
