"""VQA sample loading, prediction records, and metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class VQASample:
    vqa_id: str
    task_family: str
    primary_video_id: str | None
    participant_id: str | None
    question: str
    choices: list[str]
    correct_idx: int
    inputs: dict[str, Any]

    @classmethod
    def from_row(cls, row: pd.Series) -> "VQASample":
        return cls(
            vqa_id=row["vqa_id"],
            task_family=row["task_family"],
            primary_video_id=row.get("primary_video_id"),
            participant_id=row.get("participant_id"),
            question=row["question"],
            choices=json.loads(row["choices_json"]),
            correct_idx=int(row["correct_idx"]),
            inputs=json.loads(row["inputs_json"]),
        )


@dataclass
class VQAPrediction:
    sample_id: str
    baseline: str
    task_family: str
    video_id: str | None
    question: str
    choices: list[str]
    gold: int
    prediction: int
    correct: bool
    evidence_ids: list[str]
    tool_calls: list[str]
    failure_type: str | None
    attempt_count: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_vqa_samples(index_dir: Path, limit: int | None = None, task_family: str | None = None) -> list[VQASample]:
    df = pd.read_parquet(index_dir / "vqa_samples.parquet")
    if task_family:
        df = df[df["task_family"] == task_family]
    if limit is not None:
        df = df.head(limit)
    return [VQASample.from_row(row) for _, row in df.iterrows()]


def parse_choice_prediction(text: str, choices: list[str]) -> int:
    stripped = text.strip()
    if stripped.isdigit():
        idx = int(stripped)
        if 0 <= idx < len(choices):
            return idx
    lowered = stripped.lower()
    for idx, choice in enumerate(choices):
        if choice.lower() == lowered or choice.lower() in lowered:
            return idx
    return 0


def compute_metrics(predictions: list[VQAPrediction]) -> dict[str, Any]:
    if not predictions:
        return {"count": 0, "accuracy": None, "by_task_family": {}}
    correct = sum(1 for pred in predictions if pred.correct)
    by_family: dict[str, dict[str, Any]] = {}
    for pred in predictions:
        bucket = by_family.setdefault(pred.task_family, {"count": 0, "correct": 0})
        bucket["count"] += 1
        bucket["correct"] += int(pred.correct)
    for bucket in by_family.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else None
    return {
        "count": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "by_task_family": by_family,
    }
