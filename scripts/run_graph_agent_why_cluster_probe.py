#!/usr/bin/env python3
"""Run a small real probe over targeted fine-grained why semantic clusters."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths
from food_agent.vqa import VQAPrediction, compute_metrics
from scripts.run_graph_agent_batch import (
    RAW_REVISIT_TOOLS,
    STRUCTURED_QUERY_TOOLS,
    count_failed_tool_recoveries,
    count_ineffective_tool_avoidances,
    count_planner_overrides_from_result,
    count_relation_reuse_from_result,
    count_reuse_memory_from_result,
    count_tools,
    count_verifier_blocked_finish_from_result,
)


CLUSTERS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "towel-cluster": {
        "question_any": ("paper towel", "tea towel", "dish cloth", "cloth", "napkin", "towel", "hand towel"),
    },
    "access-space-cluster": {
        "question_any": ("<move ", "<pick up ", "<open ", "<remove ", "<shift "),
        "choice_any": (
            "behind",
            "retrieve",
            "look what's behind",
            "look what is behind",
            "clear the way",
            "make space",
            "make room",
            "put back",
            "right place",
            "freed slot",
            "put the",
            "put ",
            "free the",
            "free ",
        ),
        "choice_all": (
            "behind|retrieve|look what's behind|look what is behind",
            "clear the way|make space|make room|put back|right place|freed slot|put .* down|free ",
        ),
    },
    "future-use-cluster": {
        "combined_any": ("weigh", "measure", "pour", "empty", "serve", "check", "fill", "wash", "clean", "dry", "turn on", "turn off", "open ", "close "),
    },
    "state-change-cluster": {
        "question_any": ("<tap ", "<press ", "<push ", "<switch ", "<turn ", "<move tap", "<reach for tap"),
        "combined_any": ("tap kitchen scale", "scale", "tap", "press", "push", "switch", "turn on", "turn off", "open tap", "close tap"),
    },
}


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "graph_agent_why_cluster_probe")
    parser.add_argument("--cluster", choices=sorted(CLUSTERS), required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)

    run_dir = args.out_dir / args.cluster
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selection.json"
    pred_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"

    rows = load_or_build_selection(
        index_file=args.index_file,
        cluster=args.cluster,
        limit=args.limit,
        selection_path=selection_path,
        resume=args.resume,
    )
    completed = load_predictions(pred_path) if args.resume else {}
    ordered: list[VQAPrediction] = list(completed.values())

    for index, row in enumerate(rows, start=1):
        sample_id = str(row["vqa_id"])
        if sample_id in completed:
            pred = completed[sample_id]
            print(
                f"[{index}/{len(rows)}] skip sample={sample_id} pred={pred.prediction} gold={pred.gold} correct={pred.correct}",
                flush=True,
            )
            continue
        result = agent.answer_vqa_row(row, max_steps=args.max_steps)
        tool_calls = [entry.get("tool", "") for entry in result.tool_trace]
        pred = VQAPrediction(
            sample_id=sample_id,
            baseline=f"graph-agent-{args.cluster}",
            task_family=str(row["task_family"]),
            video_id=str(row["primary_video_id"]),
            question=str(row["question"]),
            choices=json.loads(row["choices_json"]),
            gold=int(row["correct_idx"]),
            prediction=int(result.prediction if result.prediction is not None else 0),
            correct=result.prediction == int(row["correct_idx"]),
            evidence_ids=[],
            tool_calls=tool_calls,
            failure_type=None if result.prediction == int(row["correct_idx"]) else "reasoning_error",
            attempt_count=1,
            reuse_memory_count=count_reuse_memory_from_result(result),
            raw_revisit_count=count_tools(tool_calls, RAW_REVISIT_TOOLS),
            structured_query_count=count_tools(tool_calls, STRUCTURED_QUERY_TOOLS),
            session_video_position=index,
            relation_reuse_count=count_relation_reuse_from_result(result),
            planner_override_count=count_planner_overrides_from_result(result),
            verifier_blocked_finish_count=count_verifier_blocked_finish_from_result(result),
            tool_failure_count=len(result.tool_failures),
            ineffective_tool_count=len(result.ineffective_tools),
            failed_tool_recovery_count=count_failed_tool_recoveries(result.tool_trace),
            ineffective_tool_avoidance_count=count_ineffective_tool_avoidances(result.tool_trace),
            tool_call_count=len(tool_calls),
            reasoning_step_count=len(result.tool_trace),
            elapsed_seconds=float(result.elapsed_seconds),
            prompt_tokens=float((result.usage or {}).get("prompt_tokens") or 0.0),
            completion_tokens=float((result.usage or {}).get("completion_tokens") or 0.0),
            total_tokens=float((result.usage or {}).get("total_tokens") or 0.0),
            estimated_cost=float((result.usage or {}).get("estimated_cost") or 0.0),
        )
        append_jsonl(pred_path, prediction_to_dict(pred))
        completed[sample_id] = pred
        ordered = [completed[str(item["vqa_id"])] for item in rows if str(item["vqa_id"]) in completed]
        print(
            f"[{index}/{len(rows)}] sample={sample_id} pred={pred.prediction} gold={pred.gold} correct={pred.correct} "
            f"tools={pred.tool_call_count} raw={pred.raw_revisit_count} structured={pred.structured_query_count}",
            flush=True,
        )
        summary_path.write_text(json.dumps(build_summary(args.cluster, rows, ordered), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = build_summary(args.cluster, rows, ordered)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_or_build_selection(
    *,
    index_file: Path,
    cluster: str,
    limit: int,
    selection_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    if resume and selection_path.exists():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
    rows = build_cluster_selection(index_file=index_file, cluster=cluster, limit=limit)
    selection_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def build_cluster_selection(*, index_file: Path, cluster: str, limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(index_file)
    subset = df[df["task_family"] == "fine_grained_why_recognition"].copy()
    subset = subset[subset.apply(lambda row: row_matches_cluster(dict(row), cluster=cluster), axis=1)].copy()
    subset = subset.sort_values(["primary_video_id", "vqa_id"]).head(limit)
    return [normalize_row(row) for row in subset.to_dict("records")]


def row_matches_cluster(row: dict[str, Any], *, cluster: str) -> bool:
    spec = CLUSTERS[cluster]
    question = str(row.get("question") or "").lower()
    choices = decode_choices(row.get("choices_json"))
    choices_text = " ".join(choices).lower()
    combined_text = f"{question} {choices_text}".strip()
    question_any = tuple(str(item).lower() for item in spec.get("question_any", ()))  # type: ignore[arg-type]
    choice_any = tuple(str(item).lower() for item in spec.get("choice_any", ()))  # type: ignore[arg-type]
    combined_any = tuple(str(item).lower() for item in spec.get("combined_any", ()))  # type: ignore[arg-type]
    choice_all = tuple(str(item) for item in spec.get("choice_all", ()))  # type: ignore[arg-type]
    if question_any and not any(token in question for token in question_any):
        return False
    if choice_any and not any(token in choices_text for token in choice_any):
        return False
    if combined_any and not any(token in combined_text for token in combined_any):
        return False
    for pattern in choice_all:
        if not re.search(pattern, choices_text, re.I):
            return False
    return True


def decode_choices(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            try:
                normalized[key] = value.item()
                continue
            except Exception:  # noqa: BLE001
                pass
        normalized[key] = value
    return normalized


def load_predictions(path: Path) -> dict[str, VQAPrediction]:
    if not path.exists():
        return {}
    completed: dict[str, VQAPrediction] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        completed[str(payload["sample_id"])] = VQAPrediction(**payload)
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_summary(cluster: str, rows: list[dict[str, Any]], predictions: list[VQAPrediction]) -> dict[str, Any]:
    metrics = compute_metrics(predictions)
    failures = [
        {
            "sample_id": pred.sample_id,
            "video_id": pred.video_id,
            "question": pred.question,
            "prediction": pred.prediction,
            "gold": pred.gold,
            "failure_type": pred.failure_type,
        }
        for pred in predictions
        if not pred.correct
    ]
    return {
        "cluster": cluster,
        "selected_count": len(rows),
        "completed_count": len(predictions),
        "metrics": metrics,
        "failures": failures,
    }


def prediction_to_dict(pred: VQAPrediction) -> dict[str, Any]:
    return {
        "sample_id": pred.sample_id,
        "baseline": pred.baseline,
        "task_family": pred.task_family,
        "video_id": pred.video_id,
        "question": pred.question,
        "choices": pred.choices,
        "gold": pred.gold,
        "prediction": pred.prediction,
        "correct": pred.correct,
        "evidence_ids": pred.evidence_ids,
        "tool_calls": pred.tool_calls,
        "failure_type": pred.failure_type,
        "attempt_count": pred.attempt_count,
        "reuse_memory_count": pred.reuse_memory_count,
        "raw_revisit_count": pred.raw_revisit_count,
        "structured_query_count": pred.structured_query_count,
        "session_video_position": pred.session_video_position,
        "relation_reuse_count": pred.relation_reuse_count,
        "planner_override_count": pred.planner_override_count,
        "verifier_blocked_finish_count": pred.verifier_blocked_finish_count,
        "tool_failure_count": pred.tool_failure_count,
        "ineffective_tool_count": pred.ineffective_tool_count,
        "failed_tool_recovery_count": pred.failed_tool_recovery_count,
        "ineffective_tool_avoidance_count": pred.ineffective_tool_avoidance_count,
        "tool_call_count": pred.tool_call_count,
        "reasoning_step_count": pred.reasoning_step_count,
        "elapsed_seconds": pred.elapsed_seconds,
        "prompt_tokens": pred.prompt_tokens,
        "completion_tokens": pred.completion_tokens,
        "total_tokens": pred.total_tokens,
        "estimated_cost": pred.estimated_cost,
    }


if __name__ == "__main__":
    raise SystemExit(main())
