#!/usr/bin/env python3
"""Run one video's VQA questions with per-family cap and token/storage reporting."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.config import load_env_file
from food_agent.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "video_token_probe")
    parser.add_argument("--video-id", default="", help="If omitted, auto-pick the video with the widest task-family coverage.")
    parser.add_argument("--per-family-limit", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    df = pd.read_parquet(args.index_file)
    video_id = args.video_id or auto_pick_video(df)
    selected = select_video_rows(df=df, video_id=video_id, per_family_limit=args.per_family_limit)
    run_name = build_run_name(video_id=video_id, per_family_limit=args.per_family_limit, suffix=args.run_suffix)
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress.json"

    completed_ids = load_completed_ids(predictions_path) if args.resume else set()
    if completed_ids:
        print(f"[resume] loaded {len(completed_ids)} completed samples from {predictions_path}", flush=True)

    agent = GraphAgent(paths=paths)
    session_state_path = paths.graph_agent_sessions_root / video_id / "session_state.json"
    selected_rows = selected.to_dict("records")
    total = len(selected_rows)
    results: list[dict[str, Any]] = load_existing_results(predictions_path) if args.resume else []
    result_by_id = {str(item["vqa_id"]): item for item in results if isinstance(item, dict) and item.get("vqa_id")}

    print(
        f"[video] video_id={video_id} total_selected={total} families={selected['task_family'].nunique()} per_family_limit={args.per_family_limit}",
        flush=True,
    )
    print_family_plan(selected)

    for index, row in enumerate(selected_rows, start=1):
        vqa_id = str(row["vqa_id"])
        if vqa_id in completed_ids:
            cached = result_by_id.get(vqa_id, {})
            print_progress(index=index, total=total, payload=cached, skipped=True)
            continue
        try:
            result = agent.answer_vqa_row(row, max_steps=args.max_steps)
            session_metrics = collect_session_metrics(session_state_path)
            payload = result.to_dict(gold=int(row["correct_idx"]), include_row=row)
            payload.update(
                {
                    "prompt_tokens": float((result.usage or {}).get("prompt_tokens") or 0.0),
                    "completion_tokens": float((result.usage or {}).get("completion_tokens") or 0.0),
                    "total_tokens": float((result.usage or {}).get("total_tokens") or 0.0),
                    "estimated_cost": float((result.usage or {}).get("estimated_cost") or 0.0),
                    "session_state_path": session_state_path.as_posix() if session_state_path.exists() else None,
                    "session_state_tokens_est": session_metrics["session_state_tokens_est"],
                    "session_memory_tokens_est": session_metrics["session_memory_tokens_est"],
                    "session_question_count": session_metrics["question_count"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            session_metrics = collect_session_metrics(session_state_path)
            payload = {
                "vqa_id": vqa_id,
                "video_id": row["primary_video_id"],
                "task_family": row["task_family"],
                "prediction": None,
                "gold": int(row["correct_idx"]),
                "correct": False,
                "answer_text": "",
                "confidence": 0.0,
                "elapsed_seconds": None,
                "tool_trace": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "evidence_bundle": [],
                "working_memory": [],
                "retrieved_frames": [],
                "visited_times": [],
                "artifacts": [],
                "verification_history": [],
                "tool_failures": [],
                "ineffective_tools": [],
                "open_questions": [],
                "raw_model_output": "",
                "question": row.get("question"),
                "choices_json": row.get("choices_json"),
                "inputs_json": row.get("inputs_json"),
                "failure_type": f"agent_error:{type(exc).__name__}",
                "failure_message": str(exc),
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "estimated_cost": 0.0,
                "session_state_path": session_state_path.as_posix() if session_state_path.exists() else None,
                "session_state_tokens_est": session_metrics["session_state_tokens_est"],
                "session_memory_tokens_est": session_metrics["session_memory_tokens_est"],
                "session_question_count": session_metrics["question_count"],
            }
        append_jsonl(predictions_path, payload)
        result_by_id[vqa_id] = payload
        results = [result_by_id[str(item["vqa_id"])] for item in selected_rows if str(item["vqa_id"]) in result_by_id]
        summary = build_summary(video_id=video_id, selected=selected, results=results)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        progress_path.write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "completed": len(results),
                    "total": total,
                    "remaining": total - len(results),
                    "prediction_path": predictions_path.as_posix(),
                    "summary_path": summary_path.as_posix(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print_progress(index=index, total=total, payload=payload, skipped=False, cumulative=summary)

    summary = build_summary(video_id=video_id, selected=selected, results=[result_by_id[str(item["vqa_id"])] for item in selected_rows if str(item["vqa_id"]) in result_by_id])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def auto_pick_video(df: pd.DataFrame) -> str:
    summary = (
        df.groupby("primary_video_id")
        .agg(total=("vqa_id", "count"), families=("task_family", "nunique"))
        .reset_index()
        .sort_values(["families", "total", "primary_video_id"], ascending=[False, False, True])
    )
    if summary.empty:
        raise RuntimeError("No VQA samples found in index file.")
    return str(summary.iloc[0]["primary_video_id"])


def select_video_rows(*, df: pd.DataFrame, video_id: str, per_family_limit: int) -> pd.DataFrame:
    video_df = df[df["primary_video_id"] == video_id].copy()
    if video_df.empty:
        raise RuntimeError(f"video_id not found in VQA index: {video_id}")
    video_df = video_df.sort_values(["task_family", "vqa_id"]).copy()
    selected = video_df.groupby("task_family", group_keys=False).head(per_family_limit).copy()
    selected = selected.sort_values(["task_family", "vqa_id"]).reset_index(drop=True)
    return selected


def build_run_name(*, video_id: str, per_family_limit: int, suffix: str) -> str:
    base = f"{video_id}_familycap{per_family_limit}"
    if suffix:
        return f"{base}_{suffix}"
    return base


def print_family_plan(selected: pd.DataFrame) -> None:
    counts = selected["task_family"].value_counts().sort_index()
    print("[family-plan]", flush=True)
    for family, count in counts.items():
        print(f"  {family}: {int(count)}", flush=True)


def load_completed_ids(path: Path) -> set[str]:
    return {str(item["vqa_id"]) for item in load_existing_results(path) if isinstance(item, dict) and item.get("vqa_id")}


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def collect_session_metrics(session_state_path: Path) -> dict[str, Any]:
    if not session_state_path.exists():
        return {
            "session_state_tokens_est": 0,
            "session_memory_tokens_est": 0,
            "question_count": 0,
        }
    try:
        payload = json.loads(session_state_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {
            "session_state_tokens_est": 0,
            "session_memory_tokens_est": 0,
            "question_count": 0,
        }
    session_memory = payload.get("session_memory")
    return {
        "session_state_tokens_est": estimate_json_tokens(payload),
        "session_memory_tokens_est": estimate_json_tokens(session_memory),
        "question_count": int(payload.get("question_count") or 0),
    }


def estimate_json_tokens(payload: Any) -> int:
    if payload is None:
        return 0
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # Fallback heuristic without tiktoken: close enough for relative storage growth tracking.
    return int(math.ceil(len(text) / 4.0))


def build_summary(*, video_id: str, selected: pd.DataFrame, results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_counts = {str(k): int(v) for k, v in selected["task_family"].value_counts().sort_index().items()}
    completed = len(results)
    correct = sum(1 for item in results if bool(item.get("correct")))
    family_metrics: dict[str, dict[str, Any]] = {}
    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        family = str(item.get("task_family") or "")
        rows_by_family.setdefault(family, []).append(item)
    for family, rows in sorted(rows_by_family.items()):
        count = len(rows)
        family_metrics[family] = {
            "count": count,
            "correct": sum(1 for item in rows if bool(item.get("correct"))),
            "accuracy": (sum(1 for item in rows if bool(item.get("correct"))) / count) if count else 0.0,
            "avg_prompt_tokens": avg(rows, "prompt_tokens"),
            "avg_completion_tokens": avg(rows, "completion_tokens"),
            "avg_total_tokens": avg(rows, "total_tokens"),
            "avg_estimated_cost": avg(rows, "estimated_cost"),
            "avg_session_state_tokens_est": avg(rows, "session_state_tokens_est"),
            "avg_session_memory_tokens_est": avg(rows, "session_memory_tokens_est"),
        }
    failure_summary = Counter(str(item.get("failure_type") or "none") for item in results)
    return {
        "video_id": video_id,
        "selected_question_count": int(len(selected)),
        "completed_question_count": completed,
        "correct": correct,
        "accuracy": (correct / completed) if completed else 0.0,
        "selected_by_task_family": selected_counts,
        "avg_prompt_tokens": avg(results, "prompt_tokens"),
        "avg_completion_tokens": avg(results, "completion_tokens"),
        "avg_total_tokens": avg(results, "total_tokens"),
        "avg_estimated_cost": avg(results, "estimated_cost"),
        "avg_session_state_tokens_est": avg(results, "session_state_tokens_est"),
        "avg_session_memory_tokens_est": avg(results, "session_memory_tokens_est"),
        "total_prompt_tokens": sum(float(item.get("prompt_tokens") or 0.0) for item in results),
        "total_completion_tokens": sum(float(item.get("completion_tokens") or 0.0) for item in results),
        "total_tokens": sum(float(item.get("total_tokens") or 0.0) for item in results),
        "total_estimated_cost": sum(float(item.get("estimated_cost") or 0.0) for item in results),
        "max_session_state_tokens_est": max((int(item.get("session_state_tokens_est") or 0) for item in results), default=0),
        "max_session_memory_tokens_est": max((int(item.get("session_memory_tokens_est") or 0) for item in results), default=0),
        "failure_summary": dict(sorted(failure_summary.items())),
        "by_task_family": family_metrics,
    }


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(item.get(key) or 0.0) for item in rows) / len(rows)


def print_progress(*, index: int, total: int, payload: dict[str, Any], skipped: bool, cumulative: dict[str, Any] | None = None) -> None:
    status = "skip" if skipped else "done"
    line = (
        f"[{index}/{total}] {status} sample={payload.get('vqa_id')} family={payload.get('task_family')} "
        f"pred={payload.get('prediction')} gold={payload.get('gold')} correct={payload.get('correct')} "
        f"prompt={int(float(payload.get('prompt_tokens') or 0.0))} "
        f"completion={int(float(payload.get('completion_tokens') or 0.0))} "
        f"total={int(float(payload.get('total_tokens') or 0.0))} "
        f"stored_state_est={int(payload.get('session_state_tokens_est') or 0)} "
        f"stored_mem_est={int(payload.get('session_memory_tokens_est') or 0)} "
        f"failure={payload.get('failure_type')}"
    )
    if cumulative:
        line += (
            f" cum_acc={int(cumulative.get('correct') or 0)}/{int(cumulative.get('completed_question_count') or 0)}"
            f"={float(cumulative.get('accuracy') or 0.0):.3f}"
            f" cum_tokens={int(float(cumulative.get('total_tokens') or 0.0))}"
            f" cum_cost={float(cumulative.get('total_estimated_cost') or 0.0):.6f}"
        )
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
