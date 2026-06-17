#!/usr/bin/env python3
"""Run a small real fine-grained why eval and emit error attribution artifacts."""

from __future__ import annotations

import argparse
import json
import os
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


CLUSTERS: tuple[str, ...] = (
    "towel-cluster",
    "access-space-cluster",
    "future-use-cluster",
    "state-change-cluster",
)

CLUSTER_SPECS: dict[str, dict[str, tuple[str, ...] | str]] = {
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
            "free the",
        ),
    },
    "future-use-cluster": {
        "combined_any": ("weigh", "measure", "pour", "empty", "serve", "check", "fill", "wash", "clean", "dry"),
    },
    "state-change-cluster": {
        "question_any": ("<tap ", "<press ", "<push ", "<switch ", "<turn "),
        "combined_any": ("scale", "display", "tap", "press", "push", "switch", "turn on", "turn off"),
    },
}


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=defaults.output_root / "results" / "graph_agent_small_real_why_eval_evidence_sufficiency",
    )
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    paths = ProjectPaths.from_env()
    agent = GraphAgent(paths=paths)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "selection.json"
    predictions_path = out_dir / "predictions_graph_agent.jsonl"
    summary_path = out_dir / "summary.json"
    error_analysis_path = out_dir / "error_analysis.md"

    rows = load_or_build_selection(index_file=args.index_file, limit=args.limit, selection_path=selection_path, resume=args.resume)
    completed = load_predictions(predictions_path) if args.resume else {}
    ordered: list[dict[str, Any]] = [completed[str(row["vqa_id"])] for row in rows if str(row["vqa_id"]) in completed]

    heartbeat_path = out_dir / "heartbeat.jsonl"
    os.environ["FOOD_AGENT_HEARTBEAT_PATH"] = heartbeat_path.as_posix()

    for index, row in enumerate(rows, start=1):
        sample_id = str(row["vqa_id"])
        if sample_id in completed:
            payload = completed[sample_id]
            print(
                f"[{index}/{len(rows)}] skip sample={sample_id} pred={payload.get('prediction')} gold={payload.get('gold')} "
                f"correct={payload.get('correct')} cluster={payload.get('cluster')}",
                flush=True,
            )
            continue
        try:
            result = agent.answer_vqa_row(row, max_steps=args.max_steps)
            payload = result.to_dict(gold=int(row["correct_idx"]), include_row=row)
            payload["failure_type"] = None if payload.get("correct") else "reasoning_error"
        except Exception as exc:  # noqa: BLE001
            payload = {
                "vqa_id": sample_id,
                "video_id": row.get("primary_video_id"),
                "task_family": row.get("task_family"),
                "prediction": None,
                "gold": int(row["correct_idx"]),
                "correct": False,
                "answer_text": "",
                "confidence": 0.0,
                "elapsed_seconds": None,
                "usage": {},
                "tool_trace": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "evidence_bundle": [],
                "working_memory": [],
                "retrieved_frames": [],
                "visited_times": [],
                "artifacts": [],
                "verification_history": [],
                "latest_verification": {},
                "tool_failures": [],
                "ineffective_tools": [],
                "open_questions": [],
                "question": row.get("question"),
                "choices_json": row.get("choices_json"),
                "inputs_json": row.get("inputs_json"),
                "failure_type": f"runner_error:{type(exc).__name__}",
                "failure_message": str(exc),
            }
        payload["cluster"] = str(row.get("cluster") or "")
        payload["sample_id"] = sample_id
        append_jsonl(predictions_path, payload)
        completed[sample_id] = payload
        ordered = [completed[str(item["vqa_id"])] for item in rows if str(item["vqa_id"]) in completed]
        print(
            f"[{index}/{len(rows)}] sample={sample_id} pred={payload.get('prediction')} gold={payload.get('gold')} "
            f"correct={payload.get('correct')} cluster={payload.get('cluster')} "
            f"finish_mode={((payload.get('latest_verification') or {}).get('sufficiency_decision') or {}).get('finish_mode')} "
            f"gaps={','.join(extract_gap_types(payload)) or 'none'}",
            flush=True,
        )
        summary = build_summary(rows=rows, predictions=ordered)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        error_analysis_path.write_text(build_error_analysis(rows=rows, predictions=ordered, summary=summary), encoding="utf-8")

    summary = build_summary(rows=rows, predictions=ordered)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    error_analysis_path.write_text(build_error_analysis(rows=rows, predictions=ordered, summary=summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_or_build_selection(*, index_file: Path, limit: int, selection_path: Path, resume: bool) -> list[dict[str, Any]]:
    if resume and selection_path.exists():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
    rows = build_selection(index_file=index_file, limit=limit)
    selection_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def build_selection(*, index_file: Path, limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(index_file)
    subset = df[df["task_family"] == "fine_grained_why_recognition"].copy()
    picked: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for cluster in CLUSTERS:
        cluster_rows = subset[subset.apply(lambda row: row_matches_cluster(dict(row), cluster=cluster), axis=1)].copy()
        cluster_rows = cluster_rows.sort_values(["primary_video_id", "vqa_id"])
        for record in cluster_rows.to_dict("records"):
            normalized = normalize_row(record)
            sample_id = str(normalized["vqa_id"])
            if sample_id in used_ids:
                continue
            normalized["cluster"] = cluster
            picked.append(normalized)
            used_ids.add(sample_id)
            break
    if len(picked) < limit:
        fallback = subset.sort_values(["primary_video_id", "vqa_id"]).to_dict("records")
        for record in fallback:
            normalized = normalize_row(record)
            sample_id = str(normalized["vqa_id"])
            if sample_id in used_ids:
                continue
            normalized["cluster"] = "fallback"
            picked.append(normalized)
            used_ids.add(sample_id)
            if len(picked) >= limit:
                break
    return picked[:limit]


def row_matches_cluster(row: dict[str, Any], *, cluster: str) -> bool:
    spec = CLUSTER_SPECS[cluster]
    question = str(row.get("question") or "").lower()
    choices = decode_choices(row.get("choices_json"))
    choices_text = " ".join(choices).lower()
    combined_text = f"{question} {choices_text}".strip()
    question_any = tuple(str(item).lower() for item in spec.get("question_any", ()))  # type: ignore[arg-type]
    choice_any = tuple(str(item).lower() for item in spec.get("choice_any", ()))  # type: ignore[arg-type]
    combined_any = tuple(str(item).lower() for item in spec.get("combined_any", ()))  # type: ignore[arg-type]
    if question_any and not any(token in question for token in question_any):
        return False
    if choice_any and not any(token in choices_text for token in choice_any):
        return False
    if combined_any and not any(token in combined_text for token in combined_any):
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


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get("vqa_id"):
            completed[str(payload["vqa_id"])] = payload
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def extract_gap_types(payload: dict[str, Any]) -> list[str]:
    latest = payload.get("latest_verification") or {}
    gaps = latest.get("evidence_gaps") or []
    if isinstance(gaps, list):
        extracted = [str(item.get("gap_type") or "") for item in gaps if isinstance(item, dict) and item.get("gap_type")]
        if extracted:
            return extracted
    trace_summary = summarize_action_intent_trace(payload)
    trace_gap_type = str(trace_summary.get("final_primary_gap_type") or "").strip()
    return [trace_gap_type] if trace_gap_type else []


def extract_final_finish_reason(payload: dict[str, Any]) -> str:
    final_metadata = payload.get("final_metadata")
    if isinstance(final_metadata, dict):
        reason = str(final_metadata.get("finish_reason") or "").strip()
        if reason:
            return reason
    latest = payload.get("latest_verification") or {}
    sufficiency = latest.get("sufficiency_decision") or {}
    return str(sufficiency.get("finish_mode") or "").strip()


def extract_budget_summary(payload: dict[str, Any]) -> dict[str, Any]:
    budget = payload.get("search_budget")
    if isinstance(budget, dict) and budget:
        return budget
    final_metadata = payload.get("final_metadata")
    if isinstance(final_metadata, dict):
        used_budget = final_metadata.get("used_budget")
        if isinstance(used_budget, dict) and used_budget:
            return used_budget
    return {}


def extract_action_intent_trace(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trace = payload.get("action_intent_trace")
    if not isinstance(trace, list):
        return []
    return [item for item in trace if isinstance(item, dict)]


def summarize_action_intent_trace(payload: dict[str, Any]) -> dict[str, Any]:
    trace = extract_action_intent_trace(payload)
    if not trace:
        return {}
    initial = trace[0]
    final = trace[-1]
    gap_change_count = 0
    action_change_count = 0
    primary_gap_recovery_trace = ""
    primary_gap_type = str(((initial.get("primary_gap") or {}).get("gap_type")) or "").strip()
    initial_next_action = str(initial.get("recommended_next_action") or "").strip()
    final_next_action = initial_next_action
    previous_gap_type = primary_gap_type
    previous_next_action = initial_next_action
    for item in trace:
        primary_gap = item.get("primary_gap") or {}
        current_gap_type = str(primary_gap.get("gap_type") or "").strip()
        if current_gap_type:
            if not primary_gap_type:
                primary_gap_type = current_gap_type
            if previous_gap_type and current_gap_type != previous_gap_type:
                gap_change_count += 1
            previous_gap_type = current_gap_type
        if not primary_gap_recovery_trace:
            primary_gap_recovery_trace = str(item.get("primary_gap_recovery_trace") or "").strip()
        current_next_action = str(item.get("recommended_next_action") or "").strip()
        if current_next_action:
            final_next_action = current_next_action
            if previous_next_action and current_next_action != previous_next_action:
                action_change_count += 1
            previous_next_action = current_next_action
    final_primary_gap_recovery_trace = str(final.get("primary_gap_recovery_trace") or "").strip()
    if final_primary_gap_recovery_trace:
        primary_gap_recovery_trace = final_primary_gap_recovery_trace
    final_primary_gap = final.get("primary_gap") or {}
    final_primary_gap_type = str(final_primary_gap.get("gap_type") or "").strip()
    if final_primary_gap_type:
        primary_gap_type = final_primary_gap_type
    return {
        "trace_steps": len(trace),
        "initial_primary_gap_type": str(((initial.get("primary_gap") or {}).get("gap_type")) or "").strip(),
        "final_primary_gap_type": primary_gap_type,
        "primary_gap_change_count": gap_change_count,
        "initial_recommended_next_action": initial_next_action,
        "final_recommended_next_action": final_next_action,
        "recommended_next_action_change_count": action_change_count,
        "final_primary_gap_type": primary_gap_type,
        "final_primary_gap_recovery_trace": primary_gap_recovery_trace,
        "final_finish_mode": str(final.get("finish_mode") or ""),
    }


def build_summary(*, rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = len(predictions)
    vqa_rows = [row for row in predictions if row.get("gold") is not None]
    correct = sum(1 for row in vqa_rows if row.get("correct") is True)
    elapsed_values = [float(row.get("elapsed_seconds")) for row in predictions if row.get("elapsed_seconds") is not None]
    tool_counts = [len(row.get("tool_calls") or []) for row in predictions]
    prompt_tokens = sum(float((row.get("usage") or {}).get("prompt_tokens") or 0.0) for row in predictions)
    completion_tokens = sum(float((row.get("usage") or {}).get("completion_tokens") or 0.0) for row in predictions)
    total_tokens = sum(float((row.get("usage") or {}).get("total_tokens") or 0.0) for row in predictions)
    estimated_cost = sum(float((row.get("usage") or {}).get("estimated_cost") or 0.0) for row in predictions)
    failure_counts = Counter(str(row.get("failure_type") or "none") for row in predictions)
    gap_counts = Counter()
    finish_mode_counts = Counter()
    cluster_counts = Counter()
    budget_exhausted_count = 0
    avg_new_frames = 0.0
    avg_long_horizon_expansions = 0.0
    action_intent_trace_count = 0
    total_trace_steps = 0.0
    primary_gap_change_count = 0
    recommended_next_action_change_count = 0
    primary_gap_type_counts = Counter()
    primary_gap_trace_rows = 0
    primary_gap_trace_counts = Counter()
    budget_rows = 0
    total_new_frames = 0.0
    total_long_horizon_expansions = 0.0
    for row in predictions:
        cluster_counts[str(row.get("cluster") or "unknown")] += 1
        finish_reason = extract_final_finish_reason(row)
        if finish_reason:
            finish_mode_counts[finish_reason] += 1
        if finish_reason == "finish_budget_exhausted_best_guess":
            budget_exhausted_count += 1
        for gap_type in extract_gap_types(row):
            gap_counts[gap_type] += 1
        budget = extract_budget_summary(row)
        if isinstance(budget, dict):
            budget_rows += 1
            total_new_frames += float(budget.get("new_frames_observed") or 0.0)
            total_long_horizon_expansions += float(budget.get("long_horizon_expansions_used") or 0.0)
        trace_summary = summarize_action_intent_trace(row)
        if trace_summary:
            action_intent_trace_count += 1
            total_trace_steps += float(trace_summary.get("trace_steps") or 0.0)
            primary_gap_change_count += int(trace_summary.get("primary_gap_change_count") or 0)
            recommended_next_action_change_count += int(trace_summary.get("recommended_next_action_change_count") or 0)
            final_primary_gap_type = str(trace_summary.get("final_primary_gap_type") or "").strip()
            if final_primary_gap_type:
                primary_gap_type_counts[final_primary_gap_type] += 1
            final_primary_gap_recovery_trace = str(trace_summary.get("final_primary_gap_recovery_trace") or "").strip()
            if final_primary_gap_recovery_trace:
                primary_gap_trace_rows += 1
                primary_gap_trace_counts[final_primary_gap_recovery_trace] += 1
    if budget_rows:
        avg_new_frames = total_new_frames / budget_rows
        avg_long_horizon_expansions = total_long_horizon_expansions / budget_rows
    avg_trace_steps = (total_trace_steps / action_intent_trace_count) if action_intent_trace_count else 0.0
    return {
        "task_family": "fine_grained_why_recognition",
        "selection_count": total,
        "completed_count": completed,
        "correct_count": correct,
        "accuracy": (correct / len(vqa_rows)) if vqa_rows else None,
        "avg_elapsed_seconds": (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0.0,
        "max_elapsed_seconds": max(elapsed_values) if elapsed_values else 0.0,
        "avg_tool_call_count": (sum(tool_counts) / len(tool_counts)) if tool_counts else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "failure_counts": dict(failure_counts),
        "gap_type_counts": dict(gap_counts),
        "finish_mode_counts": dict(finish_mode_counts),
        "budget_exhausted_count": budget_exhausted_count,
        "avg_new_frames_observed": avg_new_frames,
        "avg_long_horizon_expansions": avg_long_horizon_expansions,
        "action_intent_trace_count": action_intent_trace_count,
        "avg_trace_steps": avg_trace_steps,
        "primary_gap_change_count": primary_gap_change_count,
        "recommended_next_action_change_count": recommended_next_action_change_count,
        "trace_primary_gap_type_counts": dict(primary_gap_type_counts),
        "primary_gap_recovery_trace_count": primary_gap_trace_rows,
        "primary_gap_recovery_trace_counts": dict(primary_gap_trace_counts),
        "cluster_counts": dict(cluster_counts),
        "prediction_path": (Path.cwd() / "predictions_graph_agent.jsonl").as_posix() if False else "",
    }


def build_error_analysis(*, rows: list[dict[str, Any]], predictions: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GraphAgent Why 小样本真实验证与误差归因")
    lines.append("")
    lines.append("## 1. 评测设置")
    lines.append("")
    lines.append("- 任务：`fine_grained_why_recognition`")
    lines.append(f"- 样本数：`{len(rows)}`")
    lines.append("- 目标：验证 `EvidenceGap + SufficiencyDecision + gap 路由` 是否在真实链路中生效")
    lines.append("")
    lines.append("覆盖 cluster：")
    lines.append("")
    for cluster, count in sorted((summary.get("cluster_counts") or {}).items()):
        lines.append(f"- `{cluster}`: {count}")
    lines.append("")
    lines.append("## 2. 总体结果")
    lines.append("")
    accuracy = summary.get("accuracy")
    accuracy_text = "None" if accuracy is None else f"{accuracy * 100:.2f}%"
    lines.append(f"- 正确率：`{summary.get('correct_count', 0)} / {len(rows)} = {accuracy_text}`")
    lines.append(f"- 平均耗时：`{float(summary.get('avg_elapsed_seconds') or 0.0):.2f}s / 题`")
    lines.append(f"- 最长耗时：`{float(summary.get('max_elapsed_seconds') or 0.0):.2f}s`")
    lines.append(f"- 平均工具调用数：`{float(summary.get('avg_tool_call_count') or 0.0):.2f}`")
    lines.append(f"- 总输入 tokens：`{int(float(summary.get('prompt_tokens') or 0.0)):,}`")
    lines.append(f"- 总输出 tokens：`{int(float(summary.get('completion_tokens') or 0.0)):,}`")
    lines.append(f"- 总 tokens：`{int(float(summary.get('total_tokens') or 0.0)):,}`")
    lines.append("")
    lines.append("Gap / Finish 统计：")
    lines.append("")
    lines.append(f"- gap_type_counts：`{json.dumps(summary.get('gap_type_counts') or {}, ensure_ascii=False)}`")
    lines.append(f"- finish_mode_counts：`{json.dumps(summary.get('finish_mode_counts') or {}, ensure_ascii=False)}`")
    lines.append(f"- budget_exhausted_count：`{summary.get('budget_exhausted_count', 0)}`")
    lines.append(f"- avg_new_frames_observed：`{float(summary.get('avg_new_frames_observed') or 0.0):.2f}`")
    lines.append(f"- avg_long_horizon_expansions：`{float(summary.get('avg_long_horizon_expansions') or 0.0):.2f}`")
    lines.append(f"- action_intent_trace_count：`{int(summary.get('action_intent_trace_count') or 0)}`")
    lines.append(f"- avg_trace_steps：`{float(summary.get('avg_trace_steps') or 0.0):.2f}`")
    lines.append(f"- primary_gap_change_count：`{int(summary.get('primary_gap_change_count') or 0)}`")
    lines.append(f"- recommended_next_action_change_count：`{int(summary.get('recommended_next_action_change_count') or 0)}`")
    lines.append(f"- trace_primary_gap_type_counts：`{json.dumps(summary.get('trace_primary_gap_type_counts') or {}, ensure_ascii=False)}`")
    lines.append(f"- primary_gap_recovery_trace_count：`{int(summary.get('primary_gap_recovery_trace_count') or 0)}`")
    lines.append(f"- primary_gap_recovery_trace_counts：`{json.dumps(summary.get('primary_gap_recovery_trace_counts') or {}, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## 3. 分样本结果")
    lines.append("")
    for row in predictions:
        latest = row.get("latest_verification") or {}
        sufficiency = latest.get("sufficiency_decision") or {}
        final_finish_reason = extract_final_finish_reason(row)
        gap_types = extract_gap_types(row)
        trace_summary = summarize_action_intent_trace(row)
        lines.append(f"### {row.get('vqa_id')}")
        lines.append("")
        lines.append(f"- cluster: `{row.get('cluster')}`")
        lines.append(f"- 结果: `pred={row.get('prediction')} gold={row.get('gold')} correct={row.get('correct')}`")
        lines.append(f"- failure_type: `{row.get('failure_type')}`")
        lines.append(f"- elapsed_seconds: `{row.get('elapsed_seconds')}`")
        lines.append(f"- tool_calls: `{row.get('tool_calls') or []}`")
        lines.append(f"- gap_types: `{gap_types}`")
        lines.append(f"- finish_mode: `{sufficiency.get('finish_mode')}`")
        lines.append(f"- final_finish_reason: `{final_finish_reason}`")
        lines.append(f"- recommended_next_step: `{sufficiency.get('recommended_next_step')}`")
        lines.append(f"- latest_missing: `{latest.get('missing_evidence_types') or []}`")
        lines.append(f"- search_budget: `{extract_budget_summary(row)}`")
        if trace_summary:
            lines.append(
                f"- observation_trace: `steps={trace_summary.get('trace_steps')} "
                f"gap {trace_summary.get('initial_primary_gap_type')}->{trace_summary.get('final_primary_gap_type')} "
                f"next_action {trace_summary.get('initial_recommended_next_action')}->{trace_summary.get('final_recommended_next_action')} "
                f"gap_changes={trace_summary.get('primary_gap_change_count')} "
                f"action_changes={trace_summary.get('recommended_next_action_change_count')}`"
            )
            lines.append(f"- final_primary_gap_type: `{trace_summary.get('final_primary_gap_type')}`")
            lines.append(f"- final_primary_gap_recovery_trace: `{trace_summary.get('final_primary_gap_recovery_trace')}`")
        lines.append("")
    reasoning_failures = [row for row in predictions if row.get("failure_type") == "reasoning_error"]
    runtime_failures = [row for row in predictions if str(row.get("failure_type") or "").startswith("runner_error:")]
    lines.append("## 4. 误差归因")
    lines.append("")
    lines.append(f"- 逻辑错答数：`{len(reasoning_failures)}`")
    lines.append(f"- 运行时失败数：`{len(runtime_failures)}`")
    lines.append("")
    if runtime_failures:
        lines.append("运行时失败样本：")
        lines.append("")
        for row in runtime_failures:
            lines.append(f"- `{row.get('vqa_id')}`: `{row.get('failure_type')}`")
        lines.append("")
    if reasoning_failures:
        lines.append("逻辑错答样本：")
        lines.append("")
        for row in reasoning_failures:
            lines.append(
                f"- `{row.get('vqa_id')}`: gap=`{extract_gap_types(row)}` finish=`{extract_final_finish_reason(row)}`"
            )
        lines.append("")
    lines.append("## 5. 本轮结论")
    lines.append("")
    lines.append("- 这轮重点不是追求大样本分数，而是确认通用 `gap-driven` 机制已经进入真实执行链。")
    lines.append("- 如果 `gap_type_counts`、`finish_mode_counts`、`search_budget`、`action_intent_trace` 在真实样本中非空，说明 observation-centric agent 的关键状态已经被真实写出。")
    lines.append("- 如果 `action_intent_trace_count`、gap 演化摘要和 `primary_gap_recovery_trace` 在报告中非空，说明 agent 已经能把“为什么继续搜、观测缺口如何演化”写进真实误差分析。")
    lines.append("- 如果工具链里出现 `query_object / query_spatial_context / extract_frames_for_range` 等与 gap 对应的动作，说明 planner 主路由已经开始生效。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
