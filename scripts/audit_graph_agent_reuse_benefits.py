#!/usr/bin/env python3
"""Audit real reuse benefits for graph/session/artifact memory in graph-agent outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


RAW_REVISIT_TOOLS = {
    "extract_frame_at_time",
    "extract_frames_for_range",
    "sample_sparse_frames",
    "sample_frames_around_peaks",
    "extract_input_reference_frames",
    "render_bbox_overlay",
    "extract_region_with_context",
    "run_ocr_on_image",
    "run_ocr_on_region",
    "detect_audio_peaks",
}


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=defaults.output_root / "results")
    parser.add_argument("--graph-agent-runs-root", type=Path, default=defaults.output_root / "graph_agent_runs")
    parser.add_argument("--report-path", type=Path, default=defaults.output_root / "reports" / "graph_agent_reuse_benefit_audit.json")
    parser.add_argument("--example-limit", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_prediction_records(args.results_root)
    run_records = load_graph_run_records(args.graph_agent_runs_root)
    report = build_reuse_audit_report(predictions=predictions, run_records=run_records, example_limit=args.example_limit)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def load_prediction_records(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("predictions_graph_agent.jsonl")):
        run_name = path.parent.name
        per_video_position: Counter[str] = Counter()
        for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            video_id = str(payload.get("video_id") or "")
            if video_id:
                per_video_position[video_id] += 1
            payload.setdefault("run_name", run_name)
            payload.setdefault("source_path", path.as_posix())
            payload.setdefault("line_index", line_index)
            payload.setdefault("inferred_session_video_position", per_video_position.get(video_id, 0))
            rows.append(payload)
    return rows


def load_graph_run_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"task_compress.json", "task_restore.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        payload.setdefault("source_path", path.as_posix())
        rows.append(payload)
    return rows


def build_reuse_audit_report(
    *,
    predictions: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
    example_limit: int,
) -> dict[str, Any]:
    indexed_runs = index_by_vqa_id(run_records)
    prediction_reuse = summarize_prediction_reuse(predictions=predictions, indexed_runs=indexed_runs, example_limit=example_limit)
    run_record_reuse = summarize_run_record_reuse(run_records=run_records, example_limit=example_limit)
    follow_up = summarize_follow_up_cost_reduction(predictions)
    return {
        "prediction_record_count": len(predictions),
        "graph_run_record_count": len(run_records),
        "reuse_items": prediction_reuse,
        "graph_seed_and_cached_artifact_hits": run_record_reuse,
        "same_video_follow_up_cost_reduction": follow_up,
        "audit_notes": [
            "复用收益主要从真实 predictions_graph_agent.jsonl 读取；老 session_state 产物字段较轻，仅作辅助，不作为主统计来源。",
            "graph_agent_runs/*.json 用于补充 cached artifact 写回、artifact 节点 seed、真实 evidence/artifact 样例。",
            "同视频 follow-up 收益按 session_video_position 比较 first-question 与 follow-up 间 raw revisit、structured query、tool calls、latency 变化。",
        ],
    }


def summarize_prediction_reuse(
    *,
    predictions: list[dict[str, Any]],
    indexed_runs: dict[str, dict[str, Any]],
    example_limit: int,
) -> dict[str, Any]:
    reuse_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    artifact_counter: Counter[str] = Counter()
    session_reuse_examples: list[dict[str, Any]] = []
    graph_seed_examples: list[dict[str, Any]] = []
    cached_artifact_examples: list[dict[str, Any]] = []
    session_reuse_hit_count = 0
    graph_seed_hit_count = 0
    cached_artifact_hit_count = 0
    cached_artifact_tool_hit_count = 0

    for payload in predictions:
        working_memory = ensure_list(payload.get("working_memory"))
        artifacts = ensure_list(payload.get("artifacts"))
        tool_calls = ensure_list(payload.get("tool_calls"))
        vqa_id = str(payload.get("sample_id") or payload.get("vqa_id") or "")
        run_payload = indexed_runs.get(vqa_id, {})

        reuse_items = [item for item in working_memory if isinstance(item, str) and item.startswith("reuse:")]
        relation_items = [item for item in working_memory if isinstance(item, str) and item.startswith("reuse_relation:")]
        cached_memory_items = [item for item in working_memory if isinstance(item, str) and item.startswith("reuse_cached_artifact=")]
        cached_tool_used = "retrieve_cached_artifacts" in tool_calls

        for item in reuse_items:
            reuse_counter[normalize_reuse_key(item)] += 1
        for item in relation_items:
            relation_counter[normalize_reuse_key(item)] += 1
        for item in artifacts:
            if isinstance(item, str) and item:
                artifact_counter[artifact_tag(item)] += 1

        session_position = session_position_of(payload)
        if payload.get("reuse_memory_count", len(reuse_items)) or session_position >= 2:
            session_reuse_hit_count += 1
            maybe_append(
                session_reuse_examples,
                {
                    "vqa_id": vqa_id,
                    "video_id": payload.get("video_id"),
                    "task_family": payload.get("task_family"),
                    "session_video_position": session_position,
                    "reuse_memory_count": payload.get("reuse_memory_count", len(reuse_items)),
                    "reuse_items": reuse_items[:6],
                    "tool_calls": tool_calls[:10],
                    "source_path": payload.get("source_path"),
                },
                limit=example_limit,
            )

        if relation_items:
            graph_seed_hit_count += 1
            maybe_append(
                graph_seed_examples,
                {
                    "vqa_id": vqa_id,
                    "video_id": payload.get("video_id"),
                    "task_family": payload.get("task_family"),
                    "relation_reuse_count": payload.get("relation_reuse_count", len(relation_items)),
                    "relation_reuse_items": relation_items[:6],
                    "reuse_items": reuse_items[:4],
                    "source_path": payload.get("source_path"),
                },
                limit=example_limit,
            )

        if cached_memory_items or cached_tool_used or has_cached_artifact_payload(run_payload):
            cached_artifact_hit_count += 1
            if cached_tool_used:
                cached_artifact_tool_hit_count += 1
            maybe_append(
                cached_artifact_examples,
                {
                    "vqa_id": vqa_id,
                    "video_id": payload.get("video_id"),
                    "task_family": payload.get("task_family"),
                    "cached_tool_used": cached_tool_used,
                    "cached_memory_items": cached_memory_items[:6],
                    "artifact_examples": artifacts[:6],
                    "run_artifact_examples": ensure_list(run_payload.get("artifacts"))[:6],
                    "source_path": payload.get("source_path"),
                },
                limit=example_limit,
            )

    return {
        "top_reuse_items": counter_to_ranked_list(reuse_counter, example_limit),
        "top_relation_reuse_items": counter_to_ranked_list(relation_counter, example_limit),
        "top_artifact_tags": counter_to_ranked_list(artifact_counter, example_limit),
        "session_reuse_hit_count": session_reuse_hit_count,
        "graph_seed_hit_count": graph_seed_hit_count,
        "cached_artifact_hit_count": cached_artifact_hit_count,
        "cached_artifact_tool_hit_count": cached_artifact_tool_hit_count,
        "session_reuse_examples": session_reuse_examples,
        "graph_seed_examples": graph_seed_examples,
        "cached_artifact_examples": cached_artifact_examples,
    }


def summarize_run_record_reuse(run_records: list[dict[str, Any]], example_limit: int) -> dict[str, Any]:
    cached_nodes = 0
    cached_seeded_records = 0
    artifact_tags: Counter[str] = Counter()
    node_source_counter: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for payload in run_records:
        artifacts = ensure_list(payload.get("artifacts"))
        working_memory = ensure_list(payload.get("working_memory"))
        evidence_bundle = ensure_list(payload.get("evidence_bundle"))
        tool_trace = ensure_list(payload.get("tool_trace"))

        if any(isinstance(item, str) and item.startswith("reuse_cached_artifact=") for item in working_memory):
            cached_seeded_records += 1
        if any(isinstance(entry, dict) and entry.get("tool") == "retrieve_cached_artifacts" for entry in tool_trace):
            cached_nodes += 1

        for item in artifacts:
            if isinstance(item, str) and item:
                artifact_tags[artifact_tag(item)] += 1
        for item in evidence_bundle:
            if not isinstance(item, str):
                continue
            if "source=cached_artifact_reuse" in item:
                node_source_counter["cached_artifact_reuse"] += 1
            if "source=session_memory_compressor" in item:
                node_source_counter["session_memory_compressor"] += 1
            if "source=agent_timeline_summary" in item:
                node_source_counter["agent_timeline_summary"] += 1

        if (
            artifacts
            or any(isinstance(item, str) and item.startswith("reuse_cached_artifact=") for item in working_memory)
            or any("source=cached_artifact_reuse" in str(item) for item in evidence_bundle)
        ):
            maybe_append(
                examples,
                {
                    "vqa_id": payload.get("vqa_id"),
                    "video_id": payload.get("video_id"),
                    "task_family": payload.get("task_family"),
                    "artifacts": artifacts[:6],
                    "cached_memory_items": [
                        item for item in working_memory if isinstance(item, str) and item.startswith("reuse_cached_artifact=")
                    ][:6],
                    "cached_evidence_items": [
                        item for item in evidence_bundle if isinstance(item, str) and "source=cached_artifact_reuse" in item
                    ][:6],
                    "source_path": payload.get("source_path"),
                },
                limit=example_limit,
            )

    return {
        "cached_artifact_tool_record_count": cached_nodes,
        "cached_artifact_seed_record_count": cached_seeded_records,
        "artifact_tag_counter": counter_to_ranked_list(artifact_tags, example_limit),
        "evidence_source_counter": counter_to_ranked_list(node_source_counter, example_limit),
        "examples": examples,
    }


def summarize_follow_up_cost_reduction(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in predictions:
        video_id = str(payload.get("video_id") or "")
        if not video_id:
            continue
        grouped[video_id].append(payload)

    comparable_video_count = 0
    first_raw: list[float] = []
    follow_raw: list[float] = []
    first_structured: list[float] = []
    follow_structured: list[float] = []
    first_tools: list[float] = []
    follow_tools: list[float] = []
    first_latency: list[float] = []
    follow_latency: list[float] = []
    examples: list[dict[str, Any]] = []

    for video_id, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda item: (
                session_position_of(item),
                _safe_int(item.get("line_index"), fallback=0),
                str(item.get("sample_id") or item.get("vqa_id") or ""),
            ),
        )
        first = [item for item in ordered if session_position_of(item) == 1]
        follow = [item for item in ordered if session_position_of(item) >= 2]
        if not first or not follow:
            continue
        first_row = first[0]
        comparable_video_count += 1

        first_raw.append(float(first_row.get("raw_revisit_count") or 0.0))
        follow_raw.append(mean([float(item.get("raw_revisit_count") or 0.0) for item in follow]))
        first_structured.append(float(first_row.get("structured_query_count") or 0.0))
        follow_structured.append(mean([float(item.get("structured_query_count") or 0.0) for item in follow]))
        first_tools.append(mean([float(len(ensure_list(first_row.get("tool_calls"))))]))
        follow_tools.append(mean([float(len(ensure_list(item.get("tool_calls")))) for item in follow]))
        first_latency.append(_safe_float(first_row.get("elapsed_seconds")))
        follow_latency.append(mean([_safe_float(item.get("elapsed_seconds")) for item in follow]))

        if len(examples) < 12:
            examples.append(
                {
                    "video_id": video_id,
                    "first_question": compact_follow_up_row(first_row),
                    "follow_up_mean": {
                        "question_count": len(follow),
                        "avg_raw_revisit_count": round(follow_raw[-1], 3),
                        "avg_structured_query_count": round(follow_structured[-1], 3),
                        "avg_tool_calls": round(follow_tools[-1], 3),
                        "avg_elapsed_seconds": round(follow_latency[-1], 3),
                        "avg_reuse_memory_count": round(mean([float(item.get("reuse_memory_count") or 0.0) for item in follow]), 3),
                    },
                }
            )

    return {
        "comparable_video_count": comparable_video_count,
        "first_question_means": {
            "avg_raw_revisit_count": round(mean(first_raw), 3),
            "avg_structured_query_count": round(mean(first_structured), 3),
            "avg_tool_calls": round(mean(first_tools), 3),
            "avg_elapsed_seconds": round(mean(first_latency), 3),
        },
        "follow_up_means": {
            "avg_raw_revisit_count": round(mean(follow_raw), 3),
            "avg_structured_query_count": round(mean(follow_structured), 3),
            "avg_tool_calls": round(mean(follow_tools), 3),
            "avg_elapsed_seconds": round(mean(follow_latency), 3),
        },
        "delta_follow_up_minus_first": {
            "raw_revisit_count": round(mean(follow_raw) - mean(first_raw), 3),
            "structured_query_count": round(mean(follow_structured) - mean(first_structured), 3),
            "tool_calls": round(mean(follow_tools) - mean(first_tools), 3),
            "elapsed_seconds": round(mean(follow_latency) - mean(first_latency), 3),
        },
        "examples": examples,
    }


def compact_follow_up_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "vqa_id": payload.get("sample_id") or payload.get("vqa_id"),
        "task_family": payload.get("task_family"),
        "session_video_position": session_position_of(payload),
        "raw_revisit_count": payload.get("raw_revisit_count", 0),
        "structured_query_count": payload.get("structured_query_count", 0),
        "tool_calls": len(ensure_list(payload.get("tool_calls"))),
        "elapsed_seconds": round(_safe_float(payload.get("elapsed_seconds")), 3),
        "reuse_memory_count": payload.get("reuse_memory_count", 0),
    }


def normalize_reuse_key(text: str) -> str:
    head = text.split(";", 1)[0].strip()
    if " time=" in head:
        head = head.split(" time=", 1)[0].strip()
    return head


def artifact_tag(path: str) -> str:
    stem = Path(path).stem
    lowered = stem.lower()
    if "_" in lowered:
        prefix = lowered.split("_", 1)[0]
        if prefix:
            return prefix
    return lowered or "unknown"


def has_cached_artifact_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    artifacts = ensure_list(payload.get("artifacts"))
    evidence_bundle = ensure_list(payload.get("evidence_bundle"))
    working_memory = ensure_list(payload.get("working_memory"))
    return bool(
        any(isinstance(item, str) and item for item in artifacts)
        or any(isinstance(item, str) and item.startswith("reuse_cached_artifact=") for item in working_memory)
        or any(isinstance(item, str) and "source=cached_artifact_reuse" in item for item in evidence_bundle)
    )


def index_by_vqa_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for payload in rows:
        vqa_id = str(payload.get("sample_id") or payload.get("vqa_id") or "")
        if vqa_id:
            indexed[vqa_id] = payload
    return indexed


def counter_to_ranked_list(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def maybe_append(rows: list[dict[str, Any]], item: dict[str, Any], *, limit: int) -> None:
    if len(rows) < limit:
        rows.append(item)


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def mean(values: list[float]) -> float:
    filtered = [float(value) for value in values]
    if not filtered:
        return 0.0
    return float(statistics.fmean(filtered))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def _safe_int(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return fallback


def session_position_of(payload: dict[str, Any]) -> int:
    explicit = _safe_int(payload.get("session_video_position"), fallback=0)
    if explicit > 0:
        return explicit
    return _safe_int(payload.get("inferred_session_video_position"), fallback=0)


if __name__ == "__main__":
    raise SystemExit(main())
