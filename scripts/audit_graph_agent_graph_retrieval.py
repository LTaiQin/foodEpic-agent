#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.graph import VideoGraphBuilder
from food_agent.memory import GraphMemoryStore
from food_agent.paths import ProjectPaths
from food_agent.tools import AgentToolbox


REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "graph_agent_graph_retrieval_audit.json"


@dataclass(frozen=True)
class ToolAuditRecord:
    sample_id: str
    task_family: str
    video_id: str
    tool: str
    status: str
    result_count: int
    detail: dict[str, Any]


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--out-file", type=Path, default=REPORT_PATH)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--task-family-count", type=int, default=30)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--rebuild-graphs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_env()
    df = pd.read_parquet(args.index_file)
    rows = select_rows(
        df=df,
        seed=args.seed,
        task_family_count=args.task_family_count,
        samples_per_task=args.samples_per_task,
    )
    if not rows:
        raise RuntimeError("no audit rows selected")
    builder = VideoGraphBuilder(paths)
    tool_records: list[ToolAuditRecord] = []
    sampled_videos = sorted({str(row["primary_video_id"]) for row in rows})
    cached_toolboxes: dict[str, AgentToolbox] = {}

    for index, row in enumerate(rows, start=1):
        video_id = str(row["primary_video_id"])
        toolbox = cached_toolboxes.get(video_id)
        if toolbox is None:
            store = ensure_video_store(paths=paths, builder=builder, video_id=video_id, rebuild=args.rebuild_graphs)
            toolbox = AgentToolbox(store=store, paths=paths, model_client=object(), video_id=video_id)
            cached_toolboxes[video_id] = toolbox
        toolbox.set_runtime_context(
            question=str(row.get("question") or ""),
            inputs_json=str(row.get("inputs_json") or "{}"),
        )
        tool_records.extend(audit_row(toolbox=toolbox, row=row, limit=args.limit))
        if index % 20 == 0 or index == len(rows):
            print(f"[audit] processed {index}/{len(rows)} rows", flush=True)

    payload = build_report(rows=rows, records=tool_records, seed=args.seed)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = payload["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report_path={args.out_file}")
    print(f"sampled_videos={len(sampled_videos)} sampled_rows={len(rows)}")
    return 0


def select_rows(
    *,
    df: pd.DataFrame,
    seed: int,
    task_family_count: int,
    samples_per_task: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    families = sorted(str(item) for item in df["task_family"].dropna().unique().tolist())
    chosen_families = families if task_family_count >= len(families) else rng.sample(families, task_family_count)
    rows: list[dict[str, Any]] = []
    for family in sorted(chosen_families):
        subset = df[df["task_family"] == family].copy()
        if subset.empty:
            continue
        candidates = subset.to_dict("records")
        rng.shuffle(candidates)
        rows.extend(candidates[:samples_per_task])
    rows.sort(key=lambda item: (str(item.get("task_family") or ""), str(item.get("primary_video_id") or ""), str(item.get("vqa_id") or "")))
    return rows


def ensure_video_store(*, paths: ProjectPaths, builder: VideoGraphBuilder, video_id: str, rebuild: bool) -> GraphMemoryStore:
    store = GraphMemoryStore(paths.graph_memory_root / video_id)
    if rebuild or not store.query_nodes(video_id=video_id, limit=1):
        return builder.build(video_id)
    return store


def audit_row(toolbox: AgentToolbox, row: dict[str, Any], limit: int) -> list[ToolAuditRecord]:
    question = str(row.get("question") or "")
    hints = toolbox.default_hints(question, str(row.get("inputs_json") or "{}"))
    sample_id = str(row["vqa_id"])
    task_family = str(row["task_family"])
    video_id = str(row["primary_video_id"])
    anchor_times = collect_anchor_times(hints)
    records: list[ToolAuditRecord] = []
    seeded_node_ids: list[str] = []

    time_window = infer_time_window(anchor_times)
    if time_window is None:
        records.append(record_skip(sample_id, task_family, video_id, "query_time", "no_anchor_time", hints))
    else:
        result = toolbox.query_time(start_time=time_window[0], end_time=time_window[1], limit=limit)
        seeded_node_ids.extend(node_ids_from_result(result))
        records.append(record_result(sample_id, task_family, video_id, "query_time", result, {"time_window": time_window}))

    state_keyword = str(hints.get("state_keyword") or "").strip()
    if not state_keyword:
        records.append(record_skip(sample_id, task_family, video_id, "query_state", "no_state_hint", hints))
    else:
        result = toolbox.query_state(
            state_keyword=state_keyword,
            start_time=time_window[0] if time_window else None,
            end_time=time_window[1] if time_window else None,
            limit=limit,
        )
        seeded_node_ids.extend(node_ids_from_result(result))
        records.append(record_result(sample_id, task_family, video_id, "query_state", result, {"state_keyword": state_keyword}))

    location_keyword = str(hints.get("location_keyword") or "").strip()
    if not location_keyword:
        records.append(record_skip(sample_id, task_family, video_id, "query_location", "no_location_hint", hints))
    else:
        result = toolbox.query_location(
            location_keyword=location_keyword,
            start_time=time_window[0] if time_window else None,
            end_time=time_window[1] if time_window else None,
            limit=limit,
        )
        seeded_node_ids.extend(node_ids_from_result(result))
        records.append(record_result(sample_id, task_family, video_id, "query_location", result, {"location_keyword": location_keyword}))

    ocr_keyword = str(hints.get("ocr_keyword") or "").strip()
    if not ocr_keyword:
        records.append(record_skip(sample_id, task_family, video_id, "query_ocr", "no_ocr_hint", hints))
    else:
        result = toolbox.query_ocr(
            keyword=ocr_keyword,
            start_time=time_window[0] if time_window else None,
            end_time=time_window[1] if time_window else None,
            limit=limit,
        )
        seeded_node_ids.extend(node_ids_from_result(result))
        records.append(record_result(sample_id, task_family, video_id, "query_ocr", result, {"ocr_keyword": ocr_keyword}))

    ingredient_name = str(hints.get("ingredient_name") or "").strip()
    if not ingredient_name or not looks_like_measurement_question(question):
        reason = "not_measurement_question" if ingredient_name else "no_ingredient_hint"
        records.append(record_skip(sample_id, task_family, video_id, "query_ingredient_measurement", reason, hints))
    else:
        result = toolbox.query_ingredient_measurement(
            ingredient_name=ingredient_name,
            start_time=time_window[0] if time_window else None,
            end_time=time_window[1] if time_window else None,
            limit=limit,
        )
        seeded_node_ids.extend(node_ids_from_result(result))
        records.append(record_result(sample_id, task_family, video_id, "query_ingredient_measurement", result, {"ingredient_name": ingredient_name}))

    if not anchor_times:
        records.append(record_skip(sample_id, task_family, video_id, "query_spatial_context", "no_anchor_time", hints))
    else:
        object_name = str(hints.get("object_hint") or hints.get("location_keyword") or "").strip() or None
        result = toolbox.query_spatial_context(time_s=anchor_times[0], object_name=object_name, limit=limit)
        records.append(record_result(sample_id, task_family, video_id, "query_spatial_context", result, {"anchor_time": anchor_times[0], "object_name": object_name}))

    unique_node_ids = dedupe_preserve(seeded_node_ids)
    if not unique_node_ids:
        records.append(record_skip(sample_id, task_family, video_id, "expand_graph_context", "no_seed_nodes", hints))
    else:
        result = toolbox.expand_graph_context(node_ids=unique_node_ids[: min(8, len(unique_node_ids))], limit=limit)
        records.append(record_result(sample_id, task_family, video_id, "expand_graph_context", result, {"seed_node_count": len(unique_node_ids)}))

    return records


def collect_anchor_times(hints: dict[str, Any]) -> list[float]:
    raw_times = list(hints.get("times") or []) + list(hints.get("input_times") or [])
    values: list[float] = []
    for value in raw_times:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(values))


def infer_time_window(anchor_times: list[float]) -> tuple[float, float] | None:
    if not anchor_times:
        return None
    if len(anchor_times) == 1:
        time_s = anchor_times[0]
        return (max(0.0, time_s - 3.0), time_s + 3.0)
    return (max(0.0, min(anchor_times) - 1.0), max(anchor_times) + 1.0)


def looks_like_measurement_question(question: str) -> bool:
    lowered = question.lower()
    tokens = ("weigh", "weight", "gram", "grams", "kg", "digit", "number", "reading", "measure")
    return any(token in lowered for token in tokens)


def node_ids_from_result(result: dict[str, Any]) -> list[str]:
    if "nodes" in result and isinstance(result["nodes"], list):
        return [str(node.get("node_id")) for node in result["nodes"] if isinstance(node, dict) and node.get("node_id")]
    if "matches" in result and isinstance(result["matches"], list):
        return [str(node.get("node_id")) for node in result["matches"] if isinstance(node, dict) and node.get("node_id")]
    return []


def record_skip(sample_id: str, task_family: str, video_id: str, tool: str, reason: str, hints: dict[str, Any]) -> ToolAuditRecord:
    detail = {
        "reason": reason,
        "times": list(hints.get("times") or []),
        "input_times": list(hints.get("input_times") or []),
        "state_keyword": hints.get("state_keyword"),
        "location_keyword": hints.get("location_keyword"),
        "ocr_keyword": hints.get("ocr_keyword"),
        "ingredient_name": hints.get("ingredient_name"),
        "object_hint": hints.get("object_hint"),
    }
    return ToolAuditRecord(sample_id=sample_id, task_family=task_family, video_id=video_id, tool=tool, status="skipped", result_count=0, detail=detail)


def record_result(sample_id: str, task_family: str, video_id: str, tool: str, result: dict[str, Any], extra_detail: dict[str, Any]) -> ToolAuditRecord:
    result_count = infer_result_count(result)
    status = "hit" if result_count > 0 else "empty"
    detail = dict(extra_detail)
    detail["result_keys"] = sorted(result.keys())
    if "node_count" in result:
        detail["node_count"] = int(result.get("node_count") or 0)
    return ToolAuditRecord(sample_id=sample_id, task_family=task_family, video_id=video_id, tool=tool, status=status, result_count=result_count, detail=detail)


def infer_result_count(result: dict[str, Any]) -> int:
    for key in ("count", "node_count"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    if isinstance(result.get("nodes"), list):
        return len(result["nodes"])
    if isinstance(result.get("matches"), list):
        return len(result["matches"])
    return 0


def dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_report(*, rows: list[dict[str, Any]], records: list[ToolAuditRecord], seed: int) -> dict[str, Any]:
    tool_summary: dict[str, Any] = {}
    overall_status = Counter(record.status for record in records)
    by_tool: dict[str, list[ToolAuditRecord]] = defaultdict(list)
    for record in records:
        by_tool[record.tool].append(record)

    for tool, tool_records in sorted(by_tool.items()):
        calls = sum(1 for item in tool_records if item.status != "skipped")
        hits = sum(1 for item in tool_records if item.status == "hit")
        empties = sum(1 for item in tool_records if item.status == "empty")
        skips = sum(1 for item in tool_records if item.status == "skipped")
        family_counter = Counter(item.task_family for item in tool_records if item.status == "hit")
        empty_reasons = Counter(str(item.detail.get("reason") or "empty_result") for item in tool_records if item.status != "hit")
        avg_result_count = round(sum(item.result_count for item in tool_records if item.status != "skipped") / calls, 3) if calls else 0.0
        tool_summary[tool] = {
            "calls": calls,
            "hits": hits,
            "empties": empties,
            "skips": skips,
            "hit_rate": round(hits / calls, 4) if calls else None,
            "avg_result_count": avg_result_count,
            "top_hit_task_families": family_counter.most_common(8),
            "top_empty_or_skip_reasons": empty_reasons.most_common(8),
        }

    return {
        "summary": {
            "seed": seed,
            "sample_count": len(rows),
            "task_family_count": len({str(row["task_family"]) for row in rows}),
            "video_count": len({str(row["primary_video_id"]) for row in rows}),
            "tool_record_count": len(records),
            "status_breakdown": dict(overall_status),
        },
        "selection": [
            {
                "vqa_id": str(row["vqa_id"]),
                "task_family": str(row["task_family"]),
                "video_id": str(row["primary_video_id"]),
                "question": str(row["question"]),
            }
            for row in rows
        ],
        "tool_summary": tool_summary,
        "records": [asdict(record) for record in records],
    }


if __name__ == "__main__":
    raise SystemExit(main())
