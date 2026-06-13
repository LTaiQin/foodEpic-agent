#!/usr/bin/env python3
"""Audit which MCQ routes are deterministic vs residual model-backed finalization."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


DETERMINISTIC_PREFIXES = (
    "deterministic_finalize prediction=",
    "ingredient_retrieval_best_index=",
    "recipe_membership_best_index=",
    "exact_ingredient_amount_best_index=",
    "ingredient_order_best_index=",
    "action_mechanism_best_index=",
    "action_intent_best_index=",
    "recipe_catalog_best_index=",
    "recipe_nutrition_best_index=",
    "temporal_localization_best_index=",
    "visual_mcq_best_index=",
    "viewpoint_best_index=",
    "fixture_direction_best_index=",
    "gaze_best_index=",
    "object_location_best_index=",
    "itinerary_best_index=",
    "stationary_best_index=",
    "movement_count=",
    "count_candidates count=",
)


def main() -> int:
    paths = ProjectPaths.from_env()
    root = paths.graph_agent_runs_root
    report_path = paths.output_root / "reports" / "graph_agent_finalize_residual_audit.json"

    deterministic_by_task: Counter[str] = Counter()
    residual_by_task: Counter[str] = Counter()
    residual_examples: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*.json")):
        if path.name in {"task_compress.json", "task_restore.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("prediction") is None:
            continue
        task_family = str(payload.get("task_family") or "")
        working_memory = payload.get("working_memory") or []
        tool_trace = payload.get("tool_trace") or []
        if is_deterministic_record(working_memory):
            deterministic_by_task[task_family] += 1
            continue
        residual_by_task[task_family] += 1
        if len(residual_examples) < 20:
            residual_examples.append(
                {
                    "task_family": task_family,
                    "vqa_id": payload.get("vqa_id"),
                    "prediction": payload.get("prediction"),
                    "tool_tail": [entry.get("tool") for entry in tool_trace[-6:] if isinstance(entry, dict)],
                    "working_memory_tail": working_memory[-10:],
                    "source_path": path.as_posix(),
                }
            )

    report = {
        "deterministic_task_counts": counter_payload(deterministic_by_task),
        "residual_task_counts": counter_payload(residual_by_task),
        "residual_examples": residual_examples,
        "notes": [
            "这个审计主要看真实 graph_agent_runs 产物中，多选题最终预测是否带有结构化 deterministic 证据。",
            "被归为 residual 的历史样本，不一定代表当前代码仍需模型裁决；它们也可能来自早期实现，需要结合当前源码与新增测试一起解释。",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def is_deterministic_record(working_memory: Any) -> bool:
    if not isinstance(working_memory, list):
        return False
    for item in working_memory:
        if not isinstance(item, str):
            continue
        if any(item.startswith(prefix) for prefix in DETERMINISTIC_PREFIXES):
            return True
    return False


def counter_payload(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"task_family": key, "count": count} for key, count in counter.most_common()]


if __name__ == "__main__":
    raise SystemExit(main())
