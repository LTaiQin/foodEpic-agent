#!/usr/bin/env python3
"""Export a human-readable markdown report for finalize residual routes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


def main() -> int:
    paths = ProjectPaths.from_env()
    audit_path = paths.output_root / "reports" / "graph_agent_finalize_residual_audit.json"
    out_path = paths.output_root / "reports" / "graph_agent_finalize_residual_report.md"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))

    lines = [
        "# Graph Agent Finalize Residual Report",
        "",
        "## Summary",
        "",
        f"- deterministic_task_count: {sum(item['count'] for item in payload.get('deterministic_task_counts', []))}",
        f"- residual_task_count: {sum(item['count'] for item in payload.get('residual_task_counts', []))}",
        "",
        "## Deterministic Task Families",
        "",
    ]
    for item in payload.get("deterministic_task_counts", []):
        lines.append(f"- {item['task_family']}: {item['count']}")
    lines.extend(["", "## Residual Task Families", ""])
    for item in payload.get("residual_task_counts", []):
        lines.append(f"- {item['task_family']}: {item['count']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `ingredient_ingredient_weight` 这类 residual 旧样本，当前代码已能走 deterministic weight finalize，但历史产物来自补强前运行，后续应在 P7 真跑中用新代码覆盖。",
            "- `ingredient_ingredient_retrieval` 的 residual 旧样本，当前代码已新增 answer-hint overlap fallback；若视觉/OCR 已明确给出 ingredient 线索，不再需要最终模型自由裁决。",
            "- 对仍必须保留最终模型裁决的 MCQ 路线，当前代码已加入 `mcq_answer_guard`，若模型输出与已有证据或 `candidate_answer_index` 冲突，会回退到更保守的结构化候选。",
            "",
            "## Residual Examples",
            "",
        ]
    )
    for example in payload.get("residual_examples", []):
        lines.append(f"### {example['task_family']} / {example['vqa_id']}")
        lines.append("")
        lines.append(f"- prediction: {example['prediction']}")
        lines.append(f"- tool_tail: {example['tool_tail']}")
        lines.append(f"- source_path: {example['source_path']}")
        lines.append("- working_memory_tail:")
        for item in example.get("working_memory_tail", []):
            lines.append(f"  - {item}")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
