#!/usr/bin/env python3
"""Audit open-query answer critic and grounded fallback behavior with executable cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent import GraphAgent
from food_agent.agent.state import AgentState
from food_agent.model_client import ModelResponse
from food_agent.paths import ProjectPaths


class StaticModelClient:
    def __init__(self, *, answer_text: str):
        self.answer_text = answer_text

    def complete_json(self, messages, temperature=0.0):
        return {
            "thought": "direct",
            "tool": "finish",
            "args": {"prediction": 0, "answer": "0", "confidence": 0.6},
            "done": True,
        }

    def complete(self, messages, temperature=0.0):
        return ModelResponse(content=self.answer_text, raw={})


def main() -> int:
    paths = ProjectPaths.from_env()
    report_path = paths.output_root / "reports" / "graph_agent_open_query_answer_critic_audit.json"

    cases = [
        {
            "name": "location_rejects_ungrounded_answer",
            "task_family": "open_query",
            "question": "Where is the bowl?",
            "evidence_bundle": ["target_location=left side of the counter"],
            "working_memory": ["target_location=left side of the counter"],
            "model_answer": "The bowl is on the right side near the stove.",
            "expect_mode": "structured_summary",
            "expect_memory_contains": "freeform_answer_critic_blocked=answer_not_grounded_to_location_evidence",
            "expect_answer_contains": "target_location=left side of the counter",
        },
        {
            "name": "location_accepts_grounded_answer",
            "task_family": "open_query",
            "question": "Where is the bowl?",
            "evidence_bundle": ["target_location=left side of the counter"],
            "working_memory": ["target_location=left side of the counter"],
            "model_answer": "The bowl is on the left side of the counter.",
            "expect_mode": "answer_critic_passed",
            "expect_memory_contains": "freeform_answer_mode=answer_critic_passed",
            "expect_answer_contains": "left side of the counter",
        },
        {
            "name": "temporal_uses_grounded_structured_answer_before_model",
            "task_family": "open_query_temporal_summary",
            "question": "What happened after the onion was added?",
            "evidence_bundle": [
                "type=timeline_event; label=add onion; time=10.000-12.000",
                "possible_step=stirring onion",
                "state_change_hint=onion became softened",
            ],
            "working_memory": [
                "possible_step=stirring onion",
                "state_change_hint=onion became softened",
            ],
            "model_answer": "unused",
            "expect_mode": "grounded_structured_answer",
            "expect_memory_contains": "freeform_answer_mode=grounded_structured_answer",
            "expect_answer_contains": "该时间段内主要发生的是",
        },
        {
            "name": "state_rejects_non_state_claim",
            "task_family": "open_query_state",
            "question": "What state is the onion in?",
            "evidence_bundle": ["state_change_hint=onion became softened"],
            "working_memory": ["state_change_hint=onion became softened"],
            "model_answer": "The onion is still raw and dry.",
            "expect_mode": "grounded_structured_answer",
            "expect_memory_contains": "freeform_answer_mode=grounded_structured_answer",
            "expect_answer_contains": "onion became softened",
        },
    ]

    rows: list[dict[str, Any]] = []
    passed = 0
    for case in cases:
        agent = GraphAgent(paths=paths, model_client=StaticModelClient(answer_text=str(case["model_answer"])))
        state = AgentState(
            video_id="audit_vid",
            question=str(case["question"]),
            choices=["OPEN_ENDED_RESPONSE"],
            task_family=str(case["task_family"]),
        )
        state.evidence_bundle = list(case["evidence_bundle"])
        state.working_memory = list(case["working_memory"])
        answer_text, prediction = agent._finalize_state_answer(state=state, freeform=True)
        mode_items = [item for item in state.working_memory if isinstance(item, str) and item.startswith("freeform_answer_mode=")]
        blocked_items = [item for item in state.working_memory if isinstance(item, str) and item.startswith("freeform_answer_critic_blocked=")]
        ok = bool(
            case["expect_memory_contains"] in state.working_memory
            and str(case["expect_answer_contains"]) in answer_text
            and prediction is None
        )
        if ok:
            passed += 1
        rows.append(
            {
                "name": case["name"],
                "passed": ok,
                "prediction": prediction,
                "answer_text": answer_text,
                "mode_items": mode_items,
                "blocked_items": blocked_items,
                "working_memory_tail": state.working_memory[-10:],
                "verification_history": state.verification_history[-3:],
            }
        )

    report = {
        "case_count": len(cases),
        "passed_count": passed,
        "all_passed": passed == len(cases),
        "cases": rows,
        "notes": [
            "这个审计直接执行 GraphAgent._finalize_state_answer(freeform=True)，覆盖 grounded structured answer、answer critic 放行、answer critic 阻断与 structured summary 降级。",
            "它不依赖外部模型提供商，使用静态 mock answer，因此可以稳定回归回答层护栏逻辑。",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
