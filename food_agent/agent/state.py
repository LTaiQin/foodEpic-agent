"""Mutable execution state for the graph agent."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    video_id: str
    question: str
    choices: list[Any]
    task_family: str
    inputs_json: str = "{}"
    max_steps: int = 6
    current_step: int = 0
    plan_summary: str = ""
    hypotheses: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    retrieved_node_ids: list[str] = field(default_factory=list)
    retrieved_nodes: list[dict[str, Any]] = field(default_factory=list)
    retrieved_frames: list[str] = field(default_factory=list)
    evidence_bundle: list[str] = field(default_factory=list)
    working_memory: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    final_prediction: int | None = None
    confidence: float = 0.0

    def record_tool(self, name: str, args: dict[str, Any], result_summary: str, raw_result: Any | None = None) -> None:
        entry = {"tool": name, "args": args, "result_summary": result_summary}
        if raw_result is not None:
            entry["raw_result"] = raw_result
        self.tool_trace.append(entry)

    def add_node_result(self, node: dict[str, Any]) -> None:
        node_id = str(node.get("node_id") or "")
        if node_id and node_id in self.retrieved_node_ids:
            return
        self.retrieved_nodes.append(node)
        if node_id:
            self.retrieved_node_ids.append(node_id)
        for path in node.get("evidence_paths", []):
            if self._is_visual_asset(path) and path not in self.retrieved_frames:
                self.retrieved_frames.append(path)

    def add_evidence(self, text: str) -> None:
        if text and text not in self.evidence_bundle:
            self.evidence_bundle.append(text)

    def add_memory(self, text: str) -> None:
        if text and text not in self.working_memory:
            self.working_memory.append(text)

    def inputs_payload(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.inputs_json or "{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "task_family": self.task_family,
            "question": self.question,
            "choices": self.choices,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "plan_summary": self.plan_summary,
            "hypotheses": self.hypotheses,
            "open_questions": self.open_questions,
            "retrieved_node_ids": self.retrieved_node_ids[-20:],
            "retrieved_frames": self.retrieved_frames[-20:],
            "evidence_bundle": self.evidence_bundle[-20:],
            "working_memory": self.working_memory[-20:],
            "confidence": self.confidence,
        }

    def _is_visual_asset(self, path: Any) -> bool:
        if not isinstance(path, str) or not path:
            return False
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
