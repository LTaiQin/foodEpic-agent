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
    visited_times: list[float] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    evidence_bundle: list[str] = field(default_factory=list)
    working_memory: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    ineffective_tools: list[dict[str, Any]] = field(default_factory=list)
    verification_history: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    final_prediction: int | None = None
    confidence: float = 0.0

    def record_tool(self, name: str, args: dict[str, Any], result_summary: str, raw_result: Any | None = None) -> None:
        entry = {"tool": name, "args": args, "result_summary": result_summary}
        if raw_result is not None:
            entry["raw_result"] = raw_result
        self.tool_trace.append(entry)

    def record_tool_failure(self, name: str, args: dict[str, Any], error_type: str, error_message: str) -> None:
        entry = {
            "tool": name,
            "args": args,
            "error_type": error_type,
            "error_message": error_message,
        }
        self.tool_failures.append(entry)
        self.tool_trace.append(
            {
                "tool": name,
                "args": args,
                "result_summary": f"tool_failed:{error_type}",
                "raw_result": {"tool_failed": True, "error_type": error_type, "error_message": error_message},
            }
        )

    def record_ineffective_tool(self, name: str, args: dict[str, Any], reason: str) -> None:
        entry = {"tool": name, "args": args, "reason": reason}
        self.ineffective_tools.append(entry)
        self.tool_trace.append(
            {
                "tool": name,
                "args": args,
                "result_summary": f"tool_ineffective:{reason}",
                "raw_result": {"tool_ineffective": True, "reason": reason},
            }
        )

    def record_verification(
        self,
        *,
        sufficient: bool,
        confidence: float,
        missing_evidence_types: list[str],
        conflicts: list[str],
        recommend_next_action: str,
        summary: str,
    ) -> None:
        entry = {
            "sufficient": bool(sufficient),
            "confidence": float(confidence),
            "missing_evidence_types": [str(item) for item in missing_evidence_types if item],
            "conflicts": [str(item) for item in conflicts if item],
            "recommend_next_action": str(recommend_next_action or ""),
            "summary": str(summary or ""),
        }
        self.verification_history.append(entry)

    def latest_verification(self) -> dict[str, Any]:
        if not self.verification_history:
            return {}
        latest = self.verification_history[-1]
        return latest if isinstance(latest, dict) else {}

    def add_node_result(self, node: dict[str, Any]) -> None:
        node_id = str(node.get("node_id") or "")
        if node_id and node_id in self.retrieved_node_ids:
            return
        self.retrieved_nodes.append(node)
        if node_id:
            self.retrieved_node_ids.append(node_id)
        if node.get("start_time") is not None:
            self.add_visited_time(node.get("start_time"))
        if node.get("end_time") is not None:
            self.add_visited_time(node.get("end_time"))
        for path in node.get("evidence_paths", []):
            if isinstance(path, str) and path:
                self.add_artifact(path)

    def add_evidence(self, text: str) -> None:
        if text and text not in self.evidence_bundle:
            self.evidence_bundle.append(text)

    def add_visited_time(self, value: Any) -> None:
        try:
            time_s = float(value)
        except Exception:  # noqa: BLE001
            return
        rounded = round(time_s, 3)
        if rounded not in self.visited_times:
            self.visited_times.append(rounded)

    def add_artifact(self, path: str) -> None:
        if path and path not in self.artifacts:
            self.artifacts.append(path)
        if path and self._is_visual_asset(path) and path not in self.retrieved_frames:
            self.retrieved_frames.append(path)

    def add_memory(self, text: str) -> None:
        if text and text not in self.working_memory:
            self.working_memory.append(text)

    def add_hypothesis(self, text: str) -> None:
        if text and text not in self.hypotheses:
            self.hypotheses.append(text)

    def add_open_question(self, text: str) -> None:
        if text and text not in self.open_questions:
            self.open_questions.append(text)

    def prune_open_question(self, text: str) -> None:
        if not text:
            return
        self.open_questions = [item for item in self.open_questions if item != text]

    def replace_open_questions(self, items: list[str]) -> None:
        deduped: list[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        self.open_questions = deduped[-20:]

    def trim_memory(self, *, working_limit: int = 120, evidence_limit: int = 120, frame_limit: int = 80, node_limit: int = 80) -> None:
        self.working_memory = self.working_memory[-working_limit:]
        self.evidence_bundle = self.evidence_bundle[-evidence_limit:]
        self.retrieved_frames = self.retrieved_frames[-frame_limit:]
        self.artifacts = self.artifacts[-frame_limit:]
        self.visited_times = self.visited_times[-200:]
        self.retrieved_nodes = self.retrieved_nodes[-node_limit:]
        self.retrieved_node_ids = self.retrieved_node_ids[-node_limit:]
        self.hypotheses = self.hypotheses[-80:]
        self.open_questions = self.open_questions[-20:]
        self.tool_trace = self.tool_trace[-120:]
        self.tool_failures = self.tool_failures[-80:]
        self.ineffective_tools = self.ineffective_tools[-80:]
        self.verification_history = self.verification_history[-40:]

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
            "visited_times": self.visited_times[-20:],
            "artifacts": self.artifacts[-20:],
            "evidence_bundle": self.evidence_bundle[-20:],
            "working_memory": self.working_memory[-20:],
            "tool_failures": self.tool_failures[-10:],
            "ineffective_tools": self.ineffective_tools[-10:],
            "verification_history": self.verification_history[-5:],
            "confidence": self.confidence,
        }

    def export_session_memory(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "working_memory": self.working_memory[-200:],
            "evidence_bundle": self.evidence_bundle[-200:],
            "retrieved_frames": self.retrieved_frames[-200:],
            "visited_times": self.visited_times[-200:],
            "artifacts": self.artifacts[-200:],
            "retrieved_node_ids": self.retrieved_node_ids[-200:],
            "retrieved_nodes": self.retrieved_nodes[-200:],
            "hypotheses": self.hypotheses[-100:],
            "open_questions": self.open_questions[-100:],
            "tool_failures": self.tool_failures[-100:],
            "ineffective_tools": self.ineffective_tools[-100:],
            "verification_history": self.verification_history[-100:],
            "confidence": self.confidence,
        }

    def restore_session_memory(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if str(payload.get("video_id") or self.video_id) != self.video_id:
            return
        self.working_memory = self._string_list(payload.get("working_memory"), limit=200)
        self.evidence_bundle = self._string_list(payload.get("evidence_bundle"), limit=200)
        self.retrieved_frames = self._string_list(payload.get("retrieved_frames"), limit=200)
        self.artifacts = self._string_list(payload.get("artifacts"), limit=200)
        visited_times = payload.get("visited_times")
        if isinstance(visited_times, list):
            self.visited_times = []
            for item in visited_times[-200:]:
                self.add_visited_time(item)
        self.retrieved_node_ids = self._string_list(payload.get("retrieved_node_ids"), limit=200)
        retrieved_nodes = payload.get("retrieved_nodes")
        if isinstance(retrieved_nodes, list):
            self.retrieved_nodes = [item for item in retrieved_nodes[-200:] if isinstance(item, dict)]
        self.hypotheses = self._string_list(payload.get("hypotheses"), limit=100)
        self.open_questions = self._string_list(payload.get("open_questions"), limit=100)
        tool_failures = payload.get("tool_failures")
        if isinstance(tool_failures, list):
            self.tool_failures = [item for item in tool_failures[-100:] if isinstance(item, dict)]
        ineffective_tools = payload.get("ineffective_tools")
        if isinstance(ineffective_tools, list):
            self.ineffective_tools = [item for item in ineffective_tools[-100:] if isinstance(item, dict)]
        verification_history = payload.get("verification_history")
        if isinstance(verification_history, list):
            self.verification_history = [item for item in verification_history[-100:] if isinstance(item, dict)]
        try:
            self.confidence = float(payload.get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            self.confidence = 0.0

    def _is_visual_asset(self, path: Any) -> bool:
        if not isinstance(path, str) or not path:
            return False
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}

    def _string_list(self, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value[-limit:] if isinstance(item, str) and item]
