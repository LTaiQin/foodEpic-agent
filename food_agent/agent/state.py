"""Mutable execution state for the graph agent."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionIntentHypothesis:
    choice_index: int
    choice_text: str
    support_evidence: list[str] = field(default_factory=list)
    contradiction_evidence: list[str] = field(default_factory=list)
    missing_observations: list[str] = field(default_factory=list)
    comparison_summary: str = ""
    comparison: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "choice_index": int(self.choice_index),
            "choice_text": str(self.choice_text),
            "support_evidence": [str(item) for item in self.support_evidence if item],
            "contradiction_evidence": [str(item) for item in self.contradiction_evidence if item],
            "missing_observations": [str(item) for item in self.missing_observations if item],
            "comparison_summary": str(self.comparison_summary or ""),
            "comparison": dict(self.comparison or {}),
            "score": float(self.score),
            "confidence": float(self.confidence),
        }


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
    action_intent_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    search_budget: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    final_prediction: int | None = None
    confidence: float = 0.0
    final_metadata: dict[str, Any] = field(default_factory=dict)

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
        evidence_gaps: list[dict[str, Any]] | None = None,
        sufficiency_decision: dict[str, Any] | None = None,
        action_intent_hypotheses: list[dict[str, Any]] | None = None,
    ) -> None:
        primary_gap_recovery_trace = self._latest_primary_gap_recovery_trace()
        sanitized_sufficiency = self._sanitize_sufficiency_decision_for_runtime(sufficiency_decision)
        primary_gap_snapshot = self._latest_primary_gap_snapshot(
            evidence_gaps=evidence_gaps,
            sufficiency_decision=sanitized_sufficiency,
        )
        entry = {
            "sufficient": bool(sufficient),
            "confidence": float(confidence),
            "missing_evidence_types": [str(item) for item in missing_evidence_types if item],
            "conflicts": [str(item) for item in conflicts if item],
            "recommend_next_action": str(recommend_next_action or ""),
            "summary": str(summary or ""),
            "evidence_gaps": [item for item in (evidence_gaps or []) if isinstance(item, dict)],
            "sufficiency_decision": sanitized_sufficiency,
            "action_intent_hypotheses": [],
            "primary_gap_recovery_trace": primary_gap_recovery_trace,
            "primary_gap": primary_gap_snapshot,
        }
        self.verification_history.append(entry)
        self.action_intent_hypotheses = []
        self._sync_action_intent_trace_from_verification(entry)

    def latest_verification(self) -> dict[str, Any]:
        if not self.verification_history:
            return {}
        latest = self.verification_history[-1]
        return latest if isinstance(latest, dict) else {}

    def initialize_search_budget(self) -> None:
        if self.search_budget:
            return
        self.search_budget = {
            "max_tool_steps": int(self.max_steps),
            "max_new_frames": 40,
            "max_long_horizon_expansions": 5,
            "window_level": 0,
            "tool_steps_used": 0,
            "new_frames_observed": 0,
            "long_horizon_expansions_used": 0,
            "budget_exhausted": False,
        }

    def update_search_budget_after_tool(self, *, tool_name: str, newly_added_frames: int = 0) -> None:
        self.initialize_search_budget()
        self.search_budget["tool_steps_used"] = int(self.search_budget.get("tool_steps_used") or 0) + 1
        if newly_added_frames > 0:
            self.search_budget["new_frames_observed"] = int(self.search_budget.get("new_frames_observed") or 0) + int(newly_added_frames)
        if tool_name in {"query_object", "query_spatial_context", "sample_sparse_frames", "extract_frames_for_range"}:
            self.search_budget["long_horizon_expansions_used"] = int(self.search_budget.get("long_horizon_expansions_used") or 0) + 1
        if tool_name in {"sample_sparse_frames", "extract_frames_for_range"}:
            self.search_budget["window_level"] = min(2, int(self.search_budget.get("window_level") or 0) + 1)
        self.search_budget["budget_exhausted"] = bool(self.is_search_budget_exhausted())

    def is_search_budget_exhausted(self) -> bool:
        if not self.search_budget:
            return False
        return bool(
            int(self.search_budget.get("tool_steps_used") or 0) >= int(self.search_budget.get("max_tool_steps") or self.max_steps)
            or int(self.search_budget.get("new_frames_observed") or 0) >= int(self.search_budget.get("max_new_frames") or 24)
            or int(self.search_budget.get("long_horizon_expansions_used") or 0) >= int(self.search_budget.get("max_long_horizon_expansions") or 2)
        )

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
        self.action_intent_hypotheses = self.action_intent_hypotheses[-8:]

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
            "search_budget": self.search_budget,
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
            "action_intent_trace": self._export_action_intent_trace(limit=20),
            "search_budget": self.search_budget,
            "confidence": self.confidence,
        }

    def restore_session_memory(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if str(payload.get("video_id") or self.video_id) != self.video_id:
            return
        self.working_memory = self._sanitize_restored_runtime_strings(
            self._string_list(payload.get("working_memory"), limit=200)
        )
        self.evidence_bundle = self._sanitize_restored_runtime_strings(
            self._string_list(payload.get("evidence_bundle"), limit=200)
        )
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
        self.hypotheses = self._sanitize_restored_runtime_strings(
            self._string_list(payload.get("hypotheses"), limit=100)
        )
        self.open_questions = self._string_list(payload.get("open_questions"), limit=100)
        tool_failures = payload.get("tool_failures")
        if isinstance(tool_failures, list):
            self.tool_failures = [item for item in tool_failures[-100:] if isinstance(item, dict)]
        ineffective_tools = payload.get("ineffective_tools")
        if isinstance(ineffective_tools, list):
            self.ineffective_tools = [item for item in ineffective_tools[-100:] if isinstance(item, dict)]
        verification_history = payload.get("verification_history")
        if isinstance(verification_history, list):
            sanitized_history: list[dict[str, Any]] = []
            for item in verification_history[-100:]:
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                entry["action_intent_hypotheses"] = []
                entry["sufficiency_decision"] = self._sanitize_sufficiency_decision_for_runtime(
                    entry.get("sufficiency_decision")
                )
                sanitized_history.append(entry)
            self.verification_history = sanitized_history
        search_budget = payload.get("search_budget")
        if isinstance(search_budget, dict):
            self.search_budget = dict(search_budget)
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

    def _export_action_intent_trace(self, *, limit: int) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for verification in self.verification_history[-limit:]:
            if not isinstance(verification, dict):
                continue
            sufficiency = verification.get("sufficiency_decision")
            primary_gap = dict(verification.get("primary_gap") or {})
            recommended_next_action = str(
                (
                    sufficiency.get("recommended_next_step")
                    if isinstance(sufficiency, dict)
                    else ""
                )
                or verification.get("recommend_next_action")
                or ""
            )
            finish_mode = str(sufficiency.get("finish_mode") or "") if isinstance(sufficiency, dict) else ""
            if not (
                str(verification.get("summary") or "").strip()
                or primary_gap
                or str(verification.get("primary_gap_recovery_trace") or "").strip()
                or recommended_next_action
                or finish_mode
            ):
                continue
            trace.append(
                {
                    "summary": str(verification.get("summary") or ""),
                    "primary_gap": primary_gap,
                    "primary_gap_recovery_trace": str(verification.get("primary_gap_recovery_trace") or ""),
                    "recommended_next_action": recommended_next_action,
                    "finish_mode": finish_mode,
                }
            )
        return trace

    def _latest_primary_gap_recovery_trace(self) -> str:
        for item in reversed(self.hypotheses):
            if isinstance(item, str) and item.startswith("primary_gap_recovery_trace="):
                return item
        for item in reversed(self.working_memory):
            if isinstance(item, str) and item.startswith("primary_gap_recovery_trace="):
                return item
        return ""

    def _latest_primary_gap_snapshot(
        self,
        *,
        evidence_gaps: list[dict[str, Any]] | None,
        sufficiency_decision: dict[str, Any] | None,
    ) -> dict[str, Any]:
        decision = sufficiency_decision if isinstance(sufficiency_decision, dict) else {}
        recommended_next_step = str(decision.get("recommended_next_step") or "").strip()
        gaps = [item for item in (evidence_gaps or []) if isinstance(item, dict)]
        if gaps:
            priority_rank = {"high": 0, "medium": 1, "low": 2}
            preferred_gap_type = self._select_action_intent_primary_gap_type(
                missing_gap_types=[
                    str(item.get("gap_type") or "").strip()
                    for item in gaps
                    if isinstance(item, dict) and str(item.get("gap_type") or "").strip()
                ],
                recommended_next_step=recommended_next_step,
            )
            selected = sorted(
                gaps,
                key=lambda item: (
                    priority_rank.get(str(item.get("priority") or "").lower(), 3),
                    0 if str(item.get("gap_type") or "").strip() == preferred_gap_type and preferred_gap_type else 1,
                    str(item.get("gap_type") or ""),
                ),
            )[0]
            return {
                "gap_type": str(selected.get("gap_type") or ""),
                "missing_observation": str(selected.get("missing_observation") or ""),
                "target_object": str(selected.get("target_object") or ""),
                "target_fixture": str(selected.get("target_fixture") or ""),
                "time_relation": str(selected.get("time_relation") or ""),
                "source": str(selected.get("source") or ""),
                "priority": str(selected.get("priority") or ""),
            }
        missing_gap_types = [
            str(item).strip()
            for item in decision.get("missing_gap_types", [])
            if isinstance(item, str) and str(item).strip()
        ]
        if not missing_gap_types:
            return {}
        gap_type = self._select_action_intent_primary_gap_type(
            missing_gap_types=missing_gap_types,
            recommended_next_step=recommended_next_step,
        )
        if not gap_type:
            return {}
        return {
            "gap_type": gap_type,
            "missing_observation": "",
            "target_object": "",
            "target_fixture": "",
            "time_relation": "derived_from_sufficiency",
            "source": "sufficiency_missing_gap_types",
            "priority": "high",
        }

    def _select_action_intent_primary_gap_type(
        self,
        *,
        missing_gap_types: list[str],
        recommended_next_step: str,
    ) -> str:
        candidates = {
            str(item).strip()
            for item in missing_gap_types
            if isinstance(item, str) and str(item).strip()
        }
        if not candidates:
            return ""
        preferred_by_step = {
            "need_precondition_context": ("precondition",),
            "need_post_action_evidence": ("immediate_outcome", "future_outcome"),
            "need_location_evidence": ("target_discovery", "relation_confirmation", "future_outcome"),
            "need_disambiguating_evidence": ("relation_confirmation", "target_discovery", "future_outcome", "immediate_outcome"),
        }
        for candidate in preferred_by_step.get(recommended_next_step, ()):
            if candidate in candidates:
                return candidate
        for candidate in (
            "precondition",
            "immediate_outcome",
            "future_outcome",
            "relation_confirmation",
            "target_discovery",
        ):
            if candidate in candidates:
                return candidate
        return ""

    def _sanitize_sufficiency_decision_for_runtime(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        missing_gap_types = [
            str(item).strip()
            for item in value.get("missing_gap_types", [])
            if isinstance(item, str) and str(item).strip()
        ]
        return {
            "sufficient": bool(value.get("sufficient")) if "sufficient" in value else False,
            "missing_gap_types": missing_gap_types,
            "blocking_hypotheses": [],
            "blocking_comparisons": [],
            "recommended_next_step": str(value.get("recommended_next_step") or "").strip(),
            "finish_mode": str(value.get("finish_mode") or "").strip(),
            "summary": str(value.get("summary") or "").strip(),
        }

    def _sanitize_restored_runtime_strings(self, items: list[str]) -> list[str]:
        blocked_prefixes = (
            "action_intent_pending_resolution=",
            "action_intent_pending_candidates=",
            "action_intent_future_use_candidates=",
            "action_intent_resolution_withheld_for_",
            "action_intent_unresolved_rerank_withheld",
            "action_intent_top_hypothesis=",
            "action_intent_runner_up=",
            "action_intent_top_missing_observation=",
            "action_intent_comparison_summary=",
            "action_intent_blocking_comparison=",
            "planner_guard=",
            "planner_override ",
        )
        return [
            item
            for item in items
            if isinstance(item, str) and item and not item.startswith(blocked_prefixes)
        ]

    def _sync_action_intent_trace_from_verification(self, entry: dict[str, Any]) -> None:
        return None
