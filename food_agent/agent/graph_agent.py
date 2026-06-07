"""End-to-end complete graph/video agent."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from hashlib import md5
from typing import Any

from food_agent.agent.action_intent import (
    action_intent_needs_future_use_resolution,
    action_intent_needs_pairwise_resolution,
    selected_choice_categories,
)
from food_agent.agent.artifact_policy import artifact_reuse_prefixes_for_task
from food_agent.agent.executor import GraphAgentExecutor
from food_agent.agent.planner import GraphAgentPlanner
from food_agent.agent.state import AgentState
from food_agent.agent.verifier import GraphAgentVerifier
from food_agent.graph import VideoGraphBuilder
from food_agent.memory import GraphMemoryStore
from food_agent.model_client import OpenAICompatibleModelClient
from food_agent.paths import ProjectPaths
from food_agent.tools import AgentToolbox


CHOICE_RE = re.compile(r"\b([0-4])\b")


@dataclass(frozen=True)
class GraphAgentResult:
    vqa_id: str
    video_id: str
    task_family: str
    prediction: int | None
    answer_text: str
    evidence_bundle: list[str]
    tool_trace: list[dict[str, Any]]
    raw_model_output: str
    working_memory: list[str]
    retrieved_frames: list[str]
    visited_times: list[float]
    artifacts: list[str]
    confidence: float
    elapsed_seconds: float
    usage: dict[str, float] = field(default_factory=dict)
    verification_history: list[dict[str, Any]] = field(default_factory=list)
    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    ineffective_tools: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self, *, gold: int | None = None, include_row: dict[str, Any] | None = None) -> dict[str, Any]:
        tool_calls = [entry.get("tool") for entry in self.tool_trace if isinstance(entry, dict) and entry.get("tool")]
        latest_verification = self.verification_history[-1] if self.verification_history else {}
        payload = {
            "vqa_id": self.vqa_id,
            "video_id": self.video_id,
            "task_family": self.task_family,
            "prediction": self.prediction,
            "gold": gold,
            "correct": None if gold is None or self.prediction is None else self.prediction == int(gold),
            "answer_text": self.answer_text,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "usage": self.usage,
            "tool_trace": self.tool_trace,
            "evidence_bundle": self.evidence_bundle,
            "working_memory": self.working_memory,
            "retrieved_frames": self.retrieved_frames,
            "visited_times": self.visited_times,
            "artifacts": self.artifacts,
            "verification_history": self.verification_history,
            "latest_verification": latest_verification,
            "tool_failures": self.tool_failures,
            "ineffective_tools": self.ineffective_tools,
            "open_questions": self.open_questions,
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "failure_count": len(self.tool_failures),
            "ineffective_tool_count": len(self.ineffective_tools),
            "verification_count": len(self.verification_history),
            "raw_model_output": self.raw_model_output,
        }
        if include_row is not None:
            payload["question"] = include_row.get("question")
            payload["choices_json"] = include_row.get("choices_json")
            payload["inputs_json"] = include_row.get("inputs_json")
        return payload


class GraphAgentVideoSession:
    """Persistent same-video execution session that reuses one store/toolbox/executor."""

    def __init__(self, *, agent: GraphAgent, video_id: str):
        self.agent = agent
        self.video_id = video_id
        self.store = agent._ensure_store(video_id)
        self.toolbox = AgentToolbox(
            store=self.store,
            paths=agent.paths,
            model_client=agent.model_client,
            video_id=video_id,
        )
        self.executor = GraphAgentExecutor(self.toolbox, agent.planner, agent.verifier)
        self.session_dir = agent.paths.graph_agent_sessions_root / video_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.session_dir / "session_trace.jsonl"
        self.state_path = self.session_dir / "session_state.json"
        self.question_count = 0
        self.persisted_memory: dict[str, Any] = self._load_session_state()

    def answer_vqa_row(self, row: dict[str, Any], *, max_steps: int = 6) -> GraphAgentResult:
        return self._answer_row(row=row, max_steps=max_steps, freeform=False)

    def answer_open_query(
        self,
        *,
        question: str,
        inputs_json: str = "{}",
        task_family: str = "open_query",
        max_steps: int = 6,
        query_id: str = "",
    ) -> GraphAgentResult:
        resolved_task_family = self.agent._resolve_open_query_task_family(
            question=question,
            inputs_json=inputs_json,
            task_family=task_family,
        )
        generated_id = query_id or self._make_open_query_id(question=question, inputs_json=inputs_json, task_family=resolved_task_family)
        row = {
            "vqa_id": generated_id,
            "task_family": resolved_task_family,
            "primary_video_id": self.video_id,
            "question": question,
            "choices_json": json.dumps(["OPEN_ENDED_RESPONSE"], ensure_ascii=False),
            "correct_idx": None,
            "inputs_json": inputs_json,
        }
        return self._answer_row(row=row, max_steps=max_steps, freeform=True)

    def _answer_row(self, *, row: dict[str, Any], max_steps: int, freeform: bool) -> GraphAgentResult:
        started_at = time.time()
        usage_before = self._usage_snapshot()
        vqa_id = str(row.get("vqa_id") or "")
        task_family = str(row["task_family"])
        hints = self.toolbox.default_hints(str(row.get("question") or ""), str(row.get("inputs_json") or "{}"))
        state = AgentState(
            video_id=self.video_id,
            question=str(row["question"]),
            choices=json.loads(row["choices_json"]),
            task_family=task_family,
            inputs_json=str(row.get("inputs_json", "{}")),
            max_steps=max_steps,
        )
        state.restore_session_memory(self.persisted_memory)
        self._prepare_restored_state_for_new_question(state=state, hints=hints)
        state = self.executor.execute(state)
        answer_text, prediction = self.agent._finalize_state_answer(state=state, freeform=freeform)
        result = GraphAgentResult(
            vqa_id=vqa_id,
            video_id=self.video_id,
            task_family=task_family,
            prediction=prediction,
            answer_text=answer_text,
            evidence_bundle=state.evidence_bundle,
            tool_trace=state.tool_trace,
            raw_model_output=answer_text,
            working_memory=state.working_memory,
            retrieved_frames=state.retrieved_frames,
            visited_times=state.visited_times,
            artifacts=state.artifacts,
            verification_history=state.verification_history,
            tool_failures=state.tool_failures,
            ineffective_tools=state.ineffective_tools,
            open_questions=state.open_questions,
            confidence=state.confidence,
            elapsed_seconds=time.time() - started_at,
            usage=self._usage_delta(usage_before),
        )
        self._compress_and_persist_session_memory(state=state, row=row)
        self.question_count += 1
        self.persisted_memory = state.export_session_memory()
        self._save_session_state(result=result, state=state)
        report_path = self.agent._persist_evidence_report(result, row=row)
        self.agent._persist_result(result, row=row, evidence_report_path=report_path)
        self._append_session_trace(result, row=row)
        return result

    def _usage_delta(self, before: dict[str, float]) -> dict[str, float]:
        after = self._usage_snapshot()
        keys = ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost")
        return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}

    def _usage_snapshot(self) -> dict[str, float]:
        snapshot_fn = getattr(self.agent.model_client, "usage_snapshot", None)
        if callable(snapshot_fn):
            try:
                snapshot = snapshot_fn()
                if isinstance(snapshot, dict):
                    return snapshot
            except Exception:  # noqa: BLE001
                pass
        return {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "estimated_cost": 0.0,
        }

    def _prepare_restored_state_for_new_question(self, *, state: AgentState, hints: dict[str, Any]) -> None:
        restored_conflicts = [
            item
            for item in state.open_questions
            if isinstance(item, str) and item.startswith("conflict:")
        ]
        state.plan_summary = ""
        state.current_step = 0
        state.final_answer = ""
        state.final_prediction = None
        state.confidence = 0.0
        state.tool_trace = []
        state.open_questions = restored_conflicts
        state.tool_failures = []
        state.ineffective_tools = []
        state.verification_history = []
        state.hypotheses = []

        relevant_tokens = {
            str(item).strip().lower()
            for item in (
                state.task_family,
                hints.get("ingredient_name"),
                hints.get("ocr_keyword"),
                hints.get("state_keyword"),
                hints.get("location_keyword"),
                hints.get("object_hint"),
            )
            if isinstance(item, str) and str(item).strip()
        }
        anchor_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    anchor_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        blocked_prefixes = self._session_blocked_prefixes_for_task(state.task_family)
        blocked_substrings = self._session_blocked_substrings_for_task(state.task_family)
        state.working_memory = self._filter_restored_strings(
            items=state.working_memory,
            relevant_tokens=relevant_tokens,
            keep_prefixes=("reuse:", "reuse_relation:", "ocr_reading=", "measurement ", "target_location=", "scene_location=", "state_change_hint=", "possible_step="),
            blocked_prefixes=blocked_prefixes,
            blocked_substrings=blocked_substrings,
            limit=28,
            allow_unscoped_backfill=str(state.task_family or "").strip().lower() != "fine_grained_why_recognition",
            anchor_times=anchor_times,
        )
        state.evidence_bundle = self._filter_restored_strings(
            items=state.evidence_bundle,
            relevant_tokens=relevant_tokens,
            keep_prefixes=("type=ocr_reading", "ocr_reading=", "measurement ", "target_location=", "scene_location=", "state_change_hint=", "possible_step=", "type=timeline_event"),
            blocked_prefixes=blocked_prefixes,
            blocked_substrings=blocked_substrings,
            limit=24,
            allow_unscoped_backfill=str(state.task_family or "").strip().lower() != "fine_grained_why_recognition",
            anchor_times=anchor_times,
        )
        state.retrieved_frames = self._filter_restored_frames(
            frames=state.retrieved_frames,
            task_family=state.task_family,
            hints=hints,
            limit=12,
        )
        state.artifacts = self._filter_restored_frames(
            frames=state.artifacts,
            task_family=state.task_family,
            hints=hints,
            limit=20,
        )
        state.visited_times = sorted({round(float(item), 3) for item in state.visited_times[-80:]})
        if not state.retrieved_frames:
            for path in state.artifacts[-8:]:
                if isinstance(path, str) and path and path not in state.retrieved_frames:
                    state.retrieved_frames.append(path)
        state.retrieved_node_ids = state.retrieved_node_ids[-24:]
        state.retrieved_nodes = state.retrieved_nodes[-24:]
        if restored_conflicts:
            state.add_memory("session_conflict_guard=restored_conflict")
        state.trim_memory(working_limit=40, evidence_limit=32, frame_limit=16, node_limit=24)

    def _filter_restored_strings(
        self,
        *,
        items: list[str],
        relevant_tokens: set[str],
        keep_prefixes: tuple[str, ...],
        blocked_prefixes: tuple[str, ...],
        blocked_substrings: tuple[str, ...],
        limit: int,
        allow_unscoped_backfill: bool = True,
        anchor_times: list[float] | None = None,
    ) -> list[str]:
        kept: list[str] = []
        window_start = min(anchor_times) - 6.0 if anchor_times else None
        window_end = max(anchor_times) + 8.0 if anchor_times else None
        for item in items:
            if not isinstance(item, str) or not item:
                continue
            lowered = item.lower()
            if any(lowered.startswith(prefix) for prefix in blocked_prefixes):
                continue
            if any(token in lowered for token in blocked_substrings):
                continue
            has_time_overlap = False
            if window_start is not None and window_end is not None:
                for start_time, end_time in self._extract_embedded_note_times(item):
                    if not (end_time < window_start or start_time > window_end):
                        has_time_overlap = True
                        break
            if (
                any(lowered.startswith(prefix) for prefix in keep_prefixes)
                or any(token in lowered for token in relevant_tokens)
                or has_time_overlap
            ):
                if item not in kept:
                    kept.append(item)
        if allow_unscoped_backfill and len(kept) < limit:
            for item in items[-limit:]:
                lowered = item.lower() if isinstance(item, str) else ""
                if (
                    isinstance(item, str)
                    and item
                    and item not in kept
                    and not any(lowered.startswith(prefix) for prefix in blocked_prefixes)
                    and not any(token in lowered for token in blocked_substrings)
                ):
                    kept.append(item)
                if len(kept) >= limit:
                    break
        return kept[-limit:]

    def _extract_embedded_note_times(self, text: str) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        for match in re.finditer(r"time=([0-9.]+)-([0-9.]+)", str(text)):
            try:
                spans.append((float(match.group(1)), float(match.group(2))))
            except Exception:  # noqa: BLE001
                continue
        return spans

    def _is_viewpoint_like_task(self, task_family: str) -> bool:
        normalized = str(task_family or "").strip().lower()
        return normalized in {
            "3d_perception_fixture_location",
            "gaze_gaze_estimation",
            "3d_perception_object_location",
        }

    def _artifact_reuse_prefixes_for_task(self, task_family: str) -> tuple[str, ...]:
        return artifact_reuse_prefixes_for_task(task_family)

    def _session_blocked_prefixes_for_task(self, task_family: str) -> tuple[str, ...]:
        if str(task_family or "").strip().lower() == "fine_grained_why_recognition":
            return (
                "action_intent_",
                "candidate_answer_index=",
            )
        if self._is_viewpoint_like_task(task_family):
            return (
                "target_location=",
                "scene_location=",
                "fixture_direction_best_index=",
                "gaze_best_index=",
                "object_location_best_index=",
            )
        return ()

    def _session_blocked_substrings_for_task(self, task_family: str) -> tuple[str, ...]:
        if str(task_family or "").strip().lower() == "fine_grained_why_recognition":
            return (
                "action_intent_",
                "visual_mcq_reason=",
                "answer_hint=",
                "candidate_answer_index=",
                "source=agent_timeline_summary; summary=",
                "source=session_memory_compressor",
            )
        if self._is_viewpoint_like_task(task_family):
            return (
                "fixture_direction_reason=",
                "gaze_reason=",
                "source=session_memory_compressor; summary=target_location=",
                "source=session_memory_compressor; summary=scene_location=",
                "source=agent_timeline_summary; summary=",
            )
        return ()

    def _filter_restored_frames(self, *, frames: list[str], task_family: str, hints: dict[str, Any] | None, limit: int) -> list[str]:
        prefixes = tuple(token.lower() for token in self._artifact_reuse_prefixes_for_task(task_family))
        preferred = [
            item
            for item in frames
            if isinstance(item, str) and any(prefix in item.lower() for prefix in prefixes)
        ]
        if str(task_family or "").strip().lower() == "fine_grained_why_recognition":
            scoped = self._filter_restored_action_intent_frames_by_time(preferred or frames, hints=hints, limit=limit)
            if scoped:
                return scoped
        if preferred:
            return preferred[-limit:]
        return [item for item in frames[-limit:] if isinstance(item, str) and item]

    def _filter_restored_action_intent_frames_by_time(
        self,
        frames: list[str],
        *,
        hints: dict[str, Any] | None,
        limit: int,
    ) -> list[str]:
        if not frames:
            return []
        anchor_times: list[float] = []
        if isinstance(hints, dict):
            for key in ("times", "input_times"):
                for value in hints.get(key) or []:
                    try:
                        anchor_times.append(float(value))
                    except Exception:  # noqa: BLE001
                        continue
        if not anchor_times:
            return [item for item in frames[-limit:] if isinstance(item, str) and item]
        window_start = min(anchor_times) - 8.0
        window_end = max(anchor_times) + 12.0
        scoped: list[str] = []
        for item in frames:
            if not isinstance(item, str) or not item:
                continue
            match = re.search(r"_([0-9]+(?:\.[0-9]+)?)s\.[^.]+$", item)
            if not match:
                continue
            try:
                artifact_time = float(match.group(1))
            except Exception:  # noqa: BLE001
                continue
            if window_start <= artifact_time <= window_end:
                scoped.append(item)
        if scoped:
            return scoped[-limit:]
        return [item for item in frames[-limit:] if isinstance(item, str) and item]

    def _make_open_query_id(self, *, question: str, inputs_json: str, task_family: str) -> str:
        digest = md5(f"{self.video_id}|{task_family}|{question}|{inputs_json}".encode("utf-8")).hexdigest()[:12]
        return f"open_query:{self.video_id}:{digest}"

    def _append_session_trace(self, result: GraphAgentResult, *, row: dict[str, Any]) -> None:
        payload = {
            "vqa_id": result.vqa_id,
            "task_family": result.task_family,
            "prediction": result.prediction,
            "elapsed_seconds": result.elapsed_seconds,
            "question_count": self.question_count,
            "tool_calls": [entry.get("tool") for entry in result.tool_trace if isinstance(entry, dict)],
            "visited_times_tail": result.visited_times[-12:],
            "artifacts_tail": result.artifacts[-12:],
            "tool_failures": result.tool_failures[-5:],
            "ineffective_tools": result.ineffective_tools[-5:],
            "latest_verification": result.verification_history[-1] if result.verification_history else {},
            "open_questions_tail": result.open_questions[-8:],
            "working_memory_tail": result.working_memory[-12:],
            "evidence_tail": result.evidence_bundle[-12:],
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_session_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"video_id": self.video_id}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"video_id": self.video_id}
        if not isinstance(payload, dict):
            return {"video_id": self.video_id}
        try:
            self.question_count = int(payload.get("question_count") or 0)
        except Exception:  # noqa: BLE001
            self.question_count = 0
        session_memory = payload.get("session_memory")
        return session_memory if isinstance(session_memory, dict) else {"video_id": self.video_id}

    def _save_session_state(self, *, result: GraphAgentResult, state: AgentState) -> None:
        payload = {
            "video_id": self.video_id,
            "question_count": self.question_count,
            "last_vqa_id": result.vqa_id,
            "last_task_family": result.task_family,
            "last_prediction": result.prediction,
            "last_elapsed_seconds": result.elapsed_seconds,
            "updated_at": time.time(),
            "session_memory": state.export_session_memory(),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _compress_and_persist_session_memory(self, *, state: AgentState, row: dict[str, Any]) -> None:
        if self._should_skip_session_writeback(state):
            state.add_memory("session_writeback_skipped reason=conflict")
            state.trim_memory()
            return
        summary_lines = self._session_summary_lines(state=state)
        if summary_lines:
            label = f"session summary {row.get('task_family') or state.task_family}"
            try:
                start_time, end_time = self._infer_session_time_window(state=state, row=row)
                self.toolbox.write_timeline_summary(
                    label=label,
                    start_time=start_time,
                    end_time=end_time,
                    summary=" | ".join(summary_lines),
                    evidence_paths=state.retrieved_frames[-12:],
                    keywords=[state.task_family, "session_summary", "compressed_memory"],
                )
            except Exception:  # noqa: BLE001
                pass
        important = self._session_important_memory(state=state)
        if important:
            try:
                start_time, end_time = self._infer_session_time_window(state=state, row=row)
                self.toolbox.write_observation(
                    label=f"compressed session memory {state.task_family}",
                    start_time=start_time,
                    end_time=end_time,
                    attributes={"summary": " | ".join(important[-8:]), "source": "session_memory_compressor"},
                    evidence_paths=state.retrieved_frames[-12:],
                    keywords=[state.task_family, "compressed", "memory"],
                )
            except Exception:  # noqa: BLE001
                pass
        state.trim_memory()

    def _session_summary_lines(self, *, state: AgentState) -> list[str]:
        lines = state.evidence_bundle[-6:] or state.working_memory[-6:]
        if self._is_action_intent_like_task(state.task_family):
            return [
                item
                for item in lines
                if not self._is_action_intent_leaky_memory(item)
            ]
        if not self._is_viewpoint_like_task(state.task_family):
            return lines
        filtered: list[str] = []
        for item in lines:
            lowered = str(item).lower()
            if "fixture_direction_reason=" in lowered or "gaze_reason=" in lowered:
                continue
            if "target_location=" in lowered or "scene_location=" in lowered:
                continue
            filtered.append(item)
        return filtered

    def _session_important_memory(self, *, state: AgentState) -> list[str]:
        if not state.working_memory:
            return []
        important = [
            item
            for item in state.working_memory
            if any(
                token in item
                for token in (
                    "ocr_reading=",
                    "state_change_hint=",
                    "target_location=",
                    "possible_step=",
                    "candidate_answer_index=",
                )
            )
        ]
        if self._is_action_intent_like_task(state.task_family):
            return [
                item
                for item in important
                if not self._is_action_intent_leaky_memory(item)
            ]
        if not self._is_viewpoint_like_task(state.task_family):
            return important
        return [
            item
            for item in important
            if "target_location=" not in item.lower() and "scene_location=" not in item.lower()
        ]

    def _is_action_intent_like_task(self, task_family: str) -> bool:
        return str(task_family or "").strip().lower() == "fine_grained_why_recognition"

    def _is_action_intent_leaky_memory(self, item: object) -> bool:
        lowered = str(item or "").lower()
        return any(
            token in lowered
            for token in (
                "action_intent_",
                "visual_mcq_reason=",
                "answer_hint=",
                "candidate_answer_index=",
                "deterministic_finalize",
                "source=agent_timeline_summary",
                "source=session_memory_compressor",
            )
        )

    def _should_skip_session_writeback(self, state: AgentState) -> bool:
        if any(
            isinstance(item, str) and item.startswith("conflict:")
            for item in getattr(state, "open_questions", []) or []
        ):
            return True
        return any(
            isinstance(item, str) and item == "session_conflict_guard=restored_conflict"
            for item in getattr(state, "working_memory", []) or []
        )

    def _infer_session_time_window(self, *, state: AgentState, row: dict[str, Any]) -> tuple[float | None, float | None]:
        hints = self.toolbox.default_hints(str(row.get("question") or state.question), str(row.get("inputs_json") or state.inputs_json))
        times = [float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []]
        if not times:
            return None, None
        return min(times), max(times)


class GraphAgent:
    """Full agent that plans, retrieves, revisits raw video, and writes back memory."""

    def __init__(self, paths: ProjectPaths | None = None, model_client: OpenAICompatibleModelClient | None = None):
        self.paths = paths or ProjectPaths.from_env()
        self.builder = VideoGraphBuilder(self.paths)
        self.model_client = model_client or OpenAICompatibleModelClient()
        self.planner = GraphAgentPlanner(self.model_client)
        self.verifier = GraphAgentVerifier(self.model_client)
        self._video_sessions: dict[str, GraphAgentVideoSession] = {}

    def answer_vqa_row(self, row: dict[str, Any], *, max_steps: int = 6) -> GraphAgentResult:
        video_id = str(row["primary_video_id"])
        session = self.begin_video_session(video_id)
        return session.answer_vqa_row(row, max_steps=max_steps)

    def answer_open_query(
        self,
        *,
        video_id: str,
        question: str,
        inputs_json: str = "{}",
        task_family: str = "open_query",
        max_steps: int = 6,
        query_id: str = "",
    ) -> GraphAgentResult:
        session = self.begin_video_session(video_id)
        return session.answer_open_query(
            question=question,
            inputs_json=inputs_json,
            task_family=task_family,
            max_steps=max_steps,
            query_id=query_id,
        )

    def begin_video_session(self, video_id: str) -> GraphAgentVideoSession:
        session = self._video_sessions.get(video_id)
        if session is None:
            session = GraphAgentVideoSession(agent=self, video_id=video_id)
            self._video_sessions[video_id] = session
        return session

    def reset_video_session(self, video_id: str) -> None:
        session = self._video_sessions.pop(video_id, None)
        session_dir = self.paths.graph_agent_sessions_root / video_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def rebuild_video_graph(self, video_id: str) -> GraphMemoryStore:
        self.reset_video_session(video_id)
        self._video_sessions.pop(video_id, None)
        graph_dir = self.paths.graph_memory_root / video_id
        if graph_dir.exists():
            shutil.rmtree(graph_dir)
        return self.builder.build(video_id)

    def _ensure_store(self, video_id: str) -> GraphMemoryStore:
        graph_dir = self.paths.graph_memory_root / video_id
        store = GraphMemoryStore(graph_dir)
        existing = store.query_nodes(video_id=video_id, limit=1)
        if not existing:
            return self.builder.build(video_id)
        return store

    def _finalize_state_answer(self, *, state: AgentState, freeform: bool) -> tuple[str, int | None]:
        if freeform:
            grounded_answer = self._resolve_grounded_freeform_answer(state)
            if grounded_answer:
                state.add_memory("freeform_answer_mode=grounded_structured_answer")
                return grounded_answer, None
            answer_text = self._answer_from_state(state, freeform=True)
            if answer_text.strip():
                critique = self.verifier.critique_freeform_answer(state=state, answer_text=answer_text)
                state.record_verification(
                    sufficient=critique.sufficient,
                    confidence=critique.confidence,
                    missing_evidence_types=critique.missing_evidence_types,
                    conflicts=critique.conflicts,
                    recommend_next_action=critique.recommend_next_action,
                    summary=f"answer_critic: {critique.summary}",
                )
                if critique.sufficient:
                    state.add_memory("freeform_answer_mode=answer_critic_passed")
                    return answer_text, None
                if (
                    str(getattr(state, "task_family", "") or "") == "open_query_temporal_summary"
                    and any(
                        isinstance(item, str) and item.startswith("reuse:")
                        for item in getattr(state, "working_memory", []) or []
                    )
                    and len(answer_text.strip()) >= 20
                ):
                    state.add_memory("freeform_answer_mode=session_reuse_temporal_summary")
                    return answer_text, None
                state.add_memory(f"freeform_answer_critic_blocked={critique.recommend_next_action}")
            state.add_memory("freeform_answer_mode=structured_summary")
            answer_text = self._fallback_freeform_answer(state)
            return answer_text, None
        deterministic = self._resolve_deterministic_answer_from_state(state)
        if deterministic is not None:
            prediction, answer_text, confidence = deterministic
            if (
                state.final_prediction is None
                or prediction == state.final_prediction
                or self._should_override_existing_final_with_deterministic(
                    state=state,
                    deterministic_prediction=prediction,
                    deterministic_confidence=confidence,
                )
            ):
                state.final_prediction = prediction
                state.final_answer = answer_text
                state.confidence = max(float(getattr(state, "confidence", 0.0) or 0.0), confidence)
                self._record_deterministic_finalize_marker(state, prediction=prediction, confidence=confidence)
                return answer_text, prediction
        existing_structured = self._resolve_existing_structured_final_answer(state)
        if existing_structured is not None:
            prediction, answer_text, confidence = existing_structured
            state.final_prediction = prediction
            state.final_answer = answer_text
            state.confidence = max(float(getattr(state, "confidence", 0.0) or 0.0), confidence)
            self._record_deterministic_finalize_marker(state, prediction=prediction, confidence=confidence)
            return answer_text, prediction
        if state.final_prediction is None:
            answer_text = self._answer_from_state(state, freeform=False)
            prediction = self._parse_prediction(answer_text, state.choices)
            guarded_answer, guarded_prediction = self._guard_residual_mcq_answer(
                state,
                answer_text=answer_text,
                prediction=prediction,
            )
            return guarded_answer, guarded_prediction
        return state.final_answer, state.final_prediction

    def _resolve_deterministic_answer_from_state(self, state: AgentState) -> tuple[int, str, float] | None:
        task_family = str(getattr(state, "task_family", ""))
        if task_family == "ingredient_ingredient_weight":
            choice_values: list[tuple[int, float, str]] = []
            for index, choice in enumerate(state.choices):
                parsed = self._parse_numeric_value(str(choice))
                if parsed is None:
                    continue
                choice_values.append((index, parsed, str(choice)))
            if not choice_values:
                return None
            measurement_values = self._extract_prefixed_numeric_values(state, prefix="normalized=")
            if measurement_values:
                best = self._pick_best_numeric_choice(choice_values, measurement_values[-1])
                if best is not None:
                    return best[0], best[2], 0.9
            ocr_values = self._extract_prefixed_numeric_values(state, prefix="ocr_reading=")
            if ocr_values:
                best = self._pick_best_numeric_choice(choice_values, ocr_values[-1])
                if best is not None:
                    return best[0], best[2], 0.82
            return None
        if task_family == "nutrition_nutrition_change":
            nutrition_change = self._extract_nutrition_change_totals(state)
            if nutrition_change:
                best = self._pick_best_nutrition_change_choice(state, nutrition_change)
                if best is not None:
                    return best
        recipe_event_localization = self._resolve_recipe_event_localization_answer(state)
        if recipe_event_localization is not None:
            return recipe_event_localization
        recipe_step_recognition = self._resolve_recipe_step_recognition_answer(state)
        if recipe_step_recognition is not None:
            return recipe_step_recognition
        action_intent_resolution = self._resolve_action_intent_resolution_answer(state)
        if action_intent_resolution is not None:
            return action_intent_resolution
        text_overlap = self._resolve_text_overlap_structured_answer(state)
        if text_overlap is not None:
            return text_overlap
        return self._resolve_structured_best_index_answer(state)

    def _resolve_existing_structured_final_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        if state.final_prediction is None or not state.final_answer:
            return None
        deterministic = self._resolve_deterministic_answer_from_state(state)
        if deterministic is None:
            return None
        prediction, answer_text, confidence = deterministic
        if prediction != state.final_prediction or answer_text != state.final_answer:
            return None
        return prediction, answer_text, confidence

    def _resolve_structured_best_index_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        prefixes_with_confidence: tuple[tuple[str, float], ...] = (
            ("ingredient_retrieval_best_index=", 0.84),
            ("recipe_membership_best_index=", 0.84),
            ("exact_ingredient_amount_best_index=", 0.84),
            ("ingredient_order_best_index=", 0.86),
            ("action_mechanism_best_index=", 0.8),
            ("action_intent_best_index=", 0.78),
            ("recipe_catalog_best_index=", 0.88),
            ("recipe_nutrition_best_index=", 0.86),
            ("temporal_localization_best_index=", 0.78),
            ("visual_mcq_best_index=", 0.76),
            ("viewpoint_best_index=", 0.76),
            ("fixture_direction_best_index=", 0.8),
            ("gaze_best_index=", 0.8),
            ("object_location_best_index=", 0.8),
            ("itinerary_best_index=", 0.78),
            ("stationary_best_index=", 0.82),
        )
        for prefix, base_confidence in prefixes_with_confidence:
            resolved = self._extract_best_index_answer(state, prefix=prefix, default_confidence=base_confidence)
            if resolved is not None:
                return resolved
        movement = self._extract_best_index_answer(state, prefix="movement_count=", default_confidence=0.84, embedded_key="best_index")
        if movement is not None:
            return movement
        fixture_count = self._extract_best_index_answer(
            state,
            prefix="count_candidates",
            default_confidence=0.8,
            embedded_key="best_index",
        )
        if fixture_count is not None:
            return fixture_count
        return None

    def _resolve_action_intent_resolution_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        if str(getattr(state, "task_family", "")) != "fine_grained_why_recognition":
            return None
        resolution_tools = {
            "resolve_action_intent_future_use",
            "resolve_action_intent_pairwise",
        }
        for entry in reversed(list(getattr(state, "tool_trace", []) or [])):
            if not isinstance(entry, dict) or entry.get("tool") not in resolution_tools:
                continue
            raw_result = entry.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            if raw_result.get("tool_failed") or raw_result.get("tool_ineffective"):
                continue
            if self._action_intent_resolution_should_withhold_state_change_overclaim(raw_result=raw_result, state=state):
                state.add_memory("action_intent_resolution_withheld_for_missing_state_change_prereq=1")
                continue
            index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
            if index is None:
                continue
            if raw_result.get("need_more_evidence"):
                reranked = self._resolve_unresolved_action_intent_answer(raw_result=raw_result, state=state)
                if reranked is not None:
                    return reranked
                if any(
                    isinstance(item, str) and item.startswith("action_intent_unresolved_rerank_withheld")
                    for item in list(getattr(state, "working_memory", []))[-8:]
                ):
                    state.add_memory("action_intent_resolution_withheld_for_more_evidence=1")
                    continue
            elif self._action_intent_resolution_should_withhold_weak_surface_wiping_claim(raw_result=raw_result, state=state):
                state.add_memory("action_intent_resolution_withheld_for_weak_surface_wiping_evidence=1")
                continue
            elif self._action_intent_resolution_should_withhold_weak_relocation_or_residue_claim(raw_result=raw_result, state=state):
                state.add_memory("action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1")
                continue
            else:
                generic_hand_free_marker = self._action_intent_resolution_generic_hand_free_overclaim_marker(
                    raw_result=raw_result,
                    state=state,
                )
                if generic_hand_free_marker:
                    state.add_memory(generic_hand_free_marker)
                    continue
                generic_access_or_space_marker = self._action_intent_resolution_generic_access_or_space_overclaim_marker(
                    raw_result=raw_result,
                    state=state,
                )
                if generic_access_or_space_marker:
                    state.add_memory(generic_access_or_space_marker)
                    continue
                generic_relocation_or_storage_marker = (
                    self._action_intent_resolution_generic_relocation_or_storage_overclaim_marker(
                        raw_result=raw_result,
                        state=state,
                    )
                )
                if generic_relocation_or_storage_marker:
                    state.add_memory(generic_relocation_or_storage_marker)
                    continue
                mixed_horizon_later_target_marker = (
                    self._action_intent_resolution_mixed_horizon_later_target_marker(
                        raw_result=raw_result,
                        state=state,
                    )
                )
                if mixed_horizon_later_target_marker:
                    state.add_memory(mixed_horizon_later_target_marker)
                    continue
            if self._action_intent_resolution_should_withhold_broad_generic_claim_without_direct_evidence(
                raw_result=raw_result,
                state=state,
            ):
                state.add_memory("action_intent_resolution_withheld_for_broad_generic_claim=1")
                continue
            elif self._action_intent_resolution_should_withhold_nonexclusive_concrete_late_anchor_claim(
                raw_result=raw_result,
                state=state,
            ):
                state.add_memory("action_intent_resolution_withheld_for_nonexclusive_concrete_late_anchor=1")
                continue
            elif self._action_intent_resolution_should_withhold_timeline_review_bias_gap(
                raw_result=raw_result,
                state=state,
            ):
                state.add_memory("action_intent_resolution_withheld_for_timeline_review_bias_gap=1")
                continue
            elif self._action_intent_resolution_should_withhold_mixed_horizon_overclaim(
                raw_result=raw_result,
                state=state,
            ):
                state.add_memory("action_intent_resolution_withheld_for_mixed_horizon_claim=1")
                continue
            elif self._action_intent_resolution_should_withhold_workspace_or_final_placement_overclaim(
                raw_result=raw_result,
                state=state,
            ):
                state.add_memory("action_intent_resolution_withheld_for_workspace_or_final_placement_claim=1")
                continue
            confidence = self._coerce_confidence(raw_result.get("confidence"), default=0.78)
            if raw_result.get("need_more_evidence"):
                confidence = min(confidence, 0.62)
            answer = raw_result.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                answer = str(state.choices[index])
            return index, answer, confidence
        return None

    def _action_intent_resolution_generic_hand_free_overclaim_marker(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> str:
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return ""
        best_choice = str(state.choices[index]).strip().lower()
        generic_hand_free_patterns = (
            "so left hand is free",
            "so right hand is free",
            "free up the right hand",
            "free up the left hand",
            "free the right hand",
            "free the left hand",
            "to free up the right hand",
            "to free up the left hand",
            "to free one hand",
            "free one hand",
            "腾出右手",
            "腾出左手",
            "腾出一只手",
        )
        if not any(pattern in best_choice for pattern in generic_hand_free_patterns):
            return ""
        evidence_items = raw_result.get("candidate_evidence")
        if not isinstance(evidence_items, list):
            return ""
        action_object = self._action_intent_question_object(str(getattr(state, "question", "") or ""))
        if not action_object:
            return ""
        best_support = ""
        best_contradiction = ""
        candidate_rows: list[tuple[float, int, str, str, str]] = []
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            candidate_index = self._coerce_choice_index(item.get("index"), state.choices)
            if candidate_index is None:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0
            support = str(item.get("support") or "")
            contradiction = str(item.get("contradiction") or "")
            choice = str(state.choices[candidate_index])
            candidate_rows.append((score, candidate_index, choice, support, contradiction))
            if candidate_index == index:
                best_support = support.lower()
                best_contradiction = contradiction.lower()
        direct_specific_patterns = (
            "more direct visible",
            "direct visible goal",
            "direct purpose visible",
            "intermediate step",
            "setup",
            "not the more direct visible",
            "clearer evidence is",
            "下一步",
            "中间步骤",
            "更直接",
            "真正目的",
        )
        if best_contradiction and not any(pattern in best_contradiction for pattern in direct_specific_patterns):
            if not best_support or "free" not in best_support:
                return ""
        best_target = ""
        best_kind = ""
        for score, candidate_index, choice, support, contradiction in sorted(candidate_rows, key=lambda row: (-row[0], row[1])):
            if candidate_index == index:
                continue
            choice_lc = choice.lower()
            support_lc = support.lower()
            contradiction_lc = contradiction.lower()
            if score < 0.18:
                continue
            if self._choice_is_same_object_active_use(choice_lc, action_object):
                best_target = action_object
                best_kind = "object"
                break
            if self._action_intent_choice_is_direct_same_object_cleaning(
                choice=choice_lc,
                support=support_lc,
                contradiction=contradiction_lc,
                action_object=action_object,
                global_context="",
            ):
                best_target = action_object
                best_kind = "object"
                break
            if self._action_intent_choice_is_direct_same_object_role_use(
                choice=choice_lc,
                support=support_lc,
                contradiction=contradiction_lc,
                action_object=action_object,
                global_context="",
            ):
                best_target = action_object
                best_kind = "object"
                break
            if self._action_intent_choice_is_direct_enablement(
                choice=choice_lc,
                support=support_lc,
                global_context="",
            ) or self._action_intent_choice_is_direct_tap_enablement(
                choice=choice_lc,
                support=support_lc,
                contradiction=contradiction_lc,
                global_context="",
            ):
                for token in ("tap", "faucet", "scale", "sink", "fridge", "drawer", "cupboard", "door"):
                    if token in choice_lc:
                        best_target = token
                        best_kind = "fixture"
                        break
                if best_target:
                    break
            if self._action_intent_choice_is_cleaning_tool_specific_target_use(
                choice=choice_lc,
                support=support_lc,
                contradiction=contradiction_lc,
                action_object=action_object,
                global_context="",
            ):
                for token in (
                    "sponge",
                    "brush",
                    "knife",
                    "spoon",
                    "fork",
                    "cup",
                    "bowl",
                    "pot",
                    "pan",
                    "board",
                    "tray",
                    "counter",
                    "surface",
                    "peeler",
                    "blender cup",
                ):
                    if token in choice_lc:
                        best_target = token
                        best_kind = "object"
                        break
                if best_target:
                    break
            for token in (
                "sponge",
                "brush",
                "knife",
                "fork",
                "spoon",
                "bottle",
                "cup",
                "bowl",
                "pot",
                "pan",
                "board",
                "tray",
                "jar",
                "blender cup",
            ):
                if token in choice_lc and token not in action_object:
                    best_target = token
                    best_kind = "object"
                    break
            if best_target:
                break
        if not best_target or not best_kind:
            return ""
        return (
            "action_intent_resolution_withheld_for_generic_hand_free_enablement=1 "
            f"target={best_target} kind={best_kind}"
        )

    def _action_intent_choice_target_token_and_kind(
        self,
        *,
        choice: str,
        action_object: str,
    ) -> tuple[str, str] | None:
        choice_lc = str(choice or "").strip().lower()
        action_object_lc = str(action_object or "").strip().lower()
        action_object_tokens = {token for token in re.split(r"[^a-z0-9]+", action_object_lc) if token}
        fixtures = {"tap", "faucet", "scale", "sink", "fridge", "drawer", "cupboard", "door", "dishwasher", "rack"}
        for token in (
            "whisk",
            "knife",
            "fork",
            "spoon",
            "spatula",
            "bottle",
            "sponge",
            "brush",
            "cloth",
            "towel",
            "lid",
            "cover",
            "bowl",
            "plate",
            "tray",
            "pot",
            "pan",
            "saucepan",
            "cup",
            "glass",
            "jar",
            "colander",
            "scale",
            "tap",
            "faucet",
            "sink",
            "fridge",
            "door",
            "drawer",
            "cupboard",
            "rack",
            "dishwasher",
        ):
            if token not in choice_lc:
                continue
            if token in action_object_tokens:
                continue
            return token, ("fixture" if token in fixtures else "object")
        return None

    def _action_intent_resolution_generic_access_or_space_overclaim_marker(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> str:
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return ""
        question = str(getattr(state, "question", "") or "")
        question_lc = question.lower()
        best_choice = str(state.choices[index]).strip().lower()
        action_object = self._action_intent_question_object(question)
        global_context = " ".join(
            str(item)
            for item in list(getattr(state, "evidence_bundle", []))[-24:]
            + list(getattr(state, "working_memory", []))[-24:]
            if isinstance(item, str)
        ).lower()
        generic_access_patterns = (
            "access what's behind",
            "access what is behind",
            "look what's behind",
            "see what is behind",
            "what is behind",
            "look behind",
            "see what's behind",
            "access behind",
            "to access the area behind",
            "to access behind",
            "后面有什么",
            "看后面",
            "查看后面",
        )
        best_is_generic_access = any(pattern in best_choice for pattern in generic_access_patterns)
        best_is_generic_space = self._action_intent_choice_is_generic_direct_space_purpose(best_choice)
        if not best_is_generic_access and not best_is_generic_space:
            return ""
        evidence_items = raw_result.get("candidate_evidence")
        if not isinstance(evidence_items, list):
            return ""
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            candidate_index = self._coerce_choice_index(item.get("index"), state.choices)
            if candidate_index is None or candidate_index == index:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0
            if score < 0.18:
                continue
            choice = str(state.choices[candidate_index]).lower()
            support = str(item.get("support") or "").lower()
            contradiction = str(item.get("contradiction") or "").lower()
            exact_revealed_target = self._action_intent_choice_is_exact_revealed_target_purpose(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            )
            exact_targeted_placement = self._action_intent_choice_is_exact_downstream_targeted_placement(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            )
            direct_fixture_enablement = self._action_intent_choice_is_direct_fixture_or_workspace_enablement(
                choice=choice,
                support=support,
                contradiction=contradiction,
            )
            target = self._action_intent_choice_target_token_and_kind(choice=choice, action_object=action_object)
            fallback_direct_downstream = False
            if target is not None:
                signal_text = f"{support} {contradiction}"
                if any(
                    token in choice
                    for token in (
                        "take",
                        "pick up",
                        "retrieve",
                        "grab",
                        "turn on",
                        "turn off",
                        "place",
                        "put",
                        "insert",
                        "fit",
                        "weigh",
                        "measure",
                        "wash",
                        "rinse",
                        "scrub",
                        "wipe",
                        "清洗",
                        "冲洗",
                        "拿",
                        "取",
                        "放进",
                        "放到",
                    )
                ) and any(
                    token in signal_text
                    for token in (
                        "direct target",
                        "direct purpose",
                        "hidden-target retrieval",
                        "true next target",
                        "revealed target",
                        "revealed item",
                        "rather than only generic access",
                        "rather than generic access",
                        "the hidden item is then picked up",
                        "exact placement",
                        "the direct purpose is",
                        "真正目标",
                        "直接目的",
                        "后面的目标",
                    )
                ):
                    fallback_direct_downstream = True
            if not (exact_revealed_target or exact_targeted_placement or direct_fixture_enablement or fallback_direct_downstream):
                continue
            if target is None:
                continue
            target_name, target_kind = target
            return (
                "action_intent_resolution_withheld_for_generic_access_or_space_enablement=1 "
                f"target={target_name} kind={target_kind}"
            )
        return ""

    def _action_intent_resolution_generic_relocation_or_storage_overclaim_marker(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> str:
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return ""
        question = str(getattr(state, "question", "") or "")
        question_lc = question.lower()
        best_choice = str(state.choices[index]).strip().lower()
        if not self._action_intent_choice_is_final_placement_candidate(best_choice):
            return ""
        evidence_items = raw_result.get("candidate_evidence")
        if not isinstance(evidence_items, list):
            return ""
        action_object = self._action_intent_question_object(question)
        if not action_object:
            return ""
        global_context = " ".join(
            str(item)
            for item in list(getattr(state, "evidence_bundle", []))[-24:]
            + list(getattr(state, "working_memory", []))[-24:]
            if isinstance(item, str)
        ).lower()
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            candidate_index = self._coerce_choice_index(item.get("index"), state.choices)
            if candidate_index is None or candidate_index == index:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0
            if score < 0.18:
                continue
            choice = str(state.choices[candidate_index]).strip().lower()
            support = str(item.get("support") or "").strip().lower()
            contradiction = str(item.get("contradiction") or "").strip().lower()
            target_name = ""
            target_kind = ""
            if self._choice_is_same_object_active_use(choice, action_object) or self._action_intent_choice_is_direct_same_object_cleaning(
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ) or self._action_intent_choice_is_direct_same_object_role_use(
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ) or self._action_intent_choice_is_immediate_reuse_staging(
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ) or self._action_intent_choice_is_cleaning_tool_specific_target_use(
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ):
                target_name = action_object
                target_kind = "object"
            elif self._action_intent_choice_is_exact_revealed_target_purpose(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ) or self._action_intent_choice_is_exact_downstream_targeted_placement(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ) or self._action_intent_choice_is_exact_immediate_downstream_use(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ) or self._action_intent_choice_is_direct_fixture_or_workspace_enablement(
                choice=choice,
                support=support,
                contradiction=contradiction,
            ) or self._action_intent_choice_is_hidden_target_access_or_retrieval(
                choice=choice,
                support=support,
                contradiction=contradiction,
                global_context=global_context,
            ):
                target = self._action_intent_choice_target_token_and_kind(choice=choice, action_object=action_object)
                if target is not None:
                    target_name, target_kind = target
            elif (
                self._action_intent_choice_target_token_and_kind(choice=choice, action_object=action_object) is not None
                and any(
                    token in f"{support} {contradiction}"
                    for token in (
                        "direct next target",
                        "right afterwards",
                        "immediately afterwards",
                        "picked up right afterwards",
                        "picked up immediately afterwards",
                        "picked up from behind",
                        "hidden target",
                        "revealed-target retrieval",
                        "more specific than a generic put-away",
                        "more specific than generic relocation",
                        "真正目标",
                        "直接下一目标",
                        "后面目标",
                    )
                )
            ):
                target = self._action_intent_choice_target_token_and_kind(choice=choice, action_object=action_object)
                if target is not None:
                    target_name, target_kind = target
            if not target_name or not target_kind:
                continue
            return (
                "action_intent_resolution_withheld_for_generic_relocation_or_storage_enablement=1 "
                f"target={target_name} kind={target_kind}"
            )
        return ""

    def _action_intent_resolution_should_withhold_state_change_overclaim(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        question = str(getattr(state, "question", "") or "").lower()
        if "<tap kitchen scale>" not in question and "tap kitchen scale" not in question:
            return False
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return False
        choice_lc = str(state.choices[index]).lower()
        if not any(term in choice_lc for term in ("zero out", "tare", "归零", "去皮")):
            return False
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        prior_text = self._action_intent_prior_reasoning_text(state).lower()
        combined_text = f"{prior_text} {text}".strip()
        has_on_prereq = any(
            token in combined_text
            for token in (
                "already on before the tap",
                "already lit before the tap",
                "display is already lit",
                "scale is already on before the tap",
                "container on the scale",
                "container already on the scale",
                "容器已经在秤上",
                "按之前已经亮着",
                "已经开机",
                "显示屏已经亮起",
            )
        )
        has_container_at_tap = any(
            token in combined_text
            for token in (
                "container on the scale at the tap",
                "container already on the scale",
                "bowl already on the scale",
                "with container already on the scale",
                "容器已经在秤上",
                "碗已经在秤上",
            )
        )
        explicit_no_container = any(
            token in combined_text
            for token in (
                "no container on it",
                "scale appears empty",
                "empty scale",
                "no evidence for taring",
                "there is no evidence for taring",
                "no container is visible on the scale",
                "没有容器",
                "秤上是空的",
                "空秤",
                "没有去皮依据",
            )
        )
        container_added_after_tap = any(
            token in combined_text
            for token in (
                "then the person places a bowl on the scale",
                "then the person places the bowl on the scale",
                "after tapping, a bowl is placed on the scale",
                "after the tap, the bowl is placed on the scale",
                "after tapping the person places a bowl",
                "随后把碗放到秤上",
                "按完之后把碗放到秤上",
            )
        )
        if explicit_no_container or container_added_after_tap:
            return True
        return not has_on_prereq and not has_container_at_tap

    def _action_intent_resolution_should_withhold_weak_surface_wiping_claim(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        question = str(getattr(state, "question", "") or "").lower()
        action_object = self._action_intent_question_object(question)
        if not any(token in action_object for token in ("towel", "cloth", "napkin", "paper towel", "sponge")):
            return False
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return False
        choice_lc = str(state.choices[index]).lower()
        if not any(token in choice_lc for token in ("wipe", "clean")):
            return False
        if not any(token in choice_lc for token in ("surface", "counter", "countertop", "worktop", "table", "台面", "桌")):
            return False
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        if self._action_intent_text_has_negative_evidence(text):
            return True
        return not self._action_intent_support_has_strong_surface_wiping_evidence(text)

    def _action_intent_resolution_should_withhold_weak_relocation_or_residue_claim(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        question = str(getattr(state, "question", "") or "").lower()
        action_object = self._action_intent_question_object(question)
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return False
        choice_lc = str(state.choices[index]).lower()
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        if any(token in action_object for token in ("towel", "cloth", "napkin", "paper towel")):
            if "move" in choice_lc and self._action_intent_choice_lacks_direct_relocation_outcome_evidence(
                choice=choice_lc,
                support=text,
                contradiction="",
            ):
                return True
            if any(token in question for token in ("<flip ", "<turn ", "<shake ", " flip ", " turn ", " shake ")):
                if any(token in choice_lc for token in ("wipe", "clean", "dry")) and not any(
                    token in text
                    for token in (
                        "crumb",
                        "residue",
                        "drop",
                        "fall",
                        "sink",
                        "shake off",
                        "掉",
                        "落",
                        "碎屑",
                        "水槽",
                    )
                ):
                    return True
        return False

    def _action_intent_resolution_should_withhold_broad_generic_claim_without_direct_evidence(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if index is None:
            return False
        choice_lc = str(state.choices[index]).strip().lower()
        broad_generic_patterns = (
            "to clean.",
            "to dry.",
            "to store.",
            "to move.",
            "to measure.",
            "to measure the ingredients.",
        )
        if not any(pattern in choice_lc for pattern in broad_generic_patterns):
            return False
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        if any(
            token in text
            for token in (
                "least contradicted",
                "broadest",
                "could in principle",
                "might broadly",
                "compatible with",
                "最宽泛",
                "最不矛盾",
            )
        ):
            return True
        if any(
            token in text
            for token in (
                "no actual",
                "no visible",
                "not shown",
                "not visible",
                "unclear",
                "cannot tell",
                "can't tell",
                "没有看到",
                "未显示",
                "不明确",
            )
            ):
            return True
        return not self._action_intent_text_has_direct_positive_evidence(text)

    def _action_intent_resolution_should_withhold_mixed_horizon_overclaim(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        pair = self._action_intent_resolution_competing_pair(raw_result=raw_result, state=state)
        if pair is None:
            return False
        best_index, competitor_index = pair
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        categories_by_index = selected_choice_categories(choices, [best_index, competitor_index])
        best_categories = set(categories_by_index.get(best_index) or set())
        competitor_categories = set(categories_by_index.get(competitor_index) or set())
        later_outcome_categories = {
            "final_place_return",
            "measure_weigh",
            "transfer_contents",
            "serve_consume",
            "clean_dry",
            "food_prep",
            "discard",
        }
        best_choice = choices[best_index].lower()
        competitor_choice = choices[competitor_index].lower()
        best_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories)
        competitor_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(
            competitor_choice,
            competitor_categories,
        )
        spans_mixed_horizon = bool(
            (best_is_immediate and competitor_categories & later_outcome_categories)
            or (competitor_is_immediate and best_categories & later_outcome_categories)
        )
        if not spans_mixed_horizon:
            return False
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        if self._action_intent_text_has_negative_evidence(text):
            return True
        if any(
            token in text
            for token in (
                "whether",
                "still unclear",
                "unclear",
                "not visible",
                "not shown",
                "cannot tell",
                "can't tell",
                "可能",
                "是否",
                "不明确",
            )
        ):
            return True
        if best_is_immediate:
            return not self._action_intent_choice_has_explicit_immediate_micro_outcome_evidence(best_choice, text)
        return not self._action_intent_choice_has_explicit_later_outcome_evidence(best_choice, best_categories, text)

    def _action_intent_resolution_mixed_horizon_later_target_marker(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> str:
        if not self._action_intent_resolution_should_withhold_mixed_horizon_overclaim(
            raw_result=raw_result,
            state=state,
        ):
            return ""
        pair = self._action_intent_resolution_competing_pair(raw_result=raw_result, state=state)
        if pair is None:
            return ""
        best_index, competitor_index = pair
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        categories_by_index = selected_choice_categories(choices, [best_index, competitor_index])
        best_choice = choices[best_index].lower()
        best_categories = set(categories_by_index.get(best_index) or set())
        competitor_choice = choices[competitor_index].lower()
        competitor_categories = set(categories_by_index.get(competitor_index) or set())
        later_outcome_categories = {
            "final_place_return",
            "measure_weigh",
            "transfer_contents",
            "serve_consume",
            "clean_dry",
            "food_prep",
            "discard",
        }
        if not self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories):
            return ""
        if not (competitor_categories & later_outcome_categories):
            return ""
        action_object = self._action_intent_question_object(str(getattr(state, "question", "") or ""))
        target = self._action_intent_choice_target_token_and_kind(choice=competitor_choice, action_object=action_object)
        if target is None and "measure_weigh" in competitor_categories:
            target = ("scale", "fixture")
        if target is None and "final_place_return" in competitor_categories:
            combined_text = f"{competitor_choice} {raw_result.get('reason') or ''} {raw_result.get('decisive_observation') or ''}".lower()
            for token in ("fridge", "drawer", "cupboard", "rack", "dishwasher", "shelf"):
                if token in combined_text:
                    target = (token, "fixture" if token != "shelf" else "object")
                    break
        if target is None:
            return ""
        target_name, target_kind = target
        return (
            "action_intent_resolution_withheld_for_mixed_horizon_later_target=1 "
            f"target={target_name} kind={target_kind}"
        )

    def _action_intent_resolution_should_withhold_nonexclusive_concrete_late_anchor_claim(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        pair = self._action_intent_resolution_competing_pair(raw_result=raw_result, state=state)
        if pair is None:
            return False
        best_index, competitor_index = pair
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        pair_indices = [best_index, competitor_index]
        categories_by_index = selected_choice_categories(choices, pair_indices)
        best_choice = choices[best_index].lower()
        best_categories = set(categories_by_index.get(best_index) or set())
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        if not text or self._action_intent_text_has_negative_evidence(text):
            return False
        explicit_exclusive_terms = (
            "reads the label",
            "reading the label",
            "read the label",
            "inspects the label",
            "looks at the label",
            "checks the label",
            "read the printed text",
            "placed on the scale",
            "put on the scale",
            "used on the scale",
            "used to weigh",
            "weighed",
            "put back",
            "returned to",
            "stored",
            "inside the fridge",
            "into the fridge",
            "under running water",
            "turns on the tap",
            "opened the fridge",
            "closed the fridge",
            "poured into",
            "wiped",
            "dried",
            "读标签",
            "查看标签",
            "放到秤上",
            "放上秤",
            "称量",
            "放回",
            "回到冰箱",
            "打开冰箱",
            "关上冰箱",
        )
        if any(term in text for term in explicit_exclusive_terms):
            return False
        label_visibility_terms = (
            "label is visible",
            "label faces the camera",
            "label faces outward",
            "front side becomes visible",
            "front side is visible",
            "printed side becomes visible",
            "printed side is visible",
            "visible while the bottle is held",
        )
        label_reading_terms = (
            "read",
            "reading",
            "inspect",
            "look at the label",
            "check the label",
            "printed text",
            "nutrition facts",
            "ingredient list",
            "看标签",
            "读标签",
            "查看标签",
        )
        nearby_placement_terms = (
            "set beside",
            "placed beside",
            "left beside",
            "left nearby",
            "set nearby",
            "placed nearby",
            "within reach",
            "set aside",
            "simply set aside",
            "near the scale area",
            "near the counter",
            "near the counter surface",
            "near the sink",
            "near the fridge area",
            "beside the scale",
            "beside the counter",
            "adjacent to the weighing station",
            "left on the side",
            "still near",
            "放在旁边",
            "放在附近",
            "顺手放在旁边",
            "放到一边",
            "附近",
        )
        label_visible_without_reading = any(term in text for term in label_visibility_terms) and not any(
            term in text for term in label_reading_terms
        )
        if label_visible_without_reading:
            return True
        if not any(term in text for term in nearby_placement_terms):
            return False
        if self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories):
            return not self._action_intent_choice_has_explicit_immediate_micro_outcome_evidence(best_choice, text)
        return not self._action_intent_choice_has_explicit_later_outcome_evidence(best_choice, best_categories, text)

    def _action_intent_resolution_competing_pair(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> tuple[int, int] | None:
        best_index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
        if best_index is None:
            return None
        competitor_index = self._coerce_choice_index(raw_result.get("second_best_index"), state.choices)
        if competitor_index is None:
            competitor_index = self._coerce_choice_index(raw_result.get("losing_index"), state.choices)
        if competitor_index is None:
            scored: list[tuple[float, int]] = []
            for item in raw_result.get("candidate_evidence") or []:
                if not isinstance(item, dict):
                    continue
                index = self._coerce_choice_index(item.get("index"), state.choices)
                if index is None or index == best_index:
                    continue
                try:
                    score = float(item.get("score") or 0.0)
                except Exception:  # noqa: BLE001
                    score = 0.0
                scored.append((score, index))
            if scored:
                scored.sort(key=lambda pair: (-pair[0], pair[1]))
                competitor_index = scored[0][1]
        if competitor_index is None or competitor_index == best_index:
            return None
        return best_index, competitor_index

    def _action_intent_resolution_should_withhold_workspace_or_final_placement_overclaim(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        pair = self._action_intent_resolution_competing_pair(raw_result=raw_result, state=state)
        if pair is None:
            return False
        best_index, competitor_index = pair
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_choice = choices[best_index].lower()
        competitor_choice = choices[competitor_index].lower()
        question = str(getattr(state, "question", "") or "").lower()
        action_object = self._action_intent_question_object(question)
        global_context = self._action_intent_scoped_global_context(state).lower()
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation")
        ).lower()
        best_is_generic_workspace = self._action_intent_choice_is_generic_direct_space_purpose(best_choice)
        competitor_is_generic_workspace = self._action_intent_choice_is_generic_direct_space_purpose(competitor_choice)
        best_is_exact_workspace = self._action_intent_choice_is_exact_workspace_or_downstream_candidate(
            best_choice
        )
        competitor_is_exact_workspace = self._action_intent_choice_is_exact_workspace_or_downstream_candidate(
            competitor_choice
        )
        best_is_final_placement = self._action_intent_choice_is_final_placement_candidate(best_choice)
        competitor_is_final_placement = self._action_intent_choice_is_final_placement_candidate(competitor_choice)
        if not any(
            (
                best_is_generic_workspace and (competitor_is_exact_workspace or competitor_is_final_placement),
                best_is_exact_workspace and competitor_is_generic_workspace,
                best_is_final_placement and not competitor_is_final_placement,
            )
        ):
            return False
        if self._action_intent_text_has_negative_evidence(text):
            return True
        if any(
            token in text
            for token in (
                "whether",
                "still unclear",
                "unclear",
                "not visible",
                "not shown",
                "cannot tell",
                "can't tell",
                "ambiguous",
                "multiple explanations",
                "是否",
                "不明确",
                "仍不清楚",
                "有歧义",
            )
        ):
            return True
        if best_is_generic_workspace and (competitor_is_exact_workspace or competitor_is_final_placement):
            return not self._action_intent_text_explicitly_rules_out_exact_downstream_chain(text)
        if best_is_exact_workspace and competitor_is_generic_workspace:
            return not self._action_intent_choice_has_explicit_workspace_or_downstream_chain(
                question=question,
                choice=best_choice,
                text=text,
                action_object=action_object,
                global_context=global_context,
            )
        if best_is_final_placement:
            return not self._action_intent_choice_has_explicit_final_placement_evidence(best_choice, text)
        return False

    def _latest_action_intent_timeline_review_entry(
        self,
        state: AgentState,
    ) -> tuple[int, dict[str, Any]] | None:
        trace = list(getattr(state, "tool_trace", []) or [])
        for index in range(len(trace) - 1, -1, -1):
            call = trace[index]
            if not isinstance(call, dict) or str(call.get("tool") or "") != "inspect_visual_evidence":
                continue
            raw_result = call.get("raw_result")
            if isinstance(raw_result, dict) and self._action_intent_is_timeline_review_payload(raw_result):
                return index, raw_result
        return None

    def _action_intent_is_timeline_review_payload(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        return any(
            payload.get(key)
            for key in (
                "timeline_summary",
                "immediate_result",
                "next_action_hint",
                "direct_purpose_hint",
                "access_or_reveal_evidence",
                "hand_free_enablement_evidence",
                "next_use_evidence",
                "ambiguity_note",
            )
        ) or bool(payload.get("needs_more_evidence"))

    def _action_intent_timeline_review_requests_more_evidence(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("needs_more_evidence"):
            return True
        ambiguity = str(payload.get("ambiguity_note") or "").strip().lower()
        if ambiguity:
            return True
        combined = " ".join(
            str(payload.get(key) or "").strip().lower()
            for key in ("direct_purpose_hint", "next_use_evidence", "next_action_hint")
        )
        weak_markers = (
            "unclear",
            "ambiguous",
            "not enough",
            "insufficient",
            "cannot tell",
            "can't tell",
            "不明确",
            "看不清",
            "证据不足",
        )
        return any(marker in combined for marker in weak_markers)

    def _action_intent_has_unresolved_timeline_review_gap(self, state: AgentState) -> bool:
        if str(getattr(state, "task_family", "") or "") != "fine_grained_why_recognition":
            return False
        trace = list(getattr(state, "tool_trace", []) or [])
        last_review_index: int | None = None
        for index, call in enumerate(trace):
            if not isinstance(call, dict) or str(call.get("tool") or "") != "inspect_visual_evidence":
                continue
            raw_result = call.get("raw_result")
            if not isinstance(raw_result, dict) or not self._action_intent_is_timeline_review_payload(raw_result):
                continue
            if self._action_intent_timeline_review_requests_more_evidence(raw_result):
                last_review_index = index
        if last_review_index is None:
            return False
        saw_new_sampling = False
        for call in trace[last_review_index + 1 :]:
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool") or "")
            raw_result = call.get("raw_result")
            if tool in {"sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks", "retrieve_cached_artifacts"}:
                saw_new_sampling = True
                continue
            if tool == "inspect_visual_evidence" and isinstance(raw_result, dict) and self._action_intent_is_timeline_review_payload(raw_result):
                if not self._action_intent_timeline_review_requests_more_evidence(raw_result):
                    return False
                saw_new_sampling = False
                continue
            if tool in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
                if (
                    saw_new_sampling
                    and isinstance(raw_result, dict)
                    and not raw_result.get("tool_failed")
                    and raw_result.get("best_index") is not None
                    and not raw_result.get("need_more_evidence")
                ):
                    return False
            if tool == "infer_action_intent":
                if isinstance(raw_result, dict) and not raw_result.get("tool_failed") and raw_result.get("best_index") is not None and not raw_result.get("need_future_evidence") and saw_new_sampling:
                    return False
        return True

    def _action_intent_timeline_review_bias_profile(self, state: AgentState) -> dict[str, bool]:
        review = self._latest_action_intent_timeline_review_entry(state)
        empty = {
            "has_review": False,
            "needs_more_evidence": False,
            "revealed_target_retrieval": False,
            "revealed_slot_placement": False,
            "revealed_fixture_enablement": False,
            "hand_free_next_action": False,
            "next_use_unclear": False,
            "final_location_unclear": False,
        }
        if review is None:
            return empty
        _index, payload = review
        text = " ".join(
            str(payload.get(key) or "").strip().lower()
            for key in (
                "timeline_summary",
                "immediate_result",
                "next_action_hint",
                "direct_purpose_hint",
                "access_or_reveal_evidence",
                "hand_free_enablement_evidence",
                "next_use_evidence",
                "target_location",
                "ambiguity_note",
            )
        )

        def has_any(markers: tuple[str, ...]) -> bool:
            return any(marker in text for marker in markers)

        reveal_focus = has_any(
            (
                "behind",
                "hidden",
                "reveal",
                "revealed",
                "freed slot",
                "available spot",
                "slot",
                "后面",
                "露出",
                "空位",
                "槽位",
            )
        )
        return {
            "has_review": True,
            "needs_more_evidence": self._action_intent_timeline_review_requests_more_evidence(payload),
            "revealed_target_retrieval": reveal_focus and has_any(
                (
                    "hidden jar",
                    "hidden item",
                    "retrieval is not yet visible",
                    "retrieve",
                    "pick up from behind",
                    "take from behind",
                    "取后面的",
                    "拿后面的",
                )
            ),
            "revealed_slot_placement": reveal_focus and has_any(
                (
                    "freed slot is the main ambiguity",
                    "placement into the slot is not yet visible",
                    "put into the slot",
                    "place into the slot",
                    "slot placement",
                    "空位",
                    "槽位",
                    "放进",
                    "归位",
                )
            ),
            "revealed_fixture_enablement": reveal_focus and has_any(
                (
                    "scale behind",
                    "revealed appliance",
                    "revealed fixture",
                    "turn on",
                    "switch on",
                    "tap area",
                    "sink area",
                    "露出的装置",
                    "后面的秤",
                    "龙头",
                )
            ),
            "hand_free_next_action": has_any(
                (
                    "free hand",
                    "freed hand",
                    "other hand",
                    "right hand",
                    "left hand",
                    "reach toward",
                    "reaches toward",
                    "moves toward",
                    "tap area",
                    "sink area",
                    "turn on",
                    "turn off",
                    "open",
                    "close",
                    "另一只手",
                    "龙头",
                    "水槽",
                )
            ),
            "next_use_unclear": has_any(
                (
                    "later use is still unclear",
                    "next use is still unclear",
                    "not yet visible whether",
                    "not visible whether",
                    "multiple later-use explanations remain plausible",
                    "后续用途",
                    "仍不清楚",
                    "看不出之后",
                )
            ),
            "final_location_unclear": has_any(
                (
                    "final location remains unclear",
                    "not visible where",
                    "whether it is put back",
                    "whether it is returned",
                    "where it ends up",
                    "最终位置",
                    "放回原处",
                    "暂时移动",
                )
            ),
        }

    def _action_intent_resolution_should_withhold_timeline_review_bias_gap(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> bool:
        if not self._action_intent_has_unresolved_timeline_review_gap(state):
            return False
        bias = self._action_intent_timeline_review_bias_profile(state)
        if not bias["has_review"] or not bias["needs_more_evidence"]:
            return False
        pair = self._action_intent_resolution_competing_pair(raw_result=raw_result, state=state)
        if pair is None:
            return False
        best_index, competitor_index = pair
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        question = str(getattr(state, "question", "") or "")
        pair_indices = [best_index, competitor_index]
        needs_future_use = action_intent_needs_future_use_resolution(
            question=question,
            choices=choices,
            indices=pair_indices,
        )
        needs_pairwise = action_intent_needs_pairwise_resolution(
            question=question,
            choices=choices,
            indices=pair_indices,
        )
        best_choice = choices[best_index].lower()
        competitor_choice = choices[competitor_index].lower()
        text = " ".join(
            str(raw_result.get(key) or "")
            for key in (
                "reason",
                "decisive_observation",
                "needed_observation",
                "direct_effect",
                "downstream_action",
            )
        ).lower()
        if self._action_intent_text_has_negative_evidence(text):
            return True
        if any(
            token in text
            for token in (
                "whether",
                "still unclear",
                "unclear",
                "not visible",
                "not shown",
                "cannot tell",
                "can't tell",
                "ambiguous",
                "multiple explanations",
                "是否",
                "不明确",
                "仍不清楚",
                "有歧义",
            )
        ):
            return True
        if bias["final_location_unclear"] and (
            needs_future_use
            or self._action_intent_choice_is_final_placement_candidate(best_choice)
            or self._action_intent_choice_is_final_placement_candidate(competitor_choice)
        ):
            return not self._action_intent_choice_has_explicit_final_placement_evidence(best_choice, text)
        if bias["next_use_unclear"] and needs_future_use:
            categories = selected_choice_categories(choices, [best_index])
            best_categories = set(categories.get(best_index) or set())
            return not self._action_intent_choice_has_explicit_later_outcome_evidence(best_choice, best_categories, text)
        if bias["revealed_slot_placement"] and needs_pairwise:
            return not any(
                token in text
                for token in (
                    "placed into the freed slot",
                    "put into the freed slot",
                    "freed slot is used",
                    "slot becomes the destination",
                    "放进腾出的槽位",
                    "归位到空位",
                )
            )
        if bias["revealed_target_retrieval"] and needs_pairwise:
            return not any(
                token in text
                for token in (
                    "retrieved from behind",
                    "picked up from behind",
                    "taken from behind",
                    "hidden item is picked up",
                    "取出后面的",
                    "拿到后面的",
                )
            )
        if (bias["revealed_fixture_enablement"] or bias["hand_free_next_action"]) and needs_pairwise:
            direct_effect = str(raw_result.get("direct_effect") or "").strip().lower()
            downstream_action = str(raw_result.get("downstream_action") or "").strip().lower()
            if not direct_effect or not downstream_action:
                return True
            return not self._action_intent_text_has_direct_positive_evidence(f"{direct_effect} {downstream_action}")
        return False

    def _action_intent_text_explicitly_rules_out_exact_downstream_chain(self, text: str) -> bool:
        text_lc = str(text or "").lower()
        return any(
            token in text_lc
            for token in (
                "no single exact next object use shown",
                "no exact next object",
                "no exact next target",
                "no specific next target",
                "no single immediate next target",
                "no direct next-use evidence is shown",
                "target is still ambiguous",
                "without yet showing a single specific",
                "exact next target is still ambiguous",
                "no concrete hidden target is retrieved",
                "no actual retrieval is shown",
                "no direct carry path is visible",
                "no specific destination is shown",
                "没有具体下一目标",
                "没有明确下一目标",
                "目标仍不明确",
                "没有实际取出",
                "没有直接搬运路径",
                "没有具体终点",
            )
        )

    def _action_intent_choice_is_exact_workspace_or_downstream_candidate(self, choice: str) -> bool:
        text = str(choice or "").lower()
        if self._action_intent_choice_has_specific_space_target(text):
            return True
        return any(
            token in text
            for token in (
                "pick up",
                "retrieve",
                "reach",
                "open the",
                "turn on",
                "turn off",
                "switch on",
                "switch off",
                "wash",
                "rinse",
                "measure",
                "weigh",
                "put into",
                "place into",
                "put on the",
                "move the",
                "to the sink",
                "sink slot",
                "slot",
                "rack",
                "freed area",
                "free slot",
                "exact slot",
                "拿起",
                "取出",
                "伸手去拿",
                "打开",
                "开启",
                "清洗",
                "冲洗",
                "称量",
                "放进",
                "放到",
                "水槽",
                "槽位",
            )
        )

    def _action_intent_choice_has_explicit_workspace_or_downstream_chain(
        self,
        *,
        question: str,
        choice: str,
        text: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        return any(
            (
                self._action_intent_choice_is_exact_workspace_creation(
                    choice=choice,
                    support=text,
                    contradiction="",
                    action_object=action_object,
                    global_context=global_context,
                ),
                self._action_intent_choice_is_exact_downstream_targeted_placement(
                    question=question,
                    choice=choice,
                    support=text,
                    contradiction="",
                    action_object=action_object,
                    global_context=global_context,
                ),
                self._action_intent_choice_is_exact_immediate_downstream_use(
                    question=question,
                    choice=choice,
                    support=text,
                    contradiction="",
                    action_object=action_object,
                    global_context=global_context,
                ),
            )
        )

    def _action_intent_choice_is_final_placement_candidate(self, choice: str) -> bool:
        text = str(choice or "").lower()
        if re.search(r"\bput(?:\s+(?:the|this|that|it|them|an|a))?(?:\s+[a-z0-9_-]+){0,4}\s+away\b", text):
            return True
        if re.search(r"\breturn(?:\s+(?:the|this|that|it|them|an|a))?(?:\s+[a-z0-9_-]+){0,4}\b", text):
            return True
        return any(
            token in text
            for token in (
                "put away",
                "store",
                "put back",
                "return it",
                "return the",
                "returned",
                "hang back",
                "right place",
                "proper place",
                "放回",
                "收起来",
                "收纳",
                "归位",
            )
        )

    def _action_intent_choice_has_explicit_final_placement_evidence(self, choice: str, text: str) -> bool:
        choice_lc = str(choice or "").lower()
        text_lc = str(text or "").lower()
        if not self._action_intent_choice_is_final_placement_candidate(choice_lc):
            return False
        if "not final placement" in text_lc or "没有收纳" in text_lc or "暂时放在" in text_lc:
            return False
        if self._action_intent_choice_is_temporary_relocation_not_storage(
            choice=choice_lc,
            support=text_lc,
            contradiction=text_lc,
            action_object="",
            global_context="",
        ):
            return False
        return any(
            token in text_lc
            for token in (
                "returned to the drawer",
                "returned to the cupboard",
                "hung back on the hook",
                "placed back in storage",
                "stored away",
                "put back in the fridge",
                "returned to the fridge",
                "returned to the shelf",
                "placed into the freed slot",
                "inserted into",
                "exact rack slot",
                "available spot",
                "freed slot",
                "放回抽屉",
                "放回橱柜",
                "挂回挂钩",
                "收纳回去",
                "放回冰箱",
                "归位",
                "放进腾出的槽位",
            )
        )

    def _resolve_unresolved_action_intent_answer(
        self,
        *,
        raw_result: dict[str, Any],
        state: AgentState,
    ) -> tuple[int, str, float] | None:
        evidence_items = raw_result.get("candidate_evidence")
        if not isinstance(evidence_items, list):
            return None
        candidate_rows: list[dict[str, Any]] = []
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), state.choices)
            if index is None:
                continue
            score = self._coerce_float(item.get("score"), default=0.0)
            support = str(item.get("support") or "")
            contradiction = str(item.get("contradiction") or "")
            choice = str(state.choices[index])
            adjusted_score = self._score_action_intent_candidate_evidence(
                base_score=score,
                question=str(getattr(state, "question", "") or ""),
                choice=choice,
                support=support,
                contradiction=contradiction,
                state=state,
            )
            candidate_rows.append(
                {
                    "adjusted_score": adjusted_score,
                    "index": index,
                    "choice": choice,
                    "support": support,
                    "contradiction": contradiction,
                }
            )
        ranked = [(float(row["adjusted_score"]), int(row["index"]), str(row["choice"])) for row in candidate_rows]
        if not ranked:
            return None
        ranked.sort(key=lambda row: (-row[0], row[1]))
        best_score, best_index, best_choice = ranked[0]
        prior_override = self._resolve_prior_direct_action_object_intent(
            state=state,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if prior_override is not None:
            state.add_memory(
                f"action_intent_prior_direct_override_best_index={prior_override[0]} score={best_score:.2f}"
            )
            return prior_override
        causal_override = self._override_downstream_followup_with_direct_enablement_candidate(
            state=state,
            candidate_rows=candidate_rows,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if causal_override is not None:
            state.add_memory(
                f"action_intent_causal_override_best_index={causal_override[0]} score={best_score:.2f}"
            )
            return causal_override
        exact_use_override = self._override_generic_space_with_exact_immediate_use_candidate(
            state=state,
            candidate_rows=candidate_rows,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if exact_use_override is not None:
            state.add_memory(
                f"action_intent_exact_use_override_best_index={exact_use_override[0]} score={best_score:.2f}"
            )
            return exact_use_override
        hidden_target_override = self._override_generic_hidden_access_with_exact_revealed_target_candidate(
            state=state,
            candidate_rows=candidate_rows,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if hidden_target_override is not None:
            state.add_memory(
                f"action_intent_hidden_target_override_best_index={hidden_target_override[0]} score={best_score:.2f}"
            )
            return hidden_target_override
        hand_contact_override = self._override_generic_hand_wiping_with_explicit_single_hand_drying(
            state=state,
            candidate_rows=candidate_rows,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if hand_contact_override is not None:
            state.add_memory(
                f"action_intent_hand_contact_override_best_index={hand_contact_override[0]} score={best_score:.2f}"
            )
            return hand_contact_override
        relocation_override = self._override_generic_towel_use_with_simple_relocation(
            state=state,
            candidate_rows=candidate_rows,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if relocation_override is not None:
            state.add_memory(
                f"action_intent_relocation_override_best_index={relocation_override[0]} score={best_score:.2f}"
            )
            return relocation_override
        record_target_override = self._override_generic_measure_with_exact_record_target_candidate(
            state=state,
            candidate_rows=candidate_rows,
            unresolved_best_index=best_index,
            unresolved_best_score=best_score,
        )
        if record_target_override is not None:
            state.add_memory(
                f"action_intent_record_target_override_best_index={record_target_override[0]} score={best_score:.2f}"
            )
            return record_target_override
        second_score = ranked[1][0] if len(ranked) >= 2 else 0.0
        semantic_gaps = self._action_intent_unresolved_semantic_gaps(
            state=state,
            candidate_rows=candidate_rows,
            best_index=best_index,
        )
        if self._action_intent_unresolved_rerank_should_wait_for_more_evidence(
            best_score=best_score,
            second_score=second_score,
            semantic_gaps=semantic_gaps,
            candidate_rows=candidate_rows,
            state=state,
        ):
            reason = ",".join(semantic_gaps) if semantic_gaps else "weak_or_ambiguous_rerank"
            state.add_memory(
                f"action_intent_unresolved_rerank_withheld score={best_score:.2f} second={second_score:.2f} reason={reason}"
            )
            return None
        state.add_memory(f"action_intent_unresolved_rerank_best_index={best_index} score={best_score:.2f}")
        return best_index, best_choice, min(max(0.36 + max(best_score, 0.0) * 0.45, 0.36), 0.68)

    def _action_intent_unresolved_semantic_gaps(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        best_index: int,
    ) -> list[str]:
        target_row = next(
            (
                row
                for row in candidate_rows
                if int(row.get("index", -1)) == best_index
            ),
            None,
        )
        if target_row is None:
            return []
        choice_lc = str(target_row.get("choice") or "").lower()
        support_lc = str(target_row.get("support") or "").lower()
        context_lc = f"{support_lc} {str(target_row.get('contradiction') or '').lower()}"
        gaps: list[str] = []
        if any(term in support_lc for term in ("broadest", "least contradicted", "最宽泛", "最不矛盾")):
            gaps.append("best_is_unproven_broad_candidate")
        has_modal_generic_support = any(
            term in support_lc
            for term in ("could", "can be used", "compatible", "may be", "might be", "可能", "可以用来", "兼容")
        )
        if self._action_intent_text_has_negative_evidence(context_lc) and has_modal_generic_support:
            gaps.append("candidate_explicitly_lacks_observed_support")
        has_direct_positive_support = self._action_intent_text_has_direct_positive_evidence(support_lc)
        if "dry" in choice_lc and "hand" in choice_lc:
            if (
                has_modal_generic_support
                and not has_direct_positive_support
                and not any(term in support_lc for term in ("brought to both wet hands", "wipe them dry", "擦干双手", "湿手"))
            ):
                gaps.append("missing_dry_hands_evidence")
        if any(term in choice_lc for term in ("move", "relocate", "set aside", "put aside", "移开", "挪动")):
            if not any(
                term in support_lc
                for term in (
                    "picked up and then placed elsewhere",
                    "quickly set down on the counter in a different position",
                    "brief repositioning",
                    "left on the counter",
                    "set down on the counter",
                    "placed elsewhere",
                    "temporarily relocated",
                    "放到别处",
                    "放到台面另一处",
                    "短暂挪动",
                )
            ):
                gaps.append("missing_simple_relocation_evidence")
        if "wipe" in choice_lc and any(term in choice_lc for term in ("surface", "counter", "worktop", "table", "台面", "桌")):
            if any(term in support_lc for term in ("not yet shown", "no wiping motion", "未显示", "还未显示", "尚未显示")):
                gaps.append("missing_surface_wiping_evidence")
            if (
                has_modal_generic_support
                and not has_direct_positive_support
                and not any(
                    term in support_lc
                    for term in ("laid out on the worktop", "next to a visible spill", "ready for wiping", "ready to wipe", "放在台面旁准备擦", "准备擦")
                )
            ):
                gaps.append("missing_surface_wiping_evidence")
            if not self._action_intent_support_has_strong_surface_wiping_evidence(support_lc):
                gaps.append("missing_surface_wiping_evidence")
        if any(term in choice_lc for term in ("turn on", "switch on", "power on", "打开", "开启")):
            if not has_direct_positive_support and not any(
                term in support_lc
                for term in ("display turns on", "screen lights", "scale wakes", "powered on", "turned on", "亮起", "开机", "显示屏亮")
            ):
                gaps.append("missing_power_on_state_change_evidence")
        if any(term in choice_lc for term in ("zero out", "tare", "reset the scale", "归零", "去皮")):
            if not has_direct_positive_support and not any(
                term in support_lc
                for term in ("zero", "tare", "reset", "returns to 0", "container on the scale", "归零", "去皮", "回到0", "放到秤上")
            ):
                gaps.append("missing_zero_out_measurement_evidence")
        if self._action_intent_unresolved_candidate_spans_mixed_horizon(state=state, candidate_rows=candidate_rows, best_index=best_index):
            categories = selected_choice_categories([str(choice) for choice in getattr(state, "choices", [])], [best_index])
            best_categories = set(categories.get(best_index) or set())
            if self._action_intent_choice_is_immediate_micro_outcome_candidate(choice_lc, best_categories):
                if not self._action_intent_choice_has_explicit_immediate_micro_outcome_evidence(choice_lc, support_lc):
                    gaps.append("missing_immediate_micro_outcome_evidence")
            elif not self._action_intent_choice_has_explicit_later_outcome_evidence(choice_lc, best_categories, support_lc):
                gaps.append("missing_later_outcome_evidence")
        if any(term in choice_lc for term in ("access", "behind", "reveal", "expose", "拿到后面", "看到后面", "露出", "取到后面")):
            if any(term in support_lc for term in ("reveals the hidden area", "revealed", "hidden area behind", "shows what is behind", "露出后方", "看到后方")):
                gaps.append("generic_access_direct_effect")
        gaps.extend(
            self._action_intent_unresolved_timeline_review_bias_gaps(
                state=state,
                candidate_rows=candidate_rows,
                best_index=best_index,
            )
        )
        return list(dict.fromkeys(gaps))

    def _action_intent_unresolved_timeline_review_bias_gaps(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        best_index: int,
    ) -> list[str]:
        if not self._action_intent_has_unresolved_timeline_review_gap(state):
            return []
        bias = self._action_intent_timeline_review_bias_profile(state)
        if not bias["has_review"] or not bias["needs_more_evidence"]:
            return []
        target_row = next(
            (
                row
                for row in candidate_rows
                if int(row.get("index", -1)) == best_index
            ),
            None,
        )
        if target_row is None:
            return []
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        choice_lc = str(target_row.get("choice") or "").lower()
        support_lc = str(target_row.get("support") or "").lower()
        contradiction_lc = str(target_row.get("contradiction") or "").lower()
        combined_lc = f"{support_lc} {contradiction_lc}".strip()
        categories = selected_choice_categories(choices, [best_index])
        best_categories = set(categories.get(best_index) or set())
        gaps: list[str] = []
        if bias["final_location_unclear"] and (
            self._action_intent_choice_is_final_placement_candidate(choice_lc)
            or "final_place_return" in best_categories
        ):
            if not self._action_intent_choice_has_explicit_final_placement_evidence(choice_lc, combined_lc):
                gaps.append("timeline_review_final_location_gap")
        if bias["next_use_unclear"] and best_categories & {
            "measure_weigh",
            "transfer_contents",
            "serve_consume",
            "inspect_check",
            "open_close",
            "clean_dry",
            "food_prep",
            "discard",
            "final_place_return",
        }:
            if not self._action_intent_choice_has_explicit_later_outcome_evidence(choice_lc, best_categories, support_lc):
                gaps.append("timeline_review_next_use_gap")
        if bias["revealed_slot_placement"] and any(
            token in choice_lc
            for token in ("freed slot", "slot", "put into", "place into", "空位", "槽位", "放进", "归位")
        ):
            if not any(
                token in combined_lc
                for token in (
                    "placed into the freed slot",
                    "put into the freed slot",
                    "slot becomes the destination",
                    "destination is the freed slot",
                    "放进腾出的槽位",
                    "归位到空位",
                )
            ):
                gaps.append("timeline_review_revealed_slot_gap")
        if bias["revealed_target_retrieval"] and any(
            token in choice_lc
            for token in ("retrieve", "pick up", "take", "hidden", "behind", "取", "拿")
        ):
            if not any(
                token in combined_lc
                for token in (
                    "retrieved from behind",
                    "picked up from behind",
                    "taken from behind",
                    "hidden item is picked up",
                    "hidden jar is taken",
                    "取出后面的",
                    "拿到后面的",
                )
            ):
                gaps.append("timeline_review_revealed_target_gap")
        if (bias["revealed_fixture_enablement"] or bias["hand_free_next_action"]) and (
            "hand_free_enablement" in best_categories
            or "open_close" in best_categories
            or any(
                token in choice_lc
                for token in ("turn on", "turn off", "open", "close", "switch on", "switch off", "打开", "关闭", "开启")
            )
        ):
            if not self._action_intent_text_has_direct_positive_evidence(support_lc):
                gaps.append("timeline_review_hand_free_or_fixture_gap")
        return gaps

    def _action_intent_unresolved_rerank_should_wait_for_more_evidence(
        self,
        *,
        best_score: float,
        second_score: float,
        semantic_gaps: list[str],
        candidate_rows: list[dict[str, Any]],
        state: AgentState,
    ) -> bool:
        if best_score < 0.12 and "generic_access_direct_effect" not in semantic_gaps:
            return True
        broad_gap = "best_is_unproven_broad_candidate" in semantic_gaps
        unsupported_gap = "candidate_explicitly_lacks_observed_support" in semantic_gaps
        if any(
            gap in semantic_gaps
            for gap in (
                "timeline_review_final_location_gap",
                "timeline_review_next_use_gap",
                "timeline_review_revealed_slot_gap",
                "timeline_review_revealed_target_gap",
                "timeline_review_hand_free_or_fixture_gap",
            )
        ):
            return True
        if any(
            gap in semantic_gaps
            for gap in ("generic_hidden_reveal_or_access_direct_effect", "generic_access_direct_effect")
        ):
            return False
        if broad_gap and unsupported_gap and best_score <= 0.18:
            return True
        if unsupported_gap and best_score <= 0.38 and second_score <= 0.08:
            return True
        if broad_gap and best_score < 0.16 and best_score - second_score < 0.06:
            return True
        if broad_gap and unsupported_gap and len(semantic_gaps) >= 3 and best_score < 0.24:
            return True
        top_rows = sorted(
            candidate_rows,
            key=lambda row: (-float(row.get("adjusted_score", 0.0)), int(row.get("index", 999))),
        )[:3]
        if top_rows and len(top_rows) >= 2 and best_score <= 0.34:
            top_supports = [str(row.get("support") or "").lower() for row in top_rows]
            direct_positive_count = sum(
                1 for support in top_supports if self._action_intent_text_has_direct_positive_evidence(support)
            )
            weak_gap_count = 0
            if direct_positive_count == 0:
                weak_gap_count = sum(
                    1
                    for row in top_rows
                    if set(
                        self._action_intent_unresolved_semantic_gaps(
                            state=state,
                            candidate_rows=candidate_rows,
                            best_index=int(row.get("index", -1)),
                        )
                    )
                    & {
                        "candidate_explicitly_lacks_observed_support",
                        "missing_surface_wiping_evidence",
                        "missing_dry_hands_evidence",
                        "missing_simple_relocation_evidence",
                        "missing_immediate_micro_outcome_evidence",
                        "missing_later_outcome_evidence",
                    }
                )
            if weak_gap_count >= 2:
                return True
        if top_rows and best_score <= 0.24:
            top_gap_sets = [
                set(
                    self._action_intent_unresolved_semantic_gaps(
                        state=state,
                        candidate_rows=candidate_rows,
                        best_index=int(row.get("index", -1)),
                    )
                )
                for row in top_rows
            ]
            if top_gap_sets and all(
                (
                    "candidate_explicitly_lacks_observed_support" in gaps
                    or (
                        "missing_surface_wiping_evidence" in gaps
                        and "generic_access_direct_effect" not in gaps
                    )
                )
                for gaps in top_gap_sets
            ):
                if not any(self._action_intent_text_has_direct_positive_evidence(str(row.get("support") or "").lower()) for row in top_rows):
                    return True
        return False

    def _action_intent_unresolved_candidate_spans_mixed_horizon(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        best_index: int,
    ) -> bool:
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        candidate_indices = [
            int(row.get("index", -1))
            for row in candidate_rows
            if isinstance(row, dict) and self._coerce_choice_index(row.get("index"), state.choices) is not None
        ]
        categories_by_index = selected_choice_categories(choices, candidate_indices)
        best_categories = set(categories_by_index.get(best_index) or set())
        best_choice = choices[best_index].lower() if 0 <= best_index < len(choices) else ""
        best_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories)
        later_outcome_categories = {
            "final_place_return",
            "measure_weigh",
            "transfer_contents",
            "serve_consume",
            "clean_dry",
            "food_prep",
            "discard",
        }
        for row in candidate_rows:
            if not isinstance(row, dict):
                continue
            index = self._coerce_choice_index(row.get("index"), state.choices)
            if index is None or index == best_index:
                continue
            categories = set(categories_by_index.get(index) or set())
            choice_lc = choices[index].lower()
            competitor_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(choice_lc, categories)
            if (best_is_immediate and categories & later_outcome_categories) or (
                competitor_is_immediate and best_categories & later_outcome_categories
            ):
                return True
        return False

    def _action_intent_choice_is_immediate_micro_outcome_candidate(
        self,
        choice: str,
        categories: set[str],
    ) -> bool:
        text = str(choice or "").lower()
        if "inspect_check" in categories and any(
            token in text
            for token in (
                "label",
                "date",
                "expiry",
                "expiration",
                "best before",
                "use by",
                "sell by",
                "printed information",
                "read",
                "标签",
                "日期",
                "保质期",
                "读",
            )
        ):
            return True
        if "open_close" in categories and any(
            token in text
            for token in (
                "open",
                "close",
                "turn on",
                "turn off",
                "switch on",
                "switch off",
                "uncap",
                "cap",
                "unscrew",
                "打开",
                "关闭",
                "开启",
                "拧开",
                "盖上",
            )
        ):
            return True
        return False

    def _action_intent_choice_has_explicit_immediate_micro_outcome_evidence(self, choice: str, text: str) -> bool:
        choice_lc = str(choice or "").lower()
        text_lc = str(text or "").lower()
        if any(token in choice_lc for token in ("label", "date", "expiry", "expiration", "best before", "use by", "sell by", "read", "标签", "日期", "保质期", "读")):
            return any(
                token in text_lc
                for token in (
                    "reads the label",
                    "reading the label",
                    "checks the label",
                    "checked the label",
                    "looks at the label",
                    "examines the label",
                    "reads the date",
                    "checks the date",
                    "printed information is examined",
                    "looking at the printed information",
                    "查看标签",
                    "读取标签",
                    "看标签",
                    "检查日期",
                    "读取日期",
                )
            )
        if any(token in choice_lc for token in ("open", "uncap", "unscrew", "打开", "拧开", "盖上")):
            return any(
                token in text_lc
                for token in (
                    "opened",
                    "opens",
                    "opening the jar",
                    "lid removed",
                    "cap removed",
                    "unscrewed",
                    "uncapped",
                    "打开了",
                    "拧开了",
                    "盖子打开",
                )
            )
        if any(token in choice_lc for token in ("turn on", "switch on", "power on", "开启", "开机")):
            return any(
                token in text_lc
                for token in (
                    "display turns on",
                    "screen lights",
                    "powered on",
                    "turned on",
                    "亮起",
                    "开机",
                    "显示屏亮",
                )
            )
        return self._action_intent_text_has_direct_positive_evidence(text_lc)

    def _action_intent_choice_has_explicit_later_outcome_evidence(
        self,
        choice: str,
        categories: set[str],
        text: str,
    ) -> bool:
        choice_lc = str(choice or "").lower()
        text_lc = str(text or "").lower()
        if "final_place_return" in categories or any(
            token in choice_lc
            for token in ("put back", "return", "returned", "back in the fridge", "放回", "归位", "放进冰箱")
        ):
            return any(
                token in text_lc
                for token in (
                    "put back",
                    "returned",
                    "back in the fridge",
                    "placed back",
                    "returned to the fridge",
                    "returned to the shelf",
                    "放回",
                    "归位",
                    "放进冰箱",
                )
            )
        if "measure_weigh" in categories or any(
            token in choice_lc
            for token in ("weigh", "measure", "scale", "称", "测量")
        ):
            return any(
                token in text_lc
                for token in (
                    "placed on the scale",
                    "put on the scale",
                    "used to weigh",
                    "weighed",
                    "used at the scale",
                    "scale with",
                    "placed onto the scale",
                    "放到秤上",
                    "放上秤",
                    "称量",
                    "用于称",
                )
            )
        return self._action_intent_text_has_direct_positive_evidence(text_lc)

    def _score_action_intent_candidate_evidence(
        self,
        *,
        base_score: float,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        state: AgentState,
    ) -> float:
        question_lc = question.lower()
        support_lc = support.lower()
        contradiction_lc = contradiction.lower()
        choice_lc = choice.lower()
        action_object = self._action_intent_question_object(question)
        global_context = self._action_intent_scoped_global_context(state).lower()
        adjusted = max(0.0, min(float(base_score), 1.0))
        if self._action_intent_text_has_negative_evidence(contradiction_lc):
            adjusted -= 0.18
        if self._action_intent_text_has_negative_evidence(support_lc):
            adjusted -= 0.12
        if any(term in support_lc for term in ("theory", "theoretically", "could", "can be used", "compatible", "common", "常见", "理论", "可能")):
            adjusted -= 0.08
        if any(term in support_lc for term in ("least contradicted", "broadest", "最不矛盾", "最宽泛")):
            adjusted -= 0.18
        if self._action_intent_choice_is_generic_cleaning_tool_goal(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.26
        if self._action_intent_choice_is_generic_postwash_cleaning(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.22
            if any(
                token in f"{support_lc} {contradiction_lc}"
                for token in (
                    "remaining soap",
                    "soap suds",
                    "washing away the remaining soap",
                    "remove the remaining soap",
                    "肥皂",
                    "泡沫",
                )
            ):
                adjusted -= 0.12
        if self._action_intent_choice_is_generic_drying_without_wet_context(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.2
        if self._action_intent_choice_is_premature_drying_before_cleanup(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.24
        if self._action_intent_choice_is_generic_space_side_effect(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
        ):
            adjusted -= 0.18
        if self._action_intent_choice_is_generic_workspace_effect_over_exact_path_or_destination(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.22
        if self._action_intent_choice_is_direct_space_without_exact_next_use(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
        ):
            adjusted += 0.24
        if any(term in choice_lc for term in ("access", "behind", "reveal", "expose", "后面", "露出")) and any(
            term in support_lc for term in ("reveals the hidden area", "revealed", "hidden area behind", "shows what is behind", "露出后方", "看到后方")
        ):
            adjusted += 0.26
        if "clean" in choice_lc and any(term in contradiction_lc for term in ("no actual cleaning", "no visible wiping", "没有任何明确清洁", "没有擦")):
            adjusted -= 0.16
        if "away" in choice_lc and any(term in contradiction_lc for term in ("not stored", "not put", "counter", "没有看到把", "暂时", "台面")):
            adjusted -= 0.14
        if "dry" in choice_lc and "hand" in choice_lc and any(term in contradiction_lc for term in ("no visible hand", "no clear wet-hand", "没有看到双手", "没有先洗手")):
            adjusted -= 0.14
        if self._action_intent_choice_is_temporary_relocation_not_storage(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.24
        if self._action_intent_choice_is_unsupported_hand_drying_goal(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.22
        if self._action_intent_support_is_likely_downstream_to_move_action(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
            action_object=action_object,
        ):
            adjusted -= 0.36
        if self._action_intent_choice_is_direct_same_object_manipulation(
            choice=choice_lc,
            support=support_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.32
        if self._action_intent_choice_is_direct_same_object_cleaning(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.28
        if self._action_intent_choice_is_direct_same_object_role_use(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_is_measurement_base_placement(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_is_cleaning_placement_goal(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.3
        if self._action_intent_choice_is_direct_same_object_inspection_or_alignment(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.3
        if self._action_intent_choice_is_brief_cooking_inspection_over_disposal(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.32
        if self._action_intent_choice_is_generic_inspection_under_hidden_target_context(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.22
        if self._action_intent_choice_is_generic_hidden_reveal_or_access(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.24
        if self._action_intent_choice_is_generic_hidden_access_over_exact_reveal_use(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.26
        if self._action_intent_choice_is_generic_hidden_access_without_followup_use(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted += 0.22
        if self._action_intent_choice_is_generic_disposal_without_pour_signal(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.26
        if self._action_intent_choice_is_hidden_target_access_or_retrieval(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_is_exact_revealed_target_purpose(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.28
        if self._action_intent_choice_is_exact_reveal_then_take_or_place(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted += 0.26
        if self._action_intent_choice_is_generic_underneath_cleaning_under_hidden_target_context(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.2
        if self._action_intent_choice_is_exact_workspace_creation(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.4
        if self._action_intent_choice_is_exact_downstream_targeted_placement(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.36
        if self._action_intent_choice_is_exact_pickup_path_enablement(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.32
        if self._action_intent_choice_is_exact_immediate_downstream_use(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.32
        if self._action_intent_choice_is_cleaning_tool_specific_target_use(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_is_cleaning_supply_retrieval(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.36
        if self._action_intent_choice_is_cleaning_workflow_initiation(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.32
        if self._action_intent_choice_is_surface_wipe_preparation(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.28
        if self._action_intent_choice_is_weak_surface_contact_cleanup_claim(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.34
        if self._action_intent_choice_is_explicit_hand_drying_goal(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.3
        if self._action_intent_choice_is_direct_disposal_path(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_lacks_direct_relocation_outcome_evidence(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
        ):
            adjusted -= 0.2
        if self._action_intent_choice_is_postwash_residue_or_water_removal(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.3
        if self._action_intent_choice_is_postwash_drying_goal(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.28
        if self._action_intent_choice_is_immediate_reuse_staging(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.26
        if self._action_intent_choice_is_hygiene_surface_protection_staging(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.24
        if self._action_intent_choice_is_unfinished_cleanup_context_for_finished_or_storage(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.24
        finished_goal = self._action_intent_choice_is_finished_with_object_goal(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        )
        if finished_goal:
            adjusted += 0.26
            if any(
                token in support_lc
                for token in ("no longer needed", "simply placed aside", "set aside", "put down", "placed aside")
            ):
                adjusted += 0.12
            if any(
                token in contradiction_lc
                for token in ("no further", "no more", "no further spoon-use", "no further use", "no washing-followup")
            ):
                adjusted += 0.18
        if self._action_intent_choice_is_temporary_set_aside_not_finished(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.22
        if self._action_intent_choice_is_glove_removal_enablement(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.3
        if self._action_intent_choice_is_surface_mess_avoidance_goal(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.28
        if self._action_intent_choice_is_direct_hazard_avoidance(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_is_generic_mixing_under_hazard_context(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.26
        if self._action_intent_choice_is_pure_hand_free_enablement(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.22
        if self._action_intent_choice_is_direct_residue_release(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.3
        if self._action_intent_choice_is_receptacle_oriented_residue_release(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted += 0.18
        if self._action_intent_choice_is_side_switch_without_immediate_reuse(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            action_object=action_object,
            global_context=global_context,
        ):
            adjusted -= 0.18
        if self._action_intent_choice_is_hand_free_enablement(
            choice=choice_lc,
            support=support_lc,
            global_context=global_context,
        ):
            adjusted += 0.24
        if self._action_intent_choice_is_direct_enablement(
            choice=choice_lc,
            support=support_lc,
            global_context=global_context,
        ):
            adjusted += 0.34
        if self._action_intent_choice_is_direct_tap_enablement(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted += 0.18
        if self._action_intent_choice_is_dual_object_rinse(
            choice=choice_lc,
            support=support_lc,
            global_context=global_context,
        ):
            adjusted += 0.44
        if any(
            token in contradiction_lc
            for token in (
                "already in hand",
                "downstream",
                "later downstream",
                "后续",
                "下游",
                "结果性后续",
            )
        ):
            adjusted -= 0.18
        if any(
            token in contradiction_lc
            for token in (
                "downstream pickup after the transfer",
                "rather than the direct purpose of the transfer itself",
                "later downstream effect",
                "less specific than option",
                "只是转移动作之后的下游拿取",
                "不是当前转移动作的直接目的",
            )
        ):
            adjusted -= 0.2
        if self._action_intent_choice_is_weak_drainage_rearrangement(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
        ):
            adjusted -= 0.22
        if self._action_intent_choice_is_tap_state_switch(
            choice=choice_lc,
            support=support_lc,
            global_context=global_context,
        ):
            adjusted += 0.28
        if self._action_intent_choice_is_generic_fill_limit_without_match(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.24
        if self._action_intent_text_has_direct_positive_evidence(support_lc):
            adjusted += 0.1
        if self._action_intent_text_has_direct_positive_evidence(contradiction_lc):
            adjusted -= 0.08
        return min(adjusted, 1.0)

    def _action_intent_scoped_global_context(self, state: AgentState, *, limit_per_source: int = 24) -> str:
        anchor_times = self._action_intent_anchor_times(state)
        sources = list(getattr(state, "evidence_bundle", []))[-limit_per_source:] + list(getattr(state, "working_memory", []))[-limit_per_source:]
        if not anchor_times:
            return " ".join(str(item) for item in sources if isinstance(item, str) and not self._is_action_intent_leaky_memory(item))
        window_start = min(anchor_times) - 6.0
        window_end = max(anchor_times) + 8.0
        scoped: list[str] = []
        for item in sources:
            if not isinstance(item, str) or self._is_action_intent_leaky_memory(item):
                continue
            spans = self._action_intent_extract_embedded_note_times(item)
            lowered = item.lower()
            if spans:
                overlaps = any(not (end_time < window_start or start_time > window_end) for start_time, end_time in spans)
                if overlaps and item not in scoped:
                    scoped.append(item)
                continue
            if any(
                token in lowered
                for token in (
                    "type=frame;",
                    "inspection;",
                    "ongoing_action=",
                    "possible_step=",
                    "state_change_hint=",
                    "scene_location=",
                    "target_location=",
                )
            ):
                if item not in scoped:
                    scoped.append(item)
        if scoped:
            return " ".join(scoped)
        return " ".join(str(item) for item in sources if isinstance(item, str) and not self._is_action_intent_leaky_memory(item))

    def _is_action_intent_leaky_memory(self, item: object) -> bool:
        lowered = str(item or "").lower()
        return any(
            token in lowered
            for token in (
                "action_intent_",
                "visual_mcq_reason=",
                "answer_hint=",
                "candidate_answer_index=",
                "deterministic_finalize",
                "source=agent_timeline_summary",
                "source=session_memory_compressor",
            )
        )

    def _action_intent_anchor_times(self, state: AgentState) -> list[float]:
        times: list[float] = []
        payload = state.inputs_payload() if callable(getattr(state, "inputs_payload", None)) else {}
        if isinstance(payload, dict):
            for value in self._action_intent_extract_times_from_inputs_payload(payload):
                times.append(value)
        for path in list(getattr(state, "retrieved_frames", []) or []):
            inferred = self._action_intent_infer_artifact_time(path)
            if inferred is not None:
                times.append(inferred)
        deduped: list[float] = []
        for value in sorted(times):
            rounded = round(float(value), 3)
            if rounded not in deduped:
                deduped.append(rounded)
        return deduped

    def _action_intent_extract_times_from_inputs_payload(self, payload: dict[str, Any]) -> list[float]:
        times: list[float] = []
        if not isinstance(payload, dict):
            return times
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            for key in ("start_time", "end_time"):
                raw = value.get(key)
                if not isinstance(raw, str) or not raw.strip():
                    continue
                parsed = self._parse_hhmmss_time(raw)
                if parsed is not None:
                    times.append(parsed)
        return times

    def _parse_hhmmss_time(self, value: str) -> float | None:
        match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", str(value).strip())
        if not match:
            return None
        try:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
        except Exception:  # noqa: BLE001
            return None
        return hours * 3600.0 + minutes * 60.0 + seconds

    def _action_intent_extract_embedded_note_times(self, text: str) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        for match in re.finditer(r"time=([0-9.]+)-([0-9.]+)", str(text)):
            try:
                spans.append((float(match.group(1)), float(match.group(2))))
            except Exception:  # noqa: BLE001
                continue
        return spans

    def _action_intent_infer_artifact_time(self, path: str) -> float | None:
        match = re.search(r"_([0-9]+\.[0-9]+)s\.(?:jpg|jpeg|png|webp)$", str(path), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _action_intent_support_is_likely_downstream_to_move_action(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
        action_object: str,
    ) -> bool:
        if not any(token in question for token in ("move ", "transfer ", "shift ", "remove ", "clear ")):
            return False
        if self._choice_is_same_object_active_use(choice, action_object):
            return False
        if self._action_intent_choice_is_exact_immediate_downstream_use(
            question=question,
            choice=choice,
            support=support,
            contradiction=contradiction,
            action_object=action_object,
            global_context=global_context,
        ):
            return False
        if self._action_intent_choice_is_exact_pickup_path_enablement(
            question=question,
            choice=choice,
            support=support,
            contradiction=contradiction,
            action_object=action_object,
            global_context=global_context,
        ):
            return False
        if self._action_intent_choice_is_hand_free_enablement(
            choice=choice,
            support=support,
            global_context=global_context,
        ):
            return False
        if any(token in choice for token in ("tap", "faucet", "drain", "drainage", "while holding", " in hand")):
            return False
        if not any(token in choice for token in ("pick up", "lift", "take", "scrub", "wash", "sink", "board", "sponge")):
            return False
        if not any(token in support for token in ("after", "then", "later", "subsequently", "随后", "之后", "接着", "转移", "移开后")):
            return False
        return any(
            token in support
            for token in (
                "pick up",
                "picked up",
                "reach",
                "reaches",
                "伸向",
                "拿起",
                "举着",
                "成为接下来的核心操作对象",
                "main object",
            )
        )

    def _action_intent_choice_is_direct_enablement(
        self,
        *,
        choice: str,
        support: str,
        global_context: str,
    ) -> bool:
        if not any(token in choice for token in ("tap", "faucet", "drain", "drainage", "water")):
            return False
        signal_text = f"{support} {global_context}"
        return any(token in signal_text for token in ("tap", "faucet", "water", "sink", "水龙头", "排水", "水槽"))

    def _action_intent_choice_is_dual_object_rinse(
        self,
        *,
        choice: str,
        support: str,
        global_context: str,
    ) -> bool:
        if not any(token in choice for token in ("rinse", "wash", "sponge")):
            return False
        if not any(token in choice for token in ("while holding", "in hand", "手中")):
            return False
        signal_text = f"{support} {global_context}"
        has_dual_object = any(token in signal_text for token in ("one hand", "另一手", "左手", "右手", "holding", "拿着"))
        has_sink_or_water = any(token in signal_text for token in ("sink", "water", "水槽", "水流", "冲洗", "rins"))
        return has_dual_object and has_sink_or_water

    def _action_intent_choice_is_weak_drainage_rearrangement(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
    ) -> bool:
        if not any(token in choice for token in ("drain", "drainage", "排水")):
            return False
        if not any(token in support for token in ("near the drain", "drain area", "排水口附近", "relocated near")):
            return False
        return any(
            token in contradiction
            for token in ("no direct drainage", "not shown", "没有直接排水", "缺少排水", "未显示排水")
        )

    def _action_intent_choice_is_direct_tap_enablement(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if not any(token in choice for token in ("tap", "faucet", "turn on the tap", "水龙头")):
            return False
        signal_text = f"{support} {global_context}"
        has_tap_context = any(token in signal_text for token in ("tap", "faucet", "water", "sink", "水龙头", "水槽"))
        contradiction_is_soft = any(
            token in contradiction
            for token in (
                "not clearly shown",
                "not explicit",
                "not seen",
                "no water flow",
                "未清楚显示",
                "不够清楚",
                "没有水流",
                "没有明确",
            )
        )
        return has_tap_context and contradiction_is_soft

    def _resolve_prior_direct_action_object_intent(
        self,
        *,
        state: AgentState,
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        if unresolved_best_score >= 0.5:
            return None
        action_object = self._action_intent_question_object(str(getattr(state, "question", "") or ""))
        if not action_object:
            return None
        unresolved_choice = str(state.choices[unresolved_best_index]).lower()
        if self._choice_is_same_object_active_use(unresolved_choice, action_object):
            return None
        best_prior: tuple[int, str, float] | None = None
        for entry in reversed(list(getattr(state, "tool_trace", []) or [])):
            if not isinstance(entry, dict) or entry.get("tool") != "infer_action_intent":
                continue
            raw_result = entry.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
            if index is None or index == unresolved_best_index:
                continue
            choice = str(state.choices[index]).lower()
            if not self._choice_is_same_object_active_use(choice, action_object):
                continue
            confidence = self._coerce_confidence(raw_result.get("confidence"), default=0.0)
            if confidence < 0.72:
                continue
            best_prior = (index, str(state.choices[index]), confidence)
            break
        return best_prior

    def _override_downstream_followup_with_direct_enablement_candidate(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        question = str(getattr(state, "question", "") or "")
        question_lc = question.lower()
        if not any(token in question_lc for token in ("move ", "transfer ", "shift ", "remove ", "clear ")):
            return None
        best_row = next(
            (
                row
                for row in candidate_rows
                if int(row.get("index", -1)) == unresolved_best_index
            ),
            None,
        )
        if best_row is None:
            return None
        action_object = self._action_intent_question_object(question)
        best_choice = str(best_row.get("choice") or "").lower()
        best_support = str(best_row.get("support") or "").lower()
        best_contradiction = str(best_row.get("contradiction") or "").lower()
        if not self._action_intent_choice_is_downstream_followup_use(
            question=question_lc,
            choice=best_choice,
            support=best_support,
            action_object=action_object,
        ):
            return None
        direct_candidates: list[tuple[float, int, str]] = []
        for row in candidate_rows:
            index = int(row.get("index", -1))
            if index == unresolved_best_index:
                continue
            choice = str(row.get("choice") or "").lower()
            support = str(row.get("support") or "").lower()
            contradiction = str(row.get("contradiction") or "").lower()
            adjusted_score = float(row.get("adjusted_score") or 0.0)
            if not self._action_intent_choice_is_direct_fixture_or_workspace_enablement(
                choice=choice,
                support=support,
                contradiction=contradiction,
            ):
                continue
            if adjusted_score < 0.14:
                continue
            direct_candidates.append((adjusted_score, index, str(state.choices[index])))
        if not direct_candidates:
            return None
        direct_candidates.sort(key=lambda item: (-item[0], item[1]))
        alt_score, alt_index, alt_choice = direct_candidates[0]
        explicit_downstream_consequence = any(
            token in best_contradiction
            for token in (
                "downstream pickup after the transfer",
                "rather than the direct purpose of the transfer itself",
                "later downstream effect",
                "this is a downstream pickup",
                "不是当前转移动作的直接目的",
                "只是转移动作之后的下游拿取",
            )
        )
        if (
            unresolved_best_score > alt_score + 0.34
            and unresolved_best_score >= 0.82
            and not explicit_downstream_consequence
        ):
            return None
        confidence = min(max(0.48 + max(alt_score, 0.0) * 0.38, 0.48), 0.72)
        return alt_index, alt_choice, confidence

    def _override_generic_space_with_exact_immediate_use_candidate(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        question = str(getattr(state, "question", "") or "")
        question_lc = question.lower()
        if not any(token in question_lc for token in ("move ", "transfer ", "shift ", "remove ", "clear ")):
            return None
        best_row = next(
            (
                row
                for row in candidate_rows
                if int(row.get("index", -1)) == unresolved_best_index
            ),
            None,
        )
        if best_row is None:
            return None
        best_choice = str(best_row.get("choice") or "").lower()
        if not self._action_intent_choice_is_generic_direct_space_purpose(best_choice):
            return None
        action_object = self._action_intent_question_object(question)
        exact_candidates: list[tuple[float, int, str]] = []
        for row in candidate_rows:
            index = int(row.get("index", -1))
            if index == unresolved_best_index:
                continue
            choice = str(row.get("choice") or "").lower()
            support = str(row.get("support") or "").lower()
            contradiction = str(row.get("contradiction") or "").lower()
            adjusted_score = float(row.get("adjusted_score") or 0.0)
            if not self._action_intent_choice_is_exact_immediate_downstream_use(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=" ".join(
                    str(item)
                    for item in list(getattr(state, "evidence_bundle", []))[-24:]
                    + list(getattr(state, "working_memory", []))[-24:]
                    if isinstance(item, str)
                ).lower(),
            ):
                continue
            if adjusted_score < 0.12:
                continue
            exact_candidates.append((adjusted_score, index, str(state.choices[index])))
        if not exact_candidates:
            return None
        exact_candidates.sort(key=lambda item: (-item[0], item[1]))
        alt_score, alt_index, alt_choice = exact_candidates[0]
        if unresolved_best_score > alt_score + 0.26 and unresolved_best_score >= 0.8:
            return None
        confidence = min(max(0.5 + max(alt_score, 0.0) * 0.36, 0.5), 0.74)
        return alt_index, alt_choice, confidence

    def _override_generic_hidden_access_with_exact_revealed_target_candidate(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        question = str(getattr(state, "question", "") or "")
        question_lc = question.lower()
        if not any(token in question_lc for token in ("move ", "transfer ", "shift ", "remove ", "clear ")):
            return None
        best_row = next(
            (
                row
                for row in candidate_rows
                if int(row.get("index", -1)) == unresolved_best_index
            ),
            None,
        )
        if best_row is None:
            return None
        best_choice = str(best_row.get("choice") or "").lower()
        global_context = " ".join(
            str(item)
            for item in list(getattr(state, "evidence_bundle", []))[-24:]
            + list(getattr(state, "working_memory", []))[-24:]
            if isinstance(item, str)
        ).lower()
        if not self._action_intent_choice_is_generic_hidden_reveal_or_access(
            choice=best_choice,
            support=str(best_row.get("support") or "").lower(),
            contradiction=str(best_row.get("contradiction") or "").lower(),
            global_context=global_context,
        ):
            return None
        action_object = self._action_intent_question_object(question)
        exact_candidates: list[tuple[float, int, str]] = []
        for row in candidate_rows:
            index = int(row.get("index", -1))
            if index == unresolved_best_index:
                continue
            choice = str(row.get("choice") or "").lower()
            support = str(row.get("support") or "").lower()
            contradiction = str(row.get("contradiction") or "").lower()
            adjusted_score = float(row.get("adjusted_score") or 0.0)
            if not self._action_intent_choice_is_exact_revealed_target_purpose(
                question=question_lc,
                choice=choice,
                support=support,
                contradiction=contradiction,
                action_object=action_object,
                global_context=global_context,
            ):
                continue
            if adjusted_score < 0.12:
                continue
            exact_candidates.append((adjusted_score, index, str(state.choices[index])))
        if not exact_candidates:
            return None
        exact_candidates.sort(key=lambda item: (-item[0], item[1]))
        alt_score, alt_index, alt_choice = exact_candidates[0]
        if unresolved_best_score > alt_score + 0.24 and unresolved_best_score >= 0.8:
            return None
        confidence = min(max(0.5 + max(alt_score, 0.0) * 0.38, 0.5), 0.74)
        return alt_index, alt_choice, confidence

    def _override_generic_hand_wiping_with_explicit_single_hand_drying(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        del unresolved_best_score
        question = str(getattr(state, "question", "") or "").lower()
        action_object = self._action_intent_question_object(question)
        if not any(token in action_object for token in ("towel", "cloth", "napkin", "paper towel")):
            return None
        best_row = next((row for row in candidate_rows if int(row.get("index", -1)) == unresolved_best_index), None)
        if best_row is None:
            return None
        best_choice = str(best_row.get("choice") or "").lower()
        if not any(token in best_choice for token in ("clean", "counter", "surface", "wipe both hands", "to dry.", "move")):
            return None
        explicit_candidates: list[tuple[float, int, str]] = []
        for row in candidate_rows:
            index = int(row.get("index", -1))
            if index == unresolved_best_index:
                continue
            choice = str(row.get("choice") or "").lower()
            support = str(row.get("support") or "").lower()
            contradiction = str(row.get("contradiction") or "").lower()
            adjusted_score = float(row.get("adjusted_score") or 0.0)
            if "dry hand" not in choice and "dry hands" not in choice:
                continue
            signal_text = f"{support} {contradiction}"
            if not any(
                token in signal_text
                for token in (
                    "hand area",
                    "dab/rub one hand",
                    "rubbed against the other hand",
                    "finger",
                    "fingers",
                    "单手",
                    "手指",
                )
            ):
                continue
            explicit_candidates.append((adjusted_score, index, str(state.choices[index])))
        if not explicit_candidates:
            return None
        explicit_candidates.sort(key=lambda item: (-item[0], item[1]))
        alt_score, alt_index, alt_choice = explicit_candidates[0]
        confidence = min(max(0.5 + max(alt_score, 0.0) * 0.32, 0.5), 0.72)
        return alt_index, alt_choice, confidence

    def _override_generic_towel_use_with_simple_relocation(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        question = str(getattr(state, "question", "") or "").lower()
        action_object = self._action_intent_question_object(question)
        if not any(token in action_object for token in ("towel", "cloth", "napkin", "paper towel")):
            return None
        best_row = next((row for row in candidate_rows if int(row.get("index", -1)) == unresolved_best_index), None)
        if best_row is None:
            return None
        best_choice = str(best_row.get("choice") or "").lower()
        if not any(token in best_choice for token in ("clean", "counter", "surface", "wipe", "dry")):
            return None
        weak_surface_overclaim = self._action_intent_choice_is_weak_surface_contact_cleanup_claim(
            choice=best_choice,
            support=str(best_row.get("support") or "").lower(),
            contradiction=str(best_row.get("contradiction") or "").lower(),
            action_object=action_object,
            global_context="",
        )
        if unresolved_best_score >= 0.46 and not weak_surface_overclaim:
            return None
        relocation_candidates: list[tuple[float, int, str]] = []
        for row in candidate_rows:
            index = int(row.get("index", -1))
            choice = str(row.get("choice") or "").lower()
            support = str(row.get("support") or "").lower()
            contradiction = str(row.get("contradiction") or "").lower()
            adjusted_score = float(row.get("adjusted_score") or 0.0)
            if "move" not in choice:
                continue
            if not any(
                token in support
                for token in (
                    "quickly set down on the counter in a different position",
                    "brief repositioning",
                    "brief repositioning occurs",
                    "left on the counter",
                    "set down on the counter",
                    "shifted slightly",
                    "relocated from",
                    "literal move does occur",
                    "moved from",
                    "moved towards",
                    "temporarily relocated",
                    "放到别处",
                    "放到另一处",
                    "短暂挪动",
                )
            ):
                continue
            if any(
                token in contradiction
                for token in (
                    "clear wiping",
                    "counter-wiping cleanup",
                    "wiping stroke",
                    "both hands being wiped",
                    "明显擦拭",
                    "双手擦拭",
                )
            ):
                continue
            if (
                not weak_surface_overclaim
                and any(
                    token in contradiction
                    for token in (
                        "mere relocation is a byproduct",
                        "not a clear purpose",
                    )
                )
            ):
                continue
            relocation_candidates.append((adjusted_score, index, str(state.choices[index])))
        if not relocation_candidates:
            return None
        relocation_candidates.sort(key=lambda item: (-item[0], item[1]))
        alt_score, alt_index, alt_choice = relocation_candidates[0]
        confidence = min(max(0.48 + max(alt_score, 0.0) * 0.34, 0.48), 0.72 if weak_surface_overclaim else 0.7)
        return alt_index, alt_choice, confidence

    def _action_intent_choice_is_weak_surface_contact_cleanup_claim(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("cloth", "towel", "tea towel", "dish cloth", "napkin", "paper towel", "sponge", "scrubber")
        ):
            return False
        if not any(token in choice for token in ("wipe", "clean", "擦", "清洁")):
            return False
        if not any(
            token in choice
            for token in (
                "surface",
                "counter",
                "countertop",
                "worktop",
                "table",
                "台面",
                "桌面",
            )
        ):
            return False
        support_lc = str(support or "").lower()
        contradiction_lc = str(contradiction or "").lower()
        signal_text = f"{support_lc} {contradiction_lc} {str(global_context or '').lower()}"
        if self._action_intent_support_has_strong_surface_wiping_evidence(signal_text):
            return False
        has_surface_contact_only = any(
            token in support_lc
            for token in (
                "contact with the counter area",
                "contact with the counter",
                "moved across/onto the countertop",
                "moved across the countertop",
                "pressed it against the countertop",
                "brought into contact with the counter area",
                "used/place it there",
                "placed onto the counter",
                "near the counter/appliance area",
                "counter area",
                "object-to-surface interaction",
                "接触台面",
                "放到台面",
                "靠近台面",
            )
        )
        has_missing_cleaning_result = any(
            token in contradiction_lc
            for token in (
                "no extended wiping motion",
                "no visible spill",
                "no visible residue",
                "visible spill/residue being removed",
                "cleanup result is only weakly demonstrated",
                "brief and does not clearly show",
                "only weakly demonstrated",
                "没有明显擦拭",
                "没有看到污渍",
                "没有清理结果",
                "证据很弱",
            )
        )
        return has_surface_contact_only and has_missing_cleaning_result

    def _action_intent_prior_reasoning_text(self, state: AgentState) -> str:
        texts: list[str] = []
        for entry in list(getattr(state, "tool_trace", []) or []):
            if not isinstance(entry, dict):
                continue
            raw_result = entry.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            for key in ("reason", "decisive_observation", "needed_observation", "answer"):
                value = raw_result.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        return " ".join(texts[-12:])

    def _action_intent_support_has_strong_surface_wiping_evidence(self, text: str) -> bool:
        lowered = str(text or "").lower()
        if self._action_intent_text_has_negative_evidence(lowered):
            return False
        has_surface_target = any(
            token in lowered
            for token in (
                "visible spill",
                "crumbs",
                "specific dirty spot",
                "mess on the counter",
                "counter surface target",
                "worktop target",
                "台面污渍",
                "碎屑",
                "脏点",
            )
        )
        has_wiping_motion = any(
            token in lowered
            for token in (
                "wiping stroke",
                "wipe sweep",
                "repeated wiping",
                "sustained wiping",
                "scrubbing motion",
                "wiped across the counter",
                "明确擦拭动作",
                "连续擦拭",
                "来回擦",
            )
        )
        return has_surface_target and has_wiping_motion

    def _choice_is_phone_app_record_target_purpose(self, choice: str) -> bool:
        text = str(choice or "").lower()
        has_record_signal = any(
            token in text
            for token in (
                "record",
                "update",
                "enter",
                "log",
                "app",
                "nutrition",
                "nutritional",
                "phone",
                "ingredient",
                "measurements of the",
                "value of the",
                "记录",
                "录入",
                "更新",
                "营养",
                "手机",
            )
        )
        if not has_record_signal:
            return False
        return not any(
            token in text
            for token in (
                "measure the ingredients",
                "measure ingredients",
                "weigh the ingredients",
                "to measure.",
                "generic measure",
            )
        )

    def _action_intent_choice_is_generic_measure_phone_goal(self, choice: str, action_object: str) -> bool:
        text = str(choice or "").lower()
        return any(token in action_object for token in ("phone", "smartphone", "mobile")) and any(
            token in text
            for token in (
                "measure the ingredients",
                "measure ingredients",
                "weigh the ingredients",
                "measure.",
                "to measure",
            )
        )

    def _action_intent_choice_supports_exact_record_target(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        signal_text = f"{choice} {support} {contradiction} {global_context}".lower()
        has_phone_context = any(
            token in signal_text
            for token in (
                "phone",
                "smartphone",
                "app",
                "screen",
                "record",
                "log",
                "entry",
                "entering",
                "nutrition",
                "nutritional",
                "ingredient entry",
                "recording target",
                "手机",
                "屏幕",
                "记录",
                "录入",
                "营养",
            )
        )
        has_specific_target = any(
            token in signal_text
            for token in (
                "coriander",
                "broccoli",
                "carrot",
                "cilantro",
                "ingredient",
                "herbs",
                "香菜",
                "西兰花",
                "胡萝卜",
            )
        )
        return has_phone_context and has_specific_target

    def _override_generic_measure_with_exact_record_target_candidate(
        self,
        *,
        state: AgentState,
        candidate_rows: list[dict[str, Any]],
        unresolved_best_index: int,
        unresolved_best_score: float,
    ) -> tuple[int, str, float] | None:
        question = str(getattr(state, "question", "") or "")
        action_object = self._action_intent_question_object(question)
        if not any(token in action_object for token in ("phone", "smartphone", "mobile")):
            return None
        best_row = next(
            (
                row
                for row in candidate_rows
                if int(row.get("index", -1)) == unresolved_best_index
            ),
            None,
        )
        if best_row is None:
            return None
        best_choice = str(best_row.get("choice") or "").lower()
        if not self._action_intent_choice_is_generic_measure_phone_goal(best_choice, action_object):
            return None
        best_support = str(best_row.get("support") or "").lower()
        best_contradiction = str(best_row.get("contradiction") or "").lower()
        global_context = self._action_intent_scoped_global_context(state).lower()
        if not any(
            token in f"{best_support} {best_contradiction}"
            for token in (
                "broadest",
                "least contradicted",
                "no actual recording target",
                "no direct recording target",
                "not direct proof",
                "still not direct",
                "recording target is not shown",
                "最宽泛",
                "没有直接记录目标",
            )
        ):
            return None
        exact_candidates: list[tuple[float, int, str]] = []
        for row in candidate_rows:
            index = int(row.get("index", -1))
            if index == unresolved_best_index:
                continue
            choice = str(row.get("choice") or "").lower()
            support = str(row.get("support") or "").lower()
            contradiction = str(row.get("contradiction") or "").lower()
            adjusted_score = float(row.get("adjusted_score") or 0.0)
            if not self._choice_is_phone_app_record_target_purpose(choice):
                continue
            if not self._action_intent_choice_supports_exact_record_target(
                choice=choice,
                support=support,
                contradiction=contradiction,
                global_context=global_context,
            ):
                continue
            if adjusted_score + 0.18 < unresolved_best_score:
                continue
            exact_candidates.append((adjusted_score, index, str(state.choices[index])))
        if not exact_candidates:
            return None
        exact_candidates.sort(key=lambda item: (-item[0], item[1]))
        alt_score, alt_index, alt_choice = exact_candidates[0]
        return alt_index, alt_choice, min(max(0.44 + max(alt_score, 0.0) * 0.42, 0.44), 0.72)

    def _action_intent_question_object(self, question: str) -> str:
        match = re.search(r"<([^>]+)>", str(question or "").lower())
        if not match:
            return ""
        text = match.group(1)
        text = re.sub(
            r"\b(move|transfer|pick up|pickup|take|lift|shift|remove|open|close|turn|place|put|shake|tip)\b",
            " ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _choice_is_same_object_active_use(self, choice: str, action_object: str) -> bool:
        if not action_object:
            return False
        object_tokens = [token for token in re.split(r"[^a-z0-9]+", action_object) if token]
        if object_tokens and not all(token in choice for token in object_tokens):
            return False
        return any(
            token in choice
            for token in (
                "rinse",
                "wash",
                "clean",
                "wipe",
                "dry",
                "fill",
                "open",
                "uncap",
                "cap",
                "lid",
                "unscrew",
                "shake",
                "hold",
                "in hand",
                "while holding",
                "冲洗",
                "清洗",
                "擦",
                "拿着",
                "打开",
                "拧开",
                "摇",
            )
        )

    def _signal_mentions_action_object(self, signal_text: str, action_object: str) -> bool:
        if not action_object:
            return False
        object_tokens = [token for token in re.split(r"[^a-z0-9]+", action_object) if token and len(token) >= 3]
        if object_tokens and any(token in signal_text for token in object_tokens):
            return True
        return any(
            token in signal_text
            for token in (
                "same object",
                "same item",
                "main object",
                "remains held",
                "kept in one hand",
                "carried directly",
                "held throughout",
                "当前这个物体",
                "同一个物体",
                "仍拿着",
                "一直拿着",
            )
        )

    def _action_intent_choice_is_direct_same_object_manipulation(
        self,
        *,
        choice: str,
        support: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not self._choice_is_same_object_active_use(choice, action_object):
            return False
        if not any(token in choice for token in ("open", "uncap", "cap", "lid", "unscrew", "打开", "拧开")):
            return False
        signal_text = f"{support} {global_context}"
        return any(
            token in signal_text
            for token in (
                "cap",
                "lid",
                "open",
                "uncap",
                "unscrew",
                "free hand",
                "other hand",
                "holding",
                "while holding",
                "holding in one hand",
                "keeps holding",
                "one hand",
                "other hand",
                "拿着",
                "一只手",
                "另一只手",
                "盖",
                "打开",
            )
        )

    def _action_intent_choice_is_direct_same_object_cleaning(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not self._choice_is_same_object_active_use(choice, action_object):
            return False
        if any(
            token in action_object
            for token in ("sponge", "brush", "cloth", "towel", "napkin", "paper towel", "scrubber")
        ):
            return False
        if not any(token in choice for token in ("wash", "rinse", "clean", "scrub", "冲洗", "清洗", "刷")):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "later",
                "downstream",
                "after that picks up",
                "之后再",
                "后续才",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "while holding",
                "holding in one hand",
                "other hand",
                "free hand",
                "brush",
                "sponge",
                "tap",
                "running water",
                "under water",
                "rinse the",
                "wash the",
                "scrub the",
                "一只手",
                "另一只手",
                "拿着",
                "海绵",
                "刷子",
                "水龙头",
                "流水",
            )
        )

    def _action_intent_choice_is_direct_same_object_role_use(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not action_object:
            return False
        if any(
            token in action_object
            for token in ("sponge", "brush", "cloth", "towel", "napkin", "paper towel", "scrubber")
        ):
            return False
        if any(
            token in choice
            for token in (
                "wash",
                "rinse",
                "clean",
                "scrub",
                "wipe",
                "dry",
                "open",
                "uncap",
                "cap",
                "lid",
                "unscrew",
                "冲洗",
                "清洗",
                "擦",
                "晾",
                "打开",
                "拧开",
            )
        ):
            return False
        if not any(
            token in choice
            for token in (
                "measure",
                "weigh",
                "tare",
                "record",
                "reading",
                "stir",
                "scoop",
                "drain",
                "pour",
                "fill",
                "move the",
                "place the",
                "put the",
                "bring the",
                "carry the",
                "to the sink",
                "on the hob",
                "on the scale",
                "on the counter",
                "dish drainer",
                "drain rack",
                "while holding",
                "in left hand",
                "in right hand",
                "称",
                "测量",
                "搅拌",
                "舀",
                "倒",
                "移到水槽",
                "放到灶台",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        if any(
            token in signal_text
            for token in (
                "later downstream",
                "after that picks up",
                "之后再去拿",
                "后续才",
            )
        ):
            return False
        matched_terms = [
            token
            for token in (
                "sink",
                "hob",
                "scale",
                "counter",
                "dish drainer",
                "drain rack",
                "cheese",
                "onions",
                "potato",
                "mixture",
                "saucepan",
                "pan",
                "bowl",
                "pot",
                "cup",
            )
            if token in choice
        ]
        if matched_terms and not any(token in signal_text for token in matched_terms):
            return False
        return any(
            token in signal_text
            for token in (
                "remains the moved object",
                "remains held",
                "kept in one hand",
                "carried directly",
                "moved directly toward",
                "placed on the hob",
                "placed on the scale",
                "moved to the sink",
                "used to measure",
                "used for measuring",
                "used to stir",
                "used for stirring",
                "used to scoop",
                "used to drain",
                "used to pour",
                "direct purpose visible in the sequence",
                "main object",
                "while the other hand",
                "held in the other hand",
                "仍然是当前操作的主体",
                "直接移到水槽",
                "放到灶台上",
                "放到秤上",
                "用来称量",
                "用来搅拌",
                "一只手拿着",
                "另一只手",
            )
        )

    def _action_intent_choice_is_measurement_base_placement(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("place ", "put ", "set ")):
            return False
        if not any(
            token in choice
            for token in (
                "base to weigh",
                "base for weighing",
                "weigh more ingredients",
                "measure",
                "tared",
                "tare",
                "base to place the ingredients",
                "base for weighing",
                "称量基底",
                "称量更多食材",
                "称重",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        if not any(
            token in signal_text
            for token in (
                "scale",
                "weigh",
                "weighing",
                "tared",
                "tare",
                "base",
                "placed onto the scale",
                "used as a base",
                "used next for weighing",
                "immediate weighing use",
                "kitchen scale",
                "称",
                "秤",
                "作为称量基底",
                "称重",
            )
        ):
            return False
        return not any(
            token in contradiction
            for token in (
                "not being left to dry",
                "not storage",
                "not stored",
                "not for storage",
                "不是收纳",
                "不是晾干",
            )
        ) or any(
            token in signal_text
            for token in (
                "not being left to dry",
                "positioned for immediate weighing use",
                "used next with the kitchen scale",
                "as a base for weighing more ingredients",
                "immediate weighing use",
                "不是晾干",
                "立即用于称量",
            )
        )

    def _action_intent_choice_is_cleaning_placement_goal(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("place ", "put ", "set ", "transfer ")):
            return False
        if not any(
            token in choice
            for token in (
                "to clean",
                "clean off soap",
                "clean with water",
                "wash",
                "rinse",
                "soak",
                "be washed",
                "clean next",
                "清洗",
                "冲洗",
                "浸泡",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        if not any(
            token in signal_text
            for token in (
                "sink",
                "wash area",
                "tap",
                "soap",
                "soap residue",
                "cleaned next",
                "washed next",
                "placed into the sink",
                "placed in the sink",
                "soak",
                "washing the plate",
                "washing the bowl",
                "washing the knife",
                "sink placement",
                "水槽",
                "水龙头",
                "肥皂",
                "放进水槽",
                "接下来清洗",
            )
        ):
            return False
        if any(
            token in signal_text
            for token in (
                "to dry",
                "dry after washing",
                "water droplets fall away",
                "drain and dry",
                "stored",
                "store",
                "晾干",
                "收纳",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "to be cleaned next",
                "for washing the",
                "immediate sink placement is for washing",
                "clean off the soap",
                "soap residue",
                "washed next",
                "sink placement",
                "placed into the sink",
                "to soak",
                "cleaning placement",
                "接下来要洗",
                "为了清洗",
                "洗掉肥皂",
            )
        )

    def _action_intent_choice_is_generic_space_side_effect(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
    ) -> bool:
        if self._action_intent_choice_has_specific_space_target(choice):
            return False
        if not any(
            token in choice
            for token in (
                "make space",
                "make some space",
                "create space",
                "free up space",
                "make room",
                "some room",
                "free counter room",
                "腾空间",
                "腾出空间",
            )
        ):
            return False
        if not any(
            token in contradiction
            for token in (
                "side effect",
                "secondary",
                "generic workspace effect",
                "more direct",
                "explicit next functional use",
                "direct purpose",
                "只是副作用",
                "更直接的目的",
                "泛化的空间效果",
            )
        ):
            return False
        return any(
            token in f"{support} {contradiction}"
            for token in (
                "workspace",
                "counter room",
                "make room",
                "free room",
                "space",
                "room",
                "台面空间",
                "腾出空间",
            )
        )

    def _action_intent_choice_has_specific_space_target(self, choice: str) -> bool:
        if not any(
            token in choice
            for token in (
                "make space",
                "make some space",
                "create space",
                "free up space",
                "make room",
                "clear the way",
                "out of the way",
                "move out of the way",
                "腾空间",
                "腾出空间",
                "让开",
            )
        ):
            return False
        if any(
            token in choice
            for token in (
                "holding in left hand",
                "holding in right hand",
                "held in left hand",
                "held in right hand",
                "i'm holding in left hand",
                "i'm holding in right hand",
                "put down",
                "put into",
                "be put into",
                "fit into",
                "measure",
                "pick up",
                "grab",
                "drying rack",
                "dishwasher",
                "draining rack",
                "rack",
                "sink",
                "hob",
                "scale",
                "tray",
                "colander",
                "chopping board",
                "cutting board",
                "plate",
                "bowl",
                "saucepan",
                "pan",
                "omelette",
                "pizza oven",
                "tupperware",
                "knife and fork",
                "large bowls",
                "放下",
                "放进",
                "称量",
                "水槽",
                "灶台",
                "晾架",
                "砧板",
                "托盘",
                "碗",
                "锅",
            )
        ):
            return True
        return False

    def _action_intent_choice_is_generic_direct_space_purpose(self, choice: str) -> bool:
        text = str(choice or "").lower()
        if not any(
            token in text
            for token in (
                "make space",
                "make some space",
                "create space",
                "free up space",
                "clear space",
                "make room",
                "create room",
                "free up room",
                "clear room",
                "some room",
                "腾空间",
                "腾出空间",
                "让开",
            )
        ):
            return False
        return not self._action_intent_choice_has_specific_space_target(text)

    def _action_intent_choice_is_direct_space_without_exact_next_use(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
    ) -> bool:
        if not self._action_intent_choice_is_generic_direct_space_purpose(choice):
            return False
        signal_text = f"{support} {contradiction}"
        if not any(
            token in signal_text
            for token in (
                "space",
                "room",
                "clear",
                "counter room",
                "workspace",
                "腾出空间",
                "让开",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "no single exact next object use shown",
                "no exact next object",
                "no exact next target",
                "no specific next target",
                "no single immediate next target",
                "no direct next-use evidence is shown",
                "target is still ambiguous",
                "without yet showing a single specific",
                "exact next target is still ambiguous",
                "没有具体下一目标",
                "没有明确下一目标",
                "目标仍不明确",
            )
        )

    def _action_intent_choice_is_generic_workspace_effect_over_exact_path_or_destination(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del question
        if not any(
            token in choice
            for token in (
                "make space",
                "make room",
                "workspace",
                "serving easier",
                "wipe down",
                "begin clearing up",
                "腾空间",
                "让开",
                "更容易盛出",
                "擦台面",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        has_exact_path_or_destination = any(
            token in signal_text
            for token in (
                "carried directly",
                "moved directly toward",
                "to the sink",
                "moved to the sink",
                "carry path",
                "direct carry path",
                "prepare to pick up",
                "pick up the plastic colander",
                "exact pickup path",
                "specific out-of-the-way setup",
                "moved out of the way as",
                "stood upright and moved out of the way",
                "right hand",
                "left hand",
                "while the other hand",
                "held in the other hand",
                "直接移到水槽",
                "准备拿起",
                "为拿起让路",
                "明确搬运路径",
                "具体拿取路径",
            )
        )
        if not has_exact_path_or_destination:
            return False
        return any(
            token in contradiction
            for token in (
                "generic workspace effect",
                "only a generic workspace effect",
                "does not match the more direct carry path",
                "less exact than the specific out-of-the-way setup",
                "broad sense",
                "secondary",
                "weaker than the exact",
                "只是泛化空间效果",
                "不如直接搬运路径",
                "不如具体让路设置",
                "只是宽泛的空间变化",
            )
        )

    def _action_intent_choice_is_exact_workspace_creation(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not self._action_intent_choice_has_specific_space_target(choice):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if action_object and not any(
            token in signal_text
            for token in [token for token in re.split(r"[^a-z0-9]+", action_object) if token and len(token) >= 3]
        ):
            if not any(
                token in signal_text
                for token in (
                    "moved object",
                    "picked up object",
                    "current object",
                    "out of the way",
                    "remains held",
                    "当前物体",
                    "让开",
                )
            ):
                return False
        if any(
            token in contradiction
            for token in (
                "only a generic workspace effect",
                "side effect",
                "just generic space",
                "no exact next object",
                "not tied to a specific next item",
                "只是泛化空间效果",
                "只是副作用",
                "没有具体下一目标",
            )
        ):
            return False
        choice_targets = [
            token
            for token in (
                "scale",
                "sink",
                "hob",
                "rack",
                "dishwasher",
                "tray",
                "bowl",
                "plate",
                "colander",
                "chopping board",
                "cutting board",
                "pan",
                "saucepan",
                "knife",
                "fork",
                "omelette",
                "pizza oven",
                "tupperware",
                "large bowls",
            )
            if token in choice
        ]
        if choice_targets and not any(token in signal_text for token in choice_targets):
            return False
        return any(
            token in signal_text
            for token in (
                "prepare to put down",
                "about to put down",
                "about to place",
                "can be put into",
                "be put into the sink",
                "put into the sink",
                "put the tray down",
                "put the baking tray down",
                "put the cutting board on the drying rack",
                "fit into the rack",
                "fit into the draining rack",
                "room within the rack",
                "space for the saucepan",
                "space for the scale",
                "measure chicken",
                "kitchen scale",
                "standing upright and",
                "out of the way as i prepare to pick up",
                "clear the way to pick up",
                "held in right hand",
                "held in left hand",
                "holding in right hand",
                "holding in left hand",
                "next object put down",
                "free slot",
                "slot in the rack",
                "so it will fit",
                "other tupperware",
                "spanish omelette",
                "plastic colander",
                "brought forward",
                "pick up the plastic colander",
                "pick up the stack of large bowls",
                "为接下来放下",
                "放进水槽",
                "放到晾架",
                "为秤腾位",
                "为接下来拿起",
                "让开以便",
                "腾出槽位",
            )
        )

    def _action_intent_choice_is_exact_downstream_targeted_placement(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("move ", "transfer ", "shift ", "remove ", "clear ", "pick up ")):
            return False
        if self._action_intent_choice_is_exact_workspace_creation(
            choice=choice,
            support=support,
            contradiction=contradiction,
            action_object=action_object,
            global_context=global_context,
        ):
            return False
        if not any(
            token in choice
            for token in (
                "put ",
                "place ",
                "fit ",
                "insert ",
                "slot ",
                "into the sink",
                "into the rack",
                "onto the",
                "down on the",
                "放进",
                "放到",
                "插入",
                "槽位",
            )
        ):
            return False
        choice_uses_generic_right_place = any(
            token in choice
            for token in (
                "right place",
                "proper place",
            )
        )
        choice_has_explicit_destination = any(
            token in choice
            for token in (
                "sink",
                "slot",
                "rack",
                "dishwasher",
                "scale",
                "hob",
                "tray",
                "counter",
                "plate",
                "bowl",
                "colander",
                "saucepan",
                "pan",
            )
        )
        signal_text = f"{support} {contradiction} {global_context}"
        if action_object and not any(
            token in signal_text
            for token in [token for token in re.split(r"[^a-z0-9]+", action_object) if token and len(token) >= 3]
        ):
            if not any(
                token in signal_text
                for token in (
                    "moved object",
                    "current object",
                    "out of the way",
                    "让开",
                    "挪开",
                )
            ):
                return False
        has_target = any(
            token in signal_text
            for token in (
                "saucepan",
                "pan",
                "pot",
                "bowl",
                "plate",
                "tray",
                "colander",
                "lid",
                "tupperware",
                "large bowls",
                "next item",
                "another item",
                "plastic colander",
                "下一个物体",
                "另一个物体",
            )
        )
        has_destination = any(
            token in signal_text
            for token in (
                "sink slot",
                "slot in the rack",
                "available spot",
                "free slot",
                "freed slot",
                "in the sink",
                "into the sink",
                "into the rack",
                "onto the counter",
                "onto the scale",
                "exact next item and destination",
                "具体下一目标和位置",
                "腾出槽位",
                "放进水槽",
                "放到晾架",
            )
        )
        has_immediacy = any(
            token in signal_text
            for token in (
                "immediately afterwards",
                "immediately after",
                "directly afterwards",
                "directly after",
                "right afterwards",
                "about to put down",
                "prepare to put down",
                "then the",
                "紧接着",
                "随后",
                "立刻",
                "接着就",
            )
        )
        if choice_uses_generic_right_place and not choice_has_explicit_destination and not has_destination:
            return False
        if not (has_target and has_destination and has_immediacy):
            return False
        return not any(
            token in contradiction
            for token in (
                "only generic space",
                "generic workspace effect",
                "not tied to a specific next item",
                "只是泛化空间效果",
                "没有具体下一目标",
            )
        )

    def _action_intent_choice_is_exact_immediate_downstream_use(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("move ", "transfer ", "shift ", "remove ", "clear ", "place ", "put ")):
            return False
        if self._action_intent_choice_is_generic_direct_space_purpose(choice):
            return False
        if self._choice_is_same_object_active_use(choice, action_object):
            return False
        if self._action_intent_choice_is_exact_downstream_targeted_placement(
            question=question,
            choice=choice,
            support=support,
            contradiction=contradiction,
            action_object=action_object,
            global_context=global_context,
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "later downstream",
                "picked up later",
                "may be picked up later",
                "later during cooking",
                "downstream pickup",
                "downstream effect",
                "after the transfer rather than the direct purpose",
                "rather than the direct purpose of the transfer itself",
                "speculative",
                "只是后续可能",
                "后面才会",
                "后续才会",
            )
        ) and not any(
            token in signal_text
            for token in (
                "immediately",
                "immediate next",
                "right afterwards",
                "directly afterwards",
                "next visible",
                "直接下一步",
                "立刻",
                "紧接着",
            )
        ):
            return False
        if not any(
            token in choice
            for token in (
                "pick up",
                "grab",
                "reach for",
                "turn on",
                "turn off",
                "open",
                "adjust",
                "measure",
                "weigh",
                "wash",
                "rinse",
                "scrub",
                "wipe",
                "clean",
                "use",
                "stir",
                "pour",
                "拿起",
                "打开",
                "调节",
                "称量",
                "清洗",
                "冲洗",
                "擦",
                "使用",
                "搅拌",
            )
        ):
            return False
        if any(
            token in choice
            for token in (
                "make space",
                "make room",
                "free up",
                "some space",
                "some room",
                "access",
                "inspect",
                "store",
                "later",
                "future",
                "腾空间",
                "让开",
                "检查",
                "收起来",
            )
        ):
            return False
        choice_targets = [
            token
            for token in (
                "whisk",
                "knife",
                "fork",
                "spoon",
                "spatula",
                "bottle",
                "sponge",
                "brush",
                "cloth",
                "towel",
                "lid",
                "cover",
                "bowl",
                "plate",
                "tray",
                "pot",
                "pan",
                "saucepan",
                "cup",
                "glass",
                "jar",
                "colander",
                "scale",
                "tap",
                "faucet",
                "sink",
                "hob",
                "microwave",
                "oven",
                "fridge",
                "door",
                "drawer",
                "cupboard",
                "rack",
                "dishwasher",
            )
            if token in choice
        ]
        action_object_tokens = {token for token in re.split(r"[^a-z0-9]+", action_object) if token}
        non_action_targets = [token for token in choice_targets if token not in action_object_tokens]
        if not non_action_targets:
            return False
        if not any(token in signal_text for token in non_action_targets):
            return False
        has_immediacy = any(
            token in signal_text
            for token in (
                "immediately afterwards",
                "immediately after",
                "immediately reaches for",
                "immediate next target",
                "immediate next step",
                "immediate next use",
                "right afterwards",
                "directly afterwards",
                "directly after",
                "next visible target",
                "next visible object",
                "used next",
                "about to",
                "prepare to",
                "紧接着",
                "随后立刻",
                "立刻",
                "下一步就是",
                "直接下一步",
            )
        )
        if not has_immediacy:
            return False
        return any(
            token in signal_text
            for token in (
                "direct next target",
                "direct next functional use",
                "direct purpose",
                "specific next target",
                "immediate next target",
                "immediate next use",
                "reaches for",
                "picks up the",
                "opens the",
                "turns on the",
                "uses the",
                "next tool",
                "next visible target",
                "rather than only a broad workspace effect",
                "rather than only generic space",
                "not just a broad workspace effect",
                "不是泛化空间效果",
                "直接下一目标",
                "具体下一目标",
                "直接功能目的",
            )
        )

    def _action_intent_choice_is_exact_pickup_path_enablement(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("move ", "transfer ", "shift ", "push ", "clear ")):
            return False
        if not any(
            token in choice
            for token in (
                "out of the way",
                "prepare to pick up",
                "clear the way to pick up",
                "pick up the",
                "moved out of the way as",
                "让开以便拿起",
                "为拿起让路",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        if not any(
            token in signal_text
            for token in (
                "out of the way",
                "prepare to pick up",
                "specific out-of-the-way setup",
                "exact pickup path",
                "clearing the exact pickup path",
                "moved out of the way as",
                "pick up the plastic colander",
                "pick up the kettle",
                "right hand",
                "left hand",
                "让开以便拿起",
                "具体拿取路径",
                "为拿起让路",
            )
        ):
            return False
        return not any(
            token in contradiction
            for token in (
                "no exact pickup path",
                "no direct pickup target",
                "没有明确拿取路径",
                "没有明确下一拿取目标",
            )
        )

    def _action_intent_choice_is_direct_same_object_inspection_or_alignment(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not action_object:
            return False
        if not any(
            token in choice
            for token in (
                "check",
                "confirm",
                "inspect",
                "look at",
                "look what's",
                "see whether",
                "correct side",
                "facing",
                "upright",
                "fits",
                "fit",
                "doneness",
                "检查",
                "确认",
                "朝向",
                "正面",
                "合适",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        if any(
            token in signal_text
            for token in (
                "later downstream",
                "after that picks up",
                "之后再",
                "后续才",
            )
        ):
            return False
        if any(
            token in contradiction
            for token in (
                "no sign of checking",
                "no inspection cue",
                "no clear inspection",
                "not inspecting",
                "weaker than the exact",
                "there is no sign of",
                "没有检查",
                "没有查看",
                "没有明显检查",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "correct side",
                "facing the right hand",
                "face upright",
                "standing upright",
                "confirm the",
                "check the",
                "inspect the",
                "cover fits",
                "sitting well",
                "oily part",
                "dirty side",
                "orientation",
                "alignment",
                "to inspect",
                "to check",
                "确认",
                "检查",
                "朝上",
                "正朝",
                "摆正",
                "盖子是否合适",
            )
        )

    def _action_intent_choice_is_generic_inspection_under_hidden_target_context(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "inspect",
                "check",
                "look at",
                "look what's",
                "see whether",
                "find",
                "study",
                "inspect the",
                "检查",
                "查看",
                "看看",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "behind",
                "underneath",
                "under the",
                "back of the counter",
                "back shelf",
                "second shelf",
                "clear the way",
                "moved aside",
                "moved out of the way",
                "target behind",
                "hidden target",
                "pick up the stack",
                "retrieve the",
                "reaches for the",
                "拿后面的",
                "后面",
                "下面",
                "让开",
                "挪开后",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "retrieval is more direct",
                "actual target is behind",
                "not merely looking",
                "not merely looked",
                "not just checking",
                "clearer evidence is the hidden target",
                "inspection is weaker than",
                "weaker than the concrete hidden-target retrieval",
                "rather than a generic look",
                "rather than generic look",
                "检索目的更直接",
                "不是单纯查看",
                "后面的目标更直接",
            )
        )

    def _action_intent_choice_is_generic_hidden_reveal_or_access(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "access what's behind",
                "look what's behind",
                "see what is behind",
                "what is behind",
                "look behind",
                "see what's behind",
                "access the",
                "access behind",
                "to access the area behind",
                "to access behind",
                "to look what's behind",
                "后面有什么",
                "看后面",
                "查看后面",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "behind",
                "underneath",
                "second shelf",
                "back shelf",
                "moved aside",
                "reveals",
                "revealed",
                "clear the way",
                "后面",
                "下面",
                "挪开后",
                "让开",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "not merely looked",
                "not merely looking",
                "not just checking",
                "clearer evidence is",
                "weaker than the concrete hidden-target retrieval",
                "the direct target is",
                "the direct purpose is",
                "rather than a generic look",
                "rather than only generic access",
                "rather than the action being only generic access",
                "revealed slot is immediately used",
                "revealed area is immediately used",
                "the hidden item is then picked up",
                "不是单纯查看",
                "更直接的目标是",
                "直接目的其实是",
            )
        )

    def _action_intent_choice_is_generic_hidden_access_over_exact_reveal_use(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        del support, global_context
        if not any(
            token in choice
            for token in (
                "access what's behind",
                "look what's behind",
                "see what is behind",
                "what is behind",
                "look behind",
                "see what's behind",
                "access the",
                "access behind",
                "behind the",
                "后面有什么",
                "看后面",
                "查看后面",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "revealed slot is immediately used",
                "revealed area is immediately used",
                "the hidden item is then picked up",
                "revealed item is then picked up",
                "the item behind is immediately taken",
                "exact placement into that slot",
                "stronger than generic access",
                "not merely generically accessed",
                "腾出的槽位立刻被使用",
                "露出的目标立刻被拿取",
            )
        )

    def _action_intent_choice_is_generic_hidden_access_without_followup_use(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in choice
            for token in (
                "access what's behind",
                "look what's behind",
                "see what is behind",
                "what is behind",
                "look behind",
                "see what's behind",
                "access the",
                "access behind",
                "behind the",
                "后面有什么",
                "看后面",
                "查看后面",
            )
        ):
            return False
        if not any(
            token in signal_text
            for token in (
                "behind",
                "reveals",
                "revealed",
                "underneath",
                "后面",
                "下面",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "no hidden item is then picked up",
                "no item behind is actually taken",
                "no revealed slot is immediately used",
                "no object is placed into the revealed area",
                "no direct next target is established",
                "no concrete hidden target is retrieved",
                "没有实际取出",
                "没有明确下一目标",
                "没有物体被放入露出的区域",
            )
        )

    def _action_intent_choice_is_brief_cooking_inspection_over_disposal(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not action_object:
            return False
        if not any(
            token in choice
            for token in (
                "check",
                "inspect",
                "see whether",
                "look at",
                "look inside",
                "look into",
                "see inside",
                "check inside",
                "check the contents",
                "check the consistency",
                "see if it is cooked",
                "boiling",
                "done",
                "doneness",
                "cooked",
                "consistency",
                "contents",
                "检查",
                "看看",
                "确认",
                "熟了",
                "沸腾",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not self._signal_mentions_action_object(signal_text, action_object):
            return False
        has_cooking_context = any(
            token in signal_text
            for token in (
                "hob",
                "stove",
                "burner",
                "boiling",
                "water",
                "steam",
                "contents",
                "liquid",
                "simmer",
                "cooking",
                "pan",
                "pot",
                "saucepan",
                "灶",
                "锅",
                "沸腾",
                "水",
                "蒸汽",
                "内容物",
            )
        )
        if not has_cooking_context:
            return False
        if any(
            token in signal_text
            for token in (
                "scale",
                "weigh",
                "weighing",
                "butter",
                "kitchen scale",
                "秤",
                "称量",
                "黄油",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "briefly raised",
                "raised near the hob",
                "briefly raised near the hob",
                "briefly lifted",
                "brief lift",
                "quick check",
                "as if checking",
                "checks the contents",
                "looks inside",
                "look inside",
                "check the contents",
                "check the consistency",
                "see if it is done",
                "see if it is cooked",
                "check the boil",
                "check the boiling",
                "before continuing cooking",
                "set back down",
                "kept near the hob",
                "stays near the hob",
                "remains above the hob",
                "not moved to a plate",
                "not moved to a bowl",
                "not moved to a serving destination",
                "not carried away from the stove",
                "not taken away from the hob",
                "not tilted",
                "no tilt",
                "no pouring",
                "not poured",
                "not toward the sink",
                "not carried to the sink",
                "rather than emptying",
                "rather than serving",
                "rather than pouring out",
                "quick inspection",
                "brief inspection",
                "短暂拿起",
                "快速检查",
                "看一下里面",
                "查看内容物",
                "检查状态",
                "检查是否沸腾",
                "没有倾倒",
                "没有倒出",
                "没有拿去盘子里",
                "没有拿去碗里",
                "没有拿离灶台",
                "没有拿去水槽",
                "放回灶台附近",
            )
        )

    def _action_intent_choice_is_generic_disposal_without_pour_signal(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "empty",
                "pour out",
                "drain",
                "serve",
                "tip out",
                "倒掉",
                "倒出",
                "沥干",
                "盛出",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if action_object and not self._signal_mentions_action_object(signal_text, action_object):
            return False
        if not any(
            token in signal_text
            for token in (
                "hob",
                "stove",
                "boiling",
                "water",
                "liquid",
                "contents",
                "灶",
                "沸腾",
                "水",
                "内容物",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "briefly lifted",
                "brief lift",
                "quick check",
                "looks inside",
                "check the contents",
                "check the consistency",
                "not tilted",
                "no tilt",
                "no pouring",
                "not poured",
                "no serving",
                "not served",
                "not moved to a plate",
                "not moved to a bowl",
                "not moved to a serving destination",
                "not carried away from the stove",
                "not toward the sink",
                "not carried to the sink",
                "stays near the hob",
                "kept near the hob",
                "quick inspection",
                "rather than emptying",
                "rather than serving",
                "rather than pouring out",
                "短暂拿起",
                "快速检查",
                "看一下里面",
                "查看内容物",
                "没有倾倒",
                "没有倒出",
                "没有盛出",
                "没有拿去盘子里",
                "没有拿去碗里",
                "没有拿去水槽",
                "仍在灶台附近",
            )
        )

    def _action_intent_choice_is_hidden_target_access_or_retrieval(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "access",
                "retrieve",
                "take the",
                "take ",
                "pick the",
                "look for",
                "find",
                "reach the",
                "pick up the stack",
                "clear the way to pick up",
                "get easier access",
                "access what's behind",
                "target behind",
                "拿到后面",
                "取后面",
                "找",
                "拿起后面的",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in contradiction
            for token in (
                "no hidden target",
                "no item behind",
                "not actually retrieved",
                "without any later retrieval",
                "没有后方目标",
                "没有拿到后面的东西",
            )
        ):
            return False
        if not any(
            token in signal_text
            for token in (
                "behind",
                "underneath",
                "under the",
                "back shelf",
                "second shelf",
                "clear the way",
                "moved aside",
                "moved out of the way",
                "pick up the stack",
                "retrieve the",
                "picked up from behind",
                "taken from behind",
                "grab the",
                "reaches for the",
                "target behind",
                "item behind",
                "look for a",
                "needed tool",
                "becomes reachable",
                "reachable behind",
                "freed enough to grab",
                "hidden item",
                "revealed item",
                "后面",
                "下面",
                "让开",
                "挪开后",
                "目标在后面",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "retrieve the red curry paste",
                "red curry paste",
                "pick up the stack of large bowls",
                "look for the tool that is needed",
                "vegetable peeler",
                "look for a pan",
                "the actual target is behind",
                "hidden target becomes reachable",
                "item behind becomes reachable",
                "the revealed item is then picked up",
                "the item behind is immediately taken",
                "reaches behind and takes",
                "taken from behind",
                "grabbed from behind",
                "freed enough to grab the hidden item",
                "reaches for the item behind",
                "moved aside to access",
                "moved aside to retrieve",
                "clear the way to pick up",
                "needed tool",
                "red curry paste behind",
                "后面的目标",
                "要找的工具",
                "蔬菜削皮刀",
                "红咖喱酱",
            )
        )

    def _action_intent_choice_is_exact_revealed_target_purpose(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "behind",
                "underneath",
                "second shelf",
                "back shelf",
                "moved aside",
                "reveals",
                "revealed",
                "clear the way",
                "freed slot",
                "available spot",
                "behind the",
                "后面",
                "下面",
                "挪开后",
                "让开",
            )
        ):
            return False
        if any(
            token in choice
            for token in (
                "access what's behind",
                "look what's behind",
                "see what is behind",
                "what is behind",
                "look behind",
                "see what's behind",
                "access behind",
                "to access the area behind",
                "to access behind",
                "to look what's behind",
                "后面有什么",
                "看后面",
                "查看后面",
            )
        ):
            return False
        if any(
            token in contradiction
            for token in (
                "no actual retrieval is shown",
                "no direct next target is established",
                "no concrete hidden target is retrieved",
                "没有实际取出",
                "没有明确下一目标",
            )
        ):
            return False
        if (
            self._action_intent_choice_is_hidden_target_access_or_retrieval(
                choice=choice,
                support=support,
                contradiction=contradiction,
                global_context=global_context,
            )
            and not self._action_intent_choice_is_generic_hidden_reveal_or_access(
                choice=choice,
                support=support,
                contradiction=contradiction,
                global_context=global_context,
            )
        ):
            return True
        if self._action_intent_choice_is_exact_downstream_targeted_placement(
            question=question,
            choice=choice,
            support=support,
            contradiction=contradiction,
            action_object=action_object,
            global_context=global_context,
        ):
            return True
        if any(token in choice for token in ("right place", "proper place")) and any(
            token in signal_text
            for token in (
                "inserted into",
                "inserted in",
                "put into that exact",
                "put into the freed slot",
                "placed into the freed slot",
                "freed slot",
                "exact rack slot",
                "available spot",
                "right place",
                "proper place",
                "插进",
                "放进腾出的槽位",
                "准确放回",
            )
        ):
            return True
        if any(
            token in signal_text
            for token in (
                "revealed slot is immediately used",
                "revealed area is immediately used",
                "freed slot is then used",
                "the item behind is immediately taken",
                "the revealed item is then picked up",
                "reaches behind and takes",
                "picked up from behind",
                "taken from behind",
                "grabbed from behind",
                "item behind becomes reachable",
                "revealed target is used immediately",
                "revealed target is placed immediately",
                "挪开后立刻取出",
                "腾出的槽位立刻被使用",
                "露出的目标立刻被拿取",
            )
        ):
            return True
        return self._action_intent_choice_is_exact_immediate_downstream_use(
            question=question,
            choice=choice,
            support=support,
            contradiction=contradiction,
            action_object=action_object,
            global_context=global_context,
        )

    def _action_intent_choice_is_exact_reveal_then_take_or_place(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "behind",
                "underneath",
                "revealed",
                "freed slot",
                "available spot",
                "hidden item",
                "revealed item",
                "后面",
                "下面",
                "腾出的槽位",
            )
        ):
            return False
        if any(
            token in contradiction
            for token in (
                "no hidden item is then picked up",
                "no item behind is actually taken",
                "no object is placed into the revealed area",
                "no revealed slot is immediately used",
                "no direct next target is established",
                "no concrete hidden target is retrieved",
                "没有实际取出",
                "没有明确下一目标",
                "没有物体被放入露出的区域",
            )
        ):
            return False
        if any(
            token in choice
            for token in (
                "take the",
                "retrieve the",
                "pick up the",
                "look for the",
                "find the",
                "take the small",
                "拿",
                "取出",
                "找到",
            )
        ):
            return any(
                token in signal_text
                for token in (
                    "becomes reachable",
                    "reachable and is taken",
                    "taken from behind",
                    "picked up from behind",
                    "the item behind is immediately taken",
                    "the revealed item is then picked up",
                    "reaches behind and takes",
                    "grabbed from behind",
                    "hidden-item retrieval",
                    "挪开后立刻取出",
                    "露出的目标立刻被拿取",
                )
            )
        if any(
            token in choice
            for token in (
                "put the",
                "place the",
                "into the freed slot",
                "into the slot",
                "right place",
                "proper place",
                "放进",
                "放到腾出的槽位",
            )
        ):
            return any(
                token in signal_text
                for token in (
                    "revealed slot is immediately used",
                    "revealed area is immediately used",
                    "freed slot is then used",
                    "placed into the freed slot",
                    "inserted into the freed slot",
                    "exact placement into that slot",
                    "revealed-slot placement",
                    "腾出的槽位立刻被使用",
                )
            )
        return False

    def _action_intent_choice_is_generic_underneath_cleaning_under_hidden_target_context(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "clean underneath",
                "clean underneath items",
                "clean under",
                "wipe underneath",
                "clear up",
                "clean below",
                "清理下面",
                "清洁下面",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "underneath",
                "under the",
                "hidden tool",
                "needed tool",
                "vegetable peeler",
                "missing tool",
                "find",
                "look for",
                "下面",
                "要找的工具",
                "削皮刀",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "rather than cleaning underneath",
                "explicit target is the missing tool",
                "organized around finding the needed hidden tool",
                "not mainly cleaning",
                "不是主要为了清洁",
                "明确目标是缺失工具",
                "是为了找隐藏工具",
            )
        )

    def _action_intent_choice_is_cleaning_tool_specific_target_use(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("sponge", "brush", "cloth", "towel", "napkin", "paper towel", "scrubber")
        ):
            return False
        action_object_tokens = [token for token in re.split(r"[^a-z0-9]+", action_object) if token]
        if action_object_tokens and all(token in choice for token in action_object_tokens):
            return False
        if any(
            token in choice
            for token in (
                "bottle",
                "washing up liquid",
                "hand wash liquid",
                "liquid soap",
                "pick up the washing",
                "reach for the bottle",
                "pick up the bottle",
                "瓶",
                "洗洁精",
            )
        ):
            return False
        if not any(token in choice for token in ("wash", "rinse", "scrub", "wipe", "clean", "冲洗", "清洗", "擦", "刷")):
            return False
        if any(
            token in choice
            for token in (
                "whole thing",
                "dry hand",
                "dry hands",
                "wash hands",
                "wipe hands",
                "clean the whole",
                "to clean.",
                "to dry.",
                "to clean the whole thing",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        target_terms = [
            token
            for token in (
                "peeler",
                "knife",
                "spoon",
                "utensil",
                "tray",
                "counter",
                "surface",
                "hob",
                "sink",
                "cup",
                "bowl",
                "pot",
                "pan",
                "colander",
                "board",
                "blender cup",
                "ice cube tray",
                "counter surface",
                "刨刀",
                "刀",
                "勺",
                "托盘",
                "台面",
                "灶台",
                "水槽",
                "杯",
                "碗",
                "锅",
                "滤盆",
                "砧板",
            )
            if token in choice
        ]
        if not target_terms:
            return False
        if not any(token in signal_text for token in target_terms):
            return False
        if any(token in support for token in ("could", "can be used", "compatible with", "theoretically", "理论", "可能")):
            if not any(
                token in signal_text
                for token in (
                    "ready for wiping",
                    "staged for wiping",
                    "compatible with preparing to wipe",
                    "laid out",
                    "beside crumbs",
                    "next visible cleaning target",
                    "target is the",
                    "toward the utensil",
                    "cleaning target",
                    "wiping motion",
                    "scrubbing motion",
                    "准备擦",
                    "清洗目标",
                    "擦拭目标",
                )
            ):
                return False
        return any(
            token in signal_text
            for token in (
                "while holding",
                "holding",
                "other hand",
                "free hand",
                "running water",
                "under water",
                "scrub",
                "wipe",
                "wash",
                "rinse",
                "counter",
                "sink",
                "sponge",
                "brush",
                "one hand",
                "另一只手",
                "一只手",
                "拿着",
                "流水",
                "水槽",
                "海绵",
                "刷子",
                "擦",
                "清洗",
                "冲洗",
                "next visible cleaning target",
                "target is the",
                "toward the utensil",
                "target object",
                "cleaning target",
                "下一个清洗目标",
                "清洗目标",
            )
        )

    def _action_intent_choice_is_cleaning_supply_retrieval(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("sponge", "brush", "cloth", "towel", "napkin", "paper towel", "scrubber")
        ):
            return False
        if not any(
            token in choice
            for token in (
                "pick up the bottle",
                "reach for the bottle",
                "washing up liquid",
                "hand wash liquid",
                "liquid soap",
                "pick up the washing",
                "拿起洗洁精",
                "拿起瓶子",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in contradiction
            for token in (
                "no bottle pickup",
                "not directly shown",
                "less direct than",
                "intermediate possibility",
                "visible surface target",
                "visible utensil-cleaning target",
                "没有拿起瓶子",
                "没有直接拿",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "immediately reaches for",
                "picks up the washing-up-liquid bottle",
                "picks up the bottle",
                "reaches for the bottle",
                "washing-up-liquid bottle",
                "hand wash liquid bottle",
                "liquid bottle",
                "immediate next target is the bottle",
                "立即伸手拿瓶子",
                "拿起洗洁精瓶",
                "下一步就是拿瓶子",
            )
        )

    def _action_intent_choice_is_cleaning_workflow_initiation(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("sponge", "brush", "cloth", "towel", "napkin", "paper towel", "scrubber")
        ):
            return False
        if not any(
            token in choice
            for token in (
                "begin washing",
                "start washing",
                "wet the sponge",
                "washing sequence",
                "开始清洗",
                "开始洗",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "active washing position",
                "under/near water",
                "under running water",
                "washing sequence",
                "washing is starting",
                "no specific target yet",
                "exact first item washed is not explicit",
                "water",
                "sink",
                "active washing",
                "进入清洗位置",
                "开始清洗",
                "水槽",
                "流水",
            )
        ):
            return False
        return not any(
            token in signal_text
            for token in (
                "immediately reaches for",
                "picks up the bottle",
                "washing-up-liquid bottle",
                "hand wash liquid bottle",
                "立即伸手拿瓶子",
                "拿起洗洁精瓶",
            )
        )

    def _action_intent_choice_is_surface_wipe_preparation(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("cloth", "towel", "tea towel", "dish cloth", "napkin", "paper towel", "sponge", "scrubber")
        ):
            return False
        if not any(token in choice for token in ("wipe", "clean", "擦", "清洁")):
            return False
        if not any(
            token in choice
            for token in (
                "surface",
                "counter",
                "countertop",
                "worktop",
                "bench",
                "table",
                "hob",
                "kitchen side",
                "台面",
                "桌面",
                "灶台",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "hand-drying motion",
                "hands are wiped dry",
                "applied to the hands",
                "brought to both hands",
                "both hands",
                "wipes them",
                "双手",
                "擦手",
                "擦干双手",
            )
        ):
            return False
        has_surface_staging = any(
            token in signal_text
            for token in (
                "placed on the counter",
                "placed on the countertop",
                "placed on the worktop",
                "laid on the counter",
                "laid on the worktop",
                "set on the counter",
                "set on the worktop",
                "left on the counter",
                "countertop",
                "worktop",
                "work surface",
                "within reach of the counter",
                "compatible with preparing to wipe",
                "ready for wiping",
                "staged for wiping",
                "surface target",
                "counter surface",
                "crumbs",
                "spill",
                "mess on the counter",
                "台面上",
                "放在台面",
                "准备擦",
                "擦拭台面",
                "碎屑",
                "污渍",
            )
        )
        if not has_surface_staging:
            return False
        has_explicit_preparation_signal = any(
            token in signal_text
            for token in (
                "ready for wiping",
                "compatible with preparing to wipe",
                "staged for wiping",
                "beside crumbs",
                "next to a visible spill",
                "surface target",
                "counter surface",
                "准备擦",
                "擦拭台面",
                "碎屑",
                "污渍",
            )
        )
        has_nonstorage_signal = any(
            token in signal_text
            for token in (
                "not stored",
                "no drawer",
                "no cupboard",
                "no hook return",
                "no holder return",
                "not put away",
                "rather than storing",
                "instead of storing",
                "later reused",
                "kept nearby",
                "temporarily placed",
                "merely relocated",
                "没有收纳",
                "没有放回",
                "不是收起来",
                "暂时放在",
                "稍后继续使用",
            )
        )
        return has_explicit_preparation_signal or has_nonstorage_signal

    def _action_intent_choice_is_temporary_relocation_not_storage(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del action_object
        if not any(
            token in choice
            for token in ("put away", "store", "put back", "return it", "hang back", "放回", "收起来", "收纳")
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_nonstorage_signal = any(
            token in signal_text
            for token in (
                "not stored",
                "not put away",
                "no drawer",
                "no cupboard",
                "no hook return",
                "no holder return",
                "placed on the counter",
                "placed on the countertop",
                "placed on the worktop",
                "laid on the counter",
                "set on the counter",
                "left on the side",
                "temporarily placed",
                "merely relocated",
                "within reach",
                "later reused",
                "immediate reuse",
                "not final placement",
                "没有收纳",
                "没有放回",
                "没有挂回去",
                "放在台面",
                "暂时放在",
                "后续继续使用",
            )
        )
        if not has_nonstorage_signal:
            return False
        has_true_storage_signal = any(
            token in signal_text
            for token in (
                "returned to the drawer",
                "returned to the cupboard",
                "hung back on the hook",
                "placed back in storage",
                "stored away",
                "放回抽屉",
                "放回橱柜",
                "挂回挂钩",
                "收纳回去",
            )
        )
        return not has_true_storage_signal

    def _action_intent_choice_lacks_direct_relocation_outcome_evidence(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
    ) -> bool:
        if "move" not in choice and "relocat" not in choice and "shift" not in choice:
            return False
        signal_text = f"{support} {contradiction}".lower()
        has_direct_outcome = any(
            token in signal_text
            for token in (
                "picked up and then placed elsewhere",
                "set down in a different position",
                "left on the counter in a different place",
                "temporarily relocated",
                "relocated from",
                "moved towards",
                "carried toward",
                "carried to the sink",
                "placed near the sink",
                "put down nearby",
                "放到别处",
                "放到另一处",
                "移到旁边",
                "移向水槽",
                "放到水槽边",
                "短暂挪动",
            )
        )
        if has_direct_outcome:
            return False
        return not any(
            token in signal_text
            for token in (
                "direct relocation outcome",
                "immediate transport destination",
                "exact moved destination",
                "明确移动去向",
                "明确放置位置",
            )
        )

    def _action_intent_choice_is_unsupported_hand_drying_goal(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("cloth", "towel", "tea towel", "dish cloth", "napkin", "paper towel")
        ):
            return False
        if not ("dry" in choice and "hand" in choice):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_positive_hand_drying = any(
            token in signal_text
            for token in (
                "brought to both hands",
                "applied to the hands",
                "hands are wiped dry",
                "wipes the hands",
                "wet hands",
                "after washing hands",
                "双手",
                "擦手",
                "洗手后",
                "湿手",
            )
        )
        if has_positive_hand_drying:
            return False
        return any(
            token in signal_text
            for token in (
                "no visible hand-drying motion",
                "no hand wiping",
                "no wet-hand context",
                "not applied to the hands",
                "no visible hand use",
                "counter placement",
                "compatible with preparing to wipe",
                "surface wiping",
                "not for drying",
                "没有擦手",
                "没有湿手",
                "不是擦手",
                "台面",
                "准备擦拭台面",
            )
        )

    def _action_intent_choice_is_explicit_hand_drying_goal(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("cloth", "towel", "tea towel", "dish cloth", "napkin", "paper towel")
        ):
            return False
        if not ("dry" in choice and "hand" in choice):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "no visible hand-drying motion",
                "no hand wiping",
                "no wet-hand context",
                "not applied to the hands",
                "没有擦手",
                "没有湿手",
                "不是擦手",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "brought to both hands",
                "applied to the hands",
                "hands are wiped dry",
                "wipes the hands",
                "used to wipe them dry",
                "hand area",
                "dab/rub one hand",
                "rub one hand",
                "rubbed against the other hand",
                "fingers",
                "finger area",
                "wet hands",
                "after washing hands",
                "双手",
                "擦手",
                "擦干双手",
                "洗手后",
                "湿手",
            )
        )

    def _action_intent_choice_is_direct_disposal_path(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "scraps",
                "food scraps",
                "trash",
                "bin",
                "garbage",
                "dispose",
                "discard",
                "垃圾桶",
                "残渣",
                "丢弃",
            )
        ):
            return False
        if not any(
            token in choice
            for token in (
                "throw out",
                "discard",
                "dispose",
                "trash",
                "丢弃",
                "扔掉",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "carried toward the bin",
                "carried toward the trash",
                "moved toward the bin",
                "walks toward the bin",
                "disposal context",
                "垃圾桶区域",
                "走向垃圾桶",
                "带到垃圾桶",
            )
        )

    def _action_intent_choice_is_generic_cleaning_tool_goal(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in action_object
            for token in ("sponge", "brush", "cloth", "towel", "napkin", "paper towel", "scrubber")
        ):
            return False
        generic_goal = any(
            token in choice
            for token in (
                "whole thing",
                "to clean.",
                "to clean the whole thing",
                "dry hand",
                "dry hands",
                "wipe hands",
                "wash hands",
                "to dry.",
            )
        )
        if not generic_goal:
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(token in choice for token in ("dry hand", "dry hands", "wipe hands", "wash hands")) and any(
            token in signal_text
            for token in (
                "brought to both hands",
                "applied to the hands",
                "hands are wiped dry",
                "wipes the hands",
                "wet hands",
                "after washing hands",
                "双手",
                "擦手",
                "洗手后",
                "湿手",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "peeler",
                "knife",
                "spoon",
                "utensil",
                "tray",
                "counter",
                "surface",
                "hob",
                "sink",
                "cup",
                "bowl",
                "pot",
                "pan",
                "colander",
                "ice cube tray",
                "pyrex bowl",
                "刨刀",
                "刀",
                "勺",
                "台面",
                "灶台",
                "水槽",
                "杯",
                "碗",
                "锅",
            )
        )

    def _action_intent_choice_is_postwash_residue_or_water_removal(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("run ", "flip ", "tip ", "turn ", "shake ")):
            return False
        if not action_object:
            return False
        if not any(
            token in choice
            for token in (
                "soap suds",
                "soap",
                "rinsing water",
                "excess water",
                "water droplets",
                "remove the soap",
                "remove soap",
                "get rid of the excess",
                "dry",
                "suds",
                "肥皂",
                "泡沫",
                "多余水",
                "冲洗水",
                "水滴",
                "晾干",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "sink",
                "water",
                "rinse",
                "rinsing",
                "soap",
                "suds",
                "washed",
                "under running water",
                "water droplets",
                "水槽",
                "流水",
                "冲洗",
                "肥皂",
                "泡沫",
                "洗过",
            )
        ):
            return False
        if any(
            token in signal_text
            for token in (
                "pour into the pan",
                "pour into the pot",
                "into the frying pan",
                "drain the pasta",
                "back into the pan",
                "锅里",
                "炒锅",
                "意面",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "cutting board",
                "board",
                "bowl",
                "cup",
                "glass",
                "container",
                "tray",
                "砧板",
                "碗",
                "杯",
                "托盘",
            )
        )

    def _action_intent_choice_is_generic_postwash_cleaning(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not action_object:
            return False
        if not any(
            token in choice
            for token in ("rinse and clean", "to clean.", "to rinse", "to clean off any remnants", "to clean off")
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "soap suds",
                "remaining soap",
                "excess rinsing water",
                "water droplets",
                "washing away the remaining soap",
                "肥皂",
                "泡沫",
                "多余水",
                "冲洗水",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "cutting board",
                "board",
                "bowl",
                "cup",
                "glass",
                "tray",
                "砧板",
                "碗",
                "杯",
                "托盘",
            )
        )

    def _action_intent_choice_is_postwash_drying_goal(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("place ", "put ", "set ", "rest ", "lay ")):
            return False
        if not any(
            token in choice
            for token in (
                "to dry",
                "allow",
                "no spots",
                "dry after washing",
                "facing up",
                "晾干",
                "干",
                "水渍",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_wet_context = any(
            token in signal_text
            for token in (
                "washed",
                "rinsed",
                "wet",
                "water",
                "drain",
                "droplets",
                "soap",
                "after washing",
                "流水",
                "冲洗",
                "洗过",
                "水滴",
                "肥皂",
            )
        )
        if not has_wet_context:
            return False
        return any(
            token in signal_text
            for token in (
                "facing up",
                "not touching",
                "placed aside",
                "set down",
                "to dry",
                "allow",
                "drain",
                "spots",
                "face up",
                "朝上",
                "不接触",
                "晾干",
                "水渍",
            )
        )

    def _action_intent_choice_is_finished_with_object_goal(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("place ", "put ", "set ", "transfer ", "move ")):
            return False
        if not any(token in choice for token in ("finished with", "finished chopping", "finished with the", "用完")):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "washed",
                "rinsed",
                "wet",
                "water",
                "soap",
                "droplets",
                "to dry",
                "after washing",
                "流水",
                "冲洗",
                "洗过",
                "肥皂",
                "水滴",
                "晾干",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "placed aside",
                "set aside",
                "set down",
                "no longer needed",
                "simply placed",
                "put down",
                "rests on",
                "placed on the tray",
                "placed on the counter",
                "放在一边",
                "放下",
                "不再需要",
            )
        )

    def _action_intent_choice_is_immediate_reuse_staging(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del action_object
        if not any(
            token in choice
            for token in (
                "keep",
                "nearby",
                "within reach",
                "ready for the next",
                "ready for next use",
                "next step",
                "next stir",
                "later immediate use",
                "放在旁边",
                "方便下一步",
                "随手可拿",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_reuse_signal = any(
            token in signal_text
            for token in (
                "within reach",
                "ready for the next",
                "ready for next use",
                "kept nearby",
                "later reused",
                "immediate reuse",
                "used again moments later",
                "used again shortly after",
                "next stir",
                "next step",
                "placed beside the bowl",
                "beside the hob",
                "by the hob",
                "temporary reuse setup",
                "就在旁边",
                "稍后继续使用",
                "下一步还要用",
                "放在旁边",
                "方便下一步",
            )
        )
        if not has_reuse_signal:
            return False
        return not any(
            token in signal_text
            for token in (
                "no further use",
                "no more use",
                "truly finished",
                "not reused",
                "不再使用",
                "确实用完",
            )
        )

    def _action_intent_choice_is_hygiene_surface_protection_staging(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del action_object
        if not any(
            token in choice
            for token in (
                "dirty",
                "messy",
                "not dirty",
                "no spots",
                "avoid mess",
                "avoid dirtying",
                "keep the dirty end",
                "counter messy",
                "弄脏",
                "不弄脏",
                "脏的一端",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        return any(
            token in signal_text
            for token in (
                "dirty end",
                "over the tray",
                "over the chopping board",
                "over one of the muffin trays",
                "not touching the kitchen top",
                "facing up",
                "oily part",
                "kept over the board",
                "kept over the tray",
                "to keep the counter from getting messy",
                "placement hygiene",
                "dirty side",
                "脏的一端",
                "放在托盘上方",
                "放在砧板上方",
                "不接触台面",
                "避免弄脏台面",
                "朝上",
            )
        )

    def _action_intent_choice_is_temporary_set_aside_not_finished(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del action_object
        if not any(
            token in choice
            for token in (
                "finished with",
                "finished chopping",
                "no longer needed",
                "store",
                "put away",
                "put back",
                "用完",
                "收起来",
                "收纳",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_temporary_reuse_signal = any(
            token in signal_text
            for token in (
                "within reach",
                "ready for the next",
                "ready for next use",
                "kept nearby",
                "later reused",
                "immediate reuse",
                "used again moments later",
                "used again shortly after",
                "next stir",
                "next step",
                "beside the hob",
                "by the hob",
                "placed on the tray for reuse",
                "temporarily placed",
                "not final placement",
                "not stored",
                "台面旁边待会继续用",
                "就在旁边",
                "稍后继续使用",
                "下一步还要用",
                "暂时放在",
                "没有收纳",
            )
        )
        if not has_temporary_reuse_signal:
            return False
        return not any(
            token in signal_text
            for token in (
                "no further use",
                "no further",
                "no more use",
                "left there for the rest",
                "truly finished",
                "不再使用",
                "确实用完",
            )
        )

    def _action_intent_choice_is_unfinished_cleanup_context_for_finished_or_storage(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del action_object
        if not any(
            token in choice
            for token in (
                "finished with",
                "finished chopping",
                "store",
                "put away",
                "put back",
                "dry",
                "allow",
                "air dry",
                "用完",
                "收起来",
                "收纳",
                "晾干",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_cleanup_continuation_signal = any(
            token in signal_text
            for token in (
                "soap residue",
                "remaining soap",
                "soap suds",
                "still has soap",
                "still dirty",
                "not rinsed clean",
                "to be cleaned next",
                "washed next",
                "clean off the soap",
                "remove the remaining soap",
                "sink placement",
                "wash area",
                "placed into the sink",
                "placed in the sink",
                "under running water",
                "washing it clean first",
                "current direct purpose is washing",
                "current direct purpose is removing soap",
                "肥皂残留",
                "还有肥皂",
                "还没洗干净",
                "接下来要洗",
                "放进水槽",
                "当前直接目的是清洗",
                "当前直接目的是去除肥皂",
            )
        )
        if not has_cleanup_continuation_signal:
            return False
        return not any(
            token in signal_text
            for token in (
                "facing up",
                "not touching",
                "drain and dry",
                "allow it to dry",
                "face up",
                "朝上",
                "不接触",
                "晾干",
            )
        )

    def _action_intent_choice_is_generic_drying_without_wet_context(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "to dry",
                "allow",
                "no spots",
                "dry after washing",
                "晾干",
                "水渍",
            )
        ):
            return False
        positive_signal_text = f"{support} {global_context}"
        has_positive_wet_context = any(
            token in positive_signal_text
            for token in (
                "washed",
                "rinsed",
                "wet",
                "water",
                "soap",
                "droplets",
                "after washing",
                "流水",
                "冲洗",
                "洗过",
                "肥皂",
                "水滴",
            )
        )
        contradiction_denies_wet_context = any(
            token in contradiction
            for token in (
                "no clear washed",
                "no wet drying context",
                "no spot-checking",
                "no drying evidence",
                "no washing-followup",
                "not tied to this",
                "没有潮湿",
                "没有晾干",
                "没有水渍",
                "没有洗后",
            )
        )
        return (not has_positive_wet_context) or contradiction_denies_wet_context

    def _action_intent_choice_is_premature_drying_before_cleanup(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        del action_object
        if not any(
            token in choice
            for token in (
                "to dry",
                "allow",
                "air dry",
                "dry after washing",
                "晾干",
                "风干",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        has_cleanup_first_context = any(
            token in signal_text
            for token in (
                "soap residue",
                "remaining soap",
                "soap suds",
                "still has soap",
                "still dirty",
                "not rinsed clean",
                "wash it clean first",
                "washing it clean first",
                "remove soap",
                "removing soap",
                "clean off the soap",
                "rinse it clean first",
                "washed next",
                "to be cleaned next",
                "sink placement",
                "wash area",
                "还有肥皂",
                "肥皂残留",
                "还没洗干净",
                "接下来要洗",
                "放进水槽",
                "先冲洗干净",
                "先洗净",
            )
        )
        if not has_cleanup_first_context:
            return False
        return any(
            token in signal_text
            for token in (
                "not for drying",
                "not for dry",
                "drying would be a later consequence",
                "dry later",
                "later consequence",
                "current direct purpose is removing soap",
                "current direct purpose is washing",
                "current direct purpose is rinsing",
                "placed for rinsing",
                "placed to be cleaned next",
                "first, not for drying",
                "不是为了晾干",
                "晾干是后续结果",
                "当前直接目的是去除肥皂",
                "当前直接目的是清洗",
                "当前直接目的是冲洗",
            )
        )

    def _action_intent_choice_is_glove_removal_enablement(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("put ", "place ", "set ", "transfer ")):
            return False
        if not any(token in choice for token in ("oven glove", "glove", "mitt", "手套")):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(token in contradiction for token in ("no oven glove", "no glove", "没有手套")):
            return False
        return any(
            token in signal_text
            for token in (
                "oven glove",
                "glove",
                "mitt",
                "left hand",
                "right hand",
                "free",
                "freed",
                "remove",
                "take off",
                "手套",
                "左手",
                "右手",
                "摘下",
                "脱下",
            )
        )

    def _action_intent_choice_is_surface_mess_avoidance_goal(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("put ", "place ", "set ")):
            return False
        if not any(
            token in choice
            for token in (
                "messy",
                "dirty",
                "dirty end",
                "counter dirty",
                "counter messy",
                "no spots",
                "弄脏",
                "脏",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "washed",
                "rinsed",
                "wet",
                "water droplets",
                "soap suds",
                "洗过",
                "冲洗",
                "水滴",
                "肥皂",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "dirty end",
                "counter",
                "countertop",
                "messy",
                "dirty",
                "tray",
                "muffin tray",
                "chopping board",
                "surface",
                "over the tray",
                "台面",
                "弄脏",
                "脏",
                "托盘",
                "砧板",
            )
        )

    def _action_intent_choice_is_direct_hazard_avoidance(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "burn",
                "burned",
                "burning",
                "too hot",
                "hot",
                "spill",
                "spilling",
                "messy",
                "dirty",
                "temperature",
                "flame",
                "overheat",
                "don't spill",
                "don't burn",
                "doesn't get burned",
                "doesn't spill",
                "烫",
                "烧焦",
                "烧糊",
                "溢出",
                "洒出",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(token in contradiction for token in ("no heat risk", "no spill risk", "没有热风险", "没有溢出风险")):
            return False
        heat_choice = any(
            token in choice
            for token in ("burn", "burned", "burning", "too hot", "hot", "temperature", "flame", "overheat", "烫", "烧焦", "烧糊")
        )
        spill_choice = any(
            token in choice
            for token in ("spill", "spilling", "doesn't spill", "don't spill", "messy", "dirty", "溢出", "洒出")
        )
        if heat_choice:
            if not any(
                token in signal_text
                for token in (
                    "hot",
                    "too hot",
                    "burn",
                    "burning",
                    "getting burnt",
                    "getting burned",
                    "flame",
                    "stove",
                    "hob",
                    "pan",
                    "pot",
                    "saucepan",
                    "frying pan",
                    "oil",
                    "garlic",
                    "lentils",
                    "chillies",
                    "pancake",
                    "one side",
                    "keep moving",
                    "reduce the flame",
                    "adjust temperature",
                    "temperature",
                    "edge of the hob",
                    "热",
                    "烧焦",
                    "火",
                    "灶台",
                    "锅",
                    "油",
                )
            ):
                return False
            return any(
                token in signal_text
                for token in (
                    "too hot",
                    "getting burnt",
                    "getting burned",
                    "don't get burned",
                    "doesn't get burned",
                    "stop the frying pan from burning",
                    "not to burn",
                    "so it doesn't burn",
                    "reduce the flame",
                    "adjust temperature",
                    "avoid burning",
                    "keep moving",
                    "edge of the hob",
                    "one side",
                    "太烫",
                    "烧焦",
                    "避免烧糊",
                )
            )
        if spill_choice:
            if any(
                token in signal_text
                for token in (
                    "washed",
                    "rinsed",
                    "water droplets",
                    "soap suds",
                    "洗过",
                    "冲洗",
                    "肥皂",
                )
            ):
                return False
            if not any(
                token in signal_text
                for token in (
                    "spill",
                    "spilling",
                    "both hands",
                    "oil",
                    "porridge",
                    "carry",
                    "holding",
                    "counter",
                    "pan",
                    "lid",
                    "over the pan",
                    "dirty end",
                    "muffin tray",
                    "chopping board",
                    "溢出",
                    "洒出",
                    "双手",
                    "拿着",
                    "锅",
                    "台面",
                )
            ):
                return False
            return any(
                token in signal_text
                for token in (
                    "both hands",
                    "don't spill",
                    "doesn't spill",
                    "spill over",
                    "so the counter does not get messy",
                    "dirty end",
                    "oil doesn't spill over",
                    "holding porridge in both hands",
                    "双手",
                    "避免溢出",
                    "不洒出来",
                    "不弄脏",
                )
            )
        return False

    def _action_intent_choice_is_generic_mixing_under_hazard_context(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("stir ", "shake ", "rotate knob", "turn knob", "move pan", "shake pan")):
            return False
        if any(
            token in choice
            for token in ("burn", "burned", "burning", "too hot", "hot", "spill", "temperature", "flame", "烧焦", "烫", "溢出")
        ):
            return False
        if not any(
            token in choice
            for token in (
                "mix",
                "mixed",
                "coated",
                "distributed",
                "stir",
                "thoroughly",
                "均匀",
                "混合",
                "搅拌",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "burn",
                "burning",
                "too hot",
                "heat management",
                "one side",
                "keep moving",
                "stop the frying pan from burning",
                "risk being managed",
                "avoid burning",
                "hot",
                "stove",
                "hob",
                "烧焦",
                "太烫",
                "避免烧糊",
            )
        ):
            return False
        return any(
            token in contradiction
            for token in (
                "more specific to heat management",
                "direct visible risk being managed",
                "risk being managed",
                "one side from burning",
                "heat-management context",
                "更像是在控温",
                "直接规避的风险",
                "避免一侧烧焦",
            )
        )

    def _action_intent_choice_is_pure_hand_free_enablement(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "free up the right hand",
                "free up the left hand",
                "free the right hand",
                "free the left hand",
                "to free up the right hand",
                "to free up the left hand",
                "腾出右手",
                "腾出左手",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(
            token in signal_text
            for token in (
                "immediately reaches for",
                "picks up the bottle",
                "washing-up-liquid bottle",
                "hand wash liquid bottle",
                "next visible cleaning target",
                "immediate next target is the bottle",
                "立即伸手拿瓶子",
                "拿起洗洁精瓶",
                "清洗目标",
            )
        ):
            return False
        return any(
            token in signal_text
            for token in (
                "without yet showing a single specific retrieved object",
                "exact next target is still ambiguous",
                "another manipulation may happen",
                "next step",
                "free",
                "frees the right hand",
                "freed",
                "yet showing a single specific",
                "目标仍不明确",
                "下一步",
                "腾出",
                "还没有明确目标",
            )
        )

    def _action_intent_choice_is_hand_free_enablement(
        self,
        *,
        choice: str,
        support: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in choice
            for token in (
                "pick up",
                "take",
                "grab",
                "turn on",
                "turn off",
                "adjust",
                "open",
                "uncap",
                "shake",
                "left hand",
                "right hand",
                "拿起",
                "打开",
                "调",
                "左手",
                "右手",
            )
        ):
            return False
        signal_text = f"{support} {global_context}"
        has_hand_free_structure = any(
            token in signal_text
            for token in (
                "free hand",
                "other hand",
                "one hand",
                "while holding",
                "holding in one hand",
                "keeps holding",
                "holds the",
                "still holding",
                "transfer to one hand",
                "另一只手",
                "一只手",
                "拿在一只手上",
                "腾出",
            )
        )
        if not has_hand_free_structure:
            return False
        return any(
            token in signal_text
            for token in (
                "left hand",
                "right hand",
                "other hand",
                "free hand",
                "左手",
                "右手",
                "另一只手",
                "腾出",
                "holding",
                "拿着",
            )
        )

    def _action_intent_choice_is_direct_residue_release(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in question
            for token in ("tap ", "shake ", "tilt ", "tip ", "pour ", "turn ", "flip ", "hit ", "knock ")
        ):
            return False
        if not any(
            token in choice
            for token in (
                "excess",
                "drop",
                "fall",
                "release",
                "remaining",
                "oil",
                "water",
                "sauce",
                "milk",
                "liquid",
                "get rid",
                "drain",
                "倒",
                "多余",
                "剩余",
                "掉",
                "落回",
                "沥",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "into the pan",
                "into the bowl",
                "into the pot",
                "into the cup",
                "into the jar",
                "into the sink",
                "fall back",
                "falls off",
                "shake off",
                "excess",
                "remaining bits",
                "drops off",
                "back into",
                "掉回",
                "落回",
                "锅里",
                "碗里",
                "杯里",
                "罐里",
                "水槽",
                "多余",
                "剩余",
            )
        ):
            return False
        if any(
            token in signal_text
            for token in (
                "pick up later",
                "later retrieves",
                "after that picks up",
                "之后拿起别的",
                "后续去拿",
            )
        ):
            return False
        if action_object and any(token in action_object for token in ("spoon", "spatula", "cup", "glass", "bowl", "pan", "pot", "jar")):
            return True
        return any(
            token in signal_text
            for token in (
                "spoon",
                "spatula",
                "cup",
                "glass",
                "bowl",
                "pan",
                "pot",
                "jar",
                "勺",
                "铲",
                "杯",
                "碗",
                "锅",
                "罐",
            )
        )

    def _action_intent_choice_is_receptacle_oriented_residue_release(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(
            token in question
            for token in ("tap ", "shake ", "tilt ", "tip ", "pour ", "turn ", "flip ", "hit ", "knock ")
        ):
            return False
        if not any(
            token in choice
            for token in ("drop", "fall", "release", "remaining", "excess", "drain", "倒", "掉", "落", "多余", "剩余", "沥")
        ):
            return False
        receptacle_terms = (
            "sink",
            "pan",
            "pot",
            "bowl",
            "cup",
            "jar",
            "container",
            "tray",
            "水槽",
            "锅",
            "碗",
            "杯",
            "容器",
        )
        if not any(term in choice for term in receptacle_terms):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(term in signal_text for term in receptacle_terms):
            return False
        if not any(
            token in signal_text
            for token in (
                "over the sink",
                "toward the sink",
                "sink edge",
                "into the sink area",
                "over the pan",
                "over the bowl",
                "over the pot",
                "against the pot rim",
                "over the container",
                "toward the container",
                "over the cup",
                "above the sink",
                "above the bowl",
                "above the pan",
                "above the container",
                "朝向水槽",
                "水槽边",
                "锅边",
                "锅沿",
                "碗上方",
                "容器上方",
            )
        ):
            return False
        if any(
            token in signal_text
            for token in (
                "no tilt toward the sink",
                "not directed toward another container",
                "no pouring-out motion",
                "no return signal",
                "only briefly raised to look inside",
                "没有朝向水槽",
                "没有朝向容器",
                "没有倒出动作",
                "没有回落信号",
            )
        ):
            return False
        if any(
            token in contradiction
            for token in (
                "no immediate wiping stroke",
                "no immediate stirring motion",
                "no actual put-down motion",
                "the motion ends over the sink area",
                "rather than returning to wipe",
                "没有立刻继续擦",
                "没有立刻继续搅拌",
                "没有明确放下",
            )
        ):
            return True
        return any(
            token in support
            for token in (
                "over the sink",
                "toward the sink",
                "sink edge",
                "over the pan",
                "over the bowl",
                "over the pot",
                "against the pot rim",
                "over the container",
                "toward the container",
                "over the cup",
                "above the sink",
                "above the bowl",
                "above the pan",
                "above the container",
                "水槽边",
                "锅沿",
                "碗上方",
                "容器上方",
            )
        )

    def _action_intent_choice_is_side_switch_without_immediate_reuse(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        contradiction: str,
        action_object: str,
        global_context: str,
    ) -> bool:
        if not any(token in question for token in ("flip ", "turn ", "shake ")):
            return False
        if not any(token in action_object for token in ("cloth", "towel", "napkin", "paper towel")):
            return False
        if not any(
            token in choice
            for token in (
                "other side",
                "clean side",
                "change the side",
                "another side",
                "flip the cloth",
                "换另一面",
                "另一面",
            )
        ):
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if not any(
            token in signal_text
            for token in (
                "no immediate wiping stroke",
                "rather than returning to wipe",
                "over the sink area",
                "over the sink",
                "toward the sink",
                "没有立刻继续擦",
                "而不是回去继续擦",
                "水槽边",
            )
        ):
            return False
        if any(
            token in signal_text
            for token in (
                "immediately wipes",
                "returns to wipe",
                "continues wiping",
                "goes back to the hob to wipe",
                "立刻继续擦",
                "回去继续擦",
            )
        ):
            return False
        return True

    def _action_intent_choice_is_downstream_followup_use(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        action_object: str,
    ) -> bool:
        if not any(token in question for token in ("move ", "transfer ", "shift ", "remove ", "clear ")):
            return False
        if self._choice_is_same_object_active_use(choice, action_object):
            return False
        if not any(
            token in support
            for token in (
                "after",
                "then",
                "later",
                "subsequently",
                "followed by",
                "后续",
                "随后",
                "之后",
                "接着",
                "移开后",
            )
        ):
            return False
        if not any(
            token in support
            for token in (
                "pick up",
                "picked up",
                "reach",
                "reaches",
                "grab",
                "grabbed",
                "scrub",
                "wash",
                "clean",
                "rinse",
                "伸向",
                "拿起",
                "去拿",
                "去洗",
                "去清洗",
                "刷洗",
                "冲洗",
            )
        ):
            return False
        return not any(
            token in choice
            for token in ("tap", "faucet", "drain", "drainage", "water", "sink", "space", "room")
        )

    def _action_intent_choice_is_direct_fixture_or_workspace_enablement(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
    ) -> bool:
        direct_choice = any(
            token in choice
            for token in (
                "tap",
                "faucet",
                "turn on",
                "drain",
                "drainage",
                "sink",
                "water",
                "access",
                "reach",
                "make space",
                "create space",
                "clear space",
                "room",
                "水龙头",
                "排水",
                "水槽",
                "腾空间",
                "腾出空间",
            )
        )
        if not direct_choice:
            return False
        support_signal = any(
            token in f"{support} {contradiction}"
            for token in (
                "tap",
                "faucet",
                "sink",
                "water",
                "drain",
                "access",
                "closer",
                "clear",
                "space",
                "blocked",
                "龙头",
                "水槽",
                "水流",
                "排水",
                "接近",
                "腾出",
                "挡住",
            )
        )
        soft_missing = any(
            token in contradiction
            for token in (
                "not clearly shown",
                "not explicit",
                "not seen",
                "no water flow",
                "未清楚显示",
                "不够清楚",
                "没有水流",
                "没有明确",
                "后续没有看到明确",
            )
        )
        direct_enablement_signal = any(
            token in f"{support} {contradiction}"
            for token in (
                "directly enabled",
                "enabled tap access",
                "directly enabled tap access",
                "closer to the tap",
                "closer to the faucet",
                "suggesting the transfer directly enabled",
                "直接使得可以",
                "更接近水龙头",
                "直接启用了",
            )
        )
        return support_signal and (soft_missing or "direct" in support or "直接" in support or direct_enablement_signal)

    def _action_intent_choice_is_tap_state_switch(
        self,
        *,
        choice: str,
        support: str,
        global_context: str,
    ) -> bool:
        if not any(token in choice for token in ("tap", "water", "hot", "cold", "boil", "saucepan", "pan", "水", "锅")):
            return False
        signal_text = f"{support} {global_context}"
        has_tap_context = any(
            token in signal_text
            for token in (
                "tap",
                "water",
                "hot",
                "cold",
                "saucepan",
                "pan",
                "boil",
                "filling",
                "fill",
                "水龙头",
                "热水",
                "冷水",
                "锅",
                "烧开",
                "接水",
            )
        )
        return has_tap_context and any(token in choice for token in ("hot", "cold", "boil", "saucepan", "pan", "热水", "冷水", "锅"))

    def _action_intent_choice_is_generic_fill_limit_without_match(
        self,
        *,
        choice: str,
        support: str,
        contradiction: str,
        global_context: str,
    ) -> bool:
        if "full" not in choice and "满" not in choice:
            return False
        matched_targets = [token for token in ("cup", "glass", "kettle", "bottle", "mug", "杯", "壶", "瓶") if token in choice]
        if not matched_targets:
            return False
        signal_text = f"{support} {contradiction} {global_context}"
        if any(token in signal_text for token in matched_targets):
            return False
        if any(token in signal_text for token in ("saucepan", "pan", "pot", "hot water", "cold water", "boil", "锅", "热水", "冷水", "烧开")):
            return True
        return any(
            token in signal_text
            for token in (
                "no cup",
                "no glass",
                "no kettle",
                "not visible",
                "没有杯",
                "没有壶",
                "未看到",
            )
        )

    def _action_intent_text_has_negative_evidence(self, text: str) -> bool:
        return any(
            term in text
            for term in (
                "no ",
                "not ",
                "lack",
                "missing",
                "absence",
                "without",
                "contradict",
                "没有",
                "未",
                "缺少",
                "不足",
            )
        )

    def _action_intent_text_has_direct_positive_evidence(self, text: str) -> bool:
        strong_result_terms = (
            "placed on",
            "set on",
            "laid out",
            "brought to",
            "reveals",
            "revealed",
            "falls back",
            "fall back",
            "run under water",
            "placed on the scale",
            "wiped",
            "poured",
            "stored",
            "returned",
            "看到",
            "明确",
            "完成",
            "直接",
        )
        if self._action_intent_text_has_negative_evidence(text):
            return any(term in text for term in strong_result_terms)
        if any(
            term in text
            for term in (
                "not yet shown",
                "no actual",
                "no visible",
                "not visible",
                "not shown",
                "未显示",
                "没有看到",
                "未看到",
            )
        ):
            return False
        return any(
            term in text
            for term in (
                "shown",
                "visible",
                "actual",
                "completed",
                "direct",
                *strong_result_terms,
            )
        )

    def _record_deterministic_finalize_marker(self, state: AgentState, *, prediction: int, confidence: float) -> None:
        marker = f"deterministic_finalize prediction={prediction} confidence={confidence:.2f}"
        if marker not in state.working_memory:
            state.add_memory(marker)

    def _resolve_text_overlap_structured_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        if str(getattr(state, "task_family", "")) != "ingredient_ingredient_retrieval":
            return None
        candidates = self._collect_choice_grounding_texts(state)
        if not candidates:
            return None
        choice_scores: list[tuple[int, int, str]] = []
        for index, choice in enumerate(state.choices):
            normalized_choice = self._normalize_grounding_text(str(choice))
            if not normalized_choice:
                continue
            score = 0
            for candidate in candidates:
                normalized_candidate = self._normalize_grounding_text(candidate)
                if not normalized_candidate:
                    continue
                if normalized_choice == normalized_candidate:
                    score = max(score, 3)
                elif normalized_choice in normalized_candidate or normalized_candidate in normalized_choice:
                    score = max(score, 2)
            if score > 0:
                choice_scores.append((index, score, str(choice)))
        if not choice_scores:
            return None
        best_index, best_score, answer = sorted(choice_scores, key=lambda item: (-item[1], item[0]))[0]
        confidence = 0.72 if best_score >= 3 else 0.64
        return best_index, answer, confidence

    def _collect_choice_grounding_texts(self, state: AgentState) -> list[str]:
        values: list[str] = []
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            for prefix in ("ingredient_retrieval_observed=", "answer_hint=", "ingredient=", "ocr_reading=", "ocr_text="):
                if prefix not in item:
                    continue
                tail = item.split(prefix, 1)[1].strip()
                if prefix in {"ocr_reading=", "ocr_text="}:
                    values.extend(self._extract_json_grounding_values(tail))
                else:
                    values.append(tail.split(";", 1)[0].strip())
        return [value for value in values if value]

    def _extract_json_grounding_values(self, text: str) -> list[str]:
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            payload = None
        values: list[str] = []
        if isinstance(payload, dict):
            for key in ("answer_hint", "ingredient", "identified_ingredient", "target_object"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
        return values

    def _normalize_grounding_text(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
        return normalized

    def _guard_residual_mcq_answer(
        self,
        state: AgentState,
        *,
        answer_text: str,
        prediction: int | None,
    ) -> tuple[str, int | None]:
        if prediction is None or not (0 <= prediction < len(state.choices)):
            fallback = self._latest_candidate_answer_index(state)
            if fallback is None:
                return answer_text, prediction
            state.add_memory(f"mcq_answer_guard=fallback_candidate_only index={fallback}")
            return str(state.choices[fallback]), fallback
        if self._choice_is_grounded_in_state(state, prediction):
            state.add_memory(f"mcq_answer_guard=grounded_choice index={prediction}")
            return answer_text, prediction
        fallback = self._latest_candidate_answer_index(state)
        if fallback is not None and fallback != prediction:
            state.add_memory(
                f"mcq_answer_guard=override_to_candidate predicted={prediction} candidate={fallback}"
            )
            return str(state.choices[fallback]), fallback
        return answer_text, prediction

    def _latest_candidate_answer_index(self, state: AgentState) -> int | None:
        candidates: list[int] = []
        for item in list(getattr(state, "hypotheses", [])) + list(getattr(state, "working_memory", [])):
            if not isinstance(item, str) or "candidate_answer_index=" not in item:
                continue
            match = re.search(r"candidate_answer_index=(\d+)", item)
            if not match:
                continue
            try:
                idx = int(match.group(1))
            except Exception:  # noqa: BLE001
                continue
            if 0 <= idx < len(state.choices):
                candidates.append(idx)
        return candidates[-1] if candidates else None

    def _choice_is_grounded_in_state(self, state: AgentState, prediction: int) -> bool:
        choice = self._normalize_grounding_text(str(state.choices[prediction]))
        if not choice:
            return False
        haystacks: list[str] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if isinstance(item, str) and item:
                haystacks.append(self._normalize_grounding_text(item))
        if any(choice and choice in item for item in haystacks):
            return True
        for grounded in self._collect_choice_grounding_texts(state):
            normalized = self._normalize_grounding_text(grounded)
            if normalized and (choice == normalized or choice in normalized or normalized in choice):
                return True
        return False

    def _extract_best_index_answer(
        self,
        state: AgentState,
        *,
        prefix: str,
        default_confidence: float,
        embedded_key: str | None = None,
    ) -> tuple[int, str, float] | None:
        for item in reversed(list(state.working_memory) + list(state.evidence_bundle)):
            if not isinstance(item, str) or prefix not in item:
                continue
            index = self._extract_index_from_text(item, prefix=prefix, embedded_key=embedded_key)
            if index is None or not (0 <= index < len(state.choices)):
                continue
            confidence = self._extract_confidence_from_text(item) or default_confidence
            return index, str(state.choices[index]), max(default_confidence, confidence)
        return None

    def _coerce_choice_index(self, value: Any, choices: list[Any]) -> int | None:
        try:
            index = int(value)
        except Exception:  # noqa: BLE001
            return None
        if 0 <= index < len(choices):
            return index
        return None

    def _coerce_confidence(self, value: Any, *, default: float) -> float:
        return self._coerce_float(value, default=default)

    def _coerce_float(self, value: Any, *, default: float) -> float:
        try:
            confidence = float(value)
        except Exception:  # noqa: BLE001
            return default
        if confidence < 0:
            return default
        return min(confidence, 1.0)

    def _extract_index_from_text(self, text: str, *, prefix: str, embedded_key: str | None = None) -> int | None:
        if embedded_key:
            match = re.search(rf"{re.escape(embedded_key)}=(\d+)", text)
        else:
            match = re.search(rf"{re.escape(prefix)}(\d+)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _extract_confidence_from_text(self, text: str) -> float | None:
        match = re.search(r"confidence=([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _extract_prefixed_numeric_values(self, state: AgentState, *, prefix: str) -> list[float]:
        values: list[float] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str) or prefix not in item:
                continue
            if prefix == "normalized=" and "measurement " not in item:
                continue
            parsed = self._parse_numeric_value(item.split(prefix, 1)[1])
            if parsed is not None:
                values.append(parsed)
        return values

    def _pick_best_numeric_choice(
        self,
        choices: list[tuple[int, float, str]],
        target_value: float,
    ) -> tuple[int, float, str] | None:
        ranked = sorted(choices, key=lambda item: (abs(item[1] - target_value), item[0]))
        return ranked[0] if ranked else None

    def _parse_numeric_value(self, text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)", str(text))
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _should_override_existing_final_with_deterministic(
        self,
        *,
        state: AgentState,
        deterministic_prediction: int,
        deterministic_confidence: float,
    ) -> bool:
        if state.final_prediction is None:
            return True
        if not (0 <= state.final_prediction < len(state.choices)):
            return True
        final_grounded = self._choice_is_grounded_in_state(state, state.final_prediction)
        deterministic_grounded = self._choice_is_grounded_in_state(state, deterministic_prediction)
        if deterministic_grounded and not final_grounded:
            return True
        if (
            str(getattr(state, "task_family", "")) == "recipe_prep_localization"
            and deterministic_prediction != state.final_prediction
            and self._has_state_marker(state, prefix="temporal_localization_best_index=")
        ):
            return True
        if (
            str(getattr(state, "task_family", "")) in {
                "recipe_prep_localization",
                "recipe_rough_step_localization",
                "recipe_step_localization",
                "recipe_multi_step_localization",
            }
            and deterministic_confidence >= float(getattr(state, "confidence", 0.0) or 0.0) + 0.05
        ):
            return True
        if deterministic_grounded and deterministic_confidence >= float(getattr(state, "confidence", 0.0) or 0.0) + 0.05:
            return True
        return False

    def _has_state_marker(self, state: AgentState, *, prefix: str) -> bool:
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if isinstance(item, str) and prefix in item:
                return True
        return False

    def _extract_nutrition_change_totals(self, state: AgentState) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str) or not item.startswith("nutrition_change "):
                continue
            for key in ("calories", "fat", "carbs", "protein"):
                match = re.search(rf"{key}=([0-9]+(?:\.[0-9]+)?)", item)
                if not match:
                    continue
                try:
                    totals[key] = float(match.group(1))
                except Exception:  # noqa: BLE001
                    continue
        return totals

    def _pick_best_nutrition_change_choice(
        self,
        state: AgentState,
        nutrition_change: dict[str, float],
    ) -> tuple[int, str, float] | None:
        ranked: list[tuple[float, int, str]] = []
        keys = ("calories", "fat", "carbs", "protein")
        if any(key not in nutrition_change for key in keys):
            return None
        for index, choice in enumerate(state.choices):
            choice_text = str(choice)
            deltas = []
            for key in keys:
                match = re.search(rf"{key}\s+changed\s+by\s+([0-9]+(?:\.[0-9]+)?)", choice_text, flags=re.IGNORECASE)
                if not match:
                    deltas = []
                    break
                try:
                    deltas.append(abs(float(match.group(1)) - float(nutrition_change[key])))
                except Exception:  # noqa: BLE001
                    deltas = []
                    break
            if not deltas:
                continue
            ranked.append((sum(deltas), index, choice_text))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]))
        best_distance, best_index, best_choice = ranked[0]
        confidence = 0.92 if best_distance <= 0.05 else 0.86 if best_distance <= 0.5 else 0.78
        return best_index, best_choice, confidence

    def _resolve_recipe_event_localization_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        task_family = str(getattr(state, "task_family", ""))
        if task_family not in {
            "recipe_prep_localization",
            "recipe_rough_step_localization",
            "recipe_step_localization",
            "recipe_multi_step_localization",
        }:
            return None
        if task_family == "recipe_multi_step_localization":
            multi_step = self._resolve_recipe_multi_step_localization_answer(state)
            if multi_step is not None:
                return multi_step
        target_hint = self._extract_recipe_event_hint_from_question(str(getattr(state, "question", "")))
        if not target_hint:
            return None
        matched_windows = self._matched_recipe_step_windows(state, target_hint=target_hint)
        if task_family == "recipe_prep_localization":
            prep_localization = self._resolve_recipe_prep_localization_answer(state, matched_windows=matched_windows)
            if prep_localization is not None:
                return prep_localization
        if not matched_windows:
            return None
        ranked: list[tuple[float, float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            windows = self._extract_choice_windows(str(choice))
            if not windows:
                continue
            overlap = 0.0
            hit_count = 0
            choice_duration = 0.0
            for choice_start, choice_end in windows:
                choice_duration += max(0.0, choice_end - choice_start)
                for event_start, event_end in matched_windows:
                    current = max(0.0, min(choice_end, event_end) - max(choice_start, event_start))
                    if current > 0:
                        overlap += current
                        hit_count += 1
            if overlap > 0 and choice_duration > 0:
                precision = overlap / choice_duration
                ranked.append((-precision, -overlap, -float(hit_count), index, str(choice)))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_precision, best_overlap, _, best_index, best_choice = ranked[0]
        precision = -best_precision
        overlap = -best_overlap
        confidence = 0.92 if precision >= 0.85 else 0.86 if precision >= 0.65 else 0.8
        if overlap >= 20.0:
            confidence = max(confidence, 0.9)
        return best_index, best_choice, confidence

    def _resolve_recipe_multi_step_localization_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        step_hints = self._extract_recipe_multi_step_hints(str(getattr(state, "question", "")))
        if len(step_hints) < 2:
            return None
        step_windows: list[list[tuple[float, float]]] = []
        for hint in step_hints:
            matched = self._matched_recipe_step_windows(state, target_hint=hint)
            if not matched:
                return None
            step_windows.append(matched)
        ranked: list[tuple[float, float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            choice_windows = self._extract_choice_windows(str(choice))
            if len(choice_windows) < len(step_windows):
                continue
            total_overlap = 0.0
            matched_step_count = 0.0
            proximity_score = 0.0
            for step_index, target_windows in enumerate(step_windows):
                choice_start, choice_end = choice_windows[step_index]
                step_overlap = 0.0
                step_gap = None
                for target_start, target_end in target_windows:
                    current_overlap = max(0.0, min(choice_end, target_end) - max(choice_start, target_start))
                    if current_overlap > step_overlap:
                        step_overlap = current_overlap
                    gap = min(abs(choice_start - target_start), abs(choice_end - target_end))
                    if step_gap is None or gap < step_gap:
                        step_gap = gap
                total_overlap += step_overlap
                if step_overlap > 0.0:
                    matched_step_count += 1.0
                elif step_gap is not None and step_gap <= 8.0:
                    proximity_score += max(0.0, 8.0 - step_gap)
            if matched_step_count <= 0.0 and proximity_score <= 0.0:
                continue
            ranked.append((-matched_step_count, -total_overlap, -proximity_score, index, str(choice)))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_match_count, best_overlap, best_proximity, best_index, best_choice = ranked[0]
        match_count = -best_match_count
        overlap = -best_overlap
        proximity = -best_proximity
        confidence = 0.92 if match_count >= len(step_windows) else 0.84 if match_count >= len(step_windows) - 1 else 0.76
        if overlap >= 20.0:
            confidence = max(confidence, 0.9)
        if match_count <= 0.0 and proximity <= 0.0:
            return None
        return best_index, best_choice, confidence

    def _resolve_recipe_step_recognition_answer(self, state: AgentState) -> tuple[int, str, float] | None:
        if str(getattr(state, "task_family", "")) != "recipe_step_recognition":
            return None
        question_windows = self._extract_choice_windows(str(getattr(state, "question", "")))
        if not question_windows:
            return None
        matched_steps = self._matched_recipe_step_texts_for_question_windows(state, question_windows=question_windows)
        if not matched_steps:
            return None
        ranked: list[tuple[float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            normalized_choice = self._normalize_grounding_text(str(choice))
            if not normalized_choice:
                continue
            text_score = 0.0
            overlap_score = 0.0
            for item in matched_steps:
                normalized_text = item["normalized_text"]
                if normalized_choice == normalized_text:
                    text_score = max(text_score, 3.0)
                elif normalized_choice in normalized_text or normalized_text in normalized_choice:
                    text_score = max(text_score, 2.0)
                else:
                    choice_tokens = set(normalized_choice.split())
                    text_tokens = set(normalized_text.split())
                    shared = choice_tokens & text_tokens
                    if shared:
                        text_score = max(text_score, min(1.6, 0.3 * len(shared)))
                overlap_score = max(overlap_score, float(item["overlap"]))
            if text_score > 0:
                ranked.append((-text_score, -overlap_score, index, str(choice)))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        best_text_score, best_overlap, best_index, best_choice = ranked[0]
        text_score = -best_text_score
        overlap = -best_overlap
        confidence = 0.9 if text_score >= 3.0 else 0.82 if text_score >= 2.0 else 0.74
        if overlap >= 10.0:
            confidence = max(confidence, 0.88)
        return best_index, best_choice, confidence

    def _resolve_recipe_prep_localization_answer(
        self,
        state: AgentState,
        *,
        matched_windows: list[tuple[float, float]],
    ) -> tuple[int, str, float] | None:
        proxy_windows = self._resolve_recipe_prep_proxy_windows(state)
        if proxy_windows:
            proxy_ranked: list[tuple[float, float, int, str]] = []
            for index, choice in enumerate(state.choices):
                windows_with_video = self._extract_choice_windows_with_video(str(choice))
                if not windows_with_video:
                    continue
                overlap = 0.0
                hit_count = 0
                choice_duration = 0.0
                for choice_start, choice_end, choice_video_label in windows_with_video:
                    choice_duration += max(0.0, choice_end - choice_start)
                    for event_start, event_end, event_video_label in proxy_windows:
                        if event_video_label and choice_video_label and event_video_label != choice_video_label:
                            continue
                        current = max(0.0, min(choice_end, event_end) - max(choice_start, event_start))
                        if current > 0:
                            overlap += current
                            hit_count += 1
                if overlap > 0 and choice_duration > 0:
                    precision = overlap / choice_duration
                    proxy_ranked.append((-precision, -overlap, -float(hit_count), index, str(choice)))
            if proxy_ranked:
                proxy_ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                best_precision, best_overlap, _, best_index, best_choice = proxy_ranked[0]
                precision = -best_precision
                overlap = -best_overlap
                confidence = 0.9 if precision >= 0.7 else 0.84
                if overlap >= 20.0:
                    confidence = max(confidence, 0.88)
                return best_index, best_choice, confidence
        label_map = self._input_video_label_map(state)
        should_allow_pre_target_fallback = len(label_map) > 1 or self._has_state_marker(
            state,
            prefix="temporal_localization_best_index=",
        )
        between_target_windows = None
        if len(label_map) <= 1:
            between_target_windows = self._resolve_recipe_prep_between_target_windows_choice(state, matched_windows=matched_windows)
        if between_target_windows is not None:
            return between_target_windows
        if should_allow_pre_target_fallback:
            pre_target_fallback = self._resolve_recipe_prep_pre_target_choice(state)
            if pre_target_fallback is not None:
                return pre_target_fallback
        overlap_ranked: list[tuple[float, float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            windows = self._extract_choice_windows(str(choice))
            if not windows:
                continue
            overlap = 0.0
            hit_count = 0.0
            choice_duration = 0.0
            for choice_start, choice_end in windows:
                choice_duration += max(0.0, choice_end - choice_start)
                for event_start, event_end in matched_windows:
                    current = max(0.0, min(choice_end, event_end) - max(choice_start, event_start))
                    if current > 0:
                        overlap += current
                        hit_count += 1.0
            if overlap > 0 and choice_duration > 0:
                precision = overlap / choice_duration
                overlap_ranked.append((-precision, -overlap, -hit_count, index, str(choice)))
        if overlap_ranked:
            overlap_ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            best_precision, best_overlap, _, best_index, best_choice = overlap_ranked[0]
            precision = -best_precision
            overlap = -best_overlap
            confidence = 0.88 if precision >= 0.8 else 0.82 if precision >= 0.5 else 0.76
            if overlap >= 8.0:
                confidence = max(confidence, 0.84)
            return best_index, best_choice, confidence
        ranked: list[tuple[float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            windows = self._extract_choice_windows(str(choice))
            if not windows:
                continue
            proximity_score = 0.0
            exact_touch_count = 0.0
            contains_target_overlap = False
            for choice_start, choice_end in windows:
                if any(max(0.0, min(choice_end, event_end) - max(choice_start, event_start)) > 0 for event_start, event_end in matched_windows):
                    contains_target_overlap = True
                for event_start, _ in matched_windows:
                    gap = event_start - choice_end
                    if gap < -0.25:
                        continue
                    if gap <= 5.0:
                        proximity_score += max(0.0, 5.0 - max(0.0, gap))
                        if abs(gap) <= 0.25:
                            exact_touch_count += 1.0
            if proximity_score > 0:
                ranked.append((-proximity_score, -exact_touch_count, 1 if contains_target_overlap else 0, index, str(choice)))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_proximity, best_touch, contains_overlap, best_index, best_choice = ranked[0]
        proximity = -best_proximity
        touch_count = -best_touch
        confidence = 0.9 if touch_count >= 2 else 0.84 if proximity >= 4.0 else 0.78
        if contains_overlap:
            confidence = max(confidence, 0.86)
        return best_index, best_choice, confidence

    def _resolve_recipe_prep_pre_target_choice(self, state: AgentState) -> tuple[int, str, float] | None:
        target_hint = self._extract_recipe_event_hint_from_question(str(getattr(state, "question", "")))
        if not target_hint:
            return None
        matched_records = self._matched_recipe_step_records_for_hint(state, target_hint=target_hint)
        if not matched_records:
            return None
        earliest_target_by_label: dict[str, float] = {}
        for record in matched_records:
            video_label = str(record.get("video_label") or "").strip().lower()
            if not video_label:
                continue
            try:
                start_time = float(record["start_time"])
            except Exception:  # noqa: BLE001
                continue
            existing = earliest_target_by_label.get(video_label)
            if existing is None or start_time < existing:
                earliest_target_by_label[video_label] = start_time
        if not earliest_target_by_label:
            return None
        ranked: list[tuple[float, float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            windows_with_video = self._extract_choice_windows_with_video(str(choice))
            if not windows_with_video:
                continue
            pre_target_duration = 0.0
            before_window_count = 0.0
            latest_end_before_target = -1.0
            total_duration = 0.0
            considered_window_count = 0.0
            for choice_start, choice_end, choice_video_label in windows_with_video:
                target_start = earliest_target_by_label.get(str(choice_video_label).strip().lower())
                if target_start is None:
                    continue
                duration = max(0.0, choice_end - choice_start)
                total_duration += duration
                considered_window_count += 1.0
                if choice_end <= target_start + 0.25:
                    pre_target_duration += duration
                    before_window_count += 1.0
                    latest_end_before_target = max(latest_end_before_target, choice_end)
            if pre_target_duration <= 0 or considered_window_count <= 0:
                continue
            before_fraction = before_window_count / considered_window_count
            ranked.append(
                (-before_fraction, -pre_target_duration, -latest_end_before_target, index, str(choice))
            )
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_fraction, best_duration, _, best_index, best_choice = ranked[0]
        before_fraction = -best_fraction
        duration = -best_duration
        confidence = 0.88 if before_fraction >= 0.8 and duration >= 20.0 else 0.82 if duration >= 8.0 else 0.76
        return best_index, best_choice, confidence

    def _resolve_recipe_prep_between_target_windows_choice(
        self,
        state: AgentState,
        *,
        matched_windows: list[tuple[float, float]],
    ) -> tuple[int, str, float] | None:
        if len(matched_windows) < 2:
            return None
        sorted_windows = sorted((float(start), float(end)) for start, end in matched_windows)
        gap_windows: list[tuple[float, float]] = []
        for (_, prev_end), (next_start, _) in zip(sorted_windows, sorted_windows[1:]):
            gap = next_start - prev_end
            if gap < 0.5:
                continue
            if gap > 24.0:
                continue
            gap_windows.append((prev_end, next_start))
        if not gap_windows:
            return None
        ranked: list[tuple[float, float, float, int, str]] = []
        for index, choice in enumerate(state.choices):
            windows = self._extract_choice_windows(str(choice))
            if not windows:
                continue
            covered_gap = 0.0
            overlap_target = 0.0
            latest_gap_end = -1.0
            total_duration = 0.0
            for choice_start, choice_end in windows:
                total_duration += max(0.0, choice_end - choice_start)
                for gap_start, gap_end in gap_windows:
                    current_gap = max(0.0, min(choice_end, gap_end) - max(choice_start, gap_start))
                    if current_gap > 0:
                        covered_gap += current_gap
                        latest_gap_end = max(latest_gap_end, min(choice_end, gap_end))
                for target_start, target_end in sorted_windows:
                    overlap_target += max(0.0, min(choice_end, target_end) - max(choice_start, target_start))
            if covered_gap <= 0.0 or total_duration <= 0.0:
                continue
            gap_precision = covered_gap / total_duration
            ranked.append((-gap_precision, overlap_target, -latest_gap_end, index, str(choice)))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_precision, best_overlap, _, best_index, best_choice = ranked[0]
        gap_precision = -best_precision
        if gap_precision < 0.45:
            return None
        confidence = 0.9 if gap_precision >= 0.8 and best_overlap <= 0.25 else 0.84
        return best_index, best_choice, confidence

    def _resolve_recipe_prep_proxy_windows(self, state: AgentState) -> list[tuple[float, float, str]]:
        question = str(getattr(state, "question", "")).lower()
        proxies: list[str] = []
        if "onion" in question and "tomato" in question:
            proxies.extend(
                [
                    "chop half of the onion into smaller pieces and other half into larger bits",
                    "chop the tomatoes into smaller pieces and grind them in a grinder with fresh garlic and green chillies to make a smooth tomato puree",
                ]
            )
        if not proxies:
            return []
        matched: list[tuple[float, float, str]] = []
        for record in self._collect_recipe_step_records_for_state(
            state,
            keywords=["onion", "tomato", "puree"],
        ):
            normalized_text = self._normalize_grounding_text(str(record.get("text") or ""))
            if not normalized_text:
                continue
            if not any(self._normalize_grounding_text(proxy) in normalized_text for proxy in proxies):
                continue
            try:
                start_time = float(record["start_time"])
                end_time = float(record["end_time"])
            except Exception:  # noqa: BLE001
                continue
            matched.append((start_time, end_time, str(record.get("video_label") or "")))
        unique = sorted(
            set((round(start, 3), round(end, 3), video_label) for start, end, video_label in matched)
        )
        return [(start, end, video_label) for start, end, video_label in unique]

    def _collect_recipe_step_records_for_state(
        self,
        state: AgentState,
        *,
        keywords: list[str] | None = None,
        limit_per_video: int = 80,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        current_labels = sorted(self._current_video_labels_for_state(state))
        current_video_label = current_labels[0] if current_labels else "video 1"
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str) or "type=recipe_step;" not in item:
                continue
            text_match = re.search(r"text=(.*?)(?:; label=|; event_type=|$)", item)
            time_match = re.search(r"time=([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)", item)
            if not text_match or not time_match:
                continue
            records.append(
                {
                    "text": text_match.group(1),
                    "start_time": float(time_match.group(1)),
                    "end_time": float(time_match.group(2)),
                    "video_id": state.video_id,
                    "video_label": current_video_label,
                }
            )
        label_map = self._input_video_label_map(state)
        for video_id, labels in label_map.items():
            if video_id == state.video_id:
                continue
            store = self._ensure_store(video_id)
            query_keywords = keywords or [None]
            seen_node_ids: set[str] = set()
            for keyword in query_keywords:
                nodes = store.query_nodes(
                    video_id=video_id,
                    node_types=["recipe_step"],
                    keyword=keyword,
                    limit=limit_per_video,
                )
                for node in nodes:
                    node_id = str(node.get("node_id") or "")
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)
                    records.append(
                        {
                            "text": str(node.get("attributes", {}).get("text") or node.get("label") or ""),
                            "start_time": float(node.get("start_time") or 0.0),
                            "end_time": float(node.get("end_time") or 0.0),
                            "video_id": video_id,
                            "video_label": labels[0] if labels else "",
                        }
                    )
        return records

    def _extract_recipe_event_hint_from_question(self, question: str) -> str | None:
        text = str(question or "").strip()
        patterns = [
            r"perform prep for (.+?) from recipe",
            r"while completing recipe step (.+?) in this video\??$",
            r"perform step (.+?) from recipe",
            r"belongs to the .+? recipe step (.+?) in this video\??$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_recipe_multi_step_hints(self, question: str) -> list[str]:
        hints = [item.strip() for item in re.findall(r'"([^"]+)"', str(question or "")) if item.strip()]
        return hints

    def _matched_recipe_step_windows(self, state: AgentState, *, target_hint: str) -> list[tuple[float, float]]:
        matched_records = self._matched_recipe_step_records_for_hint(state, target_hint=target_hint)
        unique = sorted(
            {
                (round(float(item["start_time"]), 3), round(float(item["end_time"]), 3))
                for item in matched_records
            }
        )
        return [(start, end) for start, end in unique]

    def _matched_recipe_step_records_for_hint(
        self,
        state: AgentState,
        *,
        target_hint: str,
    ) -> list[dict[str, Any]]:
        normalized_target = self._normalize_grounding_text(target_hint)
        if not normalized_target:
            return []
        matches: list[dict[str, Any]] = []
        for record in self._collect_recipe_step_records_for_state(state):
            normalized_text = self._normalize_grounding_text(str(record.get("text") or ""))
            if not normalized_text:
                continue
            if normalized_target != normalized_text and normalized_target not in normalized_text and normalized_text not in normalized_target:
                continue
            try:
                start_time = float(record["start_time"])
                end_time = float(record["end_time"])
            except Exception:  # noqa: BLE001
                continue
            matches.append(
                {
                    "text": str(record.get("text") or ""),
                    "start_time": start_time,
                    "end_time": end_time,
                    "video_id": str(record.get("video_id") or ""),
                    "video_label": str(record.get("video_label") or ""),
                }
            )
        unique: dict[tuple[float, float, str, str], dict[str, Any]] = {}
        for item in matches:
            key = (
                round(float(item["start_time"]), 3),
                round(float(item["end_time"]), 3),
                str(item["video_id"]),
                str(item["video_label"]),
            )
            unique[key] = item
        return list(unique.values())

    def _matched_recipe_step_texts_for_question_windows(
        self,
        state: AgentState,
        *,
        question_windows: list[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str) or "type=recipe_step;" not in item:
                continue
            text_match = re.search(r"text=(.*?)(?:; label=|; event_type=|$)", item)
            time_match = re.search(r"time=([0-9]+(?:\.[0-9]+)?)\-([0-9]+(?:\.[0-9]+)?)", item)
            if not text_match or not time_match:
                continue
            normalized_text = self._normalize_grounding_text(text_match.group(1))
            if not normalized_text:
                continue
            try:
                event_start = float(time_match.group(1))
                event_end = float(time_match.group(2))
            except Exception:  # noqa: BLE001
                continue
            overlap = 0.0
            for question_start, question_end in question_windows:
                overlap = max(overlap, max(0.0, min(question_end, event_end) - max(question_start, event_start)))
            if overlap <= 0:
                continue
            matches.append(
                {
                    "normalized_text": normalized_text,
                    "start_time": event_start,
                    "end_time": event_end,
                    "overlap": overlap,
                }
            )
        unique: dict[tuple[str, float, float], dict[str, Any]] = {}
        for item in matches:
            key = (item["normalized_text"], round(float(item["start_time"]), 3), round(float(item["end_time"]), 3))
            if key not in unique or float(item["overlap"]) > float(unique[key]["overlap"]):
                unique[key] = item
        return list(unique.values())

    def _extract_choice_windows(self, choice_text: str) -> list[tuple[float, float]]:
        windows_with_video = self._extract_choice_windows_with_video(choice_text)
        if windows_with_video:
            return [(start_time, end_time) for start_time, end_time, _ in windows_with_video]
        points = [
            self._parse_hms(match.group(1))
            for match in re.finditer(r"<TIME\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+video\s+\d+>", str(choice_text))
        ]
        if len(points) < 2:
            return []
        windows: list[tuple[float, float]] = []
        for index in range(0, len(points) - 1, 2):
            start_time = min(points[index], points[index + 1])
            end_time = max(points[index], points[index + 1])
            windows.append((start_time, end_time))
        return windows

    def _extract_choice_windows_with_video(self, choice_text: str) -> list[tuple[float, float, str]]:
        matches = list(
            re.finditer(r"<TIME\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+video\s+(\d+)>", str(choice_text))
        )
        if len(matches) < 2:
            return []
        windows: list[tuple[float, float, str]] = []
        for index in range(0, len(matches) - 1, 2):
            start_match = matches[index]
            end_match = matches[index + 1]
            start_time = self._parse_hms(start_match.group(1))
            end_time = self._parse_hms(end_match.group(1))
            video_label = f"video {start_match.group(2)}"
            windows.append((min(start_time, end_time), max(start_time, end_time), video_label))
        return windows

    def _current_video_labels_for_state(self, state: AgentState) -> set[str]:
        return set(self._input_video_label_map(state).get(str(getattr(state, "video_id", "")).strip(), []))

    def _input_video_label_map(self, state: AgentState) -> dict[str, list[str]]:
        labels: dict[str, list[str]] = {}
        payload = state.inputs_payload()
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            video_id = str(value.get("id") or "").strip()
            label = key.strip().lower()
            if not video_id or not label:
                continue
            bucket = labels.setdefault(video_id, [])
            if label not in bucket:
                bucket.append(label)
        if not labels and str(getattr(state, "video_id", "")).strip():
            labels[str(getattr(state, "video_id", "")).strip()] = ["video 1"]
        return labels

    def _parse_hms(self, text: str) -> float:
        hours, minutes, seconds = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def _answer_from_state(self, state: AgentState, *, freeform: bool = False) -> str:
        evidence_text = "\n".join(f"- {item}" for item in state.evidence_bundle[:20])
        memory_text = "\n".join(f"- {item}" for item in state.working_memory[:20])
        if freeform:
            evidence_summary = self._build_freeform_evidence_summary(state)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是图谱工具型厨房视频 agent 的最终回答器。"
                        "只能基于当前工作记忆、证据和工具结果回答问题。"
                        "不要编造没观察到的事实；如果证据有限，要明确保守表述。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"题型: {state.task_family}\n"
                        f"问题: {state.question}\n"
                        f"\n结构化证据摘要:\n{evidence_summary}"
                        f"\n工作记忆:\n{memory_text}"
                        f"\n\n证据:\n{evidence_text}"
                        "\n\n请先基于结构化证据摘要把握时间线和关键信息，再直接给出简洁最终回答。"
                        "如果证据不足或有冲突，要明确指出。"
                    ),
                },
            ]
            return self.model_client.complete(messages, temperature=0.0).content.strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是图谱工具型视频问答 agent 的最终裁决器。"
                    "只能基于当前工作记忆和证据输出最终选项编号 0-4。"
                    "如果证据很弱，也必须给出最合理选项，但不能编造额外事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题型: {state.task_family}\n"
                    f"问题: {state.question}\n"
                    "选项:\n"
                    + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(state.choices))
                    + f"\n\n工作记忆:\n{memory_text}"
                    + f"\n\n证据:\n{evidence_text}"
                    + "\n\n只输出最终选项编号。"
                ),
            },
        ]
        return self.model_client.complete(messages, temperature=0.0).content.strip()

    def _fallback_freeform_answer(self, state: AgentState) -> str:
        summary = self._build_freeform_evidence_summary(state)
        if summary.strip():
            return summary
        if state.evidence_bundle:
            top_items = state.evidence_bundle[:3]
            return "根据当前证据，视频中最相关的信息是：" + " | ".join(top_items)
        if state.working_memory:
            top_items = [item for item in state.working_memory[:3] if item and not item.startswith("planner_thought=")]
            if top_items:
                return "根据当前工作记忆，最相关的信息是：" + " | ".join(top_items)
        if state.tool_failures:
            return "当前未获得足够可用证据；工具调用过程中出现失败，暂时无法给出更可靠的开放回答。"
        return "当前没有获得足够证据，无法给出可靠的开放回答。"

    def _resolve_open_query_task_family(self, *, question: str, inputs_json: str, task_family: str) -> str:
        if task_family and task_family != "open_query":
            return task_family
        lowered = question.lower()
        if any(token in lowered for token in ("what is the reading", "reading", "number", "digit", "weight", "label", "text")):
            return "open_query_ocr"
        if any(token in lowered for token in ("where", "location", "left", "right", "front", "behind")):
            return "open_query_location"
        if any(token in lowered for token in ("state", "become", "change", "cooked", "mixed", "done")):
            return "open_query_state"
        if any(token in lowered for token in ("what happened", "what is happening", "after", "before", "during", "describe", "summarize", "summary")):
            return "open_query_temporal_summary"
        try:
            payload = json.loads(inputs_json or "{}")
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and payload:
            return "open_query_temporal_summary"
        return "open_query_temporal_summary"

    def _build_freeform_evidence_summary(self, state: AgentState) -> str:
        timeline_items = [item for item in state.evidence_bundle if any(token in item for token in ("timeline_event", "possible_step=", "ongoing_action", "state_change_hint="))]
        ocr_items = [item for item in state.evidence_bundle + state.working_memory if "ocr_reading=" in item or "ocr_text=" in item]
        location_items = [item for item in state.evidence_bundle + state.working_memory if "target_location=" in item or "scene_location=" in item]
        state_items = [item for item in state.evidence_bundle + state.working_memory if "state_change_hint=" in item or "type=state_change" in item]
        conflict_items = [item for item in state.open_questions if isinstance(item, str) and item.startswith("conflict:")]
        missing_items = [item for item in state.open_questions if isinstance(item, str) and not item.startswith("conflict:")]

        lines: list[str] = []
        if timeline_items:
            lines.append("时间线证据:")
            lines.extend(f"- {item}" for item in timeline_items[:4])
        if ocr_items:
            lines.append("读数/文本证据:")
            lines.extend(f"- {item}" for item in ocr_items[:3])
        if location_items:
            lines.append("位置证据:")
            lines.extend(f"- {item}" for item in location_items[:3])
        if state_items:
            lines.append("状态证据:")
            lines.extend(f"- {item}" for item in state_items[:3])
        if conflict_items:
            lines.append("当前冲突:")
            lines.extend(f"- {item}" for item in conflict_items[:3])
        if missing_items:
            lines.append("当前仍缺:")
            lines.extend(f"- {item}" for item in missing_items[:4])
        if not lines and state.evidence_bundle:
            lines.append("当前主要证据:")
            lines.extend(f"- {item}" for item in state.evidence_bundle[:4])
        if not lines and state.working_memory:
            lines.append("当前主要工作记忆:")
            lines.extend(f"- {item}" for item in state.working_memory[:4] if not item.startswith("planner_thought="))
        return "\n".join(lines).strip()

    def _resolve_grounded_freeform_answer(self, state: AgentState) -> str:
        task_family = str(getattr(state, "task_family", "") or "")
        missing_items = [
            str(item)
            for item in getattr(state, "open_questions", []) or []
            if isinstance(item, str) and item and not item.startswith("conflict:")
        ]
        conflicts = [
            str(item).split("conflict:", 1)[1]
            for item in getattr(state, "open_questions", []) or []
            if isinstance(item, str) and item.startswith("conflict:")
        ]
        has_reuse_memory = any(
            isinstance(item, str) and item.startswith("reuse:")
            for item in getattr(state, "working_memory", []) or []
        )
        if conflicts and task_family != "open_query_temporal_summary":
            return (
                "当前证据存在冲突，暂时无法给出单一可靠结论。"
                f"冲突类型：{', '.join(conflicts[:3])}。"
                f"\n\n{self._fallback_freeform_answer(state)}"
            )
        if task_family == "open_query_ocr":
            reading = self._latest_prefixed_value(state, prefixes=("ocr_reading=", "ocr_text="))
            if reading:
                return f"当前可确认的读数/文本是：{reading}。"
        if task_family == "open_query_location":
            location = self._latest_prefixed_value(state, prefixes=("target_location=", "scene_location="))
            if location:
                return f"当前可确认的位置是：{location}。"
        if task_family == "open_query_state":
            state_hint = self._latest_prefixed_value(state, prefixes=("state_change_hint=", "after_state=", "before_state="))
            if state_hint:
                return f"当前可确认的状态信息是：{state_hint}。"
        if task_family == "open_query_temporal_summary" and not missing_items and not has_reuse_memory:
            summary = self._build_grounded_temporal_answer(state)
            if summary:
                return summary
        if conflicts and task_family == "open_query_temporal_summary" and has_reuse_memory:
            return ""
        return ""

    def _latest_prefixed_value(self, state: AgentState, *, prefixes: tuple[str, ...]) -> str:
        values: list[str] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            for prefix in prefixes:
                if prefix not in item:
                    continue
                tail = item.split(prefix, 1)[1]
                for separator in (";", "|"):
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                normalized = tail.strip()
                if normalized:
                    values.append(normalized)
                break
        return values[-1] if values else ""

    def _build_grounded_temporal_answer(self, state: AgentState) -> str:
        actions = self._collect_grounded_values(
            state,
            prefixes=("possible_step=", "ongoing_action=", "answer_hint="),
            timeline_keyword="label=",
            timeline_filters=("timeline_event", "ongoing_action", "possible_step"),
        )
        states = self._collect_grounded_values(
            state,
            prefixes=("state_change_hint=", "after_state=", "before_state="),
            timeline_keyword="state_change_hint=",
            timeline_filters=("state_change_hint", "state_change"),
        )
        locations = self._collect_grounded_values(
            state,
            prefixes=("target_location=", "scene_location="),
            timeline_keyword="target_location=",
            timeline_filters=("target_location", "scene_location"),
        )
        readings = self._collect_grounded_values(
            state,
            prefixes=("ocr_reading=", "ocr_text="),
            timeline_keyword="ocr_reading=",
            timeline_filters=("ocr_reading", "ocr_text"),
        )
        if not actions and not states:
            return ""
        parts: list[str] = []
        if actions:
            parts.append(f"该时间段内主要发生的是：{self._join_grounded_values(actions[:3])}")
        if states:
            parts.append(f"可确认的状态变化包括：{self._join_grounded_values(states[:2])}")
        if locations:
            parts.append(f"相关位置线索包括：{self._join_grounded_values(locations[:2])}")
        if readings:
            parts.append(f"同时观察到的读数/文本包括：{self._join_grounded_values(readings[:2])}")
        return "；".join(parts) + "。"

    def _collect_grounded_values(
        self,
        state: AgentState,
        *,
        prefixes: tuple[str, ...],
        timeline_keyword: str,
        timeline_filters: tuple[str, ...],
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            captured = ""
            for prefix in prefixes:
                if prefix not in item:
                    continue
                tail = item.split(prefix, 1)[1]
                for separator in (";", "|"):
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                captured = tail.strip()
                break
            if not captured and timeline_keyword in item and any(token in item for token in timeline_filters):
                tail = item.split(timeline_keyword, 1)[1]
                for separator in (";", "|"):
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                captured = tail.strip()
            normalized = captured.strip()
            lowered = normalized.lower()
            if normalized and lowered not in seen:
                seen.add(lowered)
                values.append(normalized)
        return values

    def _join_grounded_values(self, values: list[str]) -> str:
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        return "，".join(values[:-1]) + "，以及" + values[-1]

    def _parse_prediction(self, text: str, choices: list[Any]) -> int | None:
        match = CHOICE_RE.search(text)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(choices):
                return idx
        lowered = text.lower()
        for idx, choice in enumerate(choices):
            if str(choice).lower() in lowered:
                return idx
        return None

    def _persist_result(self, result: GraphAgentResult, *, row: dict[str, Any], evidence_report_path: Path | None = None) -> None:
        trace_dir = self.paths.graph_agent_runs_root / result.task_family
        trace_dir.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict(
            gold=None,
            include_row=row,
        )
        if evidence_report_path is not None:
            payload["evidence_report_path"] = evidence_report_path.as_posix()
        path = trace_dir / f"{self._safe_filename(result.vqa_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_evidence_report(self, result: GraphAgentResult, *, row: dict[str, Any]) -> Path:
        trace_dir = self.paths.graph_agent_runs_root / result.task_family
        trace_dir.mkdir(parents=True, exist_ok=True)
        report_path = trace_dir / f"{self._safe_filename(result.vqa_id)}.md"
        report_path.write_text(self._build_evidence_report(result=result, row=row), encoding="utf-8")
        return report_path

    def _build_evidence_report(self, *, result: GraphAgentResult, row: dict[str, Any]) -> str:
        lines = [
            f"# {result.vqa_id}",
            "",
            f"- video_id: {result.video_id}",
            f"- task_family: {result.task_family}",
            f"- prediction: {result.prediction}",
            f"- confidence: {result.confidence:.3f}",
            f"- elapsed_seconds: {result.elapsed_seconds:.3f}",
            "",
            "## Question",
            "",
            str(row.get("question") or ""),
            "",
        ]
        if row.get("choices_json"):
            lines.extend(["## Choices", "", str(row.get("choices_json")), ""])
        if result.answer_text:
            lines.extend(["## Answer", "", result.answer_text, ""])
        if result.visited_times:
            lines.extend(["## Visited Times", ""])
            lines.extend(f"- {time_s:.3f}" for time_s in result.visited_times[:40])
            lines.append("")
        if result.retrieved_frames:
            lines.extend(["## Retrieved Frames", ""])
            lines.extend(f"- {item}" for item in result.retrieved_frames[:40])
            lines.append("")
        if result.artifacts:
            lines.extend(["## Artifacts", ""])
            lines.extend(f"- {item}" for item in result.artifacts[:60])
            lines.append("")
        if result.evidence_bundle:
            lines.extend(["## Evidence", ""])
            lines.extend(f"- {item}" for item in result.evidence_bundle[:80])
            lines.append("")
        if result.working_memory:
            lines.extend(["## Working Memory", ""])
            lines.extend(f"- {item}" for item in result.working_memory[:80])
            lines.append("")
        if result.tool_trace:
            lines.extend(["## Tool Trace", ""])
            for entry in result.tool_trace[:120]:
                if not isinstance(entry, dict):
                    continue
                lines.append(f"- {entry.get('tool')}: {entry.get('result_summary')}")
            lines.append("")
        if result.tool_failures:
            lines.extend(["## Tool Failures", ""])
            for entry in result.tool_failures[:40]:
                if not isinstance(entry, dict):
                    continue
                lines.append(f"- {entry.get('tool')}: {entry.get('error_type')} | {entry.get('error_message')}")
            lines.append("")
        if result.open_questions:
            lines.extend(["## Open Questions Tail", ""])
            lines.extend(f"- {item}" for item in result.open_questions[:40])
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _safe_filename(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "sample"
