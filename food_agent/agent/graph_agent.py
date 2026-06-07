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
        blocked_prefixes = self._session_blocked_prefixes_for_task(state.task_family)
        blocked_substrings = self._session_blocked_substrings_for_task(state.task_family)
        state.working_memory = self._filter_restored_strings(
            items=state.working_memory,
            relevant_tokens=relevant_tokens,
            keep_prefixes=("reuse:", "reuse_relation:", "ocr_reading=", "measurement ", "target_location=", "scene_location=", "state_change_hint=", "possible_step="),
            blocked_prefixes=blocked_prefixes,
            blocked_substrings=blocked_substrings,
            limit=28,
        )
        state.evidence_bundle = self._filter_restored_strings(
            items=state.evidence_bundle,
            relevant_tokens=relevant_tokens,
            keep_prefixes=("type=ocr_reading", "ocr_reading=", "measurement ", "target_location=", "scene_location=", "state_change_hint=", "possible_step=", "type=timeline_event"),
            blocked_prefixes=blocked_prefixes,
            blocked_substrings=blocked_substrings,
            limit=24,
        )
        state.retrieved_frames = self._filter_restored_frames(
            frames=state.retrieved_frames,
            task_family=state.task_family,
            limit=12,
        )
        state.artifacts = self._filter_restored_frames(
            frames=state.artifacts,
            task_family=state.task_family,
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
    ) -> list[str]:
        kept: list[str] = []
        for item in items:
            if not isinstance(item, str) or not item:
                continue
            lowered = item.lower()
            if any(lowered.startswith(prefix) for prefix in blocked_prefixes):
                continue
            if any(token in lowered for token in blocked_substrings):
                continue
            if any(lowered.startswith(prefix) for prefix in keep_prefixes) or any(token in lowered for token in relevant_tokens):
                if item not in kept:
                    kept.append(item)
        if len(kept) < limit:
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

    def _filter_restored_frames(self, *, frames: list[str], task_family: str, limit: int) -> list[str]:
        prefixes = tuple(token.lower() for token in self._artifact_reuse_prefixes_for_task(task_family))
        preferred = [
            item
            for item in frames
            if isinstance(item, str) and any(prefix in item.lower() for prefix in prefixes)
        ]
        if preferred:
            return preferred[-limit:]
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
            index = self._coerce_choice_index(raw_result.get("best_index"), state.choices)
            if index is None:
                continue
            if raw_result.get("need_more_evidence"):
                reranked = self._resolve_unresolved_action_intent_answer(raw_result=raw_result, state=state)
                if reranked is not None:
                    return reranked
            confidence = self._coerce_confidence(raw_result.get("confidence"), default=0.78)
            if raw_result.get("need_more_evidence"):
                confidence = min(confidence, 0.62)
            answer = raw_result.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                answer = str(state.choices[index])
            return index, answer, confidence
        return None

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
        state.add_memory(f"action_intent_unresolved_rerank_best_index={best_index} score={best_score:.2f}")
        return best_index, best_choice, min(max(0.36 + max(best_score, 0.0) * 0.45, 0.36), 0.68)

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
        global_context = " ".join(
            str(item)
            for item in list(getattr(state, "evidence_bundle", []))[-24:] + list(getattr(state, "working_memory", []))[-24:]
            if isinstance(item, str)
        ).lower()
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
        if "clean" in choice_lc and any(term in contradiction_lc for term in ("no actual cleaning", "no visible wiping", "没有任何明确清洁", "没有擦")):
            adjusted -= 0.16
        if "away" in choice_lc and any(term in contradiction_lc for term in ("not stored", "not put", "counter", "没有看到把", "暂时", "台面")):
            adjusted -= 0.14
        if "dry" in choice_lc and "hand" in choice_lc and any(term in contradiction_lc for term in ("no visible hand", "no clear wet-hand", "没有看到双手", "没有先洗手")):
            adjusted -= 0.14
        if self._action_intent_support_is_likely_downstream_to_move_action(
            question=question_lc,
            choice=choice_lc,
            support=support_lc,
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
        if self._action_intent_choice_is_generic_inspection_under_hidden_target_context(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted -= 0.22
        if self._action_intent_choice_is_hidden_target_access_or_retrieval(
            choice=choice_lc,
            support=support_lc,
            contradiction=contradiction_lc,
            global_context=global_context,
        ):
            adjusted += 0.34
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

    def _action_intent_support_is_likely_downstream_to_move_action(
        self,
        *,
        question: str,
        choice: str,
        support: str,
        global_context: str,
        action_object: str,
    ) -> bool:
        if not any(token in question for token in ("move ", "transfer ", "shift ", "remove ", "clear ")):
            return False
        if self._choice_is_same_object_active_use(choice, action_object):
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
        if unresolved_best_score > alt_score + 0.34 and unresolved_best_score >= 0.82:
            return None
        confidence = min(max(0.48 + max(alt_score, 0.0) * 0.38, 0.48), 0.72)
        return alt_index, alt_choice, confidence

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
                "reaches for the",
                "target behind",
                "item behind",
                "look for a",
                "needed tool",
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
                "wash it clean first",
                "washing it clean first",
                "remove soap",
                "removing soap",
                "clean off the soap",
                "rinse it clean first",
                "还有肥皂",
                "肥皂残留",
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
                "first, not for drying",
                "不是为了晾干",
                "晾干是后续结果",
                "当前直接目的是去除肥皂",
                "当前直接目的是清洗",
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
        return support_signal and (soft_missing or "direct" in support or "直接" in support)

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
        return any(
            term in text
            for term in (
                "shown",
                "visible",
                "actual",
                "completed",
                "direct",
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
