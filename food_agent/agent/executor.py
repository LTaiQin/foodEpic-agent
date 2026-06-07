"""Looping executor for the complete graph/video agent."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from food_agent.agent.artifact_policy import artifact_reuse_prefixes_for_task
from food_agent.agent.planner import GraphAgentPlanner, PlannerDecision
from food_agent.agent.state import AgentState
from food_agent.agent.verifier import GraphAgentVerifier
from food_agent.tools import AgentToolbox


class GraphAgentExecutor:
    def __init__(self, toolbox: AgentToolbox, planner: GraphAgentPlanner, verifier: GraphAgentVerifier | None = None):
        self.toolbox = toolbox
        self.planner = planner
        self.verifier = verifier or GraphAgentVerifier()

    def execute(self, state: AgentState) -> AgentState:
        self.toolbox.set_runtime_context(question=state.question, inputs_json=state.inputs_json)
        hints = self.toolbox.default_hints(state.question, state.inputs_json)
        self._seed_reusable_memory(state, hints)
        self._initialize_reasoning_state(state, hints)
        self._emit_heartbeat(state=state, phase="initialized")
        for step_index in range(state.max_steps):
            state.current_step = step_index
            self._refresh_open_questions_before_planning(state)
            self._emit_heartbeat(state=state, phase="before_plan")
            decision = self.planner.next_action(state=state, tool_schemas=self.toolbox.tool_schemas(), hints=hints)
            state.plan_summary = decision.thought
            self._record_planner_reflection(state, decision)
            self._emit_heartbeat(state=state, phase="after_plan", tool=decision.tool, extra={"thought": decision.thought})
            if decision.done and decision.tool == "finish":
                verification = self.verifier.verify(state=state)
                state.record_verification(
                    sufficient=verification.sufficient,
                    confidence=verification.confidence,
                    missing_evidence_types=verification.missing_evidence_types,
                    conflicts=verification.conflicts,
                    recommend_next_action=verification.recommend_next_action,
                    summary=verification.summary,
                )
                state.add_memory(f"verifier={verification.summary}")
                self._emit_heartbeat(
                    state=state,
                    phase="after_verify",
                    tool="finish",
                    extra={
                        "sufficient": verification.sufficient,
                        "recommend_next_action": verification.recommend_next_action,
                        "missing_evidence_types": verification.missing_evidence_types,
                        "conflicts": verification.conflicts,
                    },
                )
                if not verification.sufficient:
                    state.replace_open_questions(
                        state.open_questions + [item for item in verification.missing_evidence_types if item not in state.open_questions]
                    )
                    for conflict in verification.conflicts:
                        state.add_open_question(f"conflict:{conflict}")
                    state.add_hypothesis(f"verifier_blocked_finish={verification.recommend_next_action}")
                    continue
                finish_payload = self.toolbox.finish(**decision.args)
                self._apply_finish(state, finish_payload)
                state.record_tool("finish", decision.args, self._summarize(finish_payload), raw_result=finish_payload)
                self._emit_heartbeat(state=state, phase="finished", tool="finish", extra={"prediction": state.final_prediction})
                break
            skip_result = self._maybe_skip_explicit_writeback(state, decision)
            if skip_result is not None:
                self._apply_tool_result(state, decision, skip_result)
                self._emit_heartbeat(state=state, phase="tool_skipped", tool=decision.tool, extra=skip_result)
                continue
            try:
                result = self.toolbox.run(decision.tool, decision.args)
            except Exception as exc:  # noqa: BLE001
                self._handle_tool_failure(state, decision, exc)
                self._emit_heartbeat(
                    state=state,
                    phase="tool_failed",
                    tool=decision.tool,
                    extra={"error_type": type(exc).__name__, "error_message": str(exc)},
                )
                continue
            self._apply_tool_result(state, decision, result)
            self._emit_heartbeat(
                state=state,
                phase="after_tool",
                tool=decision.tool,
                extra={
                    "done": bool(result.get("done")),
                    "result_keys": sorted(str(key) for key in result.keys())[:12],
                },
            )
            if result.get("done"):
                self._apply_finish(state, result)
                self._emit_heartbeat(state=state, phase="finished", tool=decision.tool, extra={"prediction": state.final_prediction})
                break
        return state

    def _is_weight_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredient_weight"

    def _is_viewpoint_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {"3d_perception_fixture_location", "gaze_gaze_estimation"}

    def _is_location_conclusion_sensitive_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {
            "3d_perception_fixture_location",
            "gaze_gaze_estimation",
            "3d_perception_object_location",
        }

    def _is_action_intent_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "fine_grained_why_recognition"

    def _is_object_motion_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")).startswith("object_motion_")

    def _is_spatial_tracking_task(self, state: AgentState) -> bool:
        task_family = str(getattr(state, "task_family", ""))
        return task_family.startswith("object_motion_") or task_family.startswith("3d_perception_") or task_family.startswith("gaze_")

    def _should_request_ocr_evidence(self, state: AgentState) -> bool:
        if self._is_viewpoint_task(state) or self._is_spatial_tracking_task(state):
            return False
        return True

    def _emit_heartbeat(
        self,
        *,
        state: AgentState,
        phase: str,
        tool: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        path = os.environ.get("FOOD_AGENT_HEARTBEAT_PATH")
        if not path:
            return
        payload = {
            "ts": time.time(),
            "phase": phase,
            "step": state.current_step,
            "tool": tool,
            "task_family": state.task_family,
            "open_questions": state.open_questions[-8:],
            "working_memory_tail": state.working_memory[-8:],
            "evidence_tail": state.evidence_bundle[-8:],
            "retrieved_frame_count": len(state.retrieved_frames),
            "retrieved_node_count": len(state.retrieved_nodes),
            "tool_failures": state.tool_failures[-3:],
            "ineffective_tools": state.ineffective_tools[-3:],
        }
        if extra:
            payload["extra"] = extra
        heartbeat_path = Path(path)
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        with heartbeat_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _apply_tool_result(self, state: AgentState, decision: PlannerDecision, result: dict[str, Any]) -> None:
        before_counts = self._state_counts(state)
        state.record_tool(decision.tool, decision.args, self._summarize(result), raw_result=result)
        self._merge_result_into_state(state, decision.tool, result)
        self._update_reasoning_after_tool(state, decision.tool, result)
        self._reconcile_conflict_questions(state)
        self._record_ineffective_tool_if_needed(state, decision, result, before_counts)

    def _initialize_reasoning_state(self, state: AgentState, hints: dict[str, Any]) -> None:
        supports_vision = getattr(self.toolbox.model_client, "supports_vision_requests", None)
        if callable(supports_vision):
            try:
                if not supports_vision():
                    reason_getter = getattr(self.toolbox.model_client, "vision_disable_reason", None)
                    reason = reason_getter() if callable(reason_getter) else "vision_disabled"
                    state.add_memory(f"vision_disabled={reason or 'vision_disabled'}")
            except Exception:  # noqa: BLE001
                pass
        state.add_hypothesis(f"task_family={state.task_family}")
        if state.task_family == "open_query_ocr":
            state.add_open_question("need_ocr_reading")
            state.add_open_question("need_disambiguating_evidence")
        if state.task_family == "open_query_location":
            state.add_open_question("need_location_evidence")
            state.add_open_question("need_disambiguating_evidence")
        if state.task_family == "open_query_state":
            state.add_open_question("need_state_evidence")
            state.add_open_question("need_disambiguating_evidence")
        if state.task_family == "open_query_temporal_summary":
            state.add_open_question("need_time_localization")
            state.add_open_question("need_initial_observation")
        if hints.get("times") or hints.get("input_times"):
            state.add_open_question("need_time_localization")
        if hints.get("bbox"):
            state.add_open_question("need_region_grounding")
        if hints.get("ocr_keyword") and self._should_request_ocr_evidence(state):
            state.add_open_question("need_ocr_reading")
        if hints.get("state_keyword"):
            state.add_open_question("need_state_evidence")
        if hints.get("location_keyword") and not self._is_weight_task(state):
            state.add_open_question("need_location_evidence")
        if not state.open_questions:
            state.add_open_question("need_disambiguating_evidence")

    def _refresh_open_questions_before_planning(self, state: AgentState) -> None:
        refreshed: list[str] = []
        is_weight_task = self._is_weight_task(state)
        if not state.evidence_bundle:
            refreshed.append("need_disambiguating_evidence")
        for item in self._refresh_question_targets(state, is_weight_task=is_weight_task):
            refreshed.append(item)
        if state.question.lower() and not state.retrieved_frames and not state.retrieved_nodes:
            refreshed.append("need_initial_observation")
        if refreshed:
            merged = state.open_questions + [item for item in refreshed if item not in state.open_questions]
            state.replace_open_questions(merged)

    def _refresh_question_targets(self, state: AgentState, *, is_weight_task: bool) -> tuple[str, ...]:
        refreshed: list[str] = []
        combined_memory = list(state.evidence_bundle) + list(state.working_memory)
        question = state.question.lower()
        if (
            not any(item.startswith("ocr_reading=") for item in state.working_memory)
            and any(token in question for token in ("weight", "gram", "grams", "read", "number", "digit"))
            and self._should_request_ocr_evidence(state)
        ):
            refreshed.append("need_ocr_reading")
        if (
            not any("target_location=" in item or "scene_location=" in item for item in combined_memory)
            and (not is_weight_task)
            and any(token in question for token in ("where", "location", "left", "right", "front", "behind"))
        ):
            refreshed.append("need_location_evidence")
        if (
            not any("state_change_hint=" in item or "type=state_change" in item for item in combined_memory)
            and any(token in question for token in ("state", "become", "change", "cooked", "mixed", "done"))
        ):
            refreshed.append("need_state_evidence")
        return tuple(refreshed)

    def _tool_open_question_targets(self, tool_name: str) -> tuple[str, ...]:
        mapping: dict[str, tuple[str, ...]] = {
            "query_time": ("need_time_localization", "need_initial_observation"),
            "sample_sparse_frames": ("need_time_localization", "need_initial_observation"),
            "extract_frames_for_range": ("need_time_localization", "need_initial_observation"),
            "sample_frames_around_peaks": ("need_time_localization", "need_initial_observation"),
            "query_region": ("need_region_grounding",),
            "render_bbox_overlay": ("need_region_grounding",),
            "extract_region_with_context": ("need_region_grounding",),
            "resolve_bbox_reference": ("need_region_grounding",),
            "query_ocr": ("need_ocr_reading",),
            "run_ocr_on_image": ("need_ocr_reading",),
            "run_ocr_on_region": ("need_ocr_reading",),
            "query_state": ("need_state_evidence",),
            "write_state_change": ("need_state_evidence",),
            "inspect_visual_evidence": ("need_state_evidence",),
            "query_location": ("need_location_evidence",),
            "infer_viewpoint_choice": ("need_location_evidence",),
            "infer_named_fixture_direction": ("need_location_evidence",),
            "infer_gaze_target_with_context": ("need_location_evidence",),
            "resolve_action_intent_pairwise": ("need_disambiguating_evidence",),
            "resolve_action_intent_future_use": ("need_disambiguating_evidence",),
        }
        return mapping.get(tool_name, ())

    def _apply_tool_question_prune_targets(self, state: AgentState, tool_name: str) -> None:
        for item in self._tool_open_question_targets(tool_name):
            state.prune_open_question(item)

    def _has_state_evidence(self, state: AgentState) -> bool:
        return any("state_change_hint=" in item or "type=state_change" in item for item in state.evidence_bundle + state.working_memory)

    def _has_location_evidence(self, state: AgentState) -> bool:
        return any("target_location=" in item or "scene_location=" in item for item in state.evidence_bundle + state.working_memory)

    def _has_ocr_evidence(self, state: AgentState, result: dict[str, Any]) -> bool:
        return bool(result.get("reading") or result.get("text") or any(item.startswith("ocr_reading=") for item in state.working_memory))

    def _record_planner_reflection(self, state: AgentState, decision: PlannerDecision) -> None:
        if decision.thought:
            state.add_memory(f"planner_thought={decision.thought}")
        if decision.tool:
            state.add_hypothesis(f"plan_step={state.current_step}; tool={decision.tool}")
        if decision.tool:
            self._apply_tool_question_prune_targets(state, decision.tool)

    def _update_reasoning_after_tool(self, state: AgentState, tool_name: str, result: dict[str, Any]) -> None:
        if tool_name in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
            self._record_action_intent_resolution_state(state, tool_name=tool_name, result=result)
        if result.get("nodes") or result.get("matches") or result.get("totals") or result.get("artifact_path") or result.get("artifact_paths"):
            state.prune_open_question("need_disambiguating_evidence")
        if state.retrieved_frames or state.retrieved_nodes or state.evidence_bundle:
            state.prune_open_question("need_initial_observation")
        if tool_name in {"query_time", "sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks"}:
            if state.retrieved_frames or state.retrieved_nodes:
                state.prune_open_question("need_time_localization")
                state.prune_open_question("need_initial_observation")
        if tool_name in {"query_region", "render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference"}:
            if state.retrieved_frames or result.get("association_id") or result.get("tracks"):
                state.prune_open_question("need_region_grounding")
        if tool_name in {"query_ocr", "run_ocr_on_image", "run_ocr_on_region"}:
            if self._has_ocr_evidence(state, result):
                state.prune_open_question("need_ocr_reading")
                state.add_hypothesis("ocr_evidence_collected")
        if tool_name in {"query_state", "inspect_visual_evidence", "write_state_change"}:
            if self._has_state_evidence(state):
                state.prune_open_question("need_state_evidence")
                state.add_hypothesis("state_evidence_collected")
        if tool_name in {"query_location", "infer_viewpoint_choice", "infer_named_fixture_direction", "infer_gaze_target_with_context"}:
            if self._has_location_evidence(state):
                state.prune_open_question("need_location_evidence")
                state.add_hypothesis("location_evidence_collected")
        if self._is_object_motion_task(state):
            if tool_name in {"resolve_bbox_reference", "estimate_object_movement_count", "estimate_stationary_start"}:
                state.prune_open_question("need_region_grounding")
            if tool_name == "estimate_object_movement_count" and any(
                isinstance(item, str) and item.startswith("movement_count=") for item in state.working_memory
            ):
                state.prune_open_question("need_location_evidence")
                state.prune_open_question("need_state_evidence")
                state.prune_open_question("need_alternative_evidence_path")
                state.prune_open_question("need_time_localization")
                state.add_hypothesis("object_motion_evidence_collected")
            if tool_name == "estimate_stationary_start" and any(
                isinstance(item, str) and item.startswith("stationary_best_index=") for item in state.working_memory
            ):
                state.prune_open_question("need_location_evidence")
                state.prune_open_question("need_state_evidence")
                state.prune_open_question("need_alternative_evidence_path")
                state.prune_open_question("need_time_localization")
                state.add_hypothesis("stationary_evidence_collected")
        if tool_name == "rank_choices_from_state" and result.get("best_index") is not None:
            state.add_hypothesis(f"candidate_answer_index={result.get('best_index')}")
        if self._is_weight_task(state) and self._has_stable_weight_answer_evidence(state):
            state.prune_open_question("need_alternative_evidence_path")
        if result.get("best_index") is not None and not result.get("need_more_evidence"):
            self._prune_open_questions_for_structured_answer(state, tool_name)
        if tool_name == "finish" or result.get("done"):
            state.replace_open_questions([])

    def _handle_tool_failure(self, state: AgentState, decision: PlannerDecision, exc: Exception) -> None:
        error_type = type(exc).__name__
        error_message = str(exc)
        state.record_tool_failure(decision.tool, decision.args, error_type, error_message)
        state.add_memory(f"tool_failure tool={decision.tool} error_type={error_type}")
        state.add_hypothesis(f"failed_tool={decision.tool}")
        state.add_open_question("need_alternative_evidence_path")
        for item in self._tool_open_question_targets(decision.tool):
            state.add_open_question(item)
        if decision.tool in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
            state.prune_open_question("need_alternative_evidence_path")
            state.add_open_question("need_disambiguating_evidence")
            state.working_memory = [
                item
                for item in state.working_memory
                if not (
                    isinstance(item, str)
                    and (
                        item.startswith("action_intent_need_future_evidence=")
                        or item.startswith("action_intent_second_best_index=")
                    )
                )
            ]
            state.add_memory(f"action_intent_pending_resolution={decision.tool}")
            candidate_indices = decision.args.get("candidate_indices") if isinstance(decision.args, dict) else None
            if candidate_indices:
                state.add_memory(f"action_intent_pending_candidates={candidate_indices}")
            state.add_memory("action_intent_resolution_failed_retry_pending=1")

    def _record_ineffective_tool_if_needed(
        self,
        state: AgentState,
        decision: PlannerDecision,
        result: dict[str, Any],
        before_counts: dict[str, int],
    ) -> None:
        if decision.tool in {"finish", "write_observation", "write_frame_observation", "write_region_observation", "write_ocr_reading", "write_audio_event", "write_timeline_summary", "write_state_change"}:
            return
        if result.get("done"):
            return
        after_counts = self._state_counts(state)
        no_new_evidence = before_counts == after_counts
        empty_payload = not any(
            result.get(key)
            for key in ("nodes", "matches", "totals", "artifact_path", "artifact_paths", "reading", "text", "peaks", "scores", "best_index", "association_id", "tracks")
        )
        if no_new_evidence and empty_payload:
            state.record_ineffective_tool(decision.tool, decision.args, "no_new_evidence")
            state.add_memory(f"tool_ineffective tool={decision.tool} reason=no_new_evidence")
            state.add_hypothesis(f"ineffective_tool={decision.tool}")
            state.add_open_question("need_alternative_evidence_path")

    def _state_counts(self, state: AgentState) -> dict[str, int]:
        return {
            "nodes": len(state.retrieved_nodes),
            "frames": len(state.retrieved_frames),
            "evidence": len(state.evidence_bundle),
            "memory": len(state.working_memory),
        }

    def _structured_answer_prune_targets(self, state: AgentState, tool_name: str) -> tuple[str, ...]:
        direct_map: dict[str, tuple[str, ...]] = {
            "infer_temporal_localization_choice": (
                "need_time_localization",
                "need_initial_observation",
                "need_disambiguating_evidence",
            ),
            "infer_ingredient_retrieval_choice": (
                "need_time_localization",
                "need_initial_observation",
                "need_disambiguating_evidence",
            ),
            "infer_recipe_ingredient_membership_choice": (
                "need_disambiguating_evidence",
                "need_location_evidence",
            ),
            "infer_exact_ingredient_amount_choice": (
                "need_disambiguating_evidence",
                "need_ocr_reading",
            ),
            "infer_recipe_catalog_choice": (
                "need_disambiguating_evidence",
                "need_location_evidence",
                "need_time_localization",
            ),
            "infer_recipe_nutrition_choice": (
                "need_disambiguating_evidence",
                "need_location_evidence",
            ),
            "infer_visual_mcq": (
                "need_region_grounding",
                "need_location_evidence",
            ),
            "infer_named_fixture_direction": (
                "need_location_evidence",
                "need_time_localization",
            ),
            "infer_gaze_target_with_context": (
                "need_location_evidence",
                "need_time_localization",
            ),
            "infer_object_drop_location": (
                "need_time_localization",
                "need_region_grounding",
                "need_location_evidence",
                "need_alternative_evidence_path",
            ),
            "infer_object_movement_itinerary": (
                "need_location_evidence",
                "need_region_grounding",
                "need_time_localization",
                "need_alternative_evidence_path",
            ),
            "resolve_action_intent_pairwise": (
                "need_disambiguating_evidence",
                "need_location_evidence",
                "need_state_evidence",
                "need_time_localization",
                "need_initial_observation",
            ),
            "resolve_action_intent_future_use": (
                "need_disambiguating_evidence",
                "need_location_evidence",
                "need_state_evidence",
                "need_time_localization",
                "need_initial_observation",
            ),
        }
        if tool_name in direct_map:
            return direct_map[tool_name]
        if tool_name in {"infer_action_mechanism", "infer_action_intent"}:
            return (
                "need_disambiguating_evidence",
                "need_location_evidence",
                "need_state_evidence",
                "need_time_localization",
                "need_initial_observation",
            )
        return ()

    def _prune_open_questions_for_structured_answer(self, state: AgentState, tool_name: str) -> None:
        for item in self._structured_answer_prune_targets(state, tool_name):
            state.prune_open_question(item)

    def _clear_action_intent_resolution_memory(self, state: AgentState) -> None:
        state.working_memory = [
            item
            for item in state.working_memory
            if not (
                isinstance(item, str)
                and (
                    item.startswith("action_intent_need_future_evidence=")
                    or item.startswith("action_intent_second_best_index=")
                    or item.startswith("action_intent_pending_resolution=")
                    or item.startswith("action_intent_pending_candidates=")
                    or item.startswith("action_intent_needed_observation=")
                )
            )
        ]

    def _record_action_intent_resolution_state(
        self,
        state: AgentState,
        *,
        tool_name: str,
        result: dict[str, Any],
    ) -> None:
        self._clear_action_intent_resolution_memory(state)
        if result.get("need_more_evidence"):
            state.add_memory(f"action_intent_pending_resolution={tool_name}")
            if result.get("candidate_indices"):
                state.add_memory(f"action_intent_pending_candidates={result.get('candidate_indices')}")
            if result.get("needed_observation"):
                state.add_memory(f"action_intent_needed_observation={result.get('needed_observation')}")
            state.add_open_question("need_disambiguating_evidence")
        else:
            state.prune_open_question("need_disambiguating_evidence")
        state.add_memory(
            f"action_intent_best_index={result.get('best_index')} confidence={result.get('confidence')}"
        )

    def _has_stable_weight_answer_evidence(self, state: AgentState) -> bool:
        if not self._is_weight_task(state):
            return False
        choice_values: list[tuple[int, float, str]] = []
        for index, choice in enumerate(state.choices):
            parsed = self._parse_numeric_value(str(choice))
            if parsed is None:
                continue
            choice_values.append((index, parsed, str(choice)))
        if not choice_values:
            return False
        return bool(
            self._extract_prefixed_numeric_values(state, prefix="normalized=", measurement_only=True)
            or self._extract_prefixed_numeric_values(state, prefix="ocr_reading=", measurement_only=False)
        )

    def _extract_prefixed_numeric_values(self, state: AgentState, *, prefix: str, measurement_only: bool) -> list[float]:
        values: list[float] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str) or prefix not in item:
                continue
            if measurement_only and "measurement " not in item:
                continue
            parsed = self._parse_numeric_value(item.split(prefix, 1)[1])
            if parsed is not None:
                values.append(parsed)
        return values

    def _parse_numeric_value(self, text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)", str(text))
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _merge_result_into_state(self, state: AgentState, tool_name: str, result: dict[str, Any]) -> None:
        self._record_result_times(state, result)
        nodes = result.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                state.add_node_result(node)
                evidence = self._node_to_evidence(node)
                if evidence:
                    state.add_evidence(evidence)
                    state.add_memory(evidence)
        matches = result.get("matches")
        if isinstance(matches, list):
            for match in matches[:10]:
                if not isinstance(match, dict):
                    continue
                summary = (
                    f"measurement ingredient={match.get('label')} amount={match.get('amount')} "
                    f"unit={match.get('amount_unit')} normalized={match.get('normalized_answer')}"
                )
                state.add_evidence(summary)
                state.add_memory(summary)
        totals = result.get("totals")
        if isinstance(totals, dict):
            summary = "nutrition_change " + ", ".join(f"{key}={value}" for key, value in totals.items())
            state.add_evidence(summary)
            state.add_memory(summary)
        object_tracks = result.get("object_tracks")
        object_masks = result.get("object_masks")
        gaze_priming = result.get("gaze_priming")
        audio_events = result.get("audio_events")
        if tool_name == "query_spatial_context":
            if isinstance(object_tracks, list):
                for item in object_tracks[:10]:
                    if not isinstance(item, dict):
                        continue
                    state.add_memory(
                        f"spatial_track object={item.get('object_name')} association_id={item.get('association_id')} "
                        f"time={item.get('start_time')}-{item.get('end_time')}"
                    )
            if isinstance(object_masks, list):
                for item in object_masks[:10]:
                    if not isinstance(item, dict):
                        continue
                    state.add_memory(
                        f"spatial_mask fixture={item.get('fixture')} frame={item.get('frame_number')}"
                    )
            if isinstance(gaze_priming, list):
                state.add_memory(f"gaze_priming_count={len(gaze_priming)}")
            if isinstance(audio_events, list):
                state.add_memory(f"audio_event_count={len(audio_events)}")
            if self._is_viewpoint_task(state) and result.get("count"):
                state.prune_open_question("need_time_localization")
        scores = result.get("scores")
        if tool_name == "compare_choice_nutrition" and isinstance(scores, list):
            for item in scores[:10]:
                if not isinstance(item, dict):
                    continue
                state.add_memory(
                    f"nutrition_choice index={item.get('index')} choice={item.get('choice')} "
                    f"{item.get('nutrient')}={item.get('value')}"
                )
        if tool_name == "resolve_bbox_reference":
            association_id = result.get("association_id")
            object_name = result.get("object_name")
            fixture = result.get("fixture")
            if association_id or object_name:
                state.add_evidence(
                    f"bbox_reference association_id={association_id} object_name={object_name} fixture={fixture}"
                )
                state.add_memory(
                    f"bbox_reference association_id={association_id} object_name={object_name} fixture={fixture}"
                )
            tracks = result.get("tracks")
            if isinstance(tracks, list):
                for track in tracks[:10]:
                    if not isinstance(track, dict):
                        continue
                    state.add_memory(
                        f"track association_id={track.get('association_id')} track_index={track.get('track_index')} "
                        f"time={track.get('start_time')}-{track.get('end_time')}"
                    )
        edges = result.get("edges")
        if isinstance(edges, list):
            for edge in edges[:10]:
                summary = (
                    f"edge={edge.get('edge_type')} {edge.get('source_id')} -> {edge.get('target_id')}"
                    if isinstance(edge, dict)
                    else str(edge)
                )
                state.add_memory(summary)
        artifact_path = result.get("artifact_path")
        if isinstance(artifact_path, str) and artifact_path:
            state.add_artifact(artifact_path)
            state.add_memory(f"artifact={artifact_path}")
        artifact_paths = result.get("artifact_paths")
        if isinstance(artifact_paths, list):
            for path in artifact_paths:
                if isinstance(path, str) and path:
                    state.add_artifact(path)
                    state.add_memory(f"artifact={path}")
        if tool_name == "retrieve_cached_artifacts":
            items = result.get("items")
            if isinstance(items, list):
                state.add_memory(f"cached_artifact_count={len(items)}")
                for item in items[:8]:
                    if not isinstance(item, dict):
                        continue
                    if item.get("time_s") is not None:
                        state.add_visited_time(item.get("time_s"))
                    if item.get("artifact_path"):
                        state.add_memory(
                            f"cached_artifact time={item.get('time_s')} path={item.get('artifact_path')}"
                        )
                self._auto_write_cached_artifact_reuse(state, items)
        if tool_name == "inspect_visual_evidence":
            if result.get("vision_disabled"):
                reason = str(result.get("error_message") or result.get("error_type") or "vision_disabled")
                state.add_memory(f"vision_disabled={reason}")
                state.add_hypothesis("vision_evidence_unavailable")
                state.add_open_question("need_alternative_evidence_path")
            self._record_semantic_conflicts_from_payload(state, result)
            summary = self._inspection_summary(result)
            if summary:
                state.add_evidence(summary)
                state.add_memory(summary)
            self._auto_write_visual_observation(state, result)
        if tool_name in {"run_ocr_on_image", "run_ocr_on_region"}:
            reading = result.get("reading")
            text = result.get("text")
            if reading:
                self._record_conflict_if_needed(
                    state,
                    conflict_type="conflicting_ocr_readings",
                    new_value=str(reading),
                    prefixes=("ocr_reading=",),
                )
                self._dedupe_conflicting_prefixed_entries(
                    state,
                    prefixes=("ocr_reading=",),
                    keep_value=str(reading),
                )
                state.add_evidence(f"ocr_reading={reading}")
                state.add_memory(f"ocr_reading={reading}")
            if text and text != reading:
                state.add_memory(f"ocr_text={text}")
            self._auto_write_ocr_observation(state, tool_name, result)
        if tool_name == "detect_audio_peaks":
            peaks = result.get("peaks")
            if isinstance(peaks, list):
                state.add_memory(f"audio_peak_count={len(peaks)}")
                for peak in peaks[:5]:
                    if not isinstance(peak, dict):
                        continue
                    if peak.get("time_s") is not None:
                        state.add_visited_time(peak.get("time_s"))
                    state.add_memory(
                        f"audio_peak time={peak.get('time_s')} score={peak.get('score')}"
                    )
                self._auto_write_audio_peaks(state, peaks)
        if tool_name == "rank_choices_from_state":
            best_index = result.get("best_index")
            scores = result.get("scores")
            if best_index is not None:
                state.add_memory(f"ranked_best_index={best_index}")
            if scores:
                state.add_memory(f"choice_scores={scores}")
        if tool_name == "count_visual_candidates":
            state.add_memory(
                f"count_candidates count={result.get('count')} best_index={result.get('best_index')} "
                f"matches={result.get('matching_event_indices')}"
            )
            if result.get("reason"):
                state.add_evidence(f"count_reason={result.get('reason')}")
        if tool_name == "estimate_object_movement_count":
            state.add_memory(
                f"movement_count={result.get('movement_count')} best_index={result.get('best_index')} "
                f"object_name={result.get('object_name')}"
            )
        if tool_name == "estimate_stationary_start":
            state.add_memory(
                f"stationary_best_index={result.get('best_index')} object_name={result.get('object_name')} "
                f"valid_candidates={result.get('valid_candidates')}"
            )
        if result.get("best_index") is not None and not result.get("need_more_evidence"):
            self._prune_open_questions_for_structured_answer(state, tool_name)
        if tool_name == "infer_object_movement_itinerary":
            state.add_memory(
                f"itinerary_best_index={result.get('best_index')} confidence={result.get('confidence')} "
                f"object_name={result.get('object_name')}"
            )
            if result.get("answer"):
                state.add_memory(f"target_location={result.get('answer')}")
                state.add_evidence(f"target_location={result.get('answer')}")
        if tool_name == "infer_object_drop_location":
            state.add_memory(
                f"object_location_best_index={result.get('best_index')} confidence={result.get('confidence')} "
                f"fixture={result.get('final_fixture')}"
            )
            if result.get("answer"):
                state.add_memory(f"target_location={result.get('answer')}")
                state.add_evidence(f"target_location={result.get('answer')}")
        if tool_name == "identify_image_ingredients":
            items = result.get("items")
            if isinstance(items, list):
                for item in items[:10]:
                    if not isinstance(item, dict):
                        continue
                    state.add_memory(
                        f"identified_image index={item.get('index')} ingredient={item.get('ingredient')} "
                        f"confidence={item.get('confidence')}"
                    )
        if tool_name == "infer_gaze_target_with_context":
            state.add_memory(
                f"gaze_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"gaze_reason={result.get('reason')}")
        if tool_name == "infer_viewpoint_choice":
            state.add_memory(
                f"viewpoint_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"viewpoint_reason={result.get('reason')}")
        if tool_name == "infer_named_fixture_direction":
            state.add_memory(
                f"fixture_direction_best_index={result.get('best_index')} target_match={result.get('target_match')}"
            )
            if result.get("reason"):
                state.add_evidence(f"fixture_direction_reason={result.get('reason')}")
            if result.get("best_index") is not None:
                state.add_memory(f"target_location={result.get('answer')}")
        if tool_name == "infer_visual_mcq":
            state.add_memory(
                f"visual_mcq_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"visual_mcq_reason={result.get('reason')}")
        if tool_name == "infer_temporal_localization_choice":
            state.add_memory(
                f"temporal_localization_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"temporal_localization_reason={result.get('reason')}")
        if tool_name == "infer_ingredient_order_choice":
            state.add_memory(
                f"ingredient_order_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"ingredient_order_reason={result.get('reason')}")
            if result.get("observed_order"):
                state.add_memory(f"ingredient_order_observed={result.get('observed_order')}")
        if tool_name == "infer_ingredient_retrieval_choice":
            state.add_memory(
                f"ingredient_retrieval_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"ingredient_retrieval_reason={result.get('reason')}")
            if result.get("observed_ingredients"):
                state.add_memory(f"ingredient_retrieval_observed={result.get('observed_ingredients')}")
        if tool_name == "infer_recipe_ingredient_membership_choice":
            state.add_memory(
                f"recipe_membership_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"recipe_membership_reason={result.get('reason')}")
        if tool_name == "infer_exact_ingredient_amount_choice":
            state.add_memory(
                f"exact_ingredient_amount_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"exact_ingredient_amount_reason={result.get('reason')}")
        if tool_name == "infer_recipe_catalog_choice":
            state.add_memory(
                f"recipe_catalog_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"recipe_catalog_reason={result.get('reason')}")
        if tool_name == "infer_recipe_nutrition_choice":
            state.add_memory(
                f"recipe_nutrition_best_index={result.get('best_index')} confidence={result.get('confidence')} nutrient={result.get('nutrient')}"
            )
            if result.get("reason"):
                state.add_evidence(f"recipe_nutrition_reason={result.get('reason')}")
        if tool_name == "infer_action_mechanism":
            state.add_memory(
                f"action_mechanism_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"action_mechanism_reason={result.get('reason')}")
        if tool_name == "infer_action_intent":
            state.working_memory = [
                item
                for item in state.working_memory
                if not (isinstance(item, str) and item.startswith("action_intent_need_future_evidence="))
            ]
            state.add_memory(
                f"action_intent_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("second_best_index") is not None:
                state.add_memory(f"action_intent_second_best_index={result.get('second_best_index')}")
            if result.get("need_future_evidence"):
                state.add_memory(
                    f"action_intent_need_future_evidence=1 window_s={result.get('future_window_s')} focus={result.get('followup_focus')}"
                )
                state.add_open_question("need_disambiguating_evidence")
            else:
                state.prune_open_question("need_disambiguating_evidence")
            if result.get("reason"):
                state.add_evidence(f"action_intent_reason={result.get('reason')}")
        if tool_name == "resolve_action_intent_pairwise":
            self._record_action_intent_resolution_state(state, tool_name=tool_name, result=result)
            if result.get("losing_index") is not None:
                state.add_memory(f"action_intent_losing_index={result.get('losing_index')}")
            if result.get("reason"):
                state.add_evidence(f"action_intent_pairwise_reason={result.get('reason')}")
        if tool_name == "resolve_action_intent_future_use":
            self._record_action_intent_resolution_state(state, tool_name=tool_name, result=result)
            if result.get("candidate_indices"):
                state.add_memory(f"action_intent_future_use_candidates={result.get('candidate_indices')}")
            if result.get("decisive_observation"):
                state.add_evidence(f"action_intent_future_use_observation={result.get('decisive_observation')}")
            if result.get("reason"):
                state.add_evidence(f"action_intent_future_use_reason={result.get('reason')}")
        if tool_name == "write_observation":
            node_id = result.get("node_id")
            if node_id:
                state.add_memory(f"writeback={node_id}")
        if tool_name in {
            "write_frame_observation",
            "write_region_observation",
            "write_ocr_reading",
            "write_audio_event",
            "write_timeline_summary",
            "write_state_change",
        }:
            node = result.get("node")
            node_id = result.get("node_id")
            if isinstance(node, dict):
                state.add_node_result(node)
                evidence = self._node_to_evidence(node)
                if evidence:
                    state.add_evidence(evidence)
                    state.add_memory(evidence)
            if node_id:
                state.add_memory(f"writeback={node_id}")

    def _apply_finish(self, state: AgentState, payload: dict[str, Any]) -> None:
        state.final_prediction = int(payload.get("prediction")) if payload.get("prediction") is not None else None
        state.final_answer = str(payload.get("answer") or "")
        state.confidence = float(payload.get("confidence") or 0.0)

    def _maybe_skip_explicit_writeback(self, state: AgentState, decision: PlannerDecision) -> dict[str, Any] | None:
        tool = decision.tool
        args = decision.args if isinstance(decision.args, dict) else {}
        if tool == "write_ocr_reading" and self._has_conflict(state, "conflicting_ocr_readings"):
            reading = str(args.get("reading") or "").strip().lower()
            state.add_memory(f"writeback_skipped type=ocr_reading reason=conflict value={reading}")
            return {"write_skipped": True, "reason": "conflicting_ocr_readings"}
        if tool in {"write_frame_observation", "write_region_observation", "write_timeline_summary"}:
            target_location = self._extract_write_location(args)
            if target_location and self._has_conflict(state, "conflicting_locations"):
                state.add_memory(f"writeback_skipped type=location reason=conflict value={target_location}")
                return {"write_skipped": True, "reason": "conflicting_locations"}
            state_change_hint = self._extract_write_state_hint(args)
            if state_change_hint and self._has_conflict(state, "conflicting_state_observations"):
                state.add_memory(f"writeback_skipped type=state reason=conflict value={state_change_hint}")
                return {"write_skipped": True, "reason": "conflicting_state_observations"}
        if tool == "write_state_change" and self._has_conflict(state, "conflicting_state_observations"):
            after_state = str(args.get("after_state") or args.get("before_state") or "").strip().lower()
            state.add_memory(f"writeback_skipped type=state reason=conflict value={after_state}")
            return {"write_skipped": True, "reason": "conflicting_state_observations"}
        return None

    def _node_to_evidence(self, node: dict[str, Any]) -> str:
        attrs = node.get("attributes", {})
        parts = [
            f"type={node.get('node_type')}",
            f"label={node.get('label')}",
        ]
        if node.get("start_time") is not None:
            end_time = node.get("end_time") if node.get("end_time") is not None else node.get("start_time")
            parts.append(f"time={node.get('start_time'):.3f}-{end_time:.3f}")
        for key in ("text", "label", "object_name", "event_type", "source", "scene_location", "target_object", "target_location", "reading", "summary"):
            value = attrs.get(key)
            if value:
                parts.append(f"{key}={value}")
        obs = attrs.get("observation")
        if isinstance(obs, dict):
            for key in ("scene_location", "ongoing_action", "possible_step", "state_change_hint"):
                value = obs.get(key)
                if value:
                    parts.append(f"{key}={value}")
        return "; ".join(parts)

    def _inspection_summary(self, payload: dict[str, Any]) -> str:
        preferred_keys = (
            "target_object",
            "target_location",
            "ongoing_action",
            "possible_step",
            "state_change_hint",
            "digits",
            "reading",
            "answer_hint",
        )
        parts = [f"{key}={payload[key]}" for key in preferred_keys if payload.get(key)]
        if parts:
            return "inspection; " + "; ".join(parts)
        raw = str(payload.get("raw_output") or "").strip()
        return f"inspection; raw={raw[:300]}" if raw else ""

    def _seed_reusable_memory(self, state: AgentState, hints: dict[str, Any]) -> None:
        times = [float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []]
        start_time = max(0.0, min(times) - 3.0) if times else None
        end_time = max(times) + 3.0 if times else None
        payload = self.toolbox.query_time(start_time=start_time, end_time=end_time, limit=24)
        nodes = payload.get("nodes") if isinstance(payload, dict) else []
        if not isinstance(nodes, list):
            return
        reusable_types = {"timeline_event", "observation", "ocr_reading", "state_change", "region", "audio_event"}
        seed_node_ids: list[str] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("node_type") or "") not in reusable_types:
                continue
            if not self._allow_reuse_node(state, node):
                continue
            if not self._allow_cached_artifact_for_task(state, node):
                continue
            state.add_node_result(node)
            node_id = str(node.get("node_id") or "")
            if node_id:
                seed_node_ids.append(node_id)
            evidence = self._node_to_evidence(node)
            if evidence:
                state.add_evidence(evidence)
                state.add_memory(f"reuse:{evidence}")
            self._seed_cached_artifact_from_node(state, node)
        if not seed_node_ids:
            return
        related = self.toolbox.expand_graph_context(node_ids=seed_node_ids[:12], limit=24)
        related_nodes = related.get("nodes") if isinstance(related, dict) else []
        related_edges = related.get("edges") if isinstance(related, dict) else []
        state.add_memory(f"relation_seed_anchor_count={len(seed_node_ids[:12])}")
        if isinstance(related_nodes, list):
            state.add_memory(f"relation_seed_expanded_node_count={len(related_nodes)}")
            for node in related_nodes:
                if not isinstance(node, dict):
                    continue
                if str(node.get("node_type") or "") not in reusable_types:
                    continue
                if not self._allow_reuse_node(state, node):
                    continue
                if not self._allow_cached_artifact_for_task(state, node):
                    continue
                state.add_node_result(node)
                evidence = self._node_to_evidence(node)
                if evidence:
                    state.add_evidence(evidence)
                    state.add_memory(f"reuse_relation:{evidence}")
                self._seed_cached_artifact_from_node(state, node)
        if isinstance(related_edges, list):
            state.add_memory(f"relation_seed_expanded_edge_count={len(related_edges)}")
            for edge in related_edges[:12]:
                if not isinstance(edge, dict):
                    continue
                edge_type = str(edge.get("edge_type") or "")
                source_id = str(edge.get("source_id") or "")
                target_id = str(edge.get("target_id") or "")
                if edge_type and source_id and target_id:
                    state.add_memory(f"reuse_relation_edge:{edge_type}:{source_id}->{target_id}")

    def _allow_reuse_node(self, state: AgentState, node: dict[str, Any]) -> bool:
        if self._is_action_intent_task(state):
            label = str(node.get("label") or "").strip().lower()
            attrs = node.get("attributes") or {}
            source = str(attrs.get("source") or "").strip().lower()
            summary = str(attrs.get("summary") or "").strip().lower()
            if label.startswith("session summary ") or label.startswith("compressed session memory "):
                return False
            if source in {"session_memory_compressor", "agent_timeline_summary"}:
                return False
            if any(
                token in summary
                for token in (
                    "action_intent_",
                    "visual_mcq_reason=",
                    "answer_hint=",
                    "candidate_answer_index=",
                )
            ):
                return False
        if not self._is_location_conclusion_sensitive_task(state):
            return True
        label = str(node.get("label") or "").strip().lower()
        if label.startswith("session summary ") or label.startswith("compressed session memory "):
            return False
        attrs = node.get("attributes") or {}
        source = str(attrs.get("source") or "").strip().lower()
        summary = str(attrs.get("summary") or "").strip().lower()
        if source in {"session_memory_compressor", "agent_timeline_summary"}:
            if "target_location=" in summary or "scene_location=" in summary:
                return False
            if "fixture_direction_reason=" in summary or "gaze_reason=" in summary:
                return False
        if source == "session_memory_compressor":
            return False
        return True

    def _seed_cached_artifact_from_node(self, state: AgentState, node: dict[str, Any]) -> None:
        attrs = node.get("attributes") or {}
        source = str(attrs.get("source") or "").strip().lower()
        if source != "cached_artifact_reuse":
            return
        artifact_path = str(attrs.get("artifact_path") or "").strip()
        if not artifact_path:
            evidence_paths = node.get("evidence_paths") or []
            artifact_path = str(evidence_paths[0]).strip() if evidence_paths else ""
        prefixes = tuple(token.lower() for token in artifact_reuse_prefixes_for_task(str(getattr(state, "task_family", "") or "")))
        if artifact_path and prefixes and not any(prefix in artifact_path.lower() for prefix in prefixes):
            return
        if artifact_path:
            state.add_artifact(artifact_path)
            state.add_memory(f"reuse_cached_artifact={artifact_path}")
        artifact_tag = str(attrs.get("artifact_tag") or "").strip()
        if artifact_tag:
            state.add_memory(f"reuse_cached_artifact_tag={artifact_tag}")

    def _allow_cached_artifact_for_task(self, state: AgentState, node: dict[str, Any]) -> bool:
        attrs = node.get("attributes") or {}
        source = str(attrs.get("source") or "").strip().lower()
        if source != "cached_artifact_reuse":
            return True
        artifact_path = str(attrs.get("artifact_path") or "").strip()
        if not artifact_path:
            evidence_paths = node.get("evidence_paths") or []
            artifact_path = str(evidence_paths[0]).strip() if evidence_paths else ""
        if not artifact_path:
            return True
        prefixes = tuple(token.lower() for token in artifact_reuse_prefixes_for_task(str(getattr(state, "task_family", "") or "")))
        if not prefixes:
            return True
        lowered = artifact_path.lower()
        return any(prefix in lowered for prefix in prefixes)

    def _auto_write_visual_observation(self, state: AgentState, payload: dict[str, Any]) -> None:
        label = str(
            payload.get("possible_step")
            or payload.get("ongoing_action")
            or payload.get("target_object")
            or "agent visual observation"
        ).strip()
        if not label:
            return
        attributes = {
            key: payload.get(key)
            for key in ("ongoing_action", "possible_step", "target_object", "target_location", "state_change_hint", "answer_hint", "raw_output")
            if payload.get(key)
        }
        image_path = state.retrieved_frames[-1] if state.retrieved_frames else ""
        time_s = self._infer_time_from_artifact(image_path)
        keywords = self._keywords_from_strings([label, *attributes.values()])
        if self._should_skip_visual_writeback(state, payload):
            return
        if len(state.retrieved_frames) >= 2:
            summary = "; ".join(f"{key}={value}" for key, value in attributes.items()) or label
            result = self.toolbox.write_timeline_summary(
                label=label,
                start_time=time_s,
                end_time=time_s,
                summary=summary,
                evidence_paths=state.retrieved_frames[-12:],
                keywords=keywords,
                source_tool="inspect_visual_evidence",
                confidence=self._extract_confidence(payload),
            )
            self._merge_result_into_state(state, "write_timeline_summary", result)
            return
        if image_path:
            result = self.toolbox.write_frame_observation(
                frame_path=image_path,
                time_s=time_s,
                label=label,
                observation=attributes,
                keywords=keywords,
                source_tool="inspect_visual_evidence",
                confidence=self._extract_confidence(payload),
            )
            self._merge_result_into_state(state, "write_frame_observation", result)

    def _auto_write_ocr_observation(self, state: AgentState, tool_name: str, payload: dict[str, Any]) -> None:
        reading = str(payload.get("reading") or "").strip()
        if not reading:
            return
        if self._has_conflict(state, "conflicting_ocr_readings"):
            state.add_memory(f"writeback_skipped type=ocr_reading reason=conflict value={reading}")
            return
        last_trace = state.tool_trace[-1] if state.tool_trace else {}
        args = last_trace.get("args") if isinstance(last_trace, dict) else {}
        image_path = str(payload.get("artifact_path") or (args.get("image_path") if isinstance(args, dict) else "") or (state.retrieved_frames[-1] if state.retrieved_frames else ""))
        bbox = args.get("bbox") if isinstance(args, dict) else None
        time_s = self._infer_time_from_artifact(image_path)
        result = self.toolbox.write_ocr_reading(
            label="ocr reading",
            reading=reading,
            time_s=time_s,
            image_path=image_path or None,
            bbox=bbox,
            attributes={"text": payload.get("text"), "source_tool": tool_name},
            keywords=self._keywords_from_strings([reading, payload.get("text")]),
            source_tool=tool_name,
            confidence=self._extract_confidence(payload),
        )
        self._merge_result_into_state(state, "write_ocr_reading", result)

    def _auto_write_audio_peaks(self, state: AgentState, peaks: list[dict[str, Any]]) -> None:
        for peak in peaks[:3]:
            if not isinstance(peak, dict) or peak.get("time_s") is None:
                continue
            result = self.toolbox.write_audio_event(
                label=f"audio peak {float(peak['time_s']):.3f}s",
                start_time=float(peak.get("window_start") or peak["time_s"]),
                end_time=float(peak.get("window_end") or peak["time_s"]),
                attributes={"score": peak.get("score"), "peak_time": peak.get("time_s")},
                evidence_paths=[],
                keywords=["audio_peak"],
                source_tool="detect_audio_peaks",
                confidence=self._extract_confidence(peak),
            )
            self._merge_result_into_state(state, "write_audio_event", result)

    def _auto_write_cached_artifact_reuse(self, state: AgentState, items: list[dict[str, Any]]) -> None:
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            artifact_path = str(item.get("artifact_path") or "").strip()
            if not artifact_path:
                continue
            time_s = item.get("time_s")
            tag = str(item.get("tag") or Path(artifact_path).stem).strip()
            result = self.toolbox.write_observation(
                label=f"cached artifact reuse {tag}",
                start_time=float(time_s) if time_s is not None else None,
                end_time=float(time_s) if time_s is not None else None,
                attributes={
                    "artifact_path": artifact_path,
                    "artifact_tag": tag,
                    "source": "cached_artifact_reuse",
                },
                evidence_paths=[artifact_path],
                keywords=self._keywords_from_strings([tag, artifact_path, state.task_family, "cached_artifact_reuse"]),
                source_tool="retrieve_cached_artifacts",
                confidence=0.6,
            )
            self._merge_result_into_state(state, "write_observation", result)

    def _infer_time_from_artifact(self, path: str) -> float | None:
        if not path:
            return None
        match = re.search(r"_(\d+\.\d+)s\.(?:jpg|jpeg|png|webp)$", path)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _extract_confidence(self, payload: dict[str, Any]) -> float | None:
        value = payload.get("confidence")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    def _record_result_times(self, state: AgentState, result: dict[str, Any]) -> None:
        scalar_keys = ("start_time", "end_time", "time_s", "reference_time")
        for key in scalar_keys:
            if result.get(key) is not None:
                state.add_visited_time(result.get(key))
        for list_key in ("times", "peak_times"):
            values = result.get(list_key)
            if isinstance(values, list):
                for item in values:
                    state.add_visited_time(item)
        peaks = result.get("peaks")
        if isinstance(peaks, list):
            for item in peaks:
                if not isinstance(item, dict):
                    continue
                for key in ("time_s", "window_start", "window_end"):
                    if item.get(key) is not None:
                        state.add_visited_time(item.get(key))
        items = result.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                for key in ("time_s", "start_time", "end_time"):
                    if item.get(key) is not None:
                        state.add_visited_time(item.get(key))

    def _extract_write_location(self, args: dict[str, Any]) -> str:
        observation = args.get("observation")
        if isinstance(observation, dict):
            value = observation.get("target_location") or observation.get("scene_location")
            if value:
                return str(value).strip().lower()
        summary = str(args.get("summary") or "").strip().lower()
        for marker in ("target_location=", "scene_location="):
            if marker in summary:
                tail = summary.split(marker, 1)[1]
                for separator in (";", "|"):
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                return tail.strip().lower()
        return ""

    def _extract_write_state_hint(self, args: dict[str, Any]) -> str:
        observation = args.get("observation")
        if isinstance(observation, dict):
            value = observation.get("state_change_hint")
            if value:
                return str(value).strip().lower()
        summary = str(args.get("summary") or "").strip().lower()
        if "state_change_hint=" in summary:
            tail = summary.split("state_change_hint=", 1)[1]
            for separator in (";", "|"):
                if separator in tail:
                    tail = tail.split(separator, 1)[0]
            return tail.strip().lower()
        return ""

    def _record_semantic_conflicts_from_payload(self, state: AgentState, payload: dict[str, Any]) -> None:
        reading = payload.get("reading")
        if reading:
            self._record_conflict_if_needed(
                state,
                conflict_type="conflicting_ocr_readings",
                new_value=str(reading),
                prefixes=("ocr_reading=",),
            )
        target_location = payload.get("target_location")
        if target_location:
            self._record_conflict_if_needed(
                state,
                conflict_type="conflicting_locations",
                new_value=str(target_location),
                prefixes=("target_location=", "scene_location="),
            )
            self._dedupe_conflicting_prefixed_entries(
                state,
                prefixes=("target_location=", "scene_location="),
                keep_value=str(target_location),
            )
        state_change_hint = payload.get("state_change_hint")
        if state_change_hint:
            self._record_conflict_if_needed(
                state,
                conflict_type="conflicting_state_observations",
                new_value=str(state_change_hint),
                prefixes=("state_change_hint=", "after_state=", "before_state="),
            )
            self._dedupe_conflicting_prefixed_entries(
                state,
                prefixes=("state_change_hint=", "after_state=", "before_state="),
                keep_value=str(state_change_hint),
            )

    def _record_conflict_if_needed(
        self,
        state: AgentState,
        *,
        conflict_type: str,
        new_value: str,
        prefixes: tuple[str, ...],
    ) -> None:
        normalized_new = new_value.strip().lower()
        if not normalized_new:
            return
        existing = self._existing_prefixed_values(state, prefixes=prefixes)
        conflicting = sorted(value for value in existing if value and value != normalized_new)
        if not conflicting:
            return
        state.add_open_question(f"conflict:{conflict_type}")
        state.add_memory(
            f"conflict_hint type={conflict_type} existing={conflicting[0]} new={normalized_new}"
        )
        state.add_hypothesis(f"conflict_detected={conflict_type}")

    def _dedupe_conflicting_prefixed_entries(
        self,
        state: AgentState,
        *,
        prefixes: tuple[str, ...],
        keep_value: str,
    ) -> None:
        normalized_keep = keep_value.strip().lower()
        if not normalized_keep:
            return
        state.evidence_bundle = self._filter_prefixed_entries(state.evidence_bundle, prefixes=prefixes, keep_value=normalized_keep)
        state.working_memory = self._filter_prefixed_entries(state.working_memory, prefixes=prefixes, keep_value=normalized_keep)

    def _filter_prefixed_entries(self, items: list[str], *, prefixes: tuple[str, ...], keep_value: str) -> list[str]:
        filtered: list[str] = []
        for item in items:
            if not isinstance(item, str):
                filtered.append(item)
                continue
            matched = False
            for prefix in prefixes:
                if prefix not in item:
                    continue
                matched = True
                tail = item.split(prefix, 1)[1]
                for separator in (";", "|"):
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                if tail.strip().lower() == keep_value:
                    filtered.append(item)
                break
            if not matched:
                filtered.append(item)
        return filtered

    def _should_skip_visual_writeback(self, state: AgentState, payload: dict[str, Any]) -> bool:
        target_location = str(payload.get("target_location") or "").strip()
        if target_location and self._has_conflict(state, "conflicting_locations"):
            state.add_memory(f"writeback_skipped type=location reason=conflict value={target_location.lower()}")
            return True
        state_change_hint = str(payload.get("state_change_hint") or "").strip()
        if state_change_hint and self._has_conflict(state, "conflicting_state_observations"):
            state.add_memory(f"writeback_skipped type=state reason=conflict value={state_change_hint.lower()}")
            return True
        return False

    def _has_conflict(self, state: AgentState, conflict_type: str) -> bool:
        target = f"conflict:{conflict_type}"
        return any(item == target for item in state.open_questions if isinstance(item, str))

    def _existing_prefixed_values(self, state: AgentState, *, prefixes: tuple[str, ...]) -> set[str]:
        values: set[str] = set()
        for item in state.evidence_bundle + state.working_memory:
            if not isinstance(item, str):
                continue
            for prefix in prefixes:
                if prefix not in item:
                    continue
                tail = item.split(prefix, 1)[1]
                for separator in (";", "|"):
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                normalized = tail.strip().lower()
                if normalized:
                    values.add(normalized)
        return values

    def _reconcile_conflict_questions(self, state: AgentState) -> None:
        detect_conflicts = getattr(self.verifier, "detect_conflicts", None)
        if callable(detect_conflicts):
            current_conflicts = set(detect_conflicts(state=state))
        else:
            current_conflicts = set()
        existing_conflict_questions = [
            str(item).split("conflict:", 1)[1]
            for item in state.open_questions
            if isinstance(item, str) and item.startswith("conflict:")
        ]
        for conflict in existing_conflict_questions:
            if conflict not in current_conflicts:
                state.prune_open_question(f"conflict:{conflict}")
                state.add_memory(f"conflict_resolved={conflict}")
        for conflict in current_conflicts:
            state.add_open_question(f"conflict:{conflict}")

    def _keywords_from_strings(self, values: list[Any]) -> list[str]:
        tokens: set[str] = set()
        for value in values:
            if value is None:
                continue
            text = str(value).strip().lower()
            if not text:
                continue
            tokens.add(text)
            for part in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text):
                if len(part) >= 2:
                    tokens.add(part)
        return sorted(tokens)

    def _summarize(self, result: Any) -> str:
        if isinstance(result, dict):
            if isinstance(result.get("nodes"), list):
                labels = [str(item.get("label") or item.get("node_id")) for item in result["nodes"][:3] if isinstance(item, dict)]
                return f"nodes={len(result['nodes'])} preview={labels}"
            if isinstance(result.get("edges"), list):
                labels = [str(item.get("edge_type")) for item in result["edges"][:3] if isinstance(item, dict)]
                return f"edges={len(result['edges'])} preview={labels}"
            if result.get("artifact_path"):
                return f"artifact={result['artifact_path']}"
            if result.get("artifact_paths"):
                return f"artifacts={len(result['artifact_paths'])}"
        return json.dumps(result, ensure_ascii=False)[:200]
