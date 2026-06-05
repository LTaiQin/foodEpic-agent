"""Looping executor for the complete graph/video agent."""

from __future__ import annotations

import json
import re
from typing import Any

from food_agent.agent.planner import GraphAgentPlanner, PlannerDecision
from food_agent.agent.state import AgentState
from food_agent.tools import AgentToolbox


class GraphAgentExecutor:
    def __init__(self, toolbox: AgentToolbox, planner: GraphAgentPlanner):
        self.toolbox = toolbox
        self.planner = planner

    def execute(self, state: AgentState) -> AgentState:
        self.toolbox.set_runtime_context(question=state.question, inputs_json=state.inputs_json)
        hints = self.toolbox.default_hints(state.question, state.inputs_json)
        self._seed_reusable_memory(state, hints)
        self._initialize_reasoning_state(state, hints)
        for step_index in range(state.max_steps):
            state.current_step = step_index
            self._refresh_open_questions_before_planning(state)
            decision = self.planner.next_action(state=state, tool_schemas=self.toolbox.tool_schemas(), hints=hints)
            state.plan_summary = decision.thought
            self._record_planner_reflection(state, decision)
            if decision.done and decision.tool == "finish":
                finish_payload = self.toolbox.finish(**decision.args)
                self._apply_finish(state, finish_payload)
                state.record_tool("finish", decision.args, self._summarize(finish_payload), raw_result=finish_payload)
                break
            try:
                result = self.toolbox.run(decision.tool, decision.args)
            except Exception as exc:  # noqa: BLE001
                self._handle_tool_failure(state, decision, exc)
                continue
            self._apply_tool_result(state, decision, result)
            if result.get("done"):
                self._apply_finish(state, result)
                break
        return state

    def _apply_tool_result(self, state: AgentState, decision: PlannerDecision, result: dict[str, Any]) -> None:
        state.record_tool(decision.tool, decision.args, self._summarize(result), raw_result=result)
        self._merge_result_into_state(state, decision.tool, result)
        self._update_reasoning_after_tool(state, decision.tool, result)

    def _initialize_reasoning_state(self, state: AgentState, hints: dict[str, Any]) -> None:
        state.add_hypothesis(f"task_family={state.task_family}")
        if hints.get("times") or hints.get("input_times"):
            state.add_open_question("need_time_localization")
        if hints.get("bbox"):
            state.add_open_question("need_region_grounding")
        if hints.get("ocr_keyword"):
            state.add_open_question("need_ocr_reading")
        if hints.get("state_keyword"):
            state.add_open_question("need_state_evidence")
        if hints.get("location_keyword"):
            state.add_open_question("need_location_evidence")
        if not state.open_questions:
            state.add_open_question("need_disambiguating_evidence")

    def _refresh_open_questions_before_planning(self, state: AgentState) -> None:
        refreshed: list[str] = []
        if not state.evidence_bundle:
            refreshed.append("need_disambiguating_evidence")
        if not any(item.startswith("ocr_reading=") for item in state.working_memory):
            if any(token in state.question.lower() for token in ("weight", "gram", "grams", "read", "number", "digit")):
                refreshed.append("need_ocr_reading")
        if not any("target_location=" in item or "scene_location=" in item for item in state.evidence_bundle + state.working_memory):
            if any(token in state.question.lower() for token in ("where", "location", "left", "right", "front", "behind")):
                refreshed.append("need_location_evidence")
        if not any("state_change_hint=" in item or "type=state_change" in item for item in state.evidence_bundle + state.working_memory):
            if any(token in state.question.lower() for token in ("state", "become", "change", "cooked", "mixed", "done")):
                refreshed.append("need_state_evidence")
        if state.question.lower() and not state.retrieved_frames and not state.retrieved_nodes:
            refreshed.append("need_initial_observation")
        if refreshed:
            merged = state.open_questions + [item for item in refreshed if item not in state.open_questions]
            state.replace_open_questions(merged)

    def _record_planner_reflection(self, state: AgentState, decision: PlannerDecision) -> None:
        if decision.thought:
            state.add_memory(f"planner_thought={decision.thought}")
        if decision.tool:
            state.add_hypothesis(f"plan_step={state.current_step}; tool={decision.tool}")
        tool = decision.tool
        if tool in {"query_time", "sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks"}:
            state.prune_open_question("need_time_localization")
            state.prune_open_question("need_initial_observation")
        if tool in {"query_region", "render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference"}:
            state.prune_open_question("need_region_grounding")
        if tool in {"query_ocr", "run_ocr_on_image", "run_ocr_on_region"}:
            state.prune_open_question("need_ocr_reading")
        if tool in {"query_state", "write_state_change", "inspect_visual_evidence"}:
            state.prune_open_question("need_state_evidence")
        if tool in {"query_location", "infer_viewpoint_choice", "infer_named_fixture_direction", "infer_gaze_target_with_context"}:
            state.prune_open_question("need_location_evidence")

    def _update_reasoning_after_tool(self, state: AgentState, tool_name: str, result: dict[str, Any]) -> None:
        if result.get("nodes") or result.get("matches") or result.get("totals") or result.get("artifact_path") or result.get("artifact_paths"):
            state.prune_open_question("need_disambiguating_evidence")
        if tool_name in {"query_time", "sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks"}:
            if state.retrieved_frames or state.retrieved_nodes:
                state.prune_open_question("need_time_localization")
                state.prune_open_question("need_initial_observation")
        if tool_name in {"query_region", "render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference"}:
            if state.retrieved_frames or result.get("association_id") or result.get("tracks"):
                state.prune_open_question("need_region_grounding")
        if tool_name in {"query_ocr", "run_ocr_on_image", "run_ocr_on_region"}:
            if result.get("reading") or result.get("text") or any(item.startswith("ocr_reading=") for item in state.working_memory):
                state.prune_open_question("need_ocr_reading")
                state.add_hypothesis("ocr_evidence_collected")
        if tool_name in {"query_state", "inspect_visual_evidence", "write_state_change"}:
            if any("state_change_hint=" in item for item in state.evidence_bundle + state.working_memory):
                state.prune_open_question("need_state_evidence")
                state.add_hypothesis("state_evidence_collected")
        if tool_name in {"query_location", "infer_viewpoint_choice", "infer_named_fixture_direction", "infer_gaze_target_with_context"}:
            if any("target_location=" in item or "scene_location=" in item for item in state.evidence_bundle + state.working_memory):
                state.prune_open_question("need_location_evidence")
                state.add_hypothesis("location_evidence_collected")
        if tool_name == "rank_choices_from_state" and result.get("best_index") is not None:
            state.add_hypothesis(f"candidate_answer_index={result.get('best_index')}")
        if tool_name == "finish" or result.get("done"):
            state.replace_open_questions([])

    def _handle_tool_failure(self, state: AgentState, decision: PlannerDecision, exc: Exception) -> None:
        error_type = type(exc).__name__
        error_message = str(exc)
        state.record_tool_failure(decision.tool, decision.args, error_type, error_message)
        state.add_memory(f"tool_failure tool={decision.tool} error_type={error_type}")
        state.add_hypothesis(f"failed_tool={decision.tool}")
        state.add_open_question("need_alternative_evidence_path")
        if decision.tool in {"run_ocr_on_image", "run_ocr_on_region", "query_ocr"}:
            state.add_open_question("need_ocr_reading")
        if decision.tool in {"render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference", "query_region"}:
            state.add_open_question("need_region_grounding")
        if decision.tool in {"query_state", "inspect_visual_evidence", "write_state_change"}:
            state.add_open_question("need_state_evidence")
        if decision.tool in {"query_location", "infer_viewpoint_choice", "infer_named_fixture_direction", "infer_gaze_target_with_context"}:
            state.add_open_question("need_location_evidence")
        if decision.tool in {"query_time", "sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks"}:
            state.add_open_question("need_time_localization")

    def _merge_result_into_state(self, state: AgentState, tool_name: str, result: dict[str, Any]) -> None:
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
            if artifact_path not in state.retrieved_frames:
                state.retrieved_frames.append(artifact_path)
            state.add_memory(f"artifact={artifact_path}")
        artifact_paths = result.get("artifact_paths")
        if isinstance(artifact_paths, list):
            for path in artifact_paths:
                if isinstance(path, str) and path and path not in state.retrieved_frames:
                    state.retrieved_frames.append(path)
                    state.add_memory(f"artifact={path}")
        if tool_name == "inspect_visual_evidence":
            summary = self._inspection_summary(result)
            if summary:
                state.add_evidence(summary)
                state.add_memory(summary)
            self._auto_write_visual_observation(state, result)
        if tool_name in {"run_ocr_on_image", "run_ocr_on_region"}:
            reading = result.get("reading")
            text = result.get("text")
            if reading:
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
        if tool_name == "infer_visual_mcq":
            state.add_memory(
                f"visual_mcq_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"visual_mcq_reason={result.get('reason')}")
        if tool_name == "infer_action_mechanism":
            state.add_memory(
                f"action_mechanism_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"action_mechanism_reason={result.get('reason')}")
        if tool_name == "infer_action_intent":
            state.add_memory(
                f"action_intent_best_index={result.get('best_index')} confidence={result.get('confidence')}"
            )
            if result.get("reason"):
                state.add_evidence(f"action_intent_reason={result.get('reason')}")
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
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("node_type") or "") not in reusable_types:
                continue
            state.add_node_result(node)
            evidence = self._node_to_evidence(node)
            if evidence:
                state.add_evidence(evidence)
                state.add_memory(f"reuse:{evidence}")

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
        if len(state.retrieved_frames) >= 2:
            summary = "; ".join(f"{key}={value}" for key, value in attributes.items()) or label
            result = self.toolbox.write_timeline_summary(
                label=label,
                start_time=time_s,
                end_time=time_s,
                summary=summary,
                evidence_paths=state.retrieved_frames[-12:],
                keywords=keywords,
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
            )
            self._merge_result_into_state(state, "write_frame_observation", result)

    def _auto_write_ocr_observation(self, state: AgentState, tool_name: str, payload: dict[str, Any]) -> None:
        reading = str(payload.get("reading") or "").strip()
        if not reading:
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
            )
            self._merge_result_into_state(state, "write_audio_event", result)

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
