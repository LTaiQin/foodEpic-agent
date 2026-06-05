"""Looping executor for the complete graph/video agent."""

from __future__ import annotations

import json
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
        for step_index in range(state.max_steps):
            state.current_step = step_index
            decision = self.planner.next_action(state=state, tool_schemas=self.toolbox.tool_schemas(), hints=hints)
            state.plan_summary = decision.thought
            if decision.done and decision.tool == "finish":
                finish_payload = self.toolbox.finish(**decision.args)
                self._apply_finish(state, finish_payload)
                state.record_tool("finish", decision.args, self._summarize(finish_payload), raw_result=finish_payload)
                break
            result = self.toolbox.run(decision.tool, decision.args)
            self._apply_tool_result(state, decision, result)
            if result.get("done"):
                self._apply_finish(state, result)
                break
        return state

    def _apply_tool_result(self, state: AgentState, decision: PlannerDecision, result: dict[str, Any]) -> None:
        state.record_tool(decision.tool, decision.args, self._summarize(result), raw_result=result)
        self._merge_result_into_state(state, decision.tool, result)

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
        for key in ("text", "label", "object_name", "event_type", "source", "scene_location", "target_object", "target_location"):
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
