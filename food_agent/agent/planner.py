"""LLM planner for multi-step graph/video tool calling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from food_agent.agent.state import AgentState
from food_agent.model_client import OpenAICompatibleModelClient


@dataclass(frozen=True)
class PlannerDecision:
    thought: str
    tool: str
    args: dict[str, Any]
    done: bool = False
    answer: str = ""
    prediction: int | None = None
    confidence: float = 0.0


class GraphAgentPlanner:
    """Use the model to decide the next tool call instead of hard-coded routing."""

    def __init__(self, model_client: OpenAICompatibleModelClient):
        self.model_client = model_client

    def next_action(self, *, state: AgentState, tool_schemas: list[dict[str, Any]], hints: dict[str, Any]) -> PlannerDecision:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个真实的视频问答 agent 规划器。"
                    "你不能直接假设答案，必须先决定是否需要调用工具。"
                    "你只能基于当前工作记忆、图谱证据和工具返回来决策。"
                    "如果证据不够，就继续调工具；如果证据足够，再调用 finish。"
                    "严格输出 JSON 对象，不要输出 markdown。"
                    'JSON 字段固定为 {"thought":"","tool":"","args":{},"done":false,"answer":"","prediction":null,"confidence":0.0}。'
                ),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(state=state, tool_schemas=tool_schemas, hints=hints),
            },
        ]
        try:
            payload = self.model_client.complete_json(messages, temperature=0.0)
            decision = self._payload_to_decision(payload)
        except Exception:  # noqa: BLE001
            decision = self._heuristic_fallback(state=state, hints=hints)
        return self._enforce_task_requirements(state=state, hints=hints, decision=decision)

    def _build_user_prompt(self, *, state: AgentState, tool_schemas: list[dict[str, Any]], hints: dict[str, Any]) -> str:
        prompt = {
            "video_id": state.video_id,
            "task_family": state.task_family,
            "question": state.question,
            "choices": state.choices,
            "current_step": state.current_step,
            "max_steps": state.max_steps,
            "parsed_hints": hints,
            "tool_schemas": tool_schemas,
            "working_memory": state.snapshot(),
            "last_tool_result": state.tool_trace[-1] if state.tool_trace else None,
            "instruction": (
                "先判断当前最缺什么证据，再选择一个最合适的工具。"
                "优先低成本检索；只有图谱证据不够时才抽帧、画框、放大或看图。"
                "如果已经足够区分答案，调用 finish。"
            ),
        }
        return json.dumps(prompt, ensure_ascii=False, indent=2)

    def _payload_to_decision(self, payload: dict[str, Any]) -> PlannerDecision:
        tool = str(payload.get("tool") or "").strip()
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        prediction = payload.get("prediction")
        try:
            prediction = None if prediction is None else int(prediction)
        except Exception:  # noqa: BLE001
            prediction = None
        return PlannerDecision(
            thought=str(payload.get("thought") or ""),
            tool=tool,
            args=args,
            done=bool(payload.get("done")) or tool == "finish",
            answer=str(payload.get("answer") or ""),
            prediction=prediction,
            confidence=float(payload.get("confidence") or 0.0),
        )

    def _heuristic_fallback(self, *, state: AgentState, hints: dict[str, Any]) -> PlannerDecision:
        last_tool = state.tool_trace[-1] if state.tool_trace else {}
        last_result = last_tool.get("raw_result") if isinstance(last_tool, dict) else {}
        if isinstance(last_result, dict) and last_tool.get("tool") == "count_visual_candidates" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="视觉计数已完成，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "infer_viewpoint_choice" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="视角定位已完成，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "infer_named_fixture_direction" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="具名 fixture 方位定位已完成，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "infer_visual_mcq" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="片段视觉多选判断已完成，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "infer_action_mechanism" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="动作机制判断已完成，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "infer_action_intent" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="动作目的判断已完成，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "rank_choices_from_state" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="已经有选项评分结果，直接结束。",
                tool="finish",
                args={
                    "prediction": best_index,
                    "answer": str(last_result.get("answer") or state.choices[best_index]),
                    "confidence": float(last_result.get("confidence") or 0.0),
                },
                done=True,
                answer=str(last_result.get("answer") or state.choices[best_index]),
                prediction=best_index,
                confidence=float(last_result.get("confidence") or 0.0),
            )
        times = [float(value) for value in hints.get("times") or []]
        input_times = [float(value) for value in hints.get("input_times") or []]
        combined_times = sorted(times + input_times)
        bbox = hints.get("bbox")
        ingredient_name = hints.get("ingredient_name")
        if state.current_step == 0 and combined_times:
            return PlannerDecision(
                thought="先查题目时间窗口附近的图谱节点。",
                tool="query_time",
                args={"start_time": min(combined_times), "end_time": max(combined_times), "limit": 20},
            )
        if state.current_step <= 1 and state.task_family.startswith("ingredient_"):
            if state.task_family == "ingredient_ingredient_weight" and combined_times:
                if state.current_step == 1 and ingredient_name:
                    return PlannerDecision(
                        thought="称重题先查图谱中的 ingredient weigh 记录。",
                        tool="query_ingredient_measurement",
                        args={
                            "ingredient_name": str(ingredient_name),
                            "start_time": min(combined_times),
                            "end_time": max(combined_times),
                            "limit": 10,
                        },
                    )
                if state.current_step == 2:
                    return PlannerDecision(
                        thought="先根据称量记录对候选重量评分。",
                        tool="rank_choices_from_state",
                        args={
                            "question": state.question,
                            "choices": [str(choice) for choice in state.choices],
                            "evidence": state.evidence_bundle,
                            "working_memory": state.working_memory,
                        },
                    )
                return PlannerDecision(
                    thought="称重题先回看称量时间段的原始视频。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": max(0.0, min(combined_times) - 2.0),
                        "end_time": max(combined_times) + 2.0,
                        "stride_s": 1.0,
                        "max_frames": 5,
                        "tag": f"{state.task_family}_range",
                    },
                )
            return PlannerDecision(
                thought="食材题优先检索 ingredient_event。",
                tool="query_event",
                args={
                    "event_types": ["ingredient_event"],
                    "keyword": "ingredient",
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 20,
                },
            )
        if state.current_step <= 1 and state.task_family.startswith("recipe_"):
            return PlannerDecision(
                thought="步骤题优先检索 recipe_step。",
                tool="query_event",
                args={"event_types": ["recipe_step"], "start_time": min(combined_times) if combined_times else None, "end_time": max(combined_times) if combined_times else None, "limit": 20},
            )
        if state.current_step == 1 and state.task_family.startswith("recipe_"):
            return PlannerDecision(
                thought="根据 recipe_step 证据对候选时间选项进行评分。",
                tool="rank_choices_from_state",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "evidence": state.evidence_bundle,
                    "working_memory": state.working_memory,
                },
            )
        if state.current_step <= 2 and state.task_family == "ingredient_ingredient_weight" and state.retrieved_frames:
            return PlannerDecision(
                thought="让视觉工具查看称重图片，尝试读取数字和食材。",
                tool="inspect_visual_evidence",
                args={
                    "prompt": (
                        "你在看厨房称重过程的若干图像。"
                        "请识别正在称量的食材和可能的数字读数。"
                        '输出 JSON，字段固定为 {"ongoing_action":"","reading":"","digits":"","answer_hint":"","confidence":0.0}。'
                    ),
                    "image_paths": state.retrieved_frames[-5:],
                },
            )
        if state.current_step == 3 and state.task_family == "ingredient_ingredient_weight":
            return PlannerDecision(
                thought="基于称重证据对候选重量选项评分。",
                tool="rank_choices_from_state",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "evidence": state.evidence_bundle,
                    "working_memory": state.working_memory,
                },
            )
        if state.current_step == 1 and bbox and state.task_family.startswith(("object_motion_", "3d_perception_", "gaze_")) and combined_times:
            return PlannerDecision(
                thought="先在目标参考时刻抽帧。",
                tool="extract_frame_at_time",
                args={"time_s": combined_times[0], "tag": f"{state.task_family}_anchor"},
            )
        if state.current_step == 1 and state.task_family in {"3d_perception_fixture_location", "gaze_gaze_estimation"} and combined_times and not state.retrieved_frames:
            return PlannerDecision(
                thought="先抽取当前视角关键帧。",
                tool="extract_frames_for_range",
                args={
                    "start_time": max(0.0, min(combined_times) - 0.5),
                    "end_time": max(combined_times) + 0.5,
                    "stride_s": 0.5,
                    "max_frames": 3,
                    "tag": f"{state.task_family}_view",
                },
            )
        if state.current_step == 2 and state.task_family in {"3d_perception_fixture_location", "gaze_gaze_estimation"} and state.retrieved_frames:
            if state.task_family == "3d_perception_fixture_location":
                return PlannerDecision(
                    thought="先识别具名 fixture 在当前厨房语境里最可能对应什么，再映射方向。",
                    tool="infer_named_fixture_direction",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                    },
                )
            return PlannerDecision(
                thought="直接根据视角图像在方位/注视目标选项中做视觉定位判断。",
                tool="infer_viewpoint_choice",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-3:],
                },
            )
        if state.current_step == 1 and state.task_family in {
            "gaze_interaction_anticipation",
            "fine_grained_how_recognition",
            "fine_grained_why_recognition",
            "recipe_step_recognition",
        } and combined_times and not state.retrieved_frames:
            return PlannerDecision(
                thought="先为短视频片段抽取按时间顺序排列的关键帧。",
                tool="extract_frames_for_range",
                args={
                    "start_time": max(0.0, min(combined_times)),
                    "end_time": max(combined_times),
                    "stride_s": max(0.3, (max(combined_times) - min(combined_times)) / 2) if len(combined_times) > 1 else 0.4,
                    "max_frames": 4,
                    "tag": f"{state.task_family}_segment",
                },
            )
        if state.current_step == 2 and state.task_family == "fine_grained_how_recognition" and state.retrieved_frames:
            return PlannerDecision(
                thought="对动作完成机制做专门判断。",
                tool="infer_action_mechanism",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-4:],
                },
            )
        if state.current_step == 2 and state.task_family == "fine_grained_why_recognition" and state.retrieved_frames:
            context_notes = [item for item in state.evidence_bundle if "type=" in item][:10]
            return PlannerDecision(
                thought="结合上下文活动和关键帧，对动作目的做专门判断。",
                tool="infer_action_intent",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-4:],
                    "context_notes": context_notes,
                },
            )
        if state.current_step == 2 and state.task_family in {
            "gaze_interaction_anticipation",
            "recipe_step_recognition",
        } and state.retrieved_frames:
            return PlannerDecision(
                thought="直接对该片段做视觉多选判断。",
                tool="infer_visual_mcq",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-4:],
                },
            )
        if state.current_step == 2 and bbox and state.retrieved_frames:
            return PlannerDecision(
                thought="对参考帧画出 bbox，保留原图上下文。",
                tool="render_bbox_overlay",
                args={"image_path": state.retrieved_frames[-1], "bbox": bbox, "tag": f"{state.task_family}_bbox"},
            )
        if state.current_step == 3 and bbox and state.retrieved_frames:
            return PlannerDecision(
                thought="放大 bbox 区域辅助识别目标物体。",
                tool="extract_region_with_context",
                args={"image_path": state.retrieved_frames[-1], "bbox": bbox, "expand_ratio": 0.35, "tag": f"{state.task_family}_crop"},
            )
        if state.current_step == 4 and state.task_family.startswith(("object_motion_", "3d_perception_", "gaze_")) and state.retrieved_frames:
            return PlannerDecision(
                thought="查看带框图和局部放大图，识别目标及其位置或交互。",
                tool="inspect_visual_evidence",
                args={
                    "prompt": (
                        "你在看厨房第一视角视频中同一目标的带框图与局部图。"
                        "请识别目标物体、所在位置、是否正在被交互或移动。"
                        '输出 JSON，字段固定为 {"target_object":"","target_location":"","ongoing_action":"","state_change_hint":"","answer_hint":"","confidence":0.0}。'
                    ),
                    "image_paths": state.retrieved_frames[-2:],
                },
            )
        if state.current_step == 5 and state.task_family == "3d_perception_fixture_interaction_counting":
            anchor_time = combined_times[0] if combined_times else None
            return PlannerDecision(
                thought="计数题先查询全视频 open/close 候选事件。",
                tool="query_event",
                args={
                    "event_types": ["audio_event"],
                    "keyword": "open / close",
                    "start_time": (anchor_time - 20.0) if anchor_time is not None else None,
                    "end_time": (anchor_time + 30.0) if anchor_time is not None else None,
                    "limit": 30,
                },
            )
        if state.current_step == 6 and state.task_family == "3d_perception_fixture_interaction_counting":
            last_result = state.tool_trace[-1].get("raw_result") if state.tool_trace else {}
            nodes = last_result.get("nodes", []) if isinstance(last_result, dict) else []
            candidate_times = [
                float(node.get("start_time"))
                for node in nodes
                if isinstance(node, dict) and node.get("start_time") is not None
            ]
            if not candidate_times:
                zero_index = next((idx for idx, choice in enumerate(state.choices) if str(choice).strip() == "0"), 0)
                return PlannerDecision(
                    thought="参考时刻附近没有目标交互候选，直接预测 0 次。",
                    tool="finish",
                    args={
                        "prediction": zero_index,
                        "answer": str(state.choices[zero_index]),
                        "confidence": 0.7,
                    },
                    done=True,
                    answer=str(state.choices[zero_index]),
                    prediction=zero_index,
                    confidence=0.7,
                )
            reference_paths = state.retrieved_frames[-2:] if len(state.retrieved_frames) >= 2 else state.retrieved_frames[-1:]
            return PlannerDecision(
                thought="针对候选开合事件逐帧判断是否属于目标，并完成计数。",
                tool="count_visual_candidates",
                args={
                    "reference_image_paths": reference_paths,
                    "candidate_times": candidate_times,
                    "choices": [str(choice) for choice in state.choices],
                    "action_hint": "close the referenced fixture",
                    "max_candidates": 8,
                    "tag": f"{state.task_family}_count",
                },
            )
        if state.current_step == 5 and state.task_family.startswith(("object_motion_", "3d_perception_", "gaze_")):
            return PlannerDecision(
                thought="基于当前时空与视觉证据对候选选项评分。",
                tool="rank_choices_from_state",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "evidence": state.evidence_bundle,
                    "working_memory": state.working_memory,
                },
            )
        if state.current_step <= 2 and combined_times:
            return PlannerDecision(
                thought="图谱证据不够，去视频里抽帧补证据。",
                tool="extract_frames_for_range",
                args={
                    "start_time": max(0.0, min(combined_times) - 2.0),
                    "end_time": max(combined_times) + 2.0,
                    "stride_s": 1.5,
                    "max_frames": 4,
                    "tag": f"{state.task_family}_step{state.current_step}",
                },
            )
        if state.current_step >= max(1, state.max_steps - 2):
            return PlannerDecision(
                thought="收尾阶段，直接基于当前证据对选项评分。",
                tool="rank_choices_from_state",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "evidence": state.evidence_bundle,
                    "working_memory": state.working_memory,
                },
            )
        return PlannerDecision(
            thought="兜底结束，让回答阶段基于当前证据给出结果。",
            tool="rank_choices_from_state",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "evidence": state.evidence_bundle,
                "working_memory": state.working_memory,
            },
        )

    def _enforce_task_requirements(self, *, state: AgentState, hints: dict[str, Any], decision: PlannerDecision) -> PlannerDecision:
        used_tools = [entry.get("tool") for entry in state.tool_trace if isinstance(entry, dict)]
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        bbox = hints.get("bbox")
        ingredient_name = hints.get("ingredient_name")

        if state.task_family == "ingredient_ingredient_weight" and decision.tool == "finish":
            if "query_ingredient_measurement" not in used_tools and ingredient_name and combined_times:
                return PlannerDecision(
                    thought="称重题在 finish 前必须先查称量记录。",
                    tool="query_ingredient_measurement",
                    args={
                        "ingredient_name": str(ingredient_name),
                        "start_time": min(combined_times),
                        "end_time": max(combined_times),
                        "limit": 10,
                    },
                )
            if "extract_frames_for_range" not in used_tools and combined_times:
                return PlannerDecision(
                    thought="称重题必须先回看称量时间段。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": max(0.0, min(combined_times) - 2.0),
                        "end_time": max(combined_times) + 2.0,
                        "stride_s": 1.0,
                        "max_frames": 5,
                        "tag": f"{state.task_family}_range",
                    },
                )
            if "inspect_visual_evidence" not in used_tools and state.retrieved_frames:
                return PlannerDecision(
                    thought="称重题在 finish 前必须至少做一次视觉读数检查。",
                    tool="inspect_visual_evidence",
                    args={
                        "prompt": (
                            "你在看厨房称重过程图像。"
                            "请识别正在称量的食材和可能的重量数字。"
                            '输出 JSON，字段固定为 {"ongoing_action":"","reading":"","digits":"","answer_hint":"","confidence":0.0}。'
                        ),
                        "image_paths": state.retrieved_frames[-5:],
                    },
                )

        if state.task_family.startswith(("object_motion_", "3d_perception_", "gaze_")) and bbox and decision.tool == "finish":
            if state.task_family == "3d_perception_fixture_interaction_counting" and "count_visual_candidates" not in used_tools:
                last_result = state.tool_trace[-1].get("raw_result") if state.tool_trace else {}
                nodes = last_result.get("nodes", []) if isinstance(last_result, dict) else []
                candidate_times = [
                    float(node.get("start_time"))
                    for node in nodes
                    if isinstance(node, dict) and node.get("start_time") is not None
                ]
                reference_paths = state.retrieved_frames[-2:] if len(state.retrieved_frames) >= 2 else state.retrieved_frames[-1:]
                if candidate_times and reference_paths:
                    return PlannerDecision(
                        thought="计数题在 finish 前必须先完成候选事件视觉计数。",
                        tool="count_visual_candidates",
                        args={
                            "reference_image_paths": reference_paths,
                            "candidate_times": candidate_times,
                            "choices": [str(choice) for choice in state.choices],
                            "action_hint": "close the referenced fixture",
                            "max_candidates": 8,
                            "tag": f"{state.task_family}_count",
                        },
                    )
            if "render_bbox_overlay" not in used_tools and state.retrieved_frames:
                return PlannerDecision(
                    thought="bbox 题在 finish 前至少要画一次框确认目标。",
                    tool="render_bbox_overlay",
                    args={"image_path": state.retrieved_frames[-1], "bbox": bbox, "tag": f"{state.task_family}_bbox"},
                )
            if "inspect_visual_evidence" not in used_tools and state.retrieved_frames:
                return PlannerDecision(
                    thought="bbox 题在 finish 前至少要做一次目标视觉检查。",
                    tool="inspect_visual_evidence",
                    args={
                        "prompt": (
                            "请根据带框图和局部图识别目标物体、位置和交互。"
                            '输出 JSON，字段固定为 {"target_object":"","target_location":"","ongoing_action":"","answer_hint":"","confidence":0.0}。'
                        ),
                        "image_paths": state.retrieved_frames[-2:],
                    },
                )

        if state.task_family.startswith("recipe_") and decision.tool == "finish" and "query_event" not in used_tools:
            return PlannerDecision(
                thought="步骤题在 finish 前必须先查 recipe_step 事件。",
                tool="query_event",
                args={
                    "event_types": ["recipe_step"],
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 20,
                },
            )

        if state.task_family in {"3d_perception_fixture_location", "gaze_gaze_estimation"}:
            required_tool = "infer_named_fixture_direction" if state.task_family == "3d_perception_fixture_location" else "infer_viewpoint_choice"
            if required_tool not in used_tools:
                if not state.retrieved_frames and combined_times:
                    return PlannerDecision(
                        thought="视角定位题必须先抽当前视角关键帧。",
                        tool="extract_frames_for_range",
                        args={
                            "start_time": max(0.0, min(combined_times) - 0.5),
                            "end_time": max(combined_times) + 0.5,
                            "stride_s": 0.5,
                            "max_frames": 3,
                            "tag": f"{state.task_family}_view",
                        },
                    )
                if state.retrieved_frames and decision.tool == "finish":
                    return PlannerDecision(
                        thought="视角定位题在 finish 前必须先做专用视觉方位选择。",
                        tool=required_tool,
                        args={
                            "question": state.question,
                            "choices": [str(choice) for choice in state.choices],
                            "image_paths": state.retrieved_frames[-3:],
                        },
                    )

        if state.task_family in {
            "gaze_interaction_anticipation",
            "recipe_step_recognition",
        } and "infer_visual_mcq" not in used_tools:
            if not state.retrieved_frames and combined_times:
                return PlannerDecision(
                    thought="片段类题先抽关键帧。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": max(0.0, min(combined_times)),
                        "end_time": max(combined_times),
                        "stride_s": max(0.3, (max(combined_times) - min(combined_times)) / 2) if len(combined_times) > 1 else 0.4,
                        "max_frames": 4,
                        "tag": f"{state.task_family}_segment",
                    },
                )
            if state.retrieved_frames and decision.tool == "finish":
                return PlannerDecision(
                    thought="片段类题在 finish 前必须先做视觉多选判断。",
                    tool="infer_visual_mcq",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-4:],
                    },
                )

        if state.task_family == "fine_grained_how_recognition" and "infer_action_mechanism" not in used_tools:
            if not state.retrieved_frames and combined_times:
                return PlannerDecision(
                    thought="how 题先抽关键帧。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": max(0.0, min(combined_times)),
                        "end_time": max(combined_times),
                        "stride_s": max(0.3, (max(combined_times) - min(combined_times)) / 2) if len(combined_times) > 1 else 0.4,
                        "max_frames": 4,
                        "tag": f"{state.task_family}_segment",
                    },
                )
            if state.retrieved_frames and decision.tool == "finish":
                return PlannerDecision(
                    thought="how 题在 finish 前必须先做动作机制判断。",
                    tool="infer_action_mechanism",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-4:],
                    },
                )

        if state.task_family == "fine_grained_why_recognition" and "infer_action_intent" not in used_tools:
            if not state.retrieved_frames and combined_times:
                return PlannerDecision(
                    thought="why 题先抽关键帧。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": max(0.0, min(combined_times)),
                        "end_time": max(combined_times),
                        "stride_s": max(0.3, (max(combined_times) - min(combined_times)) / 2) if len(combined_times) > 1 else 0.4,
                        "max_frames": 4,
                        "tag": f"{state.task_family}_segment",
                    },
                )
            if state.retrieved_frames and decision.tool == "finish":
                context_notes = [item for item in state.evidence_bundle if "type=" in item][:10]
                return PlannerDecision(
                    thought="why 题在 finish 前必须先做动作目的判断。",
                    tool="infer_action_intent",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-4:],
                        "context_notes": context_notes,
                    },
                )

        return decision
