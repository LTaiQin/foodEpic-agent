"""LLM planner for multi-step graph/video tool calling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from food_agent.agent.action_intent import (
    action_intent_followup_decision,
    action_intent_needs_future_use_resolution,
    action_intent_needs_pairwise_resolution,
    action_intent_needs_precondition_context,
)
from food_agent.agent.artifact_policy import artifact_reuse_prefixes_for_task
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


@dataclass(frozen=True)
class CandidatePlan:
    decision: PlannerDecision
    cost: int
    gain: int
    risk: int
    rationale: str

    @property
    def score(self) -> int:
        return self.cost - self.gain + self.risk


class GraphAgentPlanner:
    """Use the model to decide the next tool call instead of hard-coded routing."""

    def __init__(self, model_client: OpenAICompatibleModelClient):
        self.model_client = model_client

    def next_action(self, *, state: AgentState, tool_schemas: list[dict[str, Any]], hints: dict[str, Any]) -> PlannerDecision:
        if self._prefer_heuristic_planning(state):
            decision = self._heuristic_fallback(state=state, hints=hints)
        else:
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
        if decision is None:
            decision = self._safe_fallback_decision(state=state, hints=hints)
        decision = self._maybe_fast_finish_weight_task(state=state, decision=decision)
        decision = self._recover_if_low_confidence(state=state, hints=hints, decision=decision)
        decision = self._stabilize_decision(state=state, hints=hints, decision=decision)
        decision = self._enforce_task_requirements(state=state, hints=hints, decision=decision)
        return self._sanitize_decision_args(decision)

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

    def _is_weight_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredient_weight"

    def _question_explicitly_mentions_location(self, *, state: AgentState, location_keyword: Any) -> bool:
        question = str(getattr(state, "question", "") or "").lower()
        if any(token in question for token in ("where", "left", "right", "front", "behind", "beside", "inside", "outside", "near")):
            return True
        if location_keyword:
            keyword = str(location_keyword).strip().lower()
            if self._is_weight_task(state) and keyword in {"scale", "reading", "number"}:
                return False
            if self._is_ingredient_retrieval_task(state):
                return False
            if keyword and keyword in question:
                return True
        return False

    def _is_viewpoint_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {"3d_perception_fixture_location", "gaze_gaze_estimation"}

    def _is_object_motion_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")).startswith("object_motion_")

    def _is_object_itinerary_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "object_motion_object_movement_itinerary"

    def _is_object_location_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "3d_perception_object_location"

    def _is_object_contents_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "3d_perception_object_contents_retrieval"

    def _is_temporal_localization_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {
            "fine_grained_action_localization",
            "ingredient_ingredient_adding_localization",
            "recipe_multi_step_localization",
            "recipe_rough_step_localization",
            "recipe_prep_localization",
            "recipe_step_localization",
        }

    def _is_recipe_catalog_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {
            "recipe_recipe_recognition",
            "recipe_multi_recipe_recognition",
        }

    def _is_recipe_following_activity_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "recipe_following_activity_recognition"

    def _is_nutrition_change_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "nutrition_nutrition_change"

    def _is_recipe_nutrition_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "nutrition_video_nutrition_estimation"

    def _is_ingredient_order_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredients_order"

    def _is_ingredient_retrieval_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredient_retrieval"

    def _location_conflict_is_actionable(self, *, state: AgentState, hints: dict[str, Any]) -> bool:
        if self._is_weight_task(state):
            return self._question_explicitly_mentions_location(state=state, location_keyword=hints.get("location_keyword"))
        if self._is_ingredient_retrieval_task(state):
            return self._question_explicitly_mentions_location(state=state, location_keyword=hints.get("location_keyword"))
        return True

    def _state_conflict_is_actionable(self, *, state: AgentState) -> bool:
        if self._is_weight_task(state):
            return False
        if self._is_ingredient_retrieval_task(state):
            return False
        return True

    def _is_recipe_ingredient_membership_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredient_recognition"

    def _is_exact_ingredient_amount_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_exact_ingredient_recognition"

    def _is_recipe_step_evidence_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")).startswith("recipe_") and not self._is_recipe_catalog_task(state)

    def _is_action_mechanism_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "fine_grained_how_recognition"

    def _is_action_intent_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "fine_grained_why_recognition"

    def _action_intent_requires_followup(self, state: AgentState, result: dict[str, Any] | None = None) -> bool:
        if isinstance(result, dict):
            if bool(result.get("need_future_evidence")) or bool(result.get("ambiguity")):
                return True
            try:
                confidence = float(result.get("confidence") or 0.0)
            except Exception:  # noqa: BLE001
                confidence = 0.0
            candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
            needs_followup, _, _, _ = action_intent_followup_decision(
                question=str(getattr(state, "question", "") or ""),
                choices=[str(choice) for choice in getattr(state, "choices", [])],
                indices=candidate_indices if len(candidate_indices) >= 2 else None,
                confidence=confidence,
                reason_text=str(result.get("reason") or ""),
            )
            if needs_followup:
                return True
            all_needs_followup, _, _, _ = action_intent_followup_decision(
                question=str(getattr(state, "question", "") or ""),
                choices=[str(choice) for choice in getattr(state, "choices", [])],
                indices=None,
                confidence=confidence,
                reason_text=str(result.get("reason") or ""),
            )
            if all_needs_followup:
                return True
        return any(
            isinstance(item, str) and item.startswith("action_intent_need_future_evidence=1")
            for item in list(getattr(state, "working_memory", [])) + list(getattr(state, "evidence_bundle", []))
        )

    def _build_action_intent_followup_sampling_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> PlannerDecision | None:
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        if not combined_times:
            return None
        focus = ""
        semantic_need, semantic_reason, semantic_window_s, _ = action_intent_followup_decision(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
        )
        if semantic_need:
            focus = semantic_reason
        window_s = max(
            semantic_window_s,
            8.0 if self._action_intent_needs_future_use_evidence(state=state, result=None) else 4.0,
        )
        for item in reversed(list(getattr(state, "working_memory", []))):
            if not isinstance(item, str) or not item.startswith("action_intent_need_future_evidence=1"):
                continue
            focus_match = re.search(r"focus=(.*)$", item)
            if focus_match:
                focus = focus_match.group(1).strip()
            window_match = re.search(r"window_s=([0-9.]+)", item)
            if window_match:
                try:
                    window_s = max(2.0, min(8.0, float(window_match.group(1))))
                except Exception:  # noqa: BLE001
                    window_s = 4.0
            break
        if self._action_intent_needs_future_use_evidence(state=state, result=None):
            window_s = max(8.0, window_s)
        start_time = max(combined_times)
        end_time = start_time + window_s
        return PlannerDecision(
            thought=(
                "why 题当前仍存在意图歧义，补动作后的结果帧，检查后续是继续取后方物体，还是只是腾空间/整理。"
                + (f" followup_focus={focus}" if focus else "")
            ),
            tool="sample_sparse_frames",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "sample_count": 4,
                "tag": f"{state.task_family}_followup",
            },
        )

    def _action_intent_needs_precondition_context(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        if action_intent_needs_precondition_context(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
        ):
            return True
        # Precondition-dependent options such as "dry hands" can be missed by the
        # first-pass top-2 guess, so also scan the full candidate set.
        return action_intent_needs_precondition_context(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=None,
        )

    def _build_action_intent_precondition_sampling_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        focus: str = "precondition_context",
    ) -> PlannerDecision | None:
        combined_times = sorted(
            [float(value) for value in hints.get("times") or []]
            + [float(value) for value in hints.get("input_times") or []]
        )
        if not combined_times:
            return None
        action_start = min(combined_times)
        start_time = max(0.0, action_start - 6.0)
        if action_start <= start_time:
            return None
        return PlannerDecision(
            thought=(
                "why 题包含清洁/擦手/安全等前置状态依赖目的；先补动作前上下文，检查手是否刚洗过、台面是否需要擦、"
                "或是否存在热/湿/脏等触发原因。"
                f" precondition_focus={focus}"
            ),
            tool="sample_sparse_frames",
            args={
                "start_time": start_time,
                "end_time": action_start,
                "sample_count": 4,
                "tag": f"{state.task_family}_precontext",
            },
        )

    def _action_intent_has_precondition_frames(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> bool:
        combined_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    combined_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        action_start = min(combined_times) if combined_times else None
        task_tag = str(getattr(state, "task_family", "") or "").lower()
        pre_tag = f"{task_tag}_precontext" if task_tag else "_precontext"
        for path in self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or [])):
            name = Path(path).name.lower()
            if pre_tag in name:
                return True
            artifact_time = self._artifact_time_from_path(path)
            if action_start is not None and artifact_time is not None and action_start - 6.5 <= artifact_time < action_start - 0.25:
                return True
        return False

    def _latest_action_intent_followup_end_time(self, state: AgentState) -> float | None:
        latest_end: float | None = None
        for entry in getattr(state, "tool_trace", []):
            if not isinstance(entry, dict) or entry.get("tool") != "sample_sparse_frames":
                continue
            args = entry.get("args")
            if not isinstance(args, dict):
                continue
            tag = str(args.get("tag") or "")
            if not tag.startswith("fine_grained_why_recognition_followup"):
                continue
            try:
                end_time = float(args.get("end_time"))
            except Exception:  # noqa: BLE001
                continue
            latest_end = end_time if latest_end is None else max(latest_end, end_time)
        return latest_end

    def _build_action_intent_extra_followup_sampling_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        focus: str = "",
        window_s: float = 8.0,
    ) -> PlannerDecision | None:
        combined_times = sorted(
            [float(value) for value in hints.get("times") or []]
            + [float(value) for value in hints.get("input_times") or []]
        )
        if not combined_times:
            return None
        attempt_count = self._action_intent_followup_attempt_count(state)
        start_time = self._latest_action_intent_followup_end_time(state)
        if start_time is None:
            start_time = max(combined_times)
        window_s = max(4.0, min(10.0, float(window_s)))
        return PlannerDecision(
            thought=(
                "why 题专用裁决仍报告证据不足，继续向后补帧，检查动作后的最终放置、使用或取回结果。"
                + (f" followup_focus={focus}" if focus else "")
            ),
            tool="sample_sparse_frames",
            args={
                "start_time": start_time,
                "end_time": start_time + window_s,
                "sample_count": 4,
                "tag": f"{state.task_family}_followup_ext{attempt_count + 1}",
            },
        )

    def _action_intent_followup_attempt_count(self, state: AgentState) -> int:
        count = 0
        for entry in getattr(state, "tool_trace", []):
            if not isinstance(entry, dict) or entry.get("tool") != "sample_sparse_frames":
                continue
            args = entry.get("args")
            if not isinstance(args, dict):
                continue
            tag = str(args.get("tag") or "")
            if tag.startswith("fine_grained_why_recognition_followup"):
                count += 1
        return count

    def _action_intent_pending_resolution_tool(self, state: AgentState) -> str:
        for item in reversed(list(getattr(state, "working_memory", []))):
            if not isinstance(item, str):
                continue
            match = re.search(r"action_intent_pending_resolution=(\w+)", item)
            if not match:
                continue
            tool = match.group(1)
            if tool in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
                return tool
        return ""

    def _action_intent_resolution_needs_more_evidence(self, *, tool_name: str, result: dict[str, Any]) -> bool:
        if bool(result.get("need_more_evidence")):
            return True
        try:
            confidence = float(result.get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        reason = str(result.get("reason") or "")
        decisive = str(result.get("decisive_observation") or "")
        text = f"{reason} {decisive}".lower()
        hard_uncertainty_terms = (
            "not enough",
            "insufficient",
            "unclear",
            "cannot tell",
            "can't tell",
            "ambiguous",
            "uncertain",
            "no decisive",
            "hard to tell",
        )
        weak_missing_terms = (
            "not visible",
            "not shown",
            "lack",
            "missing",
        )
        if any(term in text for term in hard_uncertainty_terms):
            return True
        if tool_name == "resolve_action_intent_future_use" and not decisive.strip() and confidence < 0.85:
            return True
        if confidence < 0.78 and any(term in text for term in weak_missing_terms):
            return True
        return confidence < 0.68

    def _action_intent_resolution_should_backfill_precondition(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if self._action_intent_has_precondition_frames(state=state, hints=hints):
            return False
        if not self._action_intent_needs_precondition_context(state=state, result=result):
            return False
        text = " ".join(
            str(result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation", "answer")
        ).lower()
        precondition_terms = (
            "dry hands",
            "drying hands",
            "hand-drying",
            "hand drying",
            "hands after pickup",
            "applied to the hands",
            "applied to hands",
            "wet-hand",
            "wet hands",
            "wipe",
            "wiping",
            "wiped",
            "surface",
            "counter",
            "worktop",
            "cleaning",
            "clean",
            "washed",
            "wash",
            "rinsed",
            "sink",
            "擦手",
            "干手",
            "湿手",
            "擦台面",
            "台面",
            "清洁",
            "清洗",
            "水槽",
        )
        gap_terms = (
            "missing",
            "lack",
            "no visible",
            "no actual",
            "not shown",
            "not visible",
            "absence",
            "need",
            "缺少",
            "没有",
            "未看到",
            "需要",
        )
        return any(term in text for term in precondition_terms) and any(term in text for term in gap_terms)

    def _latest_action_intent_candidate_indices(self, state: AgentState, result: dict[str, Any] | None = None) -> list[int]:
        indices: list[int] = []
        if isinstance(result, dict):
            for value in result.get("candidate_indices") or []:
                try:
                    index = int(value)
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= index < len(state.choices) and index not in indices:
                    indices.append(index)
            for key in ("best_index", "second_best_index"):
                try:
                    index = int(result.get(key))
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= index < len(state.choices) and index not in indices:
                    indices.append(index)
        latest_trace = state.tool_trace[-1] if state.tool_trace else {}
        latest_raw = latest_trace.get("raw_result") if isinstance(latest_trace, dict) else {}
        if (
            not indices
            and isinstance(latest_trace, dict)
            and isinstance(latest_raw, dict)
            and latest_raw.get("tool_failed")
            and latest_trace.get("tool") in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}
        ):
            args = latest_trace.get("args") or {}
            if isinstance(args, dict):
                for value in args.get("candidate_indices") or []:
                    try:
                        index = int(value)
                    except Exception:  # noqa: BLE001
                        continue
                    if 0 <= index < len(state.choices) and index not in indices:
                        indices.append(index)
        for index in self._action_intent_pending_candidate_indices(state):
            if index not in indices:
                indices.append(index)
        for item in reversed(list(getattr(state, "working_memory", []))):
            if not isinstance(item, str):
                continue
            if item.startswith("action_intent_best_index="):
                match = re.search(r"action_intent_best_index=(\d+)", item)
                if match:
                    index = int(match.group(1))
                    if 0 <= index < len(state.choices) and index not in indices:
                        indices.append(index)
            if item.startswith("action_intent_second_best_index="):
                match = re.search(r"action_intent_second_best_index=(\d+)", item)
                if match:
                    index = int(match.group(1))
                    if 0 <= index < len(state.choices) and index not in indices:
                        indices.append(index)
            if len(indices) >= len(state.choices):
                break
        return indices

    def _action_intent_pending_candidate_indices(self, state: AgentState) -> list[int]:
        indices: list[int] = []
        for item in reversed(list(getattr(state, "working_memory", []))):
            if not isinstance(item, str) or not item.startswith("action_intent_pending_candidates="):
                continue
            for match in re.finditer(r"\d+", item):
                try:
                    index = int(match.group(0))
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= index < len(getattr(state, "choices", [])) and index not in indices:
                    indices.append(index)
            if indices:
                break
        return indices

    def _action_intent_pair_needs_outcome_resolution(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
        candidate_indices: list[int] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        indices = candidate_indices or self._latest_action_intent_candidate_indices(state, result=result)
        if len(indices) < 2:
            return False
        if action_intent_needs_pairwise_resolution(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=indices,
        ):
            return True
        question_text = str(getattr(state, "question", "") or "").lower()
        action_terms = (
            "move ",
            "moved ",
            "shift ",
            "clear ",
            "put ",
            "place ",
            "pick up",
            "take out",
            "remove ",
            "open ",
            "close ",
        )
        if not any(term in question_text for term in action_terms):
            return False
        choice_texts = [
            str(state.choices[index]).lower()
            for index in indices
            if 0 <= index < len(getattr(state, "choices", []))
        ]
        if len(choice_texts) < 2:
            return False
        outcome_terms = (
            "make space",
            "space",
            "access",
            "behind",
            "clear the way",
            "right place",
            "put back",
            "put",
            "place",
            "pick up",
            "remove",
            "rearrange",
        )
        hit_count = sum(1 for text in choice_texts if any(term in text for term in outcome_terms))
        joined = " | ".join(choice_texts)
        return hit_count >= 2 or (
            hit_count >= 1
            and any(pair_left in joined and pair_right in joined for pair_left, pair_right in (("behind", "space"), ("access", "space"), ("right place", "space")))
        )

    def _action_intent_needs_future_use_evidence(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        if action_intent_needs_future_use_resolution(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
        ):
            return True
        question_text = str(getattr(state, "question", "") or "").lower()
        manipulation_terms = (
            "pick up",
            "picked up",
            "pick ",
            "lift ",
            "lifted ",
            "take ",
            "took ",
            "transfer ",
            "transferred ",
            "carry ",
            "carried ",
            "grab ",
            "grabbed ",
        )
        if not any(term in question_text for term in manipulation_terms):
            return False
        use_terms = (
            "weigh",
            "measure",
            "use ",
            "serve",
            "empty",
            "drain",
            "pour",
            "check",
            "retrieve",
            "get ",
            "fill",
            "wash",
            "clean",
            "dry",
            "record",
            "scan",
            "put ",
            "place ",
            "return",
            "close",
            "open",
            "turn",
            "mix",
            "stir",
        )
        choices = [str(choice).lower() for choice in getattr(state, "choices", [])]
        matched_choices = [choice for choice in choices if any(term in choice for term in use_terms)]
        if len(matched_choices) < 2:
            return False
        return True

    def _build_action_intent_pairwise_resolution_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
        thought: str = "why 题存在动作后果型歧义，改为只在前两名候选之间结合结果帧裁决。",
    ) -> PlannerDecision | None:
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        action_frames = self._select_action_intent_frames(
            state,
            hints,
            limit=8,
            require_current_scope=True,
        )
        if len(candidate_indices) < 2 or not action_frames:
            return None
        missing_followup = self._build_action_intent_missing_post_action_followup_decision(
            state=state,
            hints=hints,
            action_frames=action_frames,
            focus="pairwise_outcome_resolution",
        )
        if missing_followup is not None:
            return missing_followup
        context_notes = self._action_intent_context_notes(state, limit=12)
        return PlannerDecision(
            thought=thought,
            tool="resolve_action_intent_pairwise",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "candidate_indices": candidate_indices,
                "image_paths": action_frames,
                "context_notes": context_notes,
            },
        )

    def _build_action_intent_future_use_resolution_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
        thought: str = "why 题属于后续用途型意图判断，必须显式比较动作后的使用证据再收口。",
    ) -> PlannerDecision | None:
        action_frames = self._select_action_intent_frames(
            state,
            hints,
            limit=8,
            require_current_scope=True,
        )
        if not action_frames:
            return None
        missing_followup = self._build_action_intent_missing_post_action_followup_decision(
            state=state,
            hints=hints,
            action_frames=action_frames,
            focus="future_use_resolution",
        )
        if missing_followup is not None:
            return missing_followup
        candidate_indices = list(range(len(getattr(state, "choices", []) or [])))
        context_notes = self._action_intent_context_notes(state, limit=12)
        return PlannerDecision(
            thought=thought,
            tool="resolve_action_intent_future_use",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "candidate_indices": candidate_indices,
                "image_paths": action_frames,
                "context_notes": context_notes,
            },
        )

    def _action_intent_failed_tool_count(self, state: AgentState, tool_name: str) -> int:
        count = 0
        for entry in getattr(state, "tool_trace", []):
            if not isinstance(entry, dict) or entry.get("tool") != tool_name:
                continue
            raw_result = entry.get("raw_result")
            if isinstance(raw_result, dict) and raw_result.get("tool_failed"):
                count += 1
        return count

    def _action_intent_context_notes(self, state: AgentState, *, limit: int) -> list[str]:
        notes: list[str] = []
        for item in getattr(state, "evidence_bundle", []):
            if not isinstance(item, str) or "type=" not in item:
                continue
            if self._is_action_intent_leaky_context_note(item):
                continue
            if item not in notes:
                notes.append(item)
        return notes[:limit]

    def _build_action_intent_missing_post_action_followup_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        action_frames: list[str],
        focus: str,
    ) -> PlannerDecision | None:
        if self._action_intent_has_post_action_frames(state=state, hints=hints, frames=action_frames):
            return None
        attempt_count = self._action_intent_followup_attempt_count(state)
        if attempt_count >= 3:
            return None
        if attempt_count == 0:
            followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
        else:
            followup = self._build_action_intent_extra_followup_sampling_decision(
                state=state,
                hints=hints,
                focus=focus,
            )
        if followup is None:
            return None
        return PlannerDecision(
            thought=(
                "why 题专用裁决缺少动作后的结果帧；先补后续帧，再判断当前动作的真实目的。"
                f" followup_focus={focus}"
            ),
            tool=followup.tool,
            args=followup.args,
            done=followup.done,
            answer=followup.answer,
            prediction=followup.prediction,
            confidence=followup.confidence,
        )

    def _action_intent_has_post_action_frames(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        frames: list[str],
    ) -> bool:
        if not frames:
            return False
        task_tag = str(getattr(state, "task_family", "") or "").lower()
        followup_tag = f"{task_tag}_followup" if task_tag else "_followup"
        combined_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    combined_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        action_end = max(combined_times) if combined_times else None
        for path in frames:
            name = Path(path).name.lower()
            if followup_tag in name:
                return True
            artifact_time = self._artifact_time_from_path(path)
            if action_end is not None and artifact_time is not None and artifact_time > action_end + 0.5:
                return True
        return False

    def _is_action_intent_leaky_context_note(self, item: str) -> bool:
        lowered = str(item or "").lower()
        leaky_tokens = (
            "action_intent_",
            "visual_mcq_reason=",
            "answer_hint=",
            "candidate_answer_index=",
            "deterministic_finalize",
            "source=agent_timeline_summary",
            "source=session_memory_compressor",
        )
        return any(token in lowered for token in leaky_tokens)

    def _build_action_intent_specialized_recovery_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        action_frames = self._select_action_intent_frames(
            state,
            hints,
            limit=8,
            require_current_scope=True,
        )
        if action_frames:
            return PlannerDecision(
                thought=thought,
                tool="infer_action_intent",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": action_frames,
                    "context_notes": self._action_intent_context_notes(state, limit=12),
                },
            )
        combined_times = sorted(
            [float(value) for value in hints.get("times") or []]
            + [float(value) for value in hints.get("input_times") or []]
        )
        if not combined_times:
            return None
        return PlannerDecision(
            thought="why 题专用判断缺少当前题时间窗帧，重新抽当前动作片段，避免复用同视频其它题旧帧。",
            tool="sample_sparse_frames",
            args={
                "start_time": min(combined_times),
                "end_time": max(combined_times),
                "sample_count": 4,
                "tag": f"{state.task_family}_segment",
            },
        )

    def _action_intent_text_fallback_ready(self, state: AgentState) -> bool:
        return (
            self._is_action_intent_task(state)
            and self._action_intent_failed_tool_count(state, "infer_action_intent") >= 3
            and not self._latest_successful_action_intent_result(state)
        )

    def _build_action_intent_text_fallback_rank_decision(self, state: AgentState, *, thought: str) -> PlannerDecision:
        evidence = self._action_intent_context_notes(state, limit=12) + list(getattr(state, "evidence_bundle", []) or [])[-12:]
        deduped_evidence = list(dict.fromkeys(str(item) for item in evidence if isinstance(item, str) and str(item).strip()))
        working_memory = [
            str(item)
            for item in list(getattr(state, "working_memory", []) or [])[-20:]
            if isinstance(item, str) and str(item).strip()
        ]
        return PlannerDecision(
            thought=thought,
            tool="rank_choices_from_state",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "evidence": deduped_evidence[:30],
                "working_memory": working_memory[:30],
            },
        )

    def _can_use_visual_inspection(self, state: AgentState) -> bool:
        if any(
            isinstance(item, str) and item.startswith("vision_disabled=")
            for item in list(getattr(state, "working_memory", [])) + list(getattr(state, "evidence_bundle", []))
        ):
            return False
        supports = getattr(self.model_client, "supports_vision_requests", None)
        if callable(supports):
            try:
                return bool(supports())
            except Exception:  # noqa: BLE001
                return True
        return True

    def _prefer_heuristic_planning(self, state: AgentState) -> bool:
        return (
            self._is_weight_task(state)
            or self._is_viewpoint_task(state)
            or self._is_object_motion_task(state)
            or self._is_object_location_task(state)
            or self._is_object_contents_task(state)
            or self._is_temporal_localization_task(state)
            or self._is_action_mechanism_task(state)
            or self._is_action_intent_task(state)
            or self._is_recipe_catalog_task(state)
            or self._is_recipe_following_activity_task(state)
            or self._is_recipe_nutrition_task(state)
            or self._is_ingredient_order_task(state)
            or self._is_nutrition_change_task(state)
        )

    def _structured_direct_inference_config(self, state: AgentState) -> tuple[str, str, dict[str, Any]] | None:
        if self._is_recipe_nutrition_task(state):
            return (
                "infer_recipe_nutrition_choice",
                "视频级营养题优先从 recipe catalog 和结构化营养记录中直接比较候选食材。",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_ingredient_order_task(state):
            return (
                "infer_ingredient_order_choice",
                "食材加入顺序题优先直接读取结构化 ingredient add 事件顺序。",
                {"question": state.question, "choices": state.choices},
            )
        if self._is_ingredient_retrieval_task(state):
            return (
                "infer_ingredient_retrieval_choice",
                "时间窗食材检索题优先直接读取该区间的结构化 ingredient add 事件。",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_recipe_ingredient_membership_task(state):
            return (
                "infer_recipe_ingredient_membership_choice",
                "菜谱食材归属题优先用 recipe catalog 判断哪个候选不属于目标菜谱。",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_exact_ingredient_amount_task(state):
            return (
                "infer_exact_ingredient_amount_choice",
                "精确食材用量题优先直接读取 recipe catalog 中的 ingredient_amounts。",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_recipe_catalog_task(state):
            return (
                "infer_recipe_catalog_choice",
                "菜谱识别题优先用 inputs 对应视频集合的 recipe catalog 做候选匹配。",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "scope": "participant" if state.task_family == "recipe_recipe_recognition" else "video",
                },
            )
        return None

    def _segment_task_inference_config(self, state: AgentState, hints: dict[str, Any] | None = None) -> tuple[str, str, dict[str, Any]] | None:
        base_args = {
            "question": state.question,
            "choices": [str(choice) for choice in state.choices],
            "image_paths": self._filter_visual_image_paths(state.retrieved_frames)[-4:],
        }
        if state.task_family in {"gaze_interaction_anticipation", "fine_grained_action_recognition", "recipe_step_recognition"}:
            return ("infer_visual_mcq", "直接对该片段做视觉多选判断。", base_args)
        if self._is_action_mechanism_task(state):
            return ("infer_action_mechanism", "对动作完成机制做专门判断。", base_args)
        if self._is_action_intent_task(state):
            return (
                "infer_action_intent",
                "结合上下文活动和关键帧，对动作目的做专门判断。",
                {
                    **base_args,
                    "image_paths": self._select_action_intent_frames(
                        state,
                        hints,
                        limit=4,
                        include_followup=False,
                        require_current_scope=True,
                    ),
                    "context_notes": self._action_intent_context_notes(state, limit=10),
                },
            )
        return None

    def _filter_visual_image_paths(self, paths: list[str]) -> list[str]:
        valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        filtered: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            normalized = str(raw_path).strip()
            if not normalized or normalized in seen:
                continue
            if Path(normalized).suffix.lower() not in valid_suffixes:
                continue
            filtered.append(normalized)
            seen.add(normalized)
        return filtered

    def _latest_visual_frame(self, state: AgentState) -> str | None:
        filtered = self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or []))
        return filtered[-1] if filtered else None

    def _artifact_time_from_path(self, path: str) -> float | None:
        match = re.search(r"_(\d+(?:\.\d+)?)s\.[^.]+$", str(path).strip())
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _sort_frames_by_artifact_time(self, frames: list[str]) -> list[str]:
        sortable: list[tuple[int, float, int, str]] = []
        for original_index, path in enumerate(frames):
            artifact_time = self._artifact_time_from_path(path)
            if artifact_time is None:
                sortable.append((1, float(original_index), original_index, path))
            else:
                sortable.append((0, artifact_time, original_index, path))
        return [path for _, _, _, path in sorted(sortable)]

    def _select_action_intent_frames(
        self,
        state: AgentState,
        hints: dict[str, Any] | None = None,
        *,
        limit: int = 8,
        include_followup: bool = True,
        require_current_scope: bool = False,
    ) -> list[str]:
        frames = self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or []))
        if not frames:
            return []
        task_tag = str(getattr(state, "task_family", "") or "").lower()
        followup_tag = f"{task_tag}_followup"
        precontext_tag = f"{task_tag}_precontext"
        tagged = []
        for path in frames:
            name = Path(path).name.lower()
            if not task_tag or task_tag not in name:
                continue
            if not include_followup and followup_tag in name:
                continue
            tagged.append(path)
        if tagged:
            frames = tagged

        combined_times: list[float] = []
        if isinstance(hints, dict):
            for key in ("times", "input_times"):
                for value in hints.get(key) or []:
                    try:
                        combined_times.append(float(value))
                    except Exception:  # noqa: BLE001
                        continue
        if combined_times:
            precondition_window_s = (
                6.0
                if include_followup and self._action_intent_needs_precondition_context(state=state, result=None)
                else 2.0
            )
            start_time = min(combined_times) - precondition_window_s
            followup_window_s = 8.0 if include_followup else 2.0
            if include_followup:
                for item in reversed(list(getattr(state, "working_memory", []))):
                    if not isinstance(item, str) or not item.startswith("action_intent_need_future_evidence=1"):
                        continue
                    window_match = re.search(r"window_s=([0-9.]+)", item)
                    if window_match:
                        try:
                            followup_window_s = max(2.0, min(8.0, float(window_match.group(1))))
                        except Exception:  # noqa: BLE001
                            followup_window_s = 8.0
                    break
                latest_followup_end = self._latest_action_intent_followup_end_time(state)
                if latest_followup_end is not None:
                    followup_window_s = max(
                        followup_window_s,
                        min(30.0, latest_followup_end - max(combined_times)),
                    )
            end_time = max(combined_times) + followup_window_s
            timed = []
            unknown_time = []
            for path in frames:
                artifact_time = self._artifact_time_from_path(path)
                if artifact_time is None:
                    unknown_time.append(path)
                    continue
                if start_time <= artifact_time <= end_time:
                    timed.append(path)
            if timed:
                frames = timed
            elif tagged and unknown_time:
                frames = unknown_time
            elif require_current_scope:
                return []
        frames = self._sort_frames_by_artifact_time(frames)
        if len(frames) > limit and self._is_action_intent_task(state) and include_followup and combined_times:
            action_cutoff = max(combined_times) + 0.75
            precontext_frames = []
            current_frames = []
            followup_frames = []
            for path in frames:
                name = Path(path).name.lower()
                if precontext_tag and precontext_tag in name:
                    precontext_frames.append(path)
                    continue
                if followup_tag and followup_tag in name:
                    followup_frames.append(path)
                    continue
                artifact_time = self._artifact_time_from_path(path)
                if artifact_time is not None and artifact_time <= action_cutoff:
                    current_frames.append(path)
                else:
                    followup_frames.append(path)
            pre_keep_count = 2 if precontext_frames and self._action_intent_needs_precondition_context(state=state, result=None) else 0
            pre_keep = self._sample_evenly_ordered(precontext_frames, pre_keep_count)
            current_keep = current_frames[-2:]
            followup_keep = self._sample_evenly_ordered(followup_frames, max(0, limit - len(pre_keep) - len(current_keep)))
            merged = pre_keep + current_keep + followup_keep
            if merged:
                return self._sort_frames_by_artifact_time(merged)
        return frames[-limit:]

    def _sample_evenly_ordered(self, items: list[str], limit: int) -> list[str]:
        if limit <= 0 or not items:
            return []
        if len(items) <= limit:
            return list(items)
        if limit == 1:
            return [items[-1]]
        selected_indices = {
            round(index * (len(items) - 1) / (limit - 1))
            for index in range(limit)
        }
        return [items[index] for index in sorted(selected_indices)]

    def _sanitize_decision_args(self, decision: PlannerDecision) -> PlannerDecision:
        if not decision.args:
            return decision
        visual_multi_tools = {
            "infer_action_mechanism",
            "infer_action_intent",
            "resolve_action_intent_pairwise",
            "resolve_action_intent_future_use",
            "infer_visual_mcq",
            "infer_viewpoint_choice",
            "infer_named_fixture_direction",
            "infer_gaze_target_with_context",
            "inspect_visual_evidence",
        }
        visual_single_tools = {
            "run_ocr_on_image",
            "run_ocr_on_region",
            "render_bbox_overlay",
            "extract_region_with_context",
        }
        if decision.tool in visual_multi_tools and "image_paths" in decision.args:
            sanitized = dict(decision.args)
            sanitized["image_paths"] = self._filter_visual_image_paths(list(sanitized.get("image_paths") or []))
            return PlannerDecision(
                thought=decision.thought,
                tool=decision.tool,
                args=sanitized,
                done=decision.done,
            )
        if decision.tool in visual_single_tools and "image_path" in decision.args:
            image_path = str(decision.args.get("image_path") or "").strip()
            if Path(image_path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                sanitized = dict(decision.args)
                sanitized["image_path"] = ""
                return PlannerDecision(
                    thought=decision.thought,
                    tool=decision.tool,
                    args=sanitized,
                    done=decision.done,
                )
        return decision

    def _is_segment_visual_task(self, state: AgentState) -> bool:
        return self._segment_task_inference_config(state) is not None

    def _has_current_segment_visual_frames(self, state: AgentState, combined_times: list[float]) -> bool:
        if self._is_action_intent_task(state):
            return bool(
                self._select_action_intent_frames(
                    state,
                    {"times": combined_times, "input_times": []},
                    limit=1,
                    include_followup=False,
                    require_current_scope=True,
                )
            )
        return bool(state.retrieved_frames)

    def _segment_task_sampling_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        combined_times: list[float],
        reuse_thought: str,
        extract_thought: str,
    ) -> PlannerDecision | None:
        if not self._is_segment_visual_task(state) or not combined_times:
            return None
        if self._has_current_segment_visual_frames(state, combined_times):
            return None
        if self._is_action_intent_task(state):
            reuse = self._build_reuse_or_extract_range_decision(
                state=state,
                used_tools=used_tools,
                tag_hint=f"{state.task_family}_segment",
                artifact_prefixes=(f"{state.task_family}_segment",),
                start_time=max(0.0, min(combined_times)),
                end_time=max(combined_times),
                reuse_thought=reuse_thought,
                extract_thought=extract_thought,
                extract_tag=f"{state.task_family}_segment",
                stride_s=self._segment_stride_s(combined_times),
                max_frames=4,
            )
            if reuse is not None and reuse.tool == "retrieve_cached_artifacts":
                return reuse
            return PlannerDecision(
                thought="why 题需要先抽当前动作时间窗关键帧，避免复用同视频其它题留下的旧帧。",
                tool="sample_sparse_frames",
                args={
                    "start_time": max(0.0, min(combined_times)),
                    "end_time": max(combined_times),
                    "sample_count": 4,
                    "tag": f"{state.task_family}_segment",
                },
            )
        return self._build_reuse_or_extract_range_decision(
            state=state,
            used_tools=used_tools,
            tag_hint=state.task_family,
            artifact_prefixes=self._artifact_reuse_prefixes(state),
            start_time=max(0.0, min(combined_times)),
            end_time=max(combined_times),
            reuse_thought=reuse_thought,
            extract_thought=extract_thought,
            extract_tag=f"{state.task_family}_segment",
            stride_s=self._segment_stride_s(combined_times),
            max_frames=4,
        )

    def _segment_task_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        combined_times: list[float],
    ) -> PlannerDecision | None:
        if decision.tool != "finish":
            return None
        config = self._segment_task_inference_config(state, {"times": combined_times, "input_times": []})
        if config is None:
            return None
        tool, thought, args = config
        if tool in used_tools:
            return None
        sampling = self._segment_task_sampling_decision(
            state=state,
            used_tools=used_tools,
            combined_times=combined_times,
            reuse_thought="片段类题先检索当前视频里已存在的片段 artifact，优先复用历史抽帧。",
            extract_thought="片段类题在 finish 前必须先抽关键帧。",
        )
        if sampling is not None:
            return sampling
        if self._filter_visual_image_paths(list(args.get("image_paths") or [])) and decision.tool == "finish":
            return PlannerDecision(
                thought=f"片段类题在 finish 前必须先完成该片段的专用推理：{thought}",
                tool=tool,
                args=args,
            )
        return None

    def _bbox_structured_task_config(
        self,
        state: AgentState,
        *,
        combined_times: list[float],
    ) -> tuple[str, str, dict[str, Any]] | None:
        if not combined_times:
            return None
        reference_time = combined_times[0]
        if self._is_object_itinerary_task(state):
            return (
                "infer_object_movement_itinerary",
                "根据目标对象的完整 fixture 路径推断移动轨迹选项。",
                {
                    "bbox": None,
                    "reference_time": reference_time,
                    "choices": [str(choice) for choice in state.choices],
                },
            )
        if state.task_family == "object_motion_object_movement_counting":
            return (
                "estimate_object_movement_count",
                "根据 object association 的全部 tracks 估计位置变化次数。",
                {
                    "bbox": None,
                    "reference_time": reference_time,
                    "choices": [str(choice) for choice in state.choices],
                },
            )
        if state.task_family == "object_motion_stationary_object_localization":
            return (
                "estimate_stationary_start",
                "根据 object tracks 判断从哪个候选时间开始保持静止超过阈值。",
                {
                    "bbox": None,
                    "reference_time": reference_time,
                    "choices": [str(choice) for choice in state.choices],
                    "threshold_s": 150.0,
                },
            )
        if self._is_object_location_task(state):
            return (
                "infer_object_drop_location",
                "根据目标对象后续轨迹的最终 fixture，推断被放到的位置选项。",
                {
                    "bbox": None,
                    "reference_time": reference_time,
                    "choices": [str(choice) for choice in state.choices],
                    "question": state.question,
                },
            )
        return None

    def _bbox_structured_task_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        combined_times: list[float],
        bbox: Any,
    ) -> PlannerDecision | None:
        if bbox is None or not combined_times:
            return None
        config = self._bbox_structured_task_config(state, combined_times=combined_times)
        if config is None:
            return None
        tool, thought, args = config
        resolve_args = {"bbox": bbox, "reference_time": combined_times[0], "limit": 5}
        if "resolve_bbox_reference" not in used_tools:
            return PlannerDecision(
                thought="该 bbox 驱动任务优先先把参考 bbox 解析成对象 association 和完整轨迹。",
                tool="resolve_bbox_reference",
                args=resolve_args,
            )
        if tool not in used_tools:
            return PlannerDecision(
                thought=thought,
                tool=tool,
                args={**args, "bbox": bbox},
            )
        return None

    def _object_contents_sampling_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        combined_times: list[float],
        bbox: Any,
        reuse_thought: str,
        extract_thought: str,
    ) -> PlannerDecision | None:
        if not self._is_object_contents_task(state) or bbox is None or not combined_times or state.retrieved_frames:
            return None
        return self._build_reuse_or_extract_range_decision(
            state=state,
            used_tools=used_tools,
            tag_hint=state.task_family,
            artifact_prefixes=self._artifact_reuse_prefixes(state),
            start_time=max(0.0, combined_times[0] - 1.0),
            end_time=combined_times[0] + 1.0,
            reuse_thought=reuse_thought,
            extract_thought=extract_thought,
            extract_tag=f"{state.task_family}_contents",
            stride_s=0.5,
            max_frames=3,
        )

    def _object_contents_visual_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        combined_times: list[float],
        bbox: Any,
    ) -> PlannerDecision | None:
        if not self._is_object_contents_task(state) or decision.tool != "finish":
            return None
        if "resolve_bbox_reference" not in used_tools and bbox is not None and combined_times:
            return PlannerDecision(
                thought="容器内容题在 finish 前必须先解析 bbox 引用。",
                tool="resolve_bbox_reference",
                args={"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
            )
        if "infer_visual_mcq" in used_tools:
            return None
        sampling = self._object_contents_sampling_decision(
            state=state,
            used_tools=used_tools,
            combined_times=combined_times,
            bbox=bbox,
            reuse_thought="容器内容题在 finish 前先复用已经生成过的容器关键帧或局部 artifact。",
            extract_thought="容器内容题在 finish 前至少要抽取容器关键帧。",
        )
        if sampling is not None:
            return sampling
        if state.retrieved_frames:
            return PlannerDecision(
                thought="容器内容题在 finish 前必须先做视觉多选判断。",
                tool="infer_visual_mcq",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-3:],
                },
            )
        return None

    def _nutrition_image_step_decision(self, state: AgentState) -> PlannerDecision | None:
        if state.task_family != "nutrition_image_nutrition_estimation":
            return None
        if state.current_step <= 1:
            return PlannerDecision(
                thought="多图营养题先提取 inputs_json 中的跨视频参考图。",
                tool="extract_input_reference_frames",
                args={"tag": f"{state.task_family}_inputs"},
            )
        if state.current_step == 2 and state.retrieved_frames:
            return PlannerDecision(
                thought="先识别每张参考图里展示的食材，避免只按选项名字硬比营养。",
                tool="identify_image_ingredients",
                args={"image_paths": state.retrieved_frames[-10:]},
            )
        if state.current_step == 3:
            nutrient = "carbs" if "carb" in state.question.lower() else "calories"
            return PlannerDecision(
                thought="在图像识别确认后，再比较候选食材的结构化营养字段。",
                tool="compare_choice_nutrition",
                args={"choices": [str(choice) for choice in state.choices], "nutrient": nutrient},
            )
        if state.current_step == 4:
            return PlannerDecision(
                thought="已经得到各候选食材的营养比较结果，直接结束。",
                tool="rank_choices_from_state",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "evidence": state.evidence_bundle,
                    "working_memory": state.working_memory,
                },
            )
        return None

    def _recipe_following_activity_step_decision(
        self,
        *,
        state: AgentState,
        combined_times: list[float],
        recipe_step_hint: Any,
        last_result: dict[str, Any],
    ) -> PlannerDecision | None:
        if not self._is_recipe_following_activity_task(state):
            return None
        if state.current_step <= 1:
            return PlannerDecision(
                thought="高层活动识别题先定位问题对应的 recipe step 时间段。",
                tool="query_event",
                args={
                    "event_types": ["recipe_step", "activity"],
                    "keyword": str(recipe_step_hint or state.question),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 10,
                },
            )
        if state.current_step == 2:
            nodes = last_result.get("nodes", []) if isinstance(last_result, dict) else []
            timed = [
                node for node in nodes
                if isinstance(node, dict) and node.get("start_time") is not None and node.get("end_time") is not None
            ]
            if timed:
                best = max(
                    timed,
                    key=lambda node: (
                        1 if str(node.get("node_type") or "") == "recipe_step" else 0,
                        float(node.get("end_time") or 0.0) - float(node.get("start_time") or 0.0),
                    ),
                )
                start_time = float(best["start_time"])
                end_time = float(best["end_time"])
                return PlannerDecision(
                    thought="已定位到相关 recipe step，回看该时间段关键帧。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": start_time,
                        "end_time": end_time,
                        "stride_s": max(0.5, (end_time - start_time) / 4),
                        "max_frames": 4,
                        "tag": f"{state.task_family}_step_window",
                    },
                )
        if state.current_step >= 3 and state.retrieved_frames:
            return PlannerDecision(
                thought="结合 step 时间段关键帧，判断对应的高层活动。",
                tool="infer_visual_mcq",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-4:],
                },
            )
        return None

    def _weight_task_step_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        combined_times: list[float],
        ingredient_name: Any,
        bbox: Any,
        ocr_keyword: Any,
    ) -> PlannerDecision | None:
        if not self._is_weight_task(state) or not combined_times:
            return None
        if state.current_step <= 1:
            if getattr(state, "retrieved_node_ids", []):
                return PlannerDecision(
                    thought="称重题当前已拿到时间锚点节点，先扩展图关系上下文，优先复用可能已经写回的 OCR/称量证据。",
                    tool="expand_graph_context",
                    args={
                        "node_ids": list(getattr(state, "retrieved_node_ids", [])[-8:]),
                        "edge_types": ["co_occurs", "same_step", "derived_from"],
                        "limit": 16,
                    },
                )
            if "query_ingredient_measurement" not in used_tools and ingredient_name:
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
            if ocr_keyword and "query_ocr" not in used_tools:
                return PlannerDecision(
                    thought="称重题优先检索已有 OCR 记忆，而不是先走位置路径。",
                    tool="query_ocr",
                    args={
                        "keyword": str(ocr_keyword),
                        "start_time": min(combined_times),
                        "end_time": max(combined_times),
                        "limit": 12,
                    },
                )
            if not state.retrieved_frames:
                if "retrieve_cached_artifacts" not in used_tools and self._task_has_reusable_artifacts(
                    state,
                    prefixes=self._artifact_reuse_prefixes(state),
                ):
                    return PlannerDecision(
                        thought="称重题先复用当前视频中已存在的称量帧 artifact，避免重复抽帧。",
                        tool="retrieve_cached_artifacts",
                        args={
                            "tag_hint": state.task_family,
                            "time_s": self._best_reusable_open_query_time(state, combined_times),
                            "start_time": max(0.0, min(combined_times) - 2.0),
                            "end_time": max(combined_times) + 2.0,
                            "limit": 6,
                        },
                    )
                return PlannerDecision(
                    thought="称重题先做稀疏抽帧，补称量过程的原始视觉证据。",
                    tool="sample_sparse_frames",
                    args={
                        "start_time": max(0.0, min(combined_times) - 2.0),
                        "end_time": max(combined_times) + 2.0,
                        "sample_count": 5,
                        "tag": f"{state.task_family}_range",
                    },
                )
        if state.retrieved_frames:
            latest_frame = self._latest_visual_frame(state)
            if latest_frame is None:
                latest_frame = ""
            if bbox and "run_ocr_on_region" not in used_tools:
                return PlannerDecision(
                    thought="称重题优先对候选显示区域做 OCR 读取数字。",
                    tool="run_ocr_on_region",
                    args={
                        "image_path": latest_frame,
                        "bbox": bbox,
                        "expand_ratio": 0.35,
                        "tag": f"{state.task_family}_ocr",
                    },
                )
            if "run_ocr_on_image" not in used_tools:
                return PlannerDecision(
                    thought="称重题先对已取回的称量帧做整图 OCR，优先拿到数字读数。",
                    tool="run_ocr_on_image",
                    args={"image_path": latest_frame},
                )
        if state.current_step >= 2:
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
        return None

    def _nutrition_change_step_decision(
        self,
        *,
        state: AgentState,
        combined_times: list[float],
    ) -> PlannerDecision | None:
        if not self._is_nutrition_change_task(state) or not combined_times:
            return None
        if state.current_step <= 1:
            return PlannerDecision(
                thought="营养变化题优先直接根据 ingredient add 事件计算窗口内营养增量。",
                tool="compute_nutrition_change",
                args={"start_time": min(combined_times), "end_time": max(combined_times)},
            )
        return PlannerDecision(
            thought="已经得到营养增量，直接对选项评分。",
            tool="rank_choices_from_state",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "evidence": state.evidence_bundle,
                "working_memory": state.working_memory,
            },
        )

    def _fixture_interaction_counting_step_decision(
        self,
        *,
        state: AgentState,
        combined_times: list[float],
        last_result: dict[str, Any],
    ) -> PlannerDecision | None:
        if state.task_family != "3d_perception_fixture_interaction_counting":
            return None
        if state.current_step == 5:
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
        if state.current_step == 6:
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
        return None

    def _audio_peak_followup_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        combined_times: list[float],
        last_tool: dict[str, Any],
        last_result: dict[str, Any],
    ) -> PlannerDecision | None:
        if (
            not state.task_family.startswith(("recipe_", "ingredient_", "nutrition_"))
            and state.task_family != "open_query_temporal_summary"
        ) or not combined_times:
            return None
        if state.retrieved_frames and "sample_frames_around_peaks" not in used_tools:
            return None
        if "detect_audio_peaks" not in used_tools:
            return PlannerDecision(
                thought="先检测时间段内的音频峰值，作为后续补证据的候选时间。",
                tool="detect_audio_peaks",
                args={
                    "start_time": max(0.0, min(combined_times) - 2.0),
                    "end_time": max(combined_times) + 2.0,
                    "window_s": 0.5,
                    "top_k": 4,
                },
            )
        last_peak_result = last_result if last_tool.get("tool") == "detect_audio_peaks" and isinstance(last_result, dict) else {}
        peaks = last_peak_result.get("peaks") or []
        peak_times = [
            float(item.get("time_s"))
            for item in peaks
            if isinstance(item, dict) and item.get("time_s") is not None
        ]
        if peak_times and "sample_frames_around_peaks" not in used_tools:
            return PlannerDecision(
                thought="围绕音频峰值再抽取候选关键帧，定位更可能的事件瞬间。",
                tool="sample_frames_around_peaks",
                args={
                    "peak_times": peak_times,
                    "radius_s": 0.7,
                    "frames_per_peak": 3,
                    "tag": f"{state.task_family}_audio_peaks",
                },
            )
        if self._can_use_visual_inspection(state) and state.retrieved_frames and "inspect_visual_evidence" not in used_tools:
            return PlannerDecision(
                thought="先根据峰值附近的关键帧做阶段观察，并写回时间线记忆。",
                tool="inspect_visual_evidence",
                args={
                    "prompt": (
                        "你在看厨房视频中若干候选关键时刻的图片。"
                        "请概括这一小段时间里最可能发生的动作、涉及对象、可能的步骤和状态变化。"
                        '输出 JSON，字段固定为 {"ongoing_action":"","possible_step":"","target_object":"","state_change_hint":"","answer_hint":"","confidence":0.0}。'
                    ),
                    "image_paths": state.retrieved_frames[-12:],
                },
            )
        if state.evidence_bundle or state.working_memory:
            return PlannerDecision(
                thought="把当前阶段的总结写回图谱，供后续问题复用。",
                tool="write_timeline_summary",
                args={
                    "label": f"{state.task_family} timeline summary",
                    "start_time": max(0.0, min(combined_times) - 2.0),
                    "end_time": max(combined_times) + 2.0,
                    "summary": " | ".join(state.evidence_bundle[-4:] or state.working_memory[-4:]),
                    "evidence_paths": state.retrieved_frames[-12:],
                    "keywords": [state.task_family, "timeline", "audio_peak"],
                },
            )
        return None

    def _bbox_visual_finalize_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        bbox: Any,
    ) -> PlannerDecision | None:
        if (
            decision.tool != "finish"
            or bbox is None
            or not state.task_family.startswith(("object_motion_", "3d_perception_", "gaze_"))
            or state.task_family == "3d_perception_object_contents_retrieval"
        ):
            return None
        if "render_bbox_overlay" not in used_tools and state.retrieved_frames:
            latest_frame = self._latest_visual_frame(state)
            if latest_frame is None:
                return None
            return PlannerDecision(
                thought="bbox 题在 finish 前至少要画一次框确认目标。",
                tool="render_bbox_overlay",
                args={"image_path": latest_frame, "bbox": bbox, "tag": f"{state.task_family}_bbox"},
            )
        if self._can_use_visual_inspection(state) and "inspect_visual_evidence" not in used_tools and state.retrieved_frames:
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
        return None

    def _maybe_fast_finish_weight_task(self, *, state: AgentState, decision: PlannerDecision) -> PlannerDecision:
        if not self._is_weight_task(state):
            return self._maybe_fast_finish_viewpoint_task(state=state, decision=decision)
        if decision.tool == "finish":
            return decision
        resolved = self._resolve_weight_choice_from_state(state)
        if resolved is None:
            return decision
        if state.current_step < 1:
            return decision
        prediction, answer, confidence, source = resolved
        if confidence < 0.78:
            return decision
        return PlannerDecision(
            thought=f"称重题已经从现有 {source} 证据中稳定解析出答案，直接结束。",
            tool="finish",
            args={"prediction": prediction, "answer": answer, "confidence": confidence},
            done=True,
            answer=answer,
            prediction=prediction,
            confidence=confidence,
        )

    def _maybe_fast_finish_viewpoint_task(self, *, state: AgentState, decision: PlannerDecision) -> PlannerDecision:
        if not self._is_viewpoint_task(state):
            return self._maybe_fast_finish_object_motion_task(state=state, decision=decision)
        if decision.tool == "finish":
            return decision
        resolved = self._resolve_viewpoint_choice_from_state(state)
        if resolved is None:
            return decision
        prediction, answer, confidence, source = resolved
        if confidence < 0.72:
            return decision
        return PlannerDecision(
            thought=f"视角/空间题已经从现有 {source} 证据中稳定解析出答案，直接结束。",
            tool="finish",
            args={"prediction": prediction, "answer": answer, "confidence": confidence},
            done=True,
            answer=answer,
            prediction=prediction,
            confidence=confidence,
        )

    def _maybe_fast_finish_object_motion_task(self, *, state: AgentState, decision: PlannerDecision) -> PlannerDecision:
        if not self._is_object_motion_task(state):
            return self._maybe_fast_finish_object_location_task(state=state, decision=decision)
        if decision.tool == "finish":
            return decision
        resolved = self._resolve_object_motion_choice_from_state(state)
        if resolved is None:
            return decision
        prediction, answer, confidence, source = resolved
        if confidence < 0.74:
            return decision
        return PlannerDecision(
            thought=f"物体运动题已经从现有 {source} 证据中稳定解析出答案，直接结束。",
            tool="finish",
            args={"prediction": prediction, "answer": answer, "confidence": confidence},
            done=True,
            answer=answer,
            prediction=prediction,
            confidence=confidence,
        )

    def _maybe_fast_finish_object_location_task(self, *, state: AgentState, decision: PlannerDecision) -> PlannerDecision:
        if not self._is_object_location_task(state):
            return decision
        if decision.tool == "finish":
            return decision
        resolved = self._resolve_object_location_choice_from_state(state)
        if resolved is None:
            return decision
        prediction, answer, confidence, source = resolved
        if confidence < 0.7:
            return decision
        return PlannerDecision(
            thought=f"物体放置位置题已经从现有 {source} 证据中稳定解析出答案，直接结束。",
            tool="finish",
            args={"prediction": prediction, "answer": answer, "confidence": confidence},
            done=True,
            answer=answer,
            prediction=prediction,
            confidence=confidence,
        )

    def _resolve_viewpoint_choice_from_state(self, state: AgentState) -> tuple[int, str, float, str] | None:
        best_index = None
        target_match = ""
        ranked_best_index = None
        gaze_best_index = None
        gaze_confidence = None
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            if item.startswith("fixture_direction_best_index="):
                match = re.search(r"fixture_direction_best_index=(\d+)", item)
                if match:
                    best_index = int(match.group(1))
                target_part = item.split("target_match=", 1)[1] if "target_match=" in item else ""
                target_match = str(target_part or "").strip()
            if item.startswith("ranked_best_index="):
                match = re.search(r"ranked_best_index=(\d+)", item)
                if match:
                    ranked_best_index = int(match.group(1))
            if item.startswith("gaze_best_index="):
                match = re.search(r"gaze_best_index=(\d+)", item)
                if match:
                    gaze_best_index = int(match.group(1))
                conf_match = re.search(r"confidence=([0-9.]+)", item)
                if conf_match:
                    try:
                        gaze_confidence = float(conf_match.group(1))
                    except Exception:  # noqa: BLE001
                        gaze_confidence = None
        if best_index is not None and 0 <= best_index < len(state.choices):
            answer = str(state.choices[best_index])
            if ranked_best_index is None or ranked_best_index == best_index:
                confidence = 0.82 if target_match else 0.74
                return best_index, answer, confidence, "viewpoint_structured"
        if gaze_best_index is not None and 0 <= gaze_best_index < len(state.choices):
            answer = str(state.choices[gaze_best_index])
            confidence = max(0.74, float(gaze_confidence or 0.0))
            return gaze_best_index, answer, confidence, "gaze_structured"
        return None

    def _resolve_object_motion_choice_from_state(self, state: AgentState) -> tuple[int, str, float, str] | None:
        if not self._is_object_motion_task(state):
            return None
        movement_best_index = None
        movement_count = None
        stationary_best_index = None
        itinerary_best_index = None
        itinerary_confidence = None
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            if item.startswith("movement_count="):
                best_match = re.search(r"best_index=(\d+)", item)
                count_match = re.search(r"movement_count=([0-9.]+)", item)
                if best_match:
                    movement_best_index = int(best_match.group(1))
                if count_match:
                    try:
                        movement_count = float(count_match.group(1))
                    except Exception:  # noqa: BLE001
                        movement_count = None
            if item.startswith("stationary_best_index="):
                best_match = re.search(r"stationary_best_index=(\d+)", item)
                if best_match:
                    stationary_best_index = int(best_match.group(1))
            if item.startswith("itinerary_best_index="):
                best_match = re.search(r"itinerary_best_index=(\d+)", item)
                conf_match = re.search(r"confidence=([0-9.]+)", item)
                if best_match:
                    itinerary_best_index = int(best_match.group(1))
                if conf_match:
                    try:
                        itinerary_confidence = float(conf_match.group(1))
                    except Exception:  # noqa: BLE001
                        itinerary_confidence = None
        if movement_best_index is not None and 0 <= movement_best_index < len(state.choices):
            answer = str(state.choices[movement_best_index])
            confidence = 0.86 if movement_count and movement_count > 0 else 0.74
            return movement_best_index, answer, confidence, "object_motion_structured"
        if itinerary_best_index is not None and 0 <= itinerary_best_index < len(state.choices):
            answer = str(state.choices[itinerary_best_index])
            return itinerary_best_index, answer, max(0.72, float(itinerary_confidence or 0.0)), "object_itinerary_structured"
        if stationary_best_index is not None and 0 <= stationary_best_index < len(state.choices):
            answer = str(state.choices[stationary_best_index])
            return stationary_best_index, answer, 0.82, "stationary_structured"
        return None

    def _resolve_object_location_choice_from_state(self, state: AgentState) -> tuple[int, str, float, str] | None:
        if not self._is_object_location_task(state):
            return None
        best_index = None
        confidence = None
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            if item.startswith("object_location_best_index="):
                best_match = re.search(r"object_location_best_index=(\d+)", item)
                conf_match = re.search(r"confidence=([0-9.]+)", item)
                if best_match:
                    best_index = int(best_match.group(1))
                if conf_match:
                    try:
                        confidence = float(conf_match.group(1))
                    except Exception:  # noqa: BLE001
                        confidence = None
        if best_index is not None and 0 <= best_index < len(state.choices):
            return best_index, str(state.choices[best_index]), max(0.72, float(confidence or 0.0)), "object_location_structured"
        return None

    def _has_stable_weight_answer_evidence(self, state: AgentState) -> bool:
        resolved = self._resolve_weight_choice_from_state(state)
        return resolved is not None and resolved[2] >= 0.78

    def _resolve_weight_choice_from_state(self, state: AgentState) -> tuple[int, str, float, str] | None:
        choice_values: list[tuple[int, float, str]] = []
        for index, choice in enumerate(state.choices):
            parsed = self._parse_numeric_value(str(choice))
            if parsed is None:
                continue
            choice_values.append((index, parsed, str(choice)))
        if not choice_values:
            return None
        ocr_values = self._extract_weight_values(state, prefixes=("ocr_reading=",))
        measurement_values = self._extract_measurement_values(state)
        if measurement_values:
            best = self._pick_best_numeric_choice(choice_values, measurement_values[-1])
            if best is not None:
                return best[0], best[2], 0.9, "measurement"
        if ocr_values:
            best = self._pick_best_numeric_choice(choice_values, ocr_values[-1])
            if best is not None:
                return best[0], best[2], 0.82, "ocr"
        return None

    def _extract_weight_values(self, state: AgentState, *, prefixes: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            for prefix in prefixes:
                if prefix not in item:
                    continue
                parsed = self._parse_numeric_value(item.split(prefix, 1)[1])
                if parsed is not None:
                    values.append(parsed)
        return values

    def _extract_measurement_values(self, state: AgentState) -> list[float]:
        values: list[float] = []
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str) or "measurement " not in item or "normalized=" not in item:
                continue
            parsed = self._parse_numeric_value(item.split("normalized=", 1)[1])
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

    def _safe_fallback_decision(self, *, state: AgentState, hints: dict[str, Any]) -> PlannerDecision:
        used_tools = self._used_tools(state)
        recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
        if isinstance(recovered, PlannerDecision):
            self._state_add_memory(state, "planner_guard=none_decision_recovered")
            return recovered
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        if combined_times:
            return PlannerDecision(
                thought="规划器未产出有效决策，退回到安全时间检索路径。",
                tool="query_time",
                args={
                    "start_time": min(combined_times),
                    "end_time": max(combined_times),
                    "limit": 12,
                },
            )
        return PlannerDecision(
            thought="规划器未产出有效决策，退回到基于当前证据的安全评分路径。",
            tool="rank_choices_from_state",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "evidence": state.evidence_bundle,
                "working_memory": state.working_memory,
            },
        )

    def _heuristic_fallback(self, *, state: AgentState, hints: dict[str, Any]) -> PlannerDecision:
        last_tool = state.tool_trace[-1] if state.tool_trace else {}
        last_result = last_tool.get("raw_result") if isinstance(last_tool, dict) else {}
        used_tools = [entry.get("tool") for entry in state.tool_trace if isinstance(entry, dict)]
        open_questions = list(getattr(state, "open_questions", []) or [])
        latest_verification = self._state_latest_verification(state)
        verifier_conflicts = {
            str(item)
            for item in latest_verification.get("conflicts", [])
            if isinstance(item, str) and item
        }
        if verifier_conflicts:
            recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
            if recovered.tool:
                self._state_add_memory(state, f"conflict_recovery_selected tool={recovered.tool}")
                return recovered
        viewpoint_decision = self._preferred_viewpoint_task_decision(
            state=state,
            hints=hints,
            used_tools=used_tools,
        )
        if viewpoint_decision is not None:
            return viewpoint_decision
        if (
            self._is_action_intent_task(state)
            and isinstance(last_result, dict)
            and last_tool.get("tool") in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}
            and last_result.get("tool_failed")
        ):
            failed_tool = str(last_tool.get("tool") or "")
            if self._action_intent_failed_tool_count(state, failed_tool) <= 1:
                if failed_tool == "resolve_action_intent_future_use":
                    retry_future_use = self._build_action_intent_future_use_resolution_decision(
                        state=state,
                        hints=hints,
                        result=last_result,
                        thought="why 题后续用途裁决工具失败，先用干净上下文重试同一专用裁决，不直接用上一轮五选一结果收口。",
                    )
                    if retry_future_use is not None:
                        return retry_future_use
                if failed_tool == "resolve_action_intent_pairwise":
                    retry_pairwise = self._build_action_intent_pairwise_resolution_decision(
                        state=state,
                        hints=hints,
                        result=last_result,
                        thought="why 题二选一后果裁决工具失败，先用干净上下文重试同一专用裁决，不直接用上一轮五选一结果收口。",
                    )
                    if retry_pairwise is not None:
                        return retry_pairwise
            recovered_intent = self._latest_successful_action_intent_result(state)
            if recovered_intent.get("best_index") is not None:
                best_index = int(recovered_intent["best_index"])
                return PlannerDecision(
                    thought="why 题专用裁决工具连续失败，保留最近一次当前题专用动作目的判断，避免退回通用视觉检查混入旧帧。",
                    tool="finish",
                    args={
                        "prediction": best_index,
                        "answer": str(recovered_intent.get("answer") or state.choices[best_index]),
                        "confidence": float(recovered_intent.get("confidence") or 0.0),
                    },
                    done=True,
                    answer=str(recovered_intent.get("answer") or state.choices[best_index]),
                    prediction=best_index,
                    confidence=float(recovered_intent.get("confidence") or 0.0),
                )
            action_frames = self._select_action_intent_frames(
                state,
                hints,
                limit=8,
                require_current_scope=True,
            )
            if action_frames:
                return PlannerDecision(
                    thought="why 题专用裁决工具失败，回到当前题时间窗的专用动作目的判断，不使用通用视觉检查。",
                    tool="infer_action_intent",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": action_frames,
                        "context_notes": self._action_intent_context_notes(state, limit=12),
                    },
                )
        if (
            self._is_action_intent_task(state)
            and isinstance(last_result, dict)
            and last_tool.get("tool") == "infer_action_intent"
            and last_result.get("tool_failed")
        ):
            retry_count = self._action_intent_failed_tool_count(state, "infer_action_intent")
            if retry_count <= 2:
                recovered = self._build_action_intent_specialized_recovery_decision(
                    state=state,
                    hints=hints,
                    thought="why 题专用动作目的判断请求失败，直接重试当前题时间窗的专用判断，不退回通用视觉检查或旧 artifact。",
                )
                if recovered is not None:
                    return recovered
            if retry_count >= 3:
                return self._build_action_intent_text_fallback_rank_decision(
                    state,
                    thought="why 题专用视觉判断连续失败，改用结构化文本因果裁决，避免继续空转 query_time。",
                )
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
            if (
                self._action_intent_needs_precondition_context(state=state, result=last_result)
                and not self._action_intent_has_precondition_frames(state=state, hints=hints)
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(last_result.get("followup_focus") or "precondition_before_future_use"),
                )
                if precondition is not None:
                    return precondition
            if self._action_intent_requires_followup(state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 1:
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                if self._action_intent_needs_future_use_evidence(state=state, result=last_result):
                    future_use = self._build_action_intent_future_use_resolution_decision(
                        state=state,
                        hints=hints,
                        result=last_result,
                        thought="why 题已补过动作后的用途帧，必须先逐项验证后续用途证据，不能直接用五选一视觉猜测收口。",
                    )
                    if future_use is not None:
                        return future_use
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题已补过一轮结果帧，改为只在前两名歧义候选之间做最终裁决。",
                )
                if pairwise is not None:
                    return pairwise
                return PlannerDecision(
                    thought="why 题已补过一轮动作后结果帧，仍有歧义；改为基于累计证据做聚合评分收口。",
                    tool="rank_choices_from_state",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "evidence": state.evidence_bundle,
                        "working_memory": state.working_memory,
                    },
                )
            if self._action_intent_pair_needs_outcome_resolution(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 1:
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题 top-2 仍是动作后果型歧义，不能仅凭高置信直接结束；改为结合结果帧二选一裁决。",
                )
                if pairwise is not None:
                    return pairwise
            if self._action_intent_needs_future_use_evidence(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 1:
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题目的依赖动作后用途，必须显式验证后续用途证据后才能结束。",
                )
                if future_use is not None:
                    return future_use
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
        if (
            isinstance(last_result, dict)
            and last_tool.get("tool") == "inspect_visual_evidence"
            and not str(getattr(state, "task_family", "")).startswith("open_query")
        ):
            if self._is_action_mechanism_task(state) and state.retrieved_frames:
                return PlannerDecision(
                    thought="how 题已经拿到关键帧，视觉检查后直接进入动作机制判断。",
                    tool="infer_action_mechanism",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-4:],
                    },
                )
            if self._is_action_intent_task(state) and state.retrieved_frames:
                action_frames = self._select_action_intent_frames(
                    state,
                    hints,
                    limit=4,
                    include_followup=False,
                    require_current_scope=True,
                )
                if not action_frames:
                    return self._segment_task_sampling_decision(
                        state=state,
                        used_tools=used_tools,
                        combined_times=sorted(
                            [float(value) for value in hints.get("times") or []]
                            + [float(value) for value in hints.get("input_times") or []]
                        ),
                        reuse_thought="why 题先检索当前动作片段 artifact。",
                        extract_thought="why 题先抽当前动作时间窗关键帧。",
                    )
                context_notes = self._action_intent_context_notes(state, limit=10)
                return PlannerDecision(
                    thought="why 题已经拿到关键帧，视觉检查后直接进入动作目的判断。",
                    tool="infer_action_intent",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": action_frames,
                        "context_notes": context_notes,
                    },
                )
        if (
            isinstance(last_result, dict)
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "retrieve_cached_artifacts"}
            and self._is_action_intent_task(state)
            and state.retrieved_frames
            and self._action_intent_has_precondition_frames(state=state, hints=hints)
            and not self._action_intent_requires_followup(state)
        ):
            action_frames = self._select_action_intent_frames(
                state,
                hints,
                limit=8,
                require_current_scope=True,
            )
            if action_frames:
                return PlannerDecision(
                    thought="why 题已补到动作前上下文，重新结合前置状态和当前动作判断目的。",
                    tool="infer_action_intent",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": action_frames,
                        "context_notes": self._action_intent_context_notes(state, limit=12),
                    },
                )
        if (
            isinstance(last_result, dict)
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "retrieve_cached_artifacts"}
            and self._is_action_intent_task(state)
            and state.retrieved_frames
            and self._action_intent_pending_resolution_tool(state)
        ):
            pending_tool = self._action_intent_pending_resolution_tool(state)
            if (
                self._action_intent_needs_precondition_context(state=state, result=None)
                and not self._action_intent_has_precondition_frames(state=state, hints=hints)
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=(
                        "precondition_before_pending_resolution"
                        if pending_tool
                        else "precondition_before_action_intent_resolution"
                    ),
                )
                if precondition is not None:
                    return precondition
            if pending_tool == "resolve_action_intent_future_use":
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题已补到额外后续帧，回到后续用途裁决，逐项检查动作后真实用途。",
                )
                if future_use is not None:
                    return future_use
            if pending_tool == "resolve_action_intent_pairwise":
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题已补到额外后续帧，回到二选一后果裁决，检查后续是否能排除竞争选项。",
                )
                if pairwise is not None:
                    return pairwise
        if (
            isinstance(last_result, dict)
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "retrieve_cached_artifacts"}
            and self._is_action_intent_task(state)
            and state.retrieved_frames
            and self._action_intent_requires_followup(state)
            and self._action_intent_followup_attempt_count(state) <= 1
        ):
            action_frames = self._select_action_intent_frames(
                state,
                hints,
                limit=8,
                require_current_scope=True,
            )
            if not action_frames:
                followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                if followup is not None:
                    return followup
            if not self._action_intent_has_post_action_frames(state=state, hints=hints, frames=action_frames):
                followup = self._build_action_intent_missing_post_action_followup_decision(
                    state=state,
                    hints=hints,
                    action_frames=action_frames,
                    focus="second_intent_pass_needs_post_action_frames",
                )
                if followup is not None:
                    return followup
            context_notes = self._action_intent_context_notes(state, limit=12)
            return PlannerDecision(
                thought="why 题已补到动作后结果帧，重新结合扩展时序证据判断动作目的。",
                tool="infer_action_intent",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": action_frames,
                    "context_notes": context_notes,
                },
            )
        if isinstance(last_result, dict) and last_tool.get("tool") == "resolve_action_intent_pairwise" and last_result.get("best_index") is not None:
            if (
                self._action_intent_resolution_needs_more_evidence(
                    tool_name="resolve_action_intent_pairwise",
                    result=last_result,
                )
                and self._action_intent_followup_attempt_count(state) < 3
            ):
                extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(last_result.get("needed_observation") or "pairwise_outcome_resolution"),
                )
                if extra_followup is not None:
                    return extra_followup
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="why 题二选一裁决已完成，直接结束。",
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
        if isinstance(last_result, dict) and last_tool.get("tool") == "resolve_action_intent_future_use" and last_result.get("best_index") is not None:
            if self._action_intent_resolution_should_backfill_precondition(
                state=state,
                hints=hints,
                result=last_result,
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(last_result.get("needed_observation") or "precondition_before_additional_followup"),
                )
                if precondition is not None:
                    return precondition
            if (
                self._action_intent_resolution_needs_more_evidence(
                    tool_name="resolve_action_intent_future_use",
                    result=last_result,
                )
                and self._action_intent_followup_attempt_count(state) < 3
            ):
                extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(last_result.get("needed_observation") or "future_use_resolution"),
                )
                if extra_followup is not None:
                    return extra_followup
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="why 题后续用途证据裁决已完成，直接结束。",
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
        if isinstance(last_result, dict) and last_tool.get("tool") == "infer_temporal_localization_choice" and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="时间定位判断已完成，直接结束。",
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
        if isinstance(last_result, dict) and last_tool.get("tool") in {
            "infer_ingredient_order_choice",
            "infer_ingredient_retrieval_choice",
            "infer_recipe_ingredient_membership_choice",
            "infer_exact_ingredient_amount_choice",
            "infer_recipe_catalog_choice",
            "infer_recipe_nutrition_choice",
        } and last_result.get("best_index") is not None:
            best_index = int(last_result["best_index"])
            return PlannerDecision(
                thought="结构化专用判别已完成，直接结束。",
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
            if self._action_intent_text_fallback_ready(state):
                best_index = int(last_result["best_index"])
                return PlannerDecision(
                    thought="why 题专用视觉判断连续失败后，结构化文本因果裁决已完成，直接结束。",
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
            if self._has_unresolved_evidence_gap(open_questions) and float(last_result.get("confidence") or 0.0) < 0.8:
                return self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
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
        candidate = self._select_state_driven_candidate(state=state, hints=hints, used_tools=used_tools)
        if candidate is not None:
            return candidate
        times = [float(value) for value in hints.get("times") or []]
        input_times = [float(value) for value in hints.get("input_times") or []]
        combined_times = sorted(times + input_times)
        bbox = hints.get("bbox")
        ingredient_name = hints.get("ingredient_name")
        recipe_step_hint = hints.get("recipe_step_hint")
        state_keyword = hints.get("state_keyword")
        location_keyword = hints.get("location_keyword")
        ocr_keyword = hints.get("ocr_keyword")
        object_hint = hints.get("object_hint")
        explicit_location_need = self._question_explicitly_mentions_location(state=state, location_keyword=location_keyword)
        direct_structured = self._structured_direct_inference_config(state)
        if state.current_step <= 1 and direct_structured is not None:
            tool, thought, args = direct_structured
            return PlannerDecision(thought=thought, tool=tool, args=args)
        if state.current_step == 0 and combined_times:
            return PlannerDecision(
                thought="先查题目时间窗口附近的图谱节点。",
                tool="query_time",
                args={"start_time": min(combined_times), "end_time": max(combined_times), "limit": 20},
            )
        recipe_following_step = self._recipe_following_activity_step_decision(
            state=state,
            combined_times=combined_times,
            recipe_step_hint=recipe_step_hint,
            last_result=last_result if isinstance(last_result, dict) else {},
        )
        if recipe_following_step is not None:
            return recipe_following_step
        nutrition_change_step = self._nutrition_change_step_decision(
            state=state,
            combined_times=combined_times,
        )
        if nutrition_change_step is not None:
            return nutrition_change_step
        if state.current_step <= 1 and state_keyword and "query_state" not in used_tools:
            return PlannerDecision(
                thought="问题明显涉及状态变化，先检索已写回或已索引的状态证据。",
                tool="query_state",
                args={
                    "state_keyword": str(state_keyword),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
        weight_step = self._weight_task_step_decision(
            state=state,
            used_tools=used_tools,
            combined_times=combined_times,
            ingredient_name=ingredient_name,
            bbox=bbox,
            ocr_keyword=ocr_keyword,
        )
        if weight_step is not None:
            return weight_step
        nutrition_image_step = self._nutrition_image_step_decision(state)
        if nutrition_image_step is not None:
            return nutrition_image_step
        bbox_structured = self._bbox_structured_task_decision(
            state=state,
            used_tools=used_tools,
            combined_times=combined_times,
            bbox=bbox,
        )
        if state.current_step <= 2 and bbox_structured is not None:
            return bbox_structured
        if self._is_object_contents_task(state) and state.current_step <= 3:
            contents_sampling = self._object_contents_sampling_decision(
                state=state,
                used_tools=used_tools,
                combined_times=combined_times,
                bbox=bbox,
                reuse_thought="容器内容题优先复用已存在的容器关键帧或局部 artifact。",
                extract_thought="容器内容题在锁定 bbox 引用后，抽参考时刻及其附近帧查看容器内部/取放交互。",
            )
            if contents_sampling is not None:
                return contents_sampling
            if state.retrieved_frames:
                return PlannerDecision(
                    thought="直接根据容器相关关键帧做视觉多选判断。",
                    tool="infer_visual_mcq",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                    },
                )
        if self._is_segment_visual_task(state) and state.current_step <= 2:
            segment_sampling = self._segment_task_sampling_decision(
                state=state,
                used_tools=used_tools,
                combined_times=combined_times,
                reuse_thought="片段类题先检索当前视频中已存在的片段关键帧 artifact，优先复用已有抽帧。",
                extract_thought="先为短视频片段抽取按时间顺序排列的关键帧。",
            )
            if segment_sampling is not None:
                if state.task_family == "fine_grained_action_recognition":
                    return segment_sampling
                if state.current_step == 1 and segment_sampling.tool == "extract_frames_for_range":
                    return PlannerDecision(
                        thought=segment_sampling.thought,
                        tool="sample_sparse_frames",
                        args={
                            "start_time": segment_sampling.args["start_time"],
                            "end_time": segment_sampling.args["end_time"],
                            "sample_count": 4,
                            "tag": f"{state.task_family}_segment",
                        },
                    )
                return segment_sampling
        audio_peak_followup = self._audio_peak_followup_decision(
            state=state,
            used_tools=used_tools,
            combined_times=combined_times,
            last_tool=last_tool,
            last_result=last_result,
        )
        if audio_peak_followup is not None:
            return audio_peak_followup
        segment_inference = self._segment_task_inference_config(state, {"times": combined_times, "input_times": []})
        if (
            state.current_step == 2
            and segment_inference is not None
            and self._filter_visual_image_paths(list(segment_inference[2].get("image_paths") or []))
        ):
            tool, thought, args = segment_inference
            return PlannerDecision(thought=thought, tool=tool, args=args)
        if state.current_step == 2 and bbox and state.retrieved_frames:
            latest_frame = self._latest_visual_frame(state)
            if latest_frame is not None:
                return PlannerDecision(
                    thought="对参考帧画出 bbox，保留原图上下文。",
                    tool="render_bbox_overlay",
                    args={"image_path": latest_frame, "bbox": bbox, "tag": f"{state.task_family}_bbox"},
                )
        if state.current_step == 3 and bbox and state.retrieved_frames:
            latest_frame = self._latest_visual_frame(state)
            if latest_frame is not None:
                return PlannerDecision(
                    thought="放大 bbox 区域辅助识别目标物体。",
                    tool="extract_region_with_context",
                    args={"image_path": latest_frame, "bbox": bbox, "expand_ratio": 0.35, "tag": f"{state.task_family}_crop"},
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
        fixture_counting_step = self._fixture_interaction_counting_step_decision(
            state=state,
            combined_times=combined_times,
            last_result=last_result if isinstance(last_result, dict) else {},
        )
        if fixture_counting_step is not None:
            return fixture_counting_step
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
                tool="sample_sparse_frames",
                args={
                    "start_time": max(0.0, min(combined_times) - 2.0),
                    "end_time": max(combined_times) + 2.0,
                    "sample_count": 4,
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

    def _preferred_viewpoint_task_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> PlannerDecision | None:
        if state.task_family not in {"3d_perception_fixture_location", "gaze_gaze_estimation"}:
            return None
        resolved = self._resolve_viewpoint_choice_from_state(state)
        if resolved is not None:
            prediction, answer, confidence, source = resolved
            return PlannerDecision(
                thought=f"视角/注视题已经从现有 {source} 证据中稳定解析出答案，直接结束。",
                tool="finish",
                args={"prediction": prediction, "answer": answer, "confidence": confidence},
                done=True,
                answer=answer,
                prediction=prediction,
                confidence=confidence,
            )
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        if not state.retrieved_frames and combined_times:
            reuse_or_extract = self._build_reuse_or_extract_range_decision(
                state=state,
                used_tools=used_tools,
                tag_hint=state.task_family,
                artifact_prefixes=self._artifact_reuse_prefixes(state),
                start_time=max(0.0, min(combined_times) - 0.5),
                end_time=max(combined_times) + 0.5,
                reuse_thought="视角类题先检索当前视频中已经抽取过的视角 artifact，优先复用已有帧。",
                extract_thought="视角定位题必须先抽当前视角关键帧。",
                extract_tag=f"{state.task_family}_view",
                stride_s=0.5,
                max_frames=3,
            )
            if reuse_or_extract is not None:
                return reuse_or_extract
        if "query_spatial_context" not in used_tools and combined_times:
            thought = "fixture 方位题在 finish 前必须先查询附近的空间候选。" if state.task_family == "3d_perception_fixture_location" else "注视目标题在 finish 前必须先查询该时刻的空间上下文。"
            return PlannerDecision(
                thought=thought,
                tool="query_spatial_context",
                args={"time_s": combined_times[0], "object_name": None, "limit": 12},
            )
        if state.retrieved_frames:
            if state.task_family == "3d_perception_fixture_location" and "infer_named_fixture_direction" not in used_tools:
                last_spatial = next(
                    (
                        entry.get("raw_result")
                        for entry in reversed(state.tool_trace)
                        if isinstance(entry, dict) and entry.get("tool") == "query_spatial_context"
                    ),
                    {},
                )
                return PlannerDecision(
                    thought="视角定位题在 finish 前必须先做具名 fixture 方向判断。",
                    tool="infer_named_fixture_direction",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                        "spatial_context": last_spatial if isinstance(last_spatial, dict) else {},
                    },
                )
            if state.task_family == "gaze_gaze_estimation" and "infer_gaze_target_with_context" not in used_tools:
                last_spatial = next(
                    (
                        entry.get("raw_result")
                        for entry in reversed(state.tool_trace)
                        if isinstance(entry, dict) and entry.get("tool") == "query_spatial_context"
                    ),
                    {},
                )
                spatial_context = last_spatial if isinstance(last_spatial, dict) else {}
                return PlannerDecision(
                    thought="注视目标题在 finish 前必须结合视角帧和空间上下文完成判断。",
                    tool="infer_gaze_target_with_context",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                        "spatial_context": spatial_context,
                    },
                )
        return None
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
                tool="sample_sparse_frames",
                args={
                    "start_time": max(0.0, min(combined_times) - 2.0),
                    "end_time": max(combined_times) + 2.0,
                    "sample_count": 4,
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

    def _recover_if_low_confidence(self, *, state: AgentState, hints: dict[str, Any], decision: PlannerDecision) -> PlannerDecision:
        open_questions = list(getattr(state, "open_questions", []) or [])
        latest_verification = self._state_latest_verification(state)
        if latest_verification and decision.tool == "finish":
            if not bool(latest_verification.get("sufficient")):
                recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=self._used_tools(state))
                if recovered is not None and recovered.tool != decision.tool:
                    self._state_add_memory(state, f"planner_override verifier_blocked_finish={decision.tool} -> {recovered.tool}")
                    return recovered
                return decision
        if not self._has_unresolved_evidence_gap(open_questions, task_family=state.task_family):
            return decision
        if decision.tool == "finish":
            if self._is_ingredient_order_task(state):
                return decision
            if self._is_ingredient_retrieval_task(state):
                return decision
            if self._is_recipe_ingredient_membership_task(state):
                return decision
            if self._is_exact_ingredient_amount_task(state):
                return decision
            if self._is_recipe_catalog_task(state):
                return decision
            if self._is_recipe_nutrition_task(state):
                return decision
            if self._is_temporal_localization_task(state):
                return decision
            if self._is_object_contents_task(state):
                return decision
            if self._is_object_location_task(state):
                resolved = self._resolve_object_location_choice_from_state(state)
                if resolved is not None and resolved[2] >= 0.72:
                    return decision
            if self._is_object_motion_task(state):
                resolved = self._resolve_object_motion_choice_from_state(state)
                if resolved is not None and resolved[2] >= 0.74:
                    return decision
        if decision.tool == "finish" and str(state.task_family).startswith("open_query"):
            recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=self._used_tools(state))
            if recovered is not None and recovered.tool != decision.tool:
                self._state_add_memory(state, f"planner_override open_query_gap_finish={decision.tool} -> {recovered.tool}")
                return recovered
        last_tool = state.tool_trace[-1] if state.tool_trace else {}
        last_result = last_tool.get("raw_result") if isinstance(last_tool, dict) else {}
        used_tools = [entry.get("tool") for entry in state.tool_trace if isinstance(entry, dict)]
        if decision.tool == "finish" and decision.confidence < 0.8:
            if (
                self._is_ingredient_order_task(state)
                or self._is_ingredient_retrieval_task(state)
                or self._is_recipe_ingredient_membership_task(state)
                or self._is_exact_ingredient_amount_task(state)
                or self._is_action_mechanism_task(state)
                or self._is_action_intent_task(state)
                or self._is_recipe_catalog_task(state)
                or self._is_recipe_nutrition_task(state)
            ):
                return decision
            if self._is_viewpoint_task(state):
                recovered = self._recover_viewpoint_low_confidence(state=state, hints=hints, used_tools=used_tools)
                if recovered is not None and recovered.tool != decision.tool:
                    self._state_add_memory(state, f"planner_override low_conf_finish={decision.tool} -> {recovered.tool}")
                    return recovered
            if self._is_recipe_following_activity_task(state) or self._is_nutrition_change_task(state):
                recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
                if recovered is not None and recovered.tool != decision.tool:
                    self._state_add_memory(state, f"planner_override low_conf_finish={decision.tool} -> {recovered.tool}")
                    return recovered
            recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
            if recovered is not None and recovered.tool != decision.tool:
                self._state_add_memory(state, f"planner_override low_conf_finish={decision.tool} -> {recovered.tool}")
                return recovered
            return decision
        if decision.tool == "rank_choices_from_state":
            if self._action_intent_text_fallback_ready(state):
                return decision
            if isinstance(last_result, dict) and last_tool.get("tool") == "rank_choices_from_state":
                if float(last_result.get("confidence") or 0.0) < 0.8:
                    recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
                    if recovered is not None and recovered.tool != decision.tool:
                        self._state_add_memory(state, f"planner_override repeated_loop={decision.tool} -> {recovered.tool}")
                        return recovered
                    return decision
            elif decision.confidence < 0.8:
                if self._is_recipe_following_activity_task(state) or self._is_nutrition_change_task(state):
                    recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
                    if recovered is not None and recovered.tool != decision.tool:
                        self._state_add_memory(state, f"planner_override low_conf_rank={decision.tool} -> {recovered.tool}")
                        return recovered
                    return decision
                recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
                if recovered is not None and recovered.tool != decision.tool:
                    self._state_add_memory(state, f"planner_override low_conf_rank={decision.tool} -> {recovered.tool}")
                    return recovered
                return decision
        return decision

    def _recover_viewpoint_low_confidence(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> PlannerDecision:
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        if state.task_family == "3d_perception_fixture_location":
            if "infer_named_fixture_direction" not in used_tools and state.retrieved_frames:
                last_spatial = self._latest_tool_result(state, "query_spatial_context")
                return PlannerDecision(
                    thought="视角定位低置信时，先补具名 fixture 专用方向判断。",
                    tool="infer_named_fixture_direction",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                        "spatial_context": last_spatial if isinstance(last_spatial, dict) else {},
                    },
                )
            if "rank_choices_from_state" not in used_tools:
                return PlannerDecision(
                    thought="视角定位低置信时，不退回通用视觉检查，先基于方向判断和空间上下文做收敛评分。",
                    tool="rank_choices_from_state",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "evidence": state.evidence_bundle,
                        "working_memory": state.working_memory,
                    },
                )
            if "query_spatial_context" not in used_tools and combined_times:
                return PlannerDecision(
                    thought="视角定位低置信时，补一次空间上下文再收敛。",
                    tool="query_spatial_context",
                    args={"time_s": combined_times[0], "object_name": None, "limit": 12},
                )
        if state.task_family == "gaze_gaze_estimation":
            if "infer_gaze_target_with_context" not in used_tools and state.retrieved_frames:
                last_spatial = self._latest_tool_result(state, "query_spatial_context")
                return PlannerDecision(
                    thought="注视目标题低置信时，先补专用 gaze 推断。",
                    tool="infer_gaze_target_with_context",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                        "spatial_context": last_spatial if isinstance(last_spatial, dict) else {},
                    },
                )
            if "rank_choices_from_state" not in used_tools:
                return PlannerDecision(
                    thought="注视目标题低置信时，先基于现有 gaze/spatial 证据收敛评分。",
                    tool="rank_choices_from_state",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "evidence": state.evidence_bundle,
                        "working_memory": state.working_memory,
                    },
                )
        if self._is_temporal_localization_task(state):
            if "infer_temporal_localization_choice" not in used_tools:
                return PlannerDecision(
                    thought="时间定位题低置信时，优先直接比较候选时间段关键帧，而不是重复通用时间检索。",
                    tool="infer_temporal_localization_choice",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "task_family": state.task_family,
                        "frames_per_choice": 2,
                        "tag": f"{state.task_family}_recover_temporal",
                    },
                )
        if self._is_ingredient_order_task(state) and "infer_ingredient_order_choice" not in used_tools:
            return PlannerDecision(
                thought="食材顺序题低置信时，回到结构化加入顺序主路径。",
                tool="infer_ingredient_order_choice",
                args={"question": state.question, "choices": state.choices},
            )
        if self._is_ingredient_retrieval_task(state) and "infer_ingredient_retrieval_choice" not in used_tools:
            return PlannerDecision(
                thought="时间窗食材检索题低置信时，回到结构化区间食材主路径。",
                tool="infer_ingredient_retrieval_choice",
                args={"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_recipe_ingredient_membership_task(state) and "infer_recipe_ingredient_membership_choice" not in used_tools:
            return PlannerDecision(
                thought="菜谱食材归属题低置信时，回到 recipe catalog 归属判断主路径。",
                tool="infer_recipe_ingredient_membership_choice",
                args={"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_exact_ingredient_amount_task(state) and "infer_exact_ingredient_amount_choice" not in used_tools:
            return PlannerDecision(
                thought="精确食材用量题低置信时，回到 recipe catalog 用量主路径。",
                tool="infer_exact_ingredient_amount_choice",
                args={"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_action_mechanism_task(state) and state.retrieved_frames and "infer_action_mechanism" not in used_tools:
            return PlannerDecision(
                thought="how 题低置信时，优先回到专用动作机制判断，而不是继续通用状态/位置检索。",
                tool="infer_action_mechanism",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-4:],
                },
            )
        if self._is_action_intent_task(state) and state.retrieved_frames and "infer_action_intent" not in used_tools:
            action_frames = self._select_action_intent_frames(
                state,
                hints,
                limit=4,
                include_followup=False,
                require_current_scope=True,
            )
            if not action_frames:
                return self._segment_task_sampling_decision(
                    state=state,
                    used_tools=used_tools,
                    combined_times=sorted(
                        [float(value) for value in hints.get("times") or []]
                        + [float(value) for value in hints.get("input_times") or []]
                    ),
                    reuse_thought="why 题低置信时先检索当前动作片段 artifact。",
                    extract_thought="why 题低置信时先抽当前动作时间窗关键帧。",
                )
            context_notes = self._action_intent_context_notes(state, limit=10)
            return PlannerDecision(
                thought="why 题低置信时，优先回到专用动作目的判断，而不是继续通用状态/位置检索。",
                tool="infer_action_intent",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": action_frames,
                    "context_notes": context_notes,
                },
            )
        if self._is_recipe_catalog_task(state) and "infer_recipe_catalog_choice" not in used_tools:
            return PlannerDecision(
                thought="菜谱识别题低置信时，回到 recipe catalog 主路径。",
                tool="infer_recipe_catalog_choice",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "scope": "participant" if state.task_family == "recipe_recipe_recognition" else "video",
                },
            )
        if self._is_recipe_nutrition_task(state) and "infer_recipe_nutrition_choice" not in used_tools:
            return PlannerDecision(
                thought="视频级营养题低置信时，回到 recipe catalog + nutrition 主路径。",
                tool="infer_recipe_nutrition_choice",
                args={"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_object_itinerary_task(state):
            combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
            bbox = hints.get("bbox")
            if bbox and combined_times and "resolve_bbox_reference" not in used_tools:
                return PlannerDecision(
                    thought="轨迹题低置信时，先补对象轨迹解析。",
                    tool="resolve_bbox_reference",
                    args={"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
                )
            if bbox and combined_times and "infer_object_movement_itinerary" not in used_tools:
                return PlannerDecision(
                    thought="轨迹题低置信时，优先补完整路径推断，而不是回到通用时间检索。",
                    tool="infer_object_movement_itinerary",
                    args={
                        "bbox": bbox,
                        "reference_time": combined_times[0],
                        "choices": [str(choice) for choice in state.choices],
                    },
                )
        return self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)

    def _stabilize_decision(self, *, state: AgentState, hints: dict[str, Any], decision: PlannerDecision) -> PlannerDecision:
        used_tools = self._used_tools(state)
        if self._should_preserve_structured_spatial_decision(state=state, decision=decision):
            return decision
        if (
            self._open_query_needs_raw_grounding(
                state=state,
                open_questions=list(getattr(state, "open_questions", []) or []),
                used_tools=used_tools,
            )
            and self._is_raw_grounding_tool(decision.tool)
        ):
            candidate_plan = self._best_state_candidate_plan(state=state, hints=hints, used_tools=used_tools)
            if (
                candidate_plan is not None
                and candidate_plan.decision.tool != decision.tool
                and self._is_raw_grounding_tool(candidate_plan.decision.tool)
                and self._prefers_cheaper_memory_path(
                    candidate_tool=candidate_plan.decision.tool,
                    decision_tool=decision.tool,
                )
            ):
                self._state_add_memory(
                    state,
                    f"planner_override cheaper_raw_grounding={decision.tool} -> {candidate_plan.decision.tool}",
                )
                return candidate_plan.decision
            return decision
        if self._decision_hits_blocked_tool(state=state, decision=decision):
            recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
            if recovered.tool != decision.tool:
                self._state_add_memory(state, f"planner_override blocked_tool={decision.tool} -> {recovered.tool}")
                return recovered
        if self._decision_repeats_stalled_loop(state=state, decision=decision):
            recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
            if recovered.tool != decision.tool:
                self._state_add_memory(state, f"planner_override repeated_loop={decision.tool} -> {recovered.tool}")
                return recovered
            candidate = self._select_state_driven_candidate(state=state, hints=hints, used_tools=used_tools)
            if candidate is not None and candidate.tool != decision.tool:
                self._state_add_memory(state, f"planner_override state_candidate={decision.tool} -> {candidate.tool}")
                return candidate
        candidate_plan = self._best_state_candidate_plan(state=state, hints=hints, used_tools=used_tools)
        if candidate_plan is not None:
            candidate = candidate_plan.decision
            needed_evidence = self._current_evidence_needs(state)
            candidate_addresses_need = self._tool_addresses_needs(
                tool=candidate.tool,
                needed_evidence=needed_evidence,
                verifier_conflicts=self._current_verifier_conflicts(state),
                recommend_next_action=str(self._state_latest_verification(state).get("recommend_next_action") or ""),
            )
            decision_addresses_need = self._tool_addresses_needs(
                tool=decision.tool,
                needed_evidence=needed_evidence,
                verifier_conflicts=self._current_verifier_conflicts(state),
                recommend_next_action=str(self._state_latest_verification(state).get("recommend_next_action") or ""),
            )
            if decision.tool == "finish" and needed_evidence and candidate.tool != decision.tool:
                self._state_add_memory(state, f"planner_override finish_before_missing_evidence={decision.tool} -> {candidate.tool}")
                return candidate
            if candidate_addresses_need and not decision_addresses_need and candidate.tool != decision.tool:
                self._state_add_memory(state, f"planner_override unmet_need={decision.tool} -> {candidate.tool}")
                return candidate
            if (
                candidate.tool != decision.tool
                and candidate_addresses_need
                and self._prefers_cheaper_memory_path(candidate_tool=candidate.tool, decision_tool=decision.tool)
            ):
                self._state_add_memory(state, f"planner_override cheaper_memory_path={decision.tool} -> {candidate.tool}")
                return candidate
        return decision

    def _should_preserve_structured_spatial_decision(self, *, state: AgentState, decision: PlannerDecision) -> bool:
        if not decision.tool:
            return False
        if self._is_object_location_task(state):
            return decision.tool in {"resolve_bbox_reference", "infer_object_drop_location", "finish"}
        if self._is_object_contents_task(state):
            return decision.tool in {"resolve_bbox_reference", "infer_visual_mcq", "finish"}
        if self._is_temporal_localization_task(state):
            return decision.tool in {"infer_temporal_localization_choice", "finish"}
        if self._is_ingredient_order_task(state):
            return decision.tool in {"infer_ingredient_order_choice", "finish"}
        if self._is_ingredient_retrieval_task(state):
            return decision.tool in {"infer_ingredient_retrieval_choice", "finish"}
        if self._is_recipe_ingredient_membership_task(state):
            return decision.tool in {"infer_recipe_ingredient_membership_choice", "finish"}
        if self._is_exact_ingredient_amount_task(state):
            return decision.tool in {"infer_exact_ingredient_amount_choice", "finish"}
        if self._is_action_mechanism_task(state):
            return decision.tool in {"infer_action_mechanism", "finish"}
        if self._is_action_intent_task(state):
            return decision.tool in {"infer_action_intent", "resolve_action_intent_pairwise", "resolve_action_intent_future_use", "finish"}
        if self._is_recipe_catalog_task(state):
            return decision.tool in {"infer_recipe_catalog_choice", "finish"}
        if self._is_recipe_nutrition_task(state):
            return decision.tool in {"infer_recipe_nutrition_choice", "finish"}
        if self._is_recipe_following_activity_task(state):
            return decision.tool in {"query_event", "extract_frames_for_range", "infer_visual_mcq", "rank_choices_from_state", "finish"}
        if self._is_nutrition_change_task(state):
            return decision.tool in {"compute_nutrition_change", "rank_choices_from_state", "finish"}
        if self._is_weight_task(state):
            return decision.tool in {
                "query_ingredient_measurement",
                "query_ocr",
                "retrieve_cached_artifacts",
                "extract_frames_for_range",
                "run_ocr_on_region",
                "run_ocr_on_image",
                "rank_choices_from_state",
                "finish",
            }
        return False

    def _has_unresolved_evidence_gap(self, open_questions: list[str], *, task_family: str = "") -> bool:
        meaningful = [
            item
            for item in open_questions
            if item and (item != "need_disambiguating_evidence" or str(task_family).startswith("open_query"))
        ]
        return bool(meaningful)

    def _decision_hits_blocked_tool(self, *, state: AgentState, decision: PlannerDecision) -> bool:
        if not decision.tool:
            return False
        blocked_tools = {
            str(item.get("tool"))
            for item in (
                [entry for entry in getattr(state, "tool_failures", []) if isinstance(entry, dict)][-5:]
                + [entry for entry in getattr(state, "ineffective_tools", []) if isinstance(entry, dict)][-5:]
            )
            if item.get("tool")
        }
        return decision.tool in blocked_tools

    def _decision_repeats_stalled_loop(self, *, state: AgentState, decision: PlannerDecision) -> bool:
        if not decision.tool:
            return False
        recent_trace = [entry for entry in getattr(state, "tool_trace", []) if isinstance(entry, dict)][-4:]
        if len(recent_trace) < 2:
            return False
        last_tool = recent_trace[-1].get("tool")
        if decision.tool == "rank_choices_from_state" and last_tool == "rank_choices_from_state":
            last_result = recent_trace[-1].get("raw_result")
            if isinstance(last_result, dict) and float(last_result.get("confidence") or 0.0) < 0.8:
                return True
        if decision.tool == last_tool:
            repeated = [entry for entry in recent_trace if entry.get("tool") == decision.tool]
            if len(repeated) >= 2 and self._recent_trace_has_no_progress(state):
                return True
        return False

    def _recent_trace_has_no_progress(self, state: AgentState) -> bool:
        recent_trace = [entry for entry in getattr(state, "tool_trace", []) if isinstance(entry, dict)][-3:]
        if not recent_trace:
            return False
        for entry in recent_trace:
            raw_result = entry.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            if any(
                raw_result.get(key)
                for key in (
                    "nodes",
                    "matches",
                    "totals",
                    "artifact_path",
                    "artifact_paths",
                    "reading",
                    "text",
                    "scores",
                    "best_index",
                    "association_id",
                    "tracks",
                )
            ):
                return False
        return True

    def _recover_from_open_questions(self, *, state: AgentState, hints: dict[str, Any], used_tools: list[str]) -> PlannerDecision:
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        bbox = hints.get("bbox")
        ingredient_name = hints.get("ingredient_name")
        open_questions = list(getattr(state, "open_questions", []) or [])
        latest_verification = self._state_latest_verification(state)
        verifier_conflicts = {
            str(item)
            for item in latest_verification.get("conflicts", [])
            if isinstance(item, str) and item
        }
        recent_failures = [item for item in getattr(state, "tool_failures", []) if isinstance(item, dict)]
        failed_tools = {str(item.get("tool")) for item in recent_failures[-5:] if item.get("tool")}
        recent_ineffective = [item for item in getattr(state, "ineffective_tools", []) if isinstance(item, dict)]
        ineffective_tools = {str(item.get("tool")) for item in recent_ineffective[-5:] if item.get("tool")}
        if self._is_recipe_following_activity_task(state):
            recipe_step_hint = hints.get("recipe_step_hint")
            if "query_event" not in used_tools and "query_event" not in failed_tools and "query_event" not in ineffective_tools:
                return PlannerDecision(
                    thought="高层 recipe-following 题被阻断时，先回到 recipe_step / activity 结构化检索主路径。",
                    tool="query_event",
                    args={
                        "event_types": ["recipe_step", "activity"],
                        "keyword": str(recipe_step_hint or state.question),
                        "start_time": min(combined_times) if combined_times else None,
                        "end_time": max(combined_times) if combined_times else None,
                        "limit": 10,
                    },
                )
            if state.retrieved_frames and "infer_visual_mcq" not in failed_tools and "infer_visual_mcq" not in ineffective_tools:
                return PlannerDecision(
                    thought="高层 recipe-following 题在已有 step 窗口关键帧后，优先直接做视觉多选判断。",
                    tool="infer_visual_mcq",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-4:],
                    },
                )
        if self._is_nutrition_change_task(state) and combined_times:
            if (
                "compute_nutrition_change" not in used_tools
                and "compute_nutrition_change" not in failed_tools
                and "compute_nutrition_change" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="营养变化题被阻断时，先回到结构化营养增量计算主路径。",
                    tool="compute_nutrition_change",
                    args={"start_time": min(combined_times), "end_time": max(combined_times)},
                )
            if "rank_choices_from_state" not in failed_tools and "rank_choices_from_state" not in ineffective_tools:
                return PlannerDecision(
                    thought="营养变化题在已有结构化营养增量后，直接对选项评分，而不是继续做通用状态/时间检索。",
                    tool="rank_choices_from_state",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "evidence": state.evidence_bundle,
                        "working_memory": state.working_memory,
                    },
                )
        if self._is_object_location_task(state) and bbox and combined_times:
            if (
                "resolve_bbox_reference" not in used_tools
                and "resolve_bbox_reference" not in failed_tools
                and "resolve_bbox_reference" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="bbox 驱动的放置位置题被阻断时，优先回到对象解析主路径，而不是退化成通用位置检索。",
                    tool="resolve_bbox_reference",
                    args={"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
                )
            if (
                "infer_object_drop_location" not in used_tools
                and "infer_object_drop_location" not in failed_tools
                and "infer_object_drop_location" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="bbox 驱动的放置位置题在已有对象解析后，优先直接推断最终落点。",
                    tool="infer_object_drop_location",
                    args={
                        "bbox": bbox,
                        "reference_time": combined_times[0],
                        "choices": [str(choice) for choice in state.choices],
                        "question": state.question,
                    },
                )
        if self._is_object_contents_task(state) and bbox and combined_times:
            if (
                "resolve_bbox_reference" not in used_tools
                and "resolve_bbox_reference" not in failed_tools
                and "resolve_bbox_reference" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="容器内容检索题被阻断时，先固定 bbox 对应的容器/对象引用，避免直接在普通位置记忆上兜圈。",
                    tool="resolve_bbox_reference",
                    args={"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
                )
            if state.retrieved_frames and "infer_visual_mcq" not in failed_tools and "infer_visual_mcq" not in ineffective_tools:
                return PlannerDecision(
                    thought="容器内容检索题在已有关键帧后，优先直接做视觉多选判断。",
                    tool="infer_visual_mcq",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                    },
                )
        if "conflicting_ocr_readings" in verifier_conflicts:
            if state.retrieved_frames and "run_ocr_on_image" not in failed_tools and "run_ocr_on_image" not in ineffective_tools:
                latest_frame = self._latest_visual_frame(state)
                if latest_frame is None:
                    return None
                return PlannerDecision(
                    thought="当前存在 OCR 读数冲突，优先重新回看当前图像做 OCR 消解冲突。",
                    tool="run_ocr_on_image",
                    args={"image_path": latest_frame},
                )
            if "query_ocr" not in failed_tools and "query_ocr" not in ineffective_tools:
                return PlannerDecision(
                    thought="当前存在 OCR 读数冲突，先重新检索已有 OCR 记忆做一致性检查。",
                    tool="query_ocr",
                    args={
                        "keyword": str(hints.get("ocr_keyword") or "reading"),
                        "start_time": min(combined_times) if combined_times else None,
                        "end_time": max(combined_times) if combined_times else None,
                        "limit": 12,
                    },
                )
        if "conflicting_locations" in verifier_conflicts and self._location_conflict_is_actionable(state=state, hints=hints):
            if "query_location" not in failed_tools and "query_location" not in ineffective_tools:
                return PlannerDecision(
                    thought="当前存在位置冲突，优先重新检索位置记忆做消解。",
                    tool="query_location",
                    args={
                        "location_keyword": str(hints.get("location_keyword") or "location"),
                        "start_time": min(combined_times) if combined_times else None,
                        "end_time": max(combined_times) if combined_times else None,
                        "limit": 12,
                    },
                )
        if "conflicting_state_observations" in verifier_conflicts and self._state_conflict_is_actionable(state=state):
            if "query_state" not in failed_tools and "query_state" not in ineffective_tools:
                return PlannerDecision(
                    thought="当前存在状态冲突，优先重新检索状态变化记忆做消解。",
                    tool="query_state",
                    args={
                        "state_keyword": str(hints.get("state_keyword") or "state"),
                        "start_time": min(combined_times) if combined_times else None,
                        "end_time": max(combined_times) if combined_times else None,
                        "limit": 12,
                    },
                )
        if self._open_query_needs_raw_grounding(state=state, open_questions=open_questions, used_tools=used_tools):
            raw_reuse_or_resample = self._build_raw_reuse_or_resample_decision(
                state=state,
                used_tools=used_tools,
                failed_tools=failed_tools,
                ineffective_tools=ineffective_tools,
                combined_times=combined_times,
                tag_hint=state.task_family,
                sample_tag=f"{state.task_family}_open_recover_frames",
                sample_count=5,
                retrieve_limit=8,
                retrieve_thought="开放问答被阻断后，先显式检索当前视频已经产出的 artifact，优先复用已有帧、局部图和画框图。",
                revisit_thought="开放问答被阻断后，先复用之前已经访问过的关键时刻重新取证，而不是立刻重新做整段稀疏抽帧。",
                resample_thought="开放问答被 verifier 阻断后，优先回看原始视频补关键帧，而不是继续只依赖图谱摘要。",
            )
            if raw_reuse_or_resample is not None:
                return raw_reuse_or_resample
            if (
                self._can_use_visual_inspection(state)
                and state.retrieved_frames
                and "inspect_visual_evidence" not in used_tools
                and "inspect_visual_evidence" not in failed_tools
                and "inspect_visual_evidence" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="开放问答被 verifier 阻断后，优先补一次保守视觉检查，避免只在已有摘要上兜圈。",
                    tool="inspect_visual_evidence",
                    args={
                        "prompt": (
                            "这是开放式视频问答的补证阶段。"
                            "请保守提取时间连续的关键动作、对象、位置、状态变化与读数。"
                            '输出 JSON，字段固定为 {"target_object":"","target_location":"","ongoing_action":"","state_change_hint":"","reading":"","answer_hint":"","confidence":0.0}。'
                        ),
                        "image_paths": state.retrieved_frames[-8:],
                    },
                )
            if (
                state.task_family == "open_query_temporal_summary"
                and combined_times
                and "detect_audio_peaks" not in used_tools
                and "detect_audio_peaks" not in failed_tools
                and "detect_audio_peaks" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="开放式时间总结仍缺区分性证据，补音频峰值帮助定位关键事件。",
                    tool="detect_audio_peaks",
                    args={
                        "start_time": max(0.0, min(combined_times) - 2.0),
                        "end_time": max(combined_times) + 2.0,
                        "window_s": 0.5,
                        "top_k": 4,
                    },
                )
        if "need_ocr_reading" in open_questions:
            if (
                self._is_weight_task(state)
                and ingredient_name
                and combined_times
                and "query_ingredient_measurement" not in used_tools
                and "query_ingredient_measurement" not in failed_tools
                and "query_ingredient_measurement" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="当前评分置信度不足，称重题先补结构化称量记录，再决定是否继续看图。",
                    tool="query_ingredient_measurement",
                    args={
                        "ingredient_name": str(ingredient_name),
                        "start_time": min(combined_times),
                        "end_time": max(combined_times),
                        "limit": 10,
                    },
                )
            if (
                bbox
                and state.retrieved_frames
                and "run_ocr_on_region" not in used_tools
                and "run_ocr_on_region" not in failed_tools
                and "run_ocr_on_region" not in ineffective_tools
            ):
                return PlannerDecision(
                    thought="当前评分置信度不足，且仍缺 OCR 证据，转为补局部 OCR。",
                    tool="run_ocr_on_region",
                    args={
                        "image_path": self._latest_visual_frame(state) or "",
                        "bbox": bbox,
                        "expand_ratio": 0.35,
                        "tag": f"{state.task_family}_recover_ocr",
                    },
                )
            if state.retrieved_frames and "run_ocr_on_image" not in failed_tools and "run_ocr_on_image" not in ineffective_tools:
                latest_frame = self._latest_visual_frame(state)
                if latest_frame is None:
                    return None
                return PlannerDecision(
                    thought="当前评分置信度不足，且仍缺 OCR 证据，转为补整图 OCR。",
                    tool="run_ocr_on_image",
                    args={"image_path": latest_frame},
                )
        if "need_region_grounding" in open_questions and bbox and state.retrieved_frames:
            region_reuse_or_recrop = self._build_region_reuse_or_recrop_decision(
                state=state,
                used_tools=used_tools,
                failed_tools=failed_tools,
                ineffective_tools=ineffective_tools,
                bbox=bbox,
                image_path=self._latest_visual_frame(state) or state.retrieved_frames[-1],
                tag_hint=state.task_family,
                overlay_tag=f"{state.task_family}_recover_bbox",
                region_tag=f"{state.task_family}_recover_region",
            )
            if region_reuse_or_recrop is not None:
                return region_reuse_or_recrop
        if (
            "need_location_evidence" in open_questions
            and not self._is_viewpoint_task(state)
            and (
                not self._is_weight_task(state)
                or self._question_explicitly_mentions_location(state=state, location_keyword=hints.get("location_keyword"))
            )
            and "query_location" not in used_tools
            and "query_location" not in failed_tools
            and "query_location" not in ineffective_tools
        ):
            return PlannerDecision(
                thought="当前评分置信度不足，且仍缺位置证据，转为检索空间/位置记忆。",
                tool="query_location",
                args={
                    "location_keyword": str(hints.get("location_keyword") or "location"),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
        if (
            "need_state_evidence" in open_questions
            and "query_state" not in used_tools
            and "query_state" not in failed_tools
            and "query_state" not in ineffective_tools
        ):
            return PlannerDecision(
                thought="当前评分置信度不足，且仍缺状态证据，转为检索状态变化记忆。",
                tool="query_state",
                args={
                    "state_keyword": str(hints.get("state_keyword") or "state"),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
        if self._open_query_needs_raw_grounding(state=state, open_questions=open_questions, used_tools=used_tools):
            raw_reuse_or_resample = self._build_raw_reuse_or_resample_decision(
                state=state,
                used_tools=used_tools,
                failed_tools=failed_tools,
                ineffective_tools=ineffective_tools,
                combined_times=combined_times,
                tag_hint=state.task_family,
                sample_tag=f"{state.task_family}_reuse_time_anchor",
                sample_count=4,
                retrieve_limit=8,
                retrieve_thought="开放问答在低置信恢复阶段，先显式复用当前视频已有 artifact，而不是直接重做整段抽帧。",
                revisit_thought="开放问答在低置信恢复阶段，先回到已访问的关键时刻取单帧，而不是立刻做整段稀疏抽帧。",
                resample_thought="开放问答在低置信恢复阶段，若没有可复用 artifact，再重新稀疏抽帧补证据。",
            )
            if raw_reuse_or_resample is not None:
                return raw_reuse_or_resample
        if ("need_time_localization" in open_questions or "need_initial_observation" in open_questions) and combined_times:
            raw_reuse_or_resample = self._build_raw_reuse_or_resample_decision(
                state=state,
                used_tools=used_tools,
                failed_tools=failed_tools,
                ineffective_tools=ineffective_tools,
                combined_times=combined_times,
                tag_hint=state.task_family,
                sample_tag=f"{state.task_family}_recover_frames",
                sample_count=4,
                retrieve_limit=6,
                retrieve_thought="当前评分置信度不足且时间证据仍弱，先复用当前视频已有时间段 artifact。",
                revisit_thought="当前评分置信度不足且时间证据仍弱，先回到已访问关键时刻补单帧，而不是直接整段重抽。",
                resample_thought="当前评分置信度不足，且时间证据仍弱，转为重新稀疏抽帧补证据。",
            )
            if raw_reuse_or_resample is not None:
                return raw_reuse_or_resample
        if (
            self._can_use_visual_inspection(state)
            and not self._is_weight_task(state)
            and not self._is_action_intent_task(state)
            and state.retrieved_frames
            and "inspect_visual_evidence" not in used_tools
            and "inspect_visual_evidence" not in failed_tools
            and "inspect_visual_evidence" not in ineffective_tools
        ):
            return PlannerDecision(
                thought="当前评分置信度不足，转为补一次视觉检查而不是直接结束。",
                tool="inspect_visual_evidence",
                args={
                    "prompt": (
                        "当前证据不足以高置信回答。"
                        "请保守地补充这组图片中的关键对象、动作、位置、状态变化或读数。"
                        '输出 JSON，字段固定为 {"target_object":"","target_location":"","ongoing_action":"","state_change_hint":"","reading":"","answer_hint":"","confidence":0.0}。'
                    ),
                    "image_paths": state.retrieved_frames[-6:],
                },
            )
        if (
            state.retrieved_frames
            and self._is_weight_task(state)
        ):
            return PlannerDecision(
                thought="称重题避免高成本通用视觉检查，回到时间检索或评分收尾。",
                tool="query_time",
                args={
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
        return PlannerDecision(
            thought="当前评分置信度不足，继续保留评分结果，但先补一个通用时间检索。",
            tool="query_time",
            args={
                "start_time": min(combined_times) if combined_times else None,
                "end_time": max(combined_times) if combined_times else None,
                "limit": 12,
            },
        )

    def _select_state_driven_candidate(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> PlannerDecision | None:
        ranked = self._rank_state_candidate_plans(state=state, hints=hints, used_tools=used_tools)
        best = ranked[0] if ranked else None
        if best is None:
            return None
        self._record_candidate_plan_comparison(state=state, ranked=ranked)
        self._state_add_memory(
            state,
            f"candidate_plan_selected tool={best.decision.tool} score={best.score} cost={best.cost} gain={best.gain} risk={best.risk}",
        )
        self._state_add_hypothesis(state, f"candidate_plan_rationale={best.rationale}")
        return best.decision

    def _best_state_candidate_plan(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> CandidatePlan | None:
        ranked = self._rank_state_candidate_plans(state=state, hints=hints, used_tools=used_tools)
        return ranked[0] if ranked else None

    def _rank_state_candidate_plans(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> list[CandidatePlan]:
        candidates = self._build_state_driven_candidates(state=state, hints=hints, used_tools=used_tools)
        if not candidates:
            return []
        return sorted(candidates, key=lambda item: (item.score, item.cost, item.risk, item.decision.tool))

    def _record_candidate_plan_comparison(self, *, state: AgentState, ranked: list[CandidatePlan]) -> None:
        if not ranked:
            return
        top = ranked[:3]
        for index, item in enumerate(top, start=1):
            self._state_add_memory(
                state,
                (
                    f"candidate_plan_rank rank={index} tool={item.decision.tool} "
                    f"score={item.score} cost={item.cost} gain={item.gain} risk={item.risk}"
                ),
            )
        if len(top) >= 2:
            winner = top[0]
            runner_up = top[1]
            self._state_add_hypothesis(
                state,
                (
                    "candidate_plan_comparison="
                    f"winner:{winner.decision.tool}[score={winner.score},gain={winner.gain},risk={winner.risk}]"
                    f" > runner_up:{runner_up.decision.tool}[score={runner_up.score},gain={runner_up.gain},risk={runner_up.risk}]"
                ),
            )
            self._state_add_hypothesis(
                state,
                f"candidate_plan_runner_up_rationale={runner_up.rationale}",
            )

    def _add_open_query_family_candidates(
        self,
        *,
        state: AgentState,
        combined_times: list[float],
        used_tools: list[str],
        ocr_keyword: Any,
        location_keyword: Any,
        state_keyword: Any,
        add_candidate: Callable[[int, int, int, str, str, str, dict[str, Any]], None],
    ) -> None:
        if state.task_family == "open_query_temporal_summary":
            if combined_times:
                add_candidate(
                    1,
                    5,
                    1,
                    "开放式时间总结题优先取时间窗口内已有图谱记忆。",
                    "先检索题目时间窗口内的已有事件与观察记忆。",
                    "query_time",
                    {"start_time": min(combined_times), "end_time": max(combined_times), "limit": 20},
                )
            if getattr(state, "retrieved_node_ids", []):
                add_candidate(
                    1,
                    5,
                    1,
                    "已有锚点后，沿 before/after/co_occurs/same_step 扩展上下文更适合做开放式总结。",
                    "沿图关系扩展当前时间段的上下文证据。",
                    "expand_graph_context",
                    {
                        "node_ids": list(getattr(state, "retrieved_node_ids", [])[-8:]),
                        "edge_types": ["same_step", "before", "after", "co_occurs"],
                        "limit": 20,
                    },
                )
            if combined_times and "detect_audio_peaks" not in used_tools:
                add_candidate(
                    3,
                    4,
                    2,
                    "若图谱不足，音频峰值可帮助定位关键事件瞬间。",
                    "图谱证据不足时，先检测音频峰值补关键时刻。",
                    "detect_audio_peaks",
                    {
                        "start_time": max(0.0, min(combined_times) - 2.0),
                        "end_time": max(combined_times) + 2.0,
                        "window_s": 0.5,
                        "top_k": 4,
                    },
                )
            if state.retrieved_frames:
                add_candidate(
                    4,
                    5,
                    2,
                    "开放式总结最终仍需要少量视觉补充来避免只复述图谱。",
                    "对当前时间段的关键帧做保守视觉总结。",
                    "inspect_visual_evidence",
                    {
                        "prompt": (
                            "请保守总结这组图片在时间上连续发生的动作、对象、位置与状态变化。"
                            '输出 JSON，字段固定为 {"ongoing_action":"","possible_step":"","target_object":"","state_change_hint":"","answer_hint":"","confidence":0.0}。'
                        ),
                        "image_paths": state.retrieved_frames[-8:],
                    },
                )
            return
        if state.task_family == "open_query_ocr" and ocr_keyword:
            add_candidate(
                1,
                5,
                1,
                "开放式读数题优先检索已有 OCR 记忆。",
                "先检索当前问题相关的 OCR 读数记忆。",
                "query_ocr",
                {
                    "keyword": str(ocr_keyword),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
            return
        if state.task_family == "open_query_location":
            add_candidate(
                1,
                5,
                1,
                "开放式位置题优先检索已有位置记忆。",
                "先检索当前问题相关的位置/空间记忆。",
                "query_location",
                {
                    "location_keyword": str(location_keyword or "location"),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
            return
        if state.task_family == "open_query_state":
            add_candidate(
                1,
                5,
                1,
                "开放式状态题优先检索状态变化记忆。",
                "先检索当前问题相关的状态变化记忆。",
                "query_state",
                {
                    "state_keyword": str(state_keyword or "state"),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )

    def _add_specialized_followup_candidates(
        self,
        *,
        state: AgentState,
        combined_times: list[float],
        bbox: Any,
        add_candidate: Callable[[int, int, int, str, str, str, dict[str, Any]], None],
    ) -> None:
        if state.task_family == "nutrition_image_nutrition_estimation" and state.retrieved_frames:
            add_candidate(
                2,
                5,
                1,
                "先识别参考图中的食材，再做营养比较更稳。",
                "已拿到参考图，先识别图中的食材。",
                "identify_image_ingredients",
                {"image_paths": state.retrieved_frames[-10:]},
            )
            nutrient = "carbs" if "carb" in state.question.lower() else "calories"
            add_candidate(
                1,
                4,
                1,
                "结构化营养比较成本低，适合作为识别后的下一跳。",
                "基于候选选项做结构化营养比较。",
                "compare_choice_nutrition",
                {"choices": [str(choice) for choice in state.choices], "nutrient": nutrient},
            )
            return
        if state.task_family == "object_motion_object_movement_counting" and bbox and combined_times:
            add_candidate(
                4,
                4,
                2,
                "轨迹计数应放在轨迹解析之后，避免还未建立对象关联就直接计数。",
                "根据对象轨迹估计位置变化次数。",
                "estimate_object_movement_count",
                {"bbox": bbox, "reference_time": combined_times[0], "choices": [str(choice) for choice in state.choices]},
            )
            return
        if state.task_family == "object_motion_stationary_object_localization" and bbox and combined_times:
            add_candidate(
                4,
                4,
                2,
                "静止起点估计同样应建立在对象轨迹已解析的前提上。",
                "根据对象轨迹判断静止起始时间。",
                "estimate_stationary_start",
                {
                    "bbox": bbox,
                    "reference_time": combined_times[0],
                    "choices": [str(choice) for choice in state.choices],
                    "threshold_s": 150.0,
                },
            )
            return
        if state.task_family == "3d_perception_object_location" and bbox and combined_times:
            add_candidate(
                4,
                5,
                2,
                "放置位置题可直接复用 reference object track 的最终 fixture，属于高价值低幻觉结构化路径。",
                "根据目标对象后续轨迹的最终落点推断位置选项。",
                "infer_object_drop_location",
                {
                    "bbox": bbox,
                    "reference_time": combined_times[0],
                    "choices": [str(choice) for choice in state.choices],
                    "question": state.question,
                },
            )
            return
        if state.task_family == "gaze_gaze_estimation" and state.retrieved_frames:
            add_candidate(
                2,
                5,
                1,
                "注视题可直接结合图像与空间上下文做专用判断。",
                "结合当前图像和空间上下文推断注视目标。",
                "infer_gaze_target_with_context",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-3:],
                    "spatial_context": self._latest_tool_result(state, "query_spatial_context"),
                },
            )
            return
        if state.task_family == "3d_perception_fixture_location" and state.retrieved_frames:
            add_candidate(
                2,
                5,
                1,
                "fixture 方位题需要从视角图像映射到具名设备方向。",
                "结合当前视角图像做具名 fixture 方向判断。",
                "infer_named_fixture_direction",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-3:],
                },
            )

    def _add_disambiguating_candidates(
        self,
        *,
        state: AgentState,
        combined_times: list[float],
        bbox: Any,
        ingredient_name: Any,
        object_hint: Any,
        add_candidate: Callable[[int, int, int, str, str, str, dict[str, Any]], None],
    ) -> None:
        if getattr(state, "retrieved_node_ids", []):
            add_candidate(
                1,
                4,
                1,
                "已有锚点节点时，先扩展图关系可低成本补区分性上下文。",
                "当前缺少区分性证据，先扩展已知节点的图关系上下文。",
                "expand_graph_context",
                {
                    "node_ids": list(getattr(state, "retrieved_node_ids", [])[-8:]),
                    "edge_types": ["same_step", "same_object", "co_occurs", "before", "after"],
                    "limit": 20,
                },
            )
        if object_hint and bbox is None:
            add_candidate(
                2,
                4,
                1,
                "对象区域检索有助于快速补区分性记忆。",
                "当前缺少区分性证据，先检索对象/区域记忆。",
                "query_region",
                {
                    "object_hint": str(object_hint),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
        if ingredient_name and state.task_family == "ingredient_ingredient_weight" and combined_times:
            add_candidate(
                1,
                5,
                1,
                "结构化称量记录对称重题收益高且成本低。",
                "当前缺少区分性证据，先查结构化称量记录。",
                "query_ingredient_measurement",
                {
                    "ingredient_name": str(ingredient_name),
                    "start_time": min(combined_times),
                    "end_time": max(combined_times),
                    "limit": 10,
                },
            )
        if state.task_family.startswith("recipe_"):
            add_candidate(
                1,
                4,
                1,
                "recipe_step 检索对步骤题通常是最便宜的第一跳。",
                "当前缺少区分性证据，先检索 recipe_step 事件。",
                "query_event",
                {
                    "event_types": ["recipe_step"],
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 20,
                },
            )
        if state.task_family == "nutrition_nutrition_change" and combined_times:
            add_candidate(
                1,
                5,
                1,
                "营养变化题直接读取结构化 ingredient add 事件，成本低且解释性强。",
                "当前缺少区分性证据，先直接计算时间窗口内营养变化。",
                "compute_nutrition_change",
                {"start_time": min(combined_times), "end_time": max(combined_times)},
            )
        if state.task_family == "nutrition_image_nutrition_estimation":
            add_candidate(
                1,
                4,
                1,
                "多图营养题优先提取题目给出的参考图，避免先猜选项。",
                "当前缺少区分性证据，先提取 inputs_json 中的参考图。",
                "extract_input_reference_frames",
                {"tag": f"{state.task_family}_inputs"},
            )
        if state.task_family == "object_motion_object_movement_counting" and bbox and combined_times:
            add_candidate(
                2,
                5,
                1,
                "物体运动计数优先接入 object track，后续可直接估计次数。",
                "当前缺少区分性证据，先解析 bbox 对应的对象轨迹。",
                "resolve_bbox_reference",
                {"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
            )
        if state.task_family == "object_motion_stationary_object_localization" and bbox and combined_times:
            add_candidate(
                2,
                5,
                1,
                "长期静止定位题先建对象轨迹，再判断静止起点。",
                "当前缺少区分性证据，先解析 bbox 对应的对象轨迹。",
                "resolve_bbox_reference",
                {"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
            )
        if state.task_family in {"3d_perception_fixture_location", "gaze_gaze_estimation"} and combined_times:
            add_candidate(
                2,
                5,
                1,
                "空间/视角题优先补当前时刻的空间上下文。",
                "当前缺少区分性证据，先查询该时刻的空间上下文。",
                "query_spatial_context",
                {"time_s": combined_times[0], "object_name": None, "limit": 12},
            )

    def _build_state_driven_candidates(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> list[CandidatePlan]:
        candidates: list[CandidatePlan] = []
        open_questions = list(getattr(state, "open_questions", []) or [])
        recent_failures = [item for item in getattr(state, "tool_failures", []) if isinstance(item, dict)]
        recent_ineffective = [item for item in getattr(state, "ineffective_tools", []) if isinstance(item, dict)]
        blocked_tools = {
            str(item.get("tool"))
            for item in (recent_failures[-5:] + recent_ineffective[-5:])
            if item.get("tool")
        }
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        bbox = hints.get("bbox")
        ingredient_name = hints.get("ingredient_name")
        state_keyword = hints.get("state_keyword")
        location_keyword = hints.get("location_keyword")
        ocr_keyword = hints.get("ocr_keyword")
        object_hint = hints.get("object_hint")
        explicit_location_need = self._question_explicitly_mentions_location(state=state, location_keyword=location_keyword)
        latest_verification = self._state_latest_verification(state)
        verifier_missing = {
            str(item)
            for item in latest_verification.get("missing_evidence_types", [])
            if isinstance(item, str) and item
        }
        verifier_conflicts = {
            str(item)
            for item in latest_verification.get("conflicts", [])
            if isinstance(item, str) and item
        }
        recommend_next_action = str(latest_verification.get("recommend_next_action") or "")

        def add_candidate(cost: int, gain: int, risk: int, rationale: str, thought: str, tool: str, args: dict[str, Any]) -> None:
            if tool in blocked_tools:
                return
            if tool in used_tools and tool not in {"query_time", "sample_sparse_frames", "extract_frames_for_range"}:
                return
            adjusted_cost = cost
            adjusted_gain = gain
            adjusted_risk = risk
            adjusted_rationale = rationale
            if self._is_weight_task(state):
                if tool == "inspect_visual_evidence":
                    return
                if tool == "query_location" and not explicit_location_need and "need_location_evidence" not in verifier_missing:
                    return
                if tool in {"query_ingredient_measurement", "query_ocr", "run_ocr_on_region", "run_ocr_on_image"}:
                    adjusted_cost = max(0, adjusted_cost - 1)
                    adjusted_gain += 1
                if "need_ocr_reading" in open_questions and getattr(state, "retrieved_node_ids", []):
                    if tool == "expand_graph_context":
                        adjusted_cost = max(0, adjusted_cost - 1)
                        adjusted_gain += 4
                        adjusted_risk = max(0, adjusted_risk - 1)
                    if tool == "query_ingredient_measurement":
                        adjusted_risk += 3
            if self._is_action_intent_task(state) and tool == "inspect_visual_evidence":
                return
            if self._tool_matches_verifier_need(
                tool=tool,
                verifier_missing=verifier_missing,
                verifier_conflicts=verifier_conflicts,
                recommend_next_action=recommend_next_action,
            ):
                adjusted_cost = max(0, adjusted_cost - 1)
                adjusted_gain += 2
                adjusted_risk = max(0, adjusted_risk - 1)
                adjusted_rationale = f"{rationale} verifier 明确指出该路径应优先修补当前证据缺口。"
            candidates.append(
                CandidatePlan(
                    decision=PlannerDecision(thought=thought, tool=tool, args=args),
                    cost=adjusted_cost,
                    gain=adjusted_gain,
                    risk=adjusted_risk,
                    rationale=adjusted_rationale,
                )
            )

        has_reusable_artifacts = self._task_has_reusable_artifacts(state)
        artifact_time = self._best_reusable_open_query_time(state, combined_times)
        if (
            has_reusable_artifacts
            and "retrieve_cached_artifacts" not in blocked_tools
            and "retrieve_cached_artifacts" not in used_tools
        ):
            generic_tag_hint = str(state.task_family or "artifact")
            if self._is_viewpoint_task(state):
                generic_tag_hint = f"{state.task_family}_view"
            elif self._is_action_mechanism_task(state) or self._is_action_intent_task(state):
                generic_tag_hint = f"{state.task_family}_segment"
            elif self._is_object_contents_task(state):
                generic_tag_hint = f"{state.task_family}_contents"
            elif self._is_weight_task(state):
                generic_tag_hint = f"{state.task_family}_range"
            elif str(state.task_family).startswith("open_query"):
                generic_tag_hint = str(state.task_family)
            if verifier_missing & {
                "need_initial_observation",
                "need_time_localization",
                "need_alternative_evidence_path",
                "need_disambiguating_evidence",
                "need_region_grounding",
                "need_state_evidence",
                "need_location_evidence",
            }:
                add_candidate(
                    0,
                    6,
                    0,
                    "已有缓存原始证据时，优先复用历史帧/裁剪图，比重新抽帧或再次视觉调用更便宜，也更符合真实 agent 的记忆复用行为。",
                    "当前仍缺少关键证据，先回收已保存的原始图像证据再决定是否继续抽帧。",
                    "retrieve_cached_artifacts",
                    {
                        "tag_hint": generic_tag_hint,
                        "time_s": artifact_time,
                        "max_results": 6,
                    },
                )

        if self._is_object_location_task(state) and bbox and combined_times:
            add_candidate(
                0,
                8,
                0,
                "放置位置题的主路径是先把 bbox 解析成对象 association / tracks，这比通用位置检索更直接也更可解释。",
                "当前题型是 bbox 驱动的放置位置判断，优先解析目标对象轨迹。",
                "resolve_bbox_reference",
                {"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
            )
            add_candidate(
                1,
                9,
                0,
                "放置位置题可直接利用后续 tracks 的最终 fixture 做选项映射，属于高价值结构化证据。",
                "基于目标对象后续轨迹的最终 fixture，直接推断落点选项。",
                "infer_object_drop_location",
                {
                    "bbox": bbox,
                    "reference_time": combined_times[0],
                    "choices": [str(choice) for choice in state.choices],
                    "question": state.question,
                },
            )
        if self._is_object_contents_task(state) and bbox and combined_times:
            add_candidate(
                0,
                7,
                0,
                "容器内容检索题首先需要把 bbox 锚定到容器或引用对象，否则后续所有检索都缺少统一语义锚点。",
                "当前题型是容器内容检索，优先解析 bbox 对应的容器/对象引用。",
                "resolve_bbox_reference",
                {"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
            )
            if state.retrieved_frames:
                add_candidate(
                    1,
                    8,
                    1,
                    "容器内容题在已有关键帧后，直接做视觉多选比通用检索更贴题。",
                    "根据容器相关关键帧直接判断内容/取放对象选项。",
                    "infer_visual_mcq",
                    {
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": state.retrieved_frames[-3:],
                    },
                )
        if self._is_temporal_localization_task(state):
            add_candidate(
                1,
                8,
                1,
                "时间定位题最值钱的是直接比较各候选时间段关键帧，而不是在全局时间记忆上空转。",
                "直接比较每个候选时间段的关键帧，判断哪一段最符合题目动作/步骤。",
                "infer_temporal_localization_choice",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "task_family": state.task_family,
                    "frames_per_choice": 2,
                    "tag": f"{state.task_family}_state_temporal",
                },
            )
        if self._is_ingredient_order_task(state):
            add_candidate(
                0,
                9,
                0,
                "食材顺序题最直接的证据就是结构化 ingredient add 事件顺序，不应退回通用时间或视觉检索。",
                "直接根据结构化 ingredient add 事件顺序判断候选顺序。",
                "infer_ingredient_order_choice",
                {"question": state.question, "choices": state.choices},
            )
        if self._is_ingredient_retrieval_task(state):
            add_candidate(
                0,
                9,
                0,
                "时间窗食材检索题最直接的证据就是该时间区间的 ingredient add 事件。",
                "直接根据区间内的 ingredient add 事件判断候选食材。",
                "infer_ingredient_retrieval_choice",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_recipe_ingredient_membership_task(state):
            add_candidate(
                0,
                9,
                0,
                "菜谱食材归属题最直接的证据是 recipe catalog 中的 ingredients 集合。",
                "直接根据 recipe catalog 判断哪个候选不属于目标菜谱。",
                "infer_recipe_ingredient_membership_choice",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_exact_ingredient_amount_task(state):
            add_candidate(
                0,
                9,
                0,
                "精确食材用量题最直接的证据是 recipe catalog 中的 ingredient_amounts。",
                "直接根据 recipe catalog 中的 ingredient_amounts 判断精确用量。",
                "infer_exact_ingredient_amount_choice",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_recipe_catalog_task(state):
            add_candidate(
                0,
                9,
                0,
                "菜谱识别题最直接的证据是 inputs 对应视频集合的 recipe catalog。",
                "直接根据 recipe catalog 判断候选菜谱。",
                "infer_recipe_catalog_choice",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "scope": "participant" if state.task_family == "recipe_recipe_recognition" else "video",
                },
            )
        if self._is_recipe_nutrition_task(state):
            add_candidate(
                0,
                9,
                0,
                "视频级营养题最直接的证据是 recipe catalog 中的食材集合与结构化营养字段。",
                "直接比较 recipe 食材候选的结构化营养值。",
                "infer_recipe_nutrition_choice",
                {"question": state.question, "choices": [str(choice) for choice in state.choices]},
            )
        if self._is_action_mechanism_task(state) and state.retrieved_frames:
            add_candidate(
                0,
                9,
                0,
                "how 题最值钱的是直接比较关键帧中的手部触发方式，而不是继续检索抽象状态节点。",
                "直接根据关键帧判断动作是按按钮、拉门还是推压完成。",
                "infer_action_mechanism",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-4:],
                },
            )
        action_intent_candidate_frames: list[str] = []
        if self._is_action_intent_task(state) and state.retrieved_frames:
            action_intent_candidate_frames = self._select_action_intent_frames(
                state,
                hints,
                limit=4,
                include_followup=False,
                require_current_scope=True,
            )
        if self._is_action_intent_task(state) and action_intent_candidate_frames:
            context_notes = self._action_intent_context_notes(state, limit=10)
            add_candidate(
                0,
                9,
                0,
                "why 题最值钱的是直接结合关键帧和上下文判断动作目的，而不是继续查通用状态节点。",
                "直接根据关键帧和上下文判断动作的最直接目的。",
                "infer_action_intent",
                {
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": action_intent_candidate_frames,
                    "context_notes": context_notes,
                },
            )
        if self._is_object_itinerary_task(state) and bbox and combined_times:
            add_candidate(
                1,
                8,
                1,
                "轨迹题最核心的是先拿到目标对象的完整轨迹，再映射到路径选项。",
                "先解析参考 bbox 对应的对象完整轨迹。",
                "resolve_bbox_reference",
                {"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
            )
            add_candidate(
                2,
                9,
                1,
                "轨迹题可直接利用完整 fixture path 做选项比对，结构化价值高。",
                "根据目标对象完整 fixture 路径直接推断移动轨迹选项。",
                "infer_object_movement_itinerary",
                {
                    "bbox": bbox,
                    "reference_time": combined_times[0],
                    "choices": [str(choice) for choice in state.choices],
                },
            )

        if "need_ocr_reading" in open_questions:
            if self._is_weight_task(state) and ingredient_name and combined_times:
                add_candidate(
                    1,
                    5,
                    1,
                    "称重题先查结构化称量记录，常能直接给出或约束候选重量。",
                    "当前最缺 OCR/称量证据，先查结构化称量记录。",
                    "query_ingredient_measurement",
                    {
                        "ingredient_name": str(ingredient_name),
                        "start_time": min(combined_times),
                        "end_time": max(combined_times),
                        "limit": 10,
                    },
                )
            if getattr(state, "retrieved_node_ids", []):
                add_candidate(
                    1,
                    4,
                    1,
                    "已有锚点节点时，先沿图关系扩展上下文，可能直接复用已写回 OCR 证据。",
                    "当前最缺 OCR 读数，先沿图关系扩展已知节点上下文。",
                    "expand_graph_context",
                    {
                        "node_ids": list(getattr(state, "retrieved_node_ids", [])[-8:]),
                        "edge_types": ["co_occurs", "same_step", "derived_from"],
                        "limit": 16,
                    },
                )
            if bbox and state.retrieved_frames:
                add_candidate(
                    2,
                    5,
                    1,
                    "已有局部目标且直接解决 OCR 缺口，收益高成本低。",
                    "当前最缺 OCR 读数，优先对局部候选区域做 OCR。",
                    "run_ocr_on_region",
                    {
                        "image_path": self._latest_visual_frame(state) or "",
                        "bbox": bbox,
                        "expand_ratio": 0.35,
                        "tag": f"{state.task_family}_state_ocr",
                    },
                )
            if state.retrieved_frames:
                latest_frame = self._latest_visual_frame(state)
                if latest_frame is None:
                    return
                add_candidate(
                    3,
                    4,
                    2,
                    "已有帧但没有局部定位，整图 OCR 成本低于重新抽帧。",
                    "当前最缺 OCR 读数，退而求其次对整图做 OCR。",
                    "run_ocr_on_image",
                    {"image_path": latest_frame},
                )
            if ocr_keyword:
                add_candidate(
                    1,
                    3,
                    1,
                    "先查已有 OCR 记忆最便宜，可先尝试复用历史证据。",
                    "当前最缺 OCR 读数，先检索已有 OCR 记忆。",
                    "query_ocr",
                    {
                        "keyword": str(ocr_keyword),
                        "start_time": min(combined_times) if combined_times else None,
                        "end_time": max(combined_times) if combined_times else None,
                        "limit": 12,
                    },
                )
        if "need_region_grounding" in open_questions and bbox and state.retrieved_frames:
            add_candidate(
                2,
                4,
                1,
                "画框可快速确认目标是否对齐，成本低。",
                "当前最缺区域定位证据，先画框确认目标。",
                "render_bbox_overlay",
                {"image_path": self._latest_visual_frame(state) or "", "bbox": bbox, "tag": f"{state.task_family}_state_bbox"},
            )
            add_candidate(
                3,
                4,
                2,
                "局部上下文图比整段重看更便宜，适合补区域细节。",
                "当前最缺区域定位证据，再补局部上下文图。",
                "extract_region_with_context",
                {"image_path": self._latest_visual_frame(state) or "", "bbox": bbox, "expand_ratio": 0.35, "tag": f"{state.task_family}_state_region"},
            )
            if combined_times:
                add_candidate(
                    4,
                    5,
                    3,
                    "解析到 object track 收益高，但依赖更强。",
                    "当前最缺区域定位证据，尝试从 bbox 解析到 object track。",
                    "resolve_bbox_reference",
                    {"bbox": bbox, "reference_time": combined_times[0], "limit": 5},
                )
        if "need_state_evidence" in open_questions:
            if getattr(state, "retrieved_node_ids", []):
                add_candidate(
                    1,
                    4,
                    1,
                    "已有锚点节点时，先沿关系边扩展同 step/前后时序上下文，可能直接补到状态线索。",
                    "当前最缺状态变化证据，先扩展已知节点的图关系上下文。",
                    "expand_graph_context",
                    {
                        "node_ids": list(getattr(state, "retrieved_node_ids", [])[-8:]),
                        "edge_types": ["same_step", "before", "after", "co_occurs"],
                        "limit": 16,
                    },
                )
            add_candidate(
                1,
                4,
                1,
                "状态检索便宜且可能直接命中已有记忆。",
                "当前最缺状态变化证据，先检索状态记忆。",
                "query_state",
                {
                    "state_keyword": str(state_keyword or "state"),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
            if state.retrieved_frames:
                if self._can_use_visual_inspection(state):
                    add_candidate(
                    4,
                    5,
                    2,
                    "视觉观察成本更高，但能补图谱检索拿不到的新状态线索。",
                    "当前最缺状态变化证据，补一次视觉观察。",
                    "inspect_visual_evidence",
                    {
                        "prompt": (
                            "请保守判断这组图像中的状态变化、关键动作和对象。"
                            '输出 JSON，字段固定为 {"ongoing_action":"","possible_step":"","target_object":"","state_change_hint":"","answer_hint":"","confidence":0.0}。'
                        ),
                        "image_paths": state.retrieved_frames[-6:],
                    },
                    )
        if "need_location_evidence" in open_questions and (explicit_location_need or "need_location_evidence" in verifier_missing):
            if self._is_viewpoint_task(state):
                if combined_times:
                    add_candidate(
                        1,
                        6,
                        1,
                        "3D/gaze 题应优先走空间上下文与专用视觉判断，而不是通用位置检索。",
                        "当前最缺位置证据，优先补当前时刻的空间上下文。",
                        "query_spatial_context",
                        {"time_s": combined_times[0], "object_name": None, "limit": 12},
                    )
                return candidates
            if getattr(state, "retrieved_node_ids", []):
                add_candidate(
                    1,
                    4,
                    1,
                    "已有锚点节点时，先扩展同对象/共现关系，可能直接补到位置线索。",
                    "当前最缺位置证据，先扩展已知节点的图关系上下文。",
                    "expand_graph_context",
                    {
                        "node_ids": list(getattr(state, "retrieved_node_ids", [])[-8:]),
                        "edge_types": ["same_object", "co_occurs", "before", "after"],
                        "limit": 16,
                    },
                )
            add_candidate(
                1,
                4,
                1,
                "位置检索便宜，可优先尝试已有空间记忆。",
                "当前最缺位置证据，先检索空间位置记忆。",
                "query_location",
                {
                    "location_keyword": str(location_keyword or "location"),
                    "start_time": min(combined_times) if combined_times else None,
                    "end_time": max(combined_times) if combined_times else None,
                    "limit": 12,
                },
            )
            if combined_times and state.task_family in {"3d_perception_fixture_location", "gaze_gaze_estimation"}:
                add_candidate(
                    2,
                    5,
                    2,
                    "空间上下文针对 3D/gaze 题收益高。",
                    "当前最缺位置证据，补当前时刻的空间上下文。",
                    "query_spatial_context",
                    {"time_s": combined_times[0], "object_name": None, "limit": 12},
                )
        if "need_time_localization" in open_questions or "need_initial_observation" in open_questions:
            if combined_times:
                add_candidate(
                    1,
                    4,
                    1,
                    "时间检索成本最低，应先试图复用已有时间记忆。",
                    "当前最缺时间定位/初始观察，先检索时间窗口记忆。",
                    "query_time",
                    {"start_time": min(combined_times), "end_time": max(combined_times), "limit": 20},
                )
                add_candidate(
                    4,
                    5,
                    2,
                    "稀疏抽帧更贵，但能直接补原始观察。",
                    "当前最缺时间定位/初始观察，回看原始视频做稀疏抽帧。",
                    "sample_sparse_frames",
                    {
                        "start_time": max(0.0, min(combined_times) - 2.0),
                        "end_time": max(combined_times) + 2.0,
                        "sample_count": 4,
                        "tag": f"{state.task_family}_state_frames",
                    },
                )
        if "need_disambiguating_evidence" in open_questions:
            self._add_disambiguating_candidates(
                state=state,
                combined_times=combined_times,
                bbox=bbox,
                ingredient_name=ingredient_name,
                object_hint=object_hint,
                add_candidate=add_candidate,
            )
        if "need_alternative_evidence_path" in open_questions and state.retrieved_frames and not self._is_weight_task(state):
            add_candidate(
                5,
                5,
                3,
                "已有路径失败后，视觉观察可提供跨模态替代证据。",
                "当前路径无效或失败，补一次视觉检查尝试换证据源。",
                "inspect_visual_evidence",
                {
                    "prompt": (
                        "已有路径失败或空转。"
                        "请保守补充当前图像中的对象、位置、动作、状态变化或读数。"
                        '输出 JSON，字段固定为 {"target_object":"","target_location":"","ongoing_action":"","state_change_hint":"","reading":"","answer_hint":"","confidence":0.0}。'
                    ),
                    "image_paths": state.retrieved_frames[-6:],
                    },
                )
        self._add_open_query_family_candidates(
            state=state,
            combined_times=combined_times,
            used_tools=used_tools,
            ocr_keyword=ocr_keyword,
            location_keyword=location_keyword,
            state_keyword=state_keyword,
            add_candidate=add_candidate,
        )
        self._add_specialized_followup_candidates(
            state=state,
            combined_times=combined_times,
            bbox=bbox,
            add_candidate=add_candidate,
        )
        return candidates

    def _tool_matches_verifier_need(
        self,
        *,
        tool: str,
        verifier_missing: set[str],
        verifier_conflicts: set[str],
        recommend_next_action: str,
    ) -> bool:
        mapping = {
            "need_ocr_reading": {"query_ocr", "run_ocr_on_image", "run_ocr_on_region"},
            "need_region_grounding": {"retrieve_cached_artifacts", "query_region", "render_bbox_overlay", "extract_region_with_context", "resolve_bbox_reference", "infer_object_drop_location", "infer_visual_mcq"},
            "need_state_evidence": {"retrieve_cached_artifacts", "query_state", "inspect_visual_evidence", "write_state_change"},
            "need_location_evidence": {"retrieve_cached_artifacts", "query_location", "query_spatial_context", "infer_viewpoint_choice", "infer_named_fixture_direction", "infer_gaze_target_with_context", "infer_object_drop_location"},
            "need_time_localization": {"retrieve_cached_artifacts", "query_time", "sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks", "infer_temporal_localization_choice"},
            "need_initial_observation": {"retrieve_cached_artifacts", "query_time", "sample_sparse_frames", "extract_frames_for_range", "inspect_visual_evidence"},
            "need_alternative_evidence_path": {"retrieve_cached_artifacts", "inspect_visual_evidence", "query_time", "sample_sparse_frames", "query_spatial_context"},
            "need_disambiguating_evidence": {
                "retrieve_cached_artifacts",
                "expand_graph_context",
                "query_region",
                "query_event",
                "query_ingredient_measurement",
                "query_time",
                "sample_sparse_frames",
                "inspect_visual_evidence",
                "detect_audio_peaks",
                "rank_choices_from_state",
                "infer_ingredient_order_choice",
                "infer_ingredient_retrieval_choice",
                "infer_recipe_ingredient_membership_choice",
                "infer_exact_ingredient_amount_choice",
                "infer_recipe_catalog_choice",
                "infer_recipe_nutrition_choice",
                "infer_visual_mcq",
                "infer_temporal_localization_choice",
            },
        }
        for missing in verifier_missing:
            if tool in mapping.get(missing, set()):
                return True
        if recommend_next_action and recommend_next_action in mapping and tool in mapping[recommend_next_action]:
            return True
        if recommend_next_action and tool == recommend_next_action:
            return True
        if "multiple_candidate_answers" in verifier_conflicts and tool in {
            "retrieve_cached_artifacts",
            "inspect_visual_evidence",
            "query_time",
            "query_state",
            "query_location",
            "query_region",
            "run_ocr_on_image",
            "run_ocr_on_region",
        }:
            return True
        if "conflicting_ocr_readings" in verifier_conflicts and tool in {
            "query_ocr",
            "run_ocr_on_image",
            "run_ocr_on_region",
            "inspect_visual_evidence",
        }:
            return True
        if "conflicting_locations" in verifier_conflicts and tool in {
            "retrieve_cached_artifacts",
            "query_location",
            "query_spatial_context",
            "infer_viewpoint_choice",
            "infer_named_fixture_direction",
            "infer_gaze_target_with_context",
            "inspect_visual_evidence",
        }:
            return True
        if "conflicting_state_observations" in verifier_conflicts and tool in {
            "retrieve_cached_artifacts",
            "query_state",
            "inspect_visual_evidence",
            "expand_graph_context",
        }:
            return True
        return False

    def _tool_addresses_needs(
        self,
        *,
        tool: str,
        needed_evidence: set[str],
        verifier_conflicts: set[str],
        recommend_next_action: str,
    ) -> bool:
        if not tool:
            return False
        return self._tool_matches_verifier_need(
            tool=tool,
            verifier_missing=needed_evidence,
            verifier_conflicts=verifier_conflicts,
            recommend_next_action=recommend_next_action,
        )

    def _current_evidence_needs(self, state: AgentState) -> set[str]:
        open_questions = {
            str(item)
            for item in getattr(state, "open_questions", []) or []
            if isinstance(item, str)
            and item
            and (item != "need_disambiguating_evidence" or str(state.task_family).startswith("open_query"))
        }
        latest_verification = self._state_latest_verification(state)
        verifier_missing = {
            str(item)
            for item in latest_verification.get("missing_evidence_types", [])
            if isinstance(item, str) and item
        }
        return open_questions | verifier_missing

    def _current_verifier_conflicts(self, state: AgentState) -> set[str]:
        latest_verification = self._state_latest_verification(state)
        return {
            str(item)
            for item in latest_verification.get("conflicts", [])
            if isinstance(item, str) and item
        }

    def _prefers_cheaper_memory_path(self, *, candidate_tool: str, decision_tool: str) -> bool:
        cheaper_tools = {
            "retrieve_cached_artifacts",
            "query_time",
            "query_event",
            "query_state",
            "query_location",
            "query_region",
            "query_ocr",
            "query_ingredient_measurement",
            "compute_nutrition_change",
            "compare_choice_nutrition",
            "query_spatial_context",
            "get_neighbors",
            "resolve_bbox_reference",
        }
        expensive_tools = {
            "sample_sparse_frames",
            "extract_frames_for_range",
            "sample_frames_around_peaks",
            "extract_frame_at_time",
            "render_bbox_overlay",
            "extract_region_with_context",
            "run_ocr_on_image",
            "run_ocr_on_region",
            "inspect_visual_evidence",
            "infer_visual_mcq",
            "infer_action_mechanism",
            "infer_action_intent",
            "infer_viewpoint_choice",
            "infer_named_fixture_direction",
            "infer_gaze_target_with_context",
        }
        return candidate_tool in cheaper_tools and decision_tool in expensive_tools

    def _open_query_needs_raw_grounding(self, *, state: AgentState, open_questions: list[str], used_tools: list[str]) -> bool:
        if not str(state.task_family).startswith("open_query"):
            return False
        latest_verification = self._state_latest_verification(state)
        if bool(latest_verification.get("sufficient")):
            return False
        if "need_disambiguating_evidence" not in open_questions and "need_initial_observation" not in open_questions:
            return False
        raw_grounding_tools = {
            "sample_sparse_frames",
            "extract_frames_for_range",
            "sample_frames_around_peaks",
            "inspect_visual_evidence",
            "run_ocr_on_image",
            "run_ocr_on_region",
            "detect_audio_peaks",
        }
        return not any(tool in raw_grounding_tools for tool in used_tools)

    def _open_query_has_reusable_raw_artifacts(self, state: AgentState) -> bool:
        if not str(getattr(state, "task_family", "")).startswith("open_query"):
            return False
        artifacts = getattr(state, "artifacts", None)
        if isinstance(artifacts, list) and any(isinstance(item, str) and item for item in artifacts):
            return True
        frames = getattr(state, "retrieved_frames", None)
        return bool(isinstance(frames, list) and any(isinstance(item, str) and item for item in frames))

    def _best_reusable_open_query_time(self, state: AgentState, combined_times: list[float]) -> float | None:
        visited = getattr(state, "visited_times", None)
        if not isinstance(visited, list) or not visited:
            return combined_times[0] if combined_times else None
        numeric = []
        for item in visited:
            try:
                numeric.append(float(item))
            except Exception:  # noqa: BLE001
                continue
        if not numeric:
            return combined_times[0] if combined_times else None
        if not combined_times:
            return numeric[-1]
        anchor = combined_times[0]
        return sorted(numeric, key=lambda value: (abs(value - anchor), value))[0]

    def _task_has_reusable_artifacts(self, state: AgentState, *, prefixes: tuple[str, ...] | None = None) -> bool:
        artifacts = getattr(state, "artifacts", None)
        if not isinstance(artifacts, list) or not artifacts:
            return False
        if not prefixes:
            return any(isinstance(item, str) and item for item in artifacts)
        lowered_prefixes = tuple(prefix.lower() for prefix in prefixes if prefix)
        for item in artifacts:
            if not isinstance(item, str) or not item:
                continue
            lowered = item.lower()
            if any(prefix in lowered for prefix in lowered_prefixes):
                return True
        return False

    def _is_raw_grounding_tool(self, tool: str) -> bool:
        return tool in {
            "retrieve_cached_artifacts",
            "sample_sparse_frames",
            "extract_frames_for_range",
            "sample_frames_around_peaks",
            "extract_frame_at_time",
            "inspect_visual_evidence",
            "run_ocr_on_image",
            "run_ocr_on_region",
            "detect_audio_peaks",
        }

    def _state_latest_verification(self, state: AgentState) -> dict[str, Any]:
        latest = getattr(state, "latest_verification", None)
        if callable(latest):
            payload = latest()
            return payload if isinstance(payload, dict) else {}
        history = getattr(state, "verification_history", None)
        if isinstance(history, list) and history:
            item = history[-1]
            return item if isinstance(item, dict) else {}
        snapshot = getattr(state, "snapshot", None)
        if callable(snapshot):
            payload = snapshot()
            if isinstance(payload, dict):
                history = payload.get("verification_history")
                if isinstance(history, list) and history:
                    item = history[-1]
                    return item if isinstance(item, dict) else {}
        return {}

    def _used_tools(self, state: AgentState) -> list[str]:
        return [entry.get("tool") for entry in getattr(state, "tool_trace", []) if isinstance(entry, dict)]

    def _latest_tool_result(self, state: AgentState, tool_name: str) -> dict[str, Any]:
        for entry in reversed(getattr(state, "tool_trace", [])):
            if isinstance(entry, dict) and entry.get("tool") == tool_name:
                payload = entry.get("raw_result")
                return payload if isinstance(payload, dict) else {}
        return {}

    def _latest_successful_action_intent_result(self, state: AgentState) -> dict[str, Any]:
        for entry in reversed(getattr(state, "tool_trace", [])):
            if not isinstance(entry, dict) or entry.get("tool") != "infer_action_intent":
                continue
            payload = entry.get("raw_result")
            if not isinstance(payload, dict) or payload.get("tool_failed"):
                continue
            if payload.get("best_index") is not None:
                return payload
        return {}

    def _state_add_memory(self, state: AgentState, text: str) -> None:
        add_memory = getattr(state, "add_memory", None)
        if callable(add_memory):
            add_memory(text)
            return
        working_memory = getattr(state, "working_memory", None)
        if isinstance(working_memory, list) and text not in working_memory:
            working_memory.append(text)

    def _state_add_hypothesis(self, state: AgentState, text: str) -> None:
        add_hypothesis = getattr(state, "add_hypothesis", None)
        if callable(add_hypothesis):
            add_hypothesis(text)
            return
        hypotheses = getattr(state, "hypotheses", None)
        if isinstance(hypotheses, list) and text not in hypotheses:
            hypotheses.append(text)

    def _segment_stride_s(self, combined_times: list[float], *, fallback: float = 0.4) -> float:
        if len(combined_times) > 1:
            return max(0.3, (max(combined_times) - min(combined_times)) / 2)
        return fallback

    def _artifact_reuse_prefixes(self, state: AgentState) -> tuple[str, ...]:
        return artifact_reuse_prefixes_for_task(str(getattr(state, "task_family", "") or ""))

    def _build_reuse_or_extract_range_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        tag_hint: str,
        artifact_prefixes: tuple[str, ...],
        start_time: float,
        end_time: float,
        reuse_thought: str,
        extract_thought: str,
        extract_tag: str,
        stride_s: float | None = None,
        max_frames: int | None = None,
        artifact_limit: int = 6,
    ) -> PlannerDecision | None:
        if state.retrieved_frames:
            return None
        if "retrieve_cached_artifacts" not in used_tools and self._task_has_reusable_artifacts(state, prefixes=artifact_prefixes):
            return PlannerDecision(
                thought=reuse_thought,
                tool="retrieve_cached_artifacts",
                args={
                    "tag_hint": tag_hint,
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": artifact_limit,
                },
            )
        duration = max(0.5, float(end_time) - float(start_time))
        resolved_max_frames = max_frames if isinstance(max_frames, int) and max_frames > 0 else 4
        resolved_stride = stride_s if isinstance(stride_s, (int, float)) and float(stride_s) > 0 else max(0.5, duration / max(resolved_max_frames, 1))
        return PlannerDecision(
            thought=extract_thought,
            tool="extract_frames_for_range",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "stride_s": float(resolved_stride),
                "max_frames": int(resolved_max_frames),
                "tag": extract_tag,
            },
        )

    def _build_raw_reuse_or_resample_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        failed_tools: set[str],
        ineffective_tools: set[str],
        combined_times: list[float],
        tag_hint: str,
        sample_tag: str,
        sample_count: int,
        retrieve_limit: int,
        retrieve_thought: str,
        revisit_thought: str,
        resample_thought: str,
    ) -> PlannerDecision | None:
        if not combined_times:
            return None
        start_time = max(0.0, min(combined_times) - 2.0)
        end_time = max(combined_times) + 2.0
        artifact_prefixes = self._artifact_reuse_prefixes(state)
        reusable_time = self._best_reusable_open_query_time(state, combined_times)
        if (
            "retrieve_cached_artifacts" not in used_tools
            and "retrieve_cached_artifacts" not in failed_tools
            and "retrieve_cached_artifacts" not in ineffective_tools
            and (
                self._task_has_reusable_artifacts(state, prefixes=artifact_prefixes)
                or self._open_query_has_reusable_raw_artifacts(state)
            )
        ):
            return PlannerDecision(
                thought=retrieve_thought,
                tool="retrieve_cached_artifacts",
                args={
                    "tag_hint": tag_hint,
                    "time_s": reusable_time,
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": retrieve_limit,
                },
            )
        if (
            reusable_time is not None
            and "extract_frame_at_time" not in used_tools
            and "extract_frame_at_time" not in failed_tools
            and "extract_frame_at_time" not in ineffective_tools
            and (
                self._task_has_reusable_artifacts(state, prefixes=artifact_prefixes)
                or bool(getattr(state, "visited_times", None))
            )
        ):
            return PlannerDecision(
                thought=revisit_thought,
                tool="extract_frame_at_time",
                args={"time_s": reusable_time, "tag": sample_tag},
            )
        if (
            "sample_sparse_frames" not in used_tools
            and "sample_sparse_frames" not in failed_tools
            and "sample_sparse_frames" not in ineffective_tools
        ):
            return PlannerDecision(
                thought=resample_thought,
                tool="sample_sparse_frames",
                args={
                    "start_time": start_time,
                    "end_time": end_time,
                    "sample_count": sample_count,
                    "tag": sample_tag,
                },
            )
        return None

    def _build_region_reuse_or_recrop_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        failed_tools: set[str],
        ineffective_tools: set[str],
        bbox: Any,
        image_path: str,
        tag_hint: str,
        overlay_tag: str,
        region_tag: str,
        region_expand_ratio: float = 0.35,
    ) -> PlannerDecision | None:
        if (
            "retrieve_cached_artifacts" not in used_tools
            and "retrieve_cached_artifacts" not in failed_tools
            and "retrieve_cached_artifacts" not in ineffective_tools
            and self._task_has_reusable_artifacts(state)
        ):
            return PlannerDecision(
                thought="当前仍缺区域定位证据，先复用当前视频已有 bbox/region artifact，而不是立即重画框或重裁剪。",
                tool="retrieve_cached_artifacts",
                args={
                    "tag_hint": tag_hint,
                    "start_time": None,
                    "end_time": None,
                    "limit": 6,
                },
            )
        if (
            "render_bbox_overlay" not in used_tools
            and "render_bbox_overlay" not in failed_tools
            and "render_bbox_overlay" not in ineffective_tools
        ):
            return PlannerDecision(
                thought="当前评分置信度不足，且仍缺区域定位证据，转为先画框确认目标。",
                tool="render_bbox_overlay",
                args={"image_path": image_path, "bbox": bbox, "tag": overlay_tag},
            )
        if (
            "extract_region_with_context" not in used_tools
            and "extract_region_with_context" not in failed_tools
            and "extract_region_with_context" not in ineffective_tools
        ):
            return PlannerDecision(
                thought="当前评分置信度不足，且仍缺区域定位证据，转为补局部上下文图。",
                tool="extract_region_with_context",
                args={"image_path": image_path, "bbox": bbox, "expand_ratio": region_expand_ratio, "tag": region_tag},
            )
        return None

    def _enforce_structured_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        thought_prefix: str,
    ) -> PlannerDecision | None:
        if decision.tool != "finish":
            return None
        config = self._structured_direct_inference_config(state)
        if config is None:
            return None
        tool, thought, args = config
        if tool in used_tools:
            return None
        return PlannerDecision(
            thought=f"{thought_prefix}{thought}",
            tool=tool,
            args=args,
        )

    def _enforce_nutrition_image_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
    ) -> PlannerDecision | None:
        if state.task_family != "nutrition_image_nutrition_estimation" or decision.tool != "finish":
            return None
        if "extract_input_reference_frames" not in used_tools:
            return PlannerDecision(
                thought="多图营养题在 finish 前必须先提取跨视频参考图。",
                tool="extract_input_reference_frames",
                args={"tag": f"{state.task_family}_inputs"},
            )
        if "identify_image_ingredients" not in used_tools and state.retrieved_frames:
            return PlannerDecision(
                thought="多图营养题在 finish 前必须先识别参考图中的食材。",
                tool="identify_image_ingredients",
                args={"image_paths": state.retrieved_frames[-10:]},
            )
        if "compare_choice_nutrition" not in used_tools:
            nutrient = "carbs" if "carb" in state.question.lower() else "calories"
            return PlannerDecision(
                thought="多图营养题在 finish 前必须先比较候选食材的结构化营养值。",
                tool="compare_choice_nutrition",
                args={"choices": [str(choice) for choice in state.choices], "nutrient": nutrient},
            )
        return None

    def _enforce_bbox_structured_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        combined_times: list[float],
        bbox: Any,
    ) -> PlannerDecision | None:
        if decision.tool != "finish" or bbox is None or not combined_times:
            return None
        if state.task_family in {
            "object_motion_object_movement_counting",
            "object_motion_stationary_object_localization",
        }:
            resolved_motion = self._resolve_object_motion_choice_from_state(state)
            if resolved_motion is None:
                return self._bbox_structured_task_decision(
                    state=state,
                    used_tools=used_tools,
                    combined_times=combined_times,
                    bbox=bbox,
                )
        if self._is_object_location_task(state):
            resolved_location = self._resolve_object_location_choice_from_state(state)
            if resolved_location is None:
                return self._bbox_structured_task_decision(
                    state=state,
                    used_tools=used_tools,
                    combined_times=combined_times,
                    bbox=bbox,
                )
        return None

    def _enforce_fixture_count_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        bbox: Any,
    ) -> PlannerDecision | None:
        if (
            state.task_family != "3d_perception_fixture_interaction_counting"
            or bbox is None
            or decision.tool != "finish"
            or "count_visual_candidates" in used_tools
        ):
            return None
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
        return None

    def _enforce_recipe_event_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        combined_times: list[float],
    ) -> PlannerDecision | None:
        if not self._is_recipe_step_evidence_task(state) or decision.tool != "finish" or "query_event" in used_tools:
            return None
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

    def _enforce_temporal_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
    ) -> PlannerDecision | None:
        if not self._is_temporal_localization_task(state) or decision.tool != "finish":
            return None
        temporal_resolved = any(
            isinstance(item, str) and item.startswith("temporal_localization_best_index=")
            for item in list(getattr(state, "working_memory", [])) + list(getattr(state, "evidence_bundle", []))
        )
        if "infer_temporal_localization_choice" not in used_tools and not temporal_resolved:
            return PlannerDecision(
                thought="时间定位题在 finish 前必须先比较候选时间段关键帧。",
                tool="infer_temporal_localization_choice",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "task_family": state.task_family,
                    "frames_per_choice": 2,
                    "tag": f"{state.task_family}_finish_temporal",
                },
            )
        return None

    def _enforce_temporal_rank_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
    ) -> PlannerDecision | None:
        if not self._is_temporal_localization_task(state) or decision.tool != "rank_choices_from_state":
            return None
        temporal_resolved = any(
            isinstance(item, str) and item.startswith("temporal_localization_best_index=")
            for item in list(getattr(state, "working_memory", [])) + list(getattr(state, "evidence_bundle", []))
        )
        if "infer_temporal_localization_choice" in used_tools or temporal_resolved:
            return None
        return PlannerDecision(
            thought="时间定位题在进入通用评分前，必须先用专用 temporal localization 工具比较候选时间段。",
            tool="infer_temporal_localization_choice",
            args={
                "question": state.question,
                "choices": [str(choice) for choice in state.choices],
                "task_family": state.task_family,
                "frames_per_choice": 2,
                "tag": f"{state.task_family}_rank_temporal",
            },
        )

    def _enforce_specialized_rank_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        combined_times: list[float],
        bbox: Any,
    ) -> PlannerDecision | None:
        if decision.tool != "rank_choices_from_state":
            return None
        if self._is_temporal_localization_task(state):
            return None
        if self._is_segment_visual_task(state):
            return self._segment_task_finish_requirement(
                state=state,
                decision=PlannerDecision(thought=decision.thought, tool="finish", args=decision.args),
                used_tools=used_tools,
                combined_times=combined_times,
            )
        if state.task_family in {"3d_perception_fixture_location", "gaze_gaze_estimation"}:
            return self._enforce_viewpoint_finish_requirement(
                state=state,
                decision=PlannerDecision(thought=decision.thought, tool="finish", args=decision.args),
                used_tools=used_tools,
                combined_times=combined_times,
            )
        if self._is_object_contents_task(state):
            return self._object_contents_visual_requirement(
                state=state,
                decision=PlannerDecision(thought=decision.thought, tool="finish", args=decision.args),
                used_tools=used_tools,
                combined_times=combined_times,
                bbox=bbox,
            )
        if state.task_family == "3d_perception_fixture_interaction_counting":
            return self._enforce_fixture_count_finish_requirement(
                state=state,
                decision=PlannerDecision(thought=decision.thought, tool="finish", args=decision.args),
                used_tools=used_tools,
                bbox=bbox,
            )
        if state.task_family in {
            "object_motion_object_movement_counting",
            "object_motion_object_movement_itinerary",
            "object_motion_stationary_object_localization",
            "3d_perception_object_location",
        }:
            return self._enforce_bbox_structured_finish_requirement(
                state=state,
                decision=PlannerDecision(thought=decision.thought, tool="finish", args=decision.args),
                used_tools=used_tools,
                combined_times=combined_times,
                bbox=bbox,
            )
        return None

    def _enforce_viewpoint_finish_requirement(
        self,
        *,
        state: AgentState,
        decision: PlannerDecision,
        used_tools: list[str],
        combined_times: list[float],
    ) -> PlannerDecision | None:
        if state.task_family not in {"3d_perception_fixture_location", "gaze_gaze_estimation"}:
            return None
        required_tool = "infer_named_fixture_direction" if state.task_family == "3d_perception_fixture_location" else "infer_gaze_target_with_context"
        resolved_viewpoint = self._resolve_viewpoint_choice_from_state(state)
        if required_tool in used_tools or resolved_viewpoint is not None:
            return None
        if not state.retrieved_frames and combined_times:
            return self._build_reuse_or_extract_range_decision(
                state=state,
                used_tools=used_tools,
                tag_hint=state.task_family,
                artifact_prefixes=self._artifact_reuse_prefixes(state),
                start_time=max(0.0, min(combined_times) - 0.5),
                end_time=max(combined_times) + 0.5,
                reuse_thought="视角类题先检索当前视频中已经抽取过的视角 artifact，优先复用已有帧。",
                extract_thought="视角定位题必须先抽当前视角关键帧。",
                extract_tag=f"{state.task_family}_view",
                stride_s=0.5,
                max_frames=3,
            )
        if state.task_family == "3d_perception_fixture_location" and "query_spatial_context" not in used_tools and combined_times:
            return PlannerDecision(
                thought="fixture 方位题在 finish 前必须先查询附近的空间候选。",
                tool="query_spatial_context",
                args={"time_s": combined_times[0], "object_name": None, "limit": 12},
            )
        if state.task_family == "gaze_gaze_estimation" and "query_spatial_context" not in used_tools and combined_times:
            return PlannerDecision(
                thought="注视目标题在 finish 前必须先查询该时刻的空间上下文。",
                tool="query_spatial_context",
                args={"time_s": combined_times[0], "object_name": None, "limit": 12},
            )
        if state.retrieved_frames and decision.tool == "finish":
            last_spatial = next(
                (
                    entry.get("raw_result")
                    for entry in reversed(state.tool_trace)
                    if isinstance(entry, dict) and entry.get("tool") == "query_spatial_context"
                ),
                {},
            )
            thought = (
                "视角定位题在 finish 前必须先做具名 fixture 方向判断。"
                if state.task_family == "3d_perception_fixture_location"
                else "注视目标题在 finish 前必须先结合空间上下文做专用判断。"
            )
            return PlannerDecision(
                thought=thought,
                tool=required_tool,
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": state.retrieved_frames[-3:],
                    "spatial_context": last_spatial if isinstance(last_spatial, dict) else {},
                },
            )
        return None

    def _enforce_task_requirements(self, *, state: AgentState, hints: dict[str, Any], decision: PlannerDecision) -> PlannerDecision:
        used_tools = [entry.get("tool") for entry in state.tool_trace if isinstance(entry, dict)]
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        bbox = hints.get("bbox")
        ingredient_name = hints.get("ingredient_name")

        structured_requirement = self._enforce_structured_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            thought_prefix="该题在 finish 前必须先走结构化主路径：",
        )
        if structured_requirement is not None:
            return structured_requirement

        if state.task_family == "ingredient_ingredient_weight" and decision.tool == "finish":
            if self._has_stable_weight_answer_evidence(state):
                return decision
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
                if "retrieve_cached_artifacts" not in used_tools and self._task_has_reusable_artifacts(
                    state,
                    prefixes=self._artifact_reuse_prefixes(state),
                ):
                    return PlannerDecision(
                        thought="称重题在 finish 前先复用当前视频里已经抽取过的称重相关 artifact。",
                        tool="retrieve_cached_artifacts",
                        args={
                            "tag_hint": state.task_family,
                            "start_time": max(0.0, min(combined_times) - 2.0),
                            "end_time": max(combined_times) + 2.0,
                            "limit": 6,
                        },
                    )
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
            if self._can_use_visual_inspection(state) and "inspect_visual_evidence" not in used_tools and state.retrieved_frames:
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

        if state.task_family == "nutrition_nutrition_change" and decision.tool == "finish":
            if "compute_nutrition_change" not in used_tools and combined_times:
                return PlannerDecision(
                    thought="营养变化题在 finish 前必须先计算时间窗口内营养增量。",
                    tool="compute_nutrition_change",
                    args={"start_time": min(combined_times), "end_time": max(combined_times)},
                )

        nutrition_image_requirement = self._enforce_nutrition_image_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
        )
        if nutrition_image_requirement is not None:
            return nutrition_image_requirement

        bbox_structured_requirement = self._enforce_bbox_structured_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            combined_times=combined_times,
            bbox=bbox,
        )
        if bbox_structured_requirement is not None:
            return bbox_structured_requirement

        fixture_count_requirement = self._enforce_fixture_count_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            bbox=bbox,
        )
        if fixture_count_requirement is not None:
            return fixture_count_requirement

        bbox_visual_requirement = self._bbox_visual_finalize_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            bbox=bbox,
        )
        if bbox_visual_requirement is not None:
            return bbox_visual_requirement

        recipe_event_requirement = self._enforce_recipe_event_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            combined_times=combined_times,
        )
        if recipe_event_requirement is not None:
            return recipe_event_requirement

        temporal_requirement = self._enforce_temporal_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
        )
        if temporal_requirement is not None:
            return temporal_requirement

        temporal_rank_requirement = self._enforce_temporal_rank_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
        )
        if temporal_rank_requirement is not None:
            return temporal_rank_requirement

        specialized_rank_requirement = self._enforce_specialized_rank_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            combined_times=combined_times,
            bbox=bbox,
        )
        if specialized_rank_requirement is not None:
            return specialized_rank_requirement

        object_contents_requirement = self._object_contents_visual_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            combined_times=combined_times,
            bbox=bbox,
        )
        if object_contents_requirement is not None:
            return object_contents_requirement

        viewpoint_requirement = self._enforce_viewpoint_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            combined_times=combined_times,
        )
        if viewpoint_requirement is not None:
            return viewpoint_requirement

        segment_requirement = self._segment_task_finish_requirement(
            state=state,
            decision=decision,
            used_tools=used_tools,
            combined_times=combined_times,
        )
        if segment_requirement is not None:
            return segment_requirement

        return decision
