"""LLM planner for multi-step graph/video tool calling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from food_agent.agent.action_intent import (
    action_intent_conflict_profile,
    action_intent_followup_decision,
    action_intent_needs_future_use_resolution,
    action_intent_needs_pairwise_resolution,
    action_intent_needs_precondition_context,
    selected_choice_categories,
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

    def _action_intent_followup_route(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
        candidate_indices: list[int] | None = None,
    ) -> tuple[bool, str, float, str]:
        if not self._is_action_intent_task(state):
            return False, "", 4.0, ""
        try:
            confidence = float((result or {}).get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        reason_text = str((result or {}).get("reason") or "")
        indices = candidate_indices or self._latest_action_intent_candidate_indices(state, result=result)
        semantic_indices = self._action_intent_route_candidate_indices(
            state=state,
            result=result,
            candidate_indices=indices,
            confidence=confidence,
            reason_text=reason_text,
        )
        timeline_hint = self._action_intent_timeline_review_resolver_hint(
            state=state,
            candidate_indices=semantic_indices or indices,
        )
        if semantic_indices is not None:
            needs, reason, window_s, resolver = action_intent_followup_decision(
                question=str(getattr(state, "question", "") or ""),
                choices=[str(choice) for choice in getattr(state, "choices", [])],
                indices=semantic_indices,
                confidence=confidence,
                reason_text=reason_text,
            )
            if needs:
                if timeline_hint == "future_use":
                    return needs, reason or "timeline_review_future_use_hint", max(window_s, 8.0), "future_use"
                if timeline_hint == "pairwise":
                    return needs, reason or "timeline_review_pairwise_hint", min(max(window_s, 4.0), 6.0), "pairwise"
                return needs, reason, window_s, resolver
            question_text = str(getattr(state, "question", "") or "").lower()
            requires_full_choice_recheck = (
                (
                    any(token in question_text for token in ("paper towel", "tea towel", "dish cloth", "cloth", "towel", "napkin"))
                    and any(token in question_text for token in ("<flip ", "<turn ", "<shake ", " flip ", " turn ", " shake "))
                )
                or any(token in question_text for token in ("<tap kitchen scale>", "tap kitchen scale"))
            )
            if requires_full_choice_recheck:
                needs, reason, window_s, resolver = action_intent_followup_decision(
                    question=str(getattr(state, "question", "") or ""),
                    choices=[str(choice) for choice in getattr(state, "choices", [])],
                    indices=None,
                    confidence=confidence,
                    reason_text=reason_text,
                )
                if needs:
                    if timeline_hint == "future_use":
                        return needs, reason or "timeline_review_future_use_hint", max(window_s, 8.0), "future_use"
                    if timeline_hint == "pairwise":
                        return needs, reason or "timeline_review_pairwise_hint", min(max(window_s, 4.0), 6.0), "pairwise"
                    return needs, reason, window_s, resolver
        if semantic_indices is None:
            needs, reason, window_s, resolver = action_intent_followup_decision(
                question=str(getattr(state, "question", "") or ""),
                choices=[str(choice) for choice in getattr(state, "choices", [])],
                indices=None,
                confidence=confidence,
                reason_text=reason_text,
            )
            if needs:
                if timeline_hint == "future_use":
                    return needs, reason or "timeline_review_future_use_hint", max(window_s, 8.0), "future_use"
                if timeline_hint == "pairwise":
                    return needs, reason or "timeline_review_pairwise_hint", min(max(window_s, 4.0), 6.0), "pairwise"
                return needs, reason, window_s, resolver
        pending_tool = self._action_intent_pending_resolution_tool(state)
        if pending_tool == "resolve_action_intent_future_use":
            return True, "pending_future_use_resolution", 8.0, "future_use"
        if pending_tool == "resolve_action_intent_pairwise":
            return True, "pending_pairwise_resolution", 4.0, "pairwise"
        return False, "", 4.0, ""

    def _action_intent_timeline_review_resolver_hint(
        self,
        *,
        state: AgentState,
        candidate_indices: list[int] | None = None,
    ) -> str:
        timeline_text = self._action_intent_timeline_review_text(state)
        if not timeline_text:
            return ""
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        indices = candidate_indices if candidate_indices else list(range(len(choices)))
        categories_by_index = selected_choice_categories(choices, indices if indices else None)
        active_categories = set()
        for cats in categories_by_index.values():
            active_categories.update(cats)
        future_use_markers = (
            "next use",
            "used next",
            "use again",
            "put back",
            "return",
            "returned",
            "placed on",
            "placed into",
            "pick up",
            "poured",
            "pour",
            "weigh",
            "scale",
            "later use",
            "后续用途",
            "放回",
            "放到",
            "称",
        )
        pairwise_markers = (
            "free hand",
            "freed hand",
            "other hand",
            "right hand",
            "left hand",
            "reach toward",
            "tap area",
            "sink area",
            "turn on",
            "turn off",
            "open",
            "close",
            "reveal",
            "visible behind",
            "make space",
            "clear space",
            "right place",
            "露出",
            "腾出",
            "另一只手",
            "后面",
            "空位",
        )
        if any(marker in timeline_text for marker in future_use_markers):
            if active_categories & {
                "measure_weigh",
                "transfer_contents",
                "serve_consume",
                "inspect_check",
                "open_close",
                "food_prep",
                "discard",
                "final_place_return",
                "access_retrieve",
                "hand_free_enablement",
            }:
                return "future_use"
        if any(marker in timeline_text for marker in pairwise_markers):
            if active_categories & {
                "hand_free_enablement",
                "access_retrieve",
                "space_clear",
                "final_place_return",
                "generic_relocation",
                "open_close",
            }:
                return "pairwise"
        return ""

    def _action_intent_route_candidate_indices(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        candidate_indices: list[int],
        confidence: float,
        reason_text: str,
    ) -> list[int] | None:
        indices = candidate_indices if len(candidate_indices) >= 2 else []
        if len(indices) < 2:
            return None
        question = str(getattr(state, "question", "") or "")
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if len(indices) <= 2 or not isinstance(result, dict):
            return indices
        try:
            best_index = int(result.get("best_index"))
            second_best_index = int(result.get("second_best_index"))
        except Exception:  # noqa: BLE001
            return indices
        if best_index not in indices or second_best_index not in indices or best_index == second_best_index:
            return indices
        top_pair = [best_index, second_best_index]
        full_needs, _, _, full_resolver = action_intent_followup_decision(
            question=question,
            choices=choices,
            indices=indices,
            confidence=confidence,
            reason_text=reason_text,
        )
        pair_needs, _, _, pair_resolver = action_intent_followup_decision(
            question=question,
            choices=choices,
            indices=top_pair,
            confidence=confidence,
            reason_text=reason_text,
        )
        full_profile = action_intent_conflict_profile(question=question, choices=choices, indices=indices)
        if not (full_needs and pair_needs):
            pair_profile = action_intent_conflict_profile(question=question, choices=choices, indices=top_pair)
        else:
            pair_profile = action_intent_conflict_profile(question=question, choices=choices, indices=top_pair)
        if (
            pair_resolver == "future_use"
            and full_resolver == "pairwise"
            and "hand_free_enablement" in set(pair_profile["active_categories"])
        ):
            return top_pair
        if "hand_free_enablement" in set(full_profile["active_categories"]):
            hand_free_candidates = [
                index
                for index in indices
                if "hand_free_enablement" in set(full_profile["categories_by_index"].get(index) or set())
            ]
            best_categories = set(full_profile["categories_by_index"].get(best_index) or set())
            if hand_free_candidates and (
                "access_retrieve" in best_categories
                or "final_place_return" in best_categories
                or "generic_relocation" in best_categories
            ):
                hand_free_pair = next(
                    ([best_index, index] for index in hand_free_candidates if index != best_index),
                    None,
                )
                if hand_free_pair is not None:
                    alt_needs, _, _, alt_resolver = action_intent_followup_decision(
                        question=question,
                        choices=choices,
                        indices=hand_free_pair,
                        confidence=confidence,
                        reason_text=reason_text,
                    )
                    alt_profile = action_intent_conflict_profile(question=question, choices=choices, indices=hand_free_pair)
                    if (
                        alt_needs
                        and alt_resolver == "future_use"
                        and "hand_free_enablement" in set(alt_profile["active_categories"])
                    ):
                        return hand_free_pair
        return indices

    def _action_intent_requires_followup(self, state: AgentState, result: dict[str, Any] | None = None) -> bool:
        if isinstance(result, dict):
            if bool(result.get("need_future_evidence")) or bool(result.get("ambiguity")):
                return True
            if self._action_intent_result_has_direct_post_action_evidence(result) and not self._action_intent_direct_evidence_still_needs_resolution(
                state=state,
                result=result,
            ):
                return False
            needs_followup, _, _, _ = self._action_intent_followup_route(state=state, result=result)
            if needs_followup:
                return True
            if self._action_intent_result_needs_generalized_disambiguation(state=state, result=result):
                return True
            try:
                confidence = float(result.get("confidence") or 0.0)
            except Exception:  # noqa: BLE001
                confidence = 0.0
            candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
            profile = action_intent_conflict_profile(
                question=str(getattr(state, "question", "") or ""),
                choices=[str(choice) for choice in getattr(state, "choices", [])],
                indices=candidate_indices if candidate_indices else None,
            )
            if (
                self._action_intent_prefers_result_driven_followup(state)
                and len(set(profile["active_categories"])) >= 2
                and confidence < 0.9
            ):
                return True
        return any(
            isinstance(item, str) and item.startswith("action_intent_need_future_evidence=1")
            for item in list(getattr(state, "working_memory", [])) + list(getattr(state, "evidence_bundle", []))
        )

    def _action_intent_result_support_text(self, result: dict[str, Any] | None) -> str:
        if not isinstance(result, dict):
            return ""
        return " ".join(
            str(result.get(key) or "")
            for key in (
                "reason",
                "decisive_observation",
                "direct_effect",
                "downstream_action",
                "needed_observation",
                "answer",
            )
        ).strip().lower()

    def _action_intent_result_has_direct_post_action_evidence(self, result: dict[str, Any] | None) -> bool:
        if not isinstance(result, dict):
            return False
        text = " ".join(
            str(result.get(key) or "")
            for key in ("reason", "decisive_observation", "direct_effect", "downstream_action")
        ).strip().lower()
        if not text:
            return False
        strong_result_terms = (
            "immediately",
            "right after",
            "shortly after",
            "next step",
            "afterwards",
            "placed on the scale",
            "used on the scale",
            "display changes",
            "changes to 0",
            "wiped",
            "wipe both hands",
            "wipe the hands",
            "wipe the counter",
            "dried the hands",
            "dry the hands",
            "under running water",
            "returned to",
            "put back",
            "stored",
            "poured",
            "falls back",
            "fall back",
            "picked up from behind",
            "taken from behind",
            "retrieved from behind",
            "placed into the freed slot",
            "put into the freed slot",
            "immediately picks up",
            "used again shortly after",
            "no further",
            "turned off",
            "turned on",
            "opened",
            "closed",
            "明确看到",
            "直接看到",
            "立刻",
            "随后",
            "接着",
            "放到秤上",
            "开始擦",
            "放回",
            "取到后面的",
        )
        blocked_terms = (
            "not enough",
            "insufficient",
            "unclear",
            "uncertain",
            "ambiguous",
            "cannot tell",
            "can't tell",
            "not visible",
            "not shown",
            "no visible",
            "no actual",
            "missing",
            "lack",
            "merely",
            "simply",
            "only briefly",
            "briefly",
            "visible in hand",
            "near the counter",
            "picked up but not yet used",
            "picked up but the next use is not shown",
            "still unclear",
            "still contested",
            "未显示",
            "没有看到",
            "看不清",
            "不明确",
            "仅仅",
            "只是",
        )
        return self._action_intent_text_has_direct_outcome_clause(
            text=text,
            strong_result_terms=strong_result_terms,
            blocked_terms=blocked_terms,
        )

    def _action_intent_text_has_direct_outcome_clause(
        self,
        *,
        text: str,
        strong_result_terms: tuple[str, ...],
        blocked_terms: tuple[str, ...],
    ) -> bool:
        if not text.strip():
            return False
        normalized = str(text).lower()
        for separator in ("\n", ";", ".", ", but ", " but ", " however ", " although ", " though "):
            normalized = normalized.replace(separator, "|")
        clauses = [clause.strip() for clause in normalized.split("|") if clause.strip()]
        if not clauses:
            clauses = [normalized.strip()]
        for clause in clauses:
            if not any(term in clause for term in strong_result_terms):
                continue
            if any(term in clause for term in blocked_terms):
                continue
            if any(
                token in clause
                for token in (
                    "whether",
                    "not yet visible whether",
                    "it may",
                    "may be",
                    "might be",
                    "could be",
                    "could still be",
                    "still remains plausible",
                    "remains plausible",
                    "possible next",
                    "it is possible",
                    "是否",
                    "可能",
                )
            ):
                continue
            return True
        return False

    def _action_intent_has_peak_guided_followup_frames(self, state: AgentState) -> bool:
        for path in self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or [])):
            if "_followup_peaks_" in Path(path).name.lower():
                return True
        return False

    def _action_intent_has_transition_followup_frames(self, state: AgentState) -> bool:
        for path in self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or [])):
            if "_followup_transition_" in Path(path).name.lower():
                return True
        return False

    def _action_intent_result_has_indecisive_post_action_support(self, result: dict[str, Any] | None) -> bool:
        text = self._action_intent_result_support_text(result)
        if not text:
            return True
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        weak_terms = (
            "not enough",
            "insufficient",
            "unclear",
            "uncertain",
            "ambiguous",
            "cannot tell",
            "can't tell",
            "not visible",
            "not shown",
            "no visible",
            "no actual",
            "missing",
            "lack",
            "need more evidence",
            "still unclear",
            "still contested",
            "merely",
            "simply",
            "only briefly",
            "briefly",
            "visible in hand",
            "near the counter",
            "after pickup",
            "picked up but not yet used",
            "picked up but the next use is not shown",
            "moved aside",
            "becomes visible",
            "becomes reachable",
            "reveals the area",
            "revealed area",
            "看不清",
            "不明确",
            "未显示",
            "没有看到",
            "缺少",
            "需要更多证据",
        )
        return any(term in text for term in weak_terms)

    def _action_intent_best_choice_is_broad_relative_to_competitors(
        self,
        *,
        state: AgentState,
        result: dict[str, Any],
        candidate_indices: list[int],
    ) -> bool:
        try:
            best_index = int(result.get("best_index"))
        except Exception:  # noqa: BLE001
            return False
        if best_index < 0 or best_index >= len(getattr(state, "choices", [])):
            return False
        categories_by_index = selected_choice_categories(
            [str(choice) for choice in getattr(state, "choices", [])],
            candidate_indices if candidate_indices else None,
        )
        best_categories = set(categories_by_index.get(best_index) or set())
        broad_categories = {"generic_relocation", "access_retrieve", "space_clear", "inspect_check", "open_close"}
        if not (best_categories & broad_categories):
            return False
        other_categories = set()
        for index, categories in categories_by_index.items():
            if index == best_index:
                continue
            other_categories.update(categories)
        best_choice = str(state.choices[best_index]).strip().lower()
        broad_choice_markers = (
            "to move.",
            "to make space",
            "to access",
            "access what's behind",
            "to check",
            "to inspect",
            "to open",
            "to close",
        )
        return bool((other_categories - best_categories) or any(marker in best_choice for marker in broad_choice_markers))

    def _action_intent_direct_evidence_still_needs_resolution(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
        candidate_indices: list[int] | None = None,
    ) -> bool:
        if not self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        if not isinstance(result, dict):
            return False
        indices = candidate_indices or self._latest_action_intent_candidate_indices(state, result=result)
        if len(indices) < 2:
            return False
        return self._action_intent_best_choice_is_broad_relative_to_competitors(
            state=state,
            result=result,
            candidate_indices=indices,
        )

    def _action_intent_result_needs_generalized_disambiguation(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return False
        try:
            best_index = int(result.get("best_index"))
        except Exception:  # noqa: BLE001
            return False
        if best_index < 0 or best_index >= len(getattr(state, "choices", [])):
            return False
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
        )
        if not bool(profile["post_action_sensitive"]):
            return False
        if self._action_intent_result_has_direct_post_action_evidence(result) and not self._action_intent_direct_evidence_still_needs_resolution(
            state=state,
            result=result,
            candidate_indices=candidate_indices,
        ):
            return False
        active_categories = set(profile["active_categories"])
        has_multi_candidate_conflict = (
            len(candidate_indices) >= 2
            or bool(profile["has_pairwise_outcome_conflict"])
            or bool(profile["has_future_use_conflict"])
            or len(active_categories) >= 2
        )
        if not has_multi_candidate_conflict:
            return False
        if self._action_intent_result_has_indecisive_post_action_support(result):
            return True
        if self._action_intent_best_choice_is_broad_relative_to_competitors(
            state=state,
            result=result,
            candidate_indices=candidate_indices,
        ):
            return True
        try:
            confidence = float(result.get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        if (bool(profile["has_pairwise_outcome_conflict"]) or bool(profile["has_future_use_conflict"])) and confidence < 0.97:
            return True
        return len(active_categories) >= 2 and confidence < 0.94

    def _build_action_intent_followup_sampling_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> PlannerDecision | None:
        combined_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        if not combined_times:
            return None
        _, focus, semantic_window_s, semantic_resolver = self._action_intent_followup_route(state=state, result=None)
        window_s = semantic_window_s
        dense_near_followup = self._action_intent_prefers_dense_near_followup(state)
        if semantic_resolver == "future_use":
            window_s = max(window_s, 8.0)
        elif semantic_resolver == "pairwise":
            window_s = max(window_s, 4.0)
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
        if self._action_intent_needs_future_use_evidence(state=state, result=None) and not dense_near_followup:
            window_s = max(8.0, window_s)
        start_time = max(combined_times)
        end_time = start_time + window_s
        question_text = str(getattr(state, "question", "") or "").lower()
        if any(token in question_text for token in ("<tap kitchen scale>", "tap kitchen scale")):
            end_time = max(end_time, start_time + 8.0)
        if dense_near_followup:
            end_time = min(end_time, start_time + 3.0)
        sample_count = 6 if dense_near_followup else 4
        if self._action_intent_prefers_result_driven_followup(state):
            sample_count = max(sample_count, 5)
        return PlannerDecision(
            thought=(
                (
                    "why 题当前仍存在意图歧义，先补动作后紧邻的高密度结果帧，检查物体是否立刻被用于擦手/擦台面，"
                    "还是只是短暂拿起后被放到别处。"
                    if dense_near_followup
                    else "why 题当前仍存在意图歧义，补动作后的结果帧，检查后续是继续取后方物体，还是只是腾空间/整理。"
                )
                + (f" followup_focus={focus}" if focus else "")
            ),
            tool="sample_sparse_frames",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "sample_count": sample_count,
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
            indices=candidate_indices if candidate_indices else None,
        ):
            return True
        # Full-set fallback is intentionally limited to clean/dry style options.
        # Safety/hazard choices that are not in the current top candidates should
        # not globally force precontext sampling, or they will swamp access/space
        # and other pairwise routes whenever a distractor "avoid heat/spill" option
        # exists somewhere in the full candidate list.
        full_profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=None,
        )
        if "clean_dry" not in set(full_profile["active_categories"]):
            return False
        # Precondition-dependent options such as "dry hands" can be missed by the
        # first-pass top-2 guess, so also scan the full candidate set for that
        # narrower class.
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
            if not isinstance(entry, dict) or entry.get("tool") not in {"sample_sparse_frames", "extract_frames_for_range"}:
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
        action_end = max(combined_times)
        start_time = self._latest_action_intent_followup_end_time(state)
        if start_time is None:
            start_time = action_end
        window_s = max(4.0, min(10.0, float(window_s)))
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        dense_near_followup = self._action_intent_prefers_dense_near_followup(state)
        result_driven_followup = self._action_intent_prefers_result_driven_followup(state)
        if needed_profile["prefer_mixed_horizon"]:
            start_time = max(0.0, max(action_end - 0.15, start_time - 0.75))
            window_s = max(window_s, 7.0)
        elif needed_profile["prefer_reveal_access"]:
            start_time = max(0.0, max(action_end - 0.18, start_time - 0.8))
            window_s = max(window_s, 4.6)
        if needed_profile["prefer_state_change_only"]:
            start_time = max(0.0, max(action_end - 0.2, start_time - 1.0))
            window_s = min(max(window_s, 4.0), 4.8)
        elif needed_profile["prefer_final_placement"] or needed_profile["prefer_future_use_outcome"]:
            window_s = max(window_s, 8.8 if attempt_count <= 1 else 8.5)
        if dense_near_followup:
            start_time = max(action_end - 0.15, start_time - 1.2)
            window_s = min(window_s, 5.5)
        if result_driven_followup:
            window_s = max(window_s, 8.5)
        sample_count = 6 if dense_near_followup and attempt_count <= 1 else 4
        if result_driven_followup:
            sample_count = max(sample_count, 5 if attempt_count <= 1 else 4)
        if (
            needed_profile["prefer_state_change_only"]
            or needed_profile["prefer_mixed_horizon"]
            or needed_profile["prefer_reveal_access"]
        ):
            sample_count = max(sample_count, 6)
        elif needed_profile["prefer_final_placement"] or needed_profile["prefer_future_use_outcome"]:
            sample_count = max(sample_count, 5)
        return PlannerDecision(
            thought=(
                "why 题专用裁决仍报告证据不足，继续向后补帧，检查动作后的最终放置、使用或取回结果。"
                + (f" followup_focus={focus}" if focus else "")
            ),
            tool="sample_sparse_frames",
            args={
                "start_time": start_time,
                "end_time": start_time + window_s,
                "sample_count": sample_count,
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

    def _action_intent_peak_probe_count(self, state: AgentState) -> int:
        count = 0
        for entry in getattr(state, "tool_trace", []):
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool") or "")
            if tool_name not in {"detect_audio_peaks", "sample_frames_around_peaks"}:
                continue
            args = entry.get("args") or {}
            if not isinstance(args, dict):
                continue
            tag = str(args.get("tag") or "")
            if tool_name == "detect_audio_peaks" or tag.startswith("fine_grained_why_recognition_followup_peaks"):
                count += 1
        return count

    def _action_intent_transition_probe_window(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> tuple[float, float, float, int] | None:
        combined_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    combined_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        if not combined_times:
            return None
        action_end = max(combined_times)
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
        )
        mode = self._action_intent_transition_probe_mode(
            state=state,
            result=result,
            profile=profile,
        )
        if mode == "state_change":
            start_time = max(0.0, action_end - 0.1)
            end_time = action_end + 3.0
            stride_s = 0.35
            max_frames = 6
        elif mode == "mixed_temporal_horizon":
            start_time = max(0.0, action_end - 0.12)
            end_time = action_end + 5.2
            stride_s = 0.5
            max_frames = 8
        elif mode == "hand_free_next_action":
            start_time = max(0.0, action_end - 0.15)
            end_time = action_end + 3.6
            stride_s = 0.4
            max_frames = 6
        elif mode == "reveal_or_access_result":
            start_time = max(0.0, action_end - 0.15)
            end_time = action_end + 3.2
            stride_s = 0.4
            max_frames = 6
        elif mode == "safety_or_spill_result":
            start_time = max(0.0, action_end - 0.08)
            end_time = action_end + 2.8
            stride_s = 0.4
            max_frames = 6
        elif mode == "final_placement_result":
            start_time = action_end + 0.2
            end_time = action_end + 4.0
            stride_s = 0.4
            max_frames = 6
        elif mode == "future_use_outcome":
            start_time = action_end + 0.1
            end_time = action_end + 4.3
            stride_s = 0.4
            max_frames = 6
        else:
            start_time = max(0.0, action_end - 0.2)
            end_time = action_end + 2.4
            stride_s = 0.4
            max_frames = 6
        return start_time, end_time, stride_s, max_frames

    def _action_intent_transition_probe_mode(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        profile: dict[str, Any],
    ) -> str:
        needed_profile = self._action_intent_needed_observation_profile(state=state, result=result)
        if self._action_intent_prefers_followup_state_change_only(state):
            return "state_change"
        if needed_profile["prefer_mixed_horizon"] or self._action_intent_pair_spans_immediate_and_later_outcomes(state=state, result=result):
            return "mixed_temporal_horizon"
        if needed_profile["prefer_hand_free_next_action"] or self._action_intent_has_hand_free_future_use_conflict(state=state, result=result):
            return "hand_free_next_action"
        timeline_text = self._action_intent_timeline_review_text(state)
        support_text = self._action_intent_result_support_text(result)
        combined_text = f"{timeline_text} {support_text}".lower()
        active_categories = set(profile["active_categories"])
        if (
            "safety_avoid" in active_categories
            and any(
                marker in combined_text
                for marker in (
                    "burn",
                    "burning",
                    "hot stove",
                    "burner",
                    "hob",
                    "heat",
                    "spill",
                    "spill risk",
                    "烫",
                    "热源",
                    "灶台",
                )
            )
        ):
            return "safety_or_spill_result"
        if bool(profile["has_hidden_access_exact_use_conflict"]):
            return "reveal_or_access_result"
        final_placement_markers = (
            "put back",
            "return",
            "returned",
            "store",
            "stored",
            "placed back",
            "placed into",
            "placed on the shelf",
            "back in the fridge",
            "final location",
            "proper place",
            "right place",
            "slot",
            "放回",
            "归位",
            "收起",
            "放进冰箱",
            "最终位置",
        )
        future_use_markers = (
            "weigh",
            "scale",
            "measure",
            "pour",
            "poured",
            "empty",
            "drain",
            "serve",
            "plate",
            "check",
            "inspect",
            "wash",
            "rinse",
            "dry",
            "later use",
            "next use",
            "used next",
            "后续用途",
            "称",
            "倒",
            "沥",
            "盛",
            "检查",
            "清洗",
        )
        if any(marker in combined_text for marker in final_placement_markers) and active_categories & {
            "final_place_return",
            "generic_relocation",
        }:
            return "final_placement_result"
        if self._action_intent_needs_future_use_evidence(state=state, result=result):
            if any(marker in combined_text for marker in final_placement_markers) and active_categories & {
                "final_place_return",
                "generic_relocation",
            }:
                return "final_placement_result"
            if any(marker in combined_text for marker in future_use_markers) or active_categories & {
                "measure_weigh",
                "transfer_contents",
                "serve_consume",
                "inspect_check",
                "clean_dry",
                "open_close",
                "food_prep",
                "discard",
            }:
                return "future_use_outcome"
        return "immediate_result"

    def _action_intent_pair_spans_immediate_and_later_outcomes(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
    ) -> bool:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return False
        best_index = self._coerce_choice_index(result.get("best_index"), state.choices)
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        if competitor_index is None:
            for index in self._latest_action_intent_candidate_indices(state, result=result):
                if best_index is not None and index != best_index:
                    competitor_index = index
                    break
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return False
        categories = selected_choice_categories(
            [str(choice) for choice in getattr(state, "choices", [])],
            [best_index, competitor_index],
        )
        best_categories = set(categories.get(best_index) or set())
        competitor_categories = set(categories.get(competitor_index) or set())
        later_outcome_categories = {
            "final_place_return",
            "measure_weigh",
            "transfer_contents",
            "serve_consume",
            "clean_dry",
            "food_prep",
            "discard",
        }
        best_choice = str(getattr(state, "choices", [])[best_index] if 0 <= best_index < len(getattr(state, "choices", [])) else "")
        competitor_choice = str(
            getattr(state, "choices", [])[competitor_index] if 0 <= competitor_index < len(getattr(state, "choices", [])) else ""
        )
        best_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories)
        competitor_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(
            competitor_choice,
            competitor_categories,
        )
        return bool(
            (best_is_immediate and competitor_categories & later_outcome_categories)
            or (competitor_is_immediate and best_categories & later_outcome_categories)
        )

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
        if "open_close" in categories and "hand_free_enablement" not in categories and any(
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

    def _action_intent_has_hand_free_future_use_conflict(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        try:
            confidence = float((result or {}).get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        reason_text = str((result or {}).get("reason") or "")
        indices = self._latest_action_intent_candidate_indices(state, result=result)
        semantic_indices = self._action_intent_route_candidate_indices(
            state=state,
            result=result,
            candidate_indices=indices,
            confidence=confidence,
            reason_text=reason_text,
        )
        if not semantic_indices or len(semantic_indices) < 2:
            return False
        profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=semantic_indices,
        )
        categories = set(profile["active_categories"])
        if "hand_free_enablement" not in categories:
            return False
        return bool(categories & {"access_retrieve", "final_place_return", "generic_relocation", "open_close"})

    def _action_intent_should_try_transition_probe(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if self._action_intent_has_transition_followup_frames(state):
            return False
        if self._action_intent_followup_attempt_count(state) < 1:
            return False
        if self._action_intent_transition_probe_window(state=state, hints=hints, result=result) is None:
            return False
        needed_profile = self._action_intent_needed_observation_profile(state=state, result=result)
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
        )
        hand_free_future_use = self._action_intent_has_hand_free_future_use_conflict(state=state, result=result)
        mixed_temporal_horizon = self._action_intent_pair_spans_immediate_and_later_outcomes(
            state=state,
            result=result,
        )
        support_text = self._action_intent_result_support_text(result)
        uncertainty_markers = (
            "still unclear",
            "unclear",
            "not visible",
            "not shown",
            "cannot tell",
            "can't tell",
            "缺少",
            "看不清",
            "不明确",
        )
        if (
            mixed_temporal_horizon
            and any(marker in support_text for marker in uncertainty_markers)
            and isinstance(result, dict)
            and (
                bool(result.get("need_future_evidence"))
                or bool(result.get("ambiguity"))
                or bool(result.get("need_more_evidence"))
            )
        ):
            return True
        if (
            isinstance(result, dict)
            and (
                bool(result.get("need_future_evidence"))
                or bool(result.get("ambiguity"))
                or bool(result.get("need_more_evidence"))
            )
            and (
                needed_profile["prefer_mixed_horizon"]
                or needed_profile["prefer_hand_free_next_action"]
            )
        ):
            return True
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        if bool(profile["has_hidden_access_exact_use_conflict"]) and not hand_free_future_use:
            return False
        if self._action_intent_needs_future_use_evidence(state=state, result=result) and not hand_free_future_use:
            if isinstance(result, dict) and "decisive_observation" in result:
                decisive_observation = str(result.get("decisive_observation") or "").strip()
                if not decisive_observation:
                    score_gap = self._action_intent_future_use_score_gap(result)
                    if score_gap < 0.18 or any(marker in support_text for marker in uncertainty_markers):
                        return True
            return False
        immediate_transition_markers = (
            "moved aside",
            "becomes visible",
            "becomes reachable",
            "reveals the area",
            "revealed area",
            "freed slot",
            "腾出",
            "露出",
            "显露",
        )
        hand_free_transition_markers = (
            "left hand",
            "right hand",
            "other hand",
            "free hand",
            "freed hand",
            "free the",
            "freed the",
            "腾出",
            "另一只手",
            "左手",
            "右手",
            "去拿",
            "去开",
            "reach",
            "pick up",
            "turn on",
            "open",
            "close",
            "use the right hand",
            "use the left hand",
        )
        if hand_free_future_use and any(marker in support_text for marker in uncertainty_markers + hand_free_transition_markers):
            return True
        if self._action_intent_pair_needs_outcome_resolution(state=state, result=result) and any(
            marker in support_text for marker in immediate_transition_markers
        ) and any(marker in support_text for marker in uncertainty_markers):
            return True
        if isinstance(result, dict) and self._action_intent_result_is_weak_generic_claim(state=state, result=result):
            return True
        return False

    def _build_action_intent_transition_probe_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
        thought: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_should_try_transition_probe(state=state, hints=hints, result=result):
            return None
        probe_window = self._action_intent_transition_probe_window(state=state, hints=hints, result=result)
        if probe_window is None:
            return None
        start_time, end_time, stride_s, max_frames = probe_window
        return PlannerDecision(
            thought=thought,
            tool="extract_frames_for_range",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "stride_s": stride_s,
                "max_frames": max_frames,
                "tag": f"{state.task_family}_followup_transition",
            },
        )

    def _build_action_intent_peak_probe_after_transition_decision(
        self,
        *,
        state: AgentState,
        last_tool: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state):
            return None
        if self._action_intent_peak_probe_count(state) >= 1:
            return None
        if not self._action_intent_has_hand_free_future_use_conflict(state=state, result=result):
            return None
        args = last_tool.get("args") or {}
        if not isinstance(args, dict):
            return None
        tag = str(args.get("tag") or "")
        if not tag.endswith("_followup_transition"):
            return None
        try:
            start_time = float(args.get("start_time"))
            end_time = float(args.get("end_time"))
        except Exception:  # noqa: BLE001
            return None
        if end_time <= start_time + 0.2:
            return None
        return PlannerDecision(
            thought="why 题的 hand-free/next-use 冲突已进入短窗口密采样；先检测这段里更可能对应拿起、开关、放下或接触的音频峰值，再围绕峰值细看另一只手的下一步操作。",
            tool="detect_audio_peaks",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "window_s": 0.35,
                "top_k": 3,
            },
        )

    def _action_intent_timeline_review_tag(self, last_tool: dict[str, Any]) -> str:
        if not isinstance(last_tool, dict):
            return ""
        args = last_tool.get("args") or {}
        if not isinstance(args, dict):
            return ""
        return str(args.get("tag") or args.get("tag_hint") or "").strip().lower()

    def _action_intent_timeline_review_candidate_paths(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> list[str]:
        frames = self._select_action_intent_frames(
            state,
            hints,
            limit=8,
            require_current_scope=True,
        )
        if not frames:
            return []
        names = [Path(path).name.lower() for path in frames]
        if not any("_followup_transition_" in name or "_followup_peaks_" in name for name in names):
            return []
        return frames

    def _action_intent_should_run_timeline_review(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        last_tool: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        tool_name = str(last_tool.get("tool") or "")
        if tool_name not in {"extract_frames_for_range", "sample_frames_around_peaks", "retrieve_cached_artifacts"}:
            return False
        tag = self._action_intent_timeline_review_tag(last_tool)
        if tool_name == "extract_frames_for_range" and not tag.endswith("_followup_transition"):
            return False
        if tool_name == "sample_frames_around_peaks" and not tag.endswith("_followup_peaks"):
            return False
        if tool_name == "retrieve_cached_artifacts" and not any(
            marker in tag for marker in ("followup_transition", "followup_peaks")
        ):
            return False
        if not self._action_intent_timeline_review_candidate_paths(state=state, hints=hints):
            return False
        latest_intent = result if isinstance(result, dict) and result.get("best_index") is not None else self._latest_successful_action_intent_result(state)
        if not isinstance(latest_intent, dict) or latest_intent.get("best_index") is None:
            return False
        if self._action_intent_has_hand_free_future_use_conflict(state=state, result=latest_intent):
            return True
        if self._action_intent_needs_future_use_evidence(state=state, result=latest_intent):
            return True
        if self._action_intent_pair_needs_outcome_resolution(state=state, result=latest_intent):
            return True
        if self._action_intent_result_has_indecisive_post_action_support(latest_intent):
            return True
        return self._action_intent_result_is_weak_generic_claim(state=state, result=latest_intent)

    def _action_intent_timeline_review_prompt(self, *, state: AgentState) -> str:
        choices = "\n".join(f"{index}. {choice}" for index, choice in enumerate(getattr(state, "choices", []) or []))
        return (
            "你在做厨房视频 why 题的短时序证据复核。"
            "这些图片按时间顺序排列，覆盖当前动作、动作尾部以及动作后紧接着发生的几步。"
            "\n你不能直接选择答案，也不能用常识脑补看不见的内容。"
            "\n你的任务是只基于图片，保守总结以下几点："
            "\n1. 当前动作结束后立刻出现了什么结果。"
            "\n2. 下一步手/物体最明显的去向或操作是什么。"
            "\n3. 证据更像是在腾出手、腾出空间/显露目标、还是为了当前物体本身的立即使用/放回。"
            "\n4. 如果目前仍有多个选项都说得通，必须明确指出还缺什么证据，不能假装已经确定。"
            f"\n题目：{state.question}"
            f"\n候选：\n{choices}"
            '\n输出 JSON，字段固定为 {"timeline_summary":"","immediate_result":"","next_action_hint":"","direct_purpose_hint":"","access_or_reveal_evidence":"","hand_free_enablement_evidence":"","next_use_evidence":"","target_object":"","target_location":"","ongoing_action":"","state_change_hint":"","ambiguity_note":"","needs_more_evidence":false,"confidence":0.0}。'
        )

    def _build_action_intent_timeline_review_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        last_tool: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> PlannerDecision | None:
        if not self._action_intent_should_run_timeline_review(
            state=state,
            hints=hints,
            last_tool=last_tool,
            result=result,
        ):
            return None
        image_paths = self._action_intent_timeline_review_candidate_paths(state=state, hints=hints)
        if not image_paths:
            return None
        return PlannerDecision(
            thought="why 题已经补到 transition/peak 关键帧，但还不能只凭局部瞬间下结论；先做短时序证据复核，明确动作后立刻发生了什么、下一步手部去做什么，再回到因果判断。",
            tool="inspect_visual_evidence",
            args={
                "prompt": self._action_intent_timeline_review_prompt(state=state),
                "image_paths": image_paths,
            },
        )

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

    def _action_intent_timeline_review_needs_more_evidence(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("needs_more_evidence"):
            return True
        ambiguity = str(payload.get("ambiguity_note") or "").strip().lower()
        if ambiguity:
            return True
        direct_purpose = str(payload.get("direct_purpose_hint") or "").strip().lower()
        next_use = str(payload.get("next_use_evidence") or "").strip().lower()
        next_action = str(payload.get("next_action_hint") or "").strip().lower()
        weak_markers = (
            "unclear",
            "ambiguous",
            "not enough",
            "insufficient",
            "still could be",
            "cannot tell",
            "can't tell",
            "不明确",
            "看不清",
            "证据不足",
        )
        combined = " ".join((direct_purpose, next_use, next_action))
        return any(marker in combined for marker in weak_markers)

    def _latest_action_intent_timeline_review(self, state: AgentState) -> dict[str, Any]:
        if not self._is_action_intent_task(state):
            return {}
        for entry in reversed(getattr(state, "tool_trace", [])):
            if not isinstance(entry, dict) or str(entry.get("tool") or "") != "inspect_visual_evidence":
                continue
            raw_result = entry.get("raw_result")
            if isinstance(raw_result, dict) and self._action_intent_is_timeline_review_payload(raw_result):
                return raw_result
        return {}

    def _action_intent_timeline_review_text(self, state: AgentState) -> str:
        payload = self._latest_action_intent_timeline_review(state)
        if not payload:
            return ""
        return " ".join(
            str(payload.get(key) or "")
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
        ).strip().lower()

    def _action_intent_needed_observation_text(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(result, dict):
            text = str(result.get("needed_observation") or "").strip().lower()
            if text:
                return text
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return ""
        _tool, payload = latest
        return str(payload.get("needed_observation") or "").strip().lower()

    def _action_intent_needed_observation_profile(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        text = self._action_intent_needed_observation_text(state=state, result=result)
        if not text:
            return {
                "prefer_dense_near": False,
                "prefer_result_driven": False,
                "prefer_state_change_only": False,
                "prefer_mixed_horizon": False,
                "prefer_reveal_access": False,
                "prefer_future_use_outcome": False,
                "prefer_final_placement": False,
                "prefer_hand_free_next_action": False,
                "prefer_safety_or_spill": False,
            }
        immediate_terms = (
            "read/checked first",
            "checked first",
            "read first",
            "check first",
            "label",
            "date",
            "expiry",
            "opened",
            "turn on",
            "turn off",
            "switch",
            "direct physical effect",
            "state change",
            "display",
            "starts running",
            "stops running",
            "applied to the hands",
            "wipe",
            "dry hands",
        )
        future_use_terms = (
            "put back",
            "returned",
            "back in the fridge",
            "final placement",
            "placed on the scale",
            "put on the scale",
            "used to pour",
            "actual use",
            "used again",
            "weigh",
            "scale",
            "measure",
            "serve",
            "plate",
            "drain",
            "retrieved from behind",
            "taken from behind",
        )
        reveal_terms = (
            "behind the glass",
            "behind the area",
            "hidden item",
            "after the reveal",
            "picked up before the area is closed",
            "picked up before the door is closed",
            "object behind",
        )
        hand_free_terms = (
            "hand",
            "tap",
            "switch",
            "turn on",
            "turn off",
        )
        safety_terms = (
            "messy",
            "spill",
            "unstable",
            "full",
            "burn",
            "hot",
            "boiling",
            "counter",
        )
        state_change_terms = (
            "direct physical effect",
            "state change",
            "display",
            "turned on",
            "turned off",
            "opened",
            "starts running",
            "stops running",
            "becomes full",
            "becomes empty",
        )
        final_placement_terms = (
            "put back",
            "returned",
            "back in the fridge",
            "final placement",
            "returned to the fridge",
            "returned to the shelf",
            "placed back",
            "placed into",
        )
        immediate = any(term in text for term in immediate_terms)
        future_use = any(term in text for term in future_use_terms)
        reveal_access = any(term in text for term in reveal_terms)
        hand_free = any(term in text for term in hand_free_terms)
        safety_or_spill = any(term in text for term in safety_terms)
        state_change_only = any(term in text for term in state_change_terms)
        final_placement = any(term in text for term in final_placement_terms)
        contrastive = ("whether" in text or "是否" in text) and any(
            token in text for token in (" or ", " first ", " before ", " after ", " versus ", " vs ")
        )
        mixed_horizon = (immediate and future_use) or (contrastive and immediate and (future_use or final_placement))
        dense_near = immediate or reveal_access or hand_free or safety_or_spill or state_change_only
        return {
            "prefer_dense_near": dense_near,
            "prefer_result_driven": future_use or final_placement or mixed_horizon,
            "prefer_state_change_only": state_change_only and not (future_use or final_placement or reveal_access),
            "prefer_mixed_horizon": mixed_horizon,
            "prefer_reveal_access": reveal_access,
            "prefer_future_use_outcome": future_use and not final_placement,
            "prefer_final_placement": final_placement,
            "prefer_hand_free_next_action": hand_free and (immediate or future_use),
            "prefer_safety_or_spill": safety_or_spill,
        }

    def _action_intent_peak_probe_window(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        window_s: float = 6.0,
    ) -> tuple[float, float] | None:
        combined_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    combined_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        if not combined_times:
            return None
        action_start = min(combined_times)
        action_end = max(combined_times)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        peak_start = max(0.0, action_end - 0.25)
        peak_end = latest_followup_end if latest_followup_end is not None else action_end + max(4.0, float(window_s))
        if peak_end <= peak_start + 0.2:
            peak_end = peak_start + max(2.5, float(window_s))
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        if self._action_intent_prefers_followup_state_change_only(state):
            peak_start = max(0.0, action_end - 0.15)
            peak_end = max(peak_end, action_end + 5.5)
        elif needed_profile["prefer_mixed_horizon"]:
            peak_start = max(0.0, action_end - 0.15)
            peak_end = max(peak_end, action_end + 6.2)
        elif self._action_intent_prefers_result_driven_followup(state) and not (
            needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"]
        ):
            peak_start = max(0.0, action_start - 0.2)
            peak_end = max(peak_end, action_end + 6.0)
        return peak_start, peak_end

    def _action_intent_should_try_peak_guided_followup(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if self._action_intent_peak_probe_count(state) >= 2:
            return False
        if self._action_intent_followup_attempt_count(state) < 1:
            return False
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        if self._action_intent_prefers_result_driven_followup(state):
            return True
        if self._action_intent_result_has_indecisive_post_action_support(result):
            return True
        return self._action_intent_result_needs_generalized_disambiguation(state=state, result=result)

    def _build_action_intent_peak_guided_followup_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        last_tool: dict[str, Any],
        last_result: dict[str, Any],
        focus: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_should_try_peak_guided_followup(state=state, result=last_result):
            return None
        if str(last_tool.get("tool") or "") == "detect_audio_peaks" and isinstance(last_result, dict):
            peaks = last_result.get("peaks") or []
            peak_times = [
                float(item.get("time_s"))
                for item in peaks
                if isinstance(item, dict) and item.get("time_s") is not None
            ]
            if peak_times:
                return PlannerDecision(
                    thought="why 题当前仍缺少决定性后续证据；围绕音频峰值抽更细的关键帧，优先捕捉放下、碰撞、开关、落回或立即后续使用瞬间。"
                    + (f" followup_focus={focus}" if focus else ""),
                    tool="sample_frames_around_peaks",
                    args={
                        "peak_times": peak_times[:4],
                        "radius_s": 0.45 if self._action_intent_prefers_followup_state_change_only(state) else 0.6,
                        "frames_per_peak": 3,
                        "tag": f"{state.task_family}_followup_peaks",
                    },
                )
        probe_window = self._action_intent_peak_probe_window(
            state=state,
            hints=hints,
            window_s=float(last_result.get("future_window_s") or 6.0) if isinstance(last_result, dict) else 6.0,
        )
        if probe_window is None:
            return None
        peak_start, peak_end = probe_window
        return PlannerDecision(
            thought="why 题当前后续结果仍不清楚；先检测这段动作后窗口的音频峰值，再围绕峰值取更关键的证据帧。"
            + (f" followup_focus={focus}" if focus else ""),
            tool="detect_audio_peaks",
            args={
                "start_time": peak_start,
                "end_time": peak_end,
                "window_s": 0.35 if self._action_intent_prefers_followup_state_change_only(state) else 0.45,
                "top_k": 4,
            },
        )

    def _action_intent_prefers_dense_near_followup(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        if (
            needed_profile["prefer_dense_near"]
            or needed_profile["prefer_mixed_horizon"]
            or needed_profile["prefer_reveal_access"]
            or needed_profile["prefer_safety_or_spill"]
        ):
            return True
        question_text = str(getattr(state, "question", "") or "").lower()
        timeline_text = self._action_intent_timeline_review_text(state)
        if any(
            token in timeline_text
            for token in (
                "free hand",
                "freed hand",
                "other hand",
                "right hand",
                "left hand",
                "tap area",
                "sink area",
                "reach toward",
                "reaches toward",
                "turn on",
                "turn off",
                "open",
                "close",
                "reveal",
                "visible behind",
                "露出",
                "腾出",
                "另一只手",
                "龙头",
                "水槽",
            )
        ):
            return True
        return any(
            token in question_text
            for token in (
                "paper towel",
                "tea towel",
                "dish cloth",
                "kitchen towel",
                "napkin",
                "cloth",
                "towel",
            )
        )

    def _action_intent_prefers_result_driven_followup(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        if needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"]:
            return True
        question_text = str(getattr(state, "question", "") or "").lower()
        timeline_text = self._action_intent_timeline_review_text(state)
        if any(
            token in timeline_text
            for token in (
                "next use",
                "used next",
                "use again",
                "put back",
                "return",
                "returned",
                "placed on",
                "placed into",
                "pick up",
                "poured",
                "pour",
                "weigh",
                "scale",
                "later use",
                "后续用途",
                "放回",
                "放到",
                "称",
            )
        ):
            return True
        towel_like = any(
            token in question_text
            for token in ("paper towel", "tea towel", "dish cloth", "cloth", "towel", "napkin", "hand towel")
        )
        if towel_like and any(
            token in question_text
            for token in ("<pick up ", "<grab ", "<lift ", "<move ", "<shift ", "<flip ", "<turn ", "<shake ")
        ):
            return True
        return any(token in question_text for token in ("<tap kitchen scale>", "tap kitchen scale"))

    def _action_intent_prefers_followup_state_change_only(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        question_text = str(getattr(state, "question", "") or "").lower()
        timeline_text = self._action_intent_timeline_review_text(state)
        if any(
            token in timeline_text
            for token in (
                "display",
                "changes to",
                "becomes full",
                "becomes empty",
                "turned on",
                "turned off",
                "switches",
                "fills up",
                "starts running",
                "stops running",
                "显示",
                "变成",
                "装满",
                "空了",
                "打开",
                "关闭",
            )
        ):
            return True
        return any(
            token in question_text
            for token in (
                "<tap kitchen scale>",
                "tap kitchen scale",
            )
        )

    def _action_intent_prefers_specialized_open_question_recovery(self, state: AgentState) -> bool:
        return self._action_intent_prefers_followup_state_change_only(state)

    def _action_intent_initial_followup_budget(self, state: AgentState) -> int:
        return 2 if self._action_intent_prefers_result_driven_followup(state) else 1

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
            "board",
            "tray",
            "cleaning",
            "clean",
            "washed",
            "wash",
            "rinsed",
            "sink",
            "dirty",
            "dirty end",
            "dirty-end",
            "mess",
            "messy",
            "mess-avoidance",
            "kept over",
            "over the board",
            "over the tray",
            "spill",
            "spilling",
            "spill-risk",
            "prevent spilling",
            "unstable",
            "full",
            "liquid",
            "soup",
            "hot",
            "burn",
            "burnt",
            "avoid mess",
            "avoid spill",
            "avoid burn",
            "擦手",
            "干手",
            "湿手",
            "擦台面",
            "台面",
            "清洁",
            "清洗",
            "水槽",
            "弄脏",
            "溢出",
            "太烫",
            "烧焦",
            "防止",
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

    def _build_action_intent_resolution_transition_recovery_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        tool_name: str,
        result: dict[str, Any],
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state):
            return None
        if not (
            self._action_intent_resolution_needs_more_evidence(tool_name=tool_name, result=result)
            or self._action_intent_result_is_weak_generic_claim(state=state, result=result)
        ):
            return None
        if tool_name == "resolve_action_intent_future_use":
            thought = (
                "why 题后续用途专用裁决仍缺决定性动作后证据；先围绕动作尾部后的短窗口主动补关键帧，"
                "确认是否真的出现称重、倒空、检查、放回或具体下游使用。"
            )
        else:
            thought = (
                "why 题二选一后果裁决仍缺决定性结果证据；先围绕动作尾部后的短窗口主动补关键帧，"
                "确认是否真的出现取后方物体、腾空间后的下一步、最终归位或直接物理效果。"
            )
        needed = str(result.get("needed_observation") or "").strip()
        if needed:
            thought = f"{thought} needed_observation={needed}"
        transition_probe = self._build_action_intent_transition_probe_decision(
            state=state,
            hints=hints,
            result=result,
            thought=thought,
        )
        if transition_probe is not None:
            return transition_probe
        if self._action_intent_has_transition_followup_frames(state):
            return None
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return None
        probe_window = self._action_intent_transition_probe_window(state=state, hints=hints, result=result)
        if probe_window is None:
            return None
        start_time, end_time, stride_s, max_frames = probe_window
        return PlannerDecision(
            thought=thought,
            tool="extract_frames_for_range",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "stride_s": stride_s,
                "max_frames": max_frames,
                "tag": f"{state.task_family}_followup_transition",
            },
        )

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
        if not indices and isinstance(latest_raw, dict):
            for value in latest_raw.get("candidate_indices") or []:
                try:
                    index = int(value)
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= index < len(state.choices) and index not in indices:
                    indices.append(index)
            for key in ("best_index", "second_best_index"):
                try:
                    index = int(latest_raw.get(key))
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= index < len(state.choices) and index not in indices:
                    indices.append(index)
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
        if self._action_intent_result_has_direct_post_action_evidence(result) and not self._action_intent_direct_evidence_still_needs_resolution(
            state=state,
            result=result,
            candidate_indices=candidate_indices,
        ):
            return False
        indices = candidate_indices or self._latest_action_intent_candidate_indices(state, result=result)
        needs_followup, _, _, resolver = self._action_intent_followup_route(
            state=state,
            result=result,
            candidate_indices=indices,
        )
        if needs_followup and resolver == "pairwise":
            return True
        if needs_followup and resolver == "future_use":
            return False
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
        if self._action_intent_result_has_direct_post_action_evidence(result) and not self._action_intent_direct_evidence_still_needs_resolution(
            state=state,
            result=result,
        ):
            return False
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        needs_followup, _, _, resolver = self._action_intent_followup_route(
            state=state,
            result=result,
            candidate_indices=candidate_indices,
        )
        if needs_followup and resolver == "future_use":
            return True
        if needs_followup and resolver == "pairwise":
            return False
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
        candidate_indices = self._action_intent_pairwise_candidate_indices(state=state, result=result)
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
        if self._action_intent_pairwise_requires_extended_followup(
            state=state,
            hints=hints,
            result=result,
            candidate_indices=candidate_indices,
        ):
            extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                state=state,
                hints=hints,
                focus="hidden_access_pairwise_outcome_resolution",
                window_s=8.0,
            )
            if extra_followup is not None:
                return extra_followup
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

    def _action_intent_pairwise_candidate_indices(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> list[int]:
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        indices = self._latest_action_intent_candidate_indices(state, result=result)
        if len(indices) < 2:
            return indices
        timeline_text = self._action_intent_timeline_review_text(state)
        if not timeline_text:
            return indices
        categories_by_index = selected_choice_categories(choices, indices)
        prioritized_categories: set[str] = set()
        if any(
            token in timeline_text
            for token in (
                "free hand",
                "freed hand",
                "other hand",
                "right hand",
                "left hand",
                "reach toward",
                "tap area",
                "sink area",
                "turn on",
                "turn off",
                "open",
                "close",
                "露出",
                "腾出",
                "另一只手",
                "龙头",
                "水槽",
            )
        ):
            prioritized_categories.update(
                {
                    "hand_free_enablement",
                    "access_retrieve",
                    "space_clear",
                    "final_place_return",
                    "generic_relocation",
                    "open_close",
                }
            )
        if any(
            token in timeline_text
            for token in (
                "reveal",
                "visible behind",
                "behind",
                "slot",
                "put back",
                "return",
                "returned",
                "make space",
                "clear space",
                "right place",
                "露出",
                "后面",
                "空位",
                "放回",
            )
        ):
            prioritized_categories.update(
                {
                    "access_retrieve",
                    "space_clear",
                    "final_place_return",
                    "generic_relocation",
                }
            )
        if not prioritized_categories:
            return indices
        prioritized = [
            index
            for index in indices
            if categories_by_index.get(index, set()) & prioritized_categories
        ]
        if len(prioritized) >= 2:
            return prioritized[:2]
        safety_pair = self._action_intent_pairwise_safety_space_candidate_indices(
            state=state,
            result=result,
            indices=indices,
        )
        if safety_pair:
            return safety_pair
        return indices

    def _action_intent_pairwise_safety_space_candidate_indices(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        indices: list[int],
    ) -> list[int]:
        if len(indices) < 2:
            return []
        question = str(getattr(state, "question", "") or "").lower()
        if not any(token in question for token in ("move ", "pick up ", "shift ", "remove ", "lift ", "transfer ")):
            return []
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        full_categories = selected_choice_categories(choices, range(len(choices)))
        current_categories = selected_choice_categories(choices, indices)
        current_union = set().union(*(current_categories.get(index, set()) for index in indices))
        if "safety_avoid" in current_union:
            return []
        if not current_union:
            return []
        allowed_current = {"space_clear", "final_place_return", "generic_relocation", "food_prep"}
        if not current_union.issubset(allowed_current):
            return []
        safety_candidates = [
            index
            for index, categories in full_categories.items()
            if "safety_avoid" in categories and index not in indices
        ]
        if not safety_candidates:
            return []
        if not any(
            token in " ".join(choices[index].lower() for index in safety_candidates)
            for token in ("burn", "burning", "hot", "stove", "hob", "spill", "烫", "烧焦")
        ):
            return []
        timeline_text = self._action_intent_timeline_review_text(state)
        reason_text = " ".join(
            str((result or {}).get(key) or "")
            for key in ("reason", "followup_focus")
        )
        combined_context = f"{reason_text} {timeline_text}".lower()
        has_workspace_signal = any(
            token in combined_context
            for token in (
                "crowded counter",
                "prep area",
                "chopping",
                "prepping",
                "workspace",
                "counter",
                "make space",
                "worktop",
                "crowded",
                "备料",
                "台面",
                "切菜",
                "腾空间",
            )
        )
        has_safety_signal = any(
            token in combined_context
            for token in (
                "hot stove",
                "kitchen stove",
                "burner",
                "hob",
                "heat",
                "hot surface",
                "too close to the stove",
                "avoid burn",
                "avoid burning",
                "burn risk",
                "spill risk",
                "烫",
                "热源",
                "灶台",
                "火边",
                "烧焦",
            )
        )
        if not has_workspace_signal and not has_safety_signal:
            return []
        best_index = indices[0]
        safety_candidates.sort(key=lambda index: index)
        return [best_index, safety_candidates[0]]

    def _action_intent_pairwise_requires_extended_followup(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
        candidate_indices: list[int] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        indices = candidate_indices or self._latest_action_intent_candidate_indices(state, result=result)
        if len(indices) < 2:
            return False
        profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=indices,
        )
        if not bool(profile["has_hidden_access_exact_use_conflict"]):
            return False
        if self._action_intent_has_peak_guided_followup_frames(state):
            return False
        attempt_count = self._action_intent_followup_attempt_count(state)
        if attempt_count < 1 or attempt_count >= 2:
            return False
        if self._action_intent_pairwise_text_has_explicit_hidden_outcome(result=result):
            return False
        combined_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    combined_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        action_end = max(combined_times) if combined_times else None
        latest_end = self._latest_action_intent_followup_end_time(state)
        if latest_end is None or action_end is None:
            return True
        return latest_end < action_end + 7.5

    def _action_intent_pairwise_text_has_explicit_hidden_outcome(
        self,
        *,
        result: dict[str, Any] | None = None,
    ) -> bool:
        text = " ".join(
            str((result or {}).get(key) or "")
            for key in ("reason", "direct_effect", "downstream_action", "needed_observation", "answer")
        ).lower()
        if not text.strip():
            return False
        explicit_target_use = any(
            token in text
            for token in (
                "picked up from behind",
                "taken from behind",
                "retrieved from behind",
                "hidden item is then picked up",
                "specific hidden target is reached",
                "small jar is taken from behind",
                "revealed target is immediately used",
                "revealed slot is immediately used",
                "placed into the freed slot",
                "put into the freed slot",
                "placed into the revealed slot",
                "right after the reveal",
                "immediately after the reveal",
                "revealed area is immediately used",
                "拿到后面的",
                "取到后面的",
                "露出的卡槽立刻被使用",
                "立刻放进腾出的空位",
            )
        )
        if not explicit_target_use:
            return False
        unresolved_or_negative = any(
            token in text
            for token in (
                "unclear",
                "cannot tell",
                "can't tell",
                "no hidden item",
                "no item behind",
                "no object is placed",
                "not visible",
                "not shown",
                "later target is unclear",
                "没有后方物体被取走",
                "没有物体被放进",
                "看不清",
                "不明确",
            )
        )
        return not unresolved_or_negative

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
        candidate_indices = self._action_intent_future_use_candidate_indices(state=state, result=result)
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

    def _action_intent_future_use_candidate_indices(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> list[int]:
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if not choices:
            return []
        latest_indices = self._latest_action_intent_candidate_indices(state, result=result)
        full_indices = list(range(len(choices)))
        base_indices = latest_indices if len(latest_indices) >= 2 else full_indices
        timeline_text = self._action_intent_timeline_review_text(state)
        if not timeline_text:
            return full_indices
        categories_by_index = selected_choice_categories(choices, base_indices)
        prioritized_categories: set[str] = set()
        if any(
            token in timeline_text
            for token in (
                "free hand",
                "freed hand",
                "other hand",
                "right hand",
                "left hand",
                "reach toward",
                "tap area",
                "sink area",
                "turn on",
                "turn off",
                "open",
                "close",
                "reveal",
                "visible behind",
                "露出",
                "腾出",
                "另一只手",
            )
        ):
            prioritized_categories.update(
                {
                    "hand_free_enablement",
                    "access_retrieve",
                    "space_clear",
                    "final_place_return",
                    "open_close",
                    "generic_relocation",
                }
            )
        if any(
            token in timeline_text
            for token in (
                "next use",
                "used next",
                "use again",
                "put back",
                "return",
                "returned",
                "placed on",
                "placed into",
                "pick up",
                "poured",
                "pour",
                "weigh",
                "scale",
                "later use",
                "后续用途",
                "放回",
                "放到",
                "称",
            )
        ):
            prioritized_categories.update(
                {
                    "measure_weigh",
                    "transfer_contents",
                    "serve_consume",
                    "inspect_check",
                    "open_close",
                    "food_prep",
                    "discard",
                    "final_place_return",
                    "access_retrieve",
                    "hand_free_enablement",
                }
            )
        if not prioritized_categories:
            return base_indices
        prioritized_indices = [
            index
            for index in base_indices
            if categories_by_index.get(index, set()) & prioritized_categories
        ]
        if len(prioritized_indices) >= 2:
            return prioritized_indices[:4]
        return base_indices

    def _build_action_intent_specialized_resolution_before_text_fallback(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state):
            return None
        if not self._select_action_intent_frames(state, hints, limit=8, require_current_scope=True):
            return None
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        question = str(getattr(state, "question", "") or "")
        if (
            action_intent_needs_precondition_context(question=question, choices=choices, indices=None)
            and not self._action_intent_has_precondition_frames(state=state, hints=hints)
        ):
            return self._build_action_intent_precondition_sampling_decision(
                state=state,
                hints=hints,
                focus="precondition_before_repeated_failure_resolution",
            )
        pairwise_candidates = self._fallback_action_intent_pairwise_candidate_indices(state)
        pairwise_ready = len(pairwise_candidates) >= 2 and action_intent_needs_pairwise_resolution(
            question=question,
            choices=choices,
            indices=pairwise_candidates,
        )
        future_use_ready = action_intent_needs_future_use_resolution(question=question, choices=choices, indices=None)
        if pairwise_ready and self._action_intent_question_prefers_pairwise_resolution(question):
            return self._build_action_intent_pairwise_resolution_decision(
                state=state,
                hints=hints,
                result={"candidate_indices": pairwise_candidates},
                thought="why 题专用视觉判断连续失败，但当前题已有足够原始帧；先走二选一后果裁决，不直接退回通用文本排序。",
            )
        if future_use_ready:
            return self._build_action_intent_future_use_resolution_decision(
                state=state,
                hints=hints,
                thought="why 题专用视觉判断连续失败，但当前题已有足够原始帧；先走后续用途专用裁决，不直接退回通用文本排序。",
            )
        if pairwise_ready:
            return self._build_action_intent_pairwise_resolution_decision(
                state=state,
                hints=hints,
                result={"candidate_indices": pairwise_candidates},
                thought="why 题专用视觉判断连续失败，但当前题已有足够原始帧；先走二选一后果裁决，不直接退回通用文本排序。",
            )
        return None

    def _action_intent_question_prefers_pairwise_resolution(self, question: str) -> bool:
        text = str(question or "").lower()
        pairwise_markers = (
            "<move ",
            "<shift ",
            "<remove ",
            "<clear ",
            "<open ",
            "<close ",
            "<put ",
            "<place ",
            "<return ",
        )
        return any(marker in text for marker in pairwise_markers)

    def _fallback_action_intent_pairwise_candidate_indices(self, state: AgentState) -> list[int]:
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        categories_by_index = selected_choice_categories(choices)
        preferred_pairs = (
            ("access_retrieve", "space_clear"),
            ("access_retrieve", "final_place_return"),
            ("space_clear", "final_place_return"),
            ("safety_avoid", "space_clear"),
            ("safety_avoid", "access_retrieve"),
        )
        for left_category, right_category in preferred_pairs:
            left_index = next((index for index, cats in categories_by_index.items() if left_category in cats), None)
            right_index = next((index for index, cats in categories_by_index.items() if right_category in cats and index != left_index), None)
            if left_index is not None and right_index is not None:
                return [left_index, right_index]
        pairwise_indices = [
            index
            for index, cats in categories_by_index.items()
            if {"access_retrieve", "space_clear", "final_place_return", "safety_avoid"} & cats
        ]
        deduped: list[int] = []
        for index in pairwise_indices:
            if index not in deduped:
                deduped.append(index)
            if len(deduped) >= 2:
                break
        return deduped

    def _action_intent_failed_tool_count(self, state: AgentState, tool_name: str) -> int:
        count = 0
        for entry in getattr(state, "tool_trace", []):
            if not isinstance(entry, dict) or entry.get("tool") != tool_name:
                continue
            raw_result = entry.get("raw_result")
            if isinstance(raw_result, dict) and raw_result.get("tool_failed"):
                count += 1
        return count

    def _action_intent_raw_context_notes(self, state: AgentState, *, limit: int) -> list[str]:
        notes: list[str] = []
        for item in getattr(state, "evidence_bundle", []):
            if not isinstance(item, str) or "type=" not in item:
                continue
            if self._is_action_intent_leaky_context_note(item):
                continue
            if item not in notes:
                notes.append(item)
        return notes[:limit]

    def _action_intent_context_notes(self, state: AgentState, *, limit: int) -> list[str]:
        scoped = self._action_intent_scoped_textual_fallback_evidence(state, limit=max(limit * 2, limit))
        notes: list[str] = []
        for item in scoped:
            if not isinstance(item, str) or self._is_action_intent_leaky_context_note(item):
                continue
            if item not in notes:
                notes.append(item)
        for item in self._action_intent_spatial_context_notes(state, limit=max(2, limit // 2)):
            if item not in notes:
                notes.append(item)
        if notes:
            return notes[:limit]
        return self._action_intent_raw_context_notes(state, limit=limit)

    def _action_intent_spatial_context_notes(self, state: AgentState, *, limit: int) -> list[str]:
        spatial = self._latest_tool_result(state, "query_spatial_context")
        if not isinstance(spatial, dict) or not spatial:
            return []
        notes: list[str] = []
        for item in spatial.get("object_tracks") or []:
            if not isinstance(item, dict):
                continue
            object_name = str(item.get("object_name") or "").strip()
            association_id = str(item.get("association_id") or "").strip()
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            note = f"spatial_context track object={object_name} association_id={association_id} time={start_time}-{end_time}".strip()
            if object_name and note not in notes:
                notes.append(note)
            if len(notes) >= limit:
                return notes[:limit]
        for item in spatial.get("object_masks") or []:
            if not isinstance(item, dict):
                continue
            fixture = str(item.get("fixture") or "").strip()
            frame_number = item.get("frame_number")
            object_name = str(item.get("object_name") or "").strip()
            note = f"spatial_context mask fixture={fixture} object={object_name} frame={frame_number}".strip()
            if (fixture or object_name) and note not in notes:
                notes.append(note)
            if len(notes) >= limit:
                return notes[:limit]
        for item in spatial.get("audio_events") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("event_type") or "").strip()
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            note = f"spatial_context audio label={label} time={start_time}-{end_time}".strip()
            if label and note not in notes:
                notes.append(note)
            if len(notes) >= limit:
                return notes[:limit]
        return notes[:limit]

    def _action_intent_spatial_probe_anchor_time(self, *, state: AgentState, hints: dict[str, Any]) -> float | None:
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        if latest_followup_end is not None:
            return float(latest_followup_end)
        combined_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    combined_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        if combined_times:
            return max(combined_times)
        return None

    def _action_intent_needs_spatial_probe(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if self._action_intent_pending_resolution_tool(state):
            return False
        if self._latest_tool_result(state, "query_spatial_context"):
            return False
        if self._action_intent_spatial_probe_anchor_time(state=state, hints=hints) is None:
            return False
        if self._action_intent_followup_attempt_count(state) < 1:
            return False
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        if isinstance(result, dict) and (bool(result.get("need_future_evidence")) or bool(result.get("ambiguity"))):
            return False
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        if self._action_intent_pair_needs_outcome_resolution(
            state=state,
            result=result,
            candidate_indices=candidate_indices,
        ):
            return False
        if self._action_intent_needs_future_use_evidence(state=state, result=result):
            return False
        if isinstance(result, dict) and self._action_intent_result_is_weak_generic_claim(state=state, result=result):
            return False
        profile = action_intent_conflict_profile(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=candidate_indices if len(candidate_indices) >= 2 else None,
        )
        active_categories = set(profile["active_categories"])
        spatial_categories = {
            "access_retrieve",
            "space_clear",
            "final_place_return",
            "hand_free_enablement",
            "open_close",
            "measure_weigh",
            "transfer_contents",
            "safety_avoid",
        }
        question_text = str(getattr(state, "question", "") or "").lower()
        spatial_question_terms = (
            "behind",
            "slot",
            "space",
            "room",
            "sink",
            "tap",
            "faucet",
            "drain",
            "scale",
            "free hand",
            "left hand",
            "right hand",
            "put back",
            "open",
            "close",
        )
        return bool(active_categories & spatial_categories) or any(term in question_text for term in spatial_question_terms)

    def _build_action_intent_spatial_probe_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
        thought: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_needs_spatial_probe(state=state, hints=hints, result=result):
            return None
        anchor_time = self._action_intent_spatial_probe_anchor_time(state=state, hints=hints)
        if anchor_time is None:
            return None
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={"time_s": anchor_time, "object_name": None, "limit": 16},
        )

    def _action_intent_scoped_textual_fallback_evidence(
        self,
        state: AgentState,
        *,
        limit: int,
    ) -> list[str]:
        anchor_times = self._action_intent_anchor_times(state)
        scoped: list[str] = []
        for item in list(getattr(state, "evidence_bundle", []) or []):
            if not isinstance(item, str):
                continue
            note = str(item).strip()
            if not note or self._is_action_intent_leaky_context_note(note):
                continue
            if note.startswith(("planner_thought=", "tool_failure tool=", "verifier=")):
                continue
            if anchor_times and "type=" in note:
                spans = self._extract_embedded_note_times(note)
                if spans:
                    window_start = min(anchor_times) - 6.0
                    window_end = max(anchor_times) + 6.0
                    if not any(not (end_time < window_start or start_time > window_end) for start_time, end_time in spans):
                        continue
            if note not in scoped:
                scoped.append(note)
        if not scoped:
            return self._action_intent_raw_context_notes(state, limit=limit)
        return scoped[:limit]

    def _action_intent_anchor_times(self, state: AgentState) -> list[float]:
        times: list[float] = []
        payload = {}
        inputs_payload = getattr(state, "inputs_payload", None)
        if callable(inputs_payload):
            raw_payload = inputs_payload()
            payload = raw_payload if isinstance(raw_payload, dict) else {}
        for key in ("times", "input_times"):
            values = payload.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                try:
                    times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
        for path in self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or [])):
            inferred = self._infer_artifact_time(path)
            if inferred is not None:
                times.append(float(inferred))
        deduped: list[float] = []
        for value in sorted(times):
            rounded = round(value, 3)
            if rounded not in deduped:
                deduped.append(rounded)
        return deduped

    def _extract_embedded_note_times(self, text: str) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        for match in re.finditer(r"time=([0-9.]+)-([0-9.]+)", str(text)):
            try:
                spans.append((float(match.group(1)), float(match.group(2))))
            except Exception:  # noqa: BLE001
                continue
        return spans

    def _infer_artifact_time(self, path: str) -> float | None:
        match = re.search(r"_([0-9]+\.[0-9]+)s\.(?:jpg|jpeg|png|webp)$", str(path), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

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

    def _action_intent_specialized_tools_used(self, used_tools: list[str]) -> bool:
        return any(
            tool in {
                "infer_action_intent",
                "resolve_action_intent_pairwise",
                "resolve_action_intent_future_use",
            }
            for tool in used_tools
        )

    def _build_initial_action_intent_specialized_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state):
            return None
        if self._action_intent_specialized_tools_used(used_tools):
            return None
        combined_times = sorted(
            [float(value) for value in hints.get("times") or []]
            + [float(value) for value in hints.get("input_times") or []]
        )
        if not combined_times:
            return None
        action_frames = self._select_action_intent_frames(
            state,
            hints,
            limit=8,
            require_current_scope=True,
        )
        if action_frames:
            return PlannerDecision(
                thought="why 题先走当前题时间窗的专用动作目的判断，不先退回 query_state/query_time，避免把同视频其它题的状态记忆混进来。",
                tool="infer_action_intent",
                args={
                    "question": state.question,
                    "choices": [str(choice) for choice in state.choices],
                    "image_paths": action_frames,
                    "context_notes": self._action_intent_context_notes(state, limit=12),
                },
            )
        return PlannerDecision(
            thought="why 题还没有当前题时间窗关键帧，先抽动作片段，再做专用动作目的判断。",
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
        evidence = self._action_intent_scoped_textual_fallback_evidence(state, limit=16)
        deduped_evidence = list(dict.fromkeys(str(item) for item in evidence if isinstance(item, str) and str(item).strip()))
        working_memory = [
            str(item)
            for item in list(getattr(state, "working_memory", []) or [])[-20:]
            if isinstance(item, str)
            and str(item).strip()
            and not str(item).startswith(("planner_thought=", "tool_failure tool=", "verifier="))
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
            question_text = str(getattr(state, "question", "") or "").lower()
            keep_precontext_without_followup = self._is_action_intent_task(state) and any(
                token in question_text
                for token in ("towel", "cloth", "paper towel", "tea towel", "dish cloth", "scale", "tap", "switch", "turn on")
            )
            precondition_window_s = 2.0
            if include_followup and self._action_intent_needs_precondition_context(state=state, result=None):
                precondition_window_s = 6.0
            elif not include_followup and (
                self._action_intent_needs_precondition_context(state=state, result=None)
                or keep_precontext_without_followup
            ):
                precondition_window_s = 4.0
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
        if self._is_action_intent_task(state):
            current_task_frames = self._action_intent_current_task_artifact_frames(frames)
            if current_task_frames:
                frames = current_task_frames
        frames = self._sort_frames_by_artifact_time(frames)
        needed_profile = self._action_intent_needed_observation_profile(state=state) if self._is_action_intent_task(state) else {}
        if (
            self._is_action_intent_task(state)
            and combined_times
            and (
                not include_followup
                or any("_precontext_" in Path(path).name.lower() for path in frames)
                or self._action_intent_prefers_followup_state_change_only(state)
                or self._action_intent_prefers_dense_near_followup(state)
                or any(bool(needed_profile.get(key)) for key in ("prefer_mixed_horizon", "prefer_reveal_access", "prefer_future_use_outcome", "prefer_final_placement"))
            )
        ):
            staged = self._stage_action_intent_frames(
                state=state,
                frames=frames,
                action_times=combined_times,
                limit=limit,
                include_followup=include_followup,
            )
            if staged:
                return staged
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

    def _action_intent_current_task_artifact_frames(self, frames: list[str]) -> list[str]:
        task_tag = "fine_grained_why_recognition"
        preferred_tags = (
            f"{task_tag}_segment",
            f"{task_tag}_precontext",
            f"{task_tag}_followup",
            f"{task_tag}_followup_transition",
            f"{task_tag}_followup_peaks",
            f"{task_tag}_followup_ext2",
            f"{task_tag}_followup_ext3",
            f"{task_tag}_followup_ext4",
            f"{task_tag}_recover_frames",
        )
        selected = [
            path
            for path in frames
            if isinstance(path, str)
            and any(tag in Path(path).name.lower() for tag in preferred_tags)
        ]
        if not selected:
            return []
        return self._sort_frames_by_artifact_time(selected)

    def _action_intent_result_is_weak_generic_claim(
        self,
        *,
        state: AgentState,
        result: dict[str, Any],
    ) -> bool:
        try:
            index = int(result.get("best_index"))
        except Exception:  # noqa: BLE001
            return False
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if index < 0 or index >= len(choices):
            return False
        choice_lc = choices[index].strip().lower()
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
            str(result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation", "answer")
        ).lower()
        if any(
            token in text
            for token in (
                "least contradicted",
                "broadest",
                "could in principle",
                "could broadly",
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
        direct_positive_terms = (
            "placed on the scale",
            "used on the scale",
            "under running water",
            "wiping motion",
            "wiped the",
            "dried the",
            "stored in",
            "returned to",
            "reveals",
            "revealed",
            "picked up from behind",
            "placed into the freed slot",
            "directly enabled",
            "明确看到",
            "直接看到",
            "放到秤上",
            "开始擦",
            "放回",
        )
        return not any(token in text for token in direct_positive_terms)

    def _stage_action_intent_frames(
        self,
        *,
        state: AgentState,
        frames: list[str],
        action_times: list[float],
        limit: int,
        include_followup: bool,
    ) -> list[str]:
        if not frames or limit <= 0:
            return []
        task_tag = str(getattr(state, "task_family", "") or "").lower()
        precontext_tag = f"{task_tag}_precontext"
        segment_tag = f"{task_tag}_segment"
        followup_transition_tag = f"{task_tag}_followup_transition"
        followup_peak_tag = f"{task_tag}_followup_peaks"
        followup_ext_tags = (
            f"{task_tag}_followup",
            f"{task_tag}_followup_ext2",
            f"{task_tag}_followup_ext3",
            f"{task_tag}_followup_ext4",
        )
        action_start = min(action_times)
        action_end = max(action_times)
        action_mid = (action_start + action_end) / 2.0
        precontext_frames: list[str] = []
        segment_frames: list[str] = []
        followup_transition_frames: list[str] = []
        followup_peak_frames: list[str] = []
        followup_ext_frames: list[str] = []
        followup_frames: list[str] = []
        unknown_frames: list[str] = []
        for path in frames:
            name = Path(path).name.lower()
            artifact_time = self._artifact_time_from_path(path)
            if precontext_tag in name:
                precontext_frames.append(path)
            elif segment_tag in name:
                segment_frames.append(path)
            elif include_followup and followup_transition_tag in name:
                followup_transition_frames.append(path)
            elif include_followup and followup_peak_tag in name:
                followup_peak_frames.append(path)
            elif include_followup and any(tag in name for tag in followup_ext_tags[1:]):
                followup_ext_frames.append(path)
            elif include_followup and followup_ext_tags[0] in name:
                followup_frames.append(path)
            elif artifact_time is None:
                unknown_frames.append(path)
            elif artifact_time < action_start - 0.1:
                precontext_frames.append(path)
            elif artifact_time <= action_end + 0.75:
                segment_frames.append(path)
            elif include_followup:
                followup_frames.append(path)

        stage_target_counts: list[tuple[str, list[str], int, float]]
        followup_only_state_change = False
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        if include_followup:
            needs_precontext = self._action_intent_needs_precondition_context(state=state, result=None)
            dense_near_followup = self._action_intent_prefers_dense_near_followup(state)
            followup_only_state_change = self._action_intent_prefers_followup_state_change_only(state)
            pre_keep = 2 if needs_precontext else 1
            segment_keep = 3
            transition_keep = 2 if followup_transition_frames and limit >= 6 else 0
            peak_keep = 1 if followup_peak_frames and limit >= 5 else 0
            ext_keep = 1 if followup_ext_frames and limit >= 6 else 0
            followup_keep = 2
            if dense_near_followup and limit >= 8:
                pre_keep = 2 if needs_precontext else 1
                segment_keep = 2
                transition_keep = min(2, len(followup_transition_frames)) if followup_transition_frames else 0
                peak_keep = min(2, len(followup_peak_frames)) if followup_peak_frames else 0
                ext_keep = 1 if followup_ext_frames else 0
                followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep - peak_keep - ext_keep)
            if followup_only_state_change and followup_frames:
                pre_keep = 0
                segment_keep = 0
                transition_keep = min(max(1, limit // 2), len(followup_transition_frames)) if followup_transition_frames else 0
                peak_keep = min(max(1, limit // 2), len(followup_peak_frames)) if followup_peak_frames else 0
                ext_keep = min(max(0, limit - transition_keep - peak_keep - 1), len(followup_ext_frames)) if followup_ext_frames else 0
                followup_keep = max(1, limit - transition_keep - peak_keep - ext_keep)
            if limit <= 4:
                if followup_only_state_change and followup_frames:
                    pre_keep = 0
                    segment_keep = 0
                    transition_keep = min(1, len(followup_transition_frames)) if followup_transition_frames else 0
                    peak_keep = min(1, len(followup_peak_frames)) if followup_peak_frames else 0
                    ext_keep = 0
                    followup_keep = max(1, limit - transition_keep - peak_keep)
                else:
                    pre_keep = 1 if needs_precontext else 0
                    segment_keep = 2
                    transition_keep = min(1, len(followup_transition_frames)) if followup_transition_frames and limit >= 4 else 0
                    peak_keep = 0
                    ext_keep = 0
                    followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep)
            elif limit == 5:
                if followup_only_state_change and followup_frames:
                    pre_keep = 0
                    segment_keep = 0
                    transition_keep = min(1, len(followup_transition_frames)) if followup_transition_frames else 0
                    peak_keep = min(2, len(followup_peak_frames)) if followup_peak_frames else 0
                    ext_keep = 0
                    followup_keep = max(1, limit - transition_keep - peak_keep)
                else:
                    pre_keep = 1 if needs_precontext else 0
                    segment_keep = 3
                    transition_keep = min(1, len(followup_transition_frames)) if followup_transition_frames else 0
                    peak_keep = min(1, len(followup_peak_frames)) if followup_peak_frames else 0
                    ext_keep = 0
                    followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep - peak_keep)
            if needed_profile["prefer_reveal_access"]:
                pre_keep = 1 if needs_precontext and limit >= 6 else 0
                segment_keep = min(segment_keep, 2)
                transition_keep = min(len(followup_transition_frames), max(transition_keep, 2 if limit >= 6 else 1))
                peak_keep = min(len(followup_peak_frames), max(peak_keep, 2 if limit >= 7 else 1))
                ext_keep = 0
                followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep - peak_keep - ext_keep)
            elif needed_profile["prefer_mixed_horizon"]:
                pre_keep = 1 if needs_precontext and limit >= 7 else 0
                segment_keep = min(segment_keep, 2)
                transition_keep = min(len(followup_transition_frames), max(transition_keep, 2 if limit >= 6 else 1))
                peak_keep = min(len(followup_peak_frames), max(peak_keep, 1))
                ext_keep = min(len(followup_ext_frames), max(ext_keep, 1 if limit >= 6 else 0))
                followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep - peak_keep - ext_keep)
            elif needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"]:
                pre_keep = 1 if needs_precontext and limit >= 7 else 0
                segment_keep = min(segment_keep, 2)
                transition_keep = min(len(followup_transition_frames), max(transition_keep, 1 if followup_transition_frames and limit >= 7 else 0))
                peak_keep = min(len(followup_peak_frames), max(peak_keep, 1 if followup_peak_frames and limit >= 7 else 0))
                ext_keep = min(len(followup_ext_frames), max(ext_keep, 2 if limit >= 6 else 1))
                followup_keep = max(2 if limit >= 6 else 1, limit - pre_keep - segment_keep - transition_keep - peak_keep - ext_keep)
            stage_target_counts = [
                ("precontext", precontext_frames, pre_keep, action_start - 0.15),
                ("segment", segment_frames, segment_keep, action_end if dense_near_followup or needed_profile["prefer_reveal_access"] else action_mid),
                ("transition", followup_transition_frames, transition_keep, action_end + 0.25),
                ("peaks", followup_peak_frames, peak_keep, action_end + (0.6 if dense_near_followup else 0.9)),
                ("ext", followup_ext_frames, ext_keep, action_end + (5.2 if (needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"] or needed_profile["prefer_mixed_horizon"]) else 3.0)),
                ("followup", followup_frames, followup_keep, action_end + 3.4 if (needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"]) else action_mid),
            ]
        else:
            question_text = str(getattr(state, "question", "") or "").lower()
            should_keep_precontext = self._action_intent_needs_precondition_context(state=state, result=None) or any(
                token in question_text
                for token in ("towel", "cloth", "paper towel", "tea towel", "dish cloth", "scale", "tap", "switch", "turn on")
            )
            pre_keep = 1 if should_keep_precontext and limit >= 4 else 0
            segment_keep = max(2, limit - pre_keep)
            stage_target_counts = [
                ("precontext", precontext_frames, pre_keep, action_start - 0.15),
                ("segment", segment_frames, segment_keep, action_end),
            ]

        if include_followup:
            if followup_only_state_change:
                priority = ["transition", "peaks", "followup", "ext", "segment", "precontext"]
            elif needed_profile["prefer_reveal_access"]:
                priority = ["transition", "peaks", "segment", "followup", "precontext", "ext"]
            elif needed_profile["prefer_mixed_horizon"]:
                priority = ["transition", "ext", "followup", "peaks", "segment", "precontext"]
            elif needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"]:
                priority = ["ext", "followup", "transition", "peaks", "segment", "precontext"]
            else:
                priority = ["precontext", "segment", "transition", "peaks", "ext", "followup"]
        else:
            priority = ["precontext", "segment"]
        priority_rank = {name: index for index, name in enumerate(priority)}
        stage_target_counts = sorted(
            stage_target_counts,
            key=lambda item: priority_rank.get(item[0], len(priority_rank)),
        )
        remaining_budget = limit
        normalized_stage_targets: list[tuple[str, list[str], int, float]] = []
        for stage_name, stage_frames, keep_count, stage_anchor in stage_target_counts:
            if remaining_budget <= 0:
                break
            keep = min(max(0, keep_count), remaining_budget)
            if keep <= 0:
                continue
            normalized_stage_targets.append((stage_name, stage_frames, keep, stage_anchor))
            remaining_budget -= keep
        stage_target_counts = normalized_stage_targets

        selected: list[str] = []
        seen: set[str] = set()
        for _stage_name, stage_frames, keep_count, stage_anchor in stage_target_counts:
            for path in self._sample_action_intent_stage_frames(
                stage_frames,
                keep_count,
                anchor_time=stage_anchor,
            ):
                if path in seen:
                    continue
                selected.append(path)
                seen.add(path)

        if include_followup and followup_only_state_change and followup_frames:
            return self._sort_frames_by_artifact_time(selected[:limit]) if selected else []

        if len(selected) < min(limit, len(frames)):
            remaining = [path for path in frames if path not in seen]
            if unknown_frames:
                remaining = [path for path in unknown_frames if path not in seen] + [
                    path for path in remaining if path not in unknown_frames
                ]
            remaining_keep = min(limit - len(selected), len(remaining))
            if include_followup and (
                needed_profile["prefer_future_use_outcome"]
                or needed_profile["prefer_final_placement"]
                or needed_profile["prefer_mixed_horizon"]
                or needed_profile["prefer_reveal_access"]
                or needed_profile["prefer_safety_or_spill"]
            ):
                remaining_anchor = action_end + 1.0
                if (
                    needed_profile["prefer_future_use_outcome"]
                    or needed_profile["prefer_final_placement"]
                    or needed_profile["prefer_mixed_horizon"]
                ):
                    remaining_anchor = action_end + 4.0
                elif needed_profile["prefer_reveal_access"] or needed_profile["prefer_safety_or_spill"]:
                    remaining_anchor = action_end + 0.35
                remaining_paths = self._sample_action_intent_stage_frames(
                    remaining,
                    remaining_keep,
                    anchor_time=remaining_anchor,
                )
            else:
                remaining_paths = self._sample_evenly_ordered(remaining, remaining_keep)
            for path in remaining_paths:
                if path in seen:
                    continue
                selected.append(path)
                seen.add(path)

        return self._sort_frames_by_artifact_time(selected[:limit]) if selected else []

    def _sample_action_intent_stage_frames(
        self,
        frames: list[str],
        limit: int,
        *,
        anchor_time: float,
    ) -> list[str]:
        ordered = self._sort_frames_by_artifact_time(frames)
        if limit <= 0 or not ordered:
            return []
        if len(ordered) <= limit:
            return ordered
        if limit == 1:
            return [self._nearest_frame_to_time(ordered, anchor_time)]
        if limit == 2:
            nearest = self._nearest_frame_to_time(ordered, anchor_time)
            selected = [ordered[0], nearest]
            deduped = []
            for path in selected:
                if path not in deduped:
                    deduped.append(path)
            if len(deduped) < 2:
                deduped.append(ordered[-1])
            return deduped[:2]
        if limit == 3:
            selected = [ordered[0], ordered[min(1, len(ordered) - 1)], ordered[-1]]
            deduped = []
            for path in selected:
                if path not in deduped:
                    deduped.append(path)
            if len(deduped) < 3:
                for path in ordered:
                    if path not in deduped:
                        deduped.append(path)
                    if len(deduped) >= 3:
                        break
            return deduped[:3]
        return self._sample_evenly_ordered(ordered, limit)

    def _nearest_frame_to_time(self, frames: list[str], target_time: float) -> str:
        best_path = frames[0]
        best_distance = float("inf")
        for path in frames:
            artifact_time = self._artifact_time_from_path(path)
            if artifact_time is None:
                continue
            distance = abs(artifact_time - target_time)
            if distance < best_distance:
                best_distance = distance
                best_path = path
        return best_path

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
        if (
            self._is_action_intent_task(state)
            and latest_verification
            and not bool(latest_verification.get("sufficient"))
            and ("need_disambiguating_evidence" in open_questions or "need_disambiguating_evidence" in verifier_missing)
        ):
            recovered = self._recover_from_open_questions(state=state, hints=hints, used_tools=used_tools)
            if recovered is not None and recovered.tool:
                self._state_add_memory(state, f"planner_override verifier_blocked_finish=finish -> {recovered.tool}")
                return recovered
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
                specialized_resolution = self._build_action_intent_specialized_resolution_before_text_fallback(
                    state=state,
                    hints=hints,
                )
                if specialized_resolution is not None:
                    return specialized_resolution
                return self._build_action_intent_text_fallback_rank_decision(
                    state,
                    thought="why 题专用视觉判断连续失败，改用结构化文本因果裁决，避免继续空转 query_time。",
                )
        if self._is_action_intent_task(state) and isinstance(last_result, dict) and last_tool.get("tool") == "detect_audio_peaks":
            peak_guided = self._build_action_intent_peak_guided_followup_decision(
                state=state,
                hints=hints,
                last_tool=last_tool,
                last_result=last_result,
                focus="audio_peak_guided_action_intent_recovery",
            )
            if peak_guided is not None:
                return peak_guided
        if (
            self._is_action_intent_task(state)
            and isinstance(last_result, dict)
            and last_tool.get("tool") == "extract_frames_for_range"
        ):
            transition_peak_probe = self._build_action_intent_peak_probe_after_transition_decision(
                state=state,
                last_tool=last_tool,
                result=self._latest_successful_action_intent_result(state),
            )
            if transition_peak_probe is not None:
                return transition_peak_probe
        if (
            self._is_action_intent_task(state)
            and isinstance(last_result, dict)
            and last_tool.get("tool") in {"extract_frames_for_range", "sample_frames_around_peaks", "retrieve_cached_artifacts"}
            and state.retrieved_frames
        ):
            timeline_review = self._build_action_intent_timeline_review_decision(
                state=state,
                hints=hints,
                last_tool=last_tool,
                result=self._latest_successful_action_intent_result(state),
            )
            if timeline_review is not None:
                return timeline_review
        if (
            self._is_action_intent_task(state)
            and isinstance(last_result, dict)
            and last_tool.get("tool") == "query_spatial_context"
            and state.retrieved_frames
        ):
            if self._action_intent_pending_resolution_tool(state) == "resolve_action_intent_future_use":
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题已补到空间上下文，回到后续用途裁决，利用对象/fixture 邻域关系判断动作后真实用途。",
                )
                if future_use is not None:
                    return future_use
            if self._action_intent_pending_resolution_tool(state) == "resolve_action_intent_pairwise":
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题已补到空间上下文，回到二选一后果裁决，利用对象/fixture 邻域关系排除竞争选项。",
                )
                if pairwise is not None:
                    return pairwise
            action_frames = self._select_action_intent_frames(
                state,
                hints,
                limit=8,
                require_current_scope=True,
            )
            if action_frames:
                return PlannerDecision(
                    thought="why 题已补到空间上下文，重新结合关键帧与对象/fixture 邻域证据判断动作目的。",
                    tool="infer_action_intent",
                    args={
                        "question": state.question,
                        "choices": [str(choice) for choice in state.choices],
                        "image_paths": action_frames,
                        "context_notes": self._action_intent_context_notes(state, limit=12),
                    },
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
        if self._is_action_intent_task(state) and not state.tool_trace:
            combined_times = sorted(
                [float(value) for value in hints.get("times") or []]
                + [float(value) for value in hints.get("input_times") or []]
            )
            question_text = str(getattr(state, "question", "") or "").lower()
            if combined_times and any(token in question_text for token in ("<tap kitchen scale>", "tap kitchen scale")):
                return PlannerDecision(
                    thought="why 题涉及电子秤按键，单帧不足以区分开机/归零；先补动作后的状态变化帧。",
                    tool="sample_sparse_frames",
                    args={
                        "start_time": max(combined_times),
                        "end_time": max(combined_times) + 8.0,
                        "sample_count": 4,
                        "tag": f"{state.task_family}_followup",
                    },
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
                if self._action_intent_followup_attempt_count(state) < self._action_intent_initial_followup_budget(state):
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题第一轮 followup 后仍缺少决定性结果；先在动作尾部和紧随其后的短窗口做更密的关键帧搜索，再决定是否进入专用裁决。",
                )
                if transition_probe is not None:
                    return transition_probe
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
                if self._action_intent_followup_attempt_count(state) < self._action_intent_initial_followup_budget(state):
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题 top-2 还停留在泛化解释层，先对动作后立刻发生的变化做密采样，避免过早进入 pairwise 收口。",
                )
                if transition_probe is not None:
                    return transition_probe
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题 top-2 仍是动作后果型歧义，不能仅凭高置信直接结束；改为结合结果帧二选一裁决。",
                )
                if pairwise is not None:
                    return pairwise
            if self._action_intent_needs_future_use_evidence(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < self._action_intent_initial_followup_budget(state):
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题的目的依赖动作后真实用途；先对动作后短窗口密采样，确认是否立刻出现称重、倒空、放回、开关或具体下游使用。",
                )
                if transition_probe is not None:
                    return transition_probe
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题目的依赖动作后用途，必须显式验证后续用途证据后才能结束。",
                )
                if future_use is not None:
                    return future_use
            if self._action_intent_result_is_weak_generic_claim(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 3:
                    extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                        state=state,
                        hints=hints,
                        focus="weak_generic_action_intent_claim_needs_direct_outcome",
                        window_s=8.0,
                    )
                    if extra_followup is not None:
                        return extra_followup
            spatial_probe = self._build_action_intent_spatial_probe_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题暂时没有更直接的时序补证路径，但当前候选仍依赖空间关系；先补对象/fixture 邻域上下文，再决定是否结束。",
            )
            if spatial_probe is not None:
                return spatial_probe
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
                if self._action_intent_is_timeline_review_payload(last_result):
                    latest_intent = self._latest_successful_action_intent_result(state)
                    if self._action_intent_timeline_review_needs_more_evidence(last_result):
                        transition_probe = self._build_action_intent_transition_probe_decision(
                            state=state,
                            hints=hints,
                            result=latest_intent,
                            thought="why 题短时序复核已经明确当前仍有多种解释；先在动作尾部和紧随其后的短窗口主动密采样关键帧，优先寻找能立刻排除竞争目的的决定性瞬间。",
                        )
                        if transition_probe is not None:
                            return transition_probe
                        if self._action_intent_followup_attempt_count(state) < 3:
                            extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                                state=state,
                                hints=hints,
                                focus="timeline_review_requested_more_evidence",
                                window_s=6.0,
                            )
                            if extra_followup is not None:
                                return PlannerDecision(
                                    thought="why 题短时序复核明确说仍有多个解释成立；继续向后补更远一点的结果帧，再决定动作真实目的。",
                                    tool=extra_followup.tool,
                                    args=extra_followup.args,
                                    done=extra_followup.done,
                                    answer=extra_followup.answer,
                                    prediction=extra_followup.prediction,
                                    confidence=extra_followup.confidence,
                                )
                    pending_tool = self._action_intent_pending_resolution_tool(state)
                    if pending_tool == "resolve_action_intent_future_use":
                        future_use = self._build_action_intent_future_use_resolution_decision(
                            state=state,
                            hints=hints,
                            result=latest_intent,
                            thought="why 题短时序复核后仍不唯一；改走后续用途专用裁决，而不是直接重做五选一判断。",
                        )
                        if future_use is not None:
                            return future_use
                    if pending_tool == "resolve_action_intent_pairwise":
                        pairwise = self._build_action_intent_pairwise_resolution_decision(
                            state=state,
                            hints=hints,
                            result=latest_intent,
                            thought="why 题短时序复核后仍不唯一；改走二选一专用裁决，而不是直接重做五选一判断。",
                        )
                        if pairwise is not None:
                            return pairwise
                    action_frames = self._select_action_intent_frames(
                        state,
                        hints,
                        limit=8,
                        include_followup=True,
                        require_current_scope=True,
                    )
                    if action_frames:
                        return PlannerDecision(
                            thought="why 题已完成短时序复核；带着动作后证据重新判断动作目的，不再退回只看当前动作片段。",
                            tool="infer_action_intent",
                            args={
                                "question": state.question,
                                "choices": [str(choice) for choice in state.choices],
                                "image_paths": action_frames,
                                "context_notes": self._action_intent_context_notes(state, limit=12),
                            },
                        )
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
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "retrieve_cached_artifacts", "sample_frames_around_peaks"}
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
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "retrieve_cached_artifacts", "sample_frames_around_peaks"}
            and self._is_action_intent_task(state)
            and state.retrieved_frames
            and self._action_intent_pending_resolution_tool(state)
        ):
            transition_peak_probe = self._build_action_intent_peak_probe_after_transition_decision(
                state=state,
                last_tool=last_tool,
                result=self._latest_successful_action_intent_result(state),
            )
            if transition_peak_probe is not None:
                return transition_peak_probe
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
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "retrieve_cached_artifacts", "sample_frames_around_peaks"}
            and self._is_action_intent_task(state)
            and state.retrieved_frames
            and self._action_intent_requires_followup(state)
            and self._action_intent_followup_attempt_count(state) <= 1
        ):
            transition_peak_probe = self._build_action_intent_peak_probe_after_transition_decision(
                state=state,
                last_tool=last_tool,
                result=self._latest_successful_action_intent_result(state),
            )
            if transition_peak_probe is not None:
                return transition_peak_probe
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
            if any(
                isinstance(item, str)
                and item.startswith("action_intent_resolution_withheld_for_missing_state_change_prereq=1")
                for item in list(getattr(state, "working_memory", []))[-12:]
            ) and self._action_intent_followup_attempt_count(state) < 3:
                extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                    state=state,
                    hints=hints,
                    focus="state_change_prereq_missing_need_more_followup",
                    window_s=6.0,
                )
                if extra_followup is not None:
                    return extra_followup
            if self._action_intent_resolution_should_backfill_precondition(
                state=state,
                hints=hints,
                result=last_result,
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(last_result.get("needed_observation") or "precondition_before_pairwise_followup"),
                )
                if precondition is not None:
                    return precondition
            transition_probe = self._build_action_intent_resolution_transition_recovery_decision(
                state=state,
                hints=hints,
                tool_name="resolve_action_intent_pairwise",
                result=last_result,
            )
            if transition_probe is not None:
                return transition_probe
            peak_guided = self._build_action_intent_peak_guided_followup_decision(
                state=state,
                hints=hints,
                last_tool=last_tool,
                last_result=last_result,
                focus=str(last_result.get("needed_observation") or "pairwise_outcome_resolution"),
            )
            if peak_guided is not None:
                return peak_guided
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
            if self._action_intent_result_is_weak_generic_claim(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 3:
                    extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                        state=state,
                        hints=hints,
                        focus="weak_generic_pairwise_claim_needs_direct_outcome",
                        window_s=6.0,
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
            if any(
                isinstance(item, str)
                and item.startswith("action_intent_resolution_withheld_for_weak_surface_wiping_evidence=1")
                for item in list(getattr(state, "working_memory", []))[-12:]
            ) and self._action_intent_followup_attempt_count(state) < 3:
                extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                    state=state,
                    hints=hints,
                    focus="weak_surface_wiping_claim_needs_stronger_post_action_evidence",
                    window_s=6.0,
                )
                if extra_followup is not None:
                    return extra_followup
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
            transition_probe = self._build_action_intent_resolution_transition_recovery_decision(
                state=state,
                hints=hints,
                tool_name="resolve_action_intent_future_use",
                result=last_result,
            )
            if transition_probe is not None:
                return transition_probe
            peak_guided = self._build_action_intent_peak_guided_followup_decision(
                state=state,
                hints=hints,
                last_tool=last_tool,
                last_result=last_result,
                focus=str(last_result.get("needed_observation") or "future_use_resolution"),
            )
            if peak_guided is not None:
                return peak_guided
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
            if self._action_intent_result_is_weak_generic_claim(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 3:
                    extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                        state=state,
                        hints=hints,
                        focus="weak_generic_future_use_claim_needs_direct_outcome",
                        window_s=8.0,
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
        initial_action_intent_route = self._build_initial_action_intent_specialized_decision(
            state=state,
            hints=hints,
            used_tools=used_tools,
        )
        if initial_action_intent_route is not None:
            return initial_action_intent_route
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
        recent_failures = [item for item in getattr(state, "tool_failures", []) if isinstance(item, dict)]
        failed_tools = {str(item.get("tool")) for item in recent_failures[-5:] if item.get("tool")}
        recent_ineffective = [item for item in getattr(state, "ineffective_tools", []) if isinstance(item, dict)]
        ineffective_tools = {str(item.get("tool")) for item in recent_ineffective[-5:] if item.get("tool")}
        if (
            self._is_action_intent_task(state)
            and combined_times
            and "need_alternative_evidence_path" in open_questions
        ):
            if self._action_intent_prefers_specialized_open_question_recovery(state):
                if self._action_intent_pending_resolution_tool(state) == "resolve_action_intent_future_use":
                    future_use = self._build_action_intent_future_use_resolution_decision(
                        state=state,
                        hints=hints,
                        thought="why 状态变化题被阻断时，优先回到后续用途专用裁决，不使用单帧 recover 路径。",
                    )
                    if future_use is not None:
                        return future_use
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    thought="why 状态变化题被阻断时，优先回到专用二选一状态变化裁决，不使用单帧 recover 路径。",
                )
                if pairwise is not None:
                    return pairwise
            raw_reuse_or_resample = self._build_raw_reuse_or_resample_decision(
                state=state,
                used_tools=used_tools,
                failed_tools=failed_tools,
                ineffective_tools=ineffective_tools,
                combined_times=combined_times,
                tag_hint=f"{state.task_family}_segment",
                sample_tag=f"{state.task_family}_recover_frames",
                sample_count=4,
                retrieve_limit=6,
                retrieve_thought="why 题文本 fallback 仍不够时，先复用当前动作片段 artifact，优先走更便宜的原始证据恢复。",
                revisit_thought="why 题文本 fallback 仍不够时，先回到已访问的动作关键时刻补单帧，而不是继续泛化时间检索。",
                resample_thought="why 题文本 fallback 仍不够且没有可复用 artifact 时，再重新稀疏抽当前动作时间窗关键帧。",
            )
            if raw_reuse_or_resample is not None:
                return raw_reuse_or_resample
        if self._is_action_intent_task(state) and (
            "need_disambiguating_evidence" in open_questions or "need_disambiguating_evidence" in verifier_missing
        ):
            targeted_action_intent_recovery = self._recover_action_intent_after_verifier_blocked_finish(
                state=state,
                hints=hints,
            )
            if targeted_action_intent_recovery is not None:
                return targeted_action_intent_recovery
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
        if self._is_action_intent_task(state) and self._action_intent_prefers_specialized_open_question_recovery(state):
            if self._action_intent_pending_resolution_tool(state) == "resolve_action_intent_future_use":
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    thought="why 状态变化题在恢复阶段仍证据不足，优先回到后续用途专用裁决，不退回通用 query_time。",
                )
                if future_use is not None:
                    return future_use
            pairwise = self._build_action_intent_pairwise_resolution_decision(
                state=state,
                hints=hints,
                thought="why 状态变化题在恢复阶段仍证据不足，优先回到状态变化专用二选一裁决，不退回通用 query_time。",
            )
            if pairwise is not None:
                return pairwise
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

        if (
            self._is_action_intent_task(state)
            and ("need_disambiguating_evidence" in open_questions or "need_disambiguating_evidence" in verifier_missing)
        ):
            targeted_recovery = self._recover_action_intent_after_verifier_blocked_finish(
                state=state,
                hints=hints,
            )
            if targeted_recovery is not None:
                add_candidate(
                    0,
                    10,
                    0,
                    "why 题已被 verifier 明确拦下且仍是 close call，优先回到专用补关键帧/专用裁决路径，避免被泛化检索或音频峰值路径稀释。",
                    targeted_recovery.thought,
                    targeted_recovery.tool,
                    targeted_recovery.args,
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
                "extract_frames_for_range",
                "sample_frames_around_peaks",
                "inspect_visual_evidence",
                "detect_audio_peaks",
                "rank_choices_from_state",
                "infer_action_intent",
                "resolve_action_intent_pairwise",
                "resolve_action_intent_future_use",
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

    def _action_intent_verifier_blocker_hint(self, state: AgentState) -> str:
        if not self._is_action_intent_task(state):
            return ""
        latest_verification = self._state_latest_verification(state)
        summary = str(latest_verification.get("summary") or "")
        match = re.search(r"why_blocker=([a-z_]+)", summary)
        if match:
            return str(match.group(1) or "")
        missing = {
            str(item)
            for item in latest_verification.get("missing_evidence_types", [])
            if isinstance(item, str) and item
        }
        if "need_precondition_context" in missing:
            return "precondition_context"
        if "need_post_action_evidence" in missing:
            return "post_action_evidence"
        return ""

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

    def _latest_action_intent_resolution_payload(self, state: AgentState) -> tuple[str, dict[str, Any]] | None:
        for entry in reversed(getattr(state, "tool_trace", [])):
            if not isinstance(entry, dict):
                continue
            tool = str(entry.get("tool") or "")
            if tool not in {
                "infer_action_intent",
                "resolve_action_intent_pairwise",
                "resolve_action_intent_future_use",
            }:
                continue
            payload = entry.get("raw_result")
            if not isinstance(payload, dict) or payload.get("tool_failed"):
                continue
            if payload.get("best_index") is None:
                continue
            return tool, payload
        return None

    def _action_intent_competing_candidate_index(self, payload: dict[str, Any], state: AgentState) -> int | None:
        second_best = self._coerce_choice_index(payload.get("second_best_index"), state.choices)
        if second_best is not None:
            return second_best
        losing = self._coerce_choice_index(payload.get("losing_index"), state.choices)
        if losing is not None:
            return losing
        candidate_scores: list[tuple[int, float]] = []
        for item in payload.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), state.choices)
            if index is None:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0
            candidate_scores.append((index, score))
        if len(candidate_scores) < 2:
            return None
        ranked = sorted(candidate_scores, key=lambda pair: (-pair[1], pair[0]))
        return ranked[1][0]

    def _action_intent_future_use_score_gap(self, payload: dict[str, Any]) -> float:
        scores: list[float] = []
        for item in payload.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            try:
                scores.append(float(item.get("score") or 0.0))
            except Exception:  # noqa: BLE001
                continue
        if len(scores) < 2:
            return 0.0
        ranked = sorted(scores, reverse=True)
        return ranked[0] - ranked[1]

    def _action_intent_competing_pair_still_needs_disambiguation(
        self,
        *,
        state: AgentState,
        best_index: int,
        competitor_index: int,
    ) -> bool:
        question = str(getattr(state, "question", "") or "")
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        pair = [best_index, competitor_index]
        if action_intent_needs_future_use_resolution(question=question, choices=choices, indices=pair):
            return True
        if action_intent_needs_pairwise_resolution(question=question, choices=choices, indices=pair):
            return True
        categories = selected_choice_categories(choices, pair)
        best_categories = set(categories.get(best_index) or set())
        competitor_categories = set(categories.get(competitor_index) or set())
        return best_categories != competitor_categories and bool(best_categories | competitor_categories)

    def _action_intent_result_is_close_call_for_recovery(
        self,
        *,
        state: AgentState,
        tool_name: str,
        payload: dict[str, Any],
    ) -> bool:
        if bool(payload.get("need_more_evidence")):
            return True
        best_index = self._coerce_choice_index(payload.get("best_index"), state.choices)
        competitor_index = self._action_intent_competing_candidate_index(payload, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return False
        if not self._action_intent_competing_pair_still_needs_disambiguation(
            state=state,
            best_index=best_index,
            competitor_index=competitor_index,
        ):
            return False
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        if tool_name == "resolve_action_intent_pairwise":
            direct_effect = str(payload.get("direct_effect") or "").strip()
            downstream_action = str(payload.get("downstream_action") or "").strip()
            return confidence < 0.84 or not direct_effect or not downstream_action
        if tool_name == "resolve_action_intent_future_use":
            decisive = str(payload.get("decisive_observation") or "").strip()
            score_gap = self._action_intent_future_use_score_gap(payload)
            return confidence < 0.84 or not decisive or score_gap < 0.18
        support_text = self._action_intent_result_support_text(payload)
        direct_result_markers = (
            "immediately",
            "right after",
            "shortly after",
            "next step",
            "afterwards",
            "put back",
            "returned",
            "placed on the scale",
            "turns on the tap",
            "turns off",
            "opened",
            "closed",
            "poured",
            "wiped",
            "dried",
            "立刻",
            "随后",
            "接着",
            "放回",
            "放到秤上",
        )
        return confidence < 0.9 or not any(marker in support_text for marker in direct_result_markers)

    def _coerce_choice_index(self, value: Any, choices: list[Any]) -> int | None:
        try:
            index = int(value)
        except Exception:  # noqa: BLE001
            return None
        if 0 <= index < len(choices):
            return index
        return None

    def _recover_action_intent_after_verifier_blocked_finish(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> PlannerDecision | None:
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return None
        tool_name, payload = latest
        blocker_hint = self._action_intent_verifier_blocker_hint(state)
        is_close_call = self._action_intent_result_is_close_call_for_recovery(
            state=state,
            tool_name=tool_name,
            payload=payload,
        )
        requires_blocker_driven_recovery = blocker_hint in {
            "precondition_context",
            "post_action_evidence",
            "future_use_close_call",
            "pairwise_close_call",
        }
        if not is_close_call and not requires_blocker_driven_recovery:
            return None
        if tool_name == "infer_action_intent":
            if (
                blocker_hint == "precondition_context"
                and self._action_intent_needs_precondition_context(state=state, result=payload)
                and not self._action_intent_has_precondition_frames(state=state, hints=hints)
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus="verifier_blocked_precondition_context",
                )
                if precondition is not None:
                    return precondition
            if blocker_hint in {"post_action_evidence", "future_use_close_call"}:
                if self._action_intent_followup_attempt_count(state) < self._action_intent_initial_followup_budget(state):
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 判为缺少动作后决定性证据；先围绕动作尾部后的短窗口主动补关键帧，确认是否真的出现称重、倒空、检查、放回或具体下游使用。",
                )
                if transition_probe is not None:
                    return transition_probe
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 判为缺少动作后证据后，回到后续用途专用裁决，用更新后的结果帧重新判断真实目的。",
                )
                if future_use is not None:
                    return future_use
                extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                    state=state,
                    hints=hints,
                    focus="verifier_blocked_post_action_evidence",
                    window_s=8.5,
                )
                if extra_followup is not None:
                    return extra_followup
            if blocker_hint == "pairwise_close_call":
                if self._action_intent_followup_attempt_count(state) < self._action_intent_initial_followup_budget(state):
                    followup = self._build_action_intent_followup_sampling_decision(state=state, hints=hints)
                    if followup is not None:
                        return followup
                transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 判为 top-2 后果型 close call；先主动补更近的结果帧，再回到二选一裁决。",
                )
                if transition_probe is not None:
                    return transition_probe
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 判为 pairwise close call；回到 top-2 专用裁决，不允许泛化结果提前收口。",
                )
                if pairwise is not None:
                    return pairwise
            transition_probe = self._build_action_intent_transition_probe_decision(
                state=state,
                hints=hints,
                result=payload,
                thought="why 题被 verifier 拦下，因为当前 top 候选仍没把竞争解释真正压下去；先按当前冲突类型主动重采样决定性关键帧。",
            )
            if transition_probe is not None:
                return transition_probe
            if self._action_intent_needs_future_use_evidence(state=state, result=payload):
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 拦下后，直接转入后续用途专用裁决，不让五选一结果提前收口。",
                )
                if future_use is not None:
                    return future_use
            if self._action_intent_pair_needs_outcome_resolution(state=state, result=payload):
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 拦下后，直接转入 top-2 专用裁决，不继续停留在泛化 best guess。",
                )
                if pairwise is not None:
                    return pairwise
            extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                state=state,
                hints=hints,
                focus="verifier_blocked_close_call_recovery",
                window_s=8.0 if self._action_intent_needs_future_use_evidence(state=state, result=payload) else 6.0,
            )
            return extra_followup
        transition_probe = self._build_action_intent_transition_probe_decision(
            state=state,
            hints=hints,
            result=payload,
            thought="why 题专用裁决被 verifier 拦下，因为当前仍是 close call；先针对当前竞争候选主动补决定性关键帧。",
        )
        if transition_probe is not None:
            return transition_probe
        extra_followup = self._build_action_intent_extra_followup_sampling_decision(
            state=state,
            hints=hints,
            focus=str(payload.get("needed_observation") or "verifier_blocked_close_call_recovery"),
            window_s=8.0 if tool_name == "resolve_action_intent_future_use" else 6.0,
        )
        if extra_followup is not None:
            return extra_followup
        if tool_name == "resolve_action_intent_future_use":
            return self._build_action_intent_future_use_resolution_decision(
                state=state,
                hints=hints,
                result=payload,
                thought="why 题被 verifier 拦下后，回到后续用途专用裁决并用更新后的证据重判。",
            )
        if tool_name == "resolve_action_intent_pairwise":
            return self._build_action_intent_pairwise_resolution_decision(
                state=state,
                hints=hints,
                result=payload,
                thought="why 题被 verifier 拦下后，回到 top-2 专用裁决并用更新后的证据重判。",
            )
        return None

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
