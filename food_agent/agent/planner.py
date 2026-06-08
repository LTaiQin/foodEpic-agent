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
    action_intent_requires_strict_visual_disambiguation,
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
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        if bias_profile["resolver_hint"]:
            return str(bias_profile["resolver_hint"])
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
        rescued_top_pair = self._action_intent_semantic_rescue_candidate_indices(
            state=state,
            indices=top_pair,
            result=result,
        )
        if len(rescued_top_pair) >= 2:
            top_pair = rescued_top_pair
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
        rescued_indices = self._action_intent_semantic_rescue_candidate_indices(
            state=state,
            indices=indices,
            result=result,
        )
        if len(rescued_indices) >= 2:
            return rescued_indices
        return indices

    def _action_intent_semantic_rescue_candidate_indices(
        self,
        *,
        state: AgentState,
        indices: list[int],
        result: dict[str, Any] | None = None,
    ) -> list[int]:
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        normalized = [index for index in indices if 0 <= int(index) < len(choices)]
        deduped: list[int] = []
        for index in normalized:
            if index not in deduped:
                deduped.append(index)
        if len(deduped) < 2:
            return deduped
        question = str(getattr(state, "question", "") or "").lower()
        full_categories = selected_choice_categories(choices, range(len(choices)))
        current_categories = selected_choice_categories(choices, deduped)
        full_union = set().union(*(full_categories.get(index) or set() for index in full_categories))
        current_union = set().union(*(current_categories.get(index) or set() for index in current_categories))
        best_index = self._coerce_choice_index((result or {}).get("best_index"), state.choices)
        if best_index is None or best_index not in range(len(choices)):
            best_index = deduped[0]

        def first_index_for_missing_semantic(kind: str) -> int | None:
            for index, cats in full_categories.items():
                if index == best_index:
                    continue
                categories = set(cats or set())
                if kind == "transfer_contents" and "transfer_contents" in categories:
                    return index
                if kind == "clean_dry" and "clean_dry" in categories:
                    return index
                if kind == "relocation" and categories & {"generic_relocation", "final_place_return"}:
                    return index
                if kind == "measure_weigh" and "measure_weigh" in categories:
                    return index
                if kind == "open_close" and "open_close" in categories:
                    return index
            return None

        if any(token in question for token in ("<flip ", "<turn ", "<shake ", "<tilt ", "<tip ", "<tap ", "<hit ", "<knock ")):
            if "clean_dry" in full_union and "transfer_contents" in full_union:
                if not ("clean_dry" in current_union and "transfer_contents" in current_union):
                    missing_kind = "transfer_contents" if "transfer_contents" not in current_union else "clean_dry"
                    alt_index = first_index_for_missing_semantic(missing_kind)
                    if alt_index is not None and alt_index != best_index:
                        return [best_index, alt_index]

        if any(token in question for token in ("towel", "cloth", "napkin", "paper towel", "tea towel", "dish cloth", "hand towel")):
            if any(token in question for token in ("<pick up ", "<grab ", "<lift ", "<take ", "<move ", "<shift ")):
                if "clean_dry" in full_union and {"generic_relocation", "final_place_return"} & full_union:
                    current_has_relocation = bool({"generic_relocation", "final_place_return"} & current_union)
                    if not ("clean_dry" in current_union and current_has_relocation):
                        missing_kind = "relocation" if not current_has_relocation else "clean_dry"
                        alt_index = first_index_for_missing_semantic(missing_kind)
                        if alt_index is not None and alt_index != best_index:
                            return [best_index, alt_index]

        if any(token in question for token in ("<tap ", "<press ", "<push ")):
            if any(token in question for token in ("scale", "button", "switch", "knob")):
                if "measure_weigh" in full_union and "open_close" in full_union:
                    if not ("measure_weigh" in current_union and "open_close" in current_union):
                        missing_kind = "measure_weigh" if "measure_weigh" not in current_union else "open_close"
                        alt_index = first_index_for_missing_semantic(missing_kind)
                        if alt_index is not None and alt_index != best_index:
                            return [best_index, alt_index]
        return deduped

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
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        dense_near_followup = self._action_intent_prefers_dense_near_followup(state)
        result_driven_followup = self._action_intent_prefers_result_driven_followup(state)
        review_transition_focus = (
            bias_profile["revealed_target_retrieval"]
            or bias_profile["revealed_slot_placement"]
            or bias_profile["revealed_fixture_enablement"]
            or (
                bias_profile["hand_free_next_action"]
                and not (bias_profile["next_use_unclear"] or bias_profile["final_location_unclear"])
            )
        )
        review_late_focus = bias_profile["next_use_unclear"] or bias_profile["final_location_unclear"]
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
        if review_transition_focus and not review_late_focus:
            start_time = max(0.0, max(action_end - 0.12, start_time - 0.9))
            window_s = min(max(window_s, 4.4 if not bias_profile["revealed_slot_placement"] else 4.2), 5.4)
        if review_late_focus:
            window_s = max(window_s, 8.8 if not bias_profile["final_location_unclear"] else 9.0)
        sample_count = 6 if dense_near_followup and attempt_count <= 1 else 4
        if result_driven_followup:
            sample_count = max(sample_count, 5 if attempt_count <= 1 else 4)
        if review_transition_focus:
            sample_count = max(sample_count, 6)
        if review_late_focus:
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
        elif mode == "revealed_target_retrieval":
            start_time = max(0.0, action_end - 0.12)
            end_time = action_end + 2.7
            stride_s = 0.35
            max_frames = 7
        elif mode == "revealed_slot_placement":
            start_time = action_end + 0.02
            end_time = action_end + 3.6
            stride_s = 0.4
            max_frames = 7
        elif mode == "revealed_fixture_enablement":
            start_time = max(0.0, action_end - 0.08)
            end_time = action_end + 3.0
            stride_s = 0.35
            max_frames = 6
        elif mode == "reveal_or_access_result":
            start_time = max(0.0, action_end - 0.15)
            end_time = action_end + 3.2
            stride_s = 0.4
            max_frames = 6
        elif mode == "receptacle_outcome":
            start_time = max(0.0, action_end - 0.12)
            end_time = action_end + 2.8
            stride_s = 0.35
            max_frames = 7
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
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        reveal_subtype = (
            self._action_intent_reveal_conflict_subtype(state=state, result=result)
            if bool(profile["has_hidden_access_exact_use_conflict"])
            else self._action_intent_reveal_conflict_subtype(state=state, result=result)
        )
        has_explicit_hand_free_conflict = self._action_intent_has_hand_free_future_use_conflict(state=state, result=result)
        if bias_profile["revealed_target_retrieval"]:
            return "revealed_target_retrieval"
        if bias_profile["revealed_slot_placement"]:
            return "revealed_slot_placement"
        if bias_profile["revealed_fixture_enablement"] and not has_explicit_hand_free_conflict:
            return "revealed_fixture_enablement"
        if self._action_intent_prefers_followup_state_change_only(state):
            return "state_change"
        if reveal_subtype == "revealed_fixture_enablement" and not has_explicit_hand_free_conflict:
            return reveal_subtype
        if (
            needed_profile["prefer_mixed_horizon"]
            or self._action_intent_pair_spans_immediate_and_later_outcomes(state=state, result=result)
            or (result is None and self._action_intent_initial_pair_spans_immediate_and_later_outcomes(state))
        ):
            return "mixed_temporal_horizon"
        if bias_profile["hand_free_next_action"]:
            return "hand_free_next_action"
        if (needed_profile["prefer_hand_free_next_action"] or has_explicit_hand_free_conflict) and not (
            reveal_subtype == "revealed_fixture_enablement" and not has_explicit_hand_free_conflict
        ):
            return "hand_free_next_action"
        if needed_profile["prefer_receptacle_outcome"]:
            return "receptacle_outcome"
        if bias_profile["final_location_unclear"]:
            return "final_placement_result"
        if bias_profile["next_use_unclear"]:
            return "future_use_outcome"
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
            if reveal_subtype:
                return reveal_subtype
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

    def _action_intent_reveal_conflict_subtype(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
    ) -> str:
        candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if len(candidate_indices) < 2:
            candidate_indices = list(range(len(choices)))
        candidate_choices = [
            choices[index].lower()
            for index in candidate_indices
            if 0 <= index < len(choices)
        ]
        if not candidate_choices:
            return ""
        support_text = self._action_intent_result_support_text(result)
        reveal_context_text = " ".join(candidate_choices) + " " + support_text + " " + str(getattr(state, "question", "") or "").lower()
        if not any(
            token in reveal_context_text
            for token in (
                "behind",
                "hidden",
                "reveals",
                "revealed",
                "reveal",
                "slot",
                "freed area",
                "freed slot",
                "available spot",
                "behind it",
                "behind the",
                "后面",
                "露出",
                "腾出的槽位",
                "空位",
            )
        ):
            return ""
        has_slot_placement = any(
            any(
                token in choice
                for token in (
                    "put into the freed slot",
                    "place into the freed slot",
                    "put the",
                    "place the",
                    "freed slot",
                    "slot behind",
                    "right place",
                    "proper place",
                    "放进腾出的槽位",
                    "放到腾出的槽位",
                    "归位",
                )
            )
            for choice in candidate_choices
        )
        has_hidden_target_retrieval = any(
            any(
                token in choice
                for token in (
                    "retrieve",
                    "take the hidden",
                    "take the small jar",
                    "pick up the hidden",
                    "pick up the small jar",
                    "take the spice jar",
                    "hidden behind",
                    "retrieve the red curry paste",
                    "retrieve the",
                    "取出后面的",
                    "拿后面的",
                    "取到后面",
                )
            )
            for choice in candidate_choices
        )
        has_fixture_enablement = any(
            any(
                token in choice
                for token in (
                    "turn on",
                    "switch on",
                    "open the",
                    "open ",
                    "turn off",
                    "switch off",
                    "tap",
                    "use the scale",
                    "turn on the scale",
                    "打开",
                    "开启",
                    "开机",
                )
            )
            for choice in candidate_choices
        )
        if has_hidden_target_retrieval and any(
            token in support_text
            for token in (
                "hidden item",
                "item behind",
                "behind it",
                "behind the",
                "retrieved from behind",
                "picked up from behind",
                "taken from behind",
                "small jar",
                "spice jar",
                "red curry paste",
                "后面物体",
                "后面的目标",
            )
        ):
            return "revealed_target_retrieval"
        if has_slot_placement and any(
            token in support_text
            for token in (
                "freed slot",
                "slot behind",
                "available spot",
                "put into the freed slot",
                "placed into the freed slot",
                "revealed slot",
                "腾出的槽位",
                "空位",
            )
        ):
            return "revealed_slot_placement"
        if has_fixture_enablement and any(
            token in support_text
            for token in (
                "scale behind",
                "turn on the scale",
                "open the",
                "switch on",
                "fixture behind",
                "revealed appliance",
                "露出的装置",
                "后面的秤",
            )
        ):
            return "revealed_fixture_enablement"
        if has_hidden_target_retrieval and not has_slot_placement and not has_fixture_enablement:
            return "revealed_target_retrieval"
        if has_slot_placement:
            return "revealed_slot_placement"
        if has_fixture_enablement:
            return "revealed_fixture_enablement"
        return ""

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

    def _action_intent_initial_pair_spans_immediate_and_later_outcomes(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if len(choices) < 2:
            return False
        categories_by_index = selected_choice_categories(choices, None)
        later_outcome_categories = {
            "final_place_return",
            "measure_weigh",
            "transfer_contents",
            "serve_consume",
            "clean_dry",
            "food_prep",
            "discard",
        }
        immediate_indices = [
            index
            for index, categories in categories_by_index.items()
            if self._action_intent_choice_is_immediate_micro_outcome_candidate(
                choices[index],
                set(categories or set()),
            )
        ]
        if not immediate_indices:
            return False
        return any(
            index not in immediate_indices and set(categories or set()) & later_outcome_categories
            for index, categories in categories_by_index.items()
        )

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
        if self._action_intent_followup_attempt_count(state) < 1 and not self._action_intent_should_preempt_initial_followup_with_transition(
            state=state,
            hints=hints,
            result=result,
        ):
            return False
        if self._action_intent_transition_probe_window(state=state, hints=hints, result=result) is None:
            return False
        needed_profile = self._action_intent_needed_observation_profile(state=state, result=result)
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
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
        if bias_profile["needs_more_evidence"] and (
            bias_profile["revealed_target_retrieval"]
            or bias_profile["revealed_slot_placement"]
            or bias_profile["revealed_fixture_enablement"]
            or bias_profile["hand_free_next_action"]
            or bias_profile["final_location_unclear"]
        ):
            return True
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
                or needed_profile["prefer_receptacle_outcome"]
            )
        ):
            return True
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        if bool(profile["has_hidden_access_exact_use_conflict"]) and not hand_free_future_use:
            return False
        if self._action_intent_needs_future_use_evidence(state=state, result=result) and not hand_free_future_use:
            if bias_profile["final_location_unclear"]:
                return True
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
        if isinstance(result, dict) and (
            self._action_intent_result_is_weak_generic_claim(state=state, result=result)
            or self._action_intent_result_is_workspace_or_final_placement_close_call(state=state, result=result)
        ):
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
        action_times: list[float] = []
        for key in ("times", "input_times"):
            for value in hints.get(key) or []:
                try:
                    action_times.append(float(value))
                except Exception:  # noqa: BLE001
                    continue
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
            latest_intent = self._latest_successful_action_intent_result(state)
            needed_profile = self._action_intent_needed_observation_profile(state=state, result=latest_intent if latest_intent else None)
            allow_late_followup_review = (
                needed_profile["prefer_mixed_horizon"]
                or needed_profile["prefer_future_use_outcome"]
                or needed_profile["prefer_final_placement"]
            )
            if not allow_late_followup_review and latest_intent:
                allow_late_followup_review = (
                    self._action_intent_needs_future_use_evidence(state=state, result=latest_intent)
                    or self._action_intent_pair_spans_immediate_and_later_outcomes(state=state, result=latest_intent)
                )
            if not allow_late_followup_review:
                return []
            has_segment = any("_segment_" in name for name in names)
            has_late_followup = any(
                marker in name
                for name in names
                for marker in ("_followup_ext2_", "_followup_ext3_", "_followup_ext4_")
            )
            has_regular_followup = any("_followup_" in name for name in names)
            if not has_segment or not has_late_followup:
                all_task_frames = self._action_intent_current_task_artifact_frames(
                    self._filter_visual_image_paths(list(getattr(state, "retrieved_frames", []) or []))
                )
                all_names = [Path(path).name.lower() for path in all_task_frames]
                has_segment = any("_segment_" in name for name in all_names)
                has_late_followup = any(
                    marker in name
                    for name in all_names
                    for marker in ("_followup_ext2_", "_followup_ext3_", "_followup_ext4_")
                )
                has_regular_followup = any("_followup_" in name for name in all_names)
                if has_segment and (has_late_followup or has_regular_followup):
                    if action_times:
                        staged = self._stage_action_intent_frames(
                            state=state,
                            frames=all_task_frames,
                            action_times=action_times,
                            limit=8,
                            include_followup=True,
                        )
                        if staged:
                            return staged
                    return all_task_frames[-8:]
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
        if tool_name not in {"extract_frames_for_range", "sample_frames_around_peaks", "retrieve_cached_artifacts", "sample_sparse_frames"}:
            return False
        tag = self._action_intent_timeline_review_tag(last_tool)
        if tool_name == "extract_frames_for_range" and not tag.endswith("_followup_transition"):
            return False
        if tool_name == "sample_frames_around_peaks" and not tag.endswith("_followup_peaks"):
            return False
        if tool_name == "sample_sparse_frames":
            if "_followup_ext" not in tag and not (
                tag.endswith("_followup") and self._action_intent_pending_resolution_tool(state)
            ):
                return False
        if tool_name == "retrieve_cached_artifacts" and not any(
            marker in tag for marker in ("followup_transition", "followup_peaks", "followup_ext", "followup")
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

    def _action_intent_timeline_review_bias_profile(self, state: AgentState) -> dict[str, Any]:
        payload = self._latest_action_intent_timeline_review(state)
        empty = {
            "has_review": False,
            "needs_more_evidence": False,
            "resolver_hint": "",
            "revealed_target_retrieval": False,
            "revealed_slot_placement": False,
            "revealed_fixture_enablement": False,
            "hand_free_next_action": False,
            "next_use_unclear": False,
            "final_location_unclear": False,
            "state_change_focus": False,
            "immediate_transition_focus": False,
        }
        if not payload:
            return empty

        def merged_text(*keys: str) -> str:
            return " ".join(str(payload.get(key) or "") for key in keys).strip().lower()

        def has_any(text: str, markers: tuple[str, ...]) -> bool:
            return any(marker in text for marker in markers)

        timeline_summary = merged_text("timeline_summary")
        immediate_result = merged_text("immediate_result")
        next_action = merged_text("next_action_hint")
        direct_purpose = merged_text("direct_purpose_hint")
        reveal_evidence = merged_text("access_or_reveal_evidence")
        hand_free_evidence = merged_text("hand_free_enablement_evidence")
        next_use_evidence = merged_text("next_use_evidence")
        state_change_hint = merged_text("state_change_hint")
        target_location = merged_text("target_location")
        ambiguity_note = merged_text("ambiguity_note")
        review_text = self._action_intent_timeline_review_text(state)
        combined_reveal = " ".join((reveal_evidence, next_action, ambiguity_note, timeline_summary, review_text))
        combined_hand_free = " ".join((hand_free_evidence, next_action, direct_purpose, timeline_summary, review_text))
        combined_next_use = " ".join((next_use_evidence, direct_purpose, ambiguity_note, review_text))
        combined_final_location = " ".join((target_location, next_use_evidence, ambiguity_note, direct_purpose, review_text))
        combined_state_change = " ".join((state_change_hint, immediate_result, timeline_summary, review_text))
        needs_more_evidence = self._action_intent_timeline_review_needs_more_evidence(payload)

        reveal_markers = (
            "behind",
            "hidden",
            "reveal",
            "revealed",
            "reachable",
            "freed slot",
            "available spot",
            "slot",
            "behind it",
            "behind the",
            "后面",
            "露出",
            "空位",
            "槽位",
        )
        hidden_target_markers = (
            "hidden jar",
            "hidden item",
            "retrieval is not yet visible",
            "retrieve",
            "pick up the hidden",
            "take the hidden",
            "take from behind",
            "pick up from behind",
            "取后面的",
            "拿后面的",
            "后面的目标",
        )
        slot_markers = (
            "freed slot",
            "available spot",
            "slot placement",
            "placement into the slot",
            "put into the slot",
            "place into the slot",
            "put back into the slot",
            "空位",
            "槽位",
            "放进",
            "归位",
        )
        fixture_markers = (
            "scale behind",
            "revealed appliance",
            "revealed fixture",
            "turn on",
            "switch on",
            "tap area",
            "sink area",
            "open the",
            "露出的装置",
            "后面的秤",
            "龙头",
        )
        hand_free_markers = (
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
        next_use_uncertain_markers = (
            "later use is still unclear",
            "next use is still unclear",
            "not yet visible whether",
            "not visible whether",
            "might be poured",
            "might be weighed",
            "might be put back",
            "multiple later-use explanations remain plausible",
            "后续用途",
            "仍不清楚",
            "看不出之后",
        )
        final_location_markers = (
            "final location remains unclear",
            "not visible where",
            "put back or only moved temporarily",
            "returned or only moved temporarily",
            "whether it is put back",
            "whether it is returned",
            "where it ends up",
            "最终位置",
            "归位",
            "放回原处",
            "暂时移动",
        )
        state_change_markers = (
            "display",
            "changes to",
            "turned on",
            "turned off",
            "opened",
            "closed",
            "fills up",
            "starts running",
            "stops running",
            "显示",
            "打开",
            "关闭",
            "变成",
        )
        immediate_transition_markers = (
            "reaches toward",
            "moves toward",
            "goes to",
            "pick up",
            "turn on",
            "open",
            "close",
            "retrieve",
            "put into the slot",
            "拿起",
            "去拿",
            "去开",
        )

        reveal_focus = has_any(combined_reveal, reveal_markers)
        revealed_target_retrieval = reveal_focus and has_any(combined_reveal, hidden_target_markers)
        revealed_slot_placement = reveal_focus and has_any(combined_reveal, slot_markers)
        revealed_fixture_enablement = reveal_focus and has_any(
            " ".join((combined_reveal, combined_hand_free)),
            fixture_markers,
        )
        hand_free_next_action = has_any(combined_hand_free, hand_free_markers)
        next_use_unclear = has_any(combined_next_use, next_use_uncertain_markers) or (
            needs_more_evidence and "later use" in review_text
        )
        final_location_unclear = has_any(combined_final_location, final_location_markers)
        state_change_focus = has_any(combined_state_change, state_change_markers)
        immediate_transition_focus = has_any(
            " ".join((next_action, reveal_evidence, hand_free_evidence, direct_purpose)),
            immediate_transition_markers,
        )

        resolver_hint = ""
        if next_use_unclear or final_location_unclear:
            resolver_hint = "future_use"
        elif revealed_target_retrieval or revealed_slot_placement or revealed_fixture_enablement or hand_free_next_action:
            resolver_hint = "pairwise"

        return {
            "has_review": True,
            "needs_more_evidence": needs_more_evidence,
            "resolver_hint": resolver_hint,
            "revealed_target_retrieval": revealed_target_retrieval,
            "revealed_slot_placement": revealed_slot_placement,
            "revealed_fixture_enablement": revealed_fixture_enablement,
            "hand_free_next_action": hand_free_next_action,
            "next_use_unclear": next_use_unclear,
            "final_location_unclear": final_location_unclear,
            "state_change_focus": state_change_focus,
            "immediate_transition_focus": immediate_transition_focus,
        }

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
        if latest is not None:
            _tool, payload = latest
            text = str(payload.get("needed_observation") or "").strip().lower()
            if text:
                return text
        for item in reversed(list(getattr(state, "working_memory", []))[-16:]):
            if not isinstance(item, str) or not item.startswith("action_intent_needed_observation="):
                continue
            return str(item.split("=", 1)[1] or "").strip().lower()
        return ""

    def _action_intent_needed_observation_target_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> tuple[str, str] | None:
        if self._action_intent_followup_attempt_count(state) < 2 and not any(
            isinstance(item, str) and item.startswith("action_intent_needed_observation=")
            for item in list(getattr(state, "working_memory", []))[-16:]
        ):
            return None
        text = self._action_intent_needed_observation_text(state=state, result=result)
        if not text:
            return None
        action_object = self._action_intent_question_object_hint(state)
        fixture_targets = self._action_intent_choice_fixture_target_candidates(choice=text, action_object=action_object)
        object_targets = [
            token
            for token in self._action_intent_choice_target_object_candidates(choice=text, action_object=action_object)
            if token not in set(fixture_targets)
        ]
        unique_targets: list[tuple[str, str]] = []
        seen_targets: set[str] = set()
        for target in fixture_targets:
            if not target or target == action_object or target in seen_targets:
                continue
            seen_targets.add(target)
            unique_targets.append((target, "fixture"))
        for target in object_targets:
            if not target or target == action_object or target in seen_targets:
                continue
            seen_targets.add(target)
            unique_targets.append((target, "object"))
        if len(unique_targets) == 1:
            return unique_targets[0]
        if len(unique_targets) > 1:
            text_lc = text
            reason_lc = self._action_intent_result_support_text(result) if isinstance(result, dict) else ""
            revealed_slot_terms = (
                "freed slot",
                "free slot",
                "sink slot",
                "specific freed slot",
                "target position",
                "placed into the freed slot",
                "put into the freed slot",
                "placed into the sink",
                "put into the sink",
            )
            if any(term in text_lc or term in reason_lc for term in revealed_slot_terms):
                for target, kind in unique_targets:
                    if kind == "object":
                        return target, kind
            inspection_vs_later_use_terms = (
                "tilted to pour",
                "toward the sink",
                "carried over the plate",
                "briefly checked near the hob",
                "stays near the hob",
                "checked near the hob",
            )
            if any(term in text_lc for term in inspection_vs_later_use_terms):
                preferred_targets = ("sink", "plate", "bowl")
                for preferred in preferred_targets:
                    for target, kind in unique_targets:
                        if target == preferred:
                            return target, kind
        return None

    def _action_intent_needed_observation_relation_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> tuple[str, str, str] | None:
        if self._action_intent_followup_attempt_count(state) < 2 and not any(
            isinstance(item, str) and item.startswith("action_intent_needed_observation=")
            for item in list(getattr(state, "working_memory", []))[-16:]
        ):
            return None
        target_hint = self._action_intent_needed_observation_target_hint(state=state, result=result)
        if target_hint is None:
            return None
        text = self._action_intent_needed_observation_text(state=state, result=result)
        if not text:
            return None
        if (
            target_hint[1] == "object"
            and any(marker in text for marker in ("freed slot", "free slot", "sink slot", "target position"))
        ):
            return None
        relation_matches: list[str] = []
        relation_specs = (
            (
                "on_target",
                (
                    "placed onto the scale",
                    "placed on the scale",
                    "put on the scale",
                    "used on the scale",
                    "onto the scale",
                    "on the scale",
                ),
            ),
            (
                "return_to_target",
                (
                    "put back in the fridge",
                    "back in the fridge",
                    "returned to the fridge",
                    "returned into the fridge",
                    "returned to the shelf",
                    "placed back on the shelf",
                ),
            ),
            (
                "into_target",
                (
                    "placed into the sink",
                    "put into the sink",
                    "into the sink",
                    "in the sink",
                    "sink wash area",
                    "into the sink area",
                ),
            ),
            (
                "over_target",
                (
                    "carried over the plate",
                    "moved over the plate",
                    "over the plate",
                    "carried over the bowl",
                    "over the bowl",
                    "over the sink",
                    "toward the sink",
                ),
            ),
        )
        for relation_name, markers in relation_specs:
            if any(marker in text for marker in markers):
                relation_matches.append(relation_name)
        unique_relations = list(dict.fromkeys(relation_matches))
        if len(unique_relations) != 1:
            return None
        target_name, target_kind = target_hint
        return target_name, target_kind, unique_relations[0]

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
                "prefer_receptacle_outcome": False,
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
        receptacle_terms = (
            "sink",
            "pan",
            "pot",
            "bowl",
            "cup",
            "jar",
            "container",
            "水槽",
            "锅",
            "碗",
            "杯",
            "容器",
        )
        residue_action_terms = (
            "crumb",
            "residue",
            "drop",
            "fall",
            "release",
            "drain",
            "碎屑",
            "残渣",
            "掉",
            "落",
            "沥",
        )
        immediate = any(term in text for term in immediate_terms)
        future_use = any(term in text for term in future_use_terms)
        reveal_access = any(term in text for term in reveal_terms)
        hand_free = any(term in text for term in hand_free_terms)
        safety_or_spill = any(term in text for term in safety_terms)
        state_change_only = any(term in text for term in state_change_terms)
        final_placement = any(term in text for term in final_placement_terms)
        question_text = str(getattr(state, "question", "") or "").lower()
        choices_text = " ".join(str(choice) for choice in getattr(state, "choices", []) or []).lower()
        receptacle_outcome = any(term in text for term in receptacle_terms) and any(
            term in text for term in residue_action_terms
        )
        if not receptacle_outcome and any(token in question_text for token in ("<flip ", "<turn ", "<shake ", "<tilt ", "<tip ", "<tap ", "<hit ", "<knock ")):
            if any(
                term in choices_text
                for term in (
                    "into the sink",
                    "into the pan",
                    "into the bowl",
                    "into the pot",
                    "into the container",
                    "掉进水槽",
                    "掉回锅",
                    "掉回碗",
                    "落回",
                )
            ) and any(
                term in choices_text
                for term in (
                    "drop",
                    "fall",
                    "release",
                    "crumb",
                    "residue",
                    "drain",
                    "掉",
                    "落",
                    "碎屑",
                    "残渣",
                    "沥",
                )
            ):
                receptacle_outcome = True
        contrastive = ("whether" in text or "是否" in text) and any(
            token in text for token in (" or ", " first ", " before ", " after ", " versus ", " vs ")
        )
        mixed_horizon = (immediate and future_use) or (contrastive and immediate and (future_use or final_placement))
        dense_near = immediate or reveal_access or receptacle_outcome or hand_free or safety_or_spill or state_change_only
        return {
            "prefer_dense_near": dense_near,
            "prefer_result_driven": future_use or final_placement or mixed_horizon,
            "prefer_state_change_only": state_change_only and not (future_use or final_placement or reveal_access),
            "prefer_mixed_horizon": mixed_horizon,
            "prefer_reveal_access": reveal_access,
            "prefer_receptacle_outcome": receptacle_outcome,
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
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        if (
            bias_profile["revealed_target_retrieval"]
            or bias_profile["revealed_slot_placement"]
            or bias_profile["revealed_fixture_enablement"]
            or bias_profile["hand_free_next_action"]
            or bias_profile["immediate_transition_focus"]
        ):
            return True
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
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        if bias_profile["next_use_unclear"] or bias_profile["final_location_unclear"]:
            return True
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
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        if bias_profile["state_change_focus"] and not (
            bias_profile["next_use_unclear"] or bias_profile["final_location_unclear"]
        ):
            return True
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
        if not self._is_action_intent_task(state):
            return False
        if self._action_intent_pending_resolution_tool(state):
            return True
        latest_result = self._latest_successful_action_intent_result(state)
        if latest_result:
            if self._action_intent_needs_future_use_evidence(state=state, result=latest_result):
                return True
            if self._action_intent_pair_needs_outcome_resolution(state=state, result=latest_result):
                return True
            if bool(latest_result.get("need_future_evidence")) or bool(latest_result.get("ambiguity")):
                return True
        if any(
            isinstance(item, str) and item.startswith("action_intent_need_future_evidence=1")
            for item in list(getattr(state, "working_memory", [])) + list(getattr(state, "evidence_bundle", []))
        ):
            return True
        return self._action_intent_prefers_followup_state_change_only(state)

    def _action_intent_candidate_inference_frames(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        require_current_scope: bool = True,
    ) -> list[str]:
        if not self._is_action_intent_task(state):
            return []
        latest_result = self._latest_successful_action_intent_result(state)
        include_followup = (
            bool(self._action_intent_pending_resolution_tool(state))
            or self._action_intent_followup_attempt_count(state) > 0
            or self._action_intent_has_transition_followup_frames(state)
            or self._action_intent_has_peak_guided_followup_frames(state)
            or bool(self._latest_action_intent_timeline_review(state))
            or self._action_intent_requires_followup(state, result=latest_result if latest_result else None)
        )
        return self._select_action_intent_frames(
            state,
            hints,
            limit=8 if include_followup else 4,
            include_followup=include_followup,
            require_current_scope=require_current_scope,
        )

    def _action_intent_initial_followup_budget(self, state: AgentState) -> int:
        return 2 if self._action_intent_prefers_result_driven_followup(state) else 1

    def _action_intent_should_preempt_initial_followup_with_transition(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if not isinstance(result, dict):
            return False
        if self._action_intent_followup_attempt_count(state) > 0:
            return False
        if self._action_intent_has_transition_followup_frames(state):
            return False
        if self._action_intent_transition_probe_window(state=state, hints=hints, result=result) is None:
            return False
        needed_profile = self._action_intent_needed_observation_profile(state=state, result=result)
        if not needed_profile["prefer_receptacle_outcome"]:
            question_text = str(getattr(state, "question", "") or "").lower()
            support_text = self._action_intent_result_support_text(result)
            needed_observation = str(result.get("needed_observation") or "").lower()
            combined_text = f"{support_text} {needed_observation}"
            state_change_markers = (
                "display",
                "turns on",
                "turned on",
                "turn on",
                "reset",
                "resets",
                "zero",
                "0",
                "tare",
                "readout",
                "显示",
                "归零",
                "开机",
            )
            towel_like = any(
                token in question_text
                for token in ("paper towel", "tea towel", "dish cloth", "cloth", "towel", "napkin", "hand towel")
            )
            towel_transport_action = towel_like and any(
                token in question_text
                for token in ("<pick up ", "<grab ", "<lift ", "<take ", "<move ", "<shift ")
            )
            candidate_indices = self._latest_action_intent_candidate_indices(state, result=result)
            profile = action_intent_conflict_profile(
                question=str(getattr(state, "question", "") or ""),
                choices=[str(choice) for choice in getattr(state, "choices", [])],
                indices=candidate_indices if len(candidate_indices) >= 2 else None,
            )
            active_categories = set(profile["active_categories"])
            transport_vs_use = towel_transport_action and "clean_dry" in active_categories and bool(
                active_categories & {"generic_relocation", "final_place_return"}
            )
            explicit_need_more = (
                bool(result.get("need_more_evidence"))
                or bool(result.get("ambiguity"))
                or bool(result.get("need_future_evidence"))
                or "whether" in needed_observation
                or "是否" in needed_observation
            )
            transport_markers = (
                "applied to the hands",
                "applied to the hand",
                "hand area",
                "hands after pickup",
                "wipe",
                "wiping",
                "dry hand",
                "dry hands",
                "set down",
                "put away",
                "placed on the counter",
                "placed on the worktop",
                "later use",
                "counter",
                "worktop",
                "left on the side",
                "手上",
                "擦手",
                "擦台面",
                "放到台面",
                "放回",
            )
            if explicit_need_more and needed_profile["prefer_state_change_only"] and any(
                marker in combined_text for marker in state_change_markers
            ):
                return True
            if explicit_need_more and transport_vs_use and any(marker in combined_text for marker in transport_markers):
                return True
            return False
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        support_text = self._action_intent_result_support_text(result)
        needed_observation = str(result.get("needed_observation") or "").lower()
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
        receptacle_markers = (
            "sink",
            "pan",
            "pot",
            "bowl",
            "container",
            "crumb",
            "residue",
            "drop",
            "fall",
            "release",
            "碎屑",
            "残渣",
            "掉",
            "落",
        )
        if any(marker in needed_observation for marker in receptacle_markers):
            return True
        return (
            (
                bool(result.get("need_more_evidence"))
                or bool(result.get("ambiguity"))
                or bool(result.get("need_future_evidence"))
            )
            and any(marker in support_text for marker in uncertainty_markers)
        )

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
        question_text = str(getattr(state, "question", "") or "").lower()
        text = " ".join(
            str(result.get(key) or "")
            for key in ("reason", "decisive_observation", "needed_observation", "answer")
        ).lower()
        if any(token in question_text for token in ("<tap kitchen scale>", "tap kitchen scale")) and any(
            term in text
            for term in (
                "before the tap",
                "before tapping",
                "already on",
                "already lit",
                "display was already lit",
                "scale was already on",
                "container was already on the scale",
                "container already on the scale",
                "bowl already on the scale",
                "按之前",
                "点击前",
                "已经亮",
                "已经开机",
                "容器已经在秤上",
                "碗已经在秤上",
            )
        ):
            return True
        if not self._action_intent_needs_precondition_context(state=state, result=result):
            return False
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
            "already on",
            "already lit",
            "before the tap",
            "before tapping",
            "display was already lit",
            "scale was already on",
            "container was already on the scale",
            "container already on the scale",
            "container on the scale before the tap",
            "bowl already on the scale",
            "动作前",
            "按之前",
            "点击前",
            "已经亮",
            "已经开机",
            "容器已经在秤上",
            "碗已经在秤上",
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
            or self._action_intent_result_is_workspace_or_final_placement_close_call(state=state, result=result)
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
        rescued_indices = self._action_intent_semantic_rescue_candidate_indices(
            state=state,
            indices=indices,
            result=result,
        )
        if len(rescued_indices) >= 2:
            indices = rescued_indices
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
        question = str(getattr(state, "question", "") or "").lower()
        categories_by_index = selected_choice_categories(choices)
        preferred_pairs: tuple[tuple[str, str], ...] = (
            ("access_retrieve", "space_clear"),
            ("access_retrieve", "final_place_return"),
            ("space_clear", "final_place_return"),
            ("safety_avoid", "space_clear"),
            ("safety_avoid", "access_retrieve"),
        )
        if any(token in question for token in ("<flip ", "<turn ", "<shake ", "<tilt ", "<tip ", "<tap ", "<hit ", "<knock ")):
            preferred_pairs = (
                ("clean_dry", "transfer_contents"),
                ("open_close", "measure_weigh"),
            ) + preferred_pairs
        if any(token in question for token in ("towel", "cloth", "napkin", "paper towel", "tea towel", "dish cloth", "hand towel")):
            if any(token in question for token in ("<pick up ", "<grab ", "<lift ", "<take ", "<move ", "<shift ")):
                preferred_pairs = (
                    ("clean_dry", "generic_relocation"),
                    ("clean_dry", "final_place_return"),
                ) + preferred_pairs
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
        if isinstance(result, dict) and (
            self._action_intent_result_is_weak_generic_claim(state=state, result=result)
            or self._action_intent_result_is_workspace_or_final_placement_close_call(state=state, result=result)
        ):
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

    def _build_initial_action_intent_transition_probe_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state):
            return None
        strict_visual_disambiguation = action_intent_requires_strict_visual_disambiguation(
            question=str(getattr(state, "question", "") or ""),
            choices=[str(choice) for choice in getattr(state, "choices", [])],
            indices=None,
        )
        initial_mixed_horizon = self._action_intent_initial_pair_spans_immediate_and_later_outcomes(state)
        if not strict_visual_disambiguation and not initial_mixed_horizon:
            return None
        probe_window = self._action_intent_transition_probe_window(state=state, hints=hints, result=None)
        if probe_window is None:
            return None
        start_time, end_time, stride_s, max_frames = probe_window
        thought = (
            "why 题一开始就属于严格视觉消歧场景；先围绕动作尾部和紧随其后的短窗口做更密的关键帧搜索，优先抓决定性结果帧，而不是只抽静态动作片段。"
            if strict_visual_disambiguation
            else "why 题一开始就同时包含立刻微结果和稍后用途/归位冲突；先用 mixed-horizon 的 transition probe 同时覆盖近窗与稍后结果，再进入专用动作目的判断。"
        )
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
        initial_transition_probe = self._build_initial_action_intent_transition_probe_decision(
            state=state,
            hints=hints,
        )
        if initial_transition_probe is not None:
            return initial_transition_probe
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

    def _build_action_intent_strict_text_fallback_recovery_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
    ) -> PlannerDecision | None:
        action_frames = self._select_action_intent_frames(
            state,
            hints,
            limit=8,
            require_current_scope=True,
        )
        if action_frames:
            missing_followup = self._build_action_intent_missing_post_action_followup_decision(
                state=state,
                hints=hints,
                action_frames=action_frames,
                focus="strict_visual_disambiguation_after_text_fallback",
            )
            if missing_followup is not None:
                return missing_followup
        specialized_resolution = self._build_action_intent_specialized_resolution_before_text_fallback(
            state=state,
            hints=hints,
        )
        if specialized_resolution is not None:
            return specialized_resolution
        spatial_probe = self._build_action_intent_spatial_probe_decision(
            state=state,
            hints=hints,
            result=None,
            thought="why 题属于高歧义动作理解桶，文本 fallback 不能直接收口；先继续补空间/后续证据再裁决。",
        )
        if spatial_probe is not None:
            return spatial_probe
        return self._build_action_intent_specialized_recovery_decision(
            state=state,
            hints=hints,
            thought="why 题属于高歧义动作理解桶，文本 fallback 不能直接收口；回到当前题专用动作目的判断继续找证据。",
        )

    def _action_intent_current_scope_artifact_prefixes(self, state: AgentState) -> tuple[str, ...]:
        task_tag = str(getattr(state, "task_family", "") or "").lower()
        if not task_tag:
            return ()
        return (
            f"{task_tag}_segment",
            f"{task_tag}_precontext",
            f"{task_tag}_followup",
            f"{task_tag}_followup_transition",
            f"{task_tag}_followup_peaks",
            f"{task_tag}_followup_ext",
            f"{task_tag}_recover_frames",
        )

    def _build_action_intent_evidence_first_recovery_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
        failed_tools: set[str] | None = None,
        ineffective_tools: set[str] | None = None,
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state):
            return None
        failed = failed_tools or set()
        ineffective = ineffective_tools or set()
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
        if not action_frames:
            artifact_prefixes = self._action_intent_current_scope_artifact_prefixes(state)
            if (
                "retrieve_cached_artifacts" not in used_tools
                and "retrieve_cached_artifacts" not in failed
                and "retrieve_cached_artifacts" not in ineffective
                and self._task_has_reusable_artifacts(state, prefixes=artifact_prefixes)
            ):
                precondition_margin = 6.0 if self._action_intent_needs_precondition_context(state=state, result=None) else 2.0
                followup_margin = 8.5 if self._action_intent_prefers_result_driven_followup(state) else 5.0
                return PlannerDecision(
                    thought="why 题当前仍缺当前题原始帧；先回收当前题 artifact，再继续补关键帧或做专用裁决，不退回 query_time。",
                    tool="retrieve_cached_artifacts",
                    args={
                        "tag_hint": f"{state.task_family}_segment",
                        "start_time": max(0.0, min(combined_times) - precondition_margin),
                        "end_time": max(combined_times) + followup_margin,
                        "limit": 8,
                    },
                )
            return self._build_action_intent_specialized_recovery_decision(
                state=state,
                hints=hints,
                thought="why 题当前仍缺当前题原始帧；先恢复当前题时间窗关键帧，不退回 query_time 或文本猜测。",
            )
        if (
            self._action_intent_needs_precondition_context(state=state, result=None)
            and not self._action_intent_has_precondition_frames(state=state, hints=hints)
        ):
            precondition = self._build_action_intent_precondition_sampling_decision(
                state=state,
                hints=hints,
                focus="evidence_first_recovery_precondition",
            )
            if precondition is not None:
                return precondition
        missing_followup = self._build_action_intent_missing_post_action_followup_decision(
            state=state,
            hints=hints,
            action_frames=action_frames,
            focus="evidence_first_recovery_missing_post_action",
        )
        if missing_followup is not None:
            return missing_followup
        latest_review = self._latest_action_intent_timeline_review(state)
        if not latest_review and self._action_intent_has_post_action_frames(state=state, hints=hints, frames=action_frames):
            image_paths = self._action_intent_timeline_review_candidate_paths(state=state, hints=hints)
            if not image_paths:
                image_paths = self._select_action_intent_frames(
                    state,
                    hints,
                    limit=8,
                    include_followup=True,
                    require_current_scope=True,
                )
            if image_paths and self._action_intent_has_post_action_frames(state=state, hints=hints, frames=image_paths):
                return PlannerDecision(
                    thought="why 题当前已有动作前后关键帧，但仍不能只凭局部瞬间定答；先做短时序证据复核，再回到因果判断。",
                    tool="inspect_visual_evidence",
                    args={
                        "prompt": self._action_intent_timeline_review_prompt(state=state),
                        "image_paths": image_paths,
                    },
                )
        if self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            long_horizon_query = self._build_action_intent_long_horizon_object_query_decision(
                state=state,
                used_tools=used_tools,
                thought="why 题近窗关键帧仍不足以区分 later use / final location；先按目标对象做全视频后续检索，再围绕它更晚的再次出现位置补帧。",
            )
            if long_horizon_query is not None:
                return long_horizon_query
            long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题之前已经检索过目标对象的后续轨迹；当前近窗证据仍不足，就继续沿更晚的目标对象出现点向后追，而不是直接收口或原地空转。",
            )
            if long_horizon_revisit is not None:
                return long_horizon_revisit
        specialized_resolution = self._build_action_intent_specialized_resolution_before_text_fallback(
            state=state,
            hints=hints,
        )
        if specialized_resolution is not None:
            return specialized_resolution
        return None

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

    def _action_intent_choice_is_generic_workspace_claim(self, choice: str) -> bool:
        text = str(choice or "").strip().lower()
        if not text:
            return False
        return any(
            token in text
            for token in (
                "to make space",
                "to make some space",
                "to create space",
                "to free up space",
                "to clear space",
                "to make room",
                "to create room",
                "to free up room",
                "to clear room",
                "to begin clearing up",
                "腾出空间",
                "让开",
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

    def _action_intent_choice_is_exact_workspace_or_downstream_candidate(self, choice: str) -> bool:
        text = str(choice or "").lower()
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

    def _action_intent_recent_workspace_or_final_placement_withheld(self, state: AgentState) -> bool:
        return any(
            isinstance(item, str)
            and item.startswith("action_intent_resolution_withheld_for_workspace_or_final_placement_claim=1")
            for item in list(getattr(state, "working_memory", []))[-12:]
        )

    def _action_intent_result_is_workspace_or_final_placement_close_call(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
    ) -> bool:
        if self._action_intent_recent_workspace_or_final_placement_withheld(state):
            return True
        if not isinstance(result, dict):
            return False
        try:
            index = int(result.get("best_index"))
        except Exception:  # noqa: BLE001
            return False
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if index < 0 or index >= len(choices):
            return False
        choice_lc = choices[index].strip().lower()
        generic_workspace = self._action_intent_choice_is_generic_workspace_claim(choice_lc)
        final_placement = self._action_intent_choice_is_final_placement_candidate(choice_lc)
        exact_workspace_or_downstream = self._action_intent_choice_is_exact_workspace_or_downstream_candidate(choice_lc)
        if not any((generic_workspace, final_placement, exact_workspace_or_downstream)):
            return False
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        text = self._action_intent_result_support_text(result)
        uncertainty_terms = (
            "unclear",
            "still unclear",
            "not visible",
            "not shown",
            "cannot tell",
            "can't tell",
            "ambiguous",
            "whether",
            "no exact next target",
            "no specific next target",
            "no single immediate next target",
            "not final placement",
            "left on the counter",
            "within reach",
            "temporarily placed",
            "merely relocated",
            "没有看到",
            "未显示",
            "不明确",
            "是否",
            "不是最终放置",
            "暂时放在",
            "放在台面",
        )
        weak_spatial_only_terms = (
            "more open",
            "extra room",
            "clears some room",
            "clear some room",
            "frees the area",
            "becomes more open",
            "changes the sink-side workspace",
            "workspace",
            "open counter space",
            "counter space",
            "sink-side area",
            "the area becomes available",
            "腾出空间",
            "更空了",
            "区域更开阔",
        )
        has_weak_support_signal = any(token in text for token in uncertainty_terms) or any(
            token in text for token in weak_spatial_only_terms
        )
        if not has_weak_support_signal:
            return False
        if generic_workspace or final_placement:
            return True
        return exact_workspace_or_downstream

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
        bias_profile = self._action_intent_timeline_review_bias_profile(state) if include_followup else {}
        if include_followup:
            needs_precontext = self._action_intent_needs_precondition_context(state=state, result=None)
            dense_near_followup = self._action_intent_prefers_dense_near_followup(state)
            followup_only_state_change = self._action_intent_prefers_followup_state_change_only(state)
            review_transition_focus = bool(
                bias_profile.get("revealed_target_retrieval")
                or bias_profile.get("revealed_slot_placement")
                or bias_profile.get("revealed_fixture_enablement")
                or (
                    bias_profile.get("hand_free_next_action")
                    and not (bias_profile.get("next_use_unclear") or bias_profile.get("final_location_unclear"))
                )
            )
            review_late_focus = bool(
                bias_profile.get("next_use_unclear") or bias_profile.get("final_location_unclear")
            )
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
            elif needed_profile["prefer_receptacle_outcome"]:
                pre_keep = 1 if needs_precontext and limit >= 6 else 0
                segment_keep = min(segment_keep, 2)
                transition_keep = min(len(followup_transition_frames), max(transition_keep, 2 if limit >= 6 else 1))
                peak_keep = min(len(followup_peak_frames), max(peak_keep, 1 if limit >= 6 else 0))
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
            if review_transition_focus and not review_late_focus:
                pre_keep = 1 if needs_precontext and limit >= 6 else 0
                segment_keep = min(segment_keep, 2)
                transition_keep = min(len(followup_transition_frames), max(transition_keep, 3 if limit >= 7 else 2))
                peak_keep = min(len(followup_peak_frames), max(peak_keep, 1 if limit >= 6 else 0))
                if bias_profile.get("revealed_slot_placement"):
                    ext_keep = 0
                    followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep - peak_keep)
                else:
                    ext_keep = min(len(followup_ext_frames), 1 if followup_ext_frames and limit >= 8 else 0)
                    followup_keep = max(1, limit - pre_keep - segment_keep - transition_keep - peak_keep - ext_keep)
            if review_late_focus:
                pre_keep = 1 if needs_precontext and limit >= 7 else 0
                segment_keep = 0 if not needs_precontext and followup_ext_frames else min(segment_keep, 1 if limit >= 6 else segment_keep)
                transition_keep = min(len(followup_transition_frames), max(transition_keep, 1 if followup_transition_frames and limit >= 7 else 0))
                peak_keep = min(len(followup_peak_frames), max(peak_keep, 1 if followup_peak_frames and limit >= 8 else peak_keep))
                ext_keep = min(len(followup_ext_frames), max(ext_keep, 3 if limit >= 6 else 1))
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
            elif bias_profile.get("revealed_slot_placement"):
                priority = ["transition", "followup", "peaks", "segment", "precontext", "ext"]
            elif (
                bias_profile.get("revealed_target_retrieval")
                or bias_profile.get("revealed_fixture_enablement")
                or (
                    bias_profile.get("hand_free_next_action")
                    and not (bias_profile.get("next_use_unclear") or bias_profile.get("final_location_unclear"))
                )
            ):
                priority = ["transition", "peaks", "followup", "segment", "precontext", "ext"]
            elif bias_profile.get("final_location_unclear"):
                priority = ["ext", "followup", "transition", "segment", "peaks", "precontext"]
            elif bias_profile.get("next_use_unclear"):
                priority = ["ext", "followup", "transition", "peaks", "segment", "precontext"]
            elif needed_profile["prefer_reveal_access"]:
                priority = ["transition", "peaks", "segment", "followup", "precontext", "ext"]
            elif needed_profile["prefer_receptacle_outcome"]:
                priority = ["transition", "peaks", "followup", "segment", "precontext", "ext"]
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
                or bool(bias_profile.get("next_use_unclear"))
                or bool(bias_profile.get("final_location_unclear"))
                or bool(bias_profile.get("revealed_target_retrieval"))
                or bool(bias_profile.get("revealed_slot_placement"))
                or bool(bias_profile.get("revealed_fixture_enablement"))
            ):
                remaining_anchor = action_end + 1.0
                if bool(bias_profile.get("next_use_unclear")) or bool(bias_profile.get("final_location_unclear")):
                    remaining_anchor = action_end + 4.0
                elif (
                    bool(bias_profile.get("revealed_target_retrieval"))
                    or bool(bias_profile.get("revealed_slot_placement"))
                    or bool(bias_profile.get("revealed_fixture_enablement"))
                ):
                    remaining_anchor = action_end + 0.35
                elif (
                    needed_profile["prefer_future_use_outcome"]
                    or needed_profile["prefer_final_placement"]
                    or needed_profile["prefer_mixed_horizon"]
                ):
                    remaining_anchor = action_end + 4.0
                elif needed_profile["prefer_reveal_access"] or needed_profile["prefer_receptacle_outcome"] or needed_profile["prefer_safety_or_spill"]:
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
            selected = [ordered[0], self._nearest_frame_to_time(ordered, anchor_time), ordered[-1]]
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

    def _action_intent_question_action_text(self, state: AgentState) -> str:
        question = str(getattr(state, "question", "") or "")
        match = re.search(r"<([^>]+)>", question)
        if match:
            return " ".join(str(match.group(1) or "").split())
        lowered = question.lower()
        marker = "performed the action"
        if marker not in lowered:
            return ""
        tail = question[lowered.index(marker) + len(marker) :].strip()
        if not tail:
            return ""
        tail = re.split(r"in video\s+\d+", tail, maxsplit=1, flags=re.IGNORECASE)[0]
        return " ".join(tail.strip(" ?.:").split())

    def _action_intent_question_object_hint(self, state: AgentState, provided_hint: Any = None) -> str:
        if provided_hint:
            return " ".join(str(provided_hint).strip().split())
        action_text = self._action_intent_question_action_text(state)
        if not action_text:
            return ""
        lowered = action_text.lower()
        prefixes = (
            "pick up ",
            "put down ",
            "turn off ",
            "turn on ",
            "switch off ",
            "switch on ",
            "move ",
            "shift ",
            "transfer ",
            "place ",
            "pick ",
            "grab ",
            "lift ",
            "take ",
            "open ",
            "close ",
            "clear ",
            "check ",
            "flip ",
            "turn ",
            "shake ",
            "stir ",
            "push ",
            "slide ",
            "tap ",
            "hit ",
            "set ",
            "put ",
            "run ",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                object_text = action_text[len(prefix) :].strip()
                return " ".join(object_text.split())
        return action_text

    def _action_intent_localization_window_from_nodes(
        self,
        *,
        state: AgentState,
        nodes: list[dict[str, Any]],
    ) -> tuple[float, float] | None:
        if not self._is_action_intent_task(state):
            return None
        timed: list[tuple[tuple[int, float, float], float, float]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            start_raw = node.get("start_time")
            end_raw = node.get("end_time")
            if start_raw is None:
                continue
            try:
                start_time = float(start_raw)
            except Exception:  # noqa: BLE001
                continue
            try:
                end_time = float(end_raw) if end_raw is not None else start_time
            except Exception:  # noqa: BLE001
                end_time = start_time
            if end_time < start_time:
                end_time = start_time
            node_type = str(node.get("node_type") or "").lower()
            if node_type in {"frame", "observation", "timeline_event"}:
                priority = 0
            elif node_type == "object_track":
                priority = 1
            elif node_type in {"segment", "activity"}:
                priority = 2
            else:
                priority = 3
            duration = max(0.0, end_time - start_time)
            timed.append(((priority, duration if priority == 0 else min(duration, 8.0), start_time), start_time, end_time))
        if not timed:
            return None
        _, start_time, end_time = min(timed, key=lambda item: item[0])
        duration = max(0.0, end_time - start_time)
        if duration <= 0.2:
            return (max(0.0, start_time - 1.0), start_time + 1.8)
        if duration <= 4.0:
            return (max(0.0, start_time - 0.6), end_time + 0.9)
        focus_end = min(end_time, start_time + 4.5)
        return (max(0.0, start_time - 0.5), focus_end)

    def _action_intent_step_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        combined_times: list[float],
        object_hint: Any,
        last_result: dict[str, Any],
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state) or combined_times:
            return None
        localization_keyword = self._action_intent_question_object_hint(state, object_hint)
        if state.current_step <= 1 and localization_keyword and "query_event" not in used_tools:
            return PlannerDecision(
                thought="why 题当前没有显式时间点；先按题目里的动作对象做结构化定位，缩小候选时间段，再抽关键帧判断动作目的。",
                tool="query_event",
                args={
                    "event_types": ["frame", "observation", "timeline_event", "object_track", "segment", "activity"],
                    "keyword": localization_keyword,
                    "start_time": None,
                    "end_time": None,
                    "limit": 12,
                },
            )
        if state.current_step == 2:
            nodes = last_result.get("nodes", []) if isinstance(last_result, dict) else []
            window = self._action_intent_localization_window_from_nodes(state=state, nodes=nodes if isinstance(nodes, list) else [])
            if window is not None:
                start_time, end_time = window
                return PlannerDecision(
                    thought="why 题已经定位到动作对象附近的候选时刻；先围绕最像动作发生点的短窗口抽关键帧，再进入动作目的判断。",
                    tool="extract_frames_for_range",
                    args={
                        "start_time": start_time,
                        "end_time": end_time,
                        "stride_s": max(0.35, (end_time - start_time) / 4),
                        "max_frames": 4,
                        "tag": f"{state.task_family}_segment",
                    },
                )
        return None

    def _action_intent_prefers_long_horizon_object_retrieval(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None = None,
    ) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if self._action_intent_followup_attempt_count(state) < 1 and not self._latest_action_intent_timeline_review(state):
            return False
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        needed_profile = self._action_intent_needed_observation_profile(state=state, result=result)
        if bias_profile["next_use_unclear"] or bias_profile["final_location_unclear"]:
            return True
        if needed_profile["prefer_future_use_outcome"] or needed_profile["prefer_final_placement"]:
            return True
        return self._action_intent_pair_spans_immediate_and_later_outcomes(state=state, result=result)

    def _build_action_intent_long_horizon_object_query_decision(
        self,
        *,
        state: AgentState,
        used_tools: list[str],
        thought: str,
        object_hint: Any = None,
    ) -> PlannerDecision | None:
        if not self._is_action_intent_task(state) or "query_object" in used_tools:
            return None
        query = self._action_intent_question_object_hint(state, object_hint)
        if not query:
            return None
        return PlannerDecision(
            thought=thought,
            tool="query_object",
            args={"query": query, "limit": 24},
        )

    def _latest_action_intent_long_horizon_nodes(
        self,
        state: AgentState,
        *,
        object_hint: Any = None,
    ) -> list[dict[str, Any]]:
        if not self._is_action_intent_task(state):
            return []
        target_query = self._action_intent_question_object_hint(state, object_hint).strip().lower()
        for entry in reversed(getattr(state, "tool_trace", [])):
            if not isinstance(entry, dict) or entry.get("tool") != "query_object":
                continue
            args = entry.get("args") or {}
            if not isinstance(args, dict):
                continue
            query = str(args.get("query") or "").strip().lower()
            if target_query and query and target_query not in query and query not in target_query:
                continue
            payload = entry.get("raw_result")
            if not isinstance(payload, dict):
                continue
            nodes = payload.get("nodes")
            if isinstance(nodes, list):
                return [node for node in nodes if isinstance(node, dict)]
        return []

    def _action_intent_long_horizon_target_tokens(self, state: AgentState, *, object_hint: Any = None) -> list[str]:
        query = self._action_intent_question_object_hint(state, object_hint)
        return [token for token in re.split(r"[\s_/:-]+", query.lower()) if token]

    def _action_intent_long_horizon_prefers_latest_candidate(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        if bias_profile["final_location_unclear"]:
            return True
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        return bool(needed_profile["prefer_final_placement"]) and not bool(needed_profile["prefer_future_use_outcome"])

    def _action_intent_long_horizon_node_match_tier(
        self,
        *,
        state: AgentState,
        node: dict[str, Any],
        object_hint: Any = None,
    ) -> int | None:
        tokens = self._action_intent_long_horizon_target_tokens(state, object_hint=object_hint)
        if not tokens:
            return 2
        attrs = node.get("attributes") or {}
        direct_parts = [
            str(node.get("object_name") or ""),
            str(attrs.get("object_name") or ""),
            str(node.get("label") or ""),
            str(attrs.get("label") or ""),
        ]
        direct_text = " ".join(part.strip().lower() for part in direct_parts if part).strip()
        summary_text = " ".join(
            str(part).strip().lower()
            for part in (
                attrs.get("summary"),
                attrs.get("payload_json"),
                node.get("summary"),
            )
            if part
        ).strip()
        if direct_text and all(token in direct_text for token in tokens):
            return 0
        if (direct_text or summary_text) and all(token in f"{direct_text} {summary_text}" for token in tokens):
            return 1
        return None

    def _action_intent_select_long_horizon_node(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        nodes: list[dict[str, Any]],
        min_start_time: float | None = None,
        object_hint: Any = None,
    ) -> tuple[dict[str, Any], float, float] | None:
        if not self._is_action_intent_task(state):
            return None
        anchor_times = sorted([float(value) for value in hints.get("times") or []] + [float(value) for value in hints.get("input_times") or []])
        if not anchor_times:
            anchor_times = self._action_intent_anchor_times(state)
        action_end = max(anchor_times) if anchor_times else None
        if action_end is None:
            return None
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        lower_bound = max(action_end + 0.35, (latest_followup_end or action_end) + 0.35)
        if min_start_time is not None:
            lower_bound = max(lower_bound, float(min_start_time))
        prefer_latest = self._action_intent_long_horizon_prefers_latest_candidate(state)
        candidates: list[tuple[tuple[int, int, float], dict[str, Any], float, float]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            start_raw = node.get("start_time")
            end_raw = node.get("end_time")
            if start_raw is None:
                continue
            try:
                start_time = float(start_raw)
            except Exception:  # noqa: BLE001
                continue
            try:
                end_time = float(end_raw) if end_raw is not None else start_time
            except Exception:  # noqa: BLE001
                end_time = start_time
            if end_time < start_time:
                end_time = start_time
            if min_start_time is not None and start_time < float(min_start_time):
                continue
            if end_time <= lower_bound:
                continue
            match_tier = self._action_intent_long_horizon_node_match_tier(state=state, node=node, object_hint=object_hint)
            if match_tier is None:
                continue
            node_type = str(node.get("node_type") or "").lower()
            if node_type in {"object_track", "frame", "observation", "timeline_event"}:
                priority = 0
            elif node_type in {"segment", "activity"}:
                priority = 1
            else:
                priority = 2
            time_key = -start_time if prefer_latest else start_time
            candidates.append(((match_tier, priority, time_key), node, start_time, end_time))
        if not candidates:
            return None
        _, node, start_time, end_time = min(candidates, key=lambda item: item[0])
        return node, start_time, end_time

    def _action_intent_long_horizon_window_from_nodes(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        nodes: list[dict[str, Any]],
    ) -> tuple[float, float] | None:
        selected = self._action_intent_select_long_horizon_node(state=state, hints=hints, nodes=nodes)
        if selected is None:
            return None
        _node, start_time, end_time = selected
        duration = max(0.0, end_time - start_time)
        if duration <= 0.25:
            return (max(0.0, start_time - 0.5), start_time + 2.0)
        if duration <= 4.0:
            return (max(0.0, start_time - 0.4), end_time + 1.0)
        return (max(0.0, start_time - 0.4), min(end_time, start_time + 4.8))

    def _build_action_intent_long_horizon_spatial_probe_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        used_tools: list[str],
        nodes: list[dict[str, Any]],
    ) -> PlannerDecision | None:
        if "query_spatial_context" in used_tools:
            return None
        selected = self._action_intent_select_long_horizon_node(state=state, hints=hints, nodes=nodes)
        if selected is None:
            return None
        _node, start_time, end_time = selected
        anchor_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought="why 题已定位到目标对象在更晚时刻的再次出现；先补这一下的空间关系，再决定是否继续抽长时域关键帧。",
            tool="query_spatial_context",
            args={
                "time_s": anchor_time,
                "object_name": self._action_intent_question_object_hint(state),
                "limit": 16,
            },
        )

    def _build_action_intent_long_horizon_sampling_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        nodes: list[dict[str, Any]],
    ) -> PlannerDecision | None:
        window = self._action_intent_long_horizon_window_from_nodes(state=state, hints=hints, nodes=nodes)
        if window is None:
            return None
        start_time, end_time = window
        attempt_count = self._action_intent_followup_attempt_count(state)
        return PlannerDecision(
            thought="why 题近窗证据仍不能排除多个 later-use / final-location 解释；按目标对象在更后时刻的再次出现位置补一段长时域关键帧。",
            tool="sample_sparse_frames",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "sample_count": 4,
                "tag": f"{state.task_family}_followup_ext{attempt_count + 1}",
            },
        )

    def _build_action_intent_cached_long_horizon_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
        after_time: float | None = None,
    ) -> PlannerDecision | None:
        if not self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            return None
        nodes = self._latest_action_intent_long_horizon_nodes(state)
        if not nodes:
            return None
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
        )
        if selected is None:
            return None
        _node, start_time, end_time = selected
        anchor_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": anchor_time,
                "object_name": self._action_intent_question_object_hint(state),
                "limit": 16,
            },
        )

    def _action_intent_spatial_target_mask_fixture(self, state: AgentState, spatial: dict[str, Any]) -> str:
        if not self._is_action_intent_task(state) or not isinstance(spatial, dict):
            return ""
        target = self._action_intent_question_object_hint(state).strip().lower()
        if not target:
            return ""
        target_tokens = [token for token in re.split(r"[\s_/:-]+", target) if token]
        if not target_tokens:
            return ""
        for item in spatial.get("object_masks") or []:
            if not isinstance(item, dict):
                continue
            object_name = str(item.get("object_name") or "").strip().lower()
            if object_name and all(token in object_name for token in target_tokens):
                return str(item.get("fixture") or "").strip()
        return ""

    def _action_intent_fixture_bucket(self, fixture: str) -> str:
        text = str(fixture or "").strip().lower()
        if not text:
            return "unknown"
        if any(token in text for token in ("fridge", "freezer", "cupboard", "cabinet", "drawer", "shelf", "rack", "pantry")):
            return "storage"
        if any(token in text for token in ("scale", "weigh")):
            return "scale"
        if any(token in text for token in ("sink", "drain", "tap", "faucet")):
            return "sink"
        if any(token in text for token in ("hob", "stove", "burner", "oven", "microwave", "airfryer", "toaster", "kettle")):
            return "appliance"
        if any(token in text for token in ("counter", "table", "board", "worktop", "surface", "island")):
            return "workspace"
        return "other"

    def _action_intent_long_horizon_spatial_context_looks_intermediate(
        self,
        *,
        state: AgentState,
        spatial: dict[str, Any],
    ) -> bool:
        if not self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            return False
        target_fixture = self._action_intent_spatial_target_mask_fixture(state, spatial)
        fixture_bucket = self._action_intent_fixture_bucket(target_fixture)
        has_target_track = False
        target = self._action_intent_question_object_hint(state).strip().lower()
        target_tokens = [token for token in re.split(r"[\s_/:-]+", target) if token]
        for item in spatial.get("object_tracks") or []:
            if not isinstance(item, dict):
                continue
            object_name = str(item.get("object_name") or "").strip().lower()
            if object_name and target_tokens and all(token in object_name for token in target_tokens):
                has_target_track = True
                break
        if not has_target_track and not target_fixture:
            return False
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        if bias_profile["final_location_unclear"]:
            return fixture_bucket in {"unknown", "workspace", "other"} or not target_fixture
        if bias_profile["next_use_unclear"]:
            return fixture_bucket in {"unknown", "workspace", "other"} or not target_fixture
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        if needed_profile["prefer_final_placement"] or needed_profile["prefer_future_use_outcome"]:
            return fixture_bucket in {"unknown", "workspace", "other"} or not target_fixture
        return False

    def _action_intent_long_horizon_anchor_node_at_time(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        anchor_time: float,
    ) -> tuple[dict[str, Any], float, float] | None:
        nodes = self._latest_action_intent_long_horizon_nodes(state)
        if not nodes:
            return None
        selected: tuple[tuple[int, float], dict[str, Any], float, float] | None = None
        for node in nodes:
            if not isinstance(node, dict):
                continue
            start_raw = node.get("start_time")
            end_raw = node.get("end_time")
            if start_raw is None:
                continue
            try:
                start_time = float(start_raw)
            except Exception:  # noqa: BLE001
                continue
            try:
                end_time = float(end_raw) if end_raw is not None else start_time
            except Exception:  # noqa: BLE001
                end_time = start_time
            if end_time < start_time:
                end_time = start_time
            if end_time + 0.25 < anchor_time or start_time - 0.25 > anchor_time:
                continue
            match_tier = self._action_intent_long_horizon_node_match_tier(state=state, node=node)
            if match_tier is None:
                continue
            distance = 0.0 if start_time <= anchor_time <= end_time else min(abs(anchor_time - start_time), abs(anchor_time - end_time))
            candidate = ((match_tier, distance), node, start_time, end_time)
            if selected is None or candidate[0] < selected[0]:
                selected = candidate
        if selected is None:
            return None
        _score, node, start_time, end_time = selected
        return node, start_time, end_time

    def _action_intent_has_later_long_horizon_node_after(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        after_time: float,
    ) -> bool:
        nodes = self._latest_action_intent_long_horizon_nodes(state)
        if not nodes:
            return False
        later = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=after_time + 0.15,
        )
        return later is not None

    def _action_intent_long_horizon_spatial_context_looks_transit_near_decisive_fixture(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        spatial: dict[str, Any],
        anchor_time: float,
    ) -> bool:
        if not self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            return False
        target_fixture = self._action_intent_spatial_target_mask_fixture(state, spatial)
        fixture_bucket = self._action_intent_fixture_bucket(target_fixture)
        if fixture_bucket not in {"sink", "appliance"}:
            return False
        anchor_node = self._action_intent_long_horizon_anchor_node_at_time(
            state=state,
            hints=hints,
            anchor_time=anchor_time,
        )
        if anchor_node is None:
            return False
        _node, start_time, end_time = anchor_node
        duration = max(0.0, end_time - start_time)
        if duration > 1.2:
            return False
        if not self._action_intent_has_later_long_horizon_node_after(
            state=state,
            hints=hints,
            after_time=end_time,
        ):
            return False
        audio_labels = " ".join(
            str(item.get("label") or item.get("event_type") or "").strip().lower()
            for item in spatial.get("audio_events") or []
            if isinstance(item, dict)
        )
        if fixture_bucket == "sink" and any(token in audio_labels for token in ("water", "tap", "sink", "pour", "drain", "liquid")):
            return False
        if fixture_bucket == "appliance" and any(token in audio_labels for token in ("door", "open", "close", "click", "microwave", "beep")):
            return False
        return True

    def _action_intent_spatial_has_storage_closure_cue(self, spatial: dict[str, Any]) -> bool:
        if not isinstance(spatial, dict):
            return False
        audio_labels = " ".join(
            str(item.get("label") or item.get("event_type") or "").strip().lower()
            for item in spatial.get("audio_events") or []
            if isinstance(item, dict)
        )
        if any(token in audio_labels for token in ("door", "close", "closed", "shut", "click", "drawer")):
            return True
        for collection_name in ("object_tracks", "object_masks"):
            for item in spatial.get(collection_name) or []:
                if not isinstance(item, dict):
                    continue
                object_name = str(item.get("object_name") or "").strip().lower()
                if any(token in object_name for token in ("door", "door handle", "drawer")):
                    return True
        return False

    def _action_intent_long_horizon_spatial_context_looks_nonexclusive_storage_anchor(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        spatial: dict[str, Any],
        anchor_time: float,
    ) -> bool:
        if not self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            return False
        target_fixture = self._action_intent_spatial_target_mask_fixture(state, spatial)
        if self._action_intent_fixture_bucket(target_fixture) != "storage":
            return False
        bias_profile = self._action_intent_timeline_review_bias_profile(state)
        needed_profile = self._action_intent_needed_observation_profile(state=state)
        if not (
            bias_profile["final_location_unclear"]
            or needed_profile["prefer_final_placement"]
            or needed_profile["prefer_future_use_outcome"]
        ):
            return False
        anchor_node = self._action_intent_long_horizon_anchor_node_at_time(
            state=state,
            hints=hints,
            anchor_time=anchor_time,
        )
        if anchor_node is None:
            return False
        _node, start_time, end_time = anchor_node
        duration = max(0.0, end_time - start_time)
        if duration > 1.5:
            return False
        if self._action_intent_spatial_has_storage_closure_cue(spatial):
            return False
        if not self._action_intent_has_later_long_horizon_node_after(
            state=state,
            hints=hints,
            after_time=end_time,
        ):
            return False
        return True

    def _latest_action_intent_target_spatial_anchor_time(self, state: AgentState) -> float | None:
        target = self._action_intent_question_object_hint(state).strip().lower()
        if not target:
            return None
        for entry in reversed(getattr(state, "tool_trace", [])):
            if not isinstance(entry, dict) or entry.get("tool") != "query_spatial_context":
                continue
            args = entry.get("args") or {}
            if not isinstance(args, dict):
                continue
            object_name = str(args.get("object_name") or "").strip().lower()
            if object_name != target:
                continue
            try:
                return float(args.get("time_s"))
            except Exception:  # noqa: BLE001
                continue
        return None

    def _action_intent_result_looks_weak_late_anchor_support(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
    ) -> bool:
        if not self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            return False
        if not isinstance(result, dict):
            return False
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        text = self._action_intent_result_support_text(result)
        if not text:
            return False
        proximity_terms = (
            "remains in hand",
            "visible in hand",
            "stays in hand",
            "stays near",
            "near the fridge",
            "near the fridge opening",
            "near the shelf",
            "near the counter",
            "near the sink",
            "near the hob",
            "within reach",
            "visible while held",
            "held near",
            "still held",
            "靠近",
            "拿在手里",
            "仍在手里",
            "附近",
        )
        uncertainty_terms = (
            "not decisively grounded",
            "still unclear",
            "it is not yet visible whether",
            "whether",
            "could still",
            "may be",
            "might be",
            "remains plausible",
            "not shown",
            "not visible",
            "unclear",
            "still contested",
            "不明确",
            "未显示",
            "可能",
            "是否",
        )
        has_proximity = any(term in text for term in proximity_terms)
        has_uncertainty = any(term in text for term in uncertainty_terms) or self._action_intent_result_has_indecisive_post_action_support(result)
        return has_proximity and has_uncertainty

    def _action_intent_result_looks_nonexclusive_concrete_late_anchor_support(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
    ) -> bool:
        if not self._action_intent_prefers_long_horizon_object_retrieval(state=state):
            return False
        if not isinstance(result, dict):
            return False
        if any(
            isinstance(item, str)
            and item.startswith("action_intent_resolution_withheld_for_nonexclusive_concrete_late_anchor=1")
            for item in list(getattr(state, "working_memory", []))[-12:]
        ):
            return True
        if self._action_intent_result_has_direct_post_action_evidence(result):
            return False
        best_index = self._coerce_choice_index(result.get("best_index"), getattr(state, "choices", []))
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return False
        if not self._action_intent_competing_pair_still_needs_disambiguation(
            state=state,
            best_index=best_index,
            competitor_index=competitor_index,
        ):
            return False
        text = self._action_intent_result_support_text(result)
        if not text:
            return False
        explicit_exclusive_terms = (
            "reads the label",
            "reading the label",
            "read the label",
            "inspects the label",
            "looks at the label",
            "read the printed text",
            "placed on the scale",
            "used on the scale",
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
            "称重",
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
            "read the bottle",
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
        nearby_without_exclusive_outcome = any(term in text for term in nearby_placement_terms)
        return label_visible_without_reading or nearby_without_exclusive_outcome

    def _build_action_intent_weak_late_anchor_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        thought: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_result_looks_weak_late_anchor_support(state=state, result=result):
            return None
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        return self._build_action_intent_cached_long_horizon_revisit_decision(
            state=state,
            hints=hints,
            thought=thought,
            after_time=after_time,
        )

    def _build_action_intent_nonexclusive_concrete_late_anchor_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        thought: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_result_looks_nonexclusive_concrete_late_anchor_support(state=state, result=result):
            return None
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        return self._build_action_intent_cached_long_horizon_revisit_decision(
            state=state,
            hints=hints,
            thought=thought,
            after_time=after_time,
        )

    def _build_action_intent_needed_observation_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_needed_observation_target_hint(state=state, result=result)
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 `needed_observation` 明确点名的判别目标 `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 `needed_observation` 指向的判别目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_needed_observation_relation_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_needed_observation_relation_hint(state=state, result=result)
        if hint is None:
            return None
        target_name, _target_kind, relation_name = hint
        action_object = self._action_intent_question_object_hint(state)
        if not action_object:
            return None
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=action_object)
        selected = None
        if nodes:
            selected = self._action_intent_select_long_horizon_node(
                state=state,
                hints=hints,
                nodes=nodes,
                min_start_time=min_start_time,
                object_hint=action_object,
            )
        if selected is None:
            target_nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=target_name)
            if target_nodes:
                selected = self._action_intent_select_long_horizon_node(
                    state=state,
                    hints=hints,
                    nodes=target_nodes,
                    min_start_time=min_start_time,
                    object_hint=target_name,
                )
        if selected is None:
            query_name = target_name if target_name and target_name != action_object else action_object
            return PlannerDecision(
                thought=(
                    f"{thought} 当前还没有足够晚的轨迹锚点，先重新定位 `{query_name}` 的更晚轨迹，"
                    f"确认 `{action_object}` 与判别目标的关系是否真的变成了 `{relation_name}`。"
                ),
                tool="query_object",
                args={"query": query_name, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": action_object,
                "limit": 18,
            },
        )

    def _action_intent_recent_later_outcome_finalize_withheld_marker(self, state: AgentState) -> str:
        recent = list(getattr(state, "working_memory", []))[-16:]
        marker_prefixes = (
            "action_intent_resolution_withheld_for_nonexclusive_concrete_late_anchor=1",
            "action_intent_resolution_withheld_for_timeline_review_bias_gap=1",
            "action_intent_resolution_withheld_for_workspace_or_final_placement_claim=1",
            "action_intent_resolution_withheld_for_generic_hand_free_enablement=1",
            "action_intent_resolution_withheld_for_generic_access_or_space_enablement=1",
            "action_intent_resolution_withheld_for_generic_relocation_or_storage_enablement=1",
        )
        for item in reversed(recent):
            if not isinstance(item, str):
                continue
            for prefix in marker_prefixes:
                if item.startswith(prefix):
                    return prefix
        return ""

    def _action_intent_recent_generic_hand_free_finalize_withheld_hint(self, state: AgentState) -> tuple[str, str] | None:
        recent = list(getattr(state, "working_memory", []))[-16:]
        prefix = "action_intent_resolution_withheld_for_generic_hand_free_enablement=1"
        for item in reversed(recent):
            if not isinstance(item, str) or not item.startswith(prefix):
                continue
            marker_match = re.search(r"\btarget=(.+?)\s+kind=(object|fixture)\b", item)
            target = str(marker_match.group(1) or "").strip() if marker_match else ""
            kind = str(marker_match.group(2) or "").strip() if marker_match else ""
            if target and kind:
                return target, kind
        return None

    def _action_intent_recent_generic_access_or_space_finalize_withheld_hint(self, state: AgentState) -> tuple[str, str] | None:
        recent = list(getattr(state, "working_memory", []))[-16:]
        prefix = "action_intent_resolution_withheld_for_generic_access_or_space_enablement=1"
        for item in reversed(recent):
            if not isinstance(item, str) or not item.startswith(prefix):
                continue
            marker_match = re.search(r"\btarget=(.+?)\s+kind=(object|fixture)\b", item)
            target = str(marker_match.group(1) or "").strip() if marker_match else ""
            kind = str(marker_match.group(2) or "").strip() if marker_match else ""
            if target and kind:
                return target, kind
        return None

    def _action_intent_recent_generic_relocation_or_storage_finalize_withheld_hint(
        self,
        state: AgentState,
    ) -> tuple[str, str] | None:
        recent = list(getattr(state, "working_memory", []))[-16:]
        prefix = "action_intent_resolution_withheld_for_generic_relocation_or_storage_enablement=1"
        for item in reversed(recent):
            if not isinstance(item, str) or not item.startswith(prefix):
                continue
            marker_match = re.search(r"\btarget=(.+?)\s+kind=(object|fixture)\b", item)
            target = str(marker_match.group(1) or "").strip() if marker_match else ""
            kind = str(marker_match.group(2) or "").strip() if marker_match else ""
            if target and kind:
                return target, kind
        return None

    def _action_intent_recent_mixed_horizon_later_target_withheld_hint(
        self,
        state: AgentState,
    ) -> tuple[str, str] | None:
        recent = list(getattr(state, "working_memory", []))[-16:]
        prefix = "action_intent_resolution_withheld_for_mixed_horizon_later_target=1"
        for item in reversed(recent):
            if not isinstance(item, str) or not item.startswith(prefix):
                continue
            marker_match = re.search(r"\btarget=(.+?)\s+kind=(object|fixture)\b", item)
            target = str(marker_match.group(1) or "").strip() if marker_match else ""
            kind = str(marker_match.group(2) or "").strip() if marker_match else ""
            if target and kind:
                return target, kind
        return None

    def _action_intent_recent_unresolved_rerank_withheld_reason(self, state: AgentState) -> str:
        recent = list(getattr(state, "working_memory", []))[-16:]
        for item in reversed(recent):
            if not isinstance(item, str) or not item.startswith("action_intent_unresolved_rerank_withheld"):
                continue
            match = re.search(r"reason=([a-z0-9_,.-]+)", item)
            if match:
                return str(match.group(1) or "").strip().lower()
        return ""

    def _action_intent_unresolved_rerank_reason_prefers_later_outcome_revisit(self, state: AgentState) -> bool:
        reason = self._action_intent_recent_unresolved_rerank_withheld_reason(state)
        if not reason:
            return False
        later_outcome_gaps = (
            "timeline_review_final_location_gap",
            "timeline_review_next_use_gap",
            "timeline_review_revealed_slot_gap",
            "missing_later_outcome_evidence",
        )
        return any(gap in reason for gap in later_outcome_gaps)

    def _action_intent_choice_target_object_candidates(self, *, choice: str, action_object: str) -> list[str]:
        choice_lc = str(choice or "").lower()
        action_object_tokens = {token for token in re.split(r"[^a-z0-9]+", str(action_object or "").lower()) if token}
        vocabulary = (
            "saucepan",
            "spatula",
            "microwave",
            "dishwasher",
            "cupboard",
            "container",
            "colander",
            "faucet",
            "whisk",
            "knife",
            "spoon",
            "bottle",
            "sponge",
            "brush",
            "cloth",
            "towel",
            "cover",
            "bowl",
            "plate",
            "tray",
            "glass",
            "scale",
            "fridge",
            "drawer",
            "fork",
            "rack",
            "door",
            "sink",
            "oven",
            "hob",
            "jar",
            "lid",
            "bin",
            "cup",
            "pot",
            "pan",
            "tap",
        )
        target_tokens = []
        for token in vocabulary:
            if token in action_object_tokens:
                continue
            if not re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", choice_lc):
                continue
            target_tokens.append(token)
        seen: set[str] = set()
        ordered: list[str] = []
        for token in target_tokens:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _action_intent_choice_fixture_target_candidates(self, *, choice: str, action_object: str) -> list[str]:
        fixture_targets = {
            "tap",
            "faucet",
            "scale",
            "sink",
            "bin",
            "hob",
            "microwave",
            "oven",
            "fridge",
            "door",
            "drawer",
            "cupboard",
            "rack",
            "dishwasher",
        }
        return [
            token
            for token in self._action_intent_choice_target_object_candidates(choice=choice, action_object=action_object)
            if token in fixture_targets
        ]

    def _action_intent_later_outcome_target_hint(
        self,
        *,
        choice: str,
        action_object: str,
        categories: set[str],
        evidence_text: str,
    ) -> tuple[str, str] | None:
        fixture_targets = self._action_intent_choice_fixture_target_candidates(choice=choice, action_object=action_object)
        if fixture_targets:
            return fixture_targets[0], "fixture"
        object_targets = [
            token
            for token in self._action_intent_choice_target_object_candidates(choice=choice, action_object=action_object)
            if token not in set(fixture_targets)
        ]
        if object_targets:
            return object_targets[0], "object"
        if "measure_weigh" in categories:
            return "scale", "fixture"
        evidence_lc = str(evidence_text or "").strip().lower()
        if "final_place_return" in categories:
            for token in ("fridge", "drawer", "cupboard", "rack", "dishwasher", "shelf"):
                if token in evidence_lc:
                    return token, ("object" if token == "shelf" else "fixture")
        if categories & {"transfer_contents", "serve_consume", "discard", "food_prep", "clean_dry"}:
            for token, kind in (
                ("sink", "fixture"),
                ("bin", "fixture"),
                ("bowl", "object"),
                ("plate", "object"),
                ("tray", "object"),
                ("pan", "object"),
                ("pot", "object"),
                ("saucepan", "object"),
                ("cup", "object"),
                ("glass", "object"),
                ("jar", "object"),
                ("colander", "object"),
                ("container", "object"),
            ):
                if token in evidence_lc:
                    return token, kind
        return None

    def _action_intent_choice_is_same_object_active_use(self, *, choice: str, action_object: str) -> bool:
        action_object_lc = str(action_object or "").strip().lower()
        if not action_object_lc:
            return False
        choice_lc = str(choice or "").strip().lower()
        object_tokens = [token for token in re.split(r"[^a-z0-9]+", action_object_lc) if token]
        same_object_component_reference = any(token in choice_lc for token in ("lid", "cover", "cap", "top"))
        component_bearing_objects = (
            "bottle",
            "jar",
            "container",
            "cup",
            "mug",
            "tin",
            "can",
            "tupperware",
            "shaker",
            "thermos",
            "flask",
            "pot",
            "pan",
            "saucepan",
            "blender",
            "box",
        )
        can_reference_same_object_component = any(token in action_object_lc for token in component_bearing_objects)
        if object_tokens and not all(token in choice_lc for token in object_tokens):
            if not (same_object_component_reference and can_reference_same_object_component):
                return False
        return any(
            token in choice_lc
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
                "cover",
                "replace",
                "fit",
                "unscrew",
                "pry",
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

    def _action_intent_unresolved_rerank_downstream_object_hint(self, state: AgentState) -> str:
        reason = self._action_intent_recent_unresolved_rerank_withheld_reason(state)
        if not reason or not any(
            gap in reason for gap in ("timeline_review_revealed_slot_gap", "timeline_review_revealed_target_gap")
        ):
            return ""
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return ""
        _tool_name, payload = latest
        best_index = self._coerce_choice_index(payload.get("best_index"), getattr(state, "choices", []))
        if best_index is None:
            return ""
        choice = str(getattr(state, "choices", [])[best_index])
        action_object = self._action_intent_question_object_hint(state)
        fixture_only = {"slot", "rack", "sink", "tap", "faucet", "fridge", "door", "drawer", "cupboard", "dishwasher"}
        for token in self._action_intent_choice_target_object_candidates(choice=choice, action_object=action_object):
            if token in fixture_only:
                continue
            return token
        return ""

    def _action_intent_unresolved_rerank_downstream_fixture_hint(self, state: AgentState) -> str:
        reason = self._action_intent_recent_unresolved_rerank_withheld_reason(state)
        if not reason or "timeline_review_hand_free_or_fixture_gap" not in reason:
            return ""
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return ""
        _tool_name, payload = latest
        best_index = self._coerce_choice_index(payload.get("best_index"), getattr(state, "choices", []))
        if best_index is None:
            return ""
        choice = str(getattr(state, "choices", [])[best_index])
        action_object = self._action_intent_question_object_hint(state)
        for token in self._action_intent_choice_fixture_target_candidates(choice=choice, action_object=action_object):
            return token
        return ""

    def _action_intent_unresolved_rerank_mixed_horizon_later_target_hint(
        self,
        state: AgentState,
    ) -> tuple[str, str] | None:
        reason = self._action_intent_recent_unresolved_rerank_withheld_reason(state)
        if not reason or not any(
            gap in reason
            for gap in (
                "missing_later_outcome_evidence",
                "timeline_review_next_use_gap",
                "timeline_review_final_location_gap",
            )
        ):
            return None
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return None
        _tool_name, payload = latest
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_index = self._coerce_choice_index(payload.get("best_index"), choices)
        competitor_index = self._action_intent_competing_candidate_index(payload, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return None
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
        best_choice = choices[best_index]
        competitor_choice = choices[competitor_index]
        best_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories)
        competitor_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(
            competitor_choice,
            competitor_categories,
        )
        later_index: int | None = None
        later_categories: set[str] = set()
        later_choice = ""
        if best_is_immediate and competitor_categories & later_outcome_categories:
            later_index = competitor_index
            later_categories = competitor_categories
            later_choice = competitor_choice
        elif competitor_is_immediate and best_categories & later_outcome_categories:
            later_index = best_index
            later_categories = best_categories
            later_choice = best_choice
        if later_index is None:
            return None
        action_object = self._action_intent_question_object_hint(state)
        combined_text = later_choice.lower()
        for item in payload.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index != later_index:
                continue
            combined_text = f"{combined_text} {str(item.get('support') or '').lower()} {str(item.get('contradiction') or '').lower()}"
            break
        return self._action_intent_later_outcome_target_hint(
            choice=later_choice,
            action_object=action_object,
            categories=later_categories,
            evidence_text=combined_text,
        )

    def _action_intent_verifier_blocked_mixed_horizon_later_target_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> tuple[str, str] | None:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return None
        if blocker_hint not in {"post_action_evidence", "future_use_close_call", "pairwise_close_call"}:
            return None
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_index = self._coerce_choice_index(result.get("best_index"), choices)
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return None
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
        best_choice = choices[best_index]
        competitor_choice = choices[competitor_index]
        best_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(best_choice, best_categories)
        competitor_is_immediate = self._action_intent_choice_is_immediate_micro_outcome_candidate(
            competitor_choice,
            competitor_categories,
        )
        later_index: int | None = None
        later_categories: set[str] = set()
        later_choice = ""
        if best_is_immediate and competitor_categories & later_outcome_categories:
            later_index = competitor_index
            later_categories = competitor_categories
            later_choice = competitor_choice
        elif competitor_is_immediate and best_categories & later_outcome_categories:
            later_index = best_index
            later_categories = best_categories
            later_choice = best_choice
        if later_index is None:
            return None
        action_object = self._action_intent_question_object_hint(state)
        reason_text = str(result.get("reason") or "").lower()
        needed_observation_text = str(result.get("needed_observation") or "").lower()
        combined_text = f"{later_choice.lower()} {reason_text} {needed_observation_text}".strip()
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index != later_index:
                continue
            combined_text = (
                f"{combined_text} {str(item.get('support') or '').lower()} {str(item.get('contradiction') or '').lower()}"
            ).strip()
            break
        ambiguity_markers = (
            "whether",
            "unclear",
            "not yet visible",
            "not visible",
            "remains unresolved",
            "still unresolved",
            "still unclear",
            "could still",
            "may be",
            "might be",
            "未明确",
            "仍不清楚",
            "还看不出",
            "证据不足",
        )
        if not any(marker in combined_text for marker in ambiguity_markers):
            return None
        # Only override same-object revisit when the current top answer is the
        # immediate micro-outcome and the later-use competitor still needs proof.
        if later_index == best_index:
            same_object_block_markers = (
                "same-object",
                "same object",
                "cap action",
                "lid action",
                "cover fit",
                "cover fits",
            )
            if "final_place_return" not in later_categories and any(
                marker in f"{reason_text} {needed_observation_text}" for marker in same_object_block_markers
            ):
                return None
        later_target = self._action_intent_later_outcome_target_hint(
            choice=later_choice,
            action_object=action_object,
            categories=later_categories,
            evidence_text=combined_text,
        )
        if later_target is None:
            return None
        return later_target

    def _action_intent_choice_has_hand_free_language(self, choice: str) -> bool:
        text = str(choice or "").strip().lower()
        return any(
            token in text
            for token in (
                "free hand",
                "free one hand",
                "left hand is free",
                "right hand is free",
                "use the left hand",
                "use the right hand",
                "腾出",
                "左手",
                "右手",
            )
        )

    def _action_intent_choice_target_or_same_object_hint(
        self,
        *,
        choice: str,
        action_object: str,
    ) -> tuple[str, str] | None:
        fixture_targets = self._action_intent_choice_fixture_target_candidates(choice=choice, action_object=action_object)
        if fixture_targets:
            return fixture_targets[0], "fixture"
        object_targets = [
            token
            for token in self._action_intent_choice_target_object_candidates(choice=choice, action_object=action_object)
            if token not in set(fixture_targets)
        ]
        if object_targets:
            return object_targets[0], "object"
        if self._action_intent_choice_is_same_object_active_use(choice=choice, action_object=action_object):
            return action_object, "object"
        return None

    def _action_intent_choice_is_generic_measurement_meta_purpose(self, choice: str) -> bool:
        text = str(choice or "").strip().lower()
        return any(
            token in text
            for token in (
                "adjust the measurements",
                "adjust measurements",
                "adjust the scale",
                "record measurements",
                "record the measurements",
                "read the measurements",
                "check the reading",
                "measurement reading",
                "调整刻度",
                "记录读数",
                "看读数",
            )
        )

    def _action_intent_choice_is_exact_measurement_role_purpose(self, choice: str) -> bool:
        text = str(choice or "").strip().lower()
        if self._action_intent_choice_is_generic_measurement_meta_purpose(choice):
            return False
        return any(
            token in text
            for token in (
                "measure the",
                "weigh the",
                "weigh more ingredients",
                "measure more ingredients",
                "base to weigh",
                "base for weighing",
                "used as a base",
                "as a base to weigh",
                "as a base for weighing",
                "on the scale",
                "tared",
                "tare",
                "measure ingredients",
                "weigh ingredients",
                "称量",
                "称重",
                "作为称量基底",
            )
        )

    def _action_intent_choice_is_generic_measure_phone_goal(self, *, choice: str, action_object: str) -> bool:
        text = str(choice or "").strip().lower()
        object_text = str(action_object or "").strip().lower()
        if not any(token in object_text for token in ("phone", "smartphone", "mobile")):
            return False
        if self._action_intent_choice_is_phone_record_target_purpose(choice):
            return False
        return any(
            token in text
            for token in (
                "measure the ingredients",
                "measure ingredients",
                "weigh the ingredients",
                "measure.",
                "to measure",
                "测量食材",
                "称量食材",
            )
        )

    def _action_intent_choice_record_target_hint(self, choice: str) -> str:
        text = str(choice or "").strip().lower()
        patterns = (
            r"(?:nutritional\s+value|nutrition(?:al)?\s+value|value)\s+of\s+the\s+([a-z0-9][a-z0-9 -]*)",
            r"(?:measurements?|entry|entries|record|update|log)\s+of\s+the\s+([a-z0-9][a-z0-9 -]*)",
            r"for\s+the\s+([a-z0-9][a-z0-9 -]*)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            target = re.split(r"[,.]", str(match.group(1) or "").strip())[0].strip()
            target = re.sub(
                r"\b(with|using|while|after|before|on|in|at|near|beside|next)\b.*$",
                "",
                target,
            ).strip()
            if target:
                return target
        return ""

    def _action_intent_choice_is_phone_record_target_purpose(self, choice: str) -> bool:
        text = str(choice or "").strip().lower()
        target_hint = self._action_intent_choice_record_target_hint(choice)
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
        return bool(target_hint) and not any(
            token in text
            for token in (
                "measure the ingredients",
                "measure ingredients",
                "weigh the ingredients",
                "to measure.",
                "generic measure",
            )
        )

    def _action_intent_verifier_blocked_hand_free_target_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> tuple[str, str] | None:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return None
        if blocker_hint not in {"post_action_evidence", "future_use_close_call", "pairwise_close_call"}:
            return None
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_index = self._coerce_choice_index(result.get("best_index"), choices)
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return None
        best_choice = choices[best_index]
        competitor_choice = choices[competitor_index]
        best_has_hand_free = self._action_intent_choice_has_hand_free_language(best_choice)
        competitor_has_hand_free = self._action_intent_choice_has_hand_free_language(competitor_choice)
        if not best_has_hand_free and not competitor_has_hand_free:
            return None
        action_object = self._action_intent_question_object_hint(state)
        reason_text = str(result.get("reason") or "").lower()
        needed_observation_text = str(result.get("needed_observation") or "").lower()
        combined_text = f"{reason_text} {needed_observation_text}".strip()
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index not in {best_index, competitor_index}:
                continue
            combined_text = (
                f"{combined_text} {str(item.get('support') or '').lower()} {str(item.get('contradiction') or '').lower()}"
            ).strip()
        uncertainty_markers = (
            "unclear",
            "not yet visible",
            "not visible",
            "remains ambiguous",
            "still ambiguous",
            "still unresolved",
            "may next",
            "could next",
            "not yet seen",
            "仍不清楚",
            "还看不出",
            "证据不足",
        )
        hand_free_markers = (
            "free hand",
            "other hand",
            "left hand",
            "right hand",
            "腾出",
            "另一只手",
            "左手",
            "右手",
        )
        if not any(marker in combined_text for marker in hand_free_markers):
            return None
        if not any(marker in combined_text for marker in uncertainty_markers):
            return None
        target_hint = None
        if best_has_hand_free:
            target_hint = self._action_intent_choice_target_or_same_object_hint(choice=competitor_choice, action_object=action_object)
            if target_hint is None:
                target_hint = self._action_intent_choice_target_or_same_object_hint(choice=best_choice, action_object=action_object)
        else:
            target_hint = self._action_intent_choice_target_or_same_object_hint(choice=best_choice, action_object=action_object)
            if target_hint is None:
                target_hint = self._action_intent_choice_target_or_same_object_hint(choice=competitor_choice, action_object=action_object)
        return target_hint

    def _action_intent_verifier_blocked_measurement_target_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> tuple[str, str] | None:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return None
        if blocker_hint != "future_use_close_call":
            return None
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_index = self._coerce_choice_index(result.get("best_index"), choices)
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return None
        best_choice = choices[best_index]
        competitor_choice = choices[competitor_index]
        best_is_generic_meta = self._action_intent_choice_is_generic_measurement_meta_purpose(best_choice)
        competitor_is_generic_meta = self._action_intent_choice_is_generic_measurement_meta_purpose(competitor_choice)
        best_is_exact_role = self._action_intent_choice_is_exact_measurement_role_purpose(best_choice)
        competitor_is_exact_role = self._action_intent_choice_is_exact_measurement_role_purpose(competitor_choice)
        if not ((best_is_generic_meta and competitor_is_exact_role) or (competitor_is_generic_meta and best_is_exact_role)):
            return None
        combined_text = (
            f"{str(result.get('reason') or '').lower()} "
            f"{str(result.get('needed_observation') or '').lower()}"
        ).strip()
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index not in {best_index, competitor_index}:
                continue
            combined_text = (
                f"{combined_text} {str(item.get('support') or '').lower()} {str(item.get('contradiction') or '').lower()}"
            ).strip()
        uncertainty_markers = (
            "unclear",
            "not yet visible",
            "not visible",
            "not yet seen",
            "still unclear",
            "still unresolved",
            "plausible",
            "missing",
            "no reading",
            "no tare",
            "no update",
            "仍不清楚",
            "还看不出",
            "证据不足",
            "没有读数",
            "没有归零",
        )
        if not any(marker in combined_text for marker in uncertainty_markers):
            return None
        return "scale", "fixture"

    def _action_intent_verifier_blocked_phone_record_target_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> tuple[str, str] | None:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return None
        if blocker_hint != "future_use_close_call":
            return None
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_index = self._coerce_choice_index(result.get("best_index"), choices)
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        if best_index is None or competitor_index is None or best_index == competitor_index:
            return None
        action_object = self._action_intent_question_object_hint(state)
        best_choice = choices[best_index]
        competitor_choice = choices[competitor_index]
        best_is_generic_measure = self._action_intent_choice_is_generic_measure_phone_goal(
            choice=best_choice,
            action_object=action_object,
        )
        competitor_is_generic_measure = self._action_intent_choice_is_generic_measure_phone_goal(
            choice=competitor_choice,
            action_object=action_object,
        )
        best_is_exact_record = self._action_intent_choice_is_phone_record_target_purpose(best_choice)
        competitor_is_exact_record = self._action_intent_choice_is_phone_record_target_purpose(competitor_choice)
        if not ((best_is_generic_measure and competitor_is_exact_record) or (competitor_is_generic_measure and best_is_exact_record)):
            return None
        combined_text = (
            f"{str(result.get('reason') or '').lower()} "
            f"{str(result.get('needed_observation') or '').lower()}"
        ).strip()
        candidate_rows: list[tuple[int, float, str, str, str]] = []
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index is None:
                continue
            choice = choices[index]
            support = str(item.get("support") or "").lower()
            contradiction = str(item.get("contradiction") or "").lower()
            try:
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0
            combined_text = (
                f"{combined_text} {support} {contradiction}"
            ).strip()
            candidate_rows.append((index, score, choice, support, contradiction))
        uncertainty_markers = (
            "unclear",
            "not yet visible",
            "not visible",
            "not yet seen",
            "still unclear",
            "still unresolved",
            "no actual recording target",
            "no direct recording target",
            "recording target is not shown",
            "no specific target",
            "no broccoli target",
            "no coriander target",
            "screen is not readable",
            "screen not readable",
            "entry is not readable",
            "仍不清楚",
            "还看不出",
            "证据不足",
            "没有直接记录目标",
            "没有具体目标",
        )
        if not any(marker in combined_text for marker in uncertainty_markers):
            return None
        exact_target_candidates: list[tuple[float, str]] = []
        for index, score, choice, support, contradiction in candidate_rows:
            if not self._action_intent_choice_is_phone_record_target_purpose(choice):
                continue
            target_hint = self._action_intent_choice_record_target_hint(choice)
            if not target_hint:
                continue
            candidate_uncertainty = f"{support} {contradiction}"
            uncertainty_bonus = 0.0
            if any(
                marker in candidate_uncertainty
                for marker in (
                    "not readable",
                    "screen not readable",
                    "entry is not readable",
                    "recording target is not shown",
                    "no direct recording target",
                    "no broccoli target",
                    "no coriander target",
                    "no carrot target",
                    "still unclear",
                    "still unresolved",
                    "not yet visible",
                    "没有直接记录目标",
                    "还看不出",
                    "证据不足",
                )
            ):
                uncertainty_bonus += 0.22
            if index == best_index:
                uncertainty_bonus += 0.06
            if index == competitor_index:
                uncertainty_bonus += 0.04
            exact_target_candidates.append((score + uncertainty_bonus, target_hint))
        if not exact_target_candidates:
            exact_choice = best_choice if best_is_exact_record else competitor_choice
            target_hint = self._action_intent_choice_record_target_hint(exact_choice)
            if not target_hint:
                return None
            return target_hint, "object"
        exact_target_candidates.sort(key=lambda item: (-item[0], item[1]))
        return exact_target_candidates[0][1], "object"

    def _action_intent_unresolved_rerank_hand_free_object_hint(self, state: AgentState) -> str:
        reason = self._action_intent_recent_unresolved_rerank_withheld_reason(state)
        if not reason or "timeline_review_hand_free_or_fixture_gap" not in reason:
            return ""
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return ""
        _tool_name, payload = latest
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        best_index = self._coerce_choice_index(payload.get("best_index"), choices)
        if best_index is None:
            return ""
        action_object = self._action_intent_question_object_hint(state)
        best_choice = choices[best_index]
        best_object_targets = [
            token
            for token in self._action_intent_choice_target_object_candidates(choice=best_choice, action_object=action_object)
            if token not in set(self._action_intent_choice_fixture_target_candidates(choice=best_choice, action_object=action_object))
        ]
        if best_object_targets:
            return best_object_targets[0]
        if self._action_intent_choice_is_same_object_active_use(choice=best_choice, action_object=action_object):
            return action_object
        candidate_rows: list[tuple[float, int]] = []
        for item in payload.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index is None or index == best_index:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:  # noqa: BLE001
                score = 0.0
            candidate_rows.append((score, index))
        for _score, index in sorted(candidate_rows, key=lambda pair: (-pair[0], pair[1])):
            choice = choices[index]
            if self._action_intent_choice_is_same_object_active_use(choice=choice, action_object=action_object):
                return action_object
            object_targets = [
                token
                for token in self._action_intent_choice_target_object_candidates(choice=choice, action_object=action_object)
                if token not in set(self._action_intent_choice_fixture_target_candidates(choice=choice, action_object=action_object))
            ]
            if object_targets:
                return object_targets[0]
        return ""

    def _build_action_intent_finalize_withheld_long_horizon_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        marker = self._action_intent_recent_later_outcome_finalize_withheld_marker(state)
        if not marker:
            return None
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        nodes = self._latest_action_intent_long_horizon_nodes(state)
        if not nodes:
            return None
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
        )
        if selected is None:
            return None
        _node, start_time, end_time = selected
        anchor_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": anchor_time,
                "object_name": self._action_intent_question_object_hint(state),
                "limit": 16,
            },
        )

    def _build_action_intent_finalize_withheld_generic_hand_free_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_recent_generic_hand_free_finalize_withheld_hint(state)
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 finalizer 指出的真实下游目标 `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 finalizer 指出的真实下游目标 `{downstream_target}` 的更晚轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_finalize_withheld_generic_access_or_space_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_recent_generic_access_or_space_finalize_withheld_hint(state)
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 finalizer 指出的 reveal/access 下游目标 `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 finalizer 指出的 reveal/access 下游目标 `{downstream_target}` 的更晚轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_finalize_withheld_generic_relocation_or_storage_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_recent_generic_relocation_or_storage_finalize_withheld_hint(state)
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 finalizer 指出的真实后续目标 `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 finalizer 指出的真实后续目标 `{downstream_target}` 的更晚轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_recent_mixed_horizon_later_target_withheld_hint(state)
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 mixed-horizon 竞争里更晚结果对应的真实目标 `{downstream_target}` 轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 mixed-horizon 竞争里更晚结果对应的真实目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        if target_kind == "fixture":
            later_selected = None
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                start_raw = node.get("start_time")
                end_raw = node.get("end_time")
                if start_raw is None:
                    continue
                try:
                    start_time = float(start_raw)
                    end_time = float(end_raw) if end_raw is not None else start_time
                except Exception:  # noqa: BLE001
                    continue
                if min_start_time is not None and start_time < min_start_time:
                    continue
                later_selected = (node, start_time, end_time)
            if later_selected is not None:
                selected = later_selected
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_verifier_blocked_mixed_horizon_later_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_verifier_blocked_mixed_horizon_later_target_hint(
            state=state,
            result=result,
            blocker_hint=blocker_hint,
        )
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已经暴露出 mixed-horizon close call；先定位更晚结果对应的真实目标 `{downstream_target}` 轨迹，而不是继续围着当前物体或近窗状态泛化补帧。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已经暴露出 mixed-horizon close call；继续重新检索更晚结果对应的真实目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        if target_kind == "fixture":
            later_selected = None
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                start_raw = node.get("start_time")
                end_raw = node.get("end_time")
                if start_raw is None:
                    continue
                try:
                    start_time = float(start_raw)
                    end_time = float(end_raw) if end_raw is not None else start_time
                except Exception:  # noqa: BLE001
                    continue
                if min_start_time is not None and start_time < min_start_time:
                    continue
                later_selected = (node, start_time, end_time)
            if later_selected is not None:
                selected = later_selected
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought="why 题被 verifier 拦下后，当前 `infer` 已经暴露出 `open/check` 对 `weigh/serve/...` 的 mixed-horizon close call；优先直接追更晚结果对应的真实目标，而不是继续围着当前物体或局部状态泛化补帧。",
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_verifier_blocked_hand_free_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_verifier_blocked_hand_free_target_hint(
            state=state,
            result=result,
            blocker_hint=blocker_hint,
        )
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已暴露出 generic hand-free 仍只是中间态；先定位真正下游目标 `{downstream_target}` 的更晚轨迹，而不是继续把“手空出来了”当结论。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已暴露出 generic hand-free 仍只是中间态；继续重新检索真正下游目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought="why 题被 verifier 拦下后，当前 `infer` 已暴露出 generic hand-free 只是 enablement 中间态；优先直接追 hand-free 背后真正要拿起、要继续操作，或同一物体后续要处理的目标。",
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_verifier_blocked_measurement_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_verifier_blocked_measurement_target_hint(
            state=state,
            result=result,
            blocker_hint=blocker_hint,
        )
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已暴露出 generic measurement-meta 仍只是宽泛量测语境；先定位真正量测目标 `{downstream_target}` 的更晚轨迹，而不是继续停留在调读数/量测语境层。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已暴露出 generic measurement-meta 仍只是宽泛量测语境；继续重新检索真正量测目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought="why 题被 verifier 拦下后，当前 `infer` 已暴露出 generic measurement-meta 只是宽泛量测语境；优先直接追更有判别力的量测目标，而不是继续停留在调读数/量测语境层。",
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_verifier_blocked_phone_record_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_verifier_blocked_phone_record_target_hint(
            state=state,
            result=result,
            blocker_hint=blocker_hint,
        )
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已暴露出 phone generic-measure 仍只是宽泛量测语境；先定位真正记录目标 `{downstream_target}` 的更晚轨迹，而不是继续停留在 generic measure 层。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.5
        selected = None
        candidate_nodes = list(nodes)
        if min_start_time is None and len(candidate_nodes) >= 2:
            candidate_nodes = list(reversed(candidate_nodes))
        for node in candidate_nodes:
            if not isinstance(node, dict):
                continue
            start_raw = node.get("start_time")
            end_raw = node.get("end_time")
            if start_raw is None:
                continue
            try:
                start_time = float(start_raw)
                end_time = float(end_raw) if end_raw is not None else start_time
            except Exception:  # noqa: BLE001
                continue
            if min_start_time is not None and start_time < min_start_time:
                continue
            selected = (node, start_time, end_time)
            break
        if selected is None:
            selected = self._action_intent_select_long_horizon_node(
                state=state,
                hints=hints,
                nodes=nodes,
                min_start_time=min_start_time,
                object_hint=downstream_target,
            )
        if selected is None:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 `infer` 已暴露出 phone generic-measure 仍只是宽泛量测语境；继续重新检索真正记录目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought="why 题被 verifier 拦下后，当前 `infer` 已暴露出 phone generic-measure 只是宽泛量测语境；优先直接追真正被记录/录入的食材目标，而不是继续停留在 generic measure 层。",
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_unresolved_rerank_long_horizon_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_unresolved_rerank_reason_prefers_later_outcome_revisit(state):
            return None
        nodes = self._latest_action_intent_long_horizon_nodes(state)
        if not nodes:
            return None
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
        )
        if selected is None:
            return None
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": self._action_intent_question_object_hint(state),
                "limit": 16,
            },
        )

    def _build_action_intent_unresolved_rerank_mixed_horizon_later_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        hint = self._action_intent_unresolved_rerank_mixed_horizon_later_target_hint(state)
        if hint is None:
            return None
        downstream_target, target_kind = hint
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 mixed-horizon 竞争里更晚结果对应的真实目标 `{downstream_target}` 轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 mixed-horizon 竞争里更晚结果对应的真实目标 `{downstream_target}`。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        if target_kind == "fixture":
            later_selected = None
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                start_raw = node.get("start_time")
                end_raw = node.get("end_time")
                if start_raw is None:
                    continue
                try:
                    start_time = float(start_raw)
                    end_time = float(end_raw) if end_raw is not None else start_time
                except Exception:  # noqa: BLE001
                    continue
                if min_start_time is not None and start_time < min_start_time:
                    continue
                later_selected = (node, start_time, end_time)
            if later_selected is not None:
                selected = later_selected
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16 if target_kind == "fixture" else 18,
            },
        )

    def _build_action_intent_unresolved_rerank_downstream_target_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        downstream_target = self._action_intent_unresolved_rerank_downstream_object_hint(state)
        if not downstream_target:
            return None
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位下游目标对象 `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索下游目标对象 `{downstream_target}` 的更晚轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16,
            },
        )

    def _build_action_intent_unresolved_rerank_downstream_fixture_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        downstream_target = self._action_intent_unresolved_rerank_downstream_fixture_hint(state)
        if not downstream_target:
            return None
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位下游装置/fixture `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索下游装置/fixture `{downstream_target}` 的更晚轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16,
            },
        )

    def _build_action_intent_unresolved_rerank_hand_free_object_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        thought: str,
    ) -> PlannerDecision | None:
        downstream_target = self._action_intent_unresolved_rerank_hand_free_object_hint(state)
        if not downstream_target:
            return None
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=downstream_target)
        if not nodes:
            return PlannerDecision(
                thought=f"{thought} 先定位 hand-free 背后的真实下游对象 `{downstream_target}` 在更晚时刻的轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=downstream_target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"{thought} 继续重新检索 hand-free 背后的真实下游对象 `{downstream_target}` 的更晚轨迹。",
                tool="query_object",
                args={"query": downstream_target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought=thought,
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": downstream_target,
                "limit": 16,
            },
        )

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
            if self._action_intent_success_result_is_ready_for_failure_finish(
                state=state,
                payload=recovered_intent,
            ):
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
                evidence_first = self._build_action_intent_evidence_first_recovery_decision(
                    state=state,
                    hints=hints,
                    used_tools=used_tools,
                )
                if evidence_first is not None:
                    return evidence_first
                specialized_resolution = self._build_action_intent_specialized_resolution_before_text_fallback(
                    state=state,
                    hints=hints,
                )
                if specialized_resolution is not None:
                    return specialized_resolution
                if action_intent_requires_strict_visual_disambiguation(
                    question=str(getattr(state, "question", "") or ""),
                    choices=[str(choice) for choice in getattr(state, "choices", [])],
                    indices=None,
                ):
                    strict_recovery = self._build_action_intent_strict_text_fallback_recovery_decision(
                        state=state,
                        hints=hints,
                    )
                    if strict_recovery is not None:
                        return strict_recovery
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
            and last_tool.get("tool") in {"sample_sparse_frames", "extract_frames_for_range", "sample_frames_around_peaks", "retrieve_cached_artifacts"}
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
            and self._action_intent_prefers_long_horizon_object_retrieval(state=state)
        ):
            args = last_tool.get("args") or {}
            if isinstance(args, dict):
                object_name = str(args.get("object_name") or "").strip().lower()
                target_name = self._action_intent_question_object_hint(state).strip().lower()
                if object_name and object_name == target_name:
                    anchor_time = args.get("time_s")
                    try:
                        anchor_value = float(anchor_time)
                    except Exception:  # noqa: BLE001
                        anchor_value = None
                    if anchor_value is not None:
                        if self._action_intent_long_horizon_spatial_context_looks_nonexclusive_storage_anchor(
                            state=state,
                            hints=hints,
                            spatial=last_result,
                            anchor_time=anchor_value,
                        ):
                            long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                                state=state,
                                hints=hints,
                                thought="why 题当前只是短暂停在 fridge/cabinet/shelf 一类 storage 附近，但还没有真正放回闭环；继续沿更晚节点向后追。",
                                after_time=anchor_value,
                            )
                            if long_horizon_revisit is not None:
                                return long_horizon_revisit
                        if self._action_intent_long_horizon_spatial_context_looks_transit_near_decisive_fixture(
                            state=state,
                            hints=hints,
                            spatial=last_result,
                            anchor_time=anchor_value,
                        ):
                            long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                                state=state,
                                hints=hints,
                                thought="why 题当前虽然靠近了有判别力的 fixture，但这一下更像短暂经过态而非真正完成使用/放置；继续沿更晚节点向后追。",
                                after_time=anchor_value,
                            )
                            if long_horizon_revisit is not None:
                                return long_horizon_revisit
                        if self._action_intent_long_horizon_spatial_context_looks_intermediate(state=state, spatial=last_result):
                            long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                                state=state,
                                hints=hints,
                                thought="why 题当前只看到目标对象还处在中间态 workspace / active-area 里，不能把这一下当成最终去向或真实用途；继续沿更晚节点向后追。",
                                after_time=anchor_value,
                            )
                            if long_horizon_revisit is not None:
                                return long_horizon_revisit
                        attempt_count = self._action_intent_followup_attempt_count(state)
                        return PlannerDecision(
                            thought="why 题已补到更晚时刻的空间上下文；继续围绕该对象后续位置抽关键帧，检查它后来到底被放回、再使用，还是仅暂时移开。",
                            tool="sample_sparse_frames",
                            args={
                                "start_time": max(0.0, anchor_value - 0.4),
                                "end_time": anchor_value + 2.0,
                                "sample_count": 4,
                                "tag": f"{state.task_family}_followup_ext{attempt_count + 1}",
                            },
                        )
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
                probe_window = self._action_intent_transition_probe_window(state=state, hints=hints, result=None)
                if probe_window is not None:
                    start_time, end_time, stride_s, max_frames = probe_window
                    return PlannerDecision(
                        thought="why 题涉及电子秤按键，先围绕点击后的短窗口做更密的关键帧搜索，确认显示是否开机、归零或出现其它决定性状态变化。",
                        tool="extract_frames_for_range",
                        args={
                            "start_time": start_time,
                            "end_time": end_time,
                            "stride_s": stride_s,
                            "max_frames": max_frames,
                            "tag": f"{state.task_family}_followup_transition",
                        },
                    )
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
                and not self._action_intent_should_preempt_initial_followup_with_transition(
                    state=state,
                    hints=hints,
                    result=last_result,
                )
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(last_result.get("followup_focus") or "precondition_before_future_use"),
                )
                if precondition is not None:
                    return precondition
            if self._action_intent_requires_followup(state, result=last_result):
                needed_observation_relation_revisit = self._build_action_intent_needed_observation_relation_revisit_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题当前已经知道要确认的不只是某个目标，而是动作物体与该目标之间的判别关系；优先去查这个关系是否真的变成 `on/into/over/returned`，而不是继续泛化补帧。",
                )
                if needed_observation_relation_revisit is not None:
                    return needed_observation_relation_revisit
                needed_observation_target_revisit = self._build_action_intent_needed_observation_target_revisit_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题当前已经知道真正需要确认的是某个判别目标/位置；优先去追 `needed_observation` 明确点名的目标，而不是继续做泛化 followup 或只沿动作物体后追。",
                )
                if needed_observation_target_revisit is not None:
                    return needed_observation_target_revisit
                weak_late_anchor_revisit = self._build_action_intent_weak_late_anchor_revisit_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题当前虽然已经补到晚锚点，但输出仍只是弱邻近/手持证据；继续沿更晚节点向后追，再决定是否补近窗或进入专用裁决。",
                )
                if weak_late_anchor_revisit is not None:
                    return weak_late_anchor_revisit
                nonexclusive_concrete_late_anchor_revisit = self._build_action_intent_nonexclusive_concrete_late_anchor_revisit_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题当前虽然给出了更具体的晚锚点描述，但本质上仍只是标签可见/物体被放在某处附近的中间态，不足以在候选间形成排他结论；继续沿更晚节点向后追。",
                )
                if nonexclusive_concrete_late_anchor_revisit is not None:
                    return nonexclusive_concrete_late_anchor_revisit
                initial_transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题一开始就落在近窗结果歧义上；先直接围绕动作尾部做更密的关键帧搜索，确认是否真的出现掉回水槽/容器这类决定性结果，再决定是否补泛化 followup。",
                )
                if initial_transition_probe is not None and self._action_intent_followup_attempt_count(state) < 1:
                    return initial_transition_probe
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
                long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题当前这轮后续帧仍不足以区分更晚用途或最终放置；沿目标对象已知的后续轨迹继续往后追，再决定是否进入专用裁决。",
                )
                if long_horizon_revisit is not None:
                    return long_horizon_revisit
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
                if not self._action_intent_intent_payload_is_ready_to_fall_back_to_text_rank(
                    state=state,
                    payload=last_result,
                ):
                    recovered = self._build_action_intent_specialized_recovery_decision(
                        state=state,
                        hints=hints,
                        thought="why 题当前专用动作目的判断自己仍承认证据未闭合，不能直接退回文本聚合评分；先恢复当前题时间窗关键帧或专用判断，再继续追后续证据。",
                    )
                    if recovered is not None:
                        return recovered
                    return PlannerDecision(
                        thought="why 题当前专用动作目的判断仍未闭合，当前又缺少可直接复用的恢复锚点；先回到当前题动作片段重抽，而不是退回文本聚合收口。",
                        tool="sample_sparse_frames",
                        args={
                            "start_time": None,
                            "end_time": None,
                            "sample_count": 4,
                            "tag": f"{state.task_family}_segment",
                        },
                    )
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
                initial_transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题 top-2 的冲突本身就取决于动作后紧接着发生的近窗结果；先直接补更密的尾部关键帧，再决定是否进入泛化 followup。",
                )
                if initial_transition_probe is not None and self._action_intent_followup_attempt_count(state) < 1:
                    return initial_transition_probe
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
                long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题 top-2 仍缺更晚的排他性结果；沿目标对象已知的后续轨迹继续向后找关键证据，再做二选一裁决。",
                )
                if long_horizon_revisit is not None:
                    return long_horizon_revisit
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题 top-2 仍是动作后果型歧义，不能仅凭高置信直接结束；改为结合结果帧二选一裁决。",
                )
                if pairwise is not None:
                    return pairwise
            if self._action_intent_needs_future_use_evidence(state=state, result=last_result):
                weak_late_anchor_revisit = self._build_action_intent_weak_late_anchor_revisit_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题当前晚锚点给出的只是弱邻近/手持证据，仍不能排除多个后续用途；继续沿更晚节点向后追，再决定是否进入用途专用裁决。",
                )
                if weak_late_anchor_revisit is not None:
                    return weak_late_anchor_revisit
                initial_transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=last_result,
                    thought="why 题当前先要确认动作后是否立刻出现关键微结果；先围绕尾部短窗口密采样，再决定是否继续拉长 followup 去看更晚用途。",
                )
                if initial_transition_probe is not None and self._action_intent_followup_attempt_count(state) < 1:
                    return initial_transition_probe
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
                long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题当前晚帧仍看不出目标对象的真实后续用途；继续沿目标对象更晚的再次出现位置向后追，再决定是否进入用途专用裁决。",
                )
                if long_horizon_revisit is not None:
                    return long_horizon_revisit
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
            and last_tool.get("tool") == "query_object"
            and self._is_action_intent_task(state)
        ):
            nodes = last_result.get("nodes", [])
            if isinstance(nodes, list):
                long_horizon_spatial = self._build_action_intent_long_horizon_spatial_probe_decision(
                    state=state,
                    hints=hints,
                    used_tools=used_tools,
                    nodes=nodes,
                )
                if long_horizon_spatial is not None:
                    return long_horizon_spatial
                long_horizon_sampling = self._build_action_intent_long_horizon_sampling_decision(
                    state=state,
                    hints=hints,
                    nodes=nodes,
                )
                if long_horizon_sampling is not None:
                    return long_horizon_sampling
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
                        long_horizon_revisit = self._build_action_intent_cached_long_horizon_revisit_decision(
                            state=state,
                            hints=hints,
                            thought="why 题在当前长时域关键帧上仍有多个 plausible 解释；继续沿目标对象更晚的再次出现位置向后追，确认它到底被放回、继续使用，还是只是暂时移开。",
                        )
                        if long_horizon_revisit is not None:
                            return long_horizon_revisit
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
            needed_observation_relation_revisit = self._build_action_intent_needed_observation_relation_revisit_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题 pairwise 已经明确真正缺的是某个关系型判别证据；优先去查动作物体与目标之间是否真的出现了 `on/into/over/returned` 这类关系，而不是继续泛化补帧。",
            )
            if needed_observation_relation_revisit is not None:
                return needed_observation_relation_revisit
            needed_observation_target_revisit = self._build_action_intent_needed_observation_target_revisit_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题 pairwise 已经明确真正缺的是某个判别目标/位置上的后续证据；优先去追 `needed_observation` 点名的目标，而不是继续只沿动作物体做泛化补帧。",
            )
            if needed_observation_target_revisit is not None:
                return needed_observation_target_revisit
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
            finalize_access_or_space_revisit = self._build_action_intent_finalize_withheld_generic_access_or_space_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题专用裁决刚被 finalizer 拦下，因为当前仍停留在 generic access / make-space 层；直接改追 finalizer 指出的真实 reveal/use 下游目标，而不是立即 finish。",
            )
            if finalize_access_or_space_revisit is not None:
                return finalize_access_or_space_revisit
            finalize_mixed_horizon_later_target_revisit = (
                self._build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题专用裁决刚被 finalizer 拦下，因为当前只看到了 `check/open` 这类立刻微结果；直接改追 mixed-horizon 竞争里更晚结果对应的真实目标，而不是继续围着动作物体做泛化补帧。",
                )
            )
            if finalize_mixed_horizon_later_target_revisit is not None:
                return finalize_mixed_horizon_later_target_revisit
            finalize_relocation_or_storage_revisit = (
                self._build_action_intent_finalize_withheld_generic_relocation_or_storage_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题专用裁决刚被 finalizer 拦下，因为当前把 generic put-away / temporary relocation 当成结论；直接改追真实的后续用途或下游目标，而不是立即 finish。",
                )
            )
            if finalize_relocation_or_storage_revisit is not None:
                return finalize_relocation_or_storage_revisit
            finalize_hand_free_revisit = self._build_action_intent_finalize_withheld_generic_hand_free_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题专用裁决刚被 finalizer 拦下，因为当前只停留在 generic hand-free 中间态；直接改追 finalizer 指出的真实下游对象，而不是立即 finish。",
            )
            if finalize_hand_free_revisit is not None:
                return finalize_hand_free_revisit
            unresolved_rerank_downstream_fixture_revisit = self._build_action_intent_unresolved_rerank_downstream_fixture_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 已指出 hand-free / fixture enablement 仍缺决定性证据；优先转去追踪真正的下游装置，而不是继续只盯动作物体或泛化补帧。",
            )
            if unresolved_rerank_downstream_fixture_revisit is not None:
                return unresolved_rerank_downstream_fixture_revisit
            unresolved_rerank_hand_free_object_revisit = self._build_action_intent_unresolved_rerank_hand_free_object_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 表明当前只是停留在泛化 hand-free 描述上；优先去追真正被拿起、被使用，或继续被操作的下游对象，而不是把“手空出来了”当成最终目的。",
            )
            if unresolved_rerank_hand_free_object_revisit is not None:
                return unresolved_rerank_hand_free_object_revisit
            unresolved_rerank_downstream_target_revisit = self._build_action_intent_unresolved_rerank_downstream_target_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 已指出 freed-slot / revealed-target 仍缺证据；优先转去追踪下游目标物体，而不是继续盯着被移动的物体本身。",
            )
            if unresolved_rerank_downstream_target_revisit is not None:
                return unresolved_rerank_downstream_target_revisit
            unresolved_rerank_mixed_horizon_later_target_revisit = (
                self._build_action_intent_unresolved_rerank_mixed_horizon_later_target_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题 unresolved rerank 已明确 mixed-horizon 的更晚结果还没看到；优先转去追 later outcome 对应的真实目标，而不是继续只沿动作物体做泛化 long-horizon 后追。",
                )
            )
            if unresolved_rerank_mixed_horizon_later_target_revisit is not None:
                return unresolved_rerank_mixed_horizon_later_target_revisit
            unresolved_rerank_long_horizon_revisit = self._build_action_intent_unresolved_rerank_long_horizon_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题当前 unresolved rerank 已明确指出更晚用途/最终落点仍缺证据；直接沿缓存的目标 object node 向后追，而不是继续做泛化 pairwise followup。",
            )
            if unresolved_rerank_long_horizon_revisit is not None:
                return unresolved_rerank_long_horizon_revisit
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
            if self._action_intent_result_is_workspace_or_final_placement_close_call(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 3:
                    extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                        state=state,
                        hints=hints,
                        focus="workspace_or_final_placement_pairwise_claim_needs_direct_outcome",
                        window_s=8.8,
                    )
                    if extra_followup is not None:
                        return extra_followup
            if not self._action_intent_resolution_payload_is_ready_to_finish(state=state, payload=last_result):
                recovered = self._build_action_intent_specialized_recovery_decision(
                    state=state,
                    hints=hints,
                    thought="why 题二选一裁决仍明确承认证据不够，不能直接结束；先恢复当前题时间窗关键帧或专用判断，再继续追决定性结果证据。",
                )
                if recovered is not None:
                    return recovered
                return PlannerDecision(
                    thought="why 题二选一裁决仍明确承认证据不够，当前又没有可直接复用的时间锚点；退回当前题动作片段重抽，而不是直接结束。",
                    tool="sample_sparse_frames",
                    args={
                        "start_time": None,
                        "end_time": None,
                        "sample_count": 4,
                        "tag": f"{state.task_family}_segment",
                    },
                )
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
            needed_observation_relation_revisit = self._build_action_intent_needed_observation_relation_revisit_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题 future-use 裁决已经明确真正缺的是某个关系型判别证据；优先去查动作物体与目标之间是否真的出现了 `on/into/over/returned` 这类关系，而不是继续泛化补帧。",
            )
            if needed_observation_relation_revisit is not None:
                return needed_observation_relation_revisit
            needed_observation_target_revisit = self._build_action_intent_needed_observation_target_revisit_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题 future-use 裁决已经明确真正缺的是某个判别目标/位置上的后续证据；优先去追 `needed_observation` 点名的目标，而不是继续只沿动作物体做泛化补帧。",
            )
            if needed_observation_target_revisit is not None:
                return needed_observation_target_revisit
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
            finalize_access_or_space_revisit = self._build_action_intent_finalize_withheld_generic_access_or_space_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题专用裁决刚被 finalizer 拦下，因为当前把 generic access / make-space 当结论；优先转去追真正的 reveal/use 下游目标，而不是直接 finish。",
            )
            if finalize_access_or_space_revisit is not None:
                return finalize_access_or_space_revisit
            finalize_mixed_horizon_later_target_revisit = (
                self._build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题专用裁决刚被 finalizer 拦下，因为当前仍停留在 `check/open` 这类立刻微结果；优先转去追 mixed-horizon 竞争里更晚结果对应的真实目标，而不是继续围着动作物体做泛化补帧。",
                )
            )
            if finalize_mixed_horizon_later_target_revisit is not None:
                return finalize_mixed_horizon_later_target_revisit
            finalize_relocation_or_storage_revisit = (
                self._build_action_intent_finalize_withheld_generic_relocation_or_storage_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题专用裁决刚被 finalizer 拦下，因为当前仍把 generic put-away / temporary relocation 当成结论；优先转去追真正的后续用途或下游目标，而不是直接 finish。",
                )
            )
            if finalize_relocation_or_storage_revisit is not None:
                return finalize_relocation_or_storage_revisit
            finalize_hand_free_revisit = self._build_action_intent_finalize_withheld_generic_hand_free_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题专用裁决刚被 finalizer 拦下，因为当前仍把 generic hand-free 当结论；优先转去追真正的后续对象/用途，而不是直接 finish。",
            )
            if finalize_hand_free_revisit is not None:
                return finalize_hand_free_revisit
            unresolved_rerank_downstream_fixture_revisit = self._build_action_intent_unresolved_rerank_downstream_fixture_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 已指出 hand-free / fixture enablement 的真正下游装置还没被确认；优先转去追踪那个 fixture，而不是继续只看动作物体。",
            )
            if unresolved_rerank_downstream_fixture_revisit is not None:
                return unresolved_rerank_downstream_fixture_revisit
            unresolved_rerank_hand_free_object_revisit = self._build_action_intent_unresolved_rerank_hand_free_object_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 表明当前只看到“腾出一只手”这类中间态；优先去追 hand-free 之后真正要操作的对象或同一物体的后续用途。",
            )
            if unresolved_rerank_hand_free_object_revisit is not None:
                return unresolved_rerank_hand_free_object_revisit
            unresolved_rerank_downstream_target_revisit = self._build_action_intent_unresolved_rerank_downstream_target_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 已指出 revealed-target / freed-slot 的真正下游目标还没被确认；优先转去追踪那个目标物体，而不是继续只看动作物体。",
            )
            if unresolved_rerank_downstream_target_revisit is not None:
                return unresolved_rerank_downstream_target_revisit
            unresolved_rerank_mixed_horizon_later_target_revisit = (
                self._build_action_intent_unresolved_rerank_mixed_horizon_later_target_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题 unresolved rerank 已明确 mixed-horizon 的更晚结果还没被确认；优先转去追 later outcome 对应的真实目标，而不是继续只沿动作物体做泛化 long-horizon 后追。",
                )
            )
            if unresolved_rerank_mixed_horizon_later_target_revisit is not None:
                return unresolved_rerank_mixed_horizon_later_target_revisit
            unresolved_rerank_long_horizon_revisit = self._build_action_intent_unresolved_rerank_long_horizon_revisit_decision(
                state=state,
                hints=hints,
                thought="why 题 unresolved rerank 已明确指出 later-use / final-location 还缺更晚证据；直接沿缓存目标节点向后追，而不是继续做泛化 future-use followup。",
            )
            if unresolved_rerank_long_horizon_revisit is not None:
                return unresolved_rerank_long_horizon_revisit
            transition_probe = self._build_action_intent_resolution_transition_recovery_decision(
                state=state,
                hints=hints,
                tool_name="resolve_action_intent_future_use",
                result=last_result,
            )
            if transition_probe is not None:
                return transition_probe
            weak_late_anchor_revisit = self._build_action_intent_weak_late_anchor_revisit_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题当前后续用途裁决仍只落在晚锚点的弱邻近/手持证据上，排他性仍不足；继续沿更晚节点向后追，再决定是否允许收口。",
            )
            if weak_late_anchor_revisit is not None:
                return weak_late_anchor_revisit
            nonexclusive_concrete_late_anchor_revisit = self._build_action_intent_nonexclusive_concrete_late_anchor_revisit_decision(
                state=state,
                hints=hints,
                result=last_result,
                thought="why 题当前后续用途裁决虽然给出了更具体的晚锚点描述，但仍只是标签显露或物体暂放邻近位置这类非排他中间态；继续沿更晚节点向后追，再决定是否允许收口。",
            )
            if nonexclusive_concrete_late_anchor_revisit is not None:
                return nonexclusive_concrete_late_anchor_revisit
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
            if self._action_intent_result_is_workspace_or_final_placement_close_call(state=state, result=last_result):
                if self._action_intent_followup_attempt_count(state) < 3:
                    extra_followup = self._build_action_intent_extra_followup_sampling_decision(
                        state=state,
                        hints=hints,
                        focus="workspace_or_final_placement_future_use_claim_needs_direct_outcome",
                        window_s=8.8,
                    )
                    if extra_followup is not None:
                        return extra_followup
            if not self._action_intent_resolution_payload_is_ready_to_finish(state=state, payload=last_result):
                recovered = self._build_action_intent_specialized_recovery_decision(
                    state=state,
                    hints=hints,
                    thought="why 题后续用途裁决仍明确承认证据不够，不能直接结束；先恢复当前题时间窗关键帧或专用判断，再继续追决定性动作后证据。",
                )
                if recovered is not None:
                    return recovered
                return PlannerDecision(
                    thought="why 题后续用途裁决仍明确承认证据不够，当前又没有可直接复用的时间锚点；退回当前题动作片段重抽，而不是直接结束。",
                    tool="sample_sparse_frames",
                    args={
                        "start_time": None,
                        "end_time": None,
                        "sample_count": 4,
                        "tag": f"{state.task_family}_segment",
                    },
                )
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
                evidence_first = self._build_action_intent_evidence_first_recovery_decision(
                    state=state,
                    hints=hints,
                    used_tools=used_tools,
                )
                if evidence_first is not None and evidence_first.tool != "rank_choices_from_state":
                    return evidence_first
                if action_intent_requires_strict_visual_disambiguation(
                    question=str(getattr(state, "question", "") or ""),
                    choices=[str(choice) for choice in getattr(state, "choices", [])],
                    indices=None,
                ):
                    strict_recovery = self._build_action_intent_strict_text_fallback_recovery_decision(
                        state=state,
                        hints=hints,
                    )
                    if strict_recovery is not None:
                        return strict_recovery
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
        precombined_times = sorted(
            [float(value) for value in hints.get("times") or []]
            + [float(value) for value in hints.get("input_times") or []]
        )
        initial_action_intent_route = self._build_initial_action_intent_specialized_decision(
            state=state,
            hints=hints,
            used_tools=used_tools,
        )
        if initial_action_intent_route is not None:
            return initial_action_intent_route
        action_intent_step = self._action_intent_step_decision(
            state=state,
            used_tools=used_tools,
            combined_times=precombined_times,
            object_hint=hints.get("object_hint"),
            last_result=last_result if isinstance(last_result, dict) else {},
        )
        if action_intent_step is not None:
            return action_intent_step
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
                recovered = self._build_action_intent_evidence_first_recovery_decision(
                    state=state,
                    hints=hints,
                    used_tools=used_tools,
                )
                if recovered is not None and recovered.tool != decision.tool:
                    self._state_add_memory(state, f"planner_override action_intent_textual_rank={decision.tool} -> {recovered.tool}")
                    return recovered
                if action_intent_requires_strict_visual_disambiguation(
                    question=str(getattr(state, "question", "") or ""),
                    choices=[str(choice) for choice in getattr(state, "choices", [])],
                    indices=None,
                ):
                    recovered = self._build_action_intent_strict_text_fallback_recovery_decision(
                        state=state,
                        hints=hints,
                    )
                    if recovered is not None and recovered.tool != decision.tool:
                        self._state_add_memory(state, f"planner_override strict_text_fallback_rank={decision.tool} -> {recovered.tool}")
                        return recovered
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
            action_frames = self._action_intent_candidate_inference_frames(
                state=state,
                hints=hints,
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
            return decision.tool in {
                "infer_action_intent",
                "resolve_action_intent_pairwise",
                "resolve_action_intent_future_use",
                "query_object",
                "query_spatial_context",
                "finish",
            }
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
            latest_resolution = self._latest_action_intent_resolution_payload(state)
            latest_action_intent_result = latest_resolution[1] if latest_resolution is not None else {}
            if (
                isinstance(latest_action_intent_result, dict)
                and any(
                    isinstance(item, str)
                    and item.startswith("action_intent_resolution_withheld_for_missing_state_change_prereq=1")
                    for item in list(getattr(state, "working_memory", []))[-12:]
                )
                and self._action_intent_resolution_should_backfill_precondition(
                    state=state,
                    hints=hints,
                    result=latest_action_intent_result,
                )
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(
                        latest_action_intent_result.get("needed_observation")
                        or "state_change_prereq_missing_precondition"
                    ),
                )
                if precondition is not None:
                    return precondition
            if self._action_intent_prefers_long_horizon_object_retrieval(state=state):
                long_horizon_query = self._build_action_intent_long_horizon_object_query_decision(
                    state=state,
                    used_tools=used_tools,
                    thought="why 题近窗证据仍不足以区分 later use / final location；先按目标对象做全视频后续检索，而不是回到通用 query_time。",
                )
                if long_horizon_query is not None:
                    return long_horizon_query
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
            latest_resolution = self._latest_action_intent_resolution_payload(state)
            latest_action_intent_result = latest_resolution[1] if latest_resolution is not None else {}
            if (
                isinstance(latest_action_intent_result, dict)
                and any(
                    isinstance(item, str)
                    and item.startswith("action_intent_resolution_withheld_for_missing_state_change_prereq=1")
                    for item in list(getattr(state, "working_memory", []))[-12:]
                )
                and self._action_intent_resolution_should_backfill_precondition(
                    state=state,
                    hints=hints,
                    result=latest_action_intent_result,
                )
            ):
                precondition = self._build_action_intent_precondition_sampling_decision(
                    state=state,
                    hints=hints,
                    focus=str(
                        latest_action_intent_result.get("needed_observation")
                        or "state_change_prereq_missing_precondition"
                    ),
                )
                if precondition is not None:
                    return precondition
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
            latest_intent = self._latest_successful_action_intent_result(state)
            if (
                self._action_intent_pending_resolution_tool(state) == "resolve_action_intent_future_use"
                or self._action_intent_needs_future_use_evidence(state=state, result=latest_intent if latest_intent else None)
            ):
                future_use = self._build_action_intent_future_use_resolution_decision(
                    state=state,
                    hints=hints,
                    thought="why 状态变化题在恢复阶段仍证据不足，优先回到后续用途专用裁决，不退回通用 query_time。",
                )
                if future_use is not None:
                    return future_use
            if (
                self._action_intent_pending_resolution_tool(state) == "resolve_action_intent_pairwise"
                or self._action_intent_pair_needs_outcome_resolution(state=state, result=latest_intent if latest_intent else None)
            ):
                pairwise = self._build_action_intent_pairwise_resolution_decision(
                    state=state,
                    hints=hints,
                    thought="why 状态变化题在恢复阶段仍证据不足，优先回到状态变化专用二选一裁决，不退回通用 query_time。",
                )
                if pairwise is not None:
                    return pairwise
        if (
            self._is_action_intent_task(state)
            and (
                self._action_intent_text_fallback_ready(state)
                or "need_alternative_evidence_path" in open_questions
                or "need_disambiguating_evidence" in open_questions
            )
        ):
            evidence_first = self._build_action_intent_evidence_first_recovery_decision(
                state=state,
                hints=hints,
                used_tools=used_tools,
                failed_tools=failed_tools,
                ineffective_tools=ineffective_tools,
            )
            if evidence_first is not None:
                return evidence_first
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
            action_intent_candidate_frames = self._action_intent_candidate_inference_frames(
                state=state,
                hints=hints,
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

    def _action_intent_success_result_is_ready_for_failure_finish(
        self,
        *,
        state: AgentState,
        payload: dict[str, Any],
    ) -> bool:
        if not self._is_action_intent_task(state) or not isinstance(payload, dict):
            return False
        if payload.get("best_index") is None:
            return False
        if any(
            bool(payload.get(key))
            for key in (
                "need_future_evidence",
                "need_more_evidence",
                "needs_more_evidence",
            )
        ):
            return False
        needed_observation = str(payload.get("needed_observation") or "").strip()
        if needed_observation:
            return False
        recent_memory = list(getattr(state, "working_memory", []) or [])[-16:]
        if any(
            isinstance(item, str)
            and (
                item.startswith("action_intent_pending_resolution=")
                or item.startswith("action_intent_resolution_withheld_for_")
                or item.startswith("action_intent_unresolved_rerank_withheld")
                or item.startswith("action_intent_needed_observation=")
            )
            for item in recent_memory
        ):
            return False
        return True

    def _action_intent_resolution_payload_is_ready_to_finish(
        self,
        *,
        state: AgentState,
        payload: dict[str, Any],
    ) -> bool:
        if not self._is_action_intent_task(state) or not isinstance(payload, dict):
            return False
        if payload.get("best_index") is None:
            return False
        if any(
            bool(payload.get(key))
            for key in (
                "need_future_evidence",
                "need_more_evidence",
                "needs_more_evidence",
            )
        ):
            return False
        needed_observation = str(payload.get("needed_observation") or "").strip()
        if needed_observation:
            return False
        return True

    def _action_intent_intent_payload_is_ready_to_fall_back_to_text_rank(
        self,
        *,
        state: AgentState,
        payload: dict[str, Any],
    ) -> bool:
        if not self._is_action_intent_task(state) or not isinstance(payload, dict):
            return False
        if payload.get("best_index") is None:
            return False
        if any(
            bool(payload.get(key))
            for key in (
                "need_future_evidence",
                "need_more_evidence",
                "needs_more_evidence",
            )
        ):
            return False
        needed_observation = str(payload.get("needed_observation") or "").strip()
        if needed_observation:
            return False
        return True

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
        needed_observation_relation_revisit = self._build_action_intent_needed_observation_relation_revisit_decision(
            state=state,
            hints=hints,
            result=payload,
            thought="why 题被 verifier 拦下后，当前真正缺的已经收敛到某个关系型判别证据；优先直接追动作物体与目标之间是否真的形成了 `on/into/over/returned` 这类关系，而不是先退回泛化补帧。",
        )
        if needed_observation_relation_revisit is not None:
            return needed_observation_relation_revisit
        needed_observation_target_revisit = self._build_action_intent_needed_observation_target_revisit_decision(
            state=state,
            hints=hints,
            result=payload,
            thought="why 题被 verifier 拦下后，当前真正缺的已经收敛到某个判别目标/位置上的后续证据；优先直接追 `needed_observation` 点名的目标，而不是先退回泛化补帧。",
        )
        if needed_observation_target_revisit is not None:
            return needed_observation_target_revisit
        finalize_access_or_space_revisit = self._build_action_intent_finalize_withheld_generic_access_or_space_revisit_decision(
            state=state,
            hints=hints,
            thought="why 题被 verifier 拦下后，当前已知 generic reveal/access 不是结论；优先直接追 finalizer 指出的真实 downstream target，而不是先退回泛化补帧。",
        )
        if finalize_access_or_space_revisit is not None:
            return finalize_access_or_space_revisit
        finalize_relocation_or_storage_revisit = self._build_action_intent_finalize_withheld_generic_relocation_or_storage_revisit_decision(
            state=state,
            hints=hints,
            thought="why 题被 verifier 拦下后，当前已知 generic relocation / put-away 不是结论；优先直接追 finalizer 指出的真实后续目标，而不是先退回泛化补帧。",
        )
        if finalize_relocation_or_storage_revisit is not None:
            return finalize_relocation_or_storage_revisit
        finalize_hand_free_revisit = self._build_action_intent_finalize_withheld_generic_hand_free_revisit_decision(
            state=state,
            hints=hints,
            thought="why 题被 verifier 拦下后，当前已知 generic hand-free 不是结论；优先直接追 finalizer 指出的真实 downstream object / same-object use，而不是先退回泛化补帧。",
        )
        if finalize_hand_free_revisit is not None:
            return finalize_hand_free_revisit
        if (
            any(
                isinstance(item, str)
                and item.startswith("action_intent_resolution_withheld_for_missing_state_change_prereq=1")
                for item in list(getattr(state, "working_memory", []))[-12:]
            )
            and self._action_intent_resolution_should_backfill_precondition(
                state=state,
                hints=hints,
                result=payload,
            )
        ):
            precondition = self._build_action_intent_precondition_sampling_decision(
                state=state,
                hints=hints,
                focus=str(payload.get("needed_observation") or "verifier_blocked_missing_state_change_prereq"),
            )
            if precondition is not None:
                return precondition
        if tool_name == "infer_action_intent" and blocker_hint in {"post_action_evidence", "future_use_close_call", "pairwise_close_call"}:
            finalize_mixed_horizon_later_target_revisit = (
                self._build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题被 verifier/finalizer 拦下，因为 `check/open` 这类近窗解释还没压过更晚结果；直接追 mixed-horizon 竞争里更晚结果对应的真实目标，而不是继续围着动作物体或局部状态泛化补帧。",
                )
            )
            if finalize_mixed_horizon_later_target_revisit is not None:
                return finalize_mixed_horizon_later_target_revisit
            infer_mixed_horizon_later_target_revisit = (
                self._build_action_intent_verifier_blocked_mixed_horizon_later_target_revisit_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    blocker_hint=blocker_hint,
                )
            )
            if infer_mixed_horizon_later_target_revisit is not None:
                return infer_mixed_horizon_later_target_revisit
            infer_hand_free_target_revisit = (
                self._build_action_intent_verifier_blocked_hand_free_target_revisit_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    blocker_hint=blocker_hint,
                )
            )
            if infer_hand_free_target_revisit is not None:
                return infer_hand_free_target_revisit
            infer_measurement_target_revisit = (
                self._build_action_intent_verifier_blocked_measurement_target_revisit_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    blocker_hint=blocker_hint,
                )
            )
            if infer_measurement_target_revisit is not None:
                return infer_measurement_target_revisit
            infer_phone_record_target_revisit = (
                self._build_action_intent_verifier_blocked_phone_record_target_revisit_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    blocker_hint=blocker_hint,
                )
            )
            if infer_phone_record_target_revisit is not None:
                return infer_phone_record_target_revisit
        forced_transition_probe = self._build_action_intent_verifier_blocked_forced_transition_probe_decision(
            state=state,
            hints=hints,
            result=payload,
            blocker_hint=blocker_hint,
        )
        if forced_transition_probe is not None:
            return forced_transition_probe
        same_object_active_use_revisit = self._build_action_intent_verifier_blocked_same_object_active_use_revisit_decision(
            state=state,
            hints=hints,
            result=payload,
            blocker_hint=blocker_hint,
        )
        if same_object_active_use_revisit is not None:
            return same_object_active_use_revisit
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
                finalize_long_horizon_revisit = self._build_action_intent_finalize_withheld_long_horizon_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题被 verifier/finalizer 拦下，因为后续用途或最终位置仍未排他；直接沿缓存的更晚目标节点向后追，而不是继续停留在近窗半成品证据上。",
                )
                if finalize_long_horizon_revisit is not None:
                    return finalize_long_horizon_revisit
                initial_transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 判为缺少动作后决定性证据，且当前歧义属于近窗结果型；先直接围绕动作尾部补更密的关键帧，再决定是否扩 followup。",
                )
                if initial_transition_probe is not None and self._action_intent_followup_attempt_count(state) < 1:
                    return initial_transition_probe
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
                finalize_long_horizon_revisit = self._build_action_intent_finalize_withheld_long_horizon_revisit_decision(
                    state=state,
                    hints=hints,
                    thought="why 题被 verifier/finalizer 拦下，因为 pairwise 竞争在更晚用途/最终位置上仍未排他；直接追更晚目标节点，而不是只在当前局部结果附近反复补帧。",
                )
                if finalize_long_horizon_revisit is not None:
                    return finalize_long_horizon_revisit
                initial_transition_probe = self._build_action_intent_transition_probe_decision(
                    state=state,
                    hints=hints,
                    result=payload,
                    thought="why 题被 verifier 判为 top-2 close call，且分歧集中在动作后立刻结果；先直接补尾部密采样关键帧，再决定是否做泛化 followup。",
                )
                if initial_transition_probe is not None and self._action_intent_followup_attempt_count(state) < 1:
                    return initial_transition_probe
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
        finalize_mixed_horizon_later_target_revisit = self._build_action_intent_finalize_withheld_mixed_horizon_later_target_revisit_decision(
            state=state,
            hints=hints,
            thought="why 题专用裁决被 verifier/finalizer 拦下，因为 `check/open` 这类近窗解释还没压过更晚结果；继续追 mixed-horizon 竞争里更晚结果对应的真实目标，优先找真正的后续落点证据。",
        )
        if finalize_mixed_horizon_later_target_revisit is not None:
            return finalize_mixed_horizon_later_target_revisit
        finalize_long_horizon_revisit = self._build_action_intent_finalize_withheld_long_horizon_revisit_decision(
            state=state,
            hints=hints,
            thought="why 题专用裁决被 verifier/finalizer 拦下，因为更晚用途/最终位置仍未排他；继续沿缓存目标节点向后追，优先找真正的后续落点证据。",
        )
        if finalize_long_horizon_revisit is not None:
            return finalize_long_horizon_revisit
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

    def _action_intent_verifier_blocked_prefers_forced_transition_probe(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> bool:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return False
        if self._action_intent_has_transition_followup_frames(state):
            return False
        if blocker_hint not in {"post_action_evidence", "future_use_close_call", "pairwise_close_call"}:
            return False
        if any(
            isinstance(item, str)
            and item.startswith("action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1")
            for item in list(getattr(state, "working_memory", []))[-12:]
        ):
            return True
        needed_profile = self._action_intent_needed_observation_profile(state=state, result=result)
        if needed_profile["prefer_state_change_only"]:
            return True
        support_text = self._action_intent_result_support_text(result)
        combined_text = f"{support_text} {str(result.get('needed_observation') or '').lower()}".strip()
        transition_first_markers = (
            "missing_direct_effect",
            "direct physical effect",
            "display state change",
            "state change",
            "display",
            "readout",
            "tare",
            "zero",
            "turn on",
            "turned on",
            "turned off",
            "opened",
            "closed",
            "reset",
            "开机",
            "归零",
            "显示",
        )
        return any(marker in combined_text for marker in transition_first_markers)

    def _build_action_intent_verifier_blocked_forced_transition_probe_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> PlannerDecision | None:
        if not self._action_intent_verifier_blocked_prefers_forced_transition_probe(
            state=state,
            result=result,
            blocker_hint=blocker_hint,
        ):
            return None
        probe_window = self._action_intent_transition_probe_window(state=state, hints=hints, result=result)
        if probe_window is None:
            return None
        start_time, end_time, stride_s, max_frames = probe_window
        return PlannerDecision(
            thought="why 题被 verifier 拦下后，当前缺口是近窗直接效果/状态变化证据；先强制围绕动作尾部做 `followup_transition` 密采样，确认是否真的出现决定性即时结果，再考虑更晚时域恢复。",
            tool="extract_frames_for_range",
            args={
                "start_time": start_time,
                "end_time": end_time,
                "stride_s": stride_s,
                "max_frames": max_frames,
                "tag": f"{state.task_family}_followup_transition",
            },
        )

    def _action_intent_verifier_blocked_same_object_active_use_hint(
        self,
        *,
        state: AgentState,
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> str:
        if not self._is_action_intent_task(state) or not isinstance(result, dict):
            return ""
        if blocker_hint not in {"post_action_evidence", "future_use_close_call", "pairwise_close_call"}:
            return ""
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        if not choices:
            return ""
        action_object = self._action_intent_question_object_hint(state)
        if not action_object:
            return ""
        best_index = self._coerce_choice_index(result.get("best_index"), choices)
        second_best_index = self._coerce_choice_index(result.get("second_best_index"), choices)
        competitor_index = self._action_intent_competing_candidate_index(result, state)
        candidate_indices: list[int] = []
        for index in (best_index, second_best_index, competitor_index):
            if index is None or index in candidate_indices:
                continue
            candidate_indices.append(index)
        for item in result.get("candidate_evidence") or []:
            if not isinstance(item, dict):
                continue
            index = self._coerce_choice_index(item.get("index"), choices)
            if index is None or index in candidate_indices:
                continue
            candidate_indices.append(index)
        for index in candidate_indices[:4]:
            choice = choices[index]
            if self._action_intent_choice_is_same_object_active_use(choice=choice, action_object=action_object):
                return action_object
        return ""

    def _build_action_intent_verifier_blocked_same_object_active_use_revisit_decision(
        self,
        *,
        state: AgentState,
        hints: dict[str, Any],
        result: dict[str, Any] | None,
        blocker_hint: str,
    ) -> PlannerDecision | None:
        target = self._action_intent_verifier_blocked_same_object_active_use_hint(
            state=state,
            result=result,
            blocker_hint=blocker_hint,
        )
        if not target:
            return None
        nodes = self._latest_action_intent_long_horizon_nodes(state, object_hint=target)
        if not nodes:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 close call 已经涉及同一物体 `{target}` 的后续打开/清洗/继续使用；先重新定位它在更晚时刻的轨迹，而不是退回泛化补帧。",
                tool="query_object",
                args={"query": target, "limit": 24},
            )
        anchor_time = self._latest_action_intent_target_spatial_anchor_time(state)
        latest_followup_end = self._latest_action_intent_followup_end_time(state)
        after_time: float | None = None
        if anchor_time is not None and latest_followup_end is not None:
            after_time = max(anchor_time, latest_followup_end)
        elif latest_followup_end is not None:
            after_time = latest_followup_end
        else:
            after_time = anchor_time
        min_start_time = None if after_time is None else float(after_time) + 0.15
        selected = self._action_intent_select_long_horizon_node(
            state=state,
            hints=hints,
            nodes=nodes,
            min_start_time=min_start_time,
            object_hint=target,
        )
        if selected is None:
            return PlannerDecision(
                thought=f"why 题被 verifier 拦下后，当前 close call 已经涉及同一物体 `{target}` 的后续打开/清洗/继续使用；继续检索它的更晚轨迹，而不是退回泛化补帧。",
                tool="query_object",
                args={"query": target, "limit": 24},
            )
        _node, start_time, end_time = selected
        query_time = start_time if abs(end_time - start_time) < 0.25 else (start_time + min(end_time, start_time + 1.2)) / 2
        return PlannerDecision(
            thought="why 题被 verifier 拦下后，当前 top 候选已涉及同一物体的后续打开/清洗/继续使用；优先直接查看这个物体在更晚时刻的状态，而不是退回泛化补帧。",
            tool="query_spatial_context",
            args={
                "time_s": query_time,
                "object_name": target,
                "limit": 18,
            },
        )

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
