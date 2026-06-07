"""Evidence sufficiency verifier for the graph agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from food_agent.agent.action_intent import (
    action_intent_needs_future_use_resolution,
    action_intent_needs_pairwise_resolution,
    action_intent_needs_precondition_context,
)
from food_agent.agent.state import AgentState
from food_agent.model_client import OpenAICompatibleModelClient


@dataclass(frozen=True)
class VerificationResult:
    sufficient: bool
    confidence: float
    missing_evidence_types: list[str]
    conflicts: list[str]
    recommend_next_action: str
    summary: str


class GraphAgentVerifier:
    """Check whether current evidence is sufficient before allowing finish."""

    def __init__(self, model_client: OpenAICompatibleModelClient | None = None):
        self.model_client = model_client

    def verify(self, *, state: AgentState) -> VerificationResult:
        heuristic = self._heuristic_verify(state)
        if self._prefer_heuristic_verification(state):
            return heuristic
        if self.model_client is None:
            return heuristic
        try:
            refined = self._model_verify(state)
        except Exception:  # noqa: BLE001
            return heuristic
        return self._merge_results(heuristic, refined)

    def detect_conflicts(self, *, state: AgentState) -> list[str]:
        return self._detect_conflicts(state)

    def critique_freeform_answer(self, *, state: AgentState, answer_text: str) -> VerificationResult:
        heuristic = self._heuristic_verify_freeform_answer(state, answer_text=answer_text)
        if self._prefer_heuristic_verification(state):
            return heuristic
        if self.model_client is None:
            return heuristic
        try:
            refined = self._model_verify_freeform_answer(state, answer_text=answer_text)
        except Exception:  # noqa: BLE001
            return heuristic
        return self._merge_results(heuristic, refined)

    def _heuristic_verify(self, state: AgentState) -> VerificationResult:
        missing = [item for item in state.open_questions if item and item != "need_disambiguating_evidence"]
        missing.extend(self._action_intent_missing_grounding_types(state))
        if self._is_weight_task(state) and self._has_stable_weight_answer_evidence(state):
            missing = [item for item in missing if item != "need_alternative_evidence_path"]
        if self._is_object_motion_task(state) and self._has_stable_object_motion_answer_evidence(state):
            irrelevant = {
                "need_region_grounding",
                "need_location_evidence",
                "need_state_evidence",
                "need_alternative_evidence_path",
                "need_time_localization",
            }
            missing = [item for item in missing if item not in irrelevant]
        if self._is_action_intent_task(state) and self._has_stable_structured_family_answer_evidence(state):
            irrelevant = {
                "need_location_evidence",
                "need_time_localization",
                "need_initial_observation",
                "need_ocr_reading",
                "need_region_grounding",
                "need_state_evidence",
                "need_alternative_evidence_path",
                "conflict:conflicting_locations",
                "conflict:conflicting_state_observations",
            }
            missing = [item for item in missing if item not in irrelevant]
        elif self._has_stable_structured_family_answer_evidence(state):
            irrelevant = {
                "need_location_evidence",
                "need_time_localization",
                "need_initial_observation",
                "need_ocr_reading",
                "need_region_grounding",
                "need_state_evidence",
                "need_alternative_evidence_path",
            }
            missing = [item for item in missing if item not in irrelevant]
        missing.extend(self._open_query_missing_grounding_types(state))
        missing = list(dict.fromkeys(item for item in missing if item))
        conflicts = self._detect_conflicts(state)
        conflicts.extend(self._open_query_claim_conflicts(state))
        conflicts = self._filter_non_blocking_conflicts(state, conflicts)
        conflicts = list(dict.fromkeys(item for item in conflicts if item))
        evidence_count = len(state.evidence_bundle)
        sufficient = not missing and not conflicts and evidence_count > 0
        confidence = min(0.95, 0.3 + 0.08 * evidence_count)
        if missing:
            confidence = min(confidence, 0.45)
        if conflicts:
            confidence = min(confidence, 0.25)
        summary = (
            f"sufficient={sufficient}; missing={missing}; conflicts={conflicts}; "
            f"evidence_count={evidence_count}; open_query_family={self._open_query_family(state)}"
        )
        recommend = "finish" if sufficient else (missing[0] if missing else (conflicts[0] if conflicts else "resolve_conflict"))
        return VerificationResult(
            sufficient=sufficient,
            confidence=confidence,
            missing_evidence_types=missing,
            conflicts=conflicts,
            recommend_next_action=recommend,
            summary=summary,
        )

    def _is_weight_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredient_weight"

    def _is_viewpoint_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {"3d_perception_fixture_location", "gaze_gaze_estimation"}

    def _is_object_motion_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")).startswith("object_motion_")

    def _is_ingredient_retrieval_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "ingredient_ingredient_retrieval"

    def _is_action_intent_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) == "fine_grained_why_recognition"

    def _is_structured_family_task(self, state: AgentState) -> bool:
        return str(getattr(state, "task_family", "")) in {
            "ingredient_ingredient_retrieval",
            "ingredient_ingredient_recognition",
            "ingredient_exact_ingredient_recognition",
            "ingredient_ingredients_order",
            "fine_grained_action_localization",
            "fine_grained_action_recognition",
            "fine_grained_how_recognition",
            "fine_grained_why_recognition",
            "recipe_recipe_recognition",
            "recipe_multi_recipe_recognition",
            "nutrition_video_nutrition_estimation",
            "3d_perception_object_location",
            "3d_perception_object_contents_retrieval",
            "gaze_interaction_anticipation",
        }

    def _is_open_query_task(self, state: AgentState) -> bool:
        return self._open_query_family(state) != "" or str(getattr(state, "task_family", "") or "") == "open_query"

    def _open_query_family(self, state: AgentState) -> str:
        task_family = str(getattr(state, "task_family", "") or "")
        if task_family.startswith("open_query_"):
            return task_family
        return ""

    def _prefer_heuristic_verification(self, state: AgentState) -> bool:
        return (
            self._is_weight_task(state)
            or self._is_viewpoint_task(state)
            or self._is_object_motion_task(state)
            or self._is_structured_family_task(state)
            or self._is_open_query_task(state)
        )

    def _has_stable_weight_answer_evidence(self, state: AgentState) -> bool:
        return bool(self._resolve_weight_choice_from_state(state))

    def _has_stable_object_motion_answer_evidence(self, state: AgentState) -> bool:
        if not self._is_object_motion_task(state):
            return False
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            if item.startswith("movement_count=") and "best_index=" in item:
                return True
            if item.startswith("stationary_best_index="):
                return True
        return False

    def _has_stable_structured_family_answer_evidence(self, state: AgentState) -> bool:
        if str(getattr(state, "task_family", "")) == "fine_grained_why_recognition":
            if self._action_intent_has_pending_evidence_gap(state):
                return False
            if self._has_action_intent_textual_rank_fallback_answer(state):
                return True
            if not self._action_intent_has_sufficient_grounding_for_stable_answer(state):
                return False
        prefixes = (
            "ingredient_retrieval_best_index=",
            "recipe_membership_best_index=",
            "exact_ingredient_amount_best_index=",
            "ingredient_order_best_index=",
            "temporal_localization_best_index=",
            "visual_mcq_best_index=",
            "action_mechanism_best_index=",
            "action_intent_best_index=",
            "recipe_catalog_best_index=",
            "recipe_nutrition_best_index=",
            "object_location_best_index=",
            "fixture_direction_best_index=",
            "gaze_best_index=",
            "viewpoint_best_index=",
            "itinerary_best_index=",
        )
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if isinstance(item, str) and item.startswith(prefixes):
                return True
            if isinstance(item, str) and item.startswith("movement_count=") and "best_index=" in item:
                return True
            if isinstance(item, str) and item.startswith("stationary_best_index="):
                return True
        return False

    def _has_action_intent_textual_rank_fallback_answer(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        has_ranked_best_index = any(
            isinstance(item, str) and item.startswith("ranked_best_index=")
            for item in list(state.working_memory) + list(state.evidence_bundle)
        )
        if not has_ranked_best_index:
            return False
        infer_failures = 0
        for call in list(getattr(state, "tool_trace", []) or []):
            if not isinstance(call, dict) or str(call.get("tool") or "") != "infer_action_intent":
                continue
            raw_result = call.get("raw_result")
            if isinstance(raw_result, dict) and raw_result.get("tool_failed"):
                infer_failures += 1
        return infer_failures >= 3

    def _action_intent_has_pending_evidence_gap(self, state: AgentState) -> bool:
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            if item.startswith("action_intent_need_future_evidence=1"):
                return True
            if item.startswith("action_intent_pending_resolution="):
                return True
        if "need_disambiguating_evidence" in list(getattr(state, "open_questions", []) or []):
            return True
        return False

    def _action_intent_missing_grounding_types(self, state: AgentState) -> list[str]:
        if not self._is_action_intent_task(state):
            return []
        if self._has_action_intent_textual_rank_fallback_answer(state):
            return []
        missing: list[str] = []
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        question = str(getattr(state, "question", "") or "")
        if self._action_intent_has_pending_evidence_gap(state):
            missing.append("need_disambiguating_evidence")
        if not self._action_intent_has_successful_specialized_resolution(state):
            if (
                action_intent_needs_precondition_context(question=question, choices=choices, indices=None)
                and not self._action_intent_has_precondition_grounding(state)
            ):
                missing.append("need_precondition_context")
            if (
                (
                    action_intent_needs_future_use_resolution(question=question, choices=choices, indices=None)
                    or action_intent_needs_pairwise_resolution(question=question, choices=choices, indices=None)
                )
                and not self._action_intent_has_post_action_grounding(state)
            ):
                missing.append("need_post_action_evidence")
        return missing

    def _action_intent_has_sufficient_grounding_for_stable_answer(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return True
        if self._action_intent_has_successful_specialized_resolution(state):
            return True
        choices = [str(choice) for choice in getattr(state, "choices", [])]
        question = str(getattr(state, "question", "") or "")
        if action_intent_needs_precondition_context(question=question, choices=choices, indices=None):
            if not self._action_intent_has_precondition_grounding(state):
                return False
        if (
            action_intent_needs_future_use_resolution(question=question, choices=choices, indices=None)
            or action_intent_needs_pairwise_resolution(question=question, choices=choices, indices=None)
        ):
            if not self._action_intent_has_post_action_grounding(state):
                return False
        return True

    def _action_intent_has_successful_specialized_resolution(self, state: AgentState) -> bool:
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            if item.startswith(
                (
                    "action_intent_pairwise_reason=",
                    "action_intent_future_use_reason=",
                    "action_intent_future_use_observation=",
                    "action_intent_unresolved_rerank_best_index=",
                    "action_intent_prior_direct_override_best_index=",
                    "action_intent_causal_override_best_index=",
                    "action_intent_exact_use_override_best_index=",
                    "action_intent_hidden_target_override_best_index=",
                )
            ):
                return True
        for call in list(getattr(state, "tool_trace", []) or []):
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool") or "")
            if tool not in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
                continue
            raw_result = call.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            if raw_result.get("tool_failed"):
                continue
            if raw_result.get("best_index") is None:
                continue
            if bool(raw_result.get("need_more_evidence")):
                continue
            return True
        return False

    def _action_intent_has_precondition_grounding(self, state: AgentState) -> bool:
        for path in list(getattr(state, "retrieved_frames", []) or []):
            name = Path(str(path)).name.lower()
            if "_precontext" in name:
                return True
        precondition_terms = (
            "wet hands",
            "wet-hand",
            "dry hands",
            "hand drying",
            "wipe",
            "wiping",
            "surface",
            "counter",
            "worktop",
            "washed",
            "wash",
            "rinsed",
            "sink",
            "water",
            "hot",
            "burn",
            "spill",
            "dirty",
            "messy",
            "擦手",
            "干手",
            "湿手",
            "擦台面",
            "台面",
            "清洁",
            "清洗",
            "水槽",
        )
        combined = " ".join(str(item) for item in list(state.evidence_bundle) + list(state.working_memory)).lower()
        return any(term in combined for term in precondition_terms)

    def _action_intent_has_post_action_grounding(self, state: AgentState) -> bool:
        for path in list(getattr(state, "retrieved_frames", []) or []):
            name = Path(str(path)).name.lower()
            if "_followup" in name:
                return True
        post_action_terms = (
            "timeline_event",
            "future_use_observation",
            "pairwise_reason",
            "picked up",
            "put on the scale",
            "used again",
            "retrieved before",
            "after putting",
            "shortly after",
            "next step",
            "follow-up",
            "followup",
        )
        combined = " ".join(str(item) for item in list(state.evidence_bundle) + list(state.working_memory)).lower()
        return any(term in combined for term in post_action_terms)

    def _filter_non_blocking_conflicts(self, state: AgentState, conflicts: list[str]) -> list[str]:
        if self._is_weight_task(state) and self._has_stable_weight_answer_evidence(state):
            return [
                item
                for item in conflicts
                if item not in {"conflicting_locations", "conflicting_state_observations"}
            ]
        if self._is_ingredient_retrieval_task(state) and self._has_stable_structured_family_answer_evidence(state):
            return [
                item
                for item in conflicts
                if item not in {"conflicting_locations", "conflicting_state_observations"}
            ]
        if self._is_action_intent_task(state) and self._has_stable_structured_family_answer_evidence(state):
            return [
                item
                for item in conflicts
                if item not in {"conflicting_locations", "conflicting_state_observations"}
            ]
        return conflicts

    def _resolve_weight_choice_from_state(self, state: AgentState) -> tuple[int, str, float] | None:
        choice_values: list[tuple[int, float, str]] = []
        for index, choice in enumerate(state.choices):
            parsed = self._parse_numeric_value(str(choice))
            if parsed is None:
                continue
            choice_values.append((index, parsed, str(choice)))
        if not choice_values:
            return None
        measurement_values = self._extract_prefixed_numeric_values(state, prefix="normalized=", measurement_only=True)
        if measurement_values:
            best = self._pick_best_numeric_choice(choice_values, measurement_values[-1])
            if best is not None:
                return best[0], best[2], 0.9
        ocr_values = self._extract_prefixed_numeric_values(state, prefix="ocr_reading=", measurement_only=False)
        if ocr_values:
            best = self._pick_best_numeric_choice(choice_values, ocr_values[-1])
            if best is not None:
                return best[0], best[2], 0.82
        return None

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

    def _pick_best_numeric_choice(
        self,
        choices: list[tuple[int, float, str]],
        target_value: float,
    ) -> tuple[int, float, str] | None:
        ranked = sorted(choices, key=lambda item: (abs(item[1] - target_value), item[0]))
        return ranked[0] if ranked else None

    def _parse_numeric_value(self, text: str) -> float | None:
        import re

        match = re.search(r"(\d+(?:\.\d+)?)", str(text))
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:  # noqa: BLE001
            return None

    def _model_verify(self, state: AgentState) -> VerificationResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是视频问答 agent 的证据验证器。"
                    "只判断当前证据是否足够支持最终回答。"
                    "不要回答题目本身。"
                    '输出 JSON: {"sufficient":false,"confidence":0.0,"missing_evidence_types":[],"conflicts":[],"recommend_next_action":"","summary":""}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_family": state.task_family,
                        "question": state.question,
                        "choices": state.choices,
                        "evidence_bundle": state.evidence_bundle[-20:],
                        "working_memory": state.working_memory[-20:],
                        "hypotheses": state.hypotheses[-20:],
                        "open_questions": state.open_questions[-20:],
                        "tool_failures": state.tool_failures[-10:],
                        "ineffective_tools": state.ineffective_tools[-10:],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        payload = self.model_client.complete_json(messages, temperature=0.0)
        return VerificationResult(
            sufficient=bool(payload.get("sufficient")),
            confidence=float(payload.get("confidence") or 0.0),
            missing_evidence_types=[str(item) for item in payload.get("missing_evidence_types", []) if item],
            conflicts=[str(item) for item in payload.get("conflicts", []) if item],
            recommend_next_action=str(payload.get("recommend_next_action") or ""),
            summary=str(payload.get("summary") or ""),
        )

    def _heuristic_verify_freeform_answer(self, state: AgentState, *, answer_text: str) -> VerificationResult:
        base = self._heuristic_verify(state)
        normalized_answer = str(answer_text or "").strip()
        if not normalized_answer:
            missing = list(dict.fromkeys(base.missing_evidence_types + ["need_grounded_freeform_answer"]))
            return VerificationResult(
                sufficient=False,
                confidence=min(base.confidence, 0.2),
                missing_evidence_types=missing,
                conflicts=base.conflicts,
                recommend_next_action=missing[0],
                summary=f"{base.summary}; answer_critic=empty_answer",
            )
        missing = list(base.missing_evidence_types)
        conflicts = list(base.conflicts)
        answer_issues = self._freeform_answer_grounding_issues(state, answer_text=normalized_answer)
        missing.extend(answer_issues["missing"])
        conflicts.extend(answer_issues["conflicts"])
        missing = list(dict.fromkeys(item for item in missing if item))
        conflicts = list(dict.fromkeys(item for item in conflicts if item))
        sufficient = not missing and not conflicts
        confidence = min(base.confidence, 0.9 if sufficient else 0.35)
        if conflicts:
            confidence = min(confidence, 0.25)
        recommend = "finish" if sufficient else (missing[0] if missing else (conflicts[0] if conflicts else "need_grounded_freeform_answer"))
        return VerificationResult(
            sufficient=sufficient,
            confidence=confidence,
            missing_evidence_types=missing,
            conflicts=conflicts,
            recommend_next_action=recommend,
            summary=f"{base.summary}; answer_critic_missing={missing}; answer_critic_conflicts={conflicts}",
        )

    def _model_verify_freeform_answer(self, state: AgentState, *, answer_text: str) -> VerificationResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是视频问答 agent 的最终答案审查器。"
                    "只判断给定开放回答是否被当前证据支持，是否超出证据范围，是否答非所问。"
                    '输出 JSON: {"sufficient":false,"confidence":0.0,"missing_evidence_types":[],"conflicts":[],"recommend_next_action":"","summary":""}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_family": state.task_family,
                        "question": state.question,
                        "answer_text": answer_text,
                        "evidence_bundle": state.evidence_bundle[-20:],
                        "working_memory": state.working_memory[-20:],
                        "open_questions": state.open_questions[-20:],
                        "verification_history": state.verification_history[-5:],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        payload = self.model_client.complete_json(messages, temperature=0.0)
        return VerificationResult(
            sufficient=bool(payload.get("sufficient")),
            confidence=float(payload.get("confidence") or 0.0),
            missing_evidence_types=[str(item) for item in payload.get("missing_evidence_types", []) if item],
            conflicts=[str(item) for item in payload.get("conflicts", []) if item],
            recommend_next_action=str(payload.get("recommend_next_action") or ""),
            summary=str(payload.get("summary") or ""),
        )

    def _merge_results(self, heuristic: VerificationResult, refined: VerificationResult) -> VerificationResult:
        missing = list(dict.fromkeys(heuristic.missing_evidence_types + refined.missing_evidence_types))
        conflicts = list(dict.fromkeys(heuristic.conflicts + refined.conflicts))
        sufficient = heuristic.sufficient and refined.sufficient and not missing and not conflicts
        confidence = min(heuristic.confidence, refined.confidence) if not sufficient else max(heuristic.confidence, refined.confidence)
        recommend = refined.recommend_next_action or heuristic.recommend_next_action
        summary = refined.summary or heuristic.summary
        return VerificationResult(
            sufficient=sufficient,
            confidence=confidence,
            missing_evidence_types=missing,
            conflicts=conflicts,
            recommend_next_action=recommend,
            summary=summary,
        )

    def _detect_conflicts(self, state: AgentState) -> list[str]:
        candidate_indices = {
            item.split("=", 1)[1]
            for item in state.hypotheses
            if isinstance(item, str) and item.startswith("candidate_answer_index=")
        }
        conflicts: list[str] = []
        if len(candidate_indices) > 1:
            conflicts.append("multiple_candidate_answers")
        ocr_readings = self._extract_prefixed_values(state, prefixes=("ocr_reading=",), separators=(";", "|"))
        if len(ocr_readings) > 1:
            conflicts.append("conflicting_ocr_readings")
        locations = self._extract_prefixed_values(
            state,
            prefixes=("target_location=", "scene_location="),
            separators=(";", "|"),
        )
        if len(locations) > 1:
            conflicts.append("conflicting_locations")
        state_hints = self._extract_prefixed_values(
            state,
            prefixes=("state_change_hint=", "after_state=", "before_state="),
            separators=(";", "|"),
        )
        if len(state_hints) > 1:
            conflicts.append("conflicting_state_observations")
        return conflicts

    def _open_query_missing_grounding_types(self, state: AgentState) -> list[str]:
        family = self._open_query_family(state)
        if not family:
            return []
        missing: list[str] = []
        if family == "open_query_ocr":
            if not self._has_grounded_ocr_answer_evidence(state):
                missing.append("need_grounded_ocr_answer")
        elif family == "open_query_location":
            if not self._has_grounded_location_answer_evidence(state):
                missing.append("need_grounded_location_answer")
        elif family == "open_query_state":
            if not self._has_grounded_state_answer_evidence(state):
                missing.append("need_grounded_state_answer")
        elif family == "open_query_temporal_summary":
            if not self._has_grounded_temporal_summary_evidence(state):
                missing.append("need_grounded_temporal_summary")
        return missing

    def _open_query_claim_conflicts(self, state: AgentState) -> list[str]:
        family = self._open_query_family(state)
        if not family:
            return []
        conflicts: list[str] = []
        if family == "open_query_ocr" and len(self._extract_prefixed_values(state, prefixes=("ocr_reading=", "ocr_text="), separators=(";", "|"))) > 1:
            conflicts.append("conflicting_ocr_readings")
        if family == "open_query_location" and len(
            self._extract_prefixed_values(state, prefixes=("target_location=", "scene_location="), separators=(";", "|"))
        ) > 1:
            conflicts.append("conflicting_locations")
        if family == "open_query_state" and len(
            self._extract_prefixed_values(state, prefixes=("state_change_hint=", "after_state=", "before_state="), separators=(";", "|"))
        ) > 1:
            conflicts.append("conflicting_state_observations")
        if family == "open_query_temporal_summary" and not self._has_grounded_temporal_summary_evidence(state):
            values = self._extract_prefixed_values(
                state,
                prefixes=("timeline_event", "possible_step=", "ongoing_action=", "state_change_hint="),
                separators=(";", "|"),
            )
            if len(values) > 1 and not self._has_temporal_anchor_evidence(state):
                conflicts.append("weak_temporal_summary_grounding")
        return conflicts

    def _freeform_answer_grounding_issues(self, state: AgentState, *, answer_text: str) -> dict[str, list[str]]:
        family = self._answer_critic_family(state)
        answer = answer_text.lower()
        missing: list[str] = []
        conflicts: list[str] = []
        if family == "open_query_ocr":
            values = self._extract_prefixed_values(state, prefixes=("ocr_reading=", "ocr_text="), separators=(";", "|"))
            if not values:
                missing.append("need_grounded_ocr_answer")
            elif not any(value in answer for value in values):
                conflicts.append("answer_not_grounded_to_ocr_evidence")
        elif family == "open_query_location":
            values = self._extract_prefixed_values(state, prefixes=("target_location=", "scene_location="), separators=(";", "|"))
            if not values:
                missing.append("need_grounded_location_answer")
            elif not any(value in answer for value in values):
                conflicts.append("answer_not_grounded_to_location_evidence")
        elif family == "open_query_state":
            values = self._extract_prefixed_values(
                state,
                prefixes=("state_change_hint=", "after_state=", "before_state="),
                separators=(";", "|"),
            )
            if not values and not self._has_grounded_state_answer_evidence(state):
                missing.append("need_grounded_state_answer")
            elif values and not any(value in answer for value in values):
                conflicts.append("answer_not_grounded_to_state_evidence")
        elif family == "open_query_temporal_summary":
            if not self._has_grounded_temporal_summary_evidence(state):
                missing.append("need_grounded_temporal_summary")
            action_values = self._extract_prefixed_values(
                state,
                prefixes=("ongoing_action=", "possible_step=", "answer_hint="),
                separators=(";", "|"),
            )
            state_values = self._extract_prefixed_values(
                state,
                prefixes=("state_change_hint=", "after_state=", "before_state="),
                separators=(";", "|"),
            )
            grounded_values = sorted(action_values | state_values)
            if grounded_values and not any(value in answer for value in grounded_values):
                conflicts.append("answer_not_grounded_to_temporal_evidence")
        if state.question:
            question = state.question.lower()
            if "where" in question and family != "open_query_location" and not any(token in answer for token in ("left", "right", "counter", "sink", "bowl", "pan")):
                conflicts.append("answer_not_responsive_to_question")
            if any(token in question for token in ("what happened", "after", "before")) and family == "open_query_temporal_summary":
                if not any(token in answer for token in ("发生", "主要", "状态", "动作", "stir", "mix", "add", "pour", "cook")):
                    conflicts.append("answer_not_responsive_to_question")
        return {"missing": missing, "conflicts": conflicts}

    def _answer_critic_family(self, state: AgentState) -> str:
        family = self._open_query_family(state)
        if family:
            return family
        task_family = str(getattr(state, "task_family", "") or "")
        if task_family == "open_query":
            question = str(getattr(state, "question", "") or "").lower()
            if any(token in question for token in ("read", "reading", "number", "digit", "text", "label", "scale")):
                return "open_query_ocr"
            if any(token in question for token in ("where", "location", "left", "right", "front", "behind", "near")):
                return "open_query_location"
            if any(token in question for token in ("state", "change", "mixed", "raw", "cooked", "done", "become")):
                return "open_query_state"
            return "open_query_temporal_summary"
        return ""

    def _has_grounded_ocr_answer_evidence(self, state: AgentState) -> bool:
        return bool(self._extract_prefixed_values(state, prefixes=("ocr_reading=", "ocr_text="), separators=(";", "|")))

    def _has_grounded_location_answer_evidence(self, state: AgentState) -> bool:
        return bool(self._extract_prefixed_values(state, prefixes=("target_location=", "scene_location="), separators=(";", "|")))

    def _has_grounded_state_answer_evidence(self, state: AgentState) -> bool:
        if self._extract_prefixed_values(state, prefixes=("state_change_hint=", "after_state=", "before_state="), separators=(";", "|")):
            return True
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if isinstance(item, str) and "type=state_change" in item:
                return True
        return False

    def _has_grounded_temporal_summary_evidence(self, state: AgentState) -> bool:
        has_temporal = self._has_temporal_anchor_evidence(state)
        has_action_or_state = False
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            if any(token in item for token in ("ongoing_action=", "possible_step=", "state_change_hint=", "timeline_event")):
                has_action_or_state = True
                break
        return has_temporal and has_action_or_state

    def _has_temporal_anchor_evidence(self, state: AgentState) -> bool:
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            if any(token in item for token in ("time=", "start=", "end=", "before=", "after=", "at ", "timeline_event")):
                return True
        if getattr(state, "visited_times", None):
            return bool(state.visited_times)
        return False

    def _extract_prefixed_values(
        self,
        state: AgentState,
        *,
        prefixes: tuple[str, ...],
        separators: tuple[str, ...],
    ) -> set[str]:
        values: set[str] = set()
        for item in list(state.evidence_bundle) + list(state.working_memory):
            if not isinstance(item, str):
                continue
            for prefix in prefixes:
                if prefix not in item:
                    continue
                tail = item.split(prefix, 1)[1]
                for separator in separators:
                    if separator in tail:
                        tail = tail.split(separator, 1)[0]
                normalized = tail.strip().lower()
                if normalized:
                    values.add(normalized)
        return values
