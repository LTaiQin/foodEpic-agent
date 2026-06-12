"""Evidence sufficiency verifier for the graph agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    sufficiency_decision: dict[str, Any] = field(default_factory=dict)
    action_intent_hypotheses: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceGap:
    gap_type: str
    priority: str
    missing_observation: str
    target_object: str = ""
    target_fixture: str = ""
    time_relation: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "gap_type": self.gap_type,
            "priority": self.priority,
            "missing_observation": self.missing_observation,
            "target_object": self.target_object,
            "target_fixture": self.target_fixture,
            "time_relation": self.time_relation,
            "source": self.source,
        }


@dataclass(frozen=True)
class SufficiencyDecision:
    sufficient: bool
    missing_gap_types: list[str]
    blocking_hypotheses: list[str]
    blocking_comparisons: list[dict[str, Any]]
    recommended_next_step: str
    finish_mode: str
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "missing_gap_types": list(self.missing_gap_types),
            "blocking_hypotheses": list(self.blocking_hypotheses),
            "blocking_comparisons": [
                dict(item) for item in self.blocking_comparisons if isinstance(item, dict)
            ],
            "recommended_next_step": self.recommended_next_step,
            "finish_mode": self.finish_mode,
            "summary": self.summary,
        }


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
            }
            if self._action_intent_can_suppress_secondary_conflicts(state):
                irrelevant.update(
                    {
                        "conflict:conflicting_locations",
                        "conflict:conflicting_state_observations",
                    }
                )
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
        evidence_gaps = self._action_intent_build_evidence_gaps(
            state=state,
            missing=missing,
            blocker_hint=self._action_intent_verifier_blocker(state, missing=missing),
        )
        action_intent_hypotheses = self._build_action_intent_hypotheses(
            state=state,
            missing=missing,
            conflicts=conflicts,
            evidence_gaps=evidence_gaps,
        )
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
        why_blocker = self._action_intent_verifier_blocker(state, missing=missing)
        if why_blocker:
            summary = f"{summary}; why_blocker={why_blocker}"
        recommend = "finish" if sufficient else (missing[0] if missing else (conflicts[0] if conflicts else "resolve_conflict"))
        evidence_gaps = self._action_intent_build_evidence_gaps(
            state=state,
            missing=missing,
            blocker_hint=why_blocker,
        )
        action_intent_hypotheses = self._build_action_intent_hypotheses(
            state=state,
            missing=missing,
            conflicts=conflicts,
            evidence_gaps=evidence_gaps,
        )
        sufficiency_decision = self._build_sufficiency_decision(
            state=state,
            sufficient=sufficient,
            missing=missing,
            conflicts=conflicts,
            recommend_next_action=recommend,
            summary=summary,
            evidence_gaps=evidence_gaps,
            action_intent_hypotheses=action_intent_hypotheses,
        )
        return VerificationResult(
            sufficient=sufficient,
            confidence=confidence,
            missing_evidence_types=missing,
            conflicts=conflicts,
            recommend_next_action=recommend,
            summary=summary,
            evidence_gaps=evidence_gaps,
            sufficiency_decision=sufficiency_decision,
            action_intent_hypotheses=action_intent_hypotheses,
        )

    def _action_intent_verifier_blocker(self, state: AgentState, *, missing: list[str]) -> str:
        if not self._is_action_intent_task(state):
            return ""
        missing_set = {str(item) for item in missing if item}
        if "need_precondition_context" in missing_set:
            return "precondition_context"
        if "need_post_action_evidence" in missing_set:
            latest = self._latest_action_intent_resolution_payload(state)
            if latest is None:
                return "post_action_evidence"
            _tool_name, payload = latest
            blocker_family = self._action_intent_payload_blocker_family(state=state, payload=payload)
            if blocker_family:
                return blocker_family
            return "post_action_evidence"
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is None:
            return ""
        _tool_name, payload = latest
        if not self._action_intent_has_plausible_competing_candidate_gap(state):
            return ""
        return self._action_intent_payload_blocker_family(state=state, payload=payload)

    def _action_intent_build_evidence_gaps(
        self,
        *,
        state: AgentState,
        missing: list[str],
        blocker_hint: str,
    ) -> list[dict[str, Any]]:
        if not self._is_action_intent_task(state):
            return []
        latest = self._latest_action_intent_resolution_payload(state)
        payload: dict[str, Any] = {}
        if latest is not None:
            _tool_name, payload = latest
        missing_set = {str(item) for item in missing if item}
        gaps: list[EvidenceGap] = []
        source = self._action_intent_gap_source(
            state=state,
            blocker_hint=blocker_hint,
            payload=payload,
        )
        target_object = self._action_intent_gap_target_object(state=state)
        target_fixture = self._action_intent_gap_target_fixture(state=state)

        immediate_gap_needed = (
            "need_post_action_evidence" in missing_set
            and blocker_hint == "post_action_evidence"
        ) or any(
            isinstance(item, str)
            and item.startswith("action_intent_resolution_withheld_for_missing_direct_outcome_evidence=1")
            for item in list(getattr(state, "working_memory", []))[-16:]
        )
        if immediate_gap_needed:
            gaps.append(
                EvidenceGap(
                    gap_type="immediate_outcome",
                    priority="high",
                    missing_observation=self._action_intent_observation_gap_text(
                        state=state,
                        gap_type="immediate_outcome",
                        target_object=target_object,
                        target_fixture=target_fixture,
                    ),
                    target_object=target_object,
                    target_fixture=target_fixture,
                    time_relation="after_action",
                    source=source,
                )
            )

        future_gap_needed = bool(payload.get("need_future_evidence")) or (
            bool(payload.get("need_more_evidence"))
            and bool(self._action_intent_payload_blocker_family(state=state, payload=payload))
        )
        if future_gap_needed:
            gaps.append(
                EvidenceGap(
                    gap_type="future_outcome",
                    priority="high",
                    missing_observation=self._action_intent_observation_gap_text(
                        state=state,
                        gap_type="future_outcome",
                        target_object=target_object,
                        target_fixture=target_fixture,
                    ),
                    target_object=target_object,
                    target_fixture=target_fixture,
                    time_relation="later_after_action",
                    source=source,
                )
            )
        if "need_location_evidence" in missing_set:
            gaps.append(
                EvidenceGap(
                    gap_type="target_discovery",
                    priority="medium",
                    missing_observation=self._action_intent_observation_gap_text(
                        state=state,
                        gap_type="target_discovery",
                        target_object=target_object,
                        target_fixture=target_fixture,
                    ),
                    target_object=target_object,
                    target_fixture=target_fixture,
                    time_relation="around_or_after_action",
                    source=source or "need_location_evidence",
                )
            )
        if "need_disambiguating_evidence" in missing_set and not gaps:
            gaps.append(
                EvidenceGap(
                    gap_type="relation_confirmation",
                    priority="medium",
                    missing_observation=self._action_intent_observation_gap_text(
                        state=state,
                        gap_type="relation_confirmation",
                        target_object=target_object,
                        target_fixture=target_fixture,
                    ),
                    target_object=target_object,
                    target_fixture=target_fixture,
                    time_relation="around_or_after_action",
                    source=source or "need_disambiguating_evidence",
                )
            )
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for gap in gaps:
            key = (gap.gap_type, gap.missing_observation, gap.target_object, gap.target_fixture)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(gap.as_dict())
        return deduped

    def _action_intent_gap_source(
        self,
        *,
        state: AgentState,
        blocker_hint: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        if blocker_hint == "precondition_context":
            return "precondition_gap"
        if self._action_intent_recent_observation_focus(state):
            return "verification_gap"
        if self._action_intent_payload_blocker_family(state=state, payload=payload or {}):
            return "resolution_followup_gap"
        return "verification_gap"

    def _action_intent_payload_blocker_family(
        self,
        *,
        state: AgentState,
        payload: dict[str, Any] | None,
    ) -> str:
        if not self._is_action_intent_task(state) or not isinstance(payload, dict):
            return ""
        if payload.get("best_index") is None or payload.get("tool_failed"):
            return ""
        if not bool(payload.get("need_more_evidence")) and not bool(payload.get("need_future_evidence")):
            return ""
        primary_gap_type = self._action_intent_primary_gap_type_from_state(state)
        if primary_gap_type == "future_outcome":
            return "future_gap_family"
        if primary_gap_type in {
            "immediate_outcome",
            "relation_confirmation",
            "target_discovery",
            "state_transition_unconfirmed",
            "workspace_change_unconfirmed",
        }:
            return "post_action_evidence"
        if self._action_intent_payload_lacks_direct_post_action_evidence(payload):
            return "post_action_evidence"
        if self._action_intent_payload_points_to_later_outcome_uncertainty(payload):
            return "future_gap_family"
        return ""

    def _action_intent_primary_gap_type_from_state(self, state: AgentState) -> str:
        latest = self._state_latest_verification(state)
        decision = latest.get("sufficiency_decision")
        if isinstance(decision, dict):
            recommended = str(decision.get("recommended_next_step") or "").strip()
            missing_gap_types = [
                str(item).strip()
                for item in decision.get("missing_gap_types", [])
                if isinstance(item, str) and str(item).strip()
            ]
            selected = self._action_intent_select_primary_gap_type(
                missing_gap_types=missing_gap_types,
                recommended_next_step=recommended,
            )
            if selected:
                return selected
        return ""

    def _state_latest_verification(self, state: AgentState) -> dict[str, Any]:
        history = list(getattr(state, "verification_history", []) or [])
        if not history:
            return {}
        latest = history[-1]
        return latest if isinstance(latest, dict) else {}

    def _action_intent_select_primary_gap_type(
        self,
        *,
        missing_gap_types: list[str],
        recommended_next_step: str,
    ) -> str:
        candidates = {
            str(item).strip()
            for item in missing_gap_types
            if isinstance(item, str) and str(item).strip()
        }
        if not candidates:
            return ""
        preferred_by_step = {
            "need_precondition_context": ("precondition",),
            "need_post_action_evidence": ("immediate_outcome", "future_outcome"),
            "need_location_evidence": ("target_discovery", "relation_confirmation", "future_outcome"),
            "need_disambiguating_evidence": ("relation_confirmation", "target_discovery", "future_outcome", "immediate_outcome"),
        }
        for candidate in preferred_by_step.get(recommended_next_step, ()):
            if candidate in candidates:
                return candidate
        for candidate in (
            "precondition",
            "immediate_outcome",
            "future_outcome",
            "relation_confirmation",
            "target_discovery",
        ):
            if candidate in candidates:
                return candidate
        return ""

    def _action_intent_payload_lacks_direct_post_action_evidence(self, payload: dict[str, Any]) -> bool:
        support_text = self._action_intent_result_support_text(payload)
        if not support_text:
            return True
        if self._action_intent_text_has_direct_outcome_clause(
            text=support_text,
            strong_result_terms=(
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
                "picked up from behind",
                "taken from behind",
                "retrieved from behind",
                "placed into the freed slot",
                "put into the freed slot",
                "immediately picks up",
                "used again shortly after",
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
            ),
            blocked_terms=(
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
            ),
        ):
            return False
        return True

    def _action_intent_payload_points_to_later_outcome_uncertainty(self, payload: dict[str, Any]) -> bool:
        support_text = self._action_intent_result_support_text(payload)
        if not support_text:
            return False
        later_markers = (
            "later",
            "afterwards",
            "next use",
            "used again",
            "returned to",
            "put back",
            "stored",
            "final location",
            "最终",
            "之后",
            "后续",
            "放回",
            "归位",
        )
        uncertainty_markers = (
            "not shown",
            "not visible",
            "cannot tell",
            "can't tell",
            "unclear",
            "uncertain",
            "missing",
            "lack",
            "still unclear",
            "未显示",
            "没有看到",
            "看不清",
            "不明确",
        )
        return any(term in support_text for term in later_markers) and any(
            term in support_text for term in uncertainty_markers
        )

    def _action_intent_gap_target_object(self, *, state: AgentState) -> str:
        focus = self._action_intent_recent_observation_focus(state)
        related_object = str(focus.get("related_object") or "").strip()
        if related_object:
            return related_object
        subject_object = str(focus.get("subject_object") or "").strip()
        if subject_object:
            return subject_object
        return ""

    def _action_intent_gap_target_fixture(self, *, state: AgentState) -> str:
        focus = self._action_intent_recent_observation_focus(state)
        return str(focus.get("fixture") or "").strip()

    def _action_intent_observation_gap_text(
        self,
        *,
        state: AgentState,
        gap_type: str,
        target_object: str,
        target_fixture: str,
    ) -> str:
        subject_object = str(self._action_intent_recent_observation_focus(state).get("subject_object") or "").strip()
        if target_object and target_object == subject_object:
            target_object = ""
        subject_text = subject_object or "该动作对象"
        if gap_type == "immediate_outcome":
            return f"需要确认{subject_text}在动作后短时间内是否出现可观察的直接结果。"
        if gap_type == "future_outcome":
            if target_object:
                return f"需要确认{subject_text}在更晚时段是否与{target_object}产生明确的后续用途、取放或交互结果。"
            if target_fixture:
                return f"需要确认{subject_text}在更晚时段是否到达{target_fixture}附近并形成明确的最终落点或后续用途。"
            return f"需要确认{subject_text}在更晚时段是否出现明确的后续用途、最终落点或下游结果。"
        if gap_type == "target_discovery":
            if target_object:
                return f"需要确认{target_object}在关键时间窗内是否重新出现，以及相关对象轨迹或位置是否闭合。"
            if target_fixture:
                return f"需要确认与{target_fixture}相关的关键对象或区域在动作前后是否被重新观察到。"
            return "需要确认关键对象或区域在动作前后是否重新出现，以及对象轨迹或位置是否闭合。"
        if gap_type == "relation_confirmation":
            if target_object:
                return f"需要确认{subject_text}与{target_object}之间是否形成了可观察的空间或功能关系。"
            if target_fixture:
                return f"需要确认{subject_text}与{target_fixture}附近区域之间是否形成了可观察的空间或功能关系。"
            return f"需要确认{subject_text}与相关对象或区域之间是否形成了可观察的空间或功能关系。"
        return "需要补充当前动作相关的原始观测证据。"

    def _action_intent_recent_observation_focus(self, state: AgentState) -> dict[str, str]:
        subject_object = self._action_intent_question_subject_object(state)
        related_object = ""
        fixture = ""
        for call in reversed(list(getattr(state, "tool_trace", []) or [])):
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            raw_result = call.get("raw_result") if isinstance(call.get("raw_result"), dict) else {}
            if not related_object and tool == "query_object":
                query_name = str(args.get("query") or "").strip()
                if query_name and query_name != subject_object:
                    related_object = query_name
            if tool == "query_spatial_context":
                if not subject_object:
                    subject_candidate = str(args.get("object_name") or "").strip()
                    if subject_candidate:
                        subject_object = subject_candidate
                if not fixture:
                    fixture = str(args.get("fixture") or "").strip()
                if not related_object:
                    related_object = self._action_intent_extract_related_object_from_spatial_result(raw_result, subject_object)
                if not fixture:
                    fixture = self._action_intent_extract_fixture_from_spatial_result(raw_result)
            if subject_object and related_object and fixture:
                break
        if not fixture:
            fixture = self._action_intent_extract_fixture_from_evidence(state)
        return {
            "subject_object": subject_object,
            "related_object": related_object,
            "fixture": fixture,
        }

    def _action_intent_question_subject_object(self, state: AgentState) -> str:
        question = str(getattr(state, "question", "") or "")
        match = re.search(r"<([^>]+)>", question)
        if not match:
            return ""
        action_text = str(match.group(1) or "").strip().lower()
        tokens = [token for token in re.split(r"[^a-z]+", action_text) if token]
        stopwords = {
            "move",
            "take",
            "pick",
            "put",
            "open",
            "close",
            "flip",
            "turn",
            "tap",
            "lift",
            "slide",
            "pull",
            "push",
            "up",
            "down",
        }
        for token in reversed(tokens):
            if token not in stopwords:
                return token
        return ""

    def _action_intent_extract_related_object_from_spatial_result(
        self,
        raw_result: dict[str, Any],
        subject_object: str,
    ) -> str:
        fixture_markers = ("fridge", "sink", "counter", "worktop", "shelf", "hob", "scale")
        for key in ("relations", "object_tracks", "object_masks"):
            values = raw_result.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                for candidate_key in ("target", "object", "object_name"):
                    candidate = str(item.get(candidate_key) or "").strip()
                    lowered = candidate.lower()
                    if candidate and candidate != subject_object and not any(marker in lowered for marker in fixture_markers):
                        return candidate
        return ""

    def _action_intent_extract_fixture_from_spatial_result(self, raw_result: dict[str, Any]) -> str:
        for key in ("object_masks", "relations"):
            values = raw_result.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("fixture") or item.get("target") or "").strip()
                if candidate and any(marker in candidate.lower() for marker in ("fridge", "sink", "counter", "worktop", "shelf", "hob", "scale")):
                    return candidate
        return ""

    def _action_intent_extract_fixture_from_evidence(self, state: AgentState) -> str:
        combined = " ".join(str(item) for item in list(getattr(state, "evidence_bundle", []))[-12:])
        lowered = combined.lower()
        for candidate in ("fridge", "sink", "counter", "worktop", "shelf", "hob", "scale"):
            if candidate in lowered:
                return candidate
        return ""

    def _action_intent_result_support_text(self, result: dict[str, Any] | None) -> str:
        if not isinstance(result, dict):
            return ""
        return " ".join(
            str(result.get(key) or "")
            for key in (
                "reason",
                "support",
                "decisive_observation",
                "direct_effect",
                "downstream_action",
            )
        ).strip().lower()

    def _build_action_intent_hypotheses(
        self,
        *,
        state: AgentState,
        missing: list[str],
        conflicts: list[str],
        evidence_gaps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return []

    def _build_sufficiency_decision(
        self,
        *,
        state: AgentState | None = None,
        sufficient: bool,
        missing: list[str],
        conflicts: list[str],
        recommend_next_action: str,
        summary: str,
        evidence_gaps: list[dict[str, Any]],
        action_intent_hypotheses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        gap_types = [
            str(item.get("gap_type") or "")
            for item in evidence_gaps
            if isinstance(item, dict) and item.get("gap_type")
        ]
        blocking_hypotheses = list(dict.fromkeys([*missing, *conflicts]))
        blocking_comparisons: list[dict[str, Any]] = []
        has_structured_blockers = bool(blocking_hypotheses)
        budget_exhausted = bool(getattr(state, "is_search_budget_exhausted", lambda: False)()) if state is not None else False
        if sufficient:
            finish_mode = "finish_confident"
        elif budget_exhausted:
            finish_mode = "finish_budget_exhausted_best_guess"
        elif conflicts:
            finish_mode = "resolve_conflict"
        elif gap_types or missing or has_structured_blockers:
            finish_mode = "needs_more_evidence"
        else:
            finish_mode = "finish_insufficient_evidence"
        decision = SufficiencyDecision(
            sufficient=sufficient,
            missing_gap_types=gap_types,
            blocking_hypotheses=list(dict.fromkeys(blocking_hypotheses)),
            blocking_comparisons=[
                item for item in blocking_comparisons if isinstance(item, dict) and item
            ],
            recommended_next_step=str(recommend_next_action or ""),
            finish_mode=finish_mode,
            summary=summary,
        )
        return decision.as_dict()

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
            if self._action_intent_has_plausible_competing_candidate_gap(state):
                return False
            if self._has_action_intent_textual_rank_fallback_answer(state):
                return self._action_intent_textual_rank_fallback_can_finish(state)
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

    def _action_intent_textual_rank_fallback_can_finish(self, state: AgentState) -> bool:
        if not self._has_action_intent_textual_rank_fallback_answer(state):
            return False
        if self._action_intent_has_pending_evidence_gap(state):
            return False
        if not self._action_intent_has_current_task_artifact_grounding(state):
            return False
        primary_gap_type = self._action_intent_primary_gap_type_from_state(state)
        blocker_hint = self._action_intent_verifier_blocker(state, missing=[])
        if primary_gap_type == "precondition" or blocker_hint == "precondition_context":
            if not self._action_intent_has_precondition_grounding(state):
                return False
        if (
            primary_gap_type in {"future_outcome", "relation_confirmation", "target_discovery"}
            or blocker_hint == "future_gap_family"
        ) and not self._action_intent_has_post_action_grounding(state):
            return False
        if (
            primary_gap_type in {"immediate_outcome", "state_transition_unconfirmed", "workspace_change_unconfirmed"}
            or blocker_hint == "post_action_evidence"
        ) and not self._action_intent_has_post_action_grounding(state):
            return False
        if self._action_intent_state_describes_unclosed_post_action_outcome(state):
            return False
        if self._action_intent_has_unresolved_secondary_conflicts(state):
            return False
        return True

    def _action_intent_state_describes_unclosed_post_action_outcome(self, state: AgentState) -> bool:
        text = " ".join(
            str(item)
            for item in list(getattr(state, "evidence_bundle", []) or []) + list(getattr(state, "working_memory", []) or [])
            if isinstance(item, str) and item
        ).lower()
        if not text:
            return False
        unresolved_markers = (
            "missing_direct_outcome",
            "direct outcome is still not explicit",
            "direct outcome after",
            "direct outcome right after",
            "direct outcome is not visible",
            "direct outcome remains unresolved",
            "still unclear",
            "still unresolved",
            "not yet visible",
            "未显示",
            "不明确",
            "看不清",
        )
        direct_outcome_scope = ("direct outcome", "state_change_hint", "timeline_event", "post_action")
        return any(marker in text for marker in unresolved_markers) and any(scope in text for scope in direct_outcome_scope)

    def _action_intent_has_current_task_artifact_grounding(self, state: AgentState) -> bool:
        task_prefix = f"{str(getattr(state, 'task_family', '') or '').lower()}_"
        frame_paths = list(getattr(state, "retrieved_frames", []) or []) + list(getattr(state, "artifacts", []) or [])
        for path in frame_paths:
            name = Path(str(path)).name.lower()
            if task_prefix and task_prefix in name and any(tag in name for tag in ("segment", "followup", "ext2")):
                return True
        for call in list(getattr(state, "tool_trace", []) or []):
            if not isinstance(call, dict):
                continue
            raw_result = call.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            artifact_paths = raw_result.get("artifact_paths")
            if not isinstance(artifact_paths, list):
                continue
            for path in artifact_paths:
                name = Path(str(path)).name.lower()
                if task_prefix and task_prefix in name and any(tag in name for tag in ("segment", "followup", "ext2")):
                    return True
        return False

    def _action_intent_has_unresolved_secondary_conflicts(self, state: AgentState) -> bool:
        pending_markers = {
            "conflict:conflicting_locations",
            "conflict:conflicting_state_observations",
        }
        if any(item in pending_markers for item in list(getattr(state, "open_questions", []) or [])):
            return True
        detected = set(self._detect_conflicts(state))
        return bool({"conflicting_locations", "conflicting_state_observations"} & detected)

    def _action_intent_has_pending_evidence_gap(self, state: AgentState) -> bool:
        for item in list(state.working_memory) + list(state.evidence_bundle):
            if not isinstance(item, str):
                continue
            if item.startswith("action_intent_need_future_evidence=1") or item.startswith("action_intent_followup_gap=1"):
                return True
            if item.startswith("action_intent_pending_resolution=") or item.startswith("action_intent_pending_resolution_profile="):
                return True
        if self._action_intent_has_unresolved_timeline_review_gap(state):
            return True
        if "need_disambiguating_evidence" in list(getattr(state, "open_questions", []) or []):
            return True
        return False

    def _action_intent_has_plausible_competing_candidate_gap(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        if not self._action_intent_competing_pair_still_needs_disambiguation(state=state):
            return False
        latest = self._latest_action_intent_resolution_payload(state)
        payload = latest[1] if latest is not None else {}
        if isinstance(payload, dict) and (bool(payload.get("tool_failed")) or bool(payload.get("need_more_evidence"))):
            return False
        try:
            confidence = float(payload.get("confidence") or 0.0) if isinstance(payload, dict) else 0.0
        except Exception:  # noqa: BLE001
            confidence = 0.0
        support_text = (
            " ".join(str(payload.get(key) or "") for key in ("reason", "decisive_observation")).lower()
            if isinstance(payload, dict)
            else ""
        )
        primary_gap_type = self._action_intent_primary_gap_type_from_state(state)
        has_structured_observation_gap = primary_gap_type in {
            "future_outcome",
            "immediate_outcome",
            "relation_confirmation",
            "target_discovery",
            "state_transition_unconfirmed",
            "workspace_change_unconfirmed",
        }
        payload_uncertainty = (
            self._action_intent_payload_points_to_later_outcome_uncertainty(payload)
            if isinstance(payload, dict)
            else False
        )
        if self._action_intent_state_describes_unclosed_post_action_outcome(state):
            return True
        if not support_text.strip() and not has_structured_observation_gap and not payload_uncertainty:
            return False
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
            "still unclear",
            "still contested",
            "未显示",
            "没有看到",
            "看不清",
            "不明确",
        )
        return confidence < 0.84 or not self._action_intent_text_has_direct_outcome_clause(
            text=support_text,
            strong_result_terms=direct_result_markers,
            blocked_terms=blocked_terms,
        )

    def _action_intent_text_has_direct_outcome_clause(
        self,
        *,
        text: str,
        strong_result_terms: tuple[str, ...],
        blocked_terms: tuple[str, ...],
    ) -> bool:
        if not str(text or "").strip():
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

    def _latest_action_intent_resolution_payload(self, state: AgentState) -> tuple[str, dict[str, Any]] | None:
        for call in reversed(list(getattr(state, "tool_trace", []) or [])):
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool") or "")
            if tool not in {
                "infer_action_intent",
                "resolve_action_intent_pairwise",
                "resolve_action_intent_future_use",
            }:
                continue
            raw_result = call.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            if raw_result.get("best_index") is None:
                continue
            return tool, raw_result
        return None

    def _action_intent_competing_pair_still_needs_disambiguation(
        self,
        *,
        state: AgentState,
    ) -> bool:
        latest = self._latest_action_intent_resolution_payload(state)
        if latest is not None:
            _tool_name, payload = latest
            if self._action_intent_payload_lacks_direct_post_action_evidence(payload):
                return True
            if self._action_intent_payload_points_to_later_outcome_uncertainty(payload):
                return True
        primary_gap_type = self._action_intent_primary_gap_type_from_state(state)
        if primary_gap_type in {
            "future_outcome",
            "immediate_outcome",
            "relation_confirmation",
            "target_discovery",
            "state_transition_unconfirmed",
            "workspace_change_unconfirmed",
        }:
            return True
        if self._action_intent_state_describes_unclosed_post_action_outcome(state):
            return True
        payload = latest[1] if latest is not None else {}
        support_text = self._action_intent_result_support_text(payload).lower() if isinstance(payload, dict) else ""
        uncertainty_markers = (
            "still unclear",
            "unclear",
            "not visible",
            "not shown",
            "cannot tell",
            "can't tell",
            "insufficient",
            "ambiguous",
            "看不清",
            "不明确",
            "证据不足",
        )
        return any(marker in support_text for marker in uncertainty_markers)

    def _action_intent_has_unresolved_timeline_review_gap(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
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
                last_review_index = None
                saw_new_sampling = False
                continue
            if tool in {"resolve_action_intent_pairwise", "resolve_action_intent_future_use"}:
                if isinstance(raw_result, dict) and not raw_result.get("tool_failed") and raw_result.get("best_index") is not None and not raw_result.get("need_more_evidence"):
                    return False
            if tool == "infer_action_intent":
                if isinstance(raw_result, dict) and not raw_result.get("tool_failed") and raw_result.get("best_index") is not None and not raw_result.get("need_future_evidence") and saw_new_sampling:
                    return False
        return True

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

    def _action_intent_missing_grounding_types(self, state: AgentState) -> list[str]:
        if not self._is_action_intent_task(state):
            return []
        if self._action_intent_textual_rank_fallback_can_finish(state):
            return []
        missing: list[str] = []
        if self._has_action_intent_textual_rank_fallback_answer(state):
            missing.append("need_alternative_evidence_path")
        if self._action_intent_has_pending_evidence_gap(state):
            missing.append("need_disambiguating_evidence")
        if self._action_intent_has_plausible_competing_candidate_gap(state):
            missing.append("need_disambiguating_evidence")
        primary_gap_type = self._action_intent_primary_gap_type_from_state(state)
        latest = self._latest_action_intent_resolution_payload(state)
        payload = latest[1] if latest is not None else {}
        blocker_hint = self._action_intent_verifier_blocker(state, missing=missing)
        needs_specialized_resolution = (
            primary_gap_type in {
                "future_outcome",
                "immediate_outcome",
                "relation_confirmation",
                "target_discovery",
                "state_transition_unconfirmed",
                "workspace_change_unconfirmed",
            }
            or blocker_hint in {"future_gap_family", "post_action_evidence"}
            or (
                isinstance(payload, dict)
                and (
                    self._action_intent_payload_lacks_direct_post_action_evidence(payload)
                    or self._action_intent_payload_points_to_later_outcome_uncertainty(payload)
                )
            )
        )
        if not self._action_intent_has_successful_specialized_resolution(state):
            if needs_specialized_resolution:
                missing.append("need_disambiguating_evidence")
            if (
                (
                    primary_gap_type == "precondition"
                    or self._action_intent_precondition_dependency_is_observation_grounded(state)
                )
                and not self._action_intent_has_precondition_grounding(state)
            ):
                missing.append("need_precondition_context")
            if (
                needs_specialized_resolution
                and not self._action_intent_has_post_action_grounding(state)
            ):
                missing.append("need_post_action_evidence")
        return missing

    def _action_intent_has_sufficient_grounding_for_stable_answer(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return True
        if self._action_intent_has_plausible_competing_candidate_gap(state):
            return False
        has_specialized_resolution = self._action_intent_has_successful_specialized_resolution(state)
        if has_specialized_resolution:
            return True
        primary_gap_type = self._action_intent_primary_gap_type_from_state(state)
        latest = self._latest_action_intent_resolution_payload(state)
        payload = latest[1] if latest is not None else {}
        if primary_gap_type == "precondition":
            if not self._action_intent_has_precondition_grounding(state):
                return False
        if primary_gap_type in {
            "future_outcome",
            "immediate_outcome",
            "relation_confirmation",
            "target_discovery",
            "state_transition_unconfirmed",
            "workspace_change_unconfirmed",
        }:
            return False
        if isinstance(payload, dict) and (
            self._action_intent_payload_lacks_direct_post_action_evidence(payload)
            or self._action_intent_payload_points_to_later_outcome_uncertainty(payload)
        ):
            return False
        blocker_hint = self._action_intent_verifier_blocker(state, missing=[])
        if blocker_hint in {"future_gap_family", "post_action_evidence"}:
            return False
        if blocker_hint == "precondition_context":
            if not self._action_intent_has_precondition_grounding(state):
                return False
        if blocker_hint in {"future_gap_family", "post_action_evidence"}:
            if not self._action_intent_has_post_action_grounding(state):
                return False
        return True

    def _action_intent_has_successful_specialized_resolution(self, state: AgentState) -> bool:
        for call in list(getattr(state, "tool_trace", []) or []):
            if not isinstance(call, dict):
                continue
            raw_result = call.get("raw_result")
            if not isinstance(raw_result, dict):
                continue
            if raw_result.get("tool_failed"):
                continue
            if raw_result.get("best_index") is None:
                continue
            if bool(raw_result.get("need_more_evidence")) or bool(raw_result.get("need_future_evidence")):
                continue
            if self._action_intent_payload_lacks_direct_post_action_evidence(raw_result):
                continue
            if self._action_intent_payload_points_to_later_outcome_uncertainty(raw_result):
                continue
            return True
        return False

    def _action_intent_can_suppress_secondary_conflicts(self, state: AgentState) -> bool:
        return (
            self._is_action_intent_task(state)
            and not self._action_intent_has_pending_evidence_gap(state)
            and not self._action_intent_has_plausible_competing_candidate_gap(state)
            and self._action_intent_has_successful_specialized_resolution(state)
        )

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

    def _action_intent_precondition_dependency_is_observation_grounded(self, state: AgentState) -> bool:
        if not self._is_action_intent_task(state):
            return False
        subject_object = str(self._action_intent_question_subject_object(state) or "").strip().lower()
        object_anchor_tokens = {
            "towel",
            "cloth",
            "napkin",
            "spoon",
            "ladle",
            "board",
            "tray",
            "scale",
            "switch",
            "button",
            "tap",
        }
        evidence_text = " ".join(str(item) for item in list(state.evidence_bundle) + list(state.working_memory)).lower()
        precondition_markers = (
            "dry hands",
            "wet hands",
            "wipe",
            "wiping",
            "wiped",
            "washed",
            "wash",
            "rinsed",
            "messy",
            "spill",
            "dirty",
            "already on",
            "already lit",
            "before the tap",
            "before tapping",
            "display was already lit",
            "scale was already on",
            "container was already on the scale",
            "container already on the scale",
            "bowl already on the scale",
            "干手",
            "湿手",
            "擦手",
            "清洗",
            "溢出",
            "已经亮",
            "已经开机",
        )
        if any(marker in evidence_text for marker in precondition_markers):
            return True
        return any(token in subject_object for token in object_anchor_tokens)

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
        if self._action_intent_can_suppress_secondary_conflicts(state):
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

    def _coerce_choice_index(self, value: Any, choices: list[Any]) -> int | None:
        try:
            index = int(value)
        except Exception:  # noqa: BLE001
            return None
        if 0 <= index < len(choices):
            return index
        return None

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
            evidence_gaps=[item for item in payload.get("evidence_gaps", []) if isinstance(item, dict)],
            sufficiency_decision=payload.get("sufficiency_decision") if isinstance(payload.get("sufficiency_decision"), dict) else {},
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
                evidence_gaps=list(base.evidence_gaps),
                sufficiency_decision=dict(base.sufficiency_decision),
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
            evidence_gaps=list(base.evidence_gaps),
            sufficiency_decision=self._build_sufficiency_decision(
                state=state,
                sufficient=sufficient,
                missing=missing,
                conflicts=conflicts,
                recommend_next_action=recommend,
                summary=f"{base.summary}; answer_critic_missing={missing}; answer_critic_conflicts={conflicts}",
                evidence_gaps=list(base.evidence_gaps),
            ),
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
            evidence_gaps=[item for item in payload.get("evidence_gaps", []) if isinstance(item, dict)],
            sufficiency_decision=payload.get("sufficiency_decision") if isinstance(payload.get("sufficiency_decision"), dict) else {},
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
            evidence_gaps=list(heuristic.evidence_gaps or refined.evidence_gaps),
            sufficiency_decision=self._build_sufficiency_decision(
                state=None,
                sufficient=sufficient,
                missing=missing,
                conflicts=conflicts,
                recommend_next_action=recommend,
                summary=summary,
                evidence_gaps=list(heuristic.evidence_gaps or refined.evidence_gaps),
            ),
        )

    def _detect_conflicts(self, state: AgentState) -> list[str]:
        conflicts: list[str] = []
        if not self._is_action_intent_task(state):
            candidate_indices = {
                item.split("=", 1)[1]
                for item in state.hypotheses
                if isinstance(item, str) and item.startswith("candidate_answer_index=")
            }
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
