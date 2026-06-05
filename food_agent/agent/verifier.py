"""Evidence sufficiency verifier for the graph agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


class GraphAgentVerifier:
    """Check whether current evidence is sufficient before allowing finish."""

    def __init__(self, model_client: OpenAICompatibleModelClient | None = None):
        self.model_client = model_client

    def verify(self, *, state: AgentState) -> VerificationResult:
        heuristic = self._heuristic_verify(state)
        if self.model_client is None:
            return heuristic
        try:
            refined = self._model_verify(state)
        except Exception:  # noqa: BLE001
            return heuristic
        return self._merge_results(heuristic, refined)

    def _heuristic_verify(self, state: AgentState) -> VerificationResult:
        missing = [item for item in state.open_questions if item and item != "need_disambiguating_evidence"]
        conflicts = self._detect_conflicts(state)
        evidence_count = len(state.evidence_bundle)
        sufficient = not missing and not conflicts and evidence_count > 0
        confidence = min(0.95, 0.3 + 0.08 * evidence_count)
        if missing:
            confidence = min(confidence, 0.45)
        if conflicts:
            confidence = min(confidence, 0.25)
        summary = f"sufficient={sufficient}; missing={missing}; conflicts={conflicts}; evidence_count={evidence_count}"
        recommend = "finish" if sufficient else (missing[0] if missing else "resolve_conflict")
        return VerificationResult(
            sufficient=sufficient,
            confidence=confidence,
            missing_evidence_types=missing,
            conflicts=conflicts,
            recommend_next_action=recommend,
            summary=summary,
        )

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
        return conflicts
