"""Metrics designed to highlight evidence-grounded food-agent advantages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vqa import VQAPrediction, compute_metrics


@dataclass(frozen=True)
class AdvantageWeights:
    accuracy: float = 0.40
    evidence: float = 0.25
    state_coverage: float = 0.15
    tool_use: float = 0.10
    reliability: float = 0.10


def evidence_rate(predictions: list[VQAPrediction]) -> float | None:
    if not predictions:
        return None
    return sum(1 for pred in predictions if pred.evidence_ids) / len(predictions)


def tool_use_rate(predictions: list[VQAPrediction]) -> float | None:
    if not predictions:
        return None
    return sum(1 for pred in predictions if pred.tool_calls) / len(predictions)


def failure_rate(predictions: list[VQAPrediction]) -> float | None:
    if not predictions:
        return None
    return sum(1 for pred in predictions if pred.failure_type) / len(predictions)


def state_coverage_score(food_state_metrics: dict[str, Any] | None, spatial_metrics: dict[str, Any] | None) -> float:
    if not food_state_metrics and not spatial_metrics:
        return 0.0
    scores: list[float] = []
    if food_state_metrics:
        video_count = max(food_state_metrics.get("video_count", 0), 1)
        recipe = food_state_metrics.get("recipe_video_count", 0) / video_count
        ingredient = food_state_metrics.get("ingredient_video_count", 0) / video_count
        scores.append((recipe + ingredient) / 2)
    if spatial_metrics:
        video_count = max(spatial_metrics.get("video_count", 0), 1)
        object_coverage = 1.0 if spatial_metrics.get("object_track_rows", 0) > 0 else 0.0
        gaze_coverage = 1.0 if spatial_metrics.get("gaze_rows", 0) > 0 else 0.0
        audio_coverage = 1.0 if spatial_metrics.get("audio_rows", 0) > 0 else 0.0
        scores.append((object_coverage + gaze_coverage + audio_coverage) / 3 if video_count else 0.0)
    return sum(scores) / len(scores)


def food_agent_advantage_score(
    predictions: list[VQAPrediction],
    food_state_metrics: dict[str, Any] | None = None,
    spatial_metrics: dict[str, Any] | None = None,
    weights: AdvantageWeights = AdvantageWeights(),
) -> dict[str, Any]:
    base = compute_metrics(predictions)
    accuracy = base["accuracy"] or 0.0
    evidence = evidence_rate(predictions) or 0.0
    tool_use = tool_use_rate(predictions) or 0.0
    reliability = 1.0 - (failure_rate(predictions) or 0.0)
    coverage = state_coverage_score(food_state_metrics, spatial_metrics)
    score = (
        weights.accuracy * accuracy
        + weights.evidence * evidence
        + weights.state_coverage * coverage
        + weights.tool_use * tool_use
        + weights.reliability * reliability
    )
    return {
        "food_agent_advantage_score": score,
        "components": {
            "accuracy": accuracy,
            "evidence_rate": evidence,
            "state_coverage": coverage,
            "tool_use_rate": tool_use,
            "reliability": reliability,
        },
        "weights": weights.__dict__,
        "base_metrics": base,
    }


def judge_advantage(score: dict[str, Any]) -> dict[str, Any]:
    """Judge whether a run demonstrates clear food-agent advantage."""
    s = score.get("food_agent_advantage_score", 0.0)
    components = score.get("components", {})
    accuracy = components.get("accuracy", 0.0)
    evidence = components.get("evidence_rate", 0.0)
    state_coverage = components.get("state_coverage", 0.0)
    tool_use = components.get("tool_use_rate", 0.0)
    reliability = components.get("reliability", 0.0)

    strong = (
        s >= 0.65
        and accuracy >= 0.55
        and evidence >= 0.60
        and state_coverage >= 0.70
        and reliability >= 0.70
    )
    usable = (
        s >= 0.55
        and accuracy >= 0.45
        and evidence >= 0.45
        and state_coverage >= 0.55
        and tool_use >= 0.40
    )
    if strong:
        verdict = "clear_advantage"
    elif usable:
        verdict = "promising"
    else:
        verdict = "not_yet"
    return {
        "verdict": verdict,
        "thresholds": {
            "clear_advantage": {
                "score": 0.65,
                "accuracy": 0.55,
                "evidence_rate": 0.60,
                "state_coverage": 0.70,
                "reliability": 0.70,
            },
            "promising": {
                "score": 0.55,
                "accuracy": 0.45,
                "evidence_rate": 0.45,
                "state_coverage": 0.55,
                "tool_use_rate": 0.40,
            },
        },
    }
