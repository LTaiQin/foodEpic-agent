"""Utilities for running comparable food-agent baselines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .spatial_store import SpatialContextStore
from .state_store import FoodStateStore
from .vqa import VQASample, parse_choice_prediction


TIME_PATTERN = re.compile(r"(\d+):(\d+):(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class SampleContext:
    video_id: str | None
    time_point: float | None
    start_time: float | None
    end_time: float | None
    object_name: str | None = None


def parse_hms(text: str) -> float | None:
    match = TIME_PATTERN.search(text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def extract_sample_context(sample: VQASample) -> SampleContext:
    video_id = sample.primary_video_id
    time_point = None
    start_time = None
    end_time = None
    for value in sample.inputs.values():
        if not isinstance(value, dict):
            continue
        if not video_id and value.get("id"):
            video_id = value["id"]
        if value.get("time") and time_point is None:
            time_point = parse_hms(str(value["time"]))
        if value.get("start_time"):
            start_time = parse_hms(str(value["start_time"]))
        if value.get("end_time"):
            end_time = parse_hms(str(value["end_time"]))
    if time_point is None and start_time is not None and end_time is not None:
        time_point = (start_time + end_time) / 2
    if start_time is None and time_point is not None:
        start_time = max(0.0, time_point - 5.0)
    if end_time is None and time_point is not None:
        end_time = time_point + 5.0
    return SampleContext(
        video_id=video_id,
        time_point=time_point,
        start_time=start_time,
        end_time=end_time,
    )


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))


def collect_evidence(
    sample: VQASample,
    state_store: FoodStateStore,
    spatial_store: SpatialContextStore,
) -> dict[str, Any]:
    ctx = extract_sample_context(sample)
    if not ctx.video_id:
        return {"context": ctx, "events": [], "recipe": None, "ingredient": None, "nutrition": None, "spatial": None}
    recipe = ingredient = nutrition = spatial = None
    evidence_ids: list[str] = []
    if ctx.time_point is not None:
        recipe = state_store.recipe_state(ctx.video_id, ctx.time_point)
        ingredient = state_store.ingredient_state(ctx.video_id, ctx.time_point)
        nutrition = state_store.nutrition_delta(ctx.video_id, ctx.time_point)
        spatial = spatial_store.combined_context(ctx.video_id, time=ctx.time_point, object_name=ctx.object_name)
        for row in recipe.active_steps + recipe.completed_steps[-2:] + ingredient.added[-2:] + spatial.audio_events[:2]:
            event_id = row.get("event_id")
            if event_id:
                evidence_ids.append(event_id)
    return {
        "context": ctx,
        "recipe": recipe,
        "ingredient": ingredient,
        "nutrition": nutrition,
        "spatial": spatial,
        "evidence_ids": evidence_ids,
    }


def build_messages(
    sample: VQASample,
    baseline: str,
    evidence: dict[str, Any],
) -> list[dict[str, str]]:
    ctx = evidence["context"]
    base = (
        "你是一个厨房视频问答助手。"
        "请只从给定信息中回答，输出最终选项编号。"
    )
    question = (
        f"问题：{sample.question}\n"
        f"候选项：\n{format_choices(sample.choices)}\n"
        "请只输出一个选项编号。"
    )
    if baseline == "textonly":
        return [{"role": "system", "content": base}, {"role": "user", "content": question}]

    if baseline == "directevidence":
        content = question + "\n\n结构化证据：\n" + _direct_evidence_text(evidence)
        return [{"role": "system", "content": base}, {"role": "user", "content": content}]

    if baseline == "foodstate":
        content = question + "\n\nFood state：\n" + _food_state_text(evidence)
        return [{"role": "system", "content": base}, {"role": "user", "content": content}]

    if baseline == "ours-foodevidence":
        content = (
            question
            + "\n\n你必须依据证据回答。如果证据不足，选择最符合证据的选项，并给出简短理由。"
            + "\n输出 JSON：{\"choice\": <编号>, \"reason\": \"...\", \"evidence_ids\": [\"...\"]}"
            + "\n\nEvidence block：\n"
            + _ours_evidence_text(evidence)
        )
        return [{"role": "system", "content": base}, {"role": "user", "content": content}]

    raise ValueError(f"unknown baseline: {baseline}")


def parse_model_output(text: str, sample: VQASample, baseline: str) -> tuple[int, list[str], str | None]:
    evidence_ids: list[str] = []
    if baseline == "ours-foodevidence":
        try:
            payload = json.loads(text)
            evidence_ids = [str(item) for item in payload.get("evidence_ids", [])]
            return int(payload.get("choice", 0)), evidence_ids, None
        except Exception:
            idx = parse_choice_prediction(text, sample.choices)
            return idx, evidence_ids, "format_error"
    idx = parse_choice_prediction(text, sample.choices)
    return idx, evidence_ids, None


def _direct_evidence_text(evidence: dict[str, Any]) -> str:
    ctx = evidence["context"]
    parts = [f"video_id={ctx.video_id}", f"time={ctx.time_point}"]
    recipe = evidence.get("recipe")
    ingredient = evidence.get("ingredient")
    spatial = evidence.get("spatial")
    if recipe:
        parts.append("recent_recipe_steps=" + json.dumps((recipe.active_steps + recipe.completed_steps[-2:]), ensure_ascii=False))
    if ingredient:
        parts.append("recent_ingredients=" + json.dumps(ingredient.added[-3:], ensure_ascii=False))
    if spatial:
        parts.append("recent_audio=" + json.dumps(spatial.audio_events[:3], ensure_ascii=False))
    return "\n".join(parts)


def _food_state_text(evidence: dict[str, Any]) -> str:
    recipe = evidence.get("recipe")
    ingredient = evidence.get("ingredient")
    nutrition = evidence.get("nutrition")
    parts = []
    if recipe:
        parts.append("active_steps=" + json.dumps(recipe.active_steps, ensure_ascii=False))
        parts.append("next_steps=" + json.dumps(recipe.next_steps[:3], ensure_ascii=False))
    if ingredient:
        parts.append("added_ingredients=" + json.dumps(ingredient.added[-5:], ensure_ascii=False))
        parts.append("pending_ingredients=" + json.dumps(ingredient.pending[:5], ensure_ascii=False))
    if nutrition:
        parts.append("nutrition=" + json.dumps(nutrition.totals, ensure_ascii=False))
        parts.append(f"unknown_nutrition_fields={nutrition.unknown_count}")
    return "\n".join(parts)


def _ours_evidence_text(evidence: dict[str, Any]) -> str:
    recipe = evidence.get("recipe")
    ingredient = evidence.get("ingredient")
    nutrition = evidence.get("nutrition")
    spatial = evidence.get("spatial")
    blocks = []
    if recipe:
        blocks.append("recipe.active=" + json.dumps(recipe.active_steps, ensure_ascii=False))
        blocks.append("recipe.completed_recent=" + json.dumps(recipe.completed_steps[-3:], ensure_ascii=False))
    if ingredient:
        blocks.append("ingredient.added=" + json.dumps(ingredient.added[-5:], ensure_ascii=False))
        blocks.append("ingredient.pending=" + json.dumps(ingredient.pending[:5], ensure_ascii=False))
    if nutrition:
        blocks.append("nutrition.delta=" + json.dumps(nutrition.totals, ensure_ascii=False))
    if spatial:
        blocks.append("audio.events=" + json.dumps(spatial.audio_events[:5], ensure_ascii=False))
        blocks.append("gaze.events=" + json.dumps(spatial.gaze_priming[:5], ensure_ascii=False))
        blocks.append("object.tracks=" + json.dumps(spatial.object_tracks[:5], ensure_ascii=False))
    blocks.append("suggested_evidence_ids=" + json.dumps(evidence.get("evidence_ids", []), ensure_ascii=False))
    return "\n".join(blocks)

