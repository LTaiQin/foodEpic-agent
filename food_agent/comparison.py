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
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
EVIDENCE_LINE_PATTERN = re.compile(r"(?:event_id|evidence_ids?)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
EVIDENCE_TOKEN_PATTERN = re.compile(r"[A-Za-z]+[A-Za-z0-9:_/\-]*")

TASK_SPECIALIZATION = {
    "ingredient_ingredient_retrieval": "ingredient",
    "ingredient_exact_ingredient_recognition": "ingredient",
    "ingredient_ingredient_recognition": "ingredient",
    "ingredient_ingredients_order": "ingredient",
    "ingredient_ingredient_weight": "ingredient",
    "recipe_step_recognition": "recipe",
    "recipe_recipe_recognition": "recipe",
    "recipe_following_activity_recognition": "recipe",
    "recipe_multi_step_localization": "recipe",
    "nutrition_nutrition_change": "nutrition",
    "nutrition_video_nutrition_estimation": "nutrition",
    "nutrition_image_nutrition_estimation": "nutrition",
}


@dataclass(frozen=True)
class SampleContext:
    video_id: str | None
    time_point: float | None
    start_time: float | None
    end_time: float | None
    object_name: str | None = None


@dataclass(frozen=True)
class ChoiceHint:
    choice_idx: int
    choice_text: str
    score: float
    reason: str


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
        for event_id in nutrition.evidence_ids[:2]:
            if event_id and event_id not in evidence_ids:
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
        specialization = infer_specialization(sample.task_family)
        hints = rank_choice_hints(sample, evidence, specialization)
        content = (
            question
            + "\n\n"
            + _specialized_instruction(specialization)
            + "\n先参考 choice_hints 中得分更高的候选，再结合完整证据做最后判断。"
            + "\n你必须依据证据回答。优先引用给定的 suggested_evidence_ids。"
            + "\n如果证据不足，也必须选择最符合证据的选项，且 evidence_ids 至少填写 1 个最相关证据。"
            + "\n只输出 JSON，不要输出 Markdown。格式：{\"choice\": <编号>, \"reason\": \"...\", \"evidence_ids\": [\"...\"]}"
            + "\n\nchoice_hints=\n"
            + _choice_hints_text(hints)
            + "\n\nEvidence block：\n"
            + _ours_evidence_text(evidence)
        )
        return [{"role": "system", "content": base}, {"role": "user", "content": content}]

    raise ValueError(f"unknown baseline: {baseline}")


def parse_model_output(text: str, sample: VQASample, baseline: str) -> tuple[int, list[str], str | None]:
    evidence_ids: list[str] = []
    if baseline == "ours-foodevidence":
        try:
            payload = _extract_json_payload(text)
            evidence_ids = [str(item) for item in payload.get("evidence_ids", [])]
            choice = payload.get("choice", 0)
            if isinstance(choice, str) and not choice.isdigit():
                choice_idx = parse_choice_prediction(choice, sample.choices)
            else:
                choice_idx = int(choice)
            return choice_idx, evidence_ids, None
        except Exception:
            idx = parse_choice_prediction(text, sample.choices)
            recovered_ids = _extract_evidence_ids_from_text(text)
            failure = "format_error" if idx == 0 and not recovered_ids else None
            return idx, recovered_ids, failure
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


def infer_specialization(task_family: str) -> str:
    return TASK_SPECIALIZATION.get(task_family, "general")


def rank_choice_hints(sample: VQASample, evidence: dict[str, Any], specialization: str) -> list[ChoiceHint]:
    if specialization == "ingredient":
        return _rank_ingredient_choices(sample, evidence)
    if specialization == "recipe":
        return _rank_recipe_choices(sample, evidence)
    if specialization == "nutrition":
        return _rank_nutrition_choices(sample, evidence)
    return [ChoiceHint(idx, choice, 0.0, "no_special_hint") for idx, choice in enumerate(sample.choices)]


def _specialized_instruction(specialization: str) -> str:
    if specialization == "ingredient":
        return (
            "这是食材题。优先根据 added_ingredients、pending_ingredients、ingredient.added 和 recipe 步骤判断"
            " 哪个食材被加入、未加入、或数量最匹配。"
        )
    if specialization == "recipe":
        return (
            "这是菜谱步骤题。优先根据 active_steps、completed_recent、next_steps 和高层 activity 语义"
            " 判断当前或对应时间段最匹配的步骤/菜谱。"
        )
    if specialization == "nutrition":
        return (
            "这是营养变化题。优先根据 nutrition.delta 和最近新增食材判断热量、脂肪、碳水、蛋白质变化。"
        )
    return "这是厨房任务题。优先根据结构化证据判断最符合的选项。"


def _rank_ingredient_choices(sample: VQASample, evidence: dict[str, Any]) -> list[ChoiceHint]:
    ingredient = evidence.get("ingredient")
    added_text = " ".join(_row_text(row) for row in getattr(ingredient, "added", []))
    pending_text = " ".join(_row_text(row) for row in getattr(ingredient, "pending", []))
    recipe = evidence.get("recipe")
    recipe_text = " ".join(_row_text(row) for row in getattr(recipe, "active_steps", []) + getattr(recipe, "completed_steps", [])[-3:])
    spatial = evidence.get("spatial")
    audio_text = " ".join(_row_text(row) for row in getattr(spatial, "audio_events", []))
    hints: list[ChoiceHint] = []
    for idx, choice in enumerate(sample.choices):
        score = 0.0
        reasons = []
        norm_choice = choice.lower()
        if norm_choice and norm_choice in added_text.lower():
            score += 3.0
            reasons.append("match_added")
        if norm_choice and norm_choice in recipe_text.lower():
            score += 1.5
            reasons.append("match_recipe")
        if any(word in audio_text.lower() for word in ["pour", "rustle", "clink", "glass"]) and norm_choice in added_text.lower():
            score += 0.5
            reasons.append("audio_support")
        if "not used" in sample.question.lower() and norm_choice not in added_text.lower() and norm_choice not in recipe_text.lower():
            score += 2.0
            reasons.append("absent_support")
        if norm_choice and norm_choice in pending_text.lower():
            score -= 1.0
            reasons.append("still_pending")
        hints.append(ChoiceHint(idx, choice, score, ",".join(reasons) or "weak_match"))
    return sorted(hints, key=lambda item: (-item.score, item.choice_idx))


def _rank_recipe_choices(sample: VQASample, evidence: dict[str, Any]) -> list[ChoiceHint]:
    recipe = evidence.get("recipe")
    active_text = " ".join(_row_text(row) for row in getattr(recipe, "active_steps", []))
    completed_text = " ".join(_row_text(row) for row in getattr(recipe, "completed_steps", [])[-3:])
    next_text = " ".join(_row_text(row) for row in getattr(recipe, "next_steps", [])[:3])
    spatial = evidence.get("spatial")
    object_text = " ".join(_row_text(row) for row in getattr(spatial, "object_tracks", []))
    audio_text = " ".join(_row_text(row) for row in getattr(spatial, "audio_events", []))
    hints: list[ChoiceHint] = []
    for idx, choice in enumerate(sample.choices):
        score = 0.0
        reasons = []
        overlap_active = _token_overlap_score(choice, active_text)
        overlap_completed = _token_overlap_score(choice, completed_text)
        overlap_next = _token_overlap_score(choice, next_text)
        score += overlap_active * 3.0
        score += overlap_completed * 2.0
        score += overlap_next * 1.0
        score += _token_overlap_score(choice, object_text) * 0.5
        score += _token_overlap_score(choice, audio_text) * 0.5
        if overlap_active:
            reasons.append("match_active")
        if overlap_completed:
            reasons.append("match_completed")
        if overlap_next:
            reasons.append("match_next")
        if _token_overlap_score(choice, object_text):
            reasons.append("match_object")
        if _token_overlap_score(choice, audio_text):
            reasons.append("match_audio")
        hints.append(ChoiceHint(idx, choice, score, ",".join(reasons) or "weak_match"))
    return sorted(hints, key=lambda item: (-item.score, item.choice_idx))


def _rank_nutrition_choices(sample: VQASample, evidence: dict[str, Any]) -> list[ChoiceHint]:
    nutrition = evidence.get("nutrition")
    totals = getattr(nutrition, "totals", {}) or {}
    hints: list[ChoiceHint] = []
    for idx, choice in enumerate(sample.choices):
        score = 0.0
        reasons = []
        for key in ("calories", "fat", "carbs", "protein"):
            value = totals.get(key)
            if value is not None and f"{value:.1f}" in choice:
                score += 2.0
                reasons.append(f"match_{key}")
        hints.append(ChoiceHint(idx, choice, score, ",".join(reasons) or "weak_match"))
    return sorted(hints, key=lambda item: (-item.score, item.choice_idx))


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


def _choice_hints_text(hints: list[ChoiceHint]) -> str:
    return json.dumps(
        [
            {
                "choice_idx": hint.choice_idx,
                "choice_text": hint.choice_text,
                "score": hint.score,
                "reason": hint.reason,
            }
            for hint in hints[:3]
        ],
        ensure_ascii=False,
    )


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(value) for value in row.values() if value is not None)


def _token_overlap_score(choice: str, source: str) -> float:
    choice_tokens = {token for token in re.findall(r"[a-zA-Z]+", choice.lower()) if len(token) > 2}
    if not choice_tokens:
        return 0.0
    source_tokens = set(re.findall(r"[a-zA-Z]+", source.lower()))
    overlap = choice_tokens & source_tokens
    return len(overlap) / len(choice_tokens)


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no json payload found")


def _extract_evidence_ids_from_text(text: str) -> list[str]:
    seen: list[str] = []
    for match in EVIDENCE_LINE_PATTERN.finditer(text):
        raw = match.group(1)
        for evidence_id in EVIDENCE_TOKEN_PATTERN.findall(raw):
            if ":" not in evidence_id:
                continue
            if evidence_id not in seen:
                seen.append(evidence_id)
    return seen
