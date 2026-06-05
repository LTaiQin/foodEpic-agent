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
QUESTION_TIME_PATTERN = re.compile(r"<TIME\s+(\d+:\d+:\d+(?:\.\d+)?)")
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
EVIDENCE_LINE_PATTERN = re.compile(r"(?:event_id|evidence_ids?)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
EVIDENCE_TOKEN_PATTERN = re.compile(r"[A-Za-z]+[A-Za-z0-9:_/\-]*")
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
BBOX_PATTERN = re.compile(r"<BBOX\s+([0-9.\s]+)>", re.IGNORECASE)

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

INGREDIENT_ALIASES = {
    "cinnamon sticks": ["cinnamon"],
    "spring onions": ["spring onion", "scallion", "scallions"],
    "red onions": ["red onion"],
    "peas": ["green peas"],
    "olive oil": ["extra virgin olive oil"],
}


@dataclass(frozen=True)
class SampleContext:
    video_id: str | None
    time_point: float | None
    start_time: float | None
    end_time: float | None
    object_name: str | None = None
    video_ids: list[str] | None = None


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
    video_ids: list[str] = []
    time_point = None
    start_time = None
    end_time = None
    for value in sample.inputs.values():
        if not isinstance(value, dict):
            continue
        if value.get("id"):
            video_ids.append(value["id"])
            if not video_id:
                video_id = value["id"]
        if value.get("time") and time_point is None:
            time_point = parse_hms(str(value["time"]))
        if value.get("start_time"):
            start_time = parse_hms(str(value["start_time"]))
        if value.get("end_time"):
            end_time = parse_hms(str(value["end_time"]))
    if time_point is None and start_time is not None and end_time is not None:
        time_point = (start_time + end_time) / 2
    question_times = [parse_hms(match.group(1)) for match in QUESTION_TIME_PATTERN.finditer(sample.question)]
    question_times = [value for value in question_times if value is not None]
    if start_time is None and len(question_times) >= 1:
        start_time = question_times[0]
    if end_time is None and len(question_times) >= 2:
        end_time = question_times[1]
    if time_point is None and len(question_times) == 1:
        time_point = question_times[0]
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
        video_ids=video_ids or ([video_id] if video_id else []),
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
    ingredient_interval = []
    recipe_catalog = []
    activity_window = None
    step_focus = None
    object_reference = []
    video_activities: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    if ctx.video_ids:
        recipe_catalog = state_store.recipe_catalog(ctx.video_ids)
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
    if ctx.video_id and ctx.start_time is not None and ctx.end_time is not None:
        ingredient_interval = state_store.ingredient_interval(ctx.video_id, ctx.start_time, ctx.end_time)
        activity_window = state_store.activity_window(ctx.video_id, ctx.start_time, ctx.end_time)
        for row in ingredient_interval[:3]:
            event_id = row.get("event_id")
            if event_id and event_id not in evidence_ids:
                evidence_ids.append(event_id)
        if activity_window:
            for row in activity_window.activities[:3]:
                event_id = row.get("event_id")
                if event_id and event_id not in evidence_ids:
                    evidence_ids.append(event_id)
    if (
        sample.task_family == "recipe_following_activity_recognition"
        and ctx.video_id
        and activity_window is None
    ):
        target_step = _extract_question_recipe_step_name(sample.question)
        if target_step:
            step_lookup = state_store.recipe_step_matches(ctx.video_id, target_step)
            if step_lookup.matches:
                step_focus = step_lookup.matches[-1]
                focus_start = float(step_focus.get("start_time") or 0.0)
                focus_end = float(step_focus.get("end_time") or focus_start)
                focus_time = (focus_start + focus_end) / 2
                recipe = state_store.recipe_state(ctx.video_id, focus_time)
                ingredient = state_store.ingredient_state(ctx.video_id, focus_time)
                nutrition = state_store.nutrition_delta(ctx.video_id, focus_time)
                spatial = spatial_store.combined_context(ctx.video_id, time=focus_time, object_name=ctx.object_name)
                activity_window = state_store.activity_window(ctx.video_id, focus_start, focus_end)
                step_event_id = step_focus.get("event_id")
                if step_event_id and step_event_id not in evidence_ids:
                    evidence_ids.append(step_event_id)
                if activity_window:
                    for row in activity_window.activities[:3]:
                        event_id = row.get("event_id")
                        if event_id and event_id not in evidence_ids:
                            evidence_ids.append(event_id)
    if ctx.video_id:
        video_activities = state_store.all_video_activities(ctx.video_id)
        bbox = _extract_question_bbox(sample.question)
        ref_time = ctx.time_point if ctx.time_point is not None else ctx.start_time
        if bbox:
            object_reference = spatial_store.resolve_object_reference(ctx.video_id, bbox, time=ref_time, limit=5)
    return {
        "context": ctx,
        "recipe": recipe,
        "ingredient": ingredient,
        "ingredient_interval": ingredient_interval,
        "recipe_catalog": recipe_catalog,
        "activity_window": activity_window,
        "step_focus": step_focus,
        "object_reference": object_reference,
        "video_activities": video_activities,
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
            + _ours_evidence_text(sample, evidence)
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
        if sample.task_family == "ingredient_exact_ingredient_recognition":
            return _rank_exact_ingredient_quantity_choices(sample, evidence)
        if sample.task_family == "ingredient_ingredient_recognition":
            return _rank_recipe_ingredient_membership_choices(sample, evidence)
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
    interval_rows = evidence.get("ingredient_interval", [])
    added_rows = interval_rows or getattr(ingredient, "added", [])
    added_text = " ".join(_row_text(row) for row in added_rows)
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
        match_score, match_reason = _ingredient_match_score(choice, added_rows)
        if match_score:
            score += match_score
            reasons.append(match_reason)
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


def _rank_recipe_ingredient_membership_choices(sample: VQASample, evidence: dict[str, Any]) -> list[ChoiceHint]:
    recipe_catalog = evidence.get("recipe_catalog", [])
    target_recipe = _select_target_recipe(sample.question, recipe_catalog)
    target_ingredients = target_recipe.get("ingredients", []) if target_recipe else []
    hints: list[ChoiceHint] = []
    asks_not_used = "not used" in sample.question.lower()
    for idx, choice in enumerate(sample.choices):
        score = 0.0
        reasons = []
        match_score, match_reason = _ingredient_name_list_match(choice, target_ingredients)
        if asks_not_used:
            if match_score > 0:
                score -= 3.0
                reasons.append("present_in_recipe_catalog")
            else:
                score += 5.0
                reasons.append("absent_from_recipe_catalog")
        else:
            score += match_score
            if match_score > 0:
                reasons.append(match_reason)
        hints.append(ChoiceHint(idx, choice, score, ",".join(reasons) or "weak_match"))
    return sorted(hints, key=lambda item: (-item.score, item.choice_idx))


def _rank_exact_ingredient_quantity_choices(sample: VQASample, evidence: dict[str, Any]) -> list[ChoiceHint]:
    recipe_catalog = evidence.get("recipe_catalog", [])
    target_recipe = _select_target_recipe(sample.question, recipe_catalog)
    target_ingredient = _extract_question_ingredient_name(sample.question)
    amount_rows = target_recipe.get("ingredient_amounts", []) if target_recipe else []
    target_amount = None
    target_unit = None
    if target_ingredient:
        for row in amount_rows:
            name = str(row.get("name", ""))
            if _token_overlap_score(target_ingredient, name) >= 0.5 or _token_overlap_score(name, target_ingredient) >= 0.5:
                target_amount = row.get("amount")
                target_unit = row.get("amount_unit")
                break
    hints: list[ChoiceHint] = []
    for idx, choice in enumerate(sample.choices):
        score = 0.0
        reasons = []
        quantity = _parse_choice_quantity(choice)
        if quantity and target_amount is not None:
            choice_value, choice_unit = quantity
            if str(choice_unit).lower() == str(target_unit).lower():
                score += 2.0
                reasons.append("match_unit")
            if _numeric_equal(choice_value, target_amount):
                score += 5.0
                reasons.append("match_exact_amount")
            else:
                diff = abs(float(choice_value) - float(target_amount))
                score += max(0.0, 2.0 - diff)
                reasons.append("near_amount")
        hints.append(ChoiceHint(idx, choice, score, ",".join(reasons) or "weak_match"))
    return sorted(hints, key=lambda item: (-item.score, item.choice_idx))


def _rank_recipe_choices(sample: VQASample, evidence: dict[str, Any]) -> list[ChoiceHint]:
    recipe = evidence.get("recipe")
    active_text = " ".join(_row_text(row) for row in getattr(recipe, "active_steps", []))
    completed_text = " ".join(_row_text(row) for row in getattr(recipe, "completed_steps", [])[-3:])
    next_text = " ".join(_row_text(row) for row in getattr(recipe, "next_steps", [])[:3])
    activity_window = evidence.get("activity_window")
    activity_text = " ".join(_row_text(row) for row in getattr(activity_window, "activities", []))
    video_activity_text = " ".join(_row_text(row) for row in evidence.get("video_activities", [])[:20])
    recipe_names = [str(item.get("name", "")) for item in evidence.get("recipe_catalog", [])]
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
        overlap_activity = _token_overlap_score(choice, activity_text)
        overlap_video_activity = _token_overlap_score(choice, video_activity_text)
        recipe_match = max((_token_overlap_score(choice, name) for name in recipe_names), default=0.0)
        score += overlap_active * 3.0
        score += overlap_completed * 2.0
        score += overlap_next * 1.0
        score += overlap_activity * 3.0
        score += overlap_video_activity * 1.5
        score += recipe_match * 4.0
        score += _token_overlap_score(choice, object_text) * 0.5
        score += _token_overlap_score(choice, audio_text) * 0.5
        if overlap_active:
            reasons.append("match_active")
        if overlap_completed:
            reasons.append("match_completed")
        if overlap_next:
            reasons.append("match_next")
        if overlap_activity:
            reasons.append("match_activity_window")
        if overlap_video_activity:
            reasons.append("match_video_activity")
        if recipe_match:
            reasons.append("match_recipe_name")
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
    if evidence.get("recipe_catalog"):
        parts.append("recipe_catalog=" + json.dumps(evidence["recipe_catalog"], ensure_ascii=False))
    if evidence.get("activity_window"):
        parts.append("activity_window=" + json.dumps(evidence["activity_window"].activities[:5], ensure_ascii=False))
    if evidence.get("step_focus"):
        parts.append("step_focus=" + json.dumps(evidence["step_focus"], ensure_ascii=False))
    if evidence.get("object_reference"):
        parts.append("object_reference=" + json.dumps(evidence["object_reference"][:3], ensure_ascii=False))
    if ingredient:
        parts.append("added_ingredients=" + json.dumps(ingredient.added[-5:], ensure_ascii=False))
        parts.append("pending_ingredients=" + json.dumps(ingredient.pending[:5], ensure_ascii=False))
    if nutrition:
        parts.append("nutrition=" + json.dumps(nutrition.totals, ensure_ascii=False))
        parts.append(f"unknown_nutrition_fields={nutrition.unknown_count}")
    return "\n".join(parts)


def _ours_evidence_text(sample: VQASample, evidence: dict[str, Any]) -> str:
    recipe = evidence.get("recipe")
    ingredient = evidence.get("ingredient")
    nutrition = evidence.get("nutrition")
    spatial = evidence.get("spatial")
    blocks = []
    task_family = sample.task_family
    target_recipe = _select_target_recipe(sample.question, evidence.get("recipe_catalog", []))
    if recipe:
        blocks.append("recipe.active=" + json.dumps(recipe.active_steps, ensure_ascii=False))
        blocks.append("recipe.completed_recent=" + json.dumps(recipe.completed_steps[-3:], ensure_ascii=False))
    if task_family == "recipe_recipe_recognition" and evidence.get("recipe_catalog"):
        compact_catalog = [
            {
                "recipe_id": item.get("recipe_id"),
                "name": item.get("name"),
                "video_ids": item.get("video_ids"),
            }
            for item in evidence["recipe_catalog"]
        ]
        blocks.append("recipe.catalog=" + json.dumps(compact_catalog, ensure_ascii=False))
    if task_family == "ingredient_ingredient_recognition" and target_recipe:
        blocks.append("recipe.target=" + json.dumps(target_recipe, ensure_ascii=False))
    if evidence.get("activity_window"):
        blocks.append("activity.window=" + json.dumps(evidence["activity_window"].activities[:5], ensure_ascii=False))
    if evidence.get("step_focus"):
        blocks.append("recipe.step_focus=" + json.dumps(evidence["step_focus"], ensure_ascii=False))
    if evidence.get("object_reference"):
        blocks.append("object.reference=" + json.dumps(evidence["object_reference"][:5], ensure_ascii=False))
    if task_family in {"recipe_following_activity_recognition", "recipe_step_recognition"} and evidence.get("video_activities"):
        blocks.append("activity.video_recent=" + json.dumps(evidence["video_activities"][:8], ensure_ascii=False))
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


def _ingredient_match_score(choice: str, rows: list[dict[str, Any]]) -> tuple[float, str]:
    if not rows:
        return 0.0, "no_match"
    aliases = {_normalize_ingredient_text(choice)}
    for alias in INGREDIENT_ALIASES.get(choice.lower(), []):
        aliases.add(_normalize_ingredient_text(alias))
    best = 0.0
    for row in rows:
        label = _normalize_ingredient_text(str(row.get("label", "")))
        text = _normalize_ingredient_text(str(row.get("text", "")))
        candidates = {label, text}
        if aliases & candidates:
            return 5.0, "match_interval_added"
        for alias in aliases:
            for candidate in candidates:
                if not candidate:
                    continue
                best = max(best, _jaccard_similarity(alias, candidate))
    if best >= 0.5:
        return 3.0 + best, "partial_alias_match"
    return 0.0, "no_match"


def _ingredient_name_list_match(choice: str, names: list[str]) -> tuple[float, str]:
    if not names:
        return 0.0, "no_recipe_catalog"
    aliases = {_normalize_ingredient_text(choice)}
    for alias in INGREDIENT_ALIASES.get(choice.lower(), []):
        aliases.add(_normalize_ingredient_text(alias))
    best = 0.0
    for name in names:
        candidate = _normalize_ingredient_text(name)
        if candidate in aliases or candidate == _normalize_ingredient_text(choice):
            return 5.0, "match_recipe_ingredient"
        for alias in aliases:
            best = max(best, _jaccard_similarity(alias, candidate))
    if best >= 0.5:
        return 3.0 + best, "partial_recipe_ingredient_match"
    return 0.0, "no_recipe_ingredient_match"


def _extract_question_ingredient_name(question: str) -> str | None:
    match = re.search(r"quantity of (.+?) used in", question.lower())
    if not match:
        return None
    return match.group(1).strip()


def _extract_question_recipe_step_name(question: str) -> str | None:
    match = re.search(r"recipe step (.+?) in this video\\?", question, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().rstrip(".")


def _extract_question_bbox(question: str) -> list[float] | None:
    match = BBOX_PATTERN.search(question)
    if not match:
        return None
    try:
        values = [float(value) for value in match.group(1).split()]
    except ValueError:
        return None
    if len(values) != 4:
        return None
    return values


def _parse_choice_quantity(choice: str) -> tuple[float, str] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", choice.strip())
    if not match:
        return None
    return float(match.group(1)), match.group(2).lower()


def _numeric_equal(a: float, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def _select_target_recipe(question: str, recipe_catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    question_text = question.lower()
    best_recipe = None
    best_score = 0.0
    for recipe in recipe_catalog:
        name = str(recipe.get("name", ""))
        score = _token_overlap_score(name, question_text)
        if name and name.lower() in question_text:
            score += 2.0
        if score > best_score:
            best_score = score
            best_recipe = recipe
    return best_recipe




def _normalize_ingredient_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


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
