"""Rule-based task routing for first-pass food agent baselines."""

from __future__ import annotations

from dataclasses import dataclass


TASK_TOOLS = {
    "recipe": ["get_video_metadata", "retrieve_events", "get_recipe_state"],
    "ingredient": ["get_video_metadata", "retrieve_events", "get_ingredient_state"],
    "nutrition": ["get_video_metadata", "retrieve_events", "get_ingredient_state"],
    "object": ["get_video_metadata", "retrieve_events", "get_object_state"],
    "gaze": ["get_video_metadata", "retrieve_events", "get_gaze_hand_context"],
    "audio": ["get_video_metadata", "retrieve_events", "get_audio_events"],
    "general": ["get_video_metadata", "retrieve_events"],
}


@dataclass(frozen=True)
class TaskRoute:
    task_family: str
    tool_names: list[str]


class FoodTaskRouter:
    """Small deterministic router used before training any learned policy."""

    def route(self, question: str, task_family: str | None = None) -> TaskRoute:
        family = (task_family or self.infer_task_family(question)).lower()
        if family not in TASK_TOOLS:
            family = "general"
        return TaskRoute(task_family=family, tool_names=TASK_TOOLS[family])

    def infer_task_family(self, question: str) -> str:
        q = question.lower()
        if any(word in q for word in ["recipe", "step", "activity", "next"]):
            return "recipe"
        if any(word in q for word in ["ingredient", "add", "weigh", "milk", "flour", "sugar"]):
            return "ingredient"
        if any(word in q for word in ["nutrition", "calorie", "protein", "carbs", "fat"]):
            return "nutrition"
        if any(word in q for word in ["object", "where", "located", "moved", "fixture"]):
            return "object"
        if any(word in q for word in ["gaze", "looking", "anticipation", "interaction"]):
            return "gaze"
        if any(word in q for word in ["sound", "audio", "noise"]):
            return "audio"
        return "general"

