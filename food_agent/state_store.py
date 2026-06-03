"""State queries for recipe, ingredient, nutrition, activities, and simple anomalies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ProjectPaths


NUTRIENT_KEYS = ("calories", "carbs", "fat", "protein")


@dataclass(frozen=True)
class RecipeState:
    video_id: str
    time: float
    active_steps: list[dict[str, Any]]
    completed_steps: list[dict[str, Any]]
    next_steps: list[dict[str, Any]]


@dataclass(frozen=True)
class IngredientState:
    video_id: str
    time: float
    added: list[dict[str, Any]]
    weighed: list[dict[str, Any]]
    pending: list[dict[str, Any]]


@dataclass(frozen=True)
class NutritionDelta:
    video_id: str
    time: float
    totals: dict[str, float]
    unknown_count: int
    evidence_ids: list[str]


@dataclass(frozen=True)
class ActivityWindow:
    video_id: str
    start_time: float
    end_time: float
    activities: list[dict[str, Any]]


@dataclass(frozen=True)
class Anomaly:
    anomaly_type: str
    severity: str
    message: str
    evidence_ids: list[str]


class FoodStateStore:
    """Deterministic state store over the generated Parquet event index."""

    def __init__(self, index_dir: Path | None = None):
        paths = ProjectPaths.from_env()
        self.index_dir = index_dir or paths.output_root / "event_index"
        self.annotation_root = paths.annotation_root
        self._cache: dict[str, pd.DataFrame] = {}

    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.index_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing index table: {path}")
            self._cache[name] = pd.read_parquet(path)
        return self._cache[name]

    def _activities(self) -> pd.DataFrame:
        if "activities_csv" not in self._cache:
            activity_dir = self.annotation_root / "high-level" / "activities"
            frames = []
            for csv_path in sorted(activity_dir.glob("*_recipe_timestamps.csv")):
                frame = pd.read_csv(csv_path)
                if frame.empty:
                    continue
                frame = frame.copy()
                frame["start_time"] = pd.to_numeric(frame["start_time"], errors="coerce")
                frame["end_time"] = pd.to_numeric(frame["end_time"], errors="coerce")
                frame["event_id"] = [f"activity:{row.video_id}:{idx}" for idx, row in frame.reset_index(drop=True).iterrows()]
                frame["text"] = frame["high_level_activity_label"]
                frames.append(frame)
            self._cache["activities_csv"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._cache["activities_csv"]

    def recipe_state(self, video_id: str, time: float, horizon: float = 120.0) -> RecipeState:
        steps = self._table("recipe_steps")
        video_steps = steps[steps["video_id"] == video_id].copy()
        active = video_steps[
            (video_steps["start_time"].fillna(float("inf")) <= time)
            & (video_steps["end_time"].fillna(float("-inf")) >= time)
        ]
        completed = video_steps[video_steps["end_time"].fillna(float("inf")) < time]
        upcoming = video_steps[
            (video_steps["start_time"].fillna(float("-inf")) > time)
            & (video_steps["start_time"].fillna(float("inf")) <= time + horizon)
        ]
        return RecipeState(
            video_id=video_id,
            time=time,
            active_steps=_records(active.sort_values(["start_time", "end_time"])),
            completed_steps=_records(completed.sort_values(["start_time", "end_time"])),
            next_steps=_records(upcoming.sort_values(["start_time", "end_time"]).head(5)),
        )

    def ingredient_state(self, video_id: str, time: float) -> IngredientState:
        ingredients = self._table("ingredients")
        video_ingredients = ingredients[ingredients["video_id"] == video_id].copy()
        observed = video_ingredients[video_ingredients["start_time"].fillna(float("inf")) <= time]
        added = observed[observed["event_type"] == "ingredient_add"]
        weighed = observed[observed["event_type"] == "ingredient_weigh"]
        added_ids = {_payload(row).get("ingredient_id") for _, row in added.iterrows()}
        pending = video_ingredients[
            (~video_ingredients.apply(lambda row: _payload(row).get("ingredient_id") in added_ids, axis=1))
            & (video_ingredients["event_type"] == "ingredient_add")
        ]
        return IngredientState(
            video_id=video_id,
            time=time,
            added=_records(added.sort_values(["start_time", "end_time"])),
            weighed=_records(weighed.sort_values(["start_time", "end_time"])),
            pending=_records(pending.sort_values(["start_time", "end_time"])),
        )

    def ingredient_interval(self, video_id: str, start_time: float, end_time: float) -> list[dict[str, Any]]:
        ingredients = self._table("ingredients")
        subset = ingredients[
            (ingredients["video_id"] == video_id)
            & (ingredients["event_type"] == "ingredient_add")
            & (ingredients["end_time"].fillna(float("inf")) >= float(start_time))
            & (ingredients["start_time"].fillna(float("-inf")) <= float(end_time))
        ].copy()
        subset = subset.sort_values(["start_time", "end_time"])
        return _records(subset)

    def recipe_catalog(self, video_ids: list[str]) -> list[dict[str, Any]]:
        if not video_ids:
            return []
        steps = self._table("recipe_steps")
        recipes = self._table("recipes")
        subset = steps[steps["video_id"].isin(video_ids)].copy()
        if subset.empty:
            return []
        subset["recipe_id"] = subset["payload_json"].apply(lambda raw: json.loads(raw).get("recipe_id") if raw else None)
        recipe_ids = [recipe_id for recipe_id in subset["recipe_id"].dropna().unique().tolist() if recipe_id]
        if not recipe_ids:
            return []
        recipe_rows = recipes[recipes["recipe_id"].isin(recipe_ids)].copy()
        recipe_map = {row["recipe_id"]: row for row in recipe_rows.to_dict(orient="records")}
        results = []
        for recipe_id in recipe_ids:
            meta = recipe_map.get(recipe_id, {})
            recipe_videos = subset[subset["recipe_id"] == recipe_id]["video_id"].dropna().unique().tolist()
            results.append(
                {
                    "recipe_id": recipe_id,
                    "name": meta.get("name"),
                    "participant_id": meta.get("participant_id"),
                    "video_ids": recipe_videos,
                    "step_count": int(meta.get("step_count") or 0),
                }
            )
        return results

    def activity_window(self, video_id: str, start_time: float, end_time: float) -> ActivityWindow:
        activities = self._activities()
        subset = activities[
            (activities["video_id"] == video_id)
            & (activities["end_time"].fillna(float("inf")) >= float(start_time))
            & (activities["start_time"].fillna(float("-inf")) <= float(end_time))
        ].copy()
        subset = subset.sort_values(["start_time", "end_time"])
        return ActivityWindow(
            video_id=video_id,
            start_time=float(start_time),
            end_time=float(end_time),
            activities=_records(subset),
        )

    def all_video_activities(self, video_id: str) -> list[dict[str, Any]]:
        activities = self._activities()
        subset = activities[activities["video_id"] == video_id].copy().sort_values(["start_time", "end_time"])
        return _records(subset)

    def nutrition_delta(self, video_id: str, time: float) -> NutritionDelta:
        state = self.ingredient_state(video_id, time)
        totals = {key: 0.0 for key in NUTRIENT_KEYS}
        unknown_count = 0
        evidence_ids: list[str] = []
        for row in state.added:
            payload = json.loads(row["payload_json"])
            evidence_ids.append(row["event_id"])
            for key in NUTRIENT_KEYS:
                value = _to_float(payload.get(key))
                if value is None:
                    unknown_count += 1
                else:
                    totals[key] += value
        return NutritionDelta(video_id=video_id, time=time, totals=totals, unknown_count=unknown_count, evidence_ids=evidence_ids)

    def detect_anomalies(self, video_id: str, time: float) -> list[Anomaly]:
        state = self.ingredient_state(video_id, time)
        anomalies: list[Anomaly] = []
        seen: dict[str, list[dict[str, Any]]] = {}
        for row in state.added:
            payload = json.loads(row["payload_json"])
            ingredient_id = payload.get("ingredient_id") or row.get("label")
            seen.setdefault(ingredient_id, []).append(row)
        for ingredient_id, rows in seen.items():
            if len(rows) > 1:
                anomalies.append(
                    Anomaly(
                        anomaly_type="duplicate_ingredient",
                        severity="medium",
                        message=f"Ingredient {ingredient_id} appears to be added {len(rows)} times before {time:.2f}s.",
                        evidence_ids=[row["event_id"] for row in rows],
                    )
                )
        for row in state.pending:
            if row.get("start_time") is not None and float(row["start_time"]) < time:
                anomalies.append(
                    Anomaly(
                        anomaly_type="missing_ingredient",
                        severity="high",
                        message=f"Expected ingredient {row.get('label')} has not been observed as added by {time:.2f}s.",
                        evidence_ids=[row["event_id"]],
                    )
                )
        return anomalies


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    raw = row.get("payload_json")
    return json.loads(raw) if raw else {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def state_to_json(state: Any) -> str:
    if isinstance(state, list):
        return json.dumps([asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in state], ensure_ascii=False, indent=2)
    return json.dumps(asdict(state), ensure_ascii=False, indent=2)
