"""State queries for recipe, ingredient, nutrition, and simple anomalies."""

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
        self._cache: dict[str, pd.DataFrame] = {}

    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.index_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing index table: {path}")
            self._cache[name] = pd.read_parquet(path)
        return self._cache[name]

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

