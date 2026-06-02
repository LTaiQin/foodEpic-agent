"""Build normalized event-index tables from HD-EPIC annotations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .loaders import load_json
from .paths import infer_participant_id


TIME_RE = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


@dataclass
class EventRow:
    event_id: str
    video_id: str | None
    participant_id: str | None
    event_type: str
    start_time: float | None
    end_time: float | None
    label: str | None
    text: str | None
    payload_json: str
    source_file: str
    evidence_ref: str


def parse_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    match = TIME_RE.search(text)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def rows_to_frame(rows: Iterable[EventRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def build_videos_table(manifest: pd.DataFrame) -> pd.DataFrame:
    videos = manifest[manifest["file_type"] == "mp4"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in videos.iterrows():
        metadata = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        rows.append(
            {
                "video_id": row["video_id"],
                "participant_id": row["participant_id"],
                "path": row["path"],
                "relative_path": row["relative_path"],
                "size_bytes": int(row["size_bytes"]),
                "fps": metadata.get("fps"),
                "frame_count": metadata.get("frame_count"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "duration_sec": metadata.get("duration_sec"),
            }
        )
    return pd.DataFrame(rows).sort_values(["participant_id", "video_id"], na_position="last")


def build_activity_events(annotation_root: Path) -> pd.DataFrame:
    rows: list[EventRow] = []
    for csv_path in sorted((annotation_root / "high-level" / "activities").glob("*_recipe_timestamps.csv")):
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            video_id = row.get("video_id")
            start = parse_seconds(row.get("start_time"))
            end = parse_seconds(row.get("end_time"))
            label = row.get("high_level_activity_label")
            recipe_id = row.get("recipe_id")
            payload = {"recipe_id": recipe_id, "row_index": int(index)}
            rows.append(
                EventRow(
                    event_id=f"activity:{video_id}:{index}",
                    video_id=video_id,
                    participant_id=infer_participant_id(video_id or csv_path.as_posix()),
                    event_type="activity",
                    start_time=start,
                    end_time=end,
                    label=str(label) if pd.notna(label) else None,
                    text=str(label) if pd.notna(label) else None,
                    payload_json=to_json(payload),
                    source_file=csv_path.as_posix(),
                    evidence_ref=f"{csv_path.name}:{index}",
                )
            )
    return rows_to_frame(rows)


def build_recipe_tables(annotation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recipe_path = annotation_root / "high-level" / "complete_recipes.json"
    recipes = load_json(recipe_path)
    recipe_rows: list[dict[str, Any]] = []
    step_rows: list[EventRow] = []
    ingredient_rows: list[EventRow] = []
    for recipe_id, recipe in recipes.items():
        recipe_rows.append(
            {
                "recipe_id": recipe_id,
                "participant_id": recipe.get("participant"),
                "name": recipe.get("name"),
                "type": recipe.get("type"),
                "source": recipe.get("source"),
                "step_count": len(recipe.get("steps", {})),
                "capture_count": len(recipe.get("captures", [])),
            }
        )
        steps = recipe.get("steps", {})
        for capture_index, capture in enumerate(recipe.get("captures", [])):
            for step_id, spans in capture.get("step_times", {}).items():
                for span_index, span in enumerate(spans):
                    video_id = span.get("video")
                    step_text = steps.get(step_id)
                    step_rows.append(
                        EventRow(
                            event_id=f"recipe_step:{recipe_id}:{step_id}:{capture_index}:{span_index}",
                            video_id=video_id,
                            participant_id=recipe.get("participant"),
                            event_type="recipe_step",
                            start_time=parse_seconds(span.get("start")),
                            end_time=parse_seconds(span.get("end")),
                            label=step_id,
                            text=step_text,
                            payload_json=to_json({"recipe_id": recipe_id, "step_id": step_id, "capture_index": capture_index}),
                            source_file=recipe_path.as_posix(),
                            evidence_ref=f"{recipe_id}/{step_id}/{capture_index}/{span_index}",
                        )
                    )
            for ingredient_id, ingredient in capture.get("ingredients", {}).items():
                for action_type in ("weigh", "add"):
                    for span_index, span in enumerate(ingredient.get(action_type, [])):
                        video_id = span.get("video")
                        payload = {
                            "recipe_id": recipe_id,
                            "ingredient_id": ingredient_id,
                            "action_type": action_type,
                            "amount": ingredient.get("amount"),
                            "amount_unit": ingredient.get("amount_unit"),
                            "calories": ingredient.get("calories"),
                            "carbs": ingredient.get("carbs"),
                            "fat": ingredient.get("fat"),
                            "protein": ingredient.get("protein"),
                        }
                        ingredient_rows.append(
                            EventRow(
                                event_id=f"ingredient:{recipe_id}:{ingredient_id}:{action_type}:{capture_index}:{span_index}",
                                video_id=video_id,
                                participant_id=recipe.get("participant"),
                                event_type=f"ingredient_{action_type}",
                                start_time=parse_seconds(span.get("start")),
                                end_time=parse_seconds(span.get("end")),
                                label=ingredient.get("name"),
                                text=f"{action_type} {ingredient.get('name')}",
                                payload_json=to_json(payload),
                                source_file=recipe_path.as_posix(),
                                evidence_ref=f"{recipe_id}/{ingredient_id}/{action_type}/{capture_index}/{span_index}",
                            )
                        )
    return pd.DataFrame(recipe_rows), rows_to_frame(step_rows), rows_to_frame(ingredient_rows)


def build_audio_events(annotation_root: Path) -> pd.DataFrame:
    path = annotation_root / "audio-annotations" / "HD_EPIC_Sounds.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows: list[EventRow] = []
    for index, row in df.iterrows():
        video_id = row.get("video_id")
        label = row.get("class")
        rows.append(
            EventRow(
                event_id=f"audio:{video_id}:{index}",
                video_id=video_id,
                participant_id=row.get("participant_id") or infer_participant_id(video_id or ""),
                event_type="audio",
                start_time=parse_seconds(row.get("start_timestamp")),
                end_time=parse_seconds(row.get("stop_timestamp")),
                label=str(label) if pd.notna(label) else None,
                text=str(label) if pd.notna(label) else None,
                payload_json=to_json(
                    {
                        "start_sample": row.get("start_sample"),
                        "stop_sample": row.get("stop_sample"),
                        "class_id": row.get("class_id"),
                    }
                ),
                source_file=path.as_posix(),
                evidence_ref=f"{path.name}:{index}",
            )
        )
    return rows_to_frame(rows)


def build_vqa_samples(annotation_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((annotation_root / "vqa-benchmark").glob("*.json")):
        data = load_json(path)
        items = data.items() if isinstance(data, dict) else enumerate(data)
        for raw_id, sample in items:
            inputs = sample.get("inputs", {})
            video_ids = []
            for value in inputs.values():
                if isinstance(value, dict) and value.get("id"):
                    video_ids.append(value["id"])
            rows.append(
                {
                    "vqa_id": f"{path.stem}:{raw_id}",
                    "task_family": path.stem,
                    "source_file": path.as_posix(),
                    "raw_id": str(raw_id),
                    "video_ids": video_ids,
                    "primary_video_id": video_ids[0] if video_ids else None,
                    "participant_id": infer_participant_id(video_ids[0]) if video_ids else None,
                    "question": sample.get("question"),
                    "choices_json": to_json(sample.get("choices", [])),
                    "correct_idx": sample.get("correct_idx"),
                    "inputs_json": to_json(inputs),
                    "others_json": to_json(sample.get("others", {})),
                }
            )
    return pd.DataFrame(rows)


def build_object_tables(annotation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assoc_path = annotation_root / "scene-and-object-movements" / "assoc_info.json"
    mask_path = annotation_root / "scene-and-object-movements" / "mask_info.json"
    assoc_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    if assoc_path.exists():
        assoc = load_json(assoc_path)
        for video_id, objects in assoc.items():
            for association_id, obj in objects.items():
                for track_index, track in enumerate(obj.get("tracks", [])):
                    segment = track.get("time_segment") or [None, None]
                    assoc_rows.append(
                        {
                            "video_id": video_id,
                            "participant_id": infer_participant_id(video_id),
                            "association_id": association_id,
                            "object_name": obj.get("name"),
                            "track_id": track.get("track_id"),
                            "track_index": track_index,
                            "start_time": parse_seconds(segment[0]),
                            "end_time": parse_seconds(segment[1]),
                            "masks_json": to_json(track.get("masks", [])),
                        }
                    )
    if mask_path.exists():
        masks = load_json(mask_path)
        for video_id, video_masks in masks.items():
            for mask_id, mask in video_masks.items():
                mask_rows.append(
                    {
                        "video_id": video_id,
                        "participant_id": infer_participant_id(video_id),
                        "mask_id": mask_id,
                        "frame_number": mask.get("frame_number"),
                        "bbox_json": to_json(mask.get("bbox")),
                        "location_3d_json": to_json(mask.get("3d_location")),
                        "fixture": mask.get("fixture"),
                    }
                )
    return pd.DataFrame(assoc_rows), pd.DataFrame(mask_rows)


def build_gaze_priming(annotation_root: Path) -> pd.DataFrame:
    path = annotation_root / "eye-gaze-priming" / "priming_info.json"
    if not path.exists():
        return pd.DataFrame()
    data = load_json(path)
    rows: list[dict[str, Any]] = []
    for video_id, objects in data.items():
        for object_id, states in objects.items():
            for state_name, state in states.items():
                if not isinstance(state, dict) or not state:
                    continue
                stats = state.get("prime_stats", {})
                rows.append(
                    {
                        "video_id": video_id,
                        "participant_id": infer_participant_id(video_id),
                        "object_id": object_id,
                        "state": state_name,
                        "frame": state.get("frame"),
                        "location_3d_json": to_json(state.get("3d_location")),
                        "frame_primed": stats.get("frame_primed"),
                        "prime_gap": stats.get("prime_gap"),
                        "prime_stats_json": to_json(stats),
                    }
                )
    return pd.DataFrame(rows)


def build_event_index(manifest: pd.DataFrame, annotation_root: Path) -> dict[str, pd.DataFrame]:
    recipes, recipe_steps, ingredients = build_recipe_tables(annotation_root)
    activities = build_activity_events(annotation_root)
    audio = build_audio_events(annotation_root)
    object_tracks, object_masks = build_object_tables(annotation_root)
    gaze_priming = build_gaze_priming(annotation_root)
    vqa_samples = build_vqa_samples(annotation_root)
    event_frames = [activities, recipe_steps, ingredients, audio]
    events = pd.concat([frame for frame in event_frames if not frame.empty], ignore_index=True)
    return {
        "videos": build_videos_table(manifest),
        "recipes": recipes,
        "events": events,
        "recipe_steps": recipe_steps,
        "ingredients": ingredients,
        "audio_events": audio,
        "object_tracks": object_tracks,
        "object_masks": object_masks,
        "gaze_priming": gaze_priming,
        "vqa_samples": vqa_samples,
    }

