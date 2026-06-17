"""Helpers for loading HD-EPIC annotation data."""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

_ANNOTATION_ROOT = None


def _get_annotation_root() -> Path:
    global _ANNOTATION_ROOT
    if _ANNOTATION_ROOT is None:
        from food_agent.config import ProjectConfig
        cfg = ProjectConfig.from_env()
        _ANNOTATION_ROOT = Path(cfg.annotation_root)
    return _ANNOTATION_ROOT


def load_audio_annotations(participant_id: str, video_id: str) -> List[Dict]:
    """Load audio event annotations for a video.

    Returns list of dicts with keys: type, start_time, end_time, duration.
    """
    root = _get_annotation_root()
    audio_dir = root / "audio-annotations"
    if not audio_dir.exists():
        return []

    # Try to find the annotation file
    csv_path = audio_dir / f"{video_id}_audio_events.csv"
    if not csv_path.exists():
        # Try alternate naming
        csv_path = audio_dir / f"{video_id}.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    events = []
    for _, row in df.iterrows():
        events.append({
            "type": str(row.get("event_type", row.get("label", "unknown"))),
            "start_time": float(row.get("start_time", row.get("onset", 0))),
            "end_time": float(row.get("end_time", row.get("offset", 0))),
            "duration": float(row.get("duration", 0)),
        })
    return events


def load_narrations(participant_id: str, video_id: str) -> List[Dict]:
    """Load narration/action segment annotations."""
    root = _get_annotation_root()
    narr_dir = root / "narrations-and-action-segments"
    if not narr_dir.exists():
        return []
    csv_path = narr_dir / f"{video_id}.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "start_time": float(row.get("start_time", 0)),
            "end_time": float(row.get("end_time", 0)),
            "narration": str(row.get("narration", "")),
            "action": str(row.get("action", "")),
        })
    return records


def load_vqa_samples(video_id: str) -> List[Dict]:
    """Load VQA benchmark questions for a video."""
    root = _get_annotation_root()
    vqa_dir = root / "vqa-benchmark"
    if not vqa_dir.exists():
        return []

    # Look for the VQA file
    for pattern in [f"{video_id}*.json", f"{video_id}*.jsonl"]:
        files = list(vqa_dir.glob(pattern))
        if files:
            break
    else:
        return []

    path = files[0]
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("questions", data.get("samples", []))
    else:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
