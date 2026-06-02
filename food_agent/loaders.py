"""Lightweight data loaders for HD-EPIC files."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import h5py


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_head(path: str | Path, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for _, line in zip(range(limit), f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def csv_head_info(path: str | Path, limit: int = 5) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        sample_count = sum(1 for _, _ in zip(range(limit), reader))
    return {"columns": header, "sample_rows": sample_count}


def video_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cap = cv2.VideoCapture(path.as_posix())
    try:
        if not cap.isOpened():
            return {"status": "blocked", "reason": "cv2 could not open video"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else None
        return {
            "status": "ok",
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration,
        }
    finally:
        cap.release()


def hdf5_metadata(path: str | Path) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            obj = h5[key]
            datasets[key] = {
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
            }
    return {"datasets": datasets, "dataset_count": len(datasets)}


def json_summary(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict):
        return {"json_type": "dict", "top_level_count": len(data), "sample_keys": list(data)[:10]}
    if isinstance(data, list):
        return {"json_type": "list", "top_level_count": len(data)}
    return {"json_type": type(data).__name__}

