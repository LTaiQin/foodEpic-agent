#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.graph import VideoGraphBuilder
from food_agent.memory import GraphMemoryStore
from food_agent.paths import ProjectPaths
from food_agent.tools import AgentToolbox


REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "graph_agent_audio_peak_audit.json"


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-file", type=Path, default=REPORT_PATH)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--samples-per-source", type=int, default=12)
    parser.add_argument("--rebuild-graphs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_env()
    ingredient_df = pd.read_parquet(paths.output_root / "event_index" / "ingredients.parquet")
    recipe_step_df = pd.read_parquet(paths.output_root / "event_index" / "recipe_steps.parquet")
    samples = select_windows(ingredient_df=ingredient_df, recipe_step_df=recipe_step_df, seed=args.seed, samples_per_source=args.samples_per_source)
    builder = VideoGraphBuilder(paths)
    cached_toolboxes: dict[str, AgentToolbox] = {}
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        video_id = str(sample["video_id"])
        toolbox = cached_toolboxes.get(video_id)
        if toolbox is None:
            store = GraphMemoryStore(paths.graph_memory_root / video_id)
            if args.rebuild_graphs or not store.query_nodes(video_id=video_id, limit=1):
                store = builder.build(video_id)
            toolbox = AgentToolbox(store=store, paths=paths, model_client=object(), video_id=video_id)
            cached_toolboxes[video_id] = toolbox
        rows.append(audit_window(toolbox=toolbox, sample=sample))
        if index % 10 == 0 or index == len(samples):
            print(f"[audio-audit] processed {index}/{len(samples)} windows", flush=True)
    payload = build_report(rows=rows, seed=args.seed)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"report_path={args.out_file}")
    return 0


def select_windows(*, ingredient_df: pd.DataFrame, recipe_step_df: pd.DataFrame, seed: int, samples_per_source: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for source_name, df, text_col in [
        ("ingredient_event", ingredient_df, "text"),
        ("recipe_step", recipe_step_df, "text"),
    ]:
        usable = df[df["start_time"].notna() & df["end_time"].notna()].copy()
        usable = usable[usable["video_id"].notna()].copy()
        records = usable.to_dict("records")
        rng.shuffle(records)
        for row in records[:samples_per_source]:
            start = float(row["start_time"])
            end = float(row["end_time"])
            window_start = max(0.0, start - 1.5)
            window_end = max(window_start + 2.0, end + 3.0)
            samples.append(
                {
                    "source_type": source_name,
                    "video_id": str(row["video_id"]),
                    "event_id": str(row.get("event_id") or ""),
                    "label": str(row.get(text_col) or row.get("label") or ""),
                    "anchor_start": start,
                    "anchor_end": end,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )
    samples.sort(key=lambda item: (item["source_type"], item["video_id"], item["anchor_start"]))
    return samples


def audit_window(*, toolbox: AgentToolbox, sample: dict[str, Any]) -> dict[str, Any]:
    detect = toolbox.detect_audio_peaks(
        start_time=float(sample["window_start"]),
        end_time=float(sample["window_end"]),
        window_s=0.4,
        top_k=4,
    )
    peaks = list(detect.get("peaks") or [])
    peak_times = [float(item.get("time_s")) for item in peaks if item.get("time_s") is not None]
    sampled = toolbox.sample_frames_around_peaks(peak_times=peak_times, radius_s=0.5, frames_per_peak=3, tag="audio_audit") if peak_times else {"count": 0, "artifact_paths": [], "items": []}
    nearby_audio_nodes = toolbox.query_time(
        start_time=float(sample["window_start"]),
        end_time=float(sample["window_end"]),
        limit=30,
    )
    audio_event_count = sum(1 for node in nearby_audio_nodes.get("nodes", []) if str(node.get("node_type") or "") == "audio_event")
    return {
        **sample,
        "peak_count": int(detect.get("count") or 0),
        "peak_times": peak_times,
        "peak_scores": [float(item.get("score") or 0.0) for item in peaks],
        "sampled_peak_windows": int(sampled.get("count") or 0),
        "sampled_artifact_count": len(sampled.get("artifact_paths") or []),
        "audio_event_count": audio_event_count,
        "has_audio_peaks": bool(peak_times),
        "has_peak_frames": bool(sampled.get("artifact_paths")),
    }


def build_report(*, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    source_counter = Counter(row["source_type"] for row in rows)
    with_peaks = [row for row in rows if row["has_audio_peaks"]]
    with_peak_frames = [row for row in rows if row["has_peak_frames"]]
    with_audio_events = [row for row in rows if row["audio_event_count"] > 0]
    return {
        "summary": {
            "seed": seed,
            "window_count": len(rows),
            "source_breakdown": dict(source_counter),
            "peak_non_empty_rate": round(len(with_peaks) / len(rows), 4) if rows else 0.0,
            "peak_frame_rate": round(len(with_peak_frames) / len(rows), 4) if rows else 0.0,
            "audio_event_overlap_rate": round(len(with_audio_events) / len(rows), 4) if rows else 0.0,
            "avg_peak_count": round(sum(row["peak_count"] for row in rows) / len(rows), 4) if rows else 0.0,
            "avg_sampled_artifact_count": round(sum(row["sampled_artifact_count"] for row in rows) / len(rows), 4) if rows else 0.0,
        },
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
