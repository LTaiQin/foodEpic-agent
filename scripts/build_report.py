#!/usr/bin/env python3
"""Build a compact Markdown report for current dataset, index, and baseline status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--out", type=Path, default=defaults.output_root / "reports" / "status_report.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    args = parse_args()
    manifest_path = args.output_root / "dataset_manifest.parquet"
    index_dir = args.output_root / "event_index"
    food_metrics = read_json(args.output_root / "results" / "food_state_metrics.json")
    spatial_metrics = read_json(args.output_root / "results" / "spatial_context_metrics.json")
    lines = ["# foodEpic-agent Status Report", ""]
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        lines += [
            "## Dataset Manifest",
            "",
            f"- files: {len(manifest)}",
            f"- total_size_gb: {manifest['size_bytes'].sum() / (1024 ** 3):.2f}",
            f"- mp4: {(manifest['file_type'] == 'mp4').sum()}",
            f"- hdf5: {(manifest['file_type'] == 'hdf5').sum()}",
            f"- deferred: {(manifest['status'] == 'deferred').sum()}",
            "",
        ]
    if index_dir.exists():
        lines += ["## Event Index", ""]
        for path in sorted(index_dir.glob("*.parquet")):
            rows = len(pd.read_parquet(path))
            lines.append(f"- {path.stem}: {rows}")
        lines.append("")
    if food_metrics:
        lines += [
            "## Food State Coverage",
            "",
            f"- recipe_video_count: {food_metrics.get('recipe_video_count')}",
            f"- ingredient_video_count: {food_metrics.get('ingredient_video_count')}",
            f"- recipe_step_event_count: {food_metrics.get('recipe_step_event_count')}",
            f"- ingredient_event_count: {food_metrics.get('ingredient_event_count')}",
            "",
        ]
    if spatial_metrics:
        lines += [
            "## Spatial Context Coverage",
            "",
            f"- object_track_rows: {spatial_metrics.get('object_track_rows')}",
            f"- object_mask_rows: {spatial_metrics.get('object_mask_rows')}",
            f"- gaze_rows: {spatial_metrics.get('gaze_rows')}",
            f"- audio_rows: {spatial_metrics.get('audio_rows')}",
            "",
        ]
    lines += [
        "## Advantage-Oriented Evaluation Criteria",
        "",
        "- Accuracy: VQA/task answer correctness.",
        "- Evidence rate: fraction of answers with event/frame/time evidence.",
        "- State coverage: fraction of videos with recipe/ingredient/object/gaze/audio state available.",
        "- Tool-use rate: fraction of answers using relevant structured tools.",
        "- Reliability: one minus failure-type rate.",
        "- FoodAgent Advantage Score = 0.40 accuracy + 0.25 evidence + 0.15 state coverage + 0.10 tool use + 0.10 reliability.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

