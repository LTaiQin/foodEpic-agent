#!/usr/bin/env python3
"""Evaluate spatial context coverage over object / gaze / audio tables."""

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
from food_agent.spatial_store import SpatialContextStore


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out", type=Path, default=defaults.output_root / "results" / "spatial_context_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = SpatialContextStore(args.index_dir)
    obj = pd.read_parquet(args.index_dir / "object_tracks.parquet")
    gaze = pd.read_parquet(args.index_dir / "gaze_priming.parquet")
    audio = pd.read_parquet(args.index_dir / "audio_events.parquet")
    videos = sorted(set(obj["video_id"].dropna()) | set(gaze["video_id"].dropna()) | set(audio["video_id"].dropna()))
    samples = []
    for video_id in videos[:10]:
        ctx = store.combined_context(video_id, time=30.0)
        samples.append(
            {
                "video_id": video_id,
                "object_tracks": len(ctx.object_tracks),
                "object_masks": len(ctx.object_masks),
                "gaze_priming": len(ctx.gaze_priming),
                "audio_events": len(ctx.audio_events),
            }
        )
    metrics = {
        "video_count": len(videos),
        "object_track_rows": int(len(obj)),
        "object_mask_rows": int(len(pd.read_parquet(args.index_dir / "object_masks.parquet"))),
        "gaze_rows": int(len(gaze)),
        "audio_rows": int(len(audio)),
        "sample_contexts": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

