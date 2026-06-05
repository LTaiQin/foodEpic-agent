#!/usr/bin/env python3
"""Export one video's VQA questions plus sparse frame probes for manual inspection."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths


TIME_PATTERN = re.compile(r"<TIME\s+(\d+:\d+:\d+(?:\.\d+)?)")


@dataclass
class AnchorCandidate:
    time_s: float
    weight: int
    source: str
    sample_id: str
    task_family: str


@dataclass
class AnchorCluster:
    times: list[float] = field(default_factory=list)
    weight: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    sample_ids: set[str] = field(default_factory=set)
    task_families: Counter[str] = field(default_factory=Counter)

    def add(self, candidate: AnchorCandidate) -> None:
        self.times.append(candidate.time_s)
        self.weight += candidate.weight
        self.reasons[candidate.source] += 1
        self.sample_ids.add(candidate.sample_id)
        self.task_families[candidate.task_family] += 1

    @property
    def center_time(self) -> float:
        values = sorted(self.times)
        return values[len(values) // 2]


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "video_vqa_probe")
    parser.add_argument("--video-id", default=None, help="Video id like P08-20240617-130401. If omitted, auto-pick a diverse video.")
    parser.add_argument("--cluster-window", type=float, default=6.0, help="Merge nearby question times into one anchor if they are within this many seconds.")
    parser.add_argument("--frame-offsets", default="-1.5,0,1.5", help="Relative offsets in seconds around each anchor.")
    parser.add_argument("--min-frame-gap", type=float, default=1.0, help="Avoid exporting frames closer than this gap.")
    parser.add_argument("--max-gap-seconds", type=float, default=120.0, help="If timeline has a larger uncovered gap, inject one midpoint anchor.")
    parser.add_argument("--max-anchors", type=int, default=80, help="Cap anchor count before expanding to frame timestamps.")
    parser.add_argument("--include-choice-times", action="store_true", help="Also mine candidate times embedded inside answer choices.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame directory.")
    return parser.parse_args()


def parse_hms(text: str) -> float | None:
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def extract_time_strings(text: str) -> list[str]:
    return [match.group(1) for match in TIME_PATTERN.finditer(text or "")]


def probe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path.as_posix(),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    fps_raw = str(video_stream.get("avg_frame_rate") or "0/1")
    fps = 0.0
    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)
    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    return {
        "duration_seconds": duration,
        "fps": fps,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
    }


def auto_pick_video(df: pd.DataFrame) -> str:
    ranking: list[tuple[str, int, int, int]] = []
    for video_id, subset in df.groupby("primary_video_id"):
        unique_total = subset["task_family"].nunique()
        unique_non_fg = subset[~subset["task_family"].str.startswith("fine_grained")]["task_family"].nunique()
        ranking.append((video_id, unique_non_fg, unique_total, len(subset)))
    ranking.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    return ranking[0][0]


def build_candidates(df: pd.DataFrame, include_choice_times: bool) -> list[AnchorCandidate]:
    candidates: list[AnchorCandidate] = []
    for _, row in df.iterrows():
        sample_id = str(row["vqa_id"])
        task_family = str(row["task_family"])
        inputs = json.loads(row["inputs_json"])
        question = str(row["question"])
        seen: set[tuple[str, float]] = set()

        def add_candidate(time_s: float | None, source: str, weight: int) -> None:
            if time_s is None:
                return
            key = (source, round(time_s, 3))
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                AnchorCandidate(
                    time_s=time_s,
                    weight=weight,
                    source=source,
                    sample_id=sample_id,
                    task_family=task_family,
                )
            )

        for value in inputs.values():
            if not isinstance(value, dict):
                continue
            for key in ("time", "start_time", "end_time"):
                if value.get(key):
                    add_candidate(parse_hms(str(value[key])), f"inputs.{key}", 3)

        for time_str in extract_time_strings(question):
            add_candidate(parse_hms(time_str), "question.time", 3)

        if include_choice_times:
            for choice in json.loads(row["choices_json"]):
                if isinstance(choice, list):
                    choice_text = " ".join(str(item) for item in choice)
                else:
                    choice_text = str(choice)
                for time_str in extract_time_strings(choice_text):
                    add_candidate(parse_hms(time_str), "choice.time", 1)
    return candidates


def cluster_candidates(candidates: list[AnchorCandidate], window: float) -> list[AnchorCluster]:
    clusters: list[AnchorCluster] = []
    for candidate in sorted(candidates, key=lambda item: item.time_s):
        if clusters and abs(candidate.time_s - clusters[-1].center_time) <= window:
            clusters[-1].add(candidate)
        else:
            cluster = AnchorCluster()
            cluster.add(candidate)
            clusters.append(cluster)
    return clusters


def reduce_clusters(clusters: list[AnchorCluster], max_anchors: int) -> list[AnchorCluster]:
    if len(clusters) <= max_anchors:
        return clusters
    buckets: list[list[AnchorCluster]] = [[] for _ in range(max_anchors)]
    for index, cluster in enumerate(clusters):
        bucket_index = min(max_anchors - 1, math.floor(index * max_anchors / len(clusters)))
        buckets[bucket_index].append(cluster)
    reduced = [max(bucket, key=lambda item: (item.weight, len(item.sample_ids), item.center_time)) for bucket in buckets if bucket]
    return sorted(reduced, key=lambda item: item.center_time)


def add_gap_fill_anchors(anchor_times: list[float], duration: float, max_gap_seconds: float) -> list[float]:
    if not anchor_times:
        return [duration / 2] if duration > 0 else []
    points = [0.0] + sorted(anchor_times) + [duration]
    added: list[float] = []
    for left, right in zip(points, points[1:]):
        gap = right - left
        if gap <= max_gap_seconds:
            continue
        steps = int(gap // max_gap_seconds)
        for step in range(steps):
            midpoint = left + ((step + 1) / (steps + 1)) * gap
            added.append(midpoint)
    return added


def build_frame_times(anchor_times: list[float], offsets: list[float], duration: float, min_gap: float) -> list[float]:
    expanded: list[float] = []
    for anchor in sorted(anchor_times):
        for offset in offsets:
            time_s = min(duration, max(0.0, anchor + offset))
            if expanded and abs(time_s - expanded[-1]) < min_gap:
                continue
            expanded.append(time_s)
    deduped: list[float] = []
    for time_s in sorted(set(round(value, 3) for value in expanded)):
        if deduped and abs(time_s - deduped[-1]) < min_gap:
            continue
        deduped.append(time_s)
    return deduped


def find_video_path(data_root: Path, video_id: str) -> Path:
    participant_id = video_id.split("-", 1)[0]
    path = data_root / "Videos" / participant_id / f"{video_id}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"missing video file: {path}")
    return path


def export_frame(video_path: Path, time_s: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{time_s:.3f}",
        "-i",
        video_path.as_posix(),
        "-frames:v",
        "1",
        out_path.as_posix(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def format_time_s(time_s: float) -> str:
    hours = int(time_s // 3600)
    minutes = int((time_s % 3600) // 60)
    seconds = time_s % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def build_markdown(
    video_id: str,
    video_path: Path,
    video_meta: dict[str, Any],
    subset: pd.DataFrame,
    frame_times: list[float],
    selected_clusters: list[AnchorCluster],
    out_dir: Path,
) -> str:
    lines = [
        f"# Video VQA Probe: {video_id}",
        "",
        "## Video",
        f"- video_path: `{video_path}`",
        f"- duration_seconds: `{video_meta['duration_seconds']:.3f}`",
        f"- fps: `{video_meta['fps']:.3f}`",
        f"- resolution: `{video_meta['width']}x{video_meta['height']}`",
        f"- total_vqa_questions: `{len(subset)}`",
        f"- exported_frames: `{len(frame_times)}`",
        "",
        "## Task Family Counts",
    ]
    counts = subset["task_family"].value_counts()
    for task_family, count in counts.items():
        lines.append(f"- `{task_family}`: {count}")
    lines.extend(["", "## Anchor Summary"])
    for index, cluster in enumerate(selected_clusters, start=1):
        lines.append(
            f"- anchor_{index:03d}: `{format_time_s(cluster.center_time)}`"
            f" weight={cluster.weight} samples={len(cluster.sample_ids)}"
            f" reasons={dict(cluster.reasons)} top_tasks={dict(cluster.task_families.most_common(3))}"
        )
    lines.extend(["", "## Exported Frames"])
    for index, time_s in enumerate(frame_times, start=1):
        rel = (out_dir / "frames" / f"frame_{index:03d}_{time_s:09.3f}s.jpg").as_posix()
        lines.append(f"- `{index:03d}` `{format_time_s(time_s)}` -> `{rel}`")
    lines.extend(["", "## VQA Questions"])
    for task_family, task_df in subset.groupby("task_family", sort=True):
        lines.extend(["", f"### {task_family}"])
        for _, row in task_df.iterrows():
            question = str(row["question"])
            choices = json.loads(row["choices_json"])
            correct_idx = int(row["correct_idx"])
            inputs = json.loads(row["inputs_json"])
            lines.append(f"- sample_id: `{row['vqa_id']}`")
            lines.append(f"  question: {question}")
            lines.append(f"  correct_idx: `{correct_idx}`")
            lines.append(f"  primary_video_id: `{row['primary_video_id']}`")
            if inputs:
                lines.append(f"  inputs: `{json.dumps(inputs, ensure_ascii=False)}`")
            lines.append("  choices:")
            for choice_index, choice in enumerate(choices):
                prefix = "*" if choice_index == correct_idx else "-"
                lines.append(f"  {prefix} [{choice_index}] {choice}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    vqa_path = args.index_dir / "vqa_samples.parquet"
    if not vqa_path.exists():
        raise FileNotFoundError(f"missing VQA index: {vqa_path}")
    df = pd.read_parquet(vqa_path)
    video_id = args.video_id or auto_pick_video(df)
    subset = df[df["primary_video_id"] == video_id].copy()
    if subset.empty:
        raise ValueError(f"no VQA samples found for video_id={video_id}")

    video_path = find_video_path(args.data_root, video_id)
    video_meta = probe_video(video_path)
    candidates = build_candidates(subset, include_choice_times=args.include_choice_times)
    clusters = cluster_candidates(candidates, window=args.cluster_window)
    selected_clusters = reduce_clusters(clusters, max_anchors=args.max_anchors)
    anchor_times = [cluster.center_time for cluster in selected_clusters]
    anchor_times.extend(add_gap_fill_anchors(anchor_times, video_meta["duration_seconds"], args.max_gap_seconds))
    anchor_times = sorted(set(round(value, 3) for value in anchor_times))
    offsets = [float(item) for item in args.frame_offsets.split(",") if item.strip()]
    frame_times = build_frame_times(anchor_times, offsets, video_meta["duration_seconds"], args.min_frame_gap)

    out_dir = args.out_dir / video_id
    frames_dir = out_dir / "frames"
    if frames_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{frames_dir} exists; pass --overwrite to refresh outputs")
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for index, time_s in enumerate(frame_times, start=1):
        out_path = frames_dir / f"frame_{index:03d}_{time_s:09.3f}s.jpg"
        export_frame(video_path, time_s, out_path)
        manifest.append(
            {
                "index": index,
                "time_seconds": time_s,
                "time_hms": format_time_s(time_s),
                "path": out_path.as_posix(),
            }
        )

    markdown = build_markdown(video_id, video_path, video_meta, subset, frame_times, selected_clusters, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "video_vqa_overview.md").write_text(markdown, encoding="utf-8")

    summary = {
        "video_id": video_id,
        "video_path": video_path.as_posix(),
        "question_count": len(subset),
        "task_family_count": int(subset["task_family"].nunique()),
        "cluster_count": len(clusters),
        "selected_anchor_count": len(selected_clusters),
        "exported_frame_count": len(frame_times),
        "duration_seconds": video_meta["duration_seconds"],
        "fps": video_meta["fps"],
        "out_dir": out_dir.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
