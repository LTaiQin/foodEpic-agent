#!/usr/bin/env python3
"""Run a small visual VQA smoke test on sparse frames from one video."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.config import ModelConfig, load_env_file
from food_agent.paths import ProjectPaths


TIME_PATTERN = re.compile(r"<TIME\s+(\d+:\d+:\d+(?:\.\d+)?)")
BBOX_PATTERN = re.compile(r"<BBOX\s+([0-9.\s]+)>", re.IGNORECASE)

MOTION_FAMILIES = {
    "object_motion_object_movement_counting",
    "object_motion_object_movement_itinerary",
    "object_motion_stationary_object_localization",
    "3d_perception_fixture_interaction_counting",
}


@dataclass(frozen=True)
class FrameRef:
    time_s: float
    path: Path


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--probe-dir", type=Path, default=defaults.output_root / "video_vqa_probe")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--out-name", default="vision_vqa_smoke_results_v2_bbox_motion.json")
    return parser.parse_args()


def parse_hms(text: str) -> float:
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_times(question: str) -> list[float]:
    return [parse_hms(match.group(1)) for match in TIME_PATTERN.finditer(question)]


def extract_bbox(question: str) -> list[float] | None:
    match = BBOX_PATTERN.search(question)
    if not match:
        return None
    try:
        values = [float(value) for value in match.group(1).split()]
    except ValueError:
        return None
    if len(values) != 4:
        return None
    return values


def parse_choice_prediction(text: str, choices: list[Any]) -> int | None:
    stripped = text.strip()
    match = re.search(r"\b([0-4])\b", stripped)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(choices):
            return idx
    lowered = stripped.lower()
    for idx, choice in enumerate(choices):
        choice_text = " ".join(str(item) for item in choice) if isinstance(choice, list) else str(choice)
        if choice_text.lower() == lowered or choice_text.lower() in lowered:
            return idx
    return None


def load_frame_manifest(probe_dir: Path, video_id: str) -> list[FrameRef]:
    manifest_path = probe_dir / video_id / "frames_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [FrameRef(time_s=float(row["time_seconds"]), path=Path(row["path"])) for row in rows]


def choose_frames(task_family: str, times: list[float], all_frames: list[FrameRef]) -> list[FrameRef]:
    if not times:
        raise ValueError("question has no <TIME ...> anchor")
    if task_family in MOTION_FAMILIES:
        center = times[0]
        targets = [center - 6.0, center - 2.0, center + 2.0, center + 6.0]
    elif len(times) >= 2:
        targets = [times[0], (times[0] + times[-1]) / 2, times[-1]]
    else:
        center = times[0]
        targets = [center - 1.5, center, center + 1.5]
    chosen: list[FrameRef] = []
    used_paths: set[Path] = set()
    for target in targets:
        nearest = sorted(all_frames, key=lambda frame: abs(frame.time_s - target))
        for frame in nearest:
            if frame.path in used_paths:
                continue
            used_paths.add(frame.path)
            chosen.append(frame)
            break
    return chosen


def render_bbox_overlay(frame: FrameRef, bbox: list[float], out_dir: Path, sample_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sample_id.replace(":", "__")
    out_path = out_dir / f"{safe_id}_{frame.time_s:09.3f}s_bbox.jpg"
    image = Image.open(frame.path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=(255, 64, 64), width=8)
    image.save(out_path, quality=92)
    return out_path


def image_content(path: Path) -> dict[str, str]:
    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{raw}"}


def build_prompt(sample_id: str, task_family: str, question: str, choices: list[Any], frames: list[FrameRef], bbox_paths: list[Path]) -> list[dict[str, Any]]:
    lines = [
        "你在回答厨房第一视角视频的多项选择题。",
        "你会看到与题目时间最相关的若干关键帧。",
        "如果题目涉及动作变化、运动轨迹或计数，请重点比较前后帧差异。",
        "如果题目涉及 BBOX，带框图片标记了目标区域，判断时必须重点关注该区域，但也要结合整图上下文。",
        "请只输出最终选项编号 0-4，不要解释。",
        "",
        f"sample_id: {sample_id}",
        f"task_family: {task_family}",
        f"问题: {question}",
        "选项:",
    ]
    lines.extend(f"{idx}. {choice}" for idx, choice in enumerate(choices))
    content: list[dict[str, Any]] = [{"type": "input_text", "text": "\n".join(lines)}]
    for index, frame in enumerate(frames, start=1):
        content.append({"type": "input_text", "text": f"原始帧 {index}，时间={frame.time_s:.3f}s"})
        content.append(image_content(frame.path))
    for index, bbox_path in enumerate(bbox_paths, start=1):
        content.append({"type": "input_text", "text": f"带框帧 {index}，用于突出目标区域"})
        content.append(image_content(bbox_path))
    return [{"role": "user", "content": content}]


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["FOOD_AGENT_PROVIDER_MODE"] = "responses"
    cfg = ModelConfig.from_env()
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    df = pd.read_parquet(args.index_dir / "vqa_samples.parquet")
    subset = df[(df["primary_video_id"] == args.video_id) & (df["question"].str.contains("<TIME", na=False))].copy()
    samples: list[dict[str, Any]] = []
    for _, group in subset.groupby("task_family", sort=True):
        samples.extend(group.head(args.per_family).to_dict("records"))
    samples = samples[: args.max_samples]
    frames = load_frame_manifest(args.probe_dir, args.video_id)
    bbox_dir = args.probe_dir / args.video_id / "bbox_overlays"

    results: list[dict[str, Any]] = []
    for index, row in enumerate(samples, start=1):
        sample_id = str(row["vqa_id"])
        task_family = str(row["task_family"])
        question = str(row["question"])
        choices = json.loads(row["choices_json"])
        gold = int(row["correct_idx"])
        times = extract_times(question)
        chosen_frames = choose_frames(task_family, times, frames)
        bbox = extract_bbox(question)
        bbox_paths: list[Path] = []
        if bbox:
            for frame in chosen_frames:
                bbox_paths.append(render_bbox_overlay(frame, bbox, bbox_dir, sample_id))
        prompt = build_prompt(sample_id, task_family, question, choices, chosen_frames, bbox_paths)
        try:
            response = client.responses.create(model=cfg.model, input=prompt, temperature=0)
            raw_output = getattr(response, "output_text", None) or ""
        except Exception as exc:
            raw_output = f"ERROR: {type(exc).__name__}: {exc}"
        pred = None if raw_output.startswith("ERROR:") else parse_choice_prediction(raw_output, choices)
        result = {
            "sample_id": sample_id,
            "task_family": task_family,
            "question": question,
            "choices": choices,
            "gold": gold,
            "prediction": pred,
            "correct": pred == gold,
            "raw_output": raw_output,
            "frame_times": [frame.time_s for frame in chosen_frames],
            "frame_paths": [frame.path.as_posix() for frame in chosen_frames],
            "bbox_paths": [path.as_posix() for path in bbox_paths],
        }
        results.append(result)
        print(
            f"[{index}/{len(samples)}] {task_family} pred={pred} gold={gold} correct={pred == gold} "
            f"frames={[round(frame.time_s, 3) for frame in chosen_frames]} output={raw_output[:120]!r}",
            flush=True,
        )

    out_path = args.probe_dir / args.video_id / args.out_name
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = [row for row in results if row["prediction"] is not None]
    correct = sum(int(row["correct"]) for row in valid)
    summary = {
        "video_id": args.video_id,
        "sample_count": len(results),
        "valid_prediction_count": len(valid),
        "correct": correct,
        "accuracy": (correct / len(valid) if valid else None),
        "results_path": out_path.as_posix(),
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
