#!/usr/bin/env python3
"""Single-video agent probe: audio-guided keyframes, frame observations, summary, and sampled VQA."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image, ImageDraw
from scipy.signal import find_peaks

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

WHOLE_VIDEO_FAMILIES = {
    "ingredient_ingredients_order",
    "nutrition_video_nutrition_estimation",
    "recipe_multi_recipe_recognition",
    "object_motion_object_movement_itinerary",
}

WHOLE_VIDEO_TRACKING_FAMILIES = {
    "3d_perception_fixture_interaction_counting",
    "object_motion_object_movement_counting",
    "object_motion_object_movement_itinerary",
}

VOLATILE_TASK_FAMILIES = {
    "3d_perception_fixture_location",
    "fine_grained_action_localization",
    "fine_grained_action_recognition",
    "gaze_gaze_estimation",
    "object_motion_object_movement_counting",
}

THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Anchor:
    time_s: float
    source: str
    related_tasks: tuple[str, ...]
    related_samples: tuple[str, ...]


@dataclass(frozen=True)
class FrameInfo:
    frame_id: str
    time_s: float
    time_hms: str
    path: Path
    source: str


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "single_video_agent_probe")
    parser.add_argument("--video-id", default="P08-20240617-130401")
    parser.add_argument("--question-anchor-limit", type=int, default=18)
    parser.add_argument("--audio-peak-limit", type=int, default=8)
    parser.add_argument("--frame-offsets", default="-2,0,2")
    parser.add_argument("--gap-fill-seconds", type=float, default=180.0)
    parser.add_argument("--min-frame-gap", type=float, default=1.0)
    parser.add_argument("--audio-window-seconds", type=float, default=0.5)
    parser.add_argument("--audio-min-peak-gap", type=float, default=8.0)
    parser.add_argument("--audio-exclusion-radius", type=float, default=10.0)
    parser.add_argument("--sample-vqa-per-family", type=int, default=1)
    parser.add_argument("--all-vqa", action="store_true")
    parser.add_argument("--max-vqa", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--cap-per-family", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def get_thread_client(cfg: ModelConfig) -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        THREAD_LOCAL.client = client
    return client


def parse_hms(text: str) -> float:
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_hms(time_s: float) -> str:
    hours = int(time_s // 3600)
    minutes = int((time_s % 3600) // 60)
    seconds = time_s % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def extract_times_from_inputs(inputs: Any, video_id: str) -> list[float]:
    times: list[float] = []
    if isinstance(inputs, dict):
        current_id = inputs.get("id")
        same_video = current_id in (None, video_id)
        for key in ("time", "start_time", "end_time"):
            value = inputs.get(key)
            if same_video and value:
                times.append(parse_hms(str(value)))
        for value in inputs.values():
            times.extend(extract_times_from_inputs(value, video_id))
    elif isinstance(inputs, list):
        for value in inputs:
            times.extend(extract_times_from_inputs(value, video_id))
    return times


def extract_row_times(row: pd.Series, video_id: str) -> list[float]:
    times = [parse_hms(match.group(1)) for match in TIME_PATTERN.finditer(str(row["question"]))]
    try:
        inputs = json.loads(row["inputs_json"])
    except Exception:
        inputs = {}
    times.extend(extract_times_from_inputs(inputs, video_id))
    deduped = sorted({round(time_s, 3) for time_s in times})
    return deduped


def probe_video(video_path: Path) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                video_path.as_posix(),
            ],
            text=True,
        )
    )
    streams = payload.get("streams", [])
    stream = next((item for item in streams if item.get("codec_type") == "video"), {})
    fps_raw = str(stream.get("avg_frame_rate") or "0/1")
    fps = 0.0
    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        if float(den) != 0:
            fps = float(num) / float(den)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    return {
        "duration_seconds": duration,
        "fps": fps,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def find_video_path(data_root: Path, video_id: str) -> Path:
    participant = video_id.split("-", 1)[0]
    path = data_root / "Videos" / participant / f"{video_id}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"missing video file: {path}")
    return path


def find_audio_path(data_root: Path, video_id: str) -> Path:
    participant = video_id.split("-", 1)[0]
    path = data_root / "Audio-HDF5" / participant / f"{participant}_audio.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"missing audio file: {path}")
    return path


def extract_question_anchors(df: pd.DataFrame, limit: int) -> list[Anchor]:
    grouped: dict[int, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sample_id = str(row["vqa_id"])
        task_family = str(row["task_family"])
        question = str(row["question"])
        raw_times = [parse_hms(match.group(1)) for match in TIME_PATTERN.finditer(question)]
        if not raw_times:
            inputs = json.loads(row["inputs_json"])
            for value in inputs.values():
                if isinstance(value, dict):
                    for key in ("time", "start_time", "end_time"):
                        if value.get(key):
                            raw_times.append(parse_hms(str(value[key])))
        for time_s in raw_times:
            bucket = grouped.setdefault(int(round(time_s)), {"tasks": set(), "samples": set(), "time": time_s, "count": 0})
            bucket["tasks"].add(task_family)
            bucket["samples"].add(sample_id)
            bucket["count"] += 1
    ranked = sorted(grouped.values(), key=lambda item: (-item["count"], len(item["tasks"]), item["time"]))
    selected = sorted(ranked[:limit], key=lambda item: item["time"])
    return [
        Anchor(
            time_s=float(item["time"]),
            source="question_anchor",
            related_tasks=tuple(sorted(item["tasks"])),
            related_samples=tuple(sorted(item["samples"])),
        )
        for item in selected
    ]


def extract_audio_peak_anchors(
    audio_path: Path,
    video_id: str,
    duration_seconds: float,
    exclusion_times: list[float],
    peak_limit: int,
    window_seconds: float,
    min_peak_gap: float,
    exclusion_radius: float,
) -> list[Anchor]:
    with h5py.File(audio_path, "r") as handle:
        samples = np.asarray(handle[video_id], dtype=np.float32)
    if duration_seconds <= 0 or samples.size == 0:
        return []
    sample_rate = int(round(samples.size / duration_seconds))
    window_size = max(1, int(window_seconds * sample_rate))
    usable = (samples.size // window_size) * window_size
    pooled = samples[:usable].reshape(-1, window_size)
    energy = np.sqrt(np.mean(np.square(pooled), axis=1))
    distance = max(1, int(min_peak_gap / window_seconds))
    peak_idx, _ = find_peaks(energy, distance=distance, prominence=np.std(energy) * 0.3)
    peak_items = sorted(((energy[index], index) for index in peak_idx), reverse=True)
    chosen: list[Anchor] = []
    for score, index in peak_items:
        time_s = (index + 0.5) * window_seconds
        if any(abs(time_s - other) <= exclusion_radius for other in exclusion_times):
            continue
        chosen.append(
            Anchor(
                time_s=time_s,
                source=f"audio_peak:{score:.4f}",
                related_tasks=(),
                related_samples=(),
            )
        )
        if len(chosen) >= peak_limit:
            break
    return sorted(chosen, key=lambda item: item.time_s)


def add_gap_fill_times(anchor_times: list[float], duration_seconds: float, gap_seconds: float) -> list[float]:
    if not anchor_times:
        return [duration_seconds / 2] if duration_seconds > 0 else []
    points = [0.0] + sorted(anchor_times) + [duration_seconds]
    fills: list[float] = []
    for left, right in zip(points, points[1:]):
        if right - left <= gap_seconds:
            continue
        fills.append((left + right) / 2)
    return fills


def build_frame_list(
    anchors: list[Anchor],
    fill_times: list[float],
    offsets: list[float],
    duration_seconds: float,
    min_gap: float,
    frames_dir: Path,
) -> list[FrameInfo]:
    time_points: list[tuple[float, str]] = []
    for anchor in anchors:
        for offset in offsets:
            time_points.append((min(duration_seconds, max(0.0, anchor.time_s + offset)), anchor.source))
    for time_s in fill_times:
        time_points.append((time_s, "gap_fill"))
    merged: list[tuple[float, str]] = []
    for time_s, source in sorted(time_points):
        if merged and abs(time_s - merged[-1][0]) < min_gap:
            continue
        merged.append((round(time_s, 3), source))
    frames: list[FrameInfo] = []
    for index, (time_s, source) in enumerate(merged, start=1):
        frame_id = f"frame_{index:03d}"
        path = frames_dir / f"{frame_id}_{time_s:09.3f}s.jpg"
        frames.append(FrameInfo(frame_id=frame_id, time_s=time_s, time_hms=format_hms(time_s), path=path, source=source))
    return frames


def export_frame(video_path: Path, frame: FrameInfo) -> None:
    frame.path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{frame.time_s:.3f}",
            "-i",
            video_path.as_posix(),
            "-frames:v",
            "1",
            frame.path.as_posix(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def image_part(path: Path) -> dict[str, str]:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{payload}"}


def create_response(
    client: OpenAI,
    model: str,
    content: list[dict[str, Any]],
    *,
    temperature: float = 0,
    timeout_seconds: float = 90.0,
) -> Any:
    return client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        temperature=temperature,
        timeout=timeout_seconds,
    )


def normalize_keywords(values: list[str]) -> list[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        for token in re.findall(r"[a-zA-Z]+", text):
            if len(token) >= 3:
                tokens.add(token)
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            tokens.add(token)
        if not re.search(r"[a-zA-Z\u4e00-\u9fff]", text):
            continue
        compact = re.sub(r"\s+", " ", text)
        if len(compact) >= 2:
            tokens.add(compact)
    return sorted(tokens)


def describe_frame(client: OpenAI, model: str, frame: FrameInfo, related_tasks: list[str]) -> dict[str, Any]:
    prompt = (
        "你在观察厨房第一视角视频中的一张关键帧。"
        "请只根据这张图做保守描述，不要编造看不见的信息。"
        "输出 JSON，字段固定为："
        '{"scene_location":"","visible_ingredients":[],"visible_tools":[],"hand_interaction":"","ongoing_action":"",'
        '"possible_step":"","attention_targets":[],"state_change_hint":"","confidence":0.0}'
        f"\n相关题型提示: {related_tasks or ['general']}"
        f"\n关键要求:"
        "\n1. ingredient 类关注食材与加料/称重/搅拌。"
        "\n2. recipe 类关注当前像哪一步、准备/烹饪/清理。"
        "\n3. gaze/3d 类关注视角朝向与空间参照。"
        "\n4. motion 类关注物体是否正在移动或发生变化。"
        f"\n帧时间: {frame.time_hms}"
    )
    response = create_response(
        client,
        model,
        [{"type": "input_text", "text": prompt}, image_part(frame.path)],
        temperature=0,
    )
    text = getattr(response, "output_text", None) or ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            payload["raw_output"] = text
            return payload
        except json.JSONDecodeError:
            pass
    return {"raw_output": text}


def describe_bbox_target(client: OpenAI, model: str, frame: FrameInfo, overlay_path: Path, focus_crop_path: Path) -> dict[str, Any]:
    prompt = (
        "你在看一个厨房第一视角视频中的参考帧，红框标出了题目中的目标物体。"
        "第一张图是整帧带框图，第二张图是同一区域的放大图。"
        "请优先识别红框中心附近、最可能被题目指代的对象，不要泛泛描述整张图。"
        "输出 JSON，字段固定为："
        '{"target_object":"","target_category":"","target_location":"","keywords":[],"is_movable":true,"state_hint":""}'
        f"\n参考时刻: {frame.time_hms}"
    )
    response = create_response(
        client,
        model,
        [{"type": "input_text", "text": prompt}, image_part(overlay_path), image_part(focus_crop_path)],
        temperature=0,
    )
    text = getattr(response, "output_text", None) or ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            payload["raw_output"] = text
            if not isinstance(payload.get("keywords"), list):
                payload["keywords"] = []
            payload["keywords"] = normalize_keywords([str(item) for item in payload.get("keywords", [])] + [str(payload.get("target_object", "")), str(payload.get("target_category", "")), str(payload.get("target_location", ""))])
            return payload
        except json.JSONDecodeError:
            pass
    return {"target_object": "", "target_category": "", "target_location": "", "keywords": [], "is_movable": True, "state_hint": "", "raw_output": text}


def task_specific_guidance(task_family: str) -> str:
    if task_family == "3d_perception_fixture_location":
        return "这是第一视角空间方位题。请按钟表方向理解选项，依据画面中心朝向判断目标在视线的几点钟方向。"
    if task_family == "gaze_gaze_estimation":
        return "这是视线估计题。重点看画面中心、手将要接触的位置、以及最清晰对准的区域，不要只看大致朝向。"
    if task_family == "3d_perception_fixture_interaction_counting":
        return (
            "这是固定设施/柜门交互计数题。优先判断对象是不是可开合的柜门、抽屉或门板，并统计开/关动作次数。"
            "不要把普通厨房物体误判成计数对象。"
        )
    if task_family == "object_motion_object_movement_itinerary":
        return (
            "这是物体移动轨迹题。先识别目标物体，再比较它在若干时刻的地点变化，"
            "优先找起点、经过位置、终点的连续迁移路径，不要把静态摆放误认为移动。"
        )
    if task_family == "object_motion_stationary_object_localization":
        return (
            "这是静止起点定位题。要判断从哪个候选起点开始，这个物体在后续很长时间里都没再改变位置。"
            "请重点比较候选时刻之后的多个检查点，而不是只看起点那一帧。"
        )
    if task_family == "ingredient_ingredient_retrieval":
        return (
            "这是食材加入检索题。重点比较时间段前后哪些食材新出现、被倒入、被加入碗/锅/盘中。"
            "不要只看静态存在，要优先判断新增食材。"
        )
    if task_family == "ingredient_ingredient_adding_localization":
        return (
            "这是食材加入定位题。你需要判断哪一个候选时间段最像该食材被加入的时刻，"
            "重点找倒入、放入、撒入、加入容器的证据。"
        )
    if task_family == "nutrition_nutrition_change":
        return (
            "这是营养变化题。先判断该时间段新增了什么食材，再用新增食材推断营养变化，"
            "不要脱离食材变化直接猜数值。"
        )
    if task_family in MOTION_FAMILIES:
        return (
            "这是物体运动/计数题。重点比较目标物体在前后帧中的位置、朝向和是否发生移动，"
            "不要只根据单帧外观判断。"
        )
    if "gaze" in task_family:
        return "这是视线相关题。重点判断第一视角正前方、最被关注的区域和接下来最可能交互的对象。"
    if "fine_grained" in task_family:
        return "这是细粒度动作题。重点比较手部动作、接触对象和动作目的。"
    if "recipe" in task_family:
        return "这是菜谱/步骤题。重点判断当前动作更像准备、加料、搅拌、烹饪还是装盘/清理。"
    return "请根据视频记忆与关键帧证据做保守判断。"


def ensure_frame_observation(
    client: OpenAI,
    model: str,
    frame: FrameInfo,
    related_tasks: list[str],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    if cache_lock is None:
        if frame.frame_id in cache:
            return cache[frame.frame_id]
        observation = describe_frame(client, model, frame, related_tasks)
        payload = {
            "frame_id": frame.frame_id,
            "time_s": frame.time_s,
            "time_hms": frame.time_hms,
            "path": frame.path.as_posix(),
            "source": frame.source,
            "observation": observation,
        }
        cache[frame.frame_id] = payload
        cache_path.write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    with cache_lock:
        if frame.frame_id in cache:
            return cache[frame.frame_id]
    observation = describe_frame(client, model, frame, related_tasks)
    payload = {
        "frame_id": frame.frame_id,
        "time_s": frame.time_s,
        "time_hms": frame.time_hms,
        "path": frame.path.as_posix(),
        "source": frame.source,
        "observation": observation,
    }
    with cache_lock:
        existing = cache.get(frame.frame_id)
        if existing is not None:
            return existing
        cache[frame.frame_id] = payload
        cache_path.write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def ensure_bbox_target_profile(
    client: OpenAI,
    model: str,
    sample_id: str,
    frame: FrameInfo,
    bbox: list[float],
    overlay_dir: Path,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    if cache_lock is None:
        if sample_id in cache:
            return cache[sample_id]
    else:
        with cache_lock:
            if sample_id in cache:
                return cache[sample_id]
    overlay = render_bbox_overlay(frame, bbox, overlay_dir, f"{sample_id}_target_profile")
    focus_crop = render_bbox_focus_crop(frame, bbox, overlay_dir, f"{sample_id}_target_profile")
    profile = describe_bbox_target(client, model, frame, overlay, focus_crop)
    payload = {
        "sample_id": sample_id,
        "reference_frame_id": frame.frame_id,
        "reference_time_hms": frame.time_hms,
        "overlay_path": overlay.as_posix(),
        "focus_crop_path": focus_crop.as_posix(),
        "profile": profile,
    }
    if cache_lock is None:
        cache[sample_id] = payload
        cache_path.write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with cache_lock:
            existing = cache.get(sample_id)
            if existing is not None:
                return existing
            cache[sample_id] = payload
            cache_path.write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize_video_memory(client: OpenAI, model: str, video_id: str, frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact_rows = [
        {
            "frame_id": row["frame_id"],
            "time_hms": row["time_hms"],
            "source": row["source"],
            "scene_location": row["observation"].get("scene_location"),
            "visible_ingredients": row["observation"].get("visible_ingredients"),
            "visible_tools": row["observation"].get("visible_tools"),
            "hand_interaction": row["observation"].get("hand_interaction"),
            "ongoing_action": row["observation"].get("ongoing_action"),
            "possible_step": row["observation"].get("possible_step"),
            "attention_targets": row["observation"].get("attention_targets"),
            "state_change_hint": row["observation"].get("state_change_hint"),
        }
        for row in frame_rows
    ]
    prompt = (
        "你将基于一个厨房第一视角视频的多张关键帧观察结果，构建该视频的过程记忆。"
        "请输出 JSON，字段固定为："
        '{"video_profile":{"likely_recipe":"","likely_stage_patterns":[],"ingredient_inventory":[],"tool_inventory":[]},'
        '"timeline_events":[{"time_hms":"","event":"","evidence_frames":[]}],'
        '"state_summary":{"ingredients_seen":[],"actions_seen":[],"spatial_focus":[]},'
        '"notes":[]}'
        f"\n视频ID: {video_id}\n关键帧观察:\n{json.dumps(compact_rows, ensure_ascii=False)}"
    )
    response = create_response(
        client,
        model,
        [{"type": "input_text", "text": prompt}],
        temperature=0,
    )
    text = getattr(response, "output_text", None) or ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            payload["raw_output"] = text
            return payload
        except json.JSONDecodeError:
            pass
    return {"raw_output": text}


def extract_bbox(question: str) -> list[float] | None:
    match = BBOX_PATTERN.search(question)
    if not match:
        return None
    try:
        values = [float(value) for value in match.group(1).split()]
    except ValueError:
        return None
    return values if len(values) == 4 else None


def render_bbox_overlay(frame: FrameInfo, bbox: list[float], out_dir: Path, sample_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id.replace(':', '__')}_{frame.frame_id}.jpg"
    image = Image.open(frame.path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=(255, 64, 64), width=8)
    image.save(out_path, quality=92)
    return out_path


def render_bbox_focus_crop(frame: FrameInfo, bbox: list[float], out_dir: Path, sample_id: str, expand_ratio: float = 1.3) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id.replace(':', '__')}_{frame.frame_id}_focus.jpg"
    image = Image.open(frame.path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    crop_w = min(float(width), box_w * expand_ratio)
    crop_h = min(float(height), box_h * expand_ratio)
    left = max(0, int(round(cx - crop_w / 2)))
    top = max(0, int(round(cy - crop_h / 2)))
    right = min(width, int(round(cx + crop_w / 2)))
    bottom = min(height, int(round(cy + crop_h / 2)))
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((x1 - left, y1 - top, x2 - left, y2 - top), outline=(255, 64, 64), width=6)
    crop.save(out_path, quality=92)
    return out_path


def choice_text(choice: Any) -> str:
    return " ".join(str(item) for item in choice) if isinstance(choice, list) else str(choice)


def is_localization_family(task_family: str) -> bool:
    return "localization" in task_family


def build_probe_targets(times: list[float], task_family: str) -> list[float]:
    if not times:
        return []
    if task_family == "gaze_gaze_estimation" and len(times) >= 2:
        start, end = times[0], times[-1]
        center = (start + end) / 2
        return [start - 0.5, start, center, end, end + 0.5]
    if task_family in MOTION_FAMILIES:
        center = times[0]
        return [center - 6.0, center - 2.0, center, center + 2.0, center + 6.0]
    if task_family == "ingredient_ingredient_retrieval":
        if len(times) >= 2:
            start, end = times[0], times[-1]
            return [start, start + (end - start) / 3, (start + end) / 2, start + 2 * (end - start) / 3, end]
    if len(times) >= 2:
        start, end = times[0], times[-1]
        return [start, (2 * start + end) / 3, (start + end) / 2, (start + 2 * end) / 3, end]
    center = times[0]
    return [center - 1.5, center, center + 1.5]


def sanitize_source(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", source)


def ensure_probe_frame(
    video_path: Path,
    time_s: float,
    source: str,
    out_dir: Path,
    duration_seconds: float,
) -> FrameInfo:
    safe_end = max(0.0, duration_seconds - 0.25)
    clipped = min(safe_end, max(0.0, time_s))
    frame_id = f"{sanitize_source(source)}_{int(round(clipped * 1000)):09d}"
    path = out_dir / f"{frame_id}_{clipped:09.3f}s.jpg"
    frame = FrameInfo(frame_id=frame_id, time_s=round(clipped, 3), time_hms=format_hms(clipped), path=path, source=source)
    if not path.exists():
        export_frame(video_path, frame)
    if not path.exists():
        raise FileNotFoundError(f"failed to export probe frame at {frame.time_hms}: {path}")
    return frame


def choose_nearest_frames(frames: list[FrameInfo], times: list[float], task_family: str) -> list[FrameInfo]:
    if not times:
        if not frames:
            return []
        if task_family in WHOLE_VIDEO_FAMILIES:
            indices = sorted({0, len(frames) // 5, (2 * len(frames)) // 5, (3 * len(frames)) // 5, (4 * len(frames)) // 5, len(frames) - 1})
            return [frames[index] for index in indices]
        indices = sorted({0, len(frames) // 2, len(frames) - 1})
        return [frames[index] for index in indices]
    targets = build_probe_targets(times, task_family)
    selected: list[FrameInfo] = []
    used: set[str] = set()
    for target in targets:
        for frame in sorted(frames, key=lambda item: abs(item.time_s - target)):
            if frame.frame_id in used:
                continue
            used.add(frame.frame_id)
            selected.append(frame)
            break
    return selected


def observation_to_text(observation: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("scene_location", "visible_ingredients", "visible_tools", "hand_interaction", "ongoing_action", "possible_step", "attention_targets", "state_change_hint"):
        value = observation.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts).lower()


def select_motion_tracking_frames(
    frames: list[FrameInfo],
    frame_lookup: dict[str, dict[str, Any]],
    reference_time: float | None,
    keywords: list[str],
) -> list[FrameInfo]:
    local_targets: list[float] = []
    if reference_time is not None:
        local_targets = [reference_time - 8.0, reference_time - 6.0, reference_time - 4.0, reference_time - 2.0, reference_time, reference_time + 2.0, reference_time + 4.0, reference_time + 6.0, reference_time + 8.0]
    selected: list[FrameInfo] = []
    used: set[str] = set()
    for target in local_targets:
        for frame in sorted(frames, key=lambda item: abs(item.time_s - target)):
            if frame.frame_id in used:
                continue
            used.add(frame.frame_id)
            selected.append(frame)
            break
    keyword_scores: list[tuple[int, float, FrameInfo]] = []
    for frame in frames:
        if frame.frame_id in used:
            continue
        obs = frame_lookup.get(frame.frame_id, {})
        text = observation_to_text(obs)
        score = sum(1 for keyword in keywords if keyword in text)
        if score <= 0:
            continue
        distance = abs(frame.time_s - reference_time) if reference_time is not None else 0.0
        keyword_scores.append((score, -distance, frame))
    for _, _, frame in sorted(keyword_scores, reverse=True)[:4]:
        if frame.frame_id in used:
            continue
        used.add(frame.frame_id)
        selected.append(frame)
    if not selected:
        return choose_nearest_frames(frames, [reference_time] if reference_time is not None else [], "object_motion_object_movement_counting")
    return selected


def build_stationary_option_targets(start_time: float) -> list[float]:
    return [start_time, start_time + 120.0, start_time + 360.0, start_time + 650.0]


def parse_choice_prediction(text: str, choices: list[Any]) -> int | None:
    match = re.search(r"\b([0-4])\b", text.strip())
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(choices):
            return idx
    lowered = text.lower()
    for idx, choice in enumerate(choices):
        choice_text = " ".join(str(item) for item in choice) if isinstance(choice, list) else str(choice)
        if choice_text.lower() == lowered or choice_text.lower() in lowered:
            return idx
    return None


def aggregate_vote_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) == 1:
        return results[0]
    counts = Counter(result.get("prediction") for result in results)
    best_prediction, _ = max(counts.items(), key=lambda item: (item[1], -1 if item[0] is None else item[0]))
    selected = next((result for result in results if result.get("prediction") == best_prediction), results[0])
    merged = dict(selected)
    merged["vote_predictions"] = [result.get("prediction") for result in results]
    merged["vote_raw_outputs"] = [result.get("raw_output") for result in results]
    merged["prediction"] = best_prediction
    merged["correct"] = best_prediction == merged["gold"]
    return merged


def extract_choice_places(choices: list[Any]) -> list[str]:
    places: list[str] = []
    seen: set[str] = set()
    for choice in choices:
        text = choice_text(choice).lower()
        for match in re.finditer(r"(?:from|to)\s+([^,]+?)(?=\s+(?:to|then)\b|$)", text):
            place = match.group(1).strip()
            if place and place not in seen:
                seen.add(place)
                places.append(place)
    return places


def answer_vqa_sample(
    client: OpenAI,
    model: str,
    row: pd.Series,
    video_memory: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
    probe_observation_cache: dict[str, dict[str, Any]],
    probe_observation_path: Path,
    frames: list[FrameInfo],
    video_path: Path,
    duration_seconds: float,
    probe_frames_dir: Path,
    overlay_dir: Path,
    video_id: str,
    bbox_target_cache: dict[str, dict[str, Any]],
    bbox_target_cache_path: Path,
    cache_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    question = str(row["question"])
    task_family = str(row["task_family"])
    sample_id = str(row["vqa_id"])
    choices = json.loads(row["choices_json"])
    times = extract_row_times(row, video_id)
    bbox = extract_bbox(question)
    bbox_profile: dict[str, Any] | None = None
    if bbox and times:
        reference_frame = ensure_probe_frame(
            video_path,
            time_s=times[0],
            source=f"{sample_id}_bbox_reference",
            out_dir=probe_frames_dir,
            duration_seconds=duration_seconds,
        )
        bbox_profile = ensure_bbox_target_profile(
            client,
            model,
            sample_id,
            reference_frame,
            bbox,
            overlay_dir,
            bbox_target_cache,
            bbox_target_cache_path,
            cache_lock,
        )
    chosen = choose_nearest_frames(frames, times, task_family)
    if task_family in WHOLE_VIDEO_TRACKING_FAMILIES:
        reference_time = times[0] if times else None
        keywords = []
        if bbox_profile:
            keywords = [str(item) for item in bbox_profile.get("profile", {}).get("keywords", [])]
        chosen = select_motion_tracking_frames(frames, frame_lookup, reference_time, keywords)
        if reference_time is not None:
            chosen = [
                ensure_probe_frame(
                    video_path,
                    time_s=reference_time,
                    source=f"{sample_id}_reference",
                    out_dir=probe_frames_dir,
                    duration_seconds=duration_seconds,
                )
            ] + chosen
    elif times:
        probe_targets = build_probe_targets(times, task_family)
        chosen = [
            ensure_probe_frame(
                video_path,
                time_s=target,
                source=f"{sample_id}_question_probe",
                out_dir=probe_frames_dir,
                duration_seconds=duration_seconds,
            )
            for target in probe_targets
        ]
    print(
        f"[vqa-start] task={task_family} sample={sample_id} anchors={','.join(format_hms(time_s) for time_s in times) or 'none'} chosen={len(chosen)}",
        flush=True,
    )
    guidance = task_specific_guidance(task_family)
    related_frames = []
    for frame in chosen:
        obs_row = frame_lookup.get(frame.frame_id)
        if obs_row is None:
            obs_row = ensure_frame_observation(
                client,
                model,
                frame,
                [task_family],
                probe_observation_cache,
                probe_observation_path,
                cache_lock,
            )
            if cache_lock is None:
                frame_lookup[frame.frame_id] = obs_row["observation"]
            else:
                with cache_lock:
                    frame_lookup[frame.frame_id] = obs_row["observation"]
        obs = obs_row if isinstance(obs_row, dict) and "observation" not in obs_row else obs_row.get("observation", {})
        related_frames.append(
            {
                "frame_id": frame.frame_id,
                "time_hms": frame.time_hms,
                "observation": {
                    "scene_location": obs.get("scene_location"),
                    "visible_ingredients": obs.get("visible_ingredients"),
                    "visible_tools": obs.get("visible_tools"),
                    "hand_interaction": obs.get("hand_interaction"),
                    "ongoing_action": obs.get("ongoing_action"),
                    "possible_step": obs.get("possible_step"),
                    "attention_targets": obs.get("attention_targets"),
                    "state_change_hint": obs.get("state_change_hint"),
                },
            }
        )
    prompt_text = (
        "你在回答一个厨房第一视角视频的多项选择题。"
        "你已经拿到该视频的全局记忆摘要和与本题最相关的关键帧。"
        "请综合这些证据，只输出最终选项编号0-4，不要解释。"
        f"\n任务提示: {guidance}"
        f"\n\n题型: {task_family}"
        f"\n问题: {question}"
        + "\n选项:\n"
        + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(choices))
        + f"\n\n视频记忆摘要:\n{json.dumps(video_memory, ensure_ascii=False)}"
        + f"\n\n相关关键帧观察:\n{json.dumps(related_frames, ensure_ascii=False)}"
    )
    if task_family == "object_motion_object_movement_itinerary":
        prompt_text += (
            "\n\n候选地点词表:\n"
            + json.dumps(extract_choice_places(choices), ensure_ascii=False)
            + "\n请尽量把每个时刻的目标位置映射到这些候选地点词上，再判断哪条路径最一致。"
        )
    if bbox_profile:
        prompt_text += f"\n\n目标物体参考描述:\n{json.dumps(bbox_profile.get('profile', {}), ensure_ascii=False)}"
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]

    if is_localization_family(task_family):
        localization_prompt = (
            "这是一个时间定位类多项选择题。"
            "你会看到每个选项对应时间段抽取出来的关键帧。"
            "请比较哪个选项与问题描述最匹配，只输出最终选项编号0-4，不要解释。"
            f"\n任务提示: {guidance}"
            f"\n题型: {task_family}\n问题: {question}"
            + "\n选项:\n"
            + "\n".join(f"{idx}. {choice_text(choice)}" for idx, choice in enumerate(choices))
            + f"\n\n视频级记忆摘要:\n{json.dumps(video_memory, ensure_ascii=False)}"
        )
        if bbox_profile:
            localization_prompt += f"\n\n目标物体参考描述:\n{json.dumps(bbox_profile.get('profile', {}), ensure_ascii=False)}"
        localization_content: list[dict[str, Any]] = [{"type": "input_text", "text": localization_prompt}]
        if bbox and times:
            reference_frame = ensure_probe_frame(
                video_path,
                time_s=times[0],
                source=f"{sample_id}_reference_bbox",
                out_dir=probe_frames_dir,
                duration_seconds=duration_seconds,
            )
            reference_overlay = render_bbox_overlay(reference_frame, bbox, overlay_dir, f"{sample_id}_reference")
            reference_focus = render_bbox_focus_crop(reference_frame, bbox, overlay_dir, f"{sample_id}_reference")
            localization_content.append({"type": "input_text", "text": f"参考目标帧 @ {reference_frame.time_hms}，红框标出目标物体"})
            localization_content.append(image_part(reference_overlay))
            localization_content.append({"type": "input_text", "text": "参考目标区域放大图"})
            localization_content.append(image_part(reference_focus))
        for idx, choice in enumerate(choices):
            option_times = [parse_hms(match.group(1)) for match in TIME_PATTERN.finditer(choice_text(choice))]
            if not option_times:
                continue
            if task_family == "object_motion_stationary_object_localization":
                option_targets = build_stationary_option_targets(option_times[0])
            else:
                option_targets = build_probe_targets(option_times, task_family)
            option_frames = [
                ensure_probe_frame(
                    video_path,
                    time_s=target,
                    source=f"{sample_id}_choice_{idx}",
                    out_dir=probe_frames_dir,
                    duration_seconds=duration_seconds,
                )
                for target in option_targets
            ]
            if task_family == "object_motion_stationary_object_localization":
                localization_content.append({"type": "input_text", "text": f"选项 {idx} 的验证帧：起点 {choice_text(choice)}，之后检查该物体在 120s/360s/650s 后是否仍在同一位置"})
            else:
                localization_content.append({"type": "input_text", "text": f"选项 {idx} 的关键帧：{choice_text(choice)}"})
            for option_frame in option_frames:
                localization_content.append({"type": "input_text", "text": f"选项 {idx} 帧 @ {option_frame.time_hms}"})
                localization_content.append(image_part(option_frame.path))
                if bbox and task_family not in MOTION_FAMILIES:
                    overlay = render_bbox_overlay(option_frame, bbox, overlay_dir, f"{sample_id}_choice_{idx}")
                    localization_content.append({"type": "input_text", "text": f"选项 {idx} 带框帧"})
                    localization_content.append(image_part(overlay))
        response = create_response(client, model, localization_content, temperature=0)
        text = getattr(response, "output_text", None) or ""
        pred = parse_choice_prediction(text, choices)
        gold = int(row["correct_idx"])
        return {
            "sample_id": sample_id,
            "task_family": task_family,
            "question": question,
            "choices": choices,
            "gold": gold,
            "prediction": pred,
            "correct": pred == gold,
            "raw_output": text,
            "used_frames": [frame.frame_id for frame in chosen],
            "used_frame_times": [frame.time_hms for frame in chosen],
            "overlay_paths": [],
        }

    for frame in chosen:
        content.append({"type": "input_text", "text": f"原始帧 {frame.frame_id} @ {frame.time_hms}"})
        content.append(image_part(frame.path))
    overlay_paths: list[str] = []
    if bbox:
        overlay_frames = chosen
        if task_family in MOTION_FAMILIES and times:
            reference_time = times[0]
            overlay_frames = [min(chosen, key=lambda item: abs(item.time_s - reference_time))]
            content.append({"type": "input_text", "text": "注意：边框只对参考时刻有效，其他时刻请通过原始帧追踪该物体。"})
        for frame in overlay_frames:
            overlay = render_bbox_overlay(frame, bbox, overlay_dir, sample_id)
            overlay_paths.append(overlay.as_posix())
            content.append({"type": "input_text", "text": f"带框帧 {frame.frame_id}，目标区域已标注"}) 
            content.append(image_part(overlay))
    response = create_response(client, model, content, temperature=0)
    text = getattr(response, "output_text", None) or ""
    pred = parse_choice_prediction(text, choices)
    gold = int(row["correct_idx"])
    return {
        "sample_id": sample_id,
        "task_family": task_family,
        "question": question,
        "choices": choices,
        "gold": gold,
        "prediction": pred,
        "correct": pred == gold,
        "raw_output": text,
        "used_frames": [frame.frame_id for frame in chosen],
        "used_frame_times": [frame.time_hms for frame in chosen],
        "overlay_paths": overlay_paths,
    }


def markdown_report(
    video_id: str,
    video_meta: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    video_memory: dict[str, Any],
    vqa_results: list[dict[str, Any]],
) -> str:
    correct = sum(int(row["correct"]) for row in vqa_results if row["prediction"] is not None)
    valid = sum(int(row["prediction"] is not None) for row in vqa_results)
    lines = [
        f"# Single Video Agent Probe: {video_id}",
        "",
        "## Video",
        f"- duration_seconds: `{video_meta['duration_seconds']:.3f}`",
        f"- fps: `{video_meta['fps']:.3f}`",
        f"- resolution: `{video_meta['width']}x{video_meta['height']}`",
        f"- keyframes: `{len(frame_rows)}`",
        f"- sampled_vqa: `{len(vqa_results)}`",
        f"- valid_predictions: `{valid}`",
        f"- correct: `{correct}`",
        f"- accuracy: `{(correct / valid) if valid else 0.0:.4f}`",
        "",
        "## Video Memory",
        "```json",
        json.dumps(video_memory, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Frame Observations",
    ]
    for row in frame_rows:
        lines.append(
            f"- `{row['frame_id']}` `{row['time_hms']}` source=`{row['source']}` observation=`{json.dumps(row['observation'], ensure_ascii=False)}`"
        )
    lines.extend(["", "## Sampled VQA Results"])
    for row in vqa_results:
        lines.append(
            f"- `{row['task_family']}` sample=`{row['sample_id']}` pred=`{row['prediction']}` gold=`{row['gold']}` correct=`{row['correct']}` frames={row['used_frames']}"
        )
        lines.append(f"  question: {row['question']}")
    lines.append("")
    return "\n".join(lines)


def run_sample_once(
    row: dict[str, Any],
    cfg: ModelConfig,
    video_memory: dict[str, Any],
    frame_lookup: dict[str, dict[str, Any]],
    probe_observations: dict[str, dict[str, Any]],
    probe_obs_path: Path,
    frames: list[FrameInfo],
    video_path: Path,
    duration_seconds: float,
    probe_frames_dir: Path,
    overlays_dir: Path,
    video_id: str,
    bbox_target_profiles: dict[str, dict[str, Any]],
    bbox_target_path: Path,
    cache_lock: threading.Lock,
) -> dict[str, Any]:
    row_series = pd.Series(row)
    task_family = str(row["task_family"])
    client = get_thread_client(cfg)
    vote_attempts = 3 if task_family in VOLATILE_TASK_FAMILIES else 1
    vote_results: list[dict[str, Any]] = []
    prediction_counts: Counter[int | None] = Counter()
    for _ in range(vote_attempts):
        result = answer_vqa_sample(
            client,
            cfg.model,
            row_series,
            video_memory,
            frame_lookup,
            probe_observations,
            probe_obs_path,
            frames,
            video_path,
            duration_seconds,
            probe_frames_dir,
            overlays_dir,
            video_id,
            bbox_target_profiles,
            bbox_target_path,
            cache_lock,
        )
        vote_results.append(result)
        prediction_counts[result.get("prediction")] += 1
        if prediction_counts[result.get("prediction")] >= 2:
            break
    return aggregate_vote_results(vote_results)


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["FOOD_AGENT_PROVIDER_MODE"] = "responses"
    cfg = ModelConfig.from_env()
    client = get_thread_client(cfg)

    vqa_df = pd.read_parquet(args.index_dir / "vqa_samples.parquet")
    video_df = vqa_df[vqa_df["primary_video_id"] == args.video_id].copy()
    if video_df.empty:
        raise ValueError(f"no VQA rows for video_id={args.video_id}")

    out_dir = args.out_dir / args.video_id
    frames_dir = out_dir / "frames"
    overlays_dir = out_dir / "bbox_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_video_path(args.data_root, args.video_id)
    audio_path = find_audio_path(args.data_root, args.video_id)
    video_meta = probe_video(video_path)

    question_anchors = extract_question_anchors(video_df, args.question_anchor_limit)
    audio_anchors = extract_audio_peak_anchors(
        audio_path,
        args.video_id,
        video_meta["duration_seconds"],
        [anchor.time_s for anchor in question_anchors],
        args.audio_peak_limit,
        args.audio_window_seconds,
        args.audio_min_peak_gap,
        args.audio_exclusion_radius,
    )
    anchors = sorted(question_anchors + audio_anchors, key=lambda item: item.time_s)
    fill_times = add_gap_fill_times([anchor.time_s for anchor in anchors], video_meta["duration_seconds"], args.gap_fill_seconds)
    offsets = [float(item) for item in args.frame_offsets.split(",") if item.strip()]
    frames = build_frame_list(anchors, fill_times, offsets, video_meta["duration_seconds"], args.min_frame_gap, frames_dir)

    for frame in frames:
        if args.resume and frame.path.exists():
            continue
        export_frame(video_path, frame)

    anchor_map: dict[str, set[str]] = {}
    for anchor in anchors:
        for offset in offsets:
            key = f"{round(min(video_meta['duration_seconds'], max(0.0, anchor.time_s + offset)), 3):.3f}"
            anchor_map.setdefault(key, set()).update(anchor.related_tasks)

    frame_obs_path = out_dir / "frame_observations.json"
    existing_observations: dict[str, dict[str, Any]] = {}
    if args.resume and frame_obs_path.exists():
        for row in json.loads(frame_obs_path.read_text(encoding="utf-8")):
            existing_observations[row["frame_id"]] = row

    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        if frame.frame_id in existing_observations:
            frame_rows.append(existing_observations[frame.frame_id])
            continue
        related_tasks = sorted(anchor_map.get(f"{frame.time_s:.3f}", []))
        observation = describe_frame(client, cfg.model, frame, related_tasks)
        row = {
            "frame_id": frame.frame_id,
            "time_s": frame.time_s,
            "time_hms": frame.time_hms,
            "path": frame.path.as_posix(),
            "source": frame.source,
            "observation": observation,
        }
        frame_rows.append(row)
        frame_obs_path.write_text(json.dumps(frame_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[frame] {frame.frame_id} {frame.time_hms} source={frame.source}", flush=True)

    probe_obs_path = out_dir / "probe_frame_observations.json"
    probe_observations: dict[str, dict[str, Any]] = {}
    if args.resume and probe_obs_path.exists():
        for row in json.loads(probe_obs_path.read_text(encoding="utf-8")):
            probe_observations[row["frame_id"]] = row

    bbox_target_path = out_dir / "bbox_target_profiles.json"
    bbox_target_profiles: dict[str, dict[str, Any]] = {}
    if args.resume and bbox_target_path.exists():
        for row in json.loads(bbox_target_path.read_text(encoding="utf-8")):
            bbox_target_profiles[row["sample_id"]] = row

    video_memory_path = out_dir / "video_memory.json"
    if args.resume and video_memory_path.exists():
        video_memory = json.loads(video_memory_path.read_text(encoding="utf-8"))
    else:
        video_memory = summarize_video_memory(client, cfg.model, args.video_id, frame_rows)
        video_memory_path.write_text(json.dumps(video_memory, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.all_vqa:
        sorted_video_df = video_df.sort_values(["task_family", "vqa_id"]).reset_index(drop=True)
        if args.start_index > 0:
            sorted_video_df = sorted_video_df.iloc[args.start_index :].copy()
        if args.cap_per_family > 0:
            capped_rows: list[dict[str, Any]] = []
            for _, group in sorted_video_df.groupby("task_family", sort=True):
                capped_rows.extend(group.head(args.cap_per_family).to_dict("records"))
            sampled = capped_rows
        else:
            sampled = sorted_video_df.to_dict("records")
        if args.max_vqa > 0:
            sampled = sampled[: args.max_vqa]

        name_parts: list[str] = []
        if args.start_index > 0:
            name_parts.append(f"start_{args.start_index}")
        if args.cap_per_family > 0:
            name_parts.append(f"cap_{args.cap_per_family}")
        if args.max_vqa > 0:
            name_parts.append(f"subset_{len(sampled)}")
        if not name_parts:
            results_path = out_dir / "full_vqa_results.json"
            summary_path = out_dir / "full_summary.json"
            report_path = out_dir / "full_report.md"
        else:
            suffix = "_".join(name_parts)
            results_path = out_dir / f"{suffix}_vqa_results.json"
            summary_path = out_dir / f"{suffix}_summary.json"
            report_path = out_dir / f"{suffix}_report.md"
    else:
        sampled = []
        for _, group in video_df.groupby("task_family", sort=True):
            sampled.extend(group.head(args.sample_vqa_per_family).to_dict("records"))
        results_path = out_dir / "sampled_vqa_results.json"
        summary_path = out_dir / "summary.json"
        report_path = out_dir / "report.md"
    existing_results: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for row in json.loads(results_path.read_text(encoding="utf-8")):
            existing_results[row["sample_id"]] = row
    frame_lookup = {row["frame_id"]: row["observation"] for row in frame_rows}
    results: list[dict[str, Any]] = list(existing_results.values())
    pending = [row for row in sampled if str(row["vqa_id"]) not in existing_results]
    total_target = len(sampled)
    completed_count = len(results)
    cache_lock = threading.Lock()
    if pending:
        print(
            f"[run] mode={'all_vqa' if args.all_vqa else 'sampled_vqa'} total_target={total_target} completed_resume={completed_count} pending={len(pending)} workers={args.workers}",
            flush=True,
        )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                run_sample_once,
                row,
                cfg,
                video_memory,
                frame_lookup,
                probe_observations,
                probe_obs_path,
                frames,
                video_path,
                video_meta["duration_seconds"],
                out_dir / "probe_frames",
                overlays_dir,
                args.video_id,
                bbox_target_profiles,
                bbox_target_path,
                cache_lock,
            ): row
            for row in pending
        }
        for future in as_completed(future_map):
            row = future_map[future]
            sample_id = str(row["vqa_id"])
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[task-error] sample={sample_id} task={row['task_family']} error={exc}", flush=True)
                raise
            results.append(result)
            completed_count += 1
            correct_so_far = sum(int(item["correct"]) for item in results if item["prediction"] is not None)
            valid_so_far = sum(int(item["prediction"] is not None) for item in results)
            accuracy_so_far = (correct_so_far / valid_so_far) if valid_so_far else 0.0
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[vqa] {result['task_family']} sample={sample_id} pred={result['prediction']} gold={result['gold']} correct={result['correct']} votes={result.get('vote_predictions', [result['prediction']])}",
                flush=True,
            )
            print(
                f"[progress] completed={completed_count}/{total_target} valid={valid_so_far} correct={correct_so_far} accuracy={accuracy_so_far:.4f}",
                flush=True,
            )

    summary = {
        "video_id": args.video_id,
        "video_path": video_path.as_posix(),
        "audio_path": audio_path.as_posix(),
        "frame_count": len(frame_rows),
        "sampled_vqa_count": len(results),
        "correct": sum(int(row["correct"]) for row in results if row["prediction"] is not None),
        "valid_prediction_count": sum(int(row["prediction"] is not None) for row in results),
    }
    if summary["valid_prediction_count"]:
        summary["accuracy"] = summary["correct"] / summary["valid_prediction_count"]
    else:
        summary["accuracy"] = None
    if args.all_vqa and (args.max_vqa > 0 or args.start_index > 0 or args.cap_per_family > 0):
        summary["mode"] = "subset_vqa"
        summary["requested_vqa_count"] = len(sampled)
        summary["start_index"] = args.start_index
        summary["cap_per_family"] = args.cap_per_family
    else:
        summary["mode"] = "all_vqa" if args.all_vqa else "sampled_vqa"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        markdown_report(args.video_id, video_meta, frame_rows, video_memory, results),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
