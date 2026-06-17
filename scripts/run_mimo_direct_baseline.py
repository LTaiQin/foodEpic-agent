#!/usr/bin/env python3
"""Run MiMo direct-answer baseline with video input (no agent loop, no tool use)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

import pandas as pd

from food_agent.config import load_env_file
from food_agent.model_client import OpenAICompatibleModelClient
from food_agent.paths import ProjectPaths
from food_agent.tools.video_tools import VideoToolbox


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "mimo_direct_baseline")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--selection-file", type=Path, default=None, help="Reuse selection from agent eval")
    parser.add_argument("--text-only", action="store_true", help="Text-only baseline (no video)")
    return parser.parse_args()


TEXT_PROMPT = """你是一个视频问答系统。请根据以下信息选择最合适的答案。

问题: {question}

选项:
{choices_text}

请直接选择最合适的答案编号。输出 JSON 格式:
{{"best_index": <0-{max_index}>, "answer": "<选项文本>", "confidence": <0.0-1.0>, "reason": "<简短理由>"}}
"""

VIDEO_PROMPT = """你在看一段厨房操作视频。请根据视频内容回答问题。

问题: {question}

选项:
{choices_text}

请仔细观察视频中的动作、物体和状态变化，选择最合适的答案。
输出 JSON 格式:
{{"best_index": <0-{max_index}>, "answer": "<选项文本>", "confidence": <0.0-1.0>, "reason": "<简短理由>"}}
"""


def parse_time(time_str: str) -> float:
    parts = str(time_str).split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def extract_clip_for_sample(row: dict, video_toolbox: VideoToolbox, paths: ProjectPaths) -> Path | None:
    inputs_json = str(row.get("inputs_json") or "{}")
    try:
        inputs = json.loads(inputs_json)
    except json.JSONDecodeError:
        return None
    for key, info in inputs.items():
        if not isinstance(info, dict):
            continue
        video_id = str(info.get("id") or row.get("primary_video_id") or "")
        start_time = parse_time(str(info.get("start_time") or "0"))
        end_time = parse_time(str(info.get("end_time") or "0"))
        if not video_id or end_time <= start_time:
            continue
        video_dir = paths.data_root / "Videos"
        video_path = None
        for subdir in video_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.glob(f"{video_id}.mp4"):
                    video_path = f
                    break
        if video_path is None or not video_path.exists():
            continue
        clip_name = f"baseline_clip_{video_id}_{start_time:.1f}_{end_time:.1f}.mp4"
        try:
            clip_path = video_toolbox.extract_video_clip(
                video_path=video_path,
                start_time=start_time - 2.0,
                end_time=end_time + 3.0,
                output_name=clip_name,
                max_duration_s=10.0,
                compress=True,
            )
            return clip_path
        except Exception:
            return None
    return None


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    config_path = ProjectPaths.from_env().project_root / ".env"
    if config_path.exists():
        from food_agent.config import load_env_file as load2
        load2(config_path)
    client = OpenAICompatibleModelClient()
    paths = ProjectPaths.from_env()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    video_toolbox = VideoToolbox(out_dir / "clips")

    if args.selection_file and args.selection_file.exists():
        rows = json.loads(args.selection_file.read_text(encoding="utf-8"))
    else:
        df = pd.read_parquet(args.index_file)
        subset = df[df["task_family"] == "fine_grained_why_recognition"].copy()
        subset = subset.sort_values(["primary_video_id", "vqa_id"]).head(args.limit)
        rows = subset.to_dict("records")

    results: list[dict] = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("vqa_id") or f"sample_{index}")
        question = str(row.get("question") or "")
        choices = json.loads(row.get("choices_json") or "[]")
        gold = int(row.get("correct_idx", 0))
        video_id = str(row.get("primary_video_id") or "")

        choices_text = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))

        use_video = not args.text_only
        clip_path = None
        if use_video:
            clip_path = extract_clip_for_sample(row, video_toolbox, paths)
            if clip_path is None:
                use_video = False

        prompt = (VIDEO_PROMPT if use_video else TEXT_PROMPT).format(
            question=question,
            choices_text=choices_text,
            max_index=len(choices) - 1,
        )

        started = time.time()
        try:
            if use_video and clip_path:
                response = client.inspect_videos(
                    prompt=prompt,
                    video_paths=[clip_path],
                    temperature=0.0,
                )
                text = response.content.strip()
                payload = client._extract_json_object(text)
            else:
                response = client.complete_json([{"role": "user", "content": prompt}], temperature=0.0)
                payload = response
            prediction = int(payload.get("best_index", 0))
            answer = str(payload.get("answer", ""))
            confidence = float(payload.get("confidence", 0.0))
            reason = str(payload.get("reason", ""))
        except Exception as exc:
            prediction = None
            answer = ""
            confidence = 0.0
            reason = f"error: {exc}"

        elapsed = time.time() - started
        correct = prediction == gold if prediction is not None else False

        result = {
            "vqa_id": sample_id,
            "video_id": video_id,
            "question": question,
            "choices_json": row.get("choices_json"),
            "prediction": prediction,
            "gold": gold,
            "correct": correct,
            "answer_text": answer,
            "confidence": confidence,
            "reason": reason,
            "elapsed_seconds": elapsed,
            "method": "video" if (use_video and clip_path) else "text_only",
        }
        results.append(result)

        print(
            f"[{index}/{len(rows)}] sample={sample_id} pred={prediction} gold={gold} "
            f"correct={correct} conf={confidence:.2f} elapsed={elapsed:.1f}s "
            f"method={'video' if (use_video and clip_path) else 'text'}",
            flush=True,
        )

    with open(predictions_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    elapsed_list = [r["elapsed_seconds"] for r in results if r["elapsed_seconds"]]
    usage = client.usage_snapshot()
    video_count = sum(1 for r in results if r.get("method") == "video")

    summary = {
        "task_family": "fine_grained_why_recognition",
        "method": "mimo_direct_with_video" if not args.text_only else "mimo_direct_text_only",
        "selection_count": total,
        "correct_count": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "avg_elapsed_seconds": sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0,
        "video_input_count": video_count,
        "text_only_count": total - video_count,
        "total_prompt_tokens": usage.get("prompt_tokens", 0),
        "total_completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "per_sample": [
            {
                "vqa_id": r["vqa_id"],
                "prediction": r["prediction"],
                "gold": r["gold"],
                "correct": r["correct"],
                "confidence": r["confidence"],
                "elapsed_seconds": r["elapsed_seconds"],
                "method": r.get("method", "unknown"),
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
