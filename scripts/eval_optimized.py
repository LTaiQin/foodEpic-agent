#!/usr/bin/env python3
"""Optimized hybrid evaluation: best method per category.

Based on analysis of all experiments:
- Tools: ingredient_weight, ingredients_order, nutrition_change, nutrition_video, recipe_step_recognition
- Direct LLM: all other categories

Usage:
    python scripts/eval_optimized.py --limit 7 --parallel 4 --out outputs/results/exp_optimized.json
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_server import send_request, SOCKET_PATH

BENCHMARK_DIR = Path("/22liushoulong/agent/hd-epic/annotations/hd-epic-annotations-main/vqa-benchmark")

# Categories where tools work better (from analysis)
TOOL_CATEGORIES = {
    "ingredient_ingredient_weight",
    "ingredient_ingredients_order",
    "nutrition_nutrition_change",
    "nutrition_video_nutrition_estimation",
    "recipe_step_recognition",
    "3d_perception_object_location",
    "fine_grained_why_recognition",
}


def load_benchmark(category: str = None) -> list:
    questions = []
    files = sorted(BENCHMARK_DIR.glob("*.json"))
    for f in files:
        cat_name = f.stem
        if category and cat_name != category:
            continue
        with open(f) as fh:
            data = json.load(fh)
        for key, q in data.items():
            video_info = q.get("inputs", {}).get("video 1", {})
            video_id = video_info.get("id", "")
            choices = q.get("choices", [])
            choice_strings = []
            for c in choices:
                if isinstance(c, list):
                    choice_strings.append(", ".join(str(item) for item in c))
                else:
                    choice_strings.append(str(c))
            questions.append({
                "id": key, "category": cat_name, "question": q["question"],
                "choices": choice_strings, "correct_idx": q.get("correct_idx", -1),
                "video_id": video_id,
            })
    return questions


def answer_direct(q: dict) -> dict:
    """Answer using direct LLM with optimized prompts."""
    from food_agent.loaders import VideoLoader
    from food_agent.evaluation.api_client import MimoClient
    from pathlib import Path

    vl = VideoLoader(Path("data/HD-EPIC/Videos"))
    mimo = MimoClient()

    video_id = q["video_id"]
    category = q.get("category", "")

    # Extract timestamp
    time_match = re.search(r'TIME\s+(\d+):(\d+):(\d+\.?\d*)', q["question"])
    if time_match:
        ts = float(time_match.group(1)) * 3600 + float(time_match.group(2)) * 60 + float(time_match.group(3))
    else:
        ts = 30

    try:
        frame = vl.get_frame(video_id, ts)
    except Exception:
        try:
            frame = vl.get_frame(video_id, 10)
        except Exception:
            return {"answer": "Error: could not load frame", "confidence": 0, "tool_calls": 0}

    # Build optimized prompt
    prompt = q["question"]
    if q["choices"]:
        choice_text = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(q["choices"]))
        prompt += f"\n\n{choice_text}\n\n"

        # Category-specific reasoning
        if "action" in category:
            prompt += "Analyze the person's movements, hand positions, and interactions. "
        elif "recipe" in category:
            prompt += "Consider the cooking context and recipe steps. "
        elif "gaze" in category:
            prompt += "Determine where the person is looking based on head direction. "
        elif "ingredient" in category:
            prompt += "Identify food items and their properties. "
        elif "nutrition" in category:
            prompt += "Consider nutritional content of food items. "
        elif "3d_perception" in category or "object" in category:
            prompt += "Analyze spatial layout and object positions. "
        elif "motion" in category:
            prompt += "Observe object movement and position changes. "

        prompt += "Select the BEST option. Reply with ONLY one letter (A, B, C, D, or E)."

    try:
        response = mimo.call_vision(frame, prompt)
        if isinstance(response, list):
            response = response[0] if response else ""
        response = str(response).strip()
    except Exception as e:
        response = f"Error: {e}"

    return {"answer": response, "confidence": 0.7, "tool_calls": 0}


def answer_with_tools(q: dict) -> dict:
    """Answer using agent with tools."""
    prompt = q["question"]
    if q["choices"]:
        choice_text = "\n".join(f"  {chr(65+j)}. {c}" for j, c in enumerate(q["choices"]))
        prompt += f"\n\n{choice_text}\n\nSelect the best option. Reply with ONLY the letter (A, B, C, D, or E), nothing else."

    try:
        result = send_request({
            "action": "answer",
            "question": prompt,
            "video_id": q["video_id"],
            "choices": q["choices"],
        })
        return result
    except Exception as e:
        return {"answer": f"Error: {e}", "confidence": 0, "tool_calls": 0}


def answer_question(q: dict) -> dict:
    """Answer using the best method for this category."""
    category = q["category"]

    if category in TOOL_CATEGORIES:
        result = answer_with_tools(q)
    else:
        result = answer_direct(q)

    pred_answer = result.get("answer", "")
    if isinstance(pred_answer, list):
        pred_answer = pred_answer[0] if pred_answer else ""
    pred_answer = str(pred_answer).strip()
    pred_idx = -1

    if q["choices"]:
        # Letter match
        if pred_answer:
            letter_match = re.match(r'^([A-E])', pred_answer.upper().strip())
            if letter_match:
                pred_idx = ord(letter_match.group(1)) - ord("A")
                if pred_idx >= len(q["choices"]):
                    pred_idx = -1

            # "option X" pattern
            if pred_idx == -1:
                option_match = re.search(r'(?:option|answer|choice)\s*(?:is)?\s*([A-E])', pred_answer, re.IGNORECASE)
                if option_match:
                    pred_idx = ord(option_match.group(1).upper()) - ord("A")
                    if pred_idx >= len(q["choices"]):
                        pred_idx = -1

        # Text match fallback
        if pred_idx == -1:
            for j, choice_str in enumerate(q["choices"]):
                if pred_answer == choice_str or pred_answer.lower() == choice_str.lower():
                    pred_idx = j
                    break
            if pred_idx == -1:
                for j, choice_str in enumerate(q["choices"]):
                    if choice_str[:30].lower() in pred_answer.lower() or pred_answer[:30].lower() in choice_str.lower():
                        pred_idx = j
                        break

    return {
        "id": q["id"], "category": q["category"], "question": q["question"][:200],
        "video_id": q["video_id"], "prediction": pred_answer[:200], "pred_idx": pred_idx,
        "correct_idx": q["correct_idx"], "is_correct": pred_idx == q["correct_idx"],
        "confidence": result.get("confidence", 0),
        "tool_calls": result.get("tool_calls", 0) if isinstance(result.get("tool_calls"), int) else len(result.get("tool_calls", [])),
        "method": "tools" if q["category"] in TOOL_CATEGORIES else "direct",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    questions = load_benchmark(args.category)
    print(f"Loaded {len(questions)} questions", flush=True)

    if args.limit > 0:
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        sampled = []
        for cat, qs in by_cat.items():
            sampled.extend(qs[:args.limit])
        questions = sampled
        print(f"Sampled {len(questions)} questions ({args.limit} per category)", flush=True)

    results = []
    correct = 0
    total = 0

    existing_ids = set()
    if args.resume and args.out and Path(args.out).exists():
        try:
            with open(args.out) as f:
                existing = json.load(f)
            results = existing.get("results", [])
            correct = existing.get("correct", 0)
            total = existing.get("total", 0)
            existing_ids = {r["id"] for r in results}
            print(f"Resuming: {len(results)} done, {correct}/{total}", flush=True)
        except Exception:
            pass

    questions = [q for q in questions if q["id"] not in existing_ids]
    print(f"Remaining: {len(questions)}\n", flush=True)

    print(f"{'#':>4} | {'Category':>30} | {'Method':>6} | {'Pred':>5} | {'GT':>5} | {'OK':>3} | {'Time':>5}", flush=True)
    print("-" * 75, flush=True)

    start_time = time.time()
    total_questions = len(questions)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_q = {executor.submit(answer_question, q): q for q in questions}
        for future in as_completed(future_to_q):
            result = future.result()
            results.append(result)
            total += 1
            if result["is_correct"]:
                correct += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / total
            remaining = (total_questions - total) * avg_time

            status = "OK" if result["is_correct"] else "MISS"
            method = result.get("method", "?")
            progress = f"[{'█' * (total * 20 // total_questions)}{' ' * (20 - total * 20 // total_questions)}]"
            print(f"{total:>4} | {result['category'][:30]:>30} | {method:>6} | {result['pred_idx']:>5} | {result['correct_idx']:>5} | {status:>3} | {elapsed:>5.0f}s | {progress} {correct}/{total}={correct/total:.0%}", flush=True)

            if args.out:
                _save(args.out, results, correct, total)

    elapsed = time.time() - start_time
    acc = correct / total if total > 0 else 0
    print(f"\n{'='*75}", flush=True)
    print(f"Result: {correct}/{total} = {acc:.1%}", flush=True)
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.0f}s per question)", flush=True)

    # Per-method stats
    tool_results = [r for r in results if r.get("method") == "tools"]
    direct_results = [r for r in results if r.get("method") == "direct"]
    tool_correct = sum(1 for r in tool_results if r["is_correct"])
    direct_correct = sum(1 for r in direct_results if r["is_correct"])
    print(f"\nTools: {tool_correct}/{len(tool_results)} = {tool_correct/max(len(tool_results),1):.1%}", flush=True)
    print(f"Direct: {direct_correct}/{len(direct_results)} = {direct_correct/max(len(direct_results),1):.1%}", flush=True)

    # Per-category
    print(f"\nPer-category:", flush=True)
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_results[r["category"]]["total"] += 1
        if r["is_correct"]:
            cat_results[r["category"]]["correct"] += 1
    for cat in sorted(cat_results, key=lambda x: cat_results[x]["correct"]/max(cat_results[x]["total"],1), reverse=True):
        c = cat_results[cat]
        pct = c["correct"]/max(c["total"],1)*100
        print(f"  {cat:50s} {c['correct']}/{c['total']} ({pct:.0f}%)", flush=True)

    if args.out:
        _save(args.out, results, correct, total)
        print(f"\nSaved to {args.out}", flush=True)


def _save(output_file, results, correct, total):
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_results[r["category"]]["total"] += 1
        if r["is_correct"]:
            cat_results[r["category"]]["correct"] += 1
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "total": total, "correct": correct,
            "accuracy": correct / total if total else 0,
            "per_category": {k: {"correct": v["correct"], "total": v["total"]} for k, v in cat_results.items()},
            "results": results,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
