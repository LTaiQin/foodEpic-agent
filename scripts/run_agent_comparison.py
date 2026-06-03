#!/usr/bin/env python3
"""Run baseline comparisons with the configured OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.advantage_metrics import food_agent_advantage_score, judge_advantage
from food_agent.comparison import build_messages, collect_evidence, parse_model_output
from food_agent.config import load_env_file
from food_agent.model_client import OpenAICompatibleModelClient
from food_agent.paths import ProjectPaths
from food_agent.spatial_store import SpatialContextStore
from food_agent.state_store import FoodStateStore
from food_agent.vqa import VQAPrediction, compute_metrics, load_vqa_samples


TASK_FAMILY_GROUPS = {
    "food-core": [
        "ingredient_ingredient_retrieval",
        "ingredient_exact_ingredient_recognition",
        "ingredient_ingredient_recognition",
        "recipe_step_recognition",
        "recipe_recipe_recognition",
        "recipe_following_activity_recognition",
        "nutrition_nutrition_change",
    ],
    "food-state": [
        "ingredient_ingredient_retrieval",
        "ingredient_ingredients_order",
        "ingredient_ingredient_weight",
        "recipe_step_recognition",
        "recipe_multi_step_localization",
        "nutrition_nutrition_change",
    ],
}


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--index-dir", type=Path, default=defaults.output_root / "event_index")
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "results" / "agent_comparison")
    parser.add_argument("--baselines", default="textonly,directevidence,foodstate,ours-foodevidence")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--task-family", default=None)
    parser.add_argument("--task-family-group", choices=sorted(TASK_FAMILY_GROUPS), default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_one_baseline(
    baseline: str,
    samples,
    model_client: OpenAICompatibleModelClient,
    state_store: FoodStateStore,
    spatial_store: SpatialContextStore,
    temperature: float,
) -> list[VQAPrediction]:
    predictions: list[VQAPrediction] = []
    for sample in samples:
        evidence = collect_evidence(sample, state_store, spatial_store)
        messages = build_messages(sample, baseline, evidence)
        response = model_client.complete(messages, temperature=temperature)
        pred_idx, answer_evidence_ids, parse_failure = parse_model_output(response.content, sample, baseline)
        evidence_ids = answer_evidence_ids or list(evidence.get("evidence_ids", [] if baseline == "textonly" else evidence.get("evidence_ids", [])))
        tool_calls = []
        if baseline in {"directevidence", "foodstate", "ours-foodevidence"}:
            tool_calls = ["state_store", "spatial_store"]
        predictions.append(
            VQAPrediction(
                sample_id=sample.vqa_id,
                baseline=baseline,
                task_family=sample.task_family,
                video_id=sample.primary_video_id,
                question=sample.question,
                choices=sample.choices,
                gold=sample.correct_idx,
                prediction=pred_idx,
                correct=pred_idx == sample.correct_idx,
                evidence_ids=evidence_ids if baseline != "textonly" else [],
                tool_calls=tool_calls,
                failure_type=parse_failure if parse_failure else (None if pred_idx == sample.correct_idx else "reasoning_error"),
            )
        )
    return predictions


def write_jsonl(path: Path, predictions: list[VQAPrediction]) -> None:
    path.write_text("\n".join(pred.to_json() for pred in predictions) + "\n", encoding="utf-8")


def load_selected_samples(index_dir: Path, limit: int | None, task_family: str | None, task_family_group: str | None):
    if task_family_group:
        samples = []
        per_family_limit = limit
        for family in TASK_FAMILY_GROUPS[task_family_group]:
            family_samples = load_vqa_samples(index_dir, limit=per_family_limit, task_family=family)
            samples.extend(family_samples)
        return samples
    return load_vqa_samples(index_dir, limit=limit, task_family=task_family)


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    model_client = OpenAICompatibleModelClient()
    state_store = FoodStateStore(args.index_dir)
    spatial_store = SpatialContextStore(args.index_dir)
    samples = load_selected_samples(args.index_dir, limit=args.limit, task_family=args.task_family, task_family_group=args.task_family_group)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    run_name = args.task_family_group or args.task_family or "mixed"
    run_out_dir = args.out_dir / run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    food_metrics = read_json(Path("outputs/results/food_state_metrics.json"))
    spatial_metrics = read_json(Path("outputs/results/spatial_context_metrics.json"))

    summary: dict[str, dict] = {}
    for baseline in baselines:
        print(f"[comparison] running baseline={baseline} samples={len(samples)}", flush=True)
        predictions = run_one_baseline(
            baseline,
            samples,
            model_client,
            state_store,
            spatial_store,
            temperature=args.temperature,
        )
        pred_path = run_out_dir / f"predictions_{baseline}.jsonl"
        metrics_path = run_out_dir / f"metrics_{baseline}.json"
        advantage_path = run_out_dir / f"advantage_{baseline}.json"
        write_jsonl(pred_path, predictions)
        metrics = compute_metrics(predictions)
        advantage = food_agent_advantage_score(predictions, food_metrics, spatial_metrics)
        verdict = judge_advantage(advantage)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        advantage_path.write_text(
            json.dumps({"score": advantage, "verdict": verdict}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[baseline] = {
            "predictions": pred_path.as_posix(),
            "metrics": metrics,
            "advantage": advantage,
            "verdict": verdict,
        }
    wrapped_summary = {
        "run_name": run_name,
        "task_family": args.task_family,
        "task_family_group": args.task_family_group,
        "sample_count": len(samples),
        "baselines": summary,
    }
    summary_path = run_out_dir / "summary.json"
    summary_path.write_text(json.dumps(wrapped_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(wrapped_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
