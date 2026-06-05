#!/usr/bin/env python3
"""Run baseline comparisons with the configured OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

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
    "multimodal-core": [
        "gaze_gaze_estimation",
        "gaze_interaction_anticipation",
        "3d_perception_object_location",
        "3d_perception_object_contents_retrieval",
        "object_motion_object_movement_counting",
        "object_motion_stationary_object_localization",
    ],
}


class PersistentModelError(RuntimeError):
    """Raised when a sample still hits model errors after the configured retry budget."""

    def __init__(self, prediction: VQAPrediction):
        super().__init__(
            f"persistent model error for sample={prediction.sample_id} baseline={prediction.baseline} after attempt_count={prediction.attempt_count}"
        )
        self.prediction = prediction


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
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing prediction files if present.")
    parser.add_argument("--max-model-error-attempts", type=int, default=5, help="Max attempts for model-error samples in a single run.")
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
    pred_path: Path | None = None,
    resume: bool = False,
    max_model_error_attempts: int = 5,
) -> list[VQAPrediction]:
    predictions: list[VQAPrediction] = []
    completed_by_id: dict[str, VQAPrediction] = {}
    running_correct = 0
    if resume and pred_path and pred_path.exists():
        for pred in load_latest_predictions_jsonl(pred_path).values():
            completed_by_id[pred.sample_id] = pred
        predictions = [
            completed_by_id[sample.vqa_id]
            for sample in samples
            if sample.vqa_id in completed_by_id and not _should_retry_existing_prediction(completed_by_id[sample.vqa_id])
        ]
        running_correct = sum(int(pred.correct) for pred in predictions)
        print(f"[resume] baseline={baseline} loaded_completed={len(completed_by_id)} from {pred_path}", flush=True)
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        if sample.vqa_id in completed_by_id and not _should_retry_existing_prediction(completed_by_id[sample.vqa_id]):
            pred = completed_by_id[sample.vqa_id]
            print(
                f"[{index}/{total}] baseline={baseline} skip task={sample.task_family} sample={sample.vqa_id} pred={pred.prediction} gold={pred.gold} correct={pred.correct} running_acc={running_correct}/{len(predictions)}={_format_ratio(running_correct, len(predictions))}",
                flush=True,
            )
            continue
        started_at = time.time()
        evidence = collect_evidence(sample, state_store, spatial_store)
        default_evidence_ids = list(evidence.get("evidence_ids", []))
        tool_calls = []
        if baseline in {"directevidence", "foodstate", "ours-foodevidence"}:
            tool_calls = ["state_store", "spatial_store"]
        prev_attempts = completed_by_id.get(sample.vqa_id).attempt_count if sample.vqa_id in completed_by_id else 0
        messages = build_messages(sample, baseline, evidence)
        pred_idx = 0
        evidence_ids = [] if baseline == "textonly" else default_evidence_ids[:2]
        failure_type: str | None = None
        attempt_count = prev_attempts
        for retry_index in range(1, max_model_error_attempts + 1):
            attempt_count = retry_index
            try:
                response = model_client.complete(messages, temperature=temperature)
                pred_idx, answer_evidence_ids, parse_failure = parse_model_output(response.content, sample, baseline)
                if _should_retry_response(baseline, pred_idx, answer_evidence_ids, parse_failure, sample):
                    retry_response = model_client.complete(messages, temperature=temperature)
                    retry_idx, retry_evidence_ids, retry_failure = parse_model_output(retry_response.content, sample, baseline)
                    if _prefer_retry_result(retry_idx, retry_evidence_ids, retry_failure):
                        pred_idx, answer_evidence_ids, parse_failure = retry_idx, retry_evidence_ids, retry_failure
                evidence_ids = answer_evidence_ids or ([] if baseline == "textonly" else default_evidence_ids[:2])
                failure_type = parse_failure if parse_failure else (None if pred_idx == sample.correct_idx else "reasoning_error")
                if not (failure_type and failure_type.startswith("model_error:")):
                    break
            except Exception as exc:
                pred_idx = 0
                evidence_ids = [] if baseline == "textonly" else default_evidence_ids[:2]
                failure_type = f"model_error:{type(exc).__name__}"
            print(
                f"[{index}/{total}] baseline={baseline} retry task={sample.task_family} sample={sample.vqa_id} attempt={retry_index}/{max_model_error_attempts} failure={failure_type}",
                flush=True,
            )
            if retry_index >= max_model_error_attempts:
                break
        prediction = VQAPrediction(
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
            failure_type=failure_type,
            attempt_count=attempt_count,
        )
        predictions.append(prediction)
        running_correct += int(prediction.correct)
        if pred_path:
            append_prediction_jsonl(pred_path, prediction)
        elapsed = time.time() - started_at
        print(
            f"[{index}/{total}] baseline={baseline} done task={sample.task_family} sample={sample.vqa_id} pred={pred_idx} gold={sample.correct_idx} correct={prediction.correct} failure={failure_type} attempt_count={attempt_count} evidence_count={len(prediction.evidence_ids)} elapsed={elapsed:.1f}s running_acc={running_correct}/{len(predictions)}={_format_ratio(running_correct, len(predictions))}",
            flush=True,
        )
        if prediction.failure_type and prediction.failure_type.startswith("model_error:") and prediction.attempt_count >= max_model_error_attempts:
            print(
                f"[pause] baseline={baseline} task={sample.task_family} sample={sample.vqa_id} failure={prediction.failure_type} attempt_count={prediction.attempt_count}; stopping run for later resume",
                flush=True,
            )
            raise PersistentModelError(prediction)
    return predictions


def _should_retry_response(
    baseline: str,
    pred_idx: int,
    evidence_ids: list[str],
    parse_failure: str | None,
    sample,
) -> bool:
    if baseline != "ours-foodevidence":
        return False
    if parse_failure == "format_error":
        return True
    if pred_idx == 0 and sample.correct_idx != 0 and not evidence_ids:
        return True
    return False


def _prefer_retry_result(pred_idx: int, evidence_ids: list[str], parse_failure: str | None) -> bool:
    if parse_failure is None and evidence_ids:
        return True
    if parse_failure is None and pred_idx != 0:
        return True
    return False


def write_jsonl(path: Path, predictions: list[VQAPrediction]) -> None:
    path.write_text("\n".join(pred.to_json() for pred in predictions) + "\n", encoding="utf-8")


def append_prediction_jsonl(path: Path, prediction: VQAPrediction) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prediction.to_json() + "\n")


def load_predictions_jsonl(path: Path) -> list[VQAPrediction]:
    predictions: list[VQAPrediction] = []
    if not path.exists():
        return predictions
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        payload.setdefault("attempt_count", 1)
        predictions.append(VQAPrediction(**payload))
    return predictions


def load_latest_predictions_jsonl(path: Path) -> dict[str, VQAPrediction]:
    latest: dict[str, VQAPrediction] = {}
    for pred in load_predictions_jsonl(path):
        latest[pred.sample_id] = pred
    return latest


def load_selected_samples(index_dir: Path, limit: int | None, task_family: str | None, task_family_group: str | None):
    if task_family_group:
        samples = []
        per_family_limit = limit
        for family in TASK_FAMILY_GROUPS[task_family_group]:
            family_samples = load_vqa_samples(index_dir, limit=per_family_limit, task_family=family)
            samples.extend(family_samples)
        return samples
    return load_vqa_samples(index_dir, limit=limit, task_family=task_family)


def merge_summary(run_out_dir: Path, baselines: Iterable[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for baseline in baselines:
        pred_path = run_out_dir / f"predictions_{baseline}.jsonl"
        metrics_path = run_out_dir / f"metrics_{baseline}.json"
        advantage_path = run_out_dir / f"advantage_{baseline}.json"
        if not pred_path.exists() or not metrics_path.exists() or not advantage_path.exists():
            continue
        advantage_payload = read_json(advantage_path)
        summary[baseline] = {
            "predictions": pred_path.as_posix(),
            "metrics": read_json(metrics_path),
            "advantage": advantage_payload.get("score"),
            "verdict": advantage_payload.get("verdict"),
        }
    return summary


def _format_ratio(correct: int, total: int) -> str:
    if not total:
        return "0.000"
    return f"{correct / total:.3f}"


def _should_retry_existing_prediction(prediction: VQAPrediction) -> bool:
    return bool(prediction.failure_type and prediction.failure_type.startswith("model_error:"))


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    model_client = OpenAICompatibleModelClient()
    state_store = FoodStateStore(args.index_dir)
    spatial_store = SpatialContextStore(args.index_dir)
    samples = load_selected_samples(args.index_dir, limit=args.limit, task_family=args.task_family, task_family_group=args.task_family_group)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    run_base = args.task_family_group or args.task_family or "mixed"
    run_name = f"{run_base}_limit{args.limit}"
    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix}"
    run_out_dir = args.out_dir / run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    food_metrics = read_json(Path("outputs/results/food_state_metrics.json"))
    spatial_metrics = read_json(Path("outputs/results/spatial_context_metrics.json"))

    for baseline in baselines:
        pred_path = run_out_dir / f"predictions_{baseline}.jsonl"
        metrics_path = run_out_dir / f"metrics_{baseline}.json"
        advantage_path = run_out_dir / f"advantage_{baseline}.json"
        if args.resume and pred_path.exists() and metrics_path.exists() and advantage_path.exists():
            existing_predictions = load_latest_predictions_jsonl(pred_path)
            completed_predictions = [pred for pred in existing_predictions.values() if not _should_retry_existing_prediction(pred)]
            if len(completed_predictions) == len(samples):
                print(f"[comparison] baseline={baseline} already complete, reusing existing files", flush=True)
                continue
        if pred_path.exists() and not args.resume:
            pred_path.unlink()
        print(f"[comparison] running baseline={baseline} samples={len(samples)} resume={args.resume}", flush=True)
        try:
            predictions = run_one_baseline(
                baseline,
                samples,
                model_client,
                state_store,
                spatial_store,
                temperature=args.temperature,
                pred_path=pred_path,
                resume=args.resume,
                max_model_error_attempts=args.max_model_error_attempts,
            )
        except PersistentModelError:
            predictions = load_predictions_jsonl(pred_path) if pred_path.exists() else []
            metrics = compute_metrics(predictions)
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = merge_summary(run_out_dir, baselines)
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
            return 2
        metrics = compute_metrics(predictions)
        advantage = food_agent_advantage_score(predictions, food_metrics, spatial_metrics)
        verdict = judge_advantage(advantage)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        advantage_path.write_text(
            json.dumps({"score": advantage, "verdict": verdict}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = merge_summary(run_out_dir, baselines)
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
