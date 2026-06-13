#!/usr/bin/env python3
"""Run a unified acceptance check over mechanism smoke and available real-run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.paths import ProjectPaths
from scripts.audit_graph_agent_mechanisms import build_audit_report
from scripts.run_graph_agent_mechanism_smoke import run_smoke
from scripts.run_graph_agent_real_subset_acceptance import run_real_subset_acceptance


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=defaults.output_root / "results" / "graph_agent_acceptance",
    )
    parser.add_argument(
        "--real-run-dir",
        type=Path,
        default=defaults.output_root / "results" / "graph_agent_mechanism_smoke_probe",
        help="已有真实或半真实产物目录；若不存在则自动回退到显式 result/session 文件。",
    )
    parser.add_argument("--run-real-subset", action="store_true", help="现场运行一个真实小样本子集并将其作为 real artifacts。")
    parser.add_argument("--env-file", type=Path, default=defaults.project_root / ".secrets" / "model.env")
    parser.add_argument("--real-index-file", type=Path, default=defaults.output_root / "event_index" / "vqa_samples.parquet")
    parser.add_argument("--real-video-id", default=None)
    parser.add_argument("--real-task-family", action="append", default=None)
    parser.add_argument("--real-limit-per-task", type=int, default=1)
    parser.add_argument("--real-max-steps", type=int, default=8)
    parser.add_argument("--real-include-open-query-probes", action="store_true")
    parser.add_argument("--real-open-query-max-steps", type=int, default=3)
    parser.add_argument("--real-open-query-timeout-seconds", type=int, default=90)
    parser.add_argument("--real-resume", action="store_true")
    parser.add_argument(
        "--real-result-file",
        type=Path,
        action="append",
        default=[],
        help="显式指定真实结果 JSON 文件，可重复传入。",
    )
    parser.add_argument("--real-session-trace", type=Path, default=None)
    parser.add_argument("--real-session-state", type=Path, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = run_acceptance(
        out_dir=args.out_dir,
        real_run_dir=args.real_run_dir,
        real_result_files=list(args.real_result_file or []),
        real_session_trace=args.real_session_trace,
        real_session_state=args.real_session_state,
        run_real_subset=args.run_real_subset,
        env_file=args.env_file,
        real_index_file=args.real_index_file,
        real_video_id=args.real_video_id,
        real_task_families=args.real_task_family,
        real_limit_per_task=args.real_limit_per_task,
        real_max_steps=args.real_max_steps,
        real_include_open_query_probes=args.real_include_open_query_probes,
        real_open_query_max_steps=args.real_open_query_max_steps,
        real_open_query_timeout_seconds=args.real_open_query_timeout_seconds,
        real_resume=args.real_resume,
        keep_temp=args.keep_temp,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run_acceptance(
    *,
    out_dir: Path,
    real_run_dir: Path | None,
    real_result_files: list[Path],
    real_session_trace: Path | None,
    real_session_state: Path | None,
    run_real_subset: bool = False,
    env_file: Path | None = None,
    real_index_file: Path | None = None,
    real_video_id: str | None = None,
    real_task_families: list[str] | None = None,
    real_limit_per_task: int = 1,
    real_max_steps: int = 8,
    real_include_open_query_probes: bool = False,
    real_open_query_max_steps: int = 3,
    real_open_query_timeout_seconds: int = 90,
    real_resume: bool = False,
    keep_temp: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mechanism_dir = out_dir / "mechanism_smoke"
    smoke_audit = run_smoke(out_dir=mechanism_dir, keep_temp=keep_temp)

    real_subset_report: dict[str, Any] | None = None
    if run_real_subset:
        if env_file is not None:
            from food_agent.config import load_env_file

            load_env_file(env_file)
        real_subset_dir = out_dir / "real_subset"
        real_subset_report = run_real_subset_acceptance(
            index_file=real_index_file or (ProjectPaths.from_env().output_root / "event_index" / "vqa_samples.parquet"),
            out_dir=real_subset_dir,
            video_id=real_video_id,
            task_families=real_task_families or ["ingredient_ingredient_weight", "ingredient_ingredient_retrieval", "recipe_step_recognition"],
            limit_per_task=real_limit_per_task,
            max_steps=real_max_steps,
            include_open_query_probes=real_include_open_query_probes,
            open_query_max_steps=real_open_query_max_steps,
            open_query_timeout_seconds=real_open_query_timeout_seconds,
            resume=real_resume,
        )
        real_run_dir = real_subset_dir
        real_result_files = []
        real_session_trace = None
        real_session_state = None

    real_audit = build_real_audit(
        real_run_dir=real_run_dir,
        real_result_files=real_result_files,
        real_session_trace=real_session_trace,
        real_session_state=real_session_state,
    )

    report = {
        "acceptance_version": 1,
        "mechanism_smoke": smoke_audit,
        "real_artifacts": real_audit,
        "real_subset_report": real_subset_report,
        "summary": build_acceptance_summary(smoke_audit, real_audit),
    }
    (out_dir / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_real_audit(
    *,
    real_run_dir: Path | None,
    real_result_files: list[Path],
    real_session_trace: Path | None,
    real_session_state: Path | None,
) -> dict[str, Any]:
    candidate_run_dir = real_run_dir if real_run_dir and real_run_dir.exists() else None
    existing_result_files = [path for path in real_result_files if path.exists()]
    if candidate_run_dir is None and not existing_result_files and real_session_trace is None and real_session_state is None:
        fallback = Path("/22liushoulong/agent/hd-epic/outputs/results/graph_agent_mechanism_smoke_probe")
        if fallback.exists():
            candidate_run_dir = fallback
        else:
            existing_result_files = [
                Path("/22liushoulong/agent/hd-epic/outputs/graph_agent_runs/recipe_step_recognition/task_restore.json"),
                Path("/22liushoulong/agent/hd-epic/outputs/graph_agent_runs/ingredient_ingredient_retrieval/ingredient_ingredient_retrieval_ingredient_ingredient_retrieval_34.json"),
            ]
            real_session_trace = Path("/22liushoulong/agent/hd-epic/outputs/graph_agent_sessions/P08-20240617-130401/session_trace.jsonl")
            real_session_state = Path("/22liushoulong/agent/hd-epic/outputs/graph_agent_sessions/P08-20240617-130401/session_state.json")
    return build_audit_report(
        run_dir=candidate_run_dir,
        predictions_file=None,
        result_files=existing_result_files,
        session_trace=real_session_trace,
        session_state=real_session_state,
    )


def build_acceptance_summary(smoke_audit: dict[str, Any], real_audit: dict[str, Any]) -> dict[str, Any]:
    smoke_summary = smoke_audit.get("summary", {})
    real_summary = real_audit.get("summary", {})
    return {
        "mechanism_requirements_satisfied": int(smoke_summary.get("requirements_satisfied") or 0),
        "mechanism_requirements_total": int(smoke_summary.get("requirements_total") or 0),
        "mechanism_coverage_ratio": float(smoke_summary.get("coverage_ratio") or 0.0),
        "real_requirements_satisfied": int(real_summary.get("requirements_satisfied") or 0),
        "real_requirements_total": int(real_summary.get("requirements_total") or 0),
        "real_coverage_ratio": float(real_summary.get("coverage_ratio") or 0.0),
        "acceptance_gate": classify_acceptance_gate(smoke_summary, real_summary),
    }


def classify_acceptance_gate(smoke_summary: dict[str, Any], real_summary: dict[str, Any]) -> str:
    smoke_ratio = float(smoke_summary.get("coverage_ratio") or 0.0)
    real_ratio = float(real_summary.get("coverage_ratio") or 0.0)
    if smoke_ratio >= 1.0 and real_ratio >= 0.75:
        return "mechanism_complete_real_partial"
    if smoke_ratio >= 1.0 and real_ratio >= 0.5:
        return "mechanism_complete_real_early"
    if smoke_ratio >= 0.8:
        return "mechanism_strong_real_weak"
    return "not_ready"


if __name__ == "__main__":
    raise SystemExit(main())
