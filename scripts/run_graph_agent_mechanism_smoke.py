#!/usr/bin/env python3
"""Generate deterministic mechanism-coverage artifacts for the graph agent."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.agent.executor import GraphAgentExecutor
from food_agent.agent.graph_agent import GraphAgent, GraphAgentResult
from food_agent.agent.planner import GraphAgentPlanner, PlannerDecision
from food_agent.agent.state import AgentState
from food_agent.agent.verifier import GraphAgentVerifier
from food_agent.config import ProjectConfig
from food_agent.memory import GraphEdgeRecord, GraphMemoryStore, GraphNodeRecord
from food_agent.paths import ProjectPaths
from food_agent.tools import AgentToolbox
from scripts.audit_graph_agent_mechanisms import build_audit_report


class FailingClient:
    def complete_json(self, messages, temperature=0.0):
        raise RuntimeError("force heuristic fallback")

    def complete(self, messages, temperature=0.0):
        class Response:
            content = "0"

        return Response()


class FinishClient:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def complete_json(self, messages, temperature=0.0):
        return dict(self.payload)

    def complete(self, messages, temperature=0.0):
        class Response:
            content = "0"

        return Response()


class ImmediateFinishPlanner:
    def next_action(self, *, state, tool_schemas, hints):
        return PlannerDecision(
            thought="已有可复用证据，直接结束。",
            tool="finish",
            args={"prediction": 0, "answer": "0", "confidence": 0.95},
            done=True,
            answer="0",
            prediction=0,
            confidence=0.95,
        )


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=defaults.output_root / "results" / "graph_agent_mechanism_smoke",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run_smoke(out_dir=args.out_dir, keep_temp=args.keep_temp)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def run_smoke(*, out_dir: Path, keep_temp: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_root_obj = tempfile.TemporaryDirectory(prefix="graph-agent-mechanism-smoke-", dir=out_dir.as_posix())
    temp_root = Path(temp_root_obj.name)
    if keep_temp:
        temp_root_obj.cleanup = lambda: None  # type: ignore[method-assign]
    paths = build_paths(temp_root)

    records, session_trace_rows = build_mechanism_records(paths)
    predictions_path = out_dir / "predictions_graph_agent.jsonl"
    session_trace_path = out_dir / "session_trace.jsonl"
    session_state_path = out_dir / "session_state.json"
    audit_path = out_dir / "mechanism_audit.json"

    write_jsonl(predictions_path, records)
    write_jsonl(session_trace_path, session_trace_rows)
    session_state_path.write_text(
        json.dumps(build_session_state(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit = build_audit_report(
        run_dir=out_dir,
        predictions_file=predictions_path,
        result_files=[],
        session_trace=session_trace_path,
        session_state=session_state_path,
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not keep_temp:
        temp_root_obj.cleanup()
    return audit


def build_paths(root: Path) -> ProjectPaths:
    output_root = root / "outputs"
    config = ProjectConfig(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data" / "HD-EPIC",
        annotation_root=PROJECT_ROOT / "annotations" / "hd-epic-annotations-main",
        output_root=output_root,
        graph_memory_root=output_root / "graph_memory",
        graph_agent_artifacts_root=output_root / "graph_agent_artifacts",
        graph_agent_sessions_root=output_root / "graph_agent_sessions",
        graph_agent_runs_root=output_root / "graph_agent_runs",
    )
    return ProjectPaths(config)


def build_mechanism_records(paths: ProjectPaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    session_trace_rows: list[dict[str, Any]] = []

    reuse_record = scenario_relation_reuse(paths)
    records.append(reuse_record)
    session_trace_rows.append(trace_row_from_record(reuse_record))

    override_record = scenario_planner_override_and_recovery()
    records.append(override_record)
    session_trace_rows.append(trace_row_from_record(override_record))

    ineffective_record = scenario_ineffective_tool_avoidance()
    records.append(ineffective_record)
    session_trace_rows.append(trace_row_from_record(ineffective_record))

    conflict_record = scenario_conflict_and_writeback_hygiene(paths)
    records.append(conflict_record)
    session_trace_rows.append(trace_row_from_record(conflict_record))

    session_record = scenario_session_writeback_skip(paths)
    records.append(session_record)
    session_trace_rows.append(trace_row_from_record(session_record))

    open_query_record = scenario_open_query_structured_summary(paths)
    records.append(open_query_record)
    session_trace_rows.append(trace_row_from_record(open_query_record))

    finalize_record = scenario_cached_artifact_reuse_and_deterministic_finalize(paths)
    records.append(finalize_record)
    session_trace_rows.append(trace_row_from_record(finalize_record))

    return records, session_trace_rows


def scenario_relation_reuse(paths: ProjectPaths) -> dict[str, Any]:
    store = GraphMemoryStore(paths.graph_memory_root / "vid_relation")
    store.replace_graph(
        [
            GraphNodeRecord(
                node_id="timeline:vid_relation:step1",
                node_type="timeline_event",
                label="add onion",
                video_id="vid_relation",
                start_time=10.0,
                end_time=12.0,
                attributes={"event_type": "timeline_event"},
                keywords=["onion", "add"],
            ),
            GraphNodeRecord(
                node_id="ocr:vid_relation:reading1",
                node_type="ocr_reading",
                label="scale reading",
                video_id="vid_relation",
                start_time=10.5,
                end_time=10.5,
                attributes={"reading": "20 g"},
                keywords=["reading", "20 g"],
            ),
            GraphNodeRecord(
                node_id="state:vid_relation:onion1",
                node_type="state_change",
                label="onion became softened",
                video_id="vid_relation",
                start_time=10.8,
                end_time=11.5,
                attributes={"state_change_hint": "softened"},
                keywords=["onion", "softened"],
            ),
        ],
        [
            GraphEdgeRecord(
                edge_id="edge:vid_relation:1",
                source_id="timeline:vid_relation:step1",
                target_id="ocr:vid_relation:reading1",
                edge_type="co_occurs",
                video_id="vid_relation",
            ),
            GraphEdgeRecord(
                edge_id="edge:vid_relation:2",
                source_id="timeline:vid_relation:step1",
                target_id="state:vid_relation:onion1",
                edge_type="same_step",
                video_id="vid_relation",
            ),
        ],
    )
    toolbox = AgentToolbox(store=store, paths=paths, model_client=FailingClient(), video_id="vid_relation")
    executor = GraphAgentExecutor(toolbox, ImmediateFinishPlanner(), GraphAgentVerifier())
    state = AgentState(
        video_id="vid_relation",
        question="What is the reading on the scale? <TIME 00:00:10.000 video 1> <TIME 00:00:12.000 video 1>",
        choices=["20 g", "10 g"],
        task_family="ingredient_ingredient_weight",
        max_steps=2,
    )
    result_state = executor.execute(state)
    result = GraphAgentResult(
        vqa_id="mechanism:relation_reuse",
        video_id="vid_relation",
        task_family="ingredient_ingredient_weight",
        prediction=result_state.final_prediction,
        answer_text=result_state.final_answer,
        evidence_bundle=result_state.evidence_bundle,
        tool_trace=result_state.tool_trace,
        raw_model_output=result_state.final_answer,
        working_memory=result_state.working_memory,
        retrieved_frames=result_state.retrieved_frames,
        confidence=result_state.confidence,
        elapsed_seconds=0.0,
        verification_history=result_state.verification_history,
        tool_failures=result_state.tool_failures,
        ineffective_tools=result_state.ineffective_tools,
        open_questions=result_state.open_questions,
        visited_times=result_state.visited_times,
        artifacts=result_state.artifacts,
    )
    return result.to_dict(gold=0)


def scenario_planner_override_and_recovery() -> dict[str, Any]:
    planner = GraphAgentPlanner(FinishClient({"thought": "直接继续 OCR。", "tool": "run_ocr_on_image", "args": {"image_path": "/tmp/frame.jpg"}, "done": False, "confidence": 0.7}))
    state = AgentState(
        video_id="vid_override",
        question="What is the reading on the scale? <TIME 00:00:10.000 video 1> <TIME 00:00:12.000 video 1>",
        choices=["10 g", "20 g"],
        task_family="open_query_ocr",
    )
    state.retrieved_frames = ["/tmp/frame.jpg"]
    state.open_questions = ["need_ocr_reading", "need_time_localization"]
    state.tool_failures = [{"tool": "run_ocr_on_image", "error_type": "RuntimeError", "error_message": "down"}]
    decision = planner.next_action(
        state=state,
        tool_schemas=[],
        hints={"times": [10.0, 12.0], "input_times": [], "bbox": None, "ingredient_name": "onions", "ocr_keyword": "reading"},
    )
    state.record_tool_failure("run_ocr_on_image", {"image_path": "/tmp/frame.jpg"}, "RuntimeError", "down")
    state.record_tool(decision.tool, decision.args, "recovered_plan", raw_result={"nodes": [{"node_id": "ocr:1"}]})
    return state_to_record(
        state,
        vqa_id="mechanism:planner_override",
        task_family="open_query_ocr",
        prediction=None,
        answer_text="",
    )


def scenario_ineffective_tool_avoidance() -> dict[str, Any]:
    planner = GraphAgentPlanner(
        FinishClient(
            {
                "thought": "继续排序。",
                "tool": "rank_choices_from_state",
                "args": {"question": "What is the reading?", "choices": ["10 g", "20 g"], "evidence": [], "working_memory": []},
                "done": False,
                "confidence": 0.3,
            }
        )
    )
    state = AgentState(
        video_id="vid_ineffective",
        question="What is the reading on the scale? <TIME 00:00:10.000 video 1> <TIME 00:00:12.000 video 1>",
        choices=["10 g", "20 g"],
        task_family="ingredient_ingredient_weight",
    )
    state.retrieved_frames = ["/tmp/frame.jpg"]
    state.open_questions = ["need_ocr_reading"]
    state.tool_trace = [
        {"tool": "sample_sparse_frames", "raw_result": {}},
        {"tool": "rank_choices_from_state", "raw_result": {"best_index": 0, "answer": "10 g", "confidence": 0.22}},
    ]
    state.record_ineffective_tool("run_ocr_on_image", {"image_path": "/tmp/frame.jpg"}, "no_new_evidence")
    decision = planner.next_action(
        state=state,
        tool_schemas=[],
        hints={"times": [10.0, 12.0], "input_times": [], "bbox": None, "ingredient_name": "onions", "ocr_keyword": "reading"},
    )
    state.record_tool(decision.tool, decision.args, "avoid_ineffective", raw_result={"nodes": [{"node_id": "ocr:2"}]})
    return state_to_record(
        state,
        vqa_id="mechanism:ineffective_avoidance",
        task_family="ingredient_ingredient_weight",
        prediction=None,
        answer_text="",
    )


def scenario_conflict_and_writeback_hygiene(paths: ProjectPaths) -> dict[str, Any]:
    store = GraphMemoryStore(paths.graph_memory_root / "vid_conflict")
    store.replace_graph([], [])
    toolbox = AgentToolbox(store=store, paths=paths, model_client=FailingClient(), video_id="vid_conflict")
    executor = GraphAgentExecutor(toolbox, GraphAgentPlanner(FailingClient()), GraphAgentVerifier())
    state = AgentState(
        video_id="vid_conflict",
        question="Where is the bowl and what is the reading?",
        choices=["a", "b"],
        task_family="open_query_state",
    )
    state.evidence_bundle = ["ocr_reading=10 g", "target_location=left side", "state_change_hint=raw onion"]
    state.working_memory = ["ocr_reading=10 g", "target_location=left side", "state_change_hint=raw onion"]
    executor._merge_result_into_state(state, "run_ocr_on_image", {"reading": "20 g", "text": "20 g", "artifact_path": "/tmp/scale.jpg"})
    executor._merge_result_into_state(
        state,
        "inspect_visual_evidence",
        {"target_location": "right side", "state_change_hint": "fully mixed", "ongoing_action": "stirring"},
    )
    state.retrieved_frames = ["/tmp/scale_10.000s.jpg", "/tmp/frame_10.000s.jpg"]
    executor._auto_write_ocr_observation(state, "run_ocr_on_image", {"reading": "20 g", "text": "20 g", "artifact_path": "/tmp/scale_10.000s.jpg"})
    executor._auto_write_visual_observation(
        state,
        {"target_location": "right side", "ongoing_action": "stirring", "state_change_hint": "fully mixed", "confidence": 0.8},
    )
    executor._merge_result_into_state(state, "run_ocr_on_image", {"reading": "10 g", "text": "10 g", "artifact_path": "/tmp/scale.jpg"})
    executor._merge_result_into_state(
        state,
        "inspect_visual_evidence",
        {"target_location": "left side", "state_change_hint": "raw onion", "ongoing_action": "stirring"},
    )
    executor._reconcile_conflict_questions(state)
    return state_to_record(
        state,
        vqa_id="mechanism:conflict_hygiene",
        task_family="open_query_state",
        prediction=None,
        answer_text="",
    )


def scenario_session_writeback_skip(paths: ProjectPaths) -> dict[str, Any]:
    test_video_id = "vid_session_conflict"
    shutil.rmtree(paths.graph_agent_sessions_root / test_video_id, ignore_errors=True)
    shutil.rmtree(paths.graph_memory_root / test_video_id, ignore_errors=True)
    store = GraphMemoryStore(paths.graph_memory_root / test_video_id)
    store.replace_graph(
        [
            GraphNodeRecord(
                node_id=f"video:{test_video_id}",
                node_type="video",
                label=test_video_id,
                video_id=test_video_id,
                attributes={"source": "mechanism_smoke"},
                keywords=[test_video_id],
            )
        ],
        [],
    )
    state = AgentState(
        video_id=test_video_id,
        question="What happened? <TIME 00:00:10.000 video 1> <TIME 00:00:12.000 video 1>",
        choices=["a"],
        task_family="recipe_step_recognition",
    )
    state.working_memory = ["ocr_reading=98 g", "possible_step=stirring bowl", "candidate_answer_index=0"]
    state.evidence_bundle = ["type=timeline_event; label=stirring bowl"]
    state.retrieved_frames = ["/tmp/frame.jpg"]
    state.open_questions = ["conflict:conflicting_ocr_readings"]
    agent = GraphAgent(paths=paths, model_client=FailingClient())
    session = agent.begin_video_session(test_video_id)
    session._compress_and_persist_session_memory(
        state=state,
        row={
            "vqa_id": "mechanism:session_writeback_skip",
            "task_family": "recipe_step_recognition",
            "primary_video_id": test_video_id,
            "question": state.question,
            "choices_json": '["a"]',
            "correct_idx": 0,
            "inputs_json": "{}",
        },
    )
    return state_to_record(
        state,
        vqa_id="mechanism:session_writeback_skip",
        task_family="recipe_step_recognition",
        prediction=None,
        answer_text="",
    )


def scenario_open_query_structured_summary(paths: ProjectPaths) -> dict[str, Any]:
    store = GraphMemoryStore(paths.graph_memory_root / "vid_open_query")
    store.replace_graph([], [])
    toolbox = AgentToolbox(store=store, paths=paths, model_client=FailingClient(), video_id="vid_open_query")
    executor = GraphAgentExecutor(toolbox, ImmediateFinishPlanner(), GraphAgentVerifier())
    state = AgentState(
        video_id="vid_open_query",
        question="What happened after the ingredient was added?",
        choices=["OPEN_ENDED_RESPONSE"],
        task_family="open_query_temporal_summary",
        max_steps=2,
    )
    state.evidence_bundle = ["type=timeline_event; label=ingredient added", "state_change_hint=mixture became smoother"]
    state.working_memory = ["possible_step=stirring bowl"]
    result_state = executor.execute(state)
    agent = GraphAgent(paths=paths, model_client=FailingClient())
    answer_text, prediction = agent._finalize_state_answer(state=result_state, freeform=True)
    result = GraphAgentResult(
        vqa_id="mechanism:open_query_summary",
        video_id="vid_open_query",
        task_family="open_query_temporal_summary",
        prediction=prediction,
        answer_text=answer_text,
        evidence_bundle=result_state.evidence_bundle,
        tool_trace=result_state.tool_trace,
        raw_model_output=answer_text,
        working_memory=result_state.working_memory,
        retrieved_frames=result_state.retrieved_frames,
        confidence=result_state.confidence,
        elapsed_seconds=0.0,
        verification_history=result_state.verification_history,
        tool_failures=result_state.tool_failures,
        ineffective_tools=result_state.ineffective_tools,
        open_questions=result_state.open_questions,
        visited_times=result_state.visited_times,
        artifacts=result_state.artifacts,
    )
    return result.to_dict(include_row={"question": state.question, "choices_json": json.dumps(state.choices, ensure_ascii=False), "inputs_json": "{}"})


def scenario_cached_artifact_reuse_and_deterministic_finalize(paths: ProjectPaths) -> dict[str, Any]:
    store = GraphMemoryStore(paths.graph_memory_root / "vid_cached_finalize")
    store.replace_graph([], [])
    toolbox = AgentToolbox(store=store, paths=paths, model_client=FailingClient(), video_id="vid_cached_finalize")
    executor = GraphAgentExecutor(toolbox, GraphAgentPlanner(FailingClient()), GraphAgentVerifier())
    state = AgentState(
        video_id="vid_cached_finalize",
        question="How was the microwave opened?",
        choices=["press button", "pull door"],
        task_family="fine_grained_how_recognition",
        max_steps=2,
    )
    cached_payload = {
        "artifact_paths": ["/tmp/fine_grained_how_recognition_segment_005.500s.jpg"],
        "items": [
            {
                "artifact_path": "/tmp/fine_grained_how_recognition_segment_005.500s.jpg",
                "time_s": 5.5,
                "tag": "fine_grained_how_recognition_segment",
            }
        ],
    }
    state.record_tool("retrieve_cached_artifacts", {}, "cached reuse", raw_result=cached_payload)
    executor._merge_result_into_state(state, "retrieve_cached_artifacts", cached_payload)
    state.working_memory.append("action_mechanism_best_index=0 confidence=0.45")
    agent = GraphAgent(paths=paths, model_client=FailingClient())
    answer_text, prediction = agent._finalize_state_answer(state=state, freeform=False)
    result = GraphAgentResult(
        vqa_id="mechanism:cached_finalize",
        video_id="vid_cached_finalize",
        task_family="fine_grained_how_recognition",
        prediction=prediction,
        answer_text=answer_text,
        evidence_bundle=state.evidence_bundle,
        tool_trace=state.tool_trace,
        raw_model_output=answer_text,
        working_memory=state.working_memory,
        retrieved_frames=state.retrieved_frames,
        confidence=state.confidence,
        elapsed_seconds=0.0,
        verification_history=state.verification_history,
        tool_failures=state.tool_failures,
        ineffective_tools=state.ineffective_tools,
        open_questions=state.open_questions,
        visited_times=state.visited_times,
        artifacts=state.artifacts,
    )
    return result.to_dict(gold=0)


def state_to_record(
    state: AgentState,
    *,
    vqa_id: str,
    task_family: str,
    prediction: int | None,
    answer_text: str,
) -> dict[str, Any]:
    result = GraphAgentResult(
        vqa_id=vqa_id,
        video_id=state.video_id,
        task_family=task_family,
        prediction=prediction,
        answer_text=answer_text,
        evidence_bundle=state.evidence_bundle,
        tool_trace=state.tool_trace,
        raw_model_output=answer_text,
        working_memory=state.working_memory,
        retrieved_frames=state.retrieved_frames,
        confidence=state.confidence,
        elapsed_seconds=0.0,
        verification_history=state.verification_history,
        tool_failures=state.tool_failures,
        ineffective_tools=state.ineffective_tools,
        open_questions=state.open_questions,
        visited_times=state.visited_times,
        artifacts=state.artifacts,
    )
    return result.to_dict()


def trace_row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "vqa_id": record.get("vqa_id"),
        "tool_calls": record.get("tool_calls") or [entry.get("tool") for entry in record.get("tool_trace", []) if isinstance(entry, dict)],
        "tool_failures": record.get("tool_failures") or [],
        "ineffective_tools": record.get("ineffective_tools") or [],
        "latest_verification": record.get("latest_verification") or (record.get("verification_history") or [{}])[-1],
        "open_questions_tail": (record.get("open_questions") or [])[-8:],
        "working_memory_tail": (record.get("working_memory") or [])[-12:],
        "evidence_tail": (record.get("evidence_bundle") or [])[-12:],
    }


def build_session_state(records: list[dict[str, Any]]) -> dict[str, Any]:
    working_memory: list[str] = []
    evidence_bundle: list[str] = []
    for record in records:
        for item in record.get("working_memory") or []:
            if isinstance(item, str) and item not in working_memory:
                working_memory.append(item)
        for item in record.get("evidence_bundle") or []:
            if isinstance(item, str) and item not in evidence_bundle:
                evidence_bundle.append(item)
    return {
        "video_id": "mechanism-smoke",
        "question_count": len(records),
        "session_memory": {
            "video_id": "mechanism-smoke",
            "working_memory": working_memory[-200:],
            "evidence_bundle": evidence_bundle[-200:],
            "retrieved_frames": [],
            "retrieved_node_ids": [],
            "retrieved_nodes": [],
            "hypotheses": [],
            "open_questions": [],
            "tool_failures": [],
            "ineffective_tools": [],
            "verification_history": [],
            "confidence": 0.0,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
