"""End-to-end complete graph/video agent."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from food_agent.agent.executor import GraphAgentExecutor
from food_agent.agent.planner import GraphAgentPlanner
from food_agent.agent.state import AgentState
from food_agent.graph import VideoGraphBuilder
from food_agent.memory import GraphMemoryStore
from food_agent.model_client import OpenAICompatibleModelClient
from food_agent.paths import ProjectPaths
from food_agent.tools import AgentToolbox


CHOICE_RE = re.compile(r"\b([0-4])\b")


@dataclass(frozen=True)
class GraphAgentResult:
    vqa_id: str
    video_id: str
    task_family: str
    prediction: int | None
    answer_text: str
    evidence_bundle: list[str]
    tool_trace: list[dict[str, Any]]
    raw_model_output: str
    working_memory: list[str]
    retrieved_frames: list[str]
    confidence: float
    elapsed_seconds: float

    def to_dict(self, *, gold: int | None = None, include_row: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "vqa_id": self.vqa_id,
            "video_id": self.video_id,
            "task_family": self.task_family,
            "prediction": self.prediction,
            "gold": gold,
            "correct": None if gold is None or self.prediction is None else self.prediction == int(gold),
            "answer_text": self.answer_text,
            "confidence": self.confidence,
            "elapsed_seconds": self.elapsed_seconds,
            "tool_trace": self.tool_trace,
            "evidence_bundle": self.evidence_bundle,
            "working_memory": self.working_memory,
            "retrieved_frames": self.retrieved_frames,
            "raw_model_output": self.raw_model_output,
        }
        if include_row:
            payload["question"] = include_row.get("question")
            payload["choices_json"] = include_row.get("choices_json")
            payload["inputs_json"] = include_row.get("inputs_json")
        return payload


class GraphAgentVideoSession:
    """Persistent same-video execution session that reuses one store/toolbox/executor."""

    def __init__(self, *, agent: GraphAgent, video_id: str):
        self.agent = agent
        self.video_id = video_id
        self.store = agent._ensure_store(video_id)
        self.toolbox = AgentToolbox(
            store=self.store,
            paths=agent.paths,
            model_client=agent.model_client,
            video_id=video_id,
        )
        self.executor = GraphAgentExecutor(self.toolbox, agent.planner)
        self.session_dir = agent.paths.output_root / "graph_agent_sessions" / video_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.session_dir / "session_trace.jsonl"

    def answer_vqa_row(self, row: dict[str, Any], *, max_steps: int = 6) -> GraphAgentResult:
        started_at = time.time()
        vqa_id = str(row.get("vqa_id") or "")
        task_family = str(row["task_family"])
        state = AgentState(
            video_id=self.video_id,
            question=str(row["question"]),
            choices=json.loads(row["choices_json"]),
            task_family=task_family,
            inputs_json=str(row.get("inputs_json", "{}")),
            max_steps=max_steps,
        )
        state = self.executor.execute(state)
        if state.final_prediction is None:
            answer_text = self.agent._answer_from_state(state)
            prediction = self.agent._parse_prediction(answer_text, state.choices)
        else:
            answer_text = state.final_answer
            prediction = state.final_prediction
        result = GraphAgentResult(
            vqa_id=vqa_id,
            video_id=self.video_id,
            task_family=task_family,
            prediction=prediction,
            answer_text=answer_text,
            evidence_bundle=state.evidence_bundle,
            tool_trace=state.tool_trace,
            raw_model_output=answer_text,
            working_memory=state.working_memory,
            retrieved_frames=state.retrieved_frames,
            confidence=state.confidence,
            elapsed_seconds=time.time() - started_at,
        )
        self.agent._persist_result(result, row=row)
        self._append_session_trace(result, row=row)
        return result

    def _append_session_trace(self, result: GraphAgentResult, *, row: dict[str, Any]) -> None:
        payload = {
            "vqa_id": result.vqa_id,
            "task_family": result.task_family,
            "prediction": result.prediction,
            "gold": int(row["correct_idx"]) if row.get("correct_idx") is not None else None,
            "correct": None if row.get("correct_idx") is None or result.prediction is None else result.prediction == int(row["correct_idx"]),
            "elapsed_seconds": result.elapsed_seconds,
            "tool_calls": [entry.get("tool") for entry in result.tool_trace if isinstance(entry, dict)],
            "working_memory_tail": result.working_memory[-12:],
            "evidence_tail": result.evidence_bundle[-12:],
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class GraphAgent:
    """Full agent that plans, retrieves, revisits raw video, and writes back memory."""

    def __init__(self, paths: ProjectPaths | None = None, model_client: OpenAICompatibleModelClient | None = None):
        self.paths = paths or ProjectPaths.from_env()
        self.builder = VideoGraphBuilder(self.paths)
        self.model_client = model_client or OpenAICompatibleModelClient()
        self.planner = GraphAgentPlanner(self.model_client)
        self._video_sessions: dict[str, GraphAgentVideoSession] = {}

    def answer_vqa_row(self, row: dict[str, Any], *, max_steps: int = 6) -> GraphAgentResult:
        video_id = str(row["primary_video_id"])
        session = self.begin_video_session(video_id)
        return session.answer_vqa_row(row, max_steps=max_steps)

    def begin_video_session(self, video_id: str) -> GraphAgentVideoSession:
        session = self._video_sessions.get(video_id)
        if session is None:
            session = GraphAgentVideoSession(agent=self, video_id=video_id)
            self._video_sessions[video_id] = session
        return session

    def reset_video_session(self, video_id: str) -> None:
        self._video_sessions.pop(video_id, None)

    def _ensure_store(self, video_id: str) -> GraphMemoryStore:
        graph_dir = self.paths.output_root / "graph_memory" / video_id
        store = GraphMemoryStore(graph_dir)
        existing = store.query_nodes(video_id=video_id, limit=1)
        if not existing:
            return self.builder.build(video_id)
        return store

    def _answer_from_state(self, state: AgentState) -> str:
        evidence_text = "\n".join(f"- {item}" for item in state.evidence_bundle[:20])
        memory_text = "\n".join(f"- {item}" for item in state.working_memory[:20])
        messages = [
            {
                "role": "system",
                "content": (
                    "你是图谱工具型视频问答 agent 的最终裁决器。"
                    "只能基于当前工作记忆和证据输出最终选项编号 0-4。"
                    "如果证据很弱，也必须给出最合理选项，但不能编造额外事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"题型: {state.task_family}\n"
                    f"问题: {state.question}\n"
                    "选项:\n"
                    + "\n".join(f"{idx}. {choice}" for idx, choice in enumerate(state.choices))
                    + f"\n\n工作记忆:\n{memory_text}"
                    + f"\n\n证据:\n{evidence_text}"
                    + "\n\n只输出最终选项编号。"
                ),
            },
        ]
        return self.model_client.complete(messages, temperature=0.0).content.strip()

    def _parse_prediction(self, text: str, choices: list[Any]) -> int | None:
        match = CHOICE_RE.search(text)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(choices):
                return idx
        lowered = text.lower()
        for idx, choice in enumerate(choices):
            if str(choice).lower() in lowered:
                return idx
        return None

    def _persist_result(self, result: GraphAgentResult, *, row: dict[str, Any]) -> None:
        trace_dir = self.paths.output_root / "graph_agent_runs" / result.task_family
        trace_dir.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict(
            gold=int(row["correct_idx"]) if row.get("correct_idx") is not None else None,
            include_row=row,
        )
        path = trace_dir / f"{self._safe_filename(result.vqa_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _safe_filename(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "sample"
