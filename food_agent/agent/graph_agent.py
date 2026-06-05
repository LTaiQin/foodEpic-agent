"""End-to-end complete graph/video agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    prediction: int | None
    answer_text: str
    evidence_bundle: list[str]
    tool_trace: list[dict[str, Any]]
    raw_model_output: str
    working_memory: list[str]
    retrieved_frames: list[str]
    confidence: float


class GraphAgent:
    """Full agent that plans, retrieves, revisits raw video, and writes back memory."""

    def __init__(self, paths: ProjectPaths | None = None, model_client: OpenAICompatibleModelClient | None = None):
        self.paths = paths or ProjectPaths.from_env()
        self.builder = VideoGraphBuilder(self.paths)
        self.model_client = model_client or OpenAICompatibleModelClient()
        self.planner = GraphAgentPlanner(self.model_client)

    def answer_vqa_row(self, row: dict[str, Any], *, max_steps: int = 6) -> GraphAgentResult:
        video_id = str(row["primary_video_id"])
        store = self._ensure_store(video_id)
        toolbox = AgentToolbox(store=store, paths=self.paths, model_client=self.model_client, video_id=video_id)
        executor = GraphAgentExecutor(toolbox, self.planner)
        state = AgentState(
            video_id=video_id,
            question=str(row["question"]),
            choices=json.loads(row["choices_json"]),
            task_family=str(row["task_family"]),
            inputs_json=str(row.get("inputs_json", "{}")),
            max_steps=max_steps,
        )
        state = executor.execute(state)
        if state.final_prediction is None:
            answer_text = self._answer_from_state(state)
            prediction = self._parse_prediction(answer_text, state.choices)
        else:
            answer_text = state.final_answer
            prediction = state.final_prediction
        return GraphAgentResult(
            prediction=prediction,
            answer_text=answer_text,
            evidence_bundle=state.evidence_bundle,
            tool_trace=state.tool_trace,
            raw_model_output=answer_text,
            working_memory=state.working_memory,
            retrieved_frames=state.retrieved_frames,
            confidence=state.confidence,
        )

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
