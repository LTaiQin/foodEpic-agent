"""Reasoning trace: record the Agent's complete decision process."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepRecord:
    """Record of a single step in the Agent's reasoning."""
    iteration: int
    action: str  # "tool_call", "evaluation", "answer_generation"
    tool_name: str = ""
    tool_params: Dict = field(default_factory=dict)
    result_summary: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    decision: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningTrace:
    """Complete trace of the Agent's reasoning process."""

    question: str = ""
    category: str = ""
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    total_iterations: int = 0
    total_tool_calls: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)

    def finalize(self, answer: str, confidence: float) -> None:
        self.final_answer = answer
        self.final_confidence = confidence
        self.total_iterations = max((s.iteration for s in self.steps), default=0)
        self.total_tool_calls = sum(1 for s in self.steps if s.action == "tool_call")
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "category": self.category,
            "final_answer": self.final_answer,
            "final_confidence": self.final_confidence,
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "duration_seconds": round(self.duration, 2),
            "steps": [
                {
                    "iteration": s.iteration,
                    "action": s.action,
                    "tool": s.tool_name,
                    "params": s.tool_params,
                    "result": s.result_summary[:200],
                    "confidence_before": s.confidence_before,
                    "confidence_after": s.confidence_after,
                    "decision": s.decision,
                }
                for s in self.steps
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)
