"""Agent state: mutable working state for a single question."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from food_agent.perception.evidence import Evidence


@dataclass
class AgentState:
    """Maintains the complete state of a single question-answering session."""

    question: str = ""
    category: str = "general"
    route: Dict = field(default_factory=dict)
    evidence_list: List[Evidence] = field(default_factory=list)
    iteration: int = 0
    tool_call_history: List[Dict] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    choices: Optional[List[str]] = None
    video_id: str = ""
    participant_id: str = ""
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence_list.append(evidence)

    def add_tool_call(self, tool_name: str, parameters: Dict, result: Any = None) -> None:
        self.tool_call_history.append({
            "iteration": self.iteration,
            "tool": tool_name,
            "parameters": parameters,
            "result_type": type(result).__name__,
            "timestamp": time.time(),
        })

    def increment_iteration(self) -> None:
        self.iteration += 1

    def record_confidence(self, confidence: float) -> None:
        self.confidence_history.append(confidence)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def unique_tools_called(self) -> set:
        return {call["tool"] for call in self.tool_call_history}

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "category": self.category,
            "route": self.route,
            "evidence_count": len(self.evidence_list),
            "iteration": self.iteration,
            "tool_calls": len(self.tool_call_history),
            "confidence_history": self.confidence_history,
            "elapsed_time": self.elapsed_time,
            "video_id": self.video_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, indent=2)
