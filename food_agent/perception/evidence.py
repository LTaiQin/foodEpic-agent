"""Unified evidence format for all perception modules."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """A piece of evidence produced by a perception module.

    All perception modules must return Evidence or List[Evidence].
    """
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_module: str = ""
    evidence_type: str = ""  # audio, visual, gaze, spatial, hand, nutrition, motion
    time_range: Dict[str, float] = field(default_factory=lambda: {"start": 0, "end": 0})
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "Evidence":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Evidence":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
