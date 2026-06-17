"""Evidence aggregator: fuse multi-modal evidence with conflict detection."""

from typing import Dict, List, Optional

from food_agent.perception.evidence import Evidence


# Module type weights for confidence fusion
MODULE_WEIGHTS = {
    "primary": 0.6,
    "secondary": 0.3,
    "default": 0.1,
}


class Aggregator:
    """Aggregate evidence from multiple perception modules.

    Responsibilities:
        - Collect evidence from all modules
        - Detect conflicts between evidence sources
        - Fuse evidence with weighted confidence
        - Produce a summary for the LLM
    """

    def __init__(self):
        self._evidence: List[Evidence] = []
        self._module_priority: Dict[str, str] = {}  # module -> primary/secondary

    def set_priority(self, primary: List[str], secondary: List[str]) -> None:
        """Set module priority from the Router's routing strategy."""
        self._module_priority = {}
        for m in primary:
            self._module_priority[m] = "primary"
        for m in secondary:
            self._module_priority[m] = "secondary"

    def add_evidence(self, evidence: Evidence) -> None:
        """Add a piece of evidence to the aggregator."""
        self._evidence.append(evidence)

    def add_evidence_list(self, evidence_list: List[Evidence]) -> None:
        """Add multiple evidence items."""
        self._evidence.extend(evidence_list)

    def clear(self) -> None:
        """Remove all evidence."""
        self._evidence.clear()

    @property
    def evidence_list(self) -> List[Evidence]:
        return list(self._evidence)

    def align_evidence(self) -> List[Evidence]:
        """Sort evidence by time range for temporal alignment."""
        return sorted(self._evidence, key=lambda e: e.time_range.get("start", 0))

    def detect_conflicts(self) -> List[Dict]:
        """Detect conflicting evidence between modules.

        Returns list of conflict dicts with evidence_ids and description.
        """
        conflicts = []
        by_type: Dict[str, List[Evidence]] = {}
        for ev in self._evidence:
            by_type.setdefault(ev.evidence_type, []).append(ev)

        # Check for same-type evidence with very different confidences
        for etype, evs in by_type.items():
            if len(evs) < 2:
                continue
            confs = [e.confidence for e in evs]
            if max(confs) - min(confs) > 0.5:
                conflicts.append({
                    "type": "confidence_divergence",
                    "evidence_type": etype,
                    "evidence_ids": [e.evidence_id for e in evs],
                    "confidences": confs,
                })

        return conflicts

    def fuse_evidence(self) -> Dict:
        """Fuse all evidence into a single weighted result.

        Returns:
            Dict with 'confidence', 'content_summary', 'evidence_count',
            'conflicts', 'by_type'.
        """
        if not self._evidence:
            return {
                "confidence": 0.0,
                "content_summary": "",
                "evidence_count": 0,
                "conflicts": [],
                "by_type": {},
            }

        conflicts = self.detect_conflicts()

        # Weighted confidence
        weighted_sum = 0.0
        weight_sum = 0.0
        for ev in self._evidence:
            priority = self._module_priority.get(ev.source_module, "default")
            weight = MODULE_WEIGHTS.get(priority, 0.1)
            weighted_sum += ev.confidence * weight
            weight_sum += weight

        fused_confidence = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # Reduce confidence if there are conflicts
        if conflicts:
            fused_confidence *= 0.8

        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        for ev in self._evidence:
            by_type.setdefault(ev.evidence_type, []).append(ev.content)

        return {
            "confidence": min(1.0, max(0.0, fused_confidence)),
            "evidence_count": len(self._evidence),
            "conflicts": conflicts,
            "by_type": {k: len(v) for k, v in by_type.items()},
        }

    def get_confidence(self) -> float:
        """Get the overall fused confidence score."""
        return self.fuse_evidence()["confidence"]

    def get_summary(self, max_items: int = 10) -> str:
        """Generate a text summary of all evidence for the LLM.

        Returns a structured string suitable for prompt injection.
        """
        if not self._evidence:
            return "No evidence collected yet."

        aligned = self.align_evidence()
        lines = []
        for i, ev in enumerate(aligned[:max_items]):
            lines.append(
                f"[{i+1}] {ev.source_module} ({ev.evidence_type}) "
                f"t={ev.time_range.get('start', 0):.1f}-{ev.time_range.get('end', 0):.1f}s "
                f"conf={ev.confidence:.2f}"
            )
            # Add key content
            for k, v in ev.content.items():
                if isinstance(v, (str, int, float, bool)):
                    lines.append(f"    {k}: {v}")
                elif isinstance(v, list) and len(v) <= 3:
                    lines.append(f"    {k}: {v}")

        if len(aligned) > max_items:
            lines.append(f"... and {len(aligned) - max_items} more evidence items")

        return "\n".join(lines)
