"""Adaptive depth control: evaluate evidence sufficiency and suggest expansion."""

from typing import Dict, List, Optional

from food_agent.perception.evidence import Evidence


# Confidence thresholds
THRESHOLD_SUFFICIENT = 0.8
THRESHOLD_EXPAND = 0.5


class Judge:
    """Evaluate whether collected evidence is sufficient to answer a question.

    Decision logic:
        confidence > 0.8  -> sufficient (answer now)
        0.5 < confidence <= 0.8 -> expand (try more modules)
        confidence <= 0.5 -> full_search (try everything)
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    def evaluate_sufficiency(
        self,
        evidence_list: List[Evidence],
        question: str,
        route: Dict,
    ) -> Dict:
        """Evaluate if we have enough evidence to answer.

        Returns:
            Dict with 'status' (sufficient/insufficient/full_search),
            'confidence', 'reason'.
        """
        if not evidence_list:
            return {
                "status": "full_search",
                "confidence": 0.0,
                "reason": "No evidence collected yet.",
            }

        # Compute weighted confidence
        from food_agent.reasoning.aggregator import Aggregator
        agg = Aggregator()
        agg.set_priority(route.get("primary", []), route.get("secondary", []))
        for ev in evidence_list:
            agg.add_evidence(ev)
        confidence = agg.get_confidence()

        # Check evidence diversity
        evidence_types = set(ev.evidence_type for ev in evidence_list)
        primary_covered = any(
            ev.source_module in route.get("primary", [])
            for ev in evidence_list
        )

        if confidence > THRESHOLD_SUFFICIENT and primary_covered:
            return {
                "status": "sufficient",
                "confidence": confidence,
                "reason": f"High confidence ({confidence:.2f}) with primary module evidence.",
            }
        elif confidence > THRESHOLD_EXPAND:
            return {
                "status": "insufficient",
                "confidence": confidence,
                "reason": f"Moderate confidence ({confidence:.2f}), need more evidence.",
            }
        else:
            return {
                "status": "full_search",
                "confidence": confidence,
                "reason": f"Low confidence ({confidence:.2f}), need comprehensive search.",
            }

    def suggest_expansion(
        self,
        evidence_list: List[Evidence],
        question: str,
        route: Dict,
    ) -> Dict:
        """Suggest which modules to call next.

        Returns:
            Dict with 'modules_to_call' (list of module names) and 'parameters'.
        """
        covered = set(ev.source_module for ev in evidence_list)
        primary = route.get("primary", [])
        secondary = route.get("secondary", [])

        # First try uncovered primary modules
        uncovered_primary = [m for m in primary if m not in covered]
        if uncovered_primary:
            return {
                "modules_to_call": uncovered_primary[:2],
                "parameters": {"reason": "primary module not yet queried"},
            }

        # Then try uncovered secondary modules
        uncovered_secondary = [m for m in secondary if m not in covered]
        if uncovered_secondary:
            return {
                "modules_to_call": uncovered_secondary[:2],
                "parameters": {"reason": "secondary module not yet queried"},
            }

        # All modules covered, suggest time expansion
        if evidence_list:
            time_ranges = [
                (ev.time_range.get("start", 0), ev.time_range.get("end", 0))
                for ev in evidence_list
            ]
            min_t = min(t[0] for t in time_ranges)
            max_t = max(t[1] for t in time_ranges)
            return {
                "modules_to_call": primary[:1],
                "parameters": {
                    "reason": "expand time range",
                    "start_time": max(0, min_t - 10),
                    "end_time": max_t + 10,
                },
            }

        return {
            "modules_to_call": primary[:1],
            "parameters": {"reason": "fallback"},
        }

    def should_stop(
        self,
        evidence_list: List[Evidence],
        iteration: int,
        question: str,
        route: Dict,
    ) -> bool:
        """Determine if the agent should stop iterating.

        Returns True if we should stop and generate an answer.
        """
        if iteration >= self.max_iterations:
            return True

        result = self.evaluate_sufficiency(evidence_list, question, route)
        return result["status"] == "sufficient"
