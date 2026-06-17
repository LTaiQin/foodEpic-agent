"""Evaluation metrics for HD-EPIC VQA."""

from typing import Dict, List, Optional


def accuracy(predictions: List, ground_truth: List) -> float:
    """Compute accuracy.

    Args:
        predictions: List of predicted answers.
        ground_truth: List of ground truth answers.

    Returns:
        Accuracy as float in [0, 1].
    """
    if not predictions or not ground_truth:
        return 0.0
    n = min(len(predictions), len(ground_truth))
    correct = sum(1 for p, g in zip(predictions[:n], ground_truth[:n]) if p == g)
    return correct / n


def accuracy_per_category(
    predictions: List,
    ground_truth: List,
    categories: List[str],
) -> Dict[str, float]:
    """Compute accuracy per category.

    Returns:
        Dict mapping category name to accuracy.
    """
    by_cat: Dict[str, List[tuple]] = {}
    n = min(len(predictions), len(ground_truth), len(categories))
    for i in range(n):
        cat = categories[i]
        by_cat.setdefault(cat, []).append((predictions[i], ground_truth[i]))

    result = {}
    for cat, pairs in by_cat.items():
        correct = sum(1 for p, g in pairs if p == g)
        result[cat] = correct / len(pairs) if pairs else 0.0
    return result


def average_confidence(predictions: List[Dict]) -> float:
    """Average confidence across predictions.

    Each prediction dict should have a 'confidence' key.
    """
    if not predictions:
        return 0.0
    confs = [p.get("confidence", 0) for p in predictions if isinstance(p, dict)]
    return sum(confs) / len(confs) if confs else 0.0


def average_tool_calls(predictions: List[Dict]) -> float:
    """Average number of tool calls per prediction."""
    if not predictions:
        return 0.0
    calls = [p.get("tool_calls", 0) for p in predictions if isinstance(p, dict)]
    return sum(calls) / len(calls) if calls else 0.0


def average_latency(predictions: List[Dict]) -> float:
    """Average latency in seconds per prediction."""
    if not predictions:
        return 0.0
    latencies = [p.get("latency", 0) for p in predictions if isinstance(p, dict)]
    return sum(latencies) / len(latencies) if latencies else 0.0
