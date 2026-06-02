from food_agent.advantage_metrics import evidence_rate, failure_rate, food_agent_advantage_score
from food_agent.vqa import VQAPrediction


def test_advantage_metrics() -> None:
    preds = [
        VQAPrediction("a", "ours", "task", "v", "q", ["x"], 0, 0, True, ["e1"], ["tool"], None),
        VQAPrediction("b", "ours", "task", "v", "q", ["x"], 0, 1, False, [], [], "reasoning_error"),
    ]
    assert evidence_rate(preds) == 0.5
    assert failure_rate(preds) == 0.5
    score = food_agent_advantage_score(preds)
    assert 0.0 <= score["food_agent_advantage_score"] <= 1.0

