from food_agent.vqa import VQAPrediction, compute_metrics, parse_choice_prediction


def test_parse_choice_prediction_index() -> None:
    assert parse_choice_prediction("1", ["a", "b"]) == 1


def test_parse_choice_prediction_text() -> None:
    assert parse_choice_prediction("the answer is milk", ["water", "milk"]) == 1


def test_compute_metrics() -> None:
    preds = [
        VQAPrediction("a", "b", "task", "v", "q", ["x"], 0, 0, True, [], [], None),
        VQAPrediction("b", "b", "task", "v", "q", ["x"], 0, 1, False, [], [], "reasoning_error"),
    ]
    metrics = compute_metrics(preds)
    assert metrics["accuracy"] == 0.5
    assert metrics["by_task_family"]["task"]["accuracy"] == 0.5

