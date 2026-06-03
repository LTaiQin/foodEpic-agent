from pathlib import Path

from food_agent.comparison import extract_sample_context, parse_hms, parse_model_output
from food_agent.vqa import VQASample
from scripts.run_agent_comparison import load_selected_samples


def make_sample() -> VQASample:
    return VQASample(
        vqa_id="sample",
        task_family="ingredient_ingredient_retrieval",
        primary_video_id="P01-20240202-110250",
        participant_id="P01",
        question="What ingredient was added?",
        choices=["water", "milk", "capsule"],
        correct_idx=2,
        inputs={"video 1": {"id": "P01-20240202-110250", "start_time": "00:00:10.000", "end_time": "00:00:20.000"}},
    )


def test_parse_hms() -> None:
    assert parse_hms("00:03:1.8") == 181.8


def test_extract_sample_context() -> None:
    ctx = extract_sample_context(make_sample())
    assert ctx.video_id == "P01-20240202-110250"
    assert ctx.time_point == 15.0


def test_parse_model_output_json() -> None:
    idx, evidence_ids, failure = parse_model_output('{"choice": 1, "evidence_ids": ["a"]}', make_sample(), "ours-foodevidence")
    assert idx == 1
    assert evidence_ids == ["a"]
    assert failure is None


def test_load_selected_samples_group(monkeypatch) -> None:
    calls = []

    def fake_load(index_dir: Path, limit: int | None = None, task_family: str | None = None):
        calls.append((index_dir, limit, task_family))
        return [make_sample()]

    monkeypatch.setattr("scripts.run_agent_comparison.load_vqa_samples", fake_load)
    samples = load_selected_samples(Path("/tmp/index"), limit=2, task_family=None, task_family_group="food-core")
    assert len(samples) == 7
    assert all(call[1] == 2 for call in calls)
    assert {call[2] for call in calls} == {
        "ingredient_ingredient_retrieval",
        "ingredient_exact_ingredient_recognition",
        "ingredient_ingredient_recognition",
        "recipe_step_recognition",
        "recipe_recipe_recognition",
        "recipe_following_activity_recognition",
        "nutrition_nutrition_change",
    }
