from pathlib import Path

from food_agent.comparison import (
    _extract_json_payload,
    _extract_evidence_ids_from_text,
    build_messages,
    extract_sample_context,
    infer_specialization,
    parse_hms,
    parse_model_output,
)
from food_agent.vqa import VQASample
from scripts.run_agent_comparison import load_selected_samples, run_one_baseline


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


def test_extract_json_payload_from_code_block() -> None:
    payload = _extract_json_payload('```json\n{"choice": 2, "evidence_ids": ["e1"]}\n```')
    assert payload["choice"] == 2
    assert payload["evidence_ids"] == ["e1"]


def test_extract_evidence_ids_from_text() -> None:
    evidence_ids = _extract_evidence_ids_from_text("reason ... evidence_ids: ingredient:abc/add/0/0 and event_id recipe_step:foo")
    assert evidence_ids == ["ingredient:abc/add/0/0", "recipe_step:foo"]


def test_infer_specialization() -> None:
    assert infer_specialization("ingredient_ingredient_retrieval") == "ingredient"
    assert infer_specialization("recipe_step_recognition") == "recipe"
    assert infer_specialization("nutrition_nutrition_change") == "nutrition"
    assert infer_specialization("unknown_task") == "general"


def test_build_messages_ours_mentions_evidence_requirement() -> None:
    sample = make_sample()
    messages = build_messages(
        sample,
        "ours-foodevidence",
        {"context": extract_sample_context(sample), "recipe": None, "ingredient": None, "nutrition": None, "spatial": None, "evidence_ids": ["e1"]},
    )
    content = messages[1]["content"]
    assert "只输出 JSON" in content
    assert "evidence_ids 至少填写 1 个" in content
    assert "这是食材题" in content


def test_parse_model_output_non_json_choice_text() -> None:
    idx, evidence_ids, failure = parse_model_output(
        "choice: capsule\nevidence_ids: ingredient:abc/add/0/0",
        make_sample(),
        "ours-foodevidence",
    )
    assert idx == 2
    assert evidence_ids == ["ingredient:abc/add/0/0"]
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


def test_run_one_baseline_continues_on_model_error() -> None:
    class DummyStateStore:
        pass

    class DummySpatialStore:
        pass

    class BrokenClient:
        def complete(self, messages, temperature=0.0):
            raise RuntimeError("boom")

    class DummyState:
        active_steps = []
        completed_steps = []
        next_steps = []
        added = []
        pending = []
        weighed = []
        evidence_ids = []
        totals = {}
        unknown_count = 0
        audio_events = []
        gaze_priming = []
        object_tracks = []

    dummy_state = DummyState()
    state_store = DummyStateStore()
    state_store.recipe_state = lambda *args, **kwargs: dummy_state
    state_store.ingredient_state = lambda *args, **kwargs: dummy_state
    state_store.nutrition_delta = lambda *args, **kwargs: dummy_state
    spatial_store = DummySpatialStore()
    spatial_store.combined_context = lambda *args, **kwargs: dummy_state

    predictions = run_one_baseline(
        "ours-foodevidence",
        [make_sample()],
        BrokenClient(),
        state_store=state_store,
        spatial_store=spatial_store,
        temperature=0.0,
    )
    assert len(predictions) == 1
    assert predictions[0].failure_type == "model_error:RuntimeError"
    assert predictions[0].prediction == 0
