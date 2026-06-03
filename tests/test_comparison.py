from pathlib import Path

from food_agent.comparison import (
    _extract_json_payload,
    _extract_evidence_ids_from_text,
    _ingredient_match_score,
    _extract_question_recipe_step_name,
    _normalize_ingredient_text,
    _token_overlap_score,
    build_messages,
    collect_evidence,
    extract_sample_context,
    infer_specialization,
    parse_hms,
    parse_model_output,
    rank_choice_hints,
)
from food_agent.vqa import VQASample
from scripts.run_agent_comparison import (
    _prefer_retry_result,
    _should_retry_response,
    load_selected_samples,
    run_one_baseline,
)


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


def make_question_time_sample() -> VQASample:
    return VQASample(
        vqa_id="sample_q",
        task_family="ingredient_ingredient_retrieval",
        primary_video_id="P01-20240202-110250",
        participant_id="P01",
        question="Between <TIME 00:00:10.000 video 1> and <TIME 00:00:20.000 video 1>, which ingredient was added?",
        choices=["water", "milk", "capsule"],
        correct_idx=2,
        inputs={"video 1": {"id": "P01-20240202-110250"}},
    )


def test_parse_hms() -> None:
    assert parse_hms("00:03:1.8") == 181.8


def test_extract_sample_context() -> None:
    ctx = extract_sample_context(make_sample())
    assert ctx.video_id == "P01-20240202-110250"
    assert ctx.time_point == 15.0


def test_extract_sample_context_from_question_times() -> None:
    ctx = extract_sample_context(make_question_time_sample())
    assert ctx.start_time == 10.0
    assert ctx.end_time == 20.0
    assert ctx.time_point == 15.0


def test_extract_sample_context_keeps_all_video_ids() -> None:
    sample = VQASample(
        vqa_id="multi",
        task_family="recipe_recipe_recognition",
        primary_video_id="P01-20240202-110250",
        participant_id="P01",
        question="Which recipe was carried out?",
        choices=["A", "B"],
        correct_idx=0,
        inputs={
            "video 1": {"id": "P01-20240202-110250"},
            "video 2": {"id": "P01-20240202-120000"},
        },
    )
    ctx = extract_sample_context(sample)
    assert ctx.video_ids == ["P01-20240202-110250", "P01-20240202-120000"]


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
    assert "choice_hints=" in content
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
    state_store.ingredient_interval = lambda *args, **kwargs: []
    state_store.recipe_catalog = lambda *args, **kwargs: []
    state_store.activity_window = lambda *args, **kwargs: type("ActivityWindow", (), {"activities": []})()
    state_store.all_video_activities = lambda *args, **kwargs: []
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


def test_should_retry_response_on_empty_zero_prediction() -> None:
    sample = make_sample()
    assert _should_retry_response("ours-foodevidence", 0, [], None, sample) is True
    assert _should_retry_response("directevidence", 0, [], None, sample) is False


def test_prefer_retry_result_prefers_valid_evidence() -> None:
    assert _prefer_retry_result(2, ["ingredient:abc/add/0/0"], None) is True
    assert _prefer_retry_result(2, [], None) is True
    assert _prefer_retry_result(0, [], "format_error") is False


def test_token_overlap_score() -> None:
    assert _token_overlap_score("add olive oil to pan", "olive oil was added to the pan") > 0.3


def test_rank_choice_hints_ingredient() -> None:
    sample = make_sample()
    evidence = {
        "ingredient_interval": [{"label": "capsule"}],
        "ingredient": type("Ingredient", (), {"added": [{"label": "capsule"}], "pending": []})(),
        "recipe": type("Recipe", (), {"active_steps": [], "completed_steps": []})(),
        "spatial": type("Spatial", (), {"audio_events": []})(),
    }
    hints = rank_choice_hints(sample, evidence, "ingredient")
    assert hints[0].choice_text == "capsule"


def test_rank_choice_hints_recipe_uses_recipe_catalog() -> None:
    sample = VQASample(
        vqa_id="recipe",
        task_family="recipe_recipe_recognition",
        primary_video_id="P01-20240202-110250",
        participant_id="P01",
        question="Which recipe was carried out?",
        choices=["Coffee", "Soup"],
        correct_idx=0,
        inputs={"video 1": {"id": "P01-20240202-110250"}},
    )
    evidence = {
        "recipe": type("Recipe", (), {"active_steps": [], "completed_steps": [], "next_steps": []})(),
        "recipe_catalog": [{"recipe_id": "P01_R01", "name": "Coffee", "video_ids": ["P01-20240202-110250"], "step_count": 3}],
        "activity_window": None,
        "video_activities": [],
        "spatial": type("Spatial", (), {"object_tracks": [], "audio_events": []})(),
    }
    hints = rank_choice_hints(sample, evidence, "recipe")
    assert hints[0].choice_text == "Coffee"


def test_rank_choice_hints_ingredient_membership_uses_target_recipe_catalog() -> None:
    sample = VQASample(
        vqa_id="ingredient_membership",
        task_family="ingredient_ingredient_recognition",
        primary_video_id="P07-20240529-191007",
        participant_id="P07",
        question="Which of these ingredients is not used in Chopped Chickpea Salad",
        choices=["garlic powder", "stilton", "cucumber", "paprika", "kale"],
        correct_idx=4,
        inputs={"video 1": {"id": "P07-20240529-191007"}},
    )
    evidence = {
        "recipe_catalog": [
            {
                "recipe_id": "P07_R02",
                "name": "Chopped Chickpea Salad",
                "video_ids": ["P07-20240529-131737", "P07-20240529-134410"],
                "step_count": 14,
                "ingredients": [
                    "garlic powder",
                    "stilton",
                    "cucumber",
                    "paprika",
                ],
            }
        ],
        "ingredient": type("Ingredient", (), {"added": [], "pending": []})(),
        "ingredient_interval": [],
        "recipe": type("Recipe", (), {"active_steps": [], "completed_steps": []})(),
        "spatial": type("Spatial", (), {"audio_events": []})(),
    }
    hints = rank_choice_hints(sample, evidence, "ingredient")
    assert hints[0].choice_text == "kale"


def test_rank_choice_hints_exact_ingredient_quantity_uses_recipe_amounts() -> None:
    sample = VQASample(
        vqa_id="exact_quantity",
        task_family="ingredient_exact_ingredient_recognition",
        primary_video_id="P08-20240617-184909",
        participant_id="P08",
        question="What was the exact quantity of garlic used in Fish Cakes and Vegetables",
        choices=["6 g", "5 g", "7 g", "3 g", "4 g"],
        correct_idx=1,
        inputs={"video 1": {"id": "P08-20240617-184909"}},
    )
    evidence = {
        "recipe_catalog": [
            {
                "recipe_id": "P08_R07",
                "name": "Fish Cakes and Vegetables",
                "video_ids": ["P08-20240617-184909"],
                "step_count": 11,
                "ingredients": ["garlic"],
                "ingredient_amounts": [{"name": "garlic", "amount": 5, "amount_unit": "g"}],
            }
        ],
        "ingredient": type("Ingredient", (), {"added": [], "pending": [], "weighed": []})(),
        "ingredient_interval": [],
        "recipe": type("Recipe", (), {"active_steps": [], "completed_steps": []})(),
        "spatial": type("Spatial", (), {"audio_events": []})(),
    }
    hints = rank_choice_hints(sample, evidence, "ingredient")
    assert hints[0].choice_text == "5 g"


def test_extract_question_recipe_step_name() -> None:
    question = (
        "Which high-level activity did the participant do while completing recipe step "
        "Add the onions and chopped garlic and brown slowly until tender and golden, about 5 minutes in this video?"
    )
    assert _extract_question_recipe_step_name(question) == (
        "Add the onions and chopped garlic and brown slowly until tender and golden, about 5 minutes"
    )


def test_collect_evidence_uses_step_focus_for_following_activity() -> None:
    sample = VQASample(
        vqa_id="follow",
        task_family="recipe_following_activity_recognition",
        primary_video_id="P02-20240209-184316",
        participant_id="P02",
        question=(
            "Which high-level activity did the participant do while completing recipe step "
            "Add the onions and chopped garlic and brown slowly until tender and golden, about 5 minutes in this video?"
        ),
        choices=["a", "b"],
        correct_idx=0,
        inputs={"video 1": {"id": "P02-20240209-184316"}},
    )

    class DummyStateStore:
        def recipe_catalog(self, video_ids):
            return []

        def recipe_step_matches(self, video_id, step_text):
            return type(
                "StepLookup",
                (),
                {
                    "matches": [
                        {
                            "event_id": "recipe_step:P02_R01:P02_R01_S03:0:8",
                            "start_time": 1418.964,
                            "end_time": 1420.644,
                        }
                    ]
                },
            )()

        def recipe_state(self, video_id, time):
            return type("Recipe", (), {"active_steps": [], "completed_steps": [], "next_steps": []})()

        def ingredient_state(self, video_id, time):
            return type("Ingredient", (), {"added": [], "pending": [], "weighed": []})()

        def nutrition_delta(self, video_id, time):
            return type("Nutrition", (), {"totals": {}, "unknown_count": 0, "evidence_ids": []})()

        def activity_window(self, video_id, start_time, end_time):
            return type(
                "ActivityWindow",
                (),
                {
                    "activities": [
                        {
                            "event_id": "activity:P02-20240209-184316:24",
                            "text": "Continue stirring mushrooms with tomato and season",
                        }
                    ]
                },
            )()

        def all_video_activities(self, video_id):
            return []

    class DummySpatialStore:
        def combined_context(self, video_id, time=None, object_name=None):
            return type("Spatial", (), {"audio_events": [], "gaze_priming": [], "object_tracks": []})()

    evidence = collect_evidence(sample, DummyStateStore(), DummySpatialStore())
    assert evidence["step_focus"]["event_id"] == "recipe_step:P02_R01:P02_R01_S03:0:8"
    assert evidence["activity_window"].activities[0]["event_id"] == "activity:P02-20240209-184316:24"
    assert "recipe_step:P02_R01:P02_R01_S03:0:8" in evidence["evidence_ids"]




def test_ingredient_match_score_alias() -> None:
    score, reason = _ingredient_match_score("cinnamon sticks", [{"label": "cinnamon", "text": "add cinnamon"}])
    assert score > 0
    assert reason in {"match_interval_added", "partial_alias_match"}


def test_normalize_ingredient_text() -> None:
    assert _normalize_ingredient_text("Extra-Virgin Olive Oil") == "extra virgin olive oil"
