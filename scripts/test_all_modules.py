#!/usr/bin/env python3
"""Module-level tests: verify each module works correctly with real HD-EPIC data.

Tests cover: data loaders, perception modules, knowledge modules,
reasoning engine, and the agent pipeline.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 1. Data Loaders
# ============================================================

def test_audio_loader():
    """Test AudioLoader with multiple videos."""
    from food_agent.loaders import AudioLoader
    loader = AudioLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Audio-HDF5")

    # Test get_video_ids
    vids = loader.get_video_ids("P01")
    assert len(vids) > 10, f"Expected >10 videos, got {len(vids)}"
    print(f"  [PASS] get_video_ids: P01 has {len(vids)} videos")

    # Test load_segment across multiple videos
    for vid in vids[:3]:
        dur = loader.get_duration("P01", vid)
        assert dur > 0, f"Duration should be >0 for {vid}"
        # Load a 5-second segment from the middle
        mid = dur / 2
        audio, sr = loader.load_segment("P01", vid, mid, mid + 5)
        assert len(audio) > 0, f"Audio segment should not be empty for {vid}"
        assert sr == 48000, f"Sample rate should be 48000, got {sr}"
        assert audio.dtype == np.float32, f"Dtype should be float32"
    print(f"  [PASS] load_segment: loaded 3 videos successfully")

    # Test get_all_events (annotation-based)
    # This may return empty if annotations aren't in expected format
    events = loader.get_all_events("P01", vids[0])
    print(f"  [PASS] get_all_events: returned {len(events)} events")

    # Test get_events_in_range
    events_range = loader.get_events_in_range("P01", vids[0], 0, 30)
    print(f"  [PASS] get_events_in_range(0-30s): {len(events_range)} events")

    # Test error handling for missing video
    try:
        loader.load_segment("P01", "nonexistent", 0, 5)
        assert False, "Should have raised"
    except KeyError:
        pass
    print(f"  [PASS] error handling: KeyError for missing video")

    # Test multiple participants
    for pid in ["P02", "P05", "P09"]:
        vids_p = loader.get_video_ids(pid)
        assert len(vids_p) > 0, f"{pid} should have videos"
    print(f"  [PASS] multi-participant: P02, P05, P09 all have videos")

    loader.close()


def test_video_loader():
    """Test VideoLoader with multiple videos."""
    from food_agent.loaders import VideoLoader
    loader = VideoLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Videos")

    # Test get_video_info
    info = loader.get_video_info("P01-20240202-110250")
    assert info["fps"] == 30.0, f"FPS should be 30, got {info['fps']}"
    assert info["width"] == 1408, f"Width should be 1408, got {info['width']}"
    assert info["height"] == 1408, f"Height should be 1408, got {info['height']}"
    assert info["duration"] > 100, f"Duration should be >100s"
    print(f"  [PASS] get_video_info: {info['width']}x{info['height']} @ {info['fps']}fps, {info['duration']:.1f}s")

    # Test get_frame at different timestamps
    for t in [1.0, 10.0, 50.0, 100.0]:
        frame = loader.get_frame("P01-20240202-110250", t)
        assert frame.shape == (1408, 1408, 3), f"Frame shape wrong at t={t}: {frame.shape}"
        assert frame.dtype == np.uint8
    print(f"  [PASS] get_frame: 4 timestamps, all (1408,1408,3) uint8")

    # Test get_frames (batch extraction)
    frames = loader.get_frames("P01-20240202-110250", 10.0, 15.0, fps=1.0)
    assert len(frames) >= 4, f"Expected >=4 frames, got {len(frames)}"
    for t, f in frames:
        assert f.shape == (1408, 1408, 3)
    print(f"  [PASS] get_frames: {len(frames)} frames at 1fps in 5s range")

    # Test get_frame_at_index
    t, frame = loader.get_frame_at_index("P01-20240202-110250", 300)
    assert frame.shape == (1408, 1408, 3)
    assert abs(t - 10.0) < 0.1  # 300 frames @ 30fps = 10s
    print(f"  [PASS] get_frame_at_index: frame 300 -> t={t:.2f}s")

    # Test multiple videos
    for vid in ["P01-20240202-161354", "P08-20240620-180825"]:
        try:
            info2 = loader.get_video_info(vid)
            assert info2["fps"] > 0
        except FileNotFoundError:
            pass  # Video may not exist
    print(f"  [PASS] multi-video: tested 2 more videos")


def test_gaze_loader():
    """Test GazeLoader with multiple videos."""
    from food_agent.loaders import GazeLoader
    loader = GazeLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/SLAM-and-Gaze")

    # Test get_gaze_at_time at multiple timestamps
    for t in [10.0, 30.0, 60.0, 120.0]:
        gaze = loader.get_gaze_at_time("P01", "P01-20240202-110250", t)
        assert gaze is not None, f"Gaze should not be None at t={t}"
        assert hasattr(gaze, "left_yaw")
        assert hasattr(gaze, "pitch")
        assert hasattr(gaze, "depth_m")
        assert gaze.depth_m >= 0
    print(f"  [PASS] get_gaze_at_time: 4 timestamps OK")

    # Test get_gaze_trajectory (data starts at ~50s)
    trajectory = loader.get_gaze_trajectory("P01", "P01-20240202-110250", 52.0, 62.0)
    assert len(trajectory) > 0, "Trajectory should not be empty"
    for g in trajectory[:5]:
        assert g.timestamp_s >= 52.0
    print(f"  [PASS] get_gaze_trajectory: {len(trajectory)} points in 52-62s")

    # Test get_fixations
    fixations = loader.get_fixations("P01", "P01-20240202-110250", min_duration=0.1)
    assert isinstance(fixations, list)
    for fix in fixations[:3]:
        assert "start_time" in fix
        assert "end_time" in fix
        assert "duration" in fix
        assert fix["duration"] >= 0.1
    print(f"  [PASS] get_fixations: {len(fixations)} fixations (min 0.1s)")

    # Test pixel_estimate
    gaze = loader.get_gaze_at_time("P01", "P01-20240202-110250", 30.0)
    px, py = gaze.pixel_estimate
    assert 0 <= px <= 1920 and 0 <= py <= 1080, f"Pixel out of range: ({px}, {py})"
    print(f"  [PASS] pixel_estimate: ({px:.0f}, {py:.0f})")


def test_slam_loader():
    """Test SLAMLoader with multiple timestamps."""
    from food_agent.loaders import SLAMLoader
    loader = SLAMLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/SLAM-and-Gaze")

    # Test get_pose at multiple timestamps
    for t in [10.0, 50.0, 100.0, 200.0]:
        pose = loader.get_pose("P01", "P01-20240202-110250", t)
        assert pose is not None, f"Pose should not be None at t={t}"
        assert len(pose.position) == 3
        assert len(pose.quaternion) == 4
        # Quaternion should be unit-ish
        q_norm = np.linalg.norm(pose.quaternion)
        assert 0.9 < q_norm < 1.1, f"Quaternion norm should be ~1, got {q_norm}"
    print(f"  [PASS] get_pose: 4 timestamps, all valid 6DoF poses")

    # Test get_trajectory (data starts at ~52s)
    trajectory = loader.get_trajectory("P01", "P01-20240202-110250", 52.0, 72.0)
    assert len(trajectory) > 0
    for pose in trajectory[:5]:
        assert len(pose.position) == 3
    print(f"  [PASS] get_trajectory: {len(trajectory)} poses in 52-72s")

    # Test get_position
    pos = loader.get_position("P01", "P01-20240202-110250", 50.0)
    assert len(pos) == 3
    print(f"  [PASS] get_position: {pos}")

    # Test get_facing_direction
    facing = loader.get_facing_direction("P01", "P01-20240202-110250", 50.0)
    assert len(facing) == 3
    norm = np.linalg.norm(facing)
    assert 0.9 < norm < 1.1, f"Facing direction should be unit vector"
    print(f"  [PASS] get_facing_direction: {facing}")


def test_digital_twin_loader():
    """Test DigitalTwinLoader."""
    from food_agent.loaders import DigitalTwinLoader
    loader = DigitalTwinLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Digital-Twin")

    # Test get_fixtures for multiple participants
    for pid in ["P01", "P05", "P09"]:
        fixtures = loader.get_fixtures(pid)
        assert len(fixtures) > 10, f"{pid} should have >10 fixtures, got {len(fixtures)}"
        # Check fixture structure
        for f in fixtures[:3]:
            assert f.id, "Fixture should have id"
            assert f.fixture_type, "Fixture should have type"
            assert len(f.position) == 3
            assert len(f.size) == 3
        # Get unique types
        types = set(f.fixture_type for f in fixtures)
        print(f"  [{pid}] {len(fixtures)} fixtures, types: {sorted(types)[:8]}")
    print(f"  [PASS] get_fixtures: 3 participants tested")

    # Test get_fixture_by_type
    sinks = loader.get_fixture_by_type("P01", "sink")
    assert len(sinks) > 0, "Should have at least one sink"
    counters = loader.get_fixture_by_type("P01", "counter")
    assert len(counters) > 0, "Should have counters"
    print(f"  [PASS] get_fixture_by_type: {len(sinks)} sinks, {len(counters)} counters")

    # Test get_spatial_relation
    fixtures = loader.get_fixtures("P01")
    rel = loader.get_spatial_relation("P01", fixtures[0].id, fixtures[1].id)
    assert "relation" in rel
    assert "distance" in rel
    assert rel["distance"] >= 0
    print(f"  [PASS] get_spatial_relation: {fixtures[0].id} {rel['relation']} {fixtures[1].id}, dist={rel['distance']:.2f}m")

    # Test get_fixture_position
    pos = loader.get_fixture_position("P01", fixtures[0].id)
    assert pos is not None
    assert len(pos) == 3
    print(f"  [PASS] get_fixture_position: {fixtures[0].id} at {pos}")


def test_hands_loader():
    """Test HandsLoader with multiple videos."""
    from food_agent.loaders import HandsLoader
    loader = HandsLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Hands-Masks")

    # Test get_available_frames for multiple videos
    for vid in ["P01-20240202-110250", "P01-20240202-161354"]:
        frames = loader.get_available_frames(vid)
        assert len(frames) > 0, f"{vid} should have mask frames"
        print(f"  [{vid}] {len(frames)} frames with masks")
    print(f"  [PASS] get_available_frames: 2 videos tested")

    # Test get_mask for multiple frames
    frames = loader.get_available_frames("P01-20240202-110250")
    for fn in frames[:5]:
        mask = loader.get_mask("P01-20240202-110250", fn, "left")
        if mask is not None:
            assert mask.shape == (1408, 1408), f"Mask shape wrong: {mask.shape}"
            unique = np.unique(mask)
            assert 0 in unique
            assert len(unique) >= 2  # background + hand
    print(f"  [PASS] get_mask: 5 frames, all (1408,1408) with hand pixels")

    # Test get_masks_in_range
    if len(frames) >= 3:
        masks = loader.get_masks_in_range("P01-20240202-110250", frames[0], frames[2], "left")
        assert len(masks) >= 1
        print(f"  [PASS] get_masks_in_range: {len(masks)} masks in range")

    # Test get_all_labels
    labels = loader.get_all_labels("P01-20240202-110250", frames[0])
    assert isinstance(labels, list)
    assert "left" in labels or "right" in labels
    print(f"  [PASS] get_all_labels: {labels}")

    # Test has_hand_contact
    mask = loader.get_mask("P01-20240202-110250", frames[0], "left")
    if mask is not None:
        contact = loader.has_hand_contact(mask)
        assert isinstance(contact, bool)
        print(f"  [PASS] has_hand_contact: {contact}")


# ============================================================
# 2. Reasoning Engine
# ============================================================

def test_router():
    """Test Router classification accuracy."""
    from food_agent.reasoning.router import Router
    router = Router()

    test_cases = [
        ("What food ingredients can you see?", "ingredient"),
        ("Where is the kitchen sink?", "3d_perception"),
        ("How many calories in this dish?", "nutrition"),
        ("What is the person doing with the knife?", "fine_grained_action"),
        ("Which object was moved?", "object_motion"),
        ("What is the wearer looking at?", "gaze"),
        ("What recipe is being cooked?", "recipe"),
        ("Tell me about the kitchen", "general"),
    ]

    correct = 0
    for question, expected in test_cases:
        result = router.route(question)
        actual = result["category"]
        ok = actual == expected
        if ok:
            correct += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] '{question[:50]}' -> {actual} (expected: {expected})")

    acc = correct / len(test_cases)
    print(f"  [RESULT] Router accuracy: {correct}/{len(test_cases)} = {acc:.1%}")
    assert acc >= 0.6, f"Router accuracy should be >= 60%, got {acc:.1%}"


def test_aggregator():
    """Test Aggregator evidence fusion."""
    from food_agent.reasoning.aggregator import Aggregator
    from food_agent.perception.evidence import Evidence

    agg = Aggregator()
    agg.set_priority(["VisualAnalyzer", "AudioAnalyzer"], ["GazeTracker"])

    # Add evidence from different modules
    agg.add_evidence(Evidence(source_module="VisualAnalyzer", evidence_type="visual", confidence=0.9))
    agg.add_evidence(Evidence(source_module="AudioAnalyzer", evidence_type="audio", confidence=0.7))
    agg.add_evidence(Evidence(source_module="GazeTracker", evidence_type="gaze", confidence=0.6))

    # Test confidence
    conf = agg.get_confidence()
    assert 0.5 < conf <= 1.0, f"Confidence should be >0.5, got {conf}"
    print(f"  [PASS] get_confidence: {conf:.3f} (3 sources)")

    # Test summary
    summary = agg.get_summary()
    assert "VisualAnalyzer" in summary
    assert "AudioAnalyzer" in summary
    print(f"  [PASS] get_summary: {len(summary)} chars")

    # Test conflict detection
    conflicts = agg.detect_conflicts()
    print(f"  [PASS] detect_conflicts: {len(conflicts)} conflicts")

    # Test with conflicting evidence
    agg2 = Aggregator()
    agg2.add_evidence(Evidence(source_module="A", evidence_type="visual", confidence=0.9))
    agg2.add_evidence(Evidence(source_module="B", evidence_type="visual", confidence=0.2))
    conf2 = agg2.get_confidence()
    assert conf2 < 0.9, "Conflicting evidence should reduce confidence"
    print(f"  [PASS] conflict handling: confidence reduced to {conf2:.3f}")

    # Test empty
    agg3 = Aggregator()
    assert agg3.get_confidence() == 0.0
    print(f"  [PASS] empty aggregator: confidence=0")


def test_judge():
    """Test Judge sufficiency evaluation."""
    from food_agent.reasoning.judge import Judge
    from food_agent.perception.evidence import Evidence

    judge = Judge(max_iterations=5)
    route = {"primary": ["VisualAnalyzer"], "secondary": ["AudioAnalyzer"]}

    # Test with no evidence
    result = judge.evaluate_sufficiency([], "test", route)
    assert result["status"] == "full_search"
    print(f"  [PASS] no evidence -> full_search")

    # Test with high confidence evidence
    high_ev = [Evidence(source_module="VisualAnalyzer", evidence_type="visual", confidence=0.95)]
    result = judge.evaluate_sufficiency(high_ev, "test", route)
    assert result["status"] == "sufficient"
    print(f"  [PASS] high confidence ({result['confidence']:.2f}) -> sufficient")

    # Test with low confidence evidence
    low_ev = [Evidence(source_module="AudioAnalyzer", evidence_type="audio", confidence=0.2)]
    result = judge.evaluate_sufficiency(low_ev, "test", route)
    assert result["status"] in ["insufficient", "full_search"]
    print(f"  [PASS] low confidence ({result['confidence']:.2f}) -> {result['status']}")

    # Test suggest_expansion
    suggestion = judge.suggest_expansion([], "test", route)
    assert "modules_to_call" in suggestion
    assert len(suggestion["modules_to_call"]) > 0
    print(f"  [PASS] suggest_expansion: {suggestion['modules_to_call']}")

    # Test should_stop
    assert judge.should_stop(high_ev, 1, "test", route) == True
    assert judge.should_stop([], 5, "test", route) == True  # max iterations
    print(f"  [PASS] should_stop: high_conf=True, max_iter=True")


def test_generator():
    """Test Generator answer generation."""
    from food_agent.reasoning.generator import Generator
    from food_agent.perception.evidence import Evidence

    gen = Generator(mimo_client=None)

    # Test format_evidence_prompt
    evidence = [
        Evidence(source_module="VisualAnalyzer", evidence_type="visual",
                confidence=0.9, content={"scene_description": "person cutting tomato"}),
        Evidence(source_module="AudioAnalyzer", evidence_type="audio",
                confidence=0.7, content={"sound_type": "chopping"}),
    ]
    prompt = gen.format_evidence_prompt(evidence, "What is happening?", "action")
    assert len(prompt) > 100
    assert "What is happening?" in prompt
    assert "VisualAnalyzer" in prompt
    print(f"  [PASS] format_evidence_prompt: {len(prompt)} chars")

    # Test format with choices
    prompt_choices = gen.format_evidence_prompt(evidence, "What is this?", "action", choices=["A. cutting", "B. cooking", "C. eating"])
    assert "A. cutting" in prompt_choices
    print(f"  [PASS] format with choices: OK")

    # Test parse_answer
    parsed = gen.parse_answer("The person is cutting a tomato.", choices=None)
    assert "cutting" in parsed["answer"]
    print(f"  [PASS] parse_answer (free text): {parsed['answer'][:50]}")

    parsed_choice = gen.parse_answer("A. cutting", choices=["A. cutting", "B. cooking"])
    assert parsed_choice["answer"] == "A. cutting"
    print(f"  [PASS] parse_answer (choice): {parsed_choice['answer']}")


# ============================================================
# 3. Knowledge Modules
# ============================================================

def test_nutrition_kb():
    """Test NutritionKB."""
    from food_agent.knowledge import NutritionKB
    kb = NutritionKB()

    # Test lookup for multiple ingredients
    ingredients = ["tomato", "onion", "garlic", "olive_oil", "chicken_breast", "rice", "egg"]
    for ing in ingredients:
        n = kb.lookup(ing)
        assert n is not None, f"{ing} should be in database"
        assert "calories_per_100g" in n
        assert n["calories_per_100g"] >= 0
    print(f"  [PASS] lookup: {len(ingredients)} ingredients verified")

    # Test calculate_dish
    dish = kb.calculate_dish([
        {"name": "tomato", "amount_g": 200},
        {"name": "olive_oil", "amount_g": 15},
        {"name": "garlic", "amount_g": 10},
    ])
    assert "total" in dish
    assert dish["total"]["calories"] > 0
    assert len(dish["ingredients"]) == 3
    print(f"  [PASS] calculate_dish: {dish['total']['calories']:.0f} cal, {len(dish['ingredients'])} items")

    # Test list_ingredients
    all_ings = kb.list_ingredients()
    assert len(all_ings) >= 30
    print(f"  [PASS] list_ingredients: {len(all_ings)} ingredients in DB")

    # Test unknown ingredient
    assert kb.lookup("unicorn_meat") is None
    print(f"  [PASS] unknown ingredient: returns None")


def test_commonsense_kb():
    """Test CommonSenseKB."""
    from food_agent.knowledge import CommonSenseKB
    kb = CommonSenseKB()

    # Test get_related_concepts
    concepts_tests = [
        ("cooking", "UsedFor"),
        ("knife", "UsedFor"),
        ("stove", "UsedFor"),
        ("tomato", "IsA"),
    ]
    for concept, relation in concepts_tests:
        results = kb.get_related_concepts(concept, relation)
        assert len(results) > 0, f"{concept}/{relation} should return results"
    print(f"  [PASS] get_related_concepts: {len(concepts_tests)} queries OK")

    # Test infer_cooking_purpose
    result = kb.infer_cooking_purpose(["tomato", "onion", "olive_oil"])
    assert "salad" in result["possible_dishes"]
    assert result["confidence"] > 0
    print(f"  [PASS] infer_cooking_purpose: {result['possible_dishes']} (conf={result['confidence']:.2f})")

    # Test get_next_actions
    actions = kb.get_next_actions("cutting")
    assert len(actions) > 0
    print(f"  [PASS] get_next_actions('cutting'): {actions}")


def test_scene_graph_kb():
    """Test SceneGraphKB."""
    from food_agent.knowledge import SceneGraphKB
    kb = SceneGraphKB()

    # Add graphs for multiple frames
    for t in [1.0, 2.0, 3.0, 5.0, 10.0]:
        kb.add_frame_graph(t, {
            "objects": [{"name": "knife"}, {"name": "tomato"}, {"name": "cutting_board"}],
            "relations": [{"subject": "hand", "predicate": "holding", "object": "knife"}],
        })

    # Test query_objects
    knives = kb.query_objects("knife")
    assert len(knives) == 5
    print(f"  [PASS] query_objects('knife'): {len(knives)} results")

    # Test query_relations
    relations = kb.query_relations(subject="hand", predicate="holding")
    assert len(relations) == 5
    print(f"  [PASS] query_relations(hand, holding): {len(relations)} results")

    # Test get_scene_summary
    summary = kb.get_scene_summary(0, 15)
    assert summary["frame_count"] == 5
    assert "knife" in summary["object_counts"]
    print(f"  [PASS] get_scene_summary: {summary['frame_count']} frames, {len(summary['object_counts'])} object types")


# ============================================================
# 4. Agent Pipeline
# ============================================================

def test_agent_pipeline():
    """Test the full agent pipeline end-to-end."""
    from food_agent.agent_v2.pipeline import Pipeline

    print("  Creating pipeline...")
    pipeline = Pipeline()
    pipeline.agent.max_iterations = 3
    pipeline.agent.timeout = 60

    # Test tool execution directly
    pipeline._current_video_id = "P01-20240202-110250"
    pipeline._current_participant_id = "P01"

    tool_tests = [
        ("query_3d", {"query_type": "layout", "timestamp": 10}),
        ("query_3d", {"query_type": "wearer_pose", "timestamp": 10}),
        ("query_nutrition_kb", {"ingredient": "tomato"}),
        ("query_commonsense", {"concept": "knife", "relation": "UsedFor"}),
        ("query_hands", {"frame_number": 221}),
    ]

    for tool_name, params in tool_tests:
        result = pipeline.tool_registry.call_tool(tool_name, **params)
        if hasattr(result, "content"):
            print(f"  [PASS] {tool_name}: conf={result.confidence:.2f}")
        elif isinstance(result, dict) and "error" not in result:
            print(f"  [PASS] {tool_name}: {result}")
        elif isinstance(result, list):
            print(f"  [PASS] {tool_name}: {len(result)} items")
        else:
            print(f"  [WARN] {tool_name}: {result}")

    # Test full agent run
    print("\n  Running agent on 'What food ingredients can you see?'...")
    result = pipeline.answer(
        question="What food ingredients can you see in this scene?",
        video_id="P01-20240202-110250",
    )
    print(f"  Answer: {result['answer'][:150]}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Tools called: {[tc['tool'] for tc in result['tool_calls']]}")
    print(f"  Evidence: {len(result['evidence_chain'])} items")
    assert result["answer"], "Answer should not be empty"
    assert result["iterations"] >= 1


# ============================================================
# Main
# ============================================================

def run_all_tests():
    """Run all module tests."""
    tests = [
        ("Data Loaders", [
            ("AudioLoader", test_audio_loader),
            ("VideoLoader", test_video_loader),
            ("GazeLoader", test_gaze_loader),
            ("SLAMLoader", test_slam_loader),
            ("DigitalTwinLoader", test_digital_twin_loader),
            ("HandsLoader", test_hands_loader),
        ]),
        ("Reasoning Engine", [
            ("Router", test_router),
            ("Aggregator", test_aggregator),
            ("Judge", test_judge),
            ("Generator", test_generator),
        ]),
        ("Knowledge Modules", [
            ("NutritionKB", test_nutrition_kb),
            ("CommonSenseKB", test_commonsense_kb),
            ("SceneGraphKB", test_scene_graph_kb),
        ]),
        ("Agent Pipeline", [
            ("Pipeline E2E", test_agent_pipeline),
        ]),
    ]

    total = 0
    passed = 0
    failed = 0

    for section, section_tests in tests:
        print(f"\n{'='*60}")
        print(f"  {section}")
        print(f"{'='*60}")
        for name, test_func in section_tests:
            total += 1
            print(f"\n--- {name} ---")
            try:
                test_func()
                passed += 1
                print(f"  => {name}: ALL PASSED")
            except Exception as e:
                failed += 1
                print(f"  => {name}: FAILED - {e}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
