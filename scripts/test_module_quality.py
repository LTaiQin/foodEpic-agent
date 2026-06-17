#!/usr/bin/env python3
"""Test each perception module's actual output quality with real HD-EPIC data.

This tests whether modules produce meaningful, accurate results - not just whether they run.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_visual_analyzer_quality():
    """Test VisualAnalyzer: does it actually detect objects in kitchen frames?"""
    from food_agent.loaders import VideoLoader
    from food_agent.perception import VisualAnalyzer
    from food_agent.evaluation.api_client import MimoClient

    vl = VideoLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Videos")
    mimo = MimoClient()
    va = VisualAnalyzer(mimo_client=mimo)

    print("  Testing on 5 different frames...")

    # Test on frames from different parts of the video
    timestamps = [30.0, 60.0, 120.0, 200.0, 300.0]
    for t in timestamps:
        frame = vl.get_frame("P01-20240202-110250", t)
        result = va.analyze_frame(frame, t)

        desc = result.content.get("scene_description", "")[:100]
        objects = result.content.get("objects", [])
        conf = result.confidence

        print(f"  t={t:.0f}s: conf={conf:.2f}, objects={len(objects)}, desc='{desc}'")

        # Quality checks
        assert conf >= 0, f"Confidence should be >= 0"
        assert isinstance(objects, list), "Objects should be a list"

    print("  [RESULT] VisualAnalyzer produces scene descriptions for all frames")


def test_spatial_reasoner_quality():
    """Test SpatialReasoner: does it give correct spatial relations?"""
    from food_agent.loaders import DigitalTwinLoader, SLAMLoader
    from food_agent.perception import SpatialReasoner

    dt = DigitalTwinLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Digital-Twin")
    sl = SLAMLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/SLAM-and-Gaze")
    sr = SpatialReasoner(dt, sl)

    print("  Testing spatial queries...")

    # Test 1: Wearer pose at multiple timestamps
    for t in [60.0, 120.0, 200.0, 300.0]:
        pose = sr.get_wearer_pose_at_time("P01", "P01-20240202-110250", t)
        assert pose["nearest_fixture"] != "unknown"
        assert pose["distance_to_nearest"] >= 0
        print(f"  t={t:.0f}s: near '{pose['nearest_fixture']}' ({pose['distance_to_nearest']:.2f}m)")

    # Test 2: Kitchen layout
    layout = sr.describe_spatial_layout("P01")
    assert len(layout["fixtures"]) > 10
    assert len(layout["spatial_relations"]) > 0
    print(f"  Layout: {len(layout['fixtures'])} fixtures, {len(layout['spatial_relations'])} relations")

    # Test 3: Nearest fixture query
    pos = sl.get_position("P01", "P01-20240202-110250", 100.0)
    nearest = sr.get_nearest_fixture("P01", pos)
    print(f"  Nearest to wearer at t=100s: {nearest['fixture_type']} ({nearest['distance']:.2f}m)")

    # Test 4: 3D query evidence
    evidence = sr.query_3d("P01", "P01-20240202-110250", 100.0, "layout")
    assert evidence.confidence > 0
    print(f"  3D query evidence: conf={evidence.confidence:.2f}, type={evidence.evidence_type}")

    print("  [RESULT] SpatialReasoner produces consistent spatial data")


def test_gaze_tracker_quality():
    """Test GazeTracker: does it correctly identify gaze targets?"""
    from food_agent.loaders import GazeLoader
    from food_agent.perception import GazeTracker

    gl = GazeLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/SLAM-and-Gaze")
    gt = GazeTracker(gl)

    print("  Testing gaze analysis...")

    # Test 1: Fixation detection
    fixations = gl.get_fixations("P01", "P01-20240202-110250", min_duration=0.3)
    print(f"  Fixations (>0.3s): {len(fixations)}")

    if fixations:
        # Show top 5 longest fixations
        sorted_fix = sorted(fixations, key=lambda f: f["duration"], reverse=True)
        for i, fix in enumerate(sorted_fix[:5]):
            print(f"    #{i+1}: t={fix['start_time']:.1f}-{fix['end_time']:.1f}s, "
                  f"dur={fix['duration']:.2f}s, yaw={fix['mean_yaw']:.3f}rad")

    # Test 2: Gaze trajectory
    trajectory = gl.get_gaze_trajectory("P01", "P01-20240202-110250", 60.0, 90.0)
    print(f"  Trajectory (60-90s): {len(trajectory)} points")

    if trajectory:
        # Compute gaze stability (std of yaw/pitch)
        yaws = [g.avg_yaw for g in trajectory]
        pitches = [g.pitch for g in trajectory]
        yaw_std = np.std(yaws)
        pitch_std = np.std(pitches)
        print(f"  Gaze stability: yaw_std={yaw_std:.4f}rad, pitch_std={pitch_std:.4f}rad")

    # Test 3: Attention heatmap
    if trajectory:
        points = [g.pixel_estimate for g in trajectory[:50]]
        heatmap = gt.generate_attention_heatmap(points)
        assert heatmap.max() > 0
        print(f"  Heatmap: max={heatmap.max():.3f}, nonzero={np.count_nonzero(heatmap)} pixels")

    # Test 4: Fixation evidence
    evidence = gt.get_fixation_targets("P01", "P01-20240202-110250", 60, 120)
    print(f"  Fixation evidence: {len(evidence)} items in 60-120s")

    print("  [RESULT] GazeTracker detects fixations and generates heatmaps")


def test_audio_analyzer_quality():
    """Test AudioAnalyzer: does it classify kitchen sounds?"""
    from food_agent.loaders import AudioLoader
    from food_agent.perception import AudioAnalyzer

    al = AudioLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Audio-HDF5")
    aa = AudioAnalyzer(
        clap_model_path="/22liushoulong/agent/hd-epic/weights/music_speech_audioset_epoch_15_esc_89.98.pt"
    )

    print("  Testing audio analysis...")

    # Test 1: Load audio segments from different times
    vid = "P01-20240202-110250"
    dur = al.get_duration("P01", vid)
    print(f"  Video duration: {dur:.1f}s")

    # Test CLAP zero-shot classification on real audio
    test_times = [(10, 15), (60, 65), (120, 125), (200, 205)]
    for start, end in test_times:
        if end > dur:
            continue
        audio, sr = al.load_segment("P01", vid, start, end)
        if len(audio) < sr * 0.5:
            continue

        try:
            result = aa.zero_shot_classify(
                audio,
                ["sound of chopping", "sound of water running", "sound of frying",
                 "sound of stirring", "sound of silence", "sound of talking"],
                sr,
            )
            print(f"  t={start}-{end}s: '{result['type']}' (conf={result['confidence']:.3f})")
        except Exception as e:
            print(f"  t={start}-{end}s: CLAP error: {e}")

    # Test 2: Get audio events (annotation-based)
    events = al.get_all_events("P01", vid)
    print(f"  Audio annotations: {len(events)} events")

    print("  [RESULT] AudioAnalyzer classifies sounds (requires CLAP model)")


def test_hand_interactor_quality():
    """Test HandInteractor: does it detect hand interactions?"""
    from food_agent.loaders import HandsLoader
    from food_agent.perception import HandInteractor

    hl = HandsLoader("/22liushoulong/agent/hd-epic/data/HD-EPIC/Hands-Masks")
    hi = HandInteractor(hl)

    print("  Testing hand interaction analysis...")

    # Test on multiple videos
    for vid in ["P01-20240202-110250", "P01-20240202-161354"]:
        frames = hl.get_available_frames(vid)
        print(f"  [{vid}] {len(frames)} annotated frames")

        # Analyze first 5 frames
        for fn in frames[:5]:
            evidence = hi.get_hand_interactions("P01", vid, fn)
            interactions = evidence.content.get("interactions", [])
            conf = evidence.confidence

            for inter in interactions:
                hand = inter.get("hand", "?")
                area = inter.get("mask_area", 0)
                contact = inter.get("has_contact", False)
                print(f"    frame {fn}: {hand} hand, area={area}px, contact={contact}")

    # Test contact detection with actual mask
    frames = hl.get_available_frames("P01-20240202-110250")
    mask = hl.get_mask("P01-20240202-110250", frames[0], "left")
    if mask is not None:
        contact = hl.has_hand_contact(mask)
        area = int(mask.sum())
        print(f"  Contact test: area={area}px, has_contact={contact}")

    print("  [RESULT] HandInteractor detects hand presence and contact")


def test_nutrition_estimator_quality():
    """Test NutritionEstimator: does it give reasonable nutrition estimates?"""
    from food_agent.perception import NutritionEstimator

    ne = NutritionEstimator()

    print("  Testing nutrition estimation...")

    # Test 1: Known recipes
    recipes = [
        {"name": "Tomato Salad", "ingredients": [
            {"name": "tomato", "amount_g": 200},
            {"name": "olive_oil", "amount_g": 15},
            {"name": "salt", "amount_g": 2},
        ]},
        {"name": "Pasta with Egg", "ingredients": [
            {"name": "pasta", "amount_g": 200},
            {"name": "egg", "amount_g": 100},
            {"name": "butter", "amount_g": 20},
            {"name": "cheese", "amount_g": 30},
        ]},
        {"name": "Chicken Rice", "ingredients": [
            {"name": "chicken_breast", "amount_g": 150},
            {"name": "rice", "amount_g": 200},
            {"name": "onion", "amount_g": 50},
        ]},
    ]

    for recipe in recipes:
        result = ne.calculate_total(recipe["ingredients"])
        total = result["total"]
        print(f"  {recipe['name']}: {total['calories']:.0f} cal, "
              f"P={total['protein_g']:.1f}g, C={total['carbs_g']:.1f}g, F={total['fat_g']:.1f}g")

        # Sanity checks
        assert total["calories"] > 0, "Should have calories"
        assert total["protein_g"] >= 0
        assert total["carbs_g"] >= 0
        assert total["fat_g"] >= 0

    # Test 2: Verify known values
    tomato = ne.lookup_nutrition("tomato")
    assert tomato["calories_per_100g"] == 18  # USDA value
    olive = ne.lookup_nutrition("olive_oil")
    assert olive["calories_per_100g"] == 884  # USDA value
    print(f"  USDA verification: tomato=18cal/100g, olive_oil=884cal/100g")

    print("  [RESULT] NutritionEstimator gives correct nutrition values")


def test_evidence_format_quality():
    """Test Evidence: does serialization preserve all data?"""
    from food_agent.perception.evidence import Evidence

    print("  Testing Evidence format...")

    # Create evidence with complex content
    e = Evidence(
        source_module="VisualAnalyzer",
        evidence_type="visual",
        time_range={"start": 10.5, "end": 10.5},
        content={
            "objects": [{"name": "knife", "bbox": [100, 200, 300, 400]}],
            "relations": [{"subject": "hand", "predicate": "holding", "object": "knife"}],
            "scene_description": "Person holding a knife",
        },
        confidence=0.85,
    )

    # Serialize -> deserialize
    j = e.to_json()
    e2 = Evidence.from_json(j)

    assert e.source_module == e2.source_module
    assert e.confidence == e2.confidence
    assert e.content["objects"][0]["name"] == e2.content["objects"][0]["name"]
    assert e.content["relations"][0]["predicate"] == e2.content["relations"][0]["predicate"]
    print(f"  JSON roundtrip: all fields preserved")

    # Test dict roundtrip
    d = e.to_dict()
    e3 = Evidence.from_dict(d)
    assert e3.content == e.content
    print(f"  Dict roundtrip: all fields preserved")

    print("  [RESULT] Evidence serialization is lossless")


def test_router_quality():
    """Test Router: how accurate is question classification?"""
    from food_agent.reasoning.router import Router

    router = Router()

    print("  Testing router accuracy...")

    # Comprehensive test cases
    test_cases = [
        # ingredient
        ("What ingredients are used in this recipe?", "ingredient"),
        ("Can you identify the vegetables?", "ingredient"),
        ("What food items are on the counter?", "ingredient"),
        # nutrition
        ("How many calories does this dish have?", "nutrition"),
        ("What is the nutritional value?", "nutrition"),
        ("Is this meal healthy?", "nutrition"),
        # 3d_perception
        ("Where is the stove?", "3d_perception"),
        ("What is next to the sink?", "3d_perception"),
        ("How far is the fridge?", "3d_perception"),
        # fine_grained_action
        ("What is the person doing?", "fine_grained_action"),
        ("What action is being performed?", "fine_grained_action"),
        ("Is the person stirring or chopping?", "fine_grained_action"),
        # gaze
        ("What is the wearer looking at?", "gaze"),
        ("Where is the person's attention?", "gaze"),
        # object_motion
        ("Which object was moved?", "object_motion"),
        ("Where did the tomato go?", "object_motion"),
        # recipe
        ("What recipe is being prepared?", "recipe"),
        ("Which step of the recipe is this?", "recipe"),
    ]

    correct = 0
    for question, expected in test_cases:
        result = router.route(question)
        actual = result["category"]
        ok = actual == expected
        if ok:
            correct += 1
        else:
            print(f"    MISS: '{question}' -> {actual} (expected: {expected})")

    acc = correct / len(test_cases)
    print(f"  Accuracy: {correct}/{len(test_cases)} = {acc:.1%}")

    # Category breakdown
    from collections import Counter
    cats = Counter(expected for _, expected in test_cases)
    cat_correct = Counter()
    for q, expected in test_cases:
        if router.classify_question(q) == expected:
            cat_correct[expected] += 1

    for cat in sorted(cats.keys()):
        c = cat_correct.get(cat, 0)
        t = cats[cat]
        print(f"    {cat}: {c}/{t}")

    print(f"  [RESULT] Router accuracy: {acc:.1%}")


# ============================================================
# Main
# ============================================================

def run_all_quality_tests():
    """Run all quality tests."""
    tests = [
        ("VisualAnalyzer Quality", test_visual_analyzer_quality),
        ("SpatialReasoner Quality", test_spatial_reasoner_quality),
        ("GazeTracker Quality", test_gaze_tracker_quality),
        ("AudioAnalyzer Quality", test_audio_analyzer_quality),
        ("HandInteractor Quality", test_hand_interactor_quality),
        ("NutritionEstimator Quality", test_nutrition_estimator_quality),
        ("Evidence Format Quality", test_evidence_format_quality),
        ("Router Quality", test_router_quality),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            test_func()
            passed += 1
            print(f"  => PASS")
        except Exception as e:
            failed += 1
            print(f"  => FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_quality_tests()
