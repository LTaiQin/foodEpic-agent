#!/usr/bin/env python3
"""Test SAM3 segmentation quality on HD-EPIC kitchen scenes."""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_sam3_multiframe():
    """Test SAM3 on multiple frames with kitchen-specific prompts."""
    from food_agent.perception.sam3_wrapper import SAM3Segmentor

    print("Loading SAM3...")
    sam3 = SAM3Segmentor("/22liushoulong/sam-weight/")

    cap = cv2.VideoCapture("/22liushoulong/agent/hd-epic/data/HD-EPIC/Videos/P01/P01-20240202-110250.mp4")

    prompts = [
        "food ingredient",
        "kitchen object",
        "cooking utensil",
        "food",
        "fruit",
    ]

    timestamps = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

    print(f"\nTesting {len(timestamps)} frames with {len(prompts)} prompts...")
    print(f"{'t(s)':>5} | {'prompt':>20} | {'total':>5} | {'high_conf':>9} | {'avg_score':>10}")
    print("-" * 65)

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        for prompt in prompts:
            objects = sam3.detect_objects(frame, prompt, threshold=0.1)
            high_conf = [o for o in objects if o["score"] > 0.3]
            avg_score = np.mean([o["score"] for o in objects]) if objects else 0

            print(f"{t:>5} | {prompt:>20} | {len(objects):>5} | {len(high_conf):>9} | {avg_score:>10.3f}")

    cap.release()

    # Summary
    print("\n--- Summary ---")
    cap = cv2.VideoCapture("/22liushoulong/agent/hd-epic/data/HD-EPIC/Videos/P01/P01-20240202-110250.mp4")
    total_detections = 0
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        # Use combined prompt for best coverage
        objects = sam3.detect_objects(frame, "kitchen object food", threshold=0.1)
        high_conf = [o for o in objects if o["score"] > 0.3]
        total_detections += len(high_conf)
        print(f"t={t}s: {len(high_conf)} high-confidence objects")
    cap.release()

    print(f"\nTotal high-confidence detections across {len(timestamps)} frames: {total_detections}")
    print(f"Average: {total_detections/len(timestamps):.1f} objects/frame")


def test_sam3_vs_mimo():
    """Compare SAM3 detection vs MiMo2.5 API detection."""
    from food_agent.perception.sam3_wrapper import SAM3Segmentor
    from food_agent.evaluation.api_client import MimoClient

    print("\nComparing SAM3 vs MiMo2.5 API...")

    sam3 = SAM3Segmentor("/22liushoulong/sam-weight/")
    mimo = MimoClient()

    cap = cv2.VideoCapture("/22liushoulong/agent/hd-epic/data/HD-EPIC/Videos/P01/P01-20240202-110250.mp4")
    cap.set(cv2.CAP_PROP_POS_MSEC, 60000)
    ret, frame = cap.read()
    cap.release()

    # SAM3 detection
    import time
    t0 = time.time()
    sam3_objects = sam3.detect_objects(frame, "kitchen object food", threshold=0.1)
    sam3_time = time.time() - t0
    sam3_high = [o for o in sam3_objects if o["score"] > 0.3]

    # MiMo API detection
    t0 = time.time()
    prompt = "List all visible objects in this kitchen scene. Return a JSON array with 'label' for each."
    mimo_response = mimo.call_vision(frame, prompt)
    mimo_time = time.time() - t0

    print(f"SAM3: {len(sam3_high)} objects in {sam3_time:.1f}s")
    for obj in sam3_high[:5]:
        print(f"  score={obj['score']:.3f}, area={obj['area']}px, bbox={obj['bbox']}")

    print(f"\nMiMo2.5: response in {mimo_time:.1f}s")
    print(f"  {mimo_response[:300]}")


if __name__ == "__main__":
    test_sam3_multiframe()
    test_sam3_vs_mimo()
