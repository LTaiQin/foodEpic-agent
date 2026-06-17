"""Hand-object interaction analysis using hand masks and object detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .evidence import Evidence

# Contact-object → action mapping
CONTACT_ACTION_MAP = {
    ("hand", "knife", "horizontal"): "cutting",
    ("hand", "knife", "vertical"): "chopping",
    ("hand", "spoon", "circular"): "stirring",
    ("hand", "pan", "upward"): "lifting_pan",
    ("hand", "pan_handle", "tilting"): "pouring_from_pan",
    ("hand", "lid", "upward"): "removing_lid",
    ("hand", "bottle", "tilting"): "pouring",
    ("hand", "vegetable", "downward"): "placing_in_pan",
    ("hand", "mixing_bowl", "circular"): "mixing",
    ("hand", "fork", "downward"): "piercing",
    ("hand", "spatula", "horizontal"): "flipping",
}


class HandInteractor:
    """Analyze hand-object interactions from masks and visual detection.

    Pipeline:
        1. Load hand mask from HandsLoader
        2. Expand hand region and detect objects with Grounding DINO
        3. Check overlap between hand mask and detected objects
        4. Infer action from contact object + motion direction + audio hint
    """

    def __init__(self, hands_loader, grounding_dino_model=None, sam2_model=None):
        self._hands = hands_loader
        self._gdino = grounding_dino_model
        self._sam2 = sam2_model

    def detect_contact_object(
        self,
        frame: np.ndarray,
        hand_mask: np.ndarray,
        text_prompt: str = "knife. spoon. fork. pan. pot. bowl. plate. cup. lid. bottle. spatula. food. vegetable. cutting board.",
        expand_ratio: float = 1.5,
    ) -> Dict:
        """Detect what object the hand is in contact with.

        Args:
            frame: BGR image (H, W, 3).
            hand_mask: Binary hand mask (H, W).
            text_prompt: Objects to detect.
            expand_ratio: How much to expand the hand region for detection.

        Returns:
            Dict with object_name, interaction_type, bbox, overlap_ratio.
        """
        if hand_mask is None or hand_mask.sum() == 0:
            return {"object_name": "none", "interaction_type": "no_contact", "bbox": None, "overlap_ratio": 0}

        h, w = hand_mask.shape
        # Get hand bounding box from mask
        ys, xs = np.where(hand_mask > 0)
        if len(ys) == 0:
            return {"object_name": "none", "interaction_type": "no_contact", "bbox": None, "overlap_ratio": 0}

        hand_x1, hand_x2 = int(xs.min()), int(xs.max())
        hand_y1, hand_y2 = int(ys.min()), int(ys.max())

        # Expand region
        bw = hand_x2 - hand_x1
        bh = hand_y2 - hand_y1
        expand_x = int(bw * (expand_ratio - 1) / 2)
        expand_y = int(bh * (expand_ratio - 1) / 2)
        roi_x1 = max(0, hand_x1 - expand_x)
        roi_y1 = max(0, hand_y1 - expand_y)
        roi_x2 = min(w, hand_x2 + expand_x)
        roi_y2 = min(h, hand_y2 + expand_y)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Detect objects in the expanded region
        detections = []
        if self._gdino is not None:
            from food_agent.perception.visual_analyzer import VisualAnalyzer
            va = VisualAnalyzer(grounding_dino_model=self._gdino)
            detections = va.detect_objects(roi, text_prompt)

        if not detections:
            return {"object_name": "unknown", "interaction_type": "possible_contact", "bbox": None, "overlap_ratio": 0}

        # Check overlap between hand mask and each detection
        best_obj = None
        best_overlap = 0

        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            # Convert from ROI coordinates to full frame coordinates
            fx1 = bx1 + roi_x1
            fy1 = by1 + roi_y1
            fx2 = bx2 + roi_x1
            fy2 = by2 + roi_y1

            # Check overlap with hand mask
            fx1 = max(0, min(fx1, w - 1))
            fx2 = max(0, min(fx2, w))
            fy1 = max(0, min(fy1, h - 1))
            fy2 = max(0, min(fy2, h))

            region = hand_mask[fy1:fy2, fx1:fx2]
            if region.size == 0:
                continue
            overlap = region.sum() / region.size

            if overlap > best_overlap:
                best_overlap = overlap
                best_obj = det

        if best_obj is None or best_overlap < 0.05:
            return {"object_name": "none", "interaction_type": "no_contact", "bbox": None, "overlap_ratio": 0}

        return {
            "object_name": best_obj["label"],
            "interaction_type": "grasping" if best_overlap > 0.2 else "near_contact",
            "bbox": best_obj["bbox"],
            "overlap_ratio": float(best_overlap),
        }

    def infer_action(
        self,
        contact_object: str,
        hand_motion: str = "unknown",
        audio_hint: str = "",
    ) -> Dict:
        """Infer the action being performed based on contact and context.

        Args:
            contact_object: Name of the object in contact.
            hand_motion: Motion direction (horizontal, vertical, circular, etc.).
            audio_hint: Audio event hint (e.g. "chopping", "frying").

        Returns:
            Dict with action, confidence, reasoning.
        """
        # Try exact match
        key = ("hand", contact_object.lower().replace(" ", "_"), hand_motion)
        if key in CONTACT_ACTION_MAP:
            return {
                "action": CONTACT_ACTION_MAP[key],
                "confidence": 0.85,
                "reasoning": f"Contact with {contact_object}, motion={hand_motion}",
            }

        # Fallback: use audio hint to disambiguate
        if audio_hint:
            for (h, obj, motion), action in CONTACT_ACTION_MAP.items():
                if obj in contact_object.lower() and action.startswith(audio_hint[:4]):
                    return {
                        "action": action,
                        "confidence": 0.7,
                        "reasoning": f"Contact with {contact_object}, audio hint={audio_hint}",
                    }

        # Generic fallback
        return {
            "action": f"handling_{contact_object}",
            "confidence": 0.4,
            "reasoning": f"Contact with {contact_object}, motion unknown",
        }

    def get_hand_interactions(
        self,
        participant_id: str,
        video_id: str,
        frame_number: int,
        fps: float = 30.0,
    ) -> Evidence:
        """Analyze hand interactions for a single frame.

        Returns:
            Evidence with hand interaction data.
        """
        timestamp = frame_number / fps

        left_mask = self._hands.get_mask(video_id, frame_number, "left")
        right_mask = self._hands.get_mask(video_id, frame_number, "right")

        interactions = []
        for hand, mask in [("left", left_mask), ("right", right_mask)]:
            if mask is not None and mask.sum() > 0:
                interactions.append({
                    "hand": hand,
                    "has_contact": self._hands.has_hand_contact(mask),
                    "mask_area": int(mask.sum()),
                })

        return Evidence(
            source_module="HandInteractor",
            evidence_type="hand",
            time_range={"start": timestamp, "end": timestamp},
            content={
                "interactions": interactions,
                "frame_number": frame_number,
            },
            confidence=0.8 if interactions else 0.0,
        )
