"""Gaze tracker: map gaze points to objects and generate attention heatmaps."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .evidence import Evidence


class GazeTracker:
    """Track gaze targets and generate attention analysis.

    Uses Grounding DINO to identify what the wearer is looking at,
    and numpy/scipy for fixation detection and heatmap generation.
    """

    def __init__(self, gaze_loader, grounding_dino_model=None):
        self._gaze_loader = gaze_loader
        self._gdino = grounding_dino_model

    def identify_gaze_target(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[float, float],
        crop_size: int = 200,
        text_prompt: str = "knife. cutting board. tomato. pan. plate. pot. spoon. food. hand. sink. stove. bowl.",
    ) -> Dict:
        """Identify what the wearer is looking at.

        Crops a region around the gaze point and runs object detection.

        Args:
            frame: BGR image (H, W, 3).
            gaze_point: (x, y) pixel coordinates of gaze.
            crop_size: Size of the crop region around gaze point.
            text_prompt: Objects to detect.

        Returns:
            Dict with target_name, confidence, bbox, position_in_frame.
        """
        h, w = frame.shape[:2]
        gx, gy = int(gaze_point[0]), int(gaze_point[1])

        # Clamp to image bounds
        gx = max(0, min(gx, w - 1))
        gy = max(0, min(gy, h - 1))

        # Crop region around gaze point
        half = crop_size // 2
        x1 = max(0, gx - half)
        y1 = max(0, gy - half)
        x2 = min(w, gx + half)
        y2 = min(h, gy + half)
        crop = frame[y1:y2, x1:x2]

        if self._gdino is not None:
            from food_agent.perception.visual_analyzer import VisualAnalyzer
            va = VisualAnalyzer(grounding_dino_model=self._gdino)
            detections = va.detect_objects(crop, text_prompt)

            if detections:
                # Find the detection whose center is closest to gaze point in crop
                crop_cx, crop_cy = (gx - x1), (gy - y1)
                best = None
                best_dist = float("inf")
                for det in detections:
                    bx1, by1, bx2, by2 = det["bbox"]
                    dcx = (bx1 + bx2) / 2 - crop_cx
                    dcy = (by1 + by2) / 2 - crop_cy
                    dist = (dcx ** 2 + dcy ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best = det

                return {
                    "target_name": best["label"],
                    "confidence": best["confidence"],
                    "bbox": best["bbox"],
                    "position_in_frame": [gx, gy],
                }

        return {
            "target_name": "unknown",
            "confidence": 0.0,
            "bbox": None,
            "position_in_frame": [gx, gy],
        }

    def generate_attention_heatmap(
        self,
        gaze_points: List[Tuple[float, float]],
        frame_size: Tuple[int, int] = (1920, 1080),
        sigma: float = 30,
    ) -> np.ndarray:
        """Generate an attention heatmap from gaze points.

        Args:
            gaze_points: List of (x, y) pixel coordinates.
            frame_size: (width, height) of the frame.
            sigma: Gaussian blur sigma.

        Returns:
            Normalized heatmap array (H, W) with values in [0, 1].
        """
        w, h = frame_size
        heatmap = np.zeros((h, w), dtype=np.float64)

        for x, y in gaze_points:
            ix, iy = int(x), int(y)
            if 0 <= ix < w and 0 <= iy < h:
                heatmap[iy, ix] += 1

        if heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            heatmap = heatmap / heatmap.max()

        return heatmap

    def get_fixation_targets(
        self,
        participant_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
        min_duration: float = 0.2,
    ) -> List[Evidence]:
        """Get gaze fixation events and their targets in a time range.

        Returns:
            List of Evidence objects, one per fixation.
        """
        fixations = self._gaze_loader.get_fixations(
            participant_id, video_id, min_duration
        )

        evidence_list = []
        for fix in fixations:
            if fix["end_time"] < start_time or fix["start_time"] > end_time:
                continue

            evidence_list.append(Evidence(
                source_module="GazeTracker",
                evidence_type="gaze",
                time_range={"start": fix["start_time"], "end": fix["end_time"]},
                content={
                    "type": "fixation",
                    "duration": fix["duration"],
                    "mean_yaw": fix["mean_yaw"],
                    "mean_pitch": fix["mean_pitch"],
                },
                confidence=min(1.0, fix["duration"] / 2.0),
            ))

        return evidence_list
