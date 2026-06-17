"""Visual analyzer: SAM3 + Grounding DINO + MiMo2.5 for kitchen scene understanding.

Pipeline:
    1. SAM3 / Grounding DINO detect and segment objects
    2. MiMo2.5 API generates scene graph and action descriptions
    3. All outputs unified as Evidence
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .evidence import Evidence


class VisualAnalyzer:
    """Analyze video frames for scene understanding, object detection, and action recognition.

    Supports three modes:
        - SAM3: open-vocabulary segmentation (fastest, gives masks)
        - Grounding DINO: open-vocabulary detection (gives bboxes)
        - MiMo2.5 API: scene graph generation (gives descriptions)
    """

    def __init__(
        self,
        grounding_dino_model=None,
        sam2_model=None,
        mimo_client=None,
        sam3_segmentor=None,
    ):
        self._gdino = grounding_dino_model
        self._sam2 = sam2_model
        self._mimo = mimo_client
        self._sam3 = sam3_segmentor

    def detect_objects(
        self,
        frame: np.ndarray,
        text_prompt: str = "food ingredient",
        method: str = "auto",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Dict]:
        """Detect objects in a frame.

        Args:
            frame: BGR image (H, W, 3).
            text_prompt: What to detect.
            method: 'sam3', 'gdino', 'mimo', or 'auto' (tries sam3 -> gdino -> mimo).
            box_threshold: Detection threshold for GDino.
            text_threshold: Text matching threshold for GDino.

        Returns:
            List of dicts with bbox, label, score, mask (optional).
        """
        if method == "auto":
            if self._sam3 is not None:
                method = "sam3"
            elif self._gdino is not None:
                method = "gdino"
            else:
                method = "mimo"

        if method == "sam3":
            return self._detect_sam3(frame, text_prompt)
        elif method == "gdino":
            return self._detect_gdino(frame, text_prompt, box_threshold, text_threshold)
        else:
            return self._detect_mimo(frame, text_prompt)

    def _detect_sam3(self, frame: np.ndarray, text_prompt: str) -> List[Dict]:
        """Detect objects using SAM3 open-vocabulary segmentation."""
        objects = self._sam3.detect_objects(frame, text_prompt, threshold=0.1)
        results = []
        for obj in objects:
            if obj["score"] > 0.2:
                results.append({
                    "bbox": obj["bbox"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "mask": obj["mask"],
                    "area": obj["area"],
                })
        return results

    def _detect_gdino(
        self, frame: np.ndarray, text_prompt: str,
        box_threshold: float, text_threshold: float,
    ) -> List[Dict]:
        """Detect objects using Grounding DINO."""
        import torch
        from groundingdino.util.inference import predict

        image_rgb = frame[:, :, ::-1].copy()
        h, w = frame.shape[:2]

        boxes, logits, phrases = predict(
            model=self._gdino,
            image=image_rgb,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections = []
        for i in range(len(boxes)):
            cx, cy, bw, bh = boxes[i].cpu().numpy()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": phrases[i] if i < len(phrases) else "unknown",
                "score": float(logits[i]),
            })
        return detections

    def _detect_mimo(self, frame: np.ndarray, text_prompt: str) -> List[Dict]:
        """Detect objects using MiMo2.5 API."""
        if self._mimo is None:
            return []
        prompt = (
            f"Detect all objects matching '{text_prompt}' in this kitchen scene.\n"
            "Return a JSON array with each object having 'label' (string) and 'bbox' [x1,y1,x2,y2].\n"
            "Example: [{\"label\": \"tomato\", \"bbox\": [100, 200, 300, 400]}]"
        )
        response = self._mimo.call_vision(frame, prompt)
        try:
            import json
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                detections = json.loads(response[start:end])
                for d in detections:
                    d.setdefault("score", 0.5)
                return detections
        except Exception:
            pass
        return []

    def generate_scene_graph(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate a scene graph using MiMo2.5 API."""
        if self._mimo is None:
            return {"objects": [], "relations": [], "scene_description": ""}

        det_text = ""
        if detections:
            det_text = "Detected objects: " + ", ".join(
                f"{d['label']}({d.get('score', 0):.2f})" for d in detections[:10]
            )

        prompt = (
            "Analyze this kitchen scene image. Return a JSON object with:\n"
            'objects: list of {name, attributes}\n'
            'relations: list of {subject, predicate, object}\n'
            'scene_description: one sentence\n'
            f"{det_text}"
        )

        response = self._mimo.call_vision(frame, prompt)
        try:
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass
        return {"objects": [], "relations": [], "scene_description": response[:200]}

    def analyze_action(
        self,
        frame: np.ndarray,
        context: str = "",
    ) -> Dict:
        """Analyze action in a single frame using MiMo2.5 API."""
        if self._mimo is None:
            return {"action": "unknown", "confidence": 0, "description": ""}

        prompt = (
            "What action is being performed in this kitchen scene? "
            "Return JSON with action, confidence (0-1), description."
        )
        if context:
            prompt += f"\nContext: {context}"

        response = self._mimo.call_vision(frame, prompt)
        try:
            import json
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass
        return {"action": "unknown", "confidence": 0, "description": response[:200]}

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        text_prompt: str = "food ingredient",
        use_sam3: bool = True,
        use_scene_graph: bool = False,
    ) -> Evidence:
        """Full frame analysis: detect objects + optional scene graph.

        Args:
            frame: BGR image (H, W, 3).
            timestamp: Frame timestamp in seconds.
            text_prompt: What to detect.
            use_sam3: Use SAM3 for detection (preferred).
            use_scene_graph: Also generate scene graph via MiMo.

        Returns:
            Evidence with detection results.
        """
        method = "sam3" if (use_sam3 and self._sam3) else "auto"
        detections = self.detect_objects(frame, text_prompt, method=method)

        content = {
            "detections": [
                {"label": d["label"], "score": d.get("score", 0), "bbox": d["bbox"]}
                for d in detections
            ],
            "detection_count": len(detections),
            "method": method,
        }

        if use_scene_graph and self._mimo:
            scene = self.generate_scene_graph(frame, detections)
            content.update({
                "scene_description": scene.get("scene_description", ""),
                "objects": scene.get("objects", []),
                "relations": scene.get("relations", []),
            })

        confidence = max((d.get("score", 0) for d in detections), default=0.0)
        if confidence == 0 and detections:
            confidence = 0.5

        return Evidence(
            source_module="VisualAnalyzer",
            evidence_type="visual",
            time_range={"start": timestamp, "end": timestamp},
            content=content,
            confidence=min(1.0, confidence),
        )
