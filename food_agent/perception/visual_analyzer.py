"""Visual analyzer using MiMo2.5 API, Grounding DINO, and SAM 2.1."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .evidence import Evidence


class VisualAnalyzer:
    """Analyze video frames for scene understanding, object detection, and action recognition.

    Pipeline:
        1. Grounding DINO detects objects (open-vocabulary)
        2. SAM 2.1 segments detected objects
        3. MiMo2.5 API generates scene graphs and action descriptions
    """

    def __init__(
        self,
        grounding_dino_model=None,
        sam2_model=None,
        mimo_client=None,
    ):
        self._gdino = grounding_dino_model
        self._sam2 = sam2_model
        self._mimo = mimo_client

    def detect_objects(
        self,
        frame: np.ndarray,
        text_prompt: str = "knife. cutting board. tomato. pan. plate. pot. spoon. fork. bowl. cup. food. hand.",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Dict]:
        """Detect objects in a frame using Grounding DINO.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            text_prompt: Period-separated object descriptions.
            box_threshold: Detection confidence threshold.
            text_threshold: Text matching threshold.

        Returns:
            List of dicts with keys: bbox (x1,y1,x2,y2), label, confidence.
        """
        if self._gdino is None:
            return self._detect_objects_fallback(frame, text_prompt)

        import torch
        from groundingdino.util.inference import load_image, predict

        # Convert BGR to RGB, then to tensor
        image_rgb = frame[:, :, ::-1].copy()
        h, w = frame.shape[:2]

        # Use groundingdino predict
        boxes, logits, phrases = predict(
            model=self._gdino,
            image=image_rgb,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert from normalized (cx, cy, w, h) to pixel (x1, y1, x2, y2)
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
                "confidence": float(logits[i]),
            })

        return detections

    def _detect_objects_fallback(self, frame: np.ndarray, text_prompt: str) -> List[Dict]:
        """Fallback: use MiMo2.5 API for object detection."""
        if self._mimo is None:
            return []
        # Use API-based detection
        prompt = (
            f"Detect the following objects in this image: {text_prompt}\n"
            "Return a JSON array of objects, each with 'label' and 'bbox' [x1,y1,x2,y2].\n"
            "Example: [{\"label\": \"knife\", \"bbox\": [100, 200, 300, 400]}]"
        )
        response = self._mimo.call_vision(frame, prompt)
        # Parse response (best effort)
        try:
            import json
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                detections = json.loads(response[start:end])
                # Ensure all detections have 'confidence' key
                for d in detections:
                    d.setdefault("confidence", 0.5)
                return detections
        except Exception:
            pass
        return []

    def segment_objects(
        self,
        frame: np.ndarray,
        boxes: List[List[int]],
    ) -> List[np.ndarray]:
        """Segment objects using SAM 2.1 with bounding box prompts.

        Args:
            frame: BGR image (H, W, 3).
            boxes: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            List of binary masks, each (H, W) uint8.
        """
        if self._sam2 is None or not boxes:
            return [np.zeros(frame.shape[:2], dtype=np.uint8) for _ in boxes]

        import torch

        image_rgb = frame[:, :, ::-1].copy()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._sam2.set_image(image_rgb)
            masks_list = []
            for box in boxes:
                box_arr = np.array(box)
                masks, scores, logits = self._sam2.predict(
                    box=box_arr,
                    multimask_output=True,
                )
                # Take the mask with highest score
                best_idx = int(np.argmax(scores))
                masks_list.append(masks[best_idx].astype(np.uint8))
            return masks_list

    def generate_scene_graph(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate a scene graph using MiMo2.5 API.

        Returns:
            Dict with objects, relations, scene_description.
        """
        if self._mimo is None:
            return {"objects": [], "relations": [], "scene_description": ""}

        det_text = ""
        if detections:
            det_text = "Detected objects: " + ", ".join(
                f"{d['label']}({d['confidence']:.2f})" for d in detections
            )

        prompt = (
            "Analyze this kitchen scene image. Return a JSON object with:\n"
            '- "objects": list of {name, attributes: []}\n'
            '- "relations": list of {subject, predicate, object}\n'
            '- "scene_description": one sentence summary\n'
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
        frames_sequence: List[np.ndarray],
        context: str = "",
    ) -> Dict:
        """Analyze a sequence of frames for action recognition.

        Args:
            frames_sequence: List of BGR frames.
            context: Additional context (e.g. audio hints).

        Returns:
            Dict with action, confidence, description.
        """
        if self._mimo is None or not frames_sequence:
            return {"action": "unknown", "confidence": 0, "description": ""}

        prompt = (
            "Analyze this sequence of kitchen video frames. "
            "What action is being performed? Return JSON with:\n"
            '- "action": short action label\n'
            '- "confidence": 0-1\n'
            '- "description": one sentence\n'
            f"Context: {context}" if context else ""
        )

        # Use middle frame for single-frame analysis
        mid_idx = len(frames_sequence) // 2
        response = self._mimo.call_vision(frames_sequence[mid_idx], prompt)
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
        text_prompt: str = "knife. cutting board. tomato. pan. plate. pot. spoon. food. hand.",
    ) -> Evidence:
        """Full frame analysis pipeline: detect → segment → scene graph.

        Returns:
            Evidence with scene graph content.
        """
        detections = self.detect_objects(frame, text_prompt)
        scene = self.generate_scene_graph(frame, detections)

        return Evidence(
            source_module="VisualAnalyzer",
            evidence_type="visual",
            time_range={"start": timestamp, "end": timestamp},
            content={
                "objects": scene.get("objects", []),
                "relations": scene.get("relations", []),
                "scene_description": scene.get("scene_description", ""),
                "detections": detections,
            },
            confidence=max((d.get("confidence", 0) for d in detections), default=0.5),
        )
