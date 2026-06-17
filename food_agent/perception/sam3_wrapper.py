"""SAM3 wrapper: open-vocabulary segmentation for kitchen scenes."""

from typing import Dict, List, Optional, Tuple

import numpy as np


class SAM3Segmentor:
    """SAM3-based open-vocabulary segmentation.

    Uses Sam3Model + Sam3Processor from transformers for
    text-prompted instance segmentation.
    """

    def __init__(self, model_path: str = "/22liushoulong/sam-weight/"):
        self._model_path = model_path
        self._model = None
        self._processor = None
        self._device = "cuda"
        self._dtype = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Processor, Sam3Model

        self._dtype = torch.bfloat16
        self._model = Sam3Model.from_pretrained(
            self._model_path, torch_dtype=self._dtype
        ).to(self._device)
        self._processor = Sam3Processor.from_pretrained(self._model_path)

    def segment(
        self,
        image: np.ndarray,
        text_prompt: str = "food ingredient",
        threshold: float = 0.1,
        mask_threshold: float = 0.2,
    ) -> Dict:
        """Segment objects matching the text prompt.

        Args:
            image: BGR numpy array (H, W, 3).
            text_prompt: Open-vocabulary text description.
            threshold: Detection confidence threshold.
            mask_threshold: Mask binarization threshold.

        Returns:
            Dict with 'masks' (N, H, W), 'scores' (N,), 'count' (int).
        """
        self._load()

        import torch
        import cv2
        from PIL import Image

        # Convert BGR to RGB PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self._processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self._device)

        # Match dtype
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                inputs[k] = v.to(self._dtype)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"].float().cpu().numpy()
        scores = results["scores"].float().cpu().numpy()

        return {
            "masks": masks,
            "scores": scores,
            "count": len(masks),
        }

    def detect_objects(
        self,
        image: np.ndarray,
        text_prompt: str = "food ingredient",
        threshold: float = 0.1,
    ) -> List[Dict]:
        """Detect and segment objects, return as list of dicts.

        Returns:
            List of dicts with 'mask' (H, W), 'score', 'area', 'bbox'.
        """
        result = self.segment(image, text_prompt, threshold)

        objects = []
        for i in range(result["count"]):
            mask = result["masks"][i]
            score = float(result["scores"][i])
            area = int(mask.sum())

            # Compute bounding box from mask
            ys, xs = np.where(mask > 0.5)
            if len(xs) > 0:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            else:
                bbox = [0, 0, 0, 0]

            objects.append({
                "mask": mask,
                "score": score,
                "area": area,
                "bbox": bbox,
                "label": text_prompt,
            })

        return objects
