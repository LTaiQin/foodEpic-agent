"""Hands-Masks loader for HD-EPIC hand/object mask data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np


def _rle_decode(rle: Dict, shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE mask (COCO-style counts + size format) to binary mask.

    Args:
        rle: Dict with 'size' [H, W] and 'counts' (list or string).
        shape: Target (H, W) for the output mask.

    Returns:
        Binary mask as uint8 numpy array.
    """
    h, w = shape
    size = rle.get("size", [h, w])
    counts = rle.get("counts", [])

    # Handle uncompressed RLE (list of run lengths)
    if isinstance(counts, list):
        total = size[0] * size[1]
        mask_flat = np.zeros(total, dtype=np.uint8)
        idx = 0
        val = 0  # starts with background
        for run_len in counts:
            run_len = int(run_len)
            if val == 1:
                mask_flat[idx:idx + run_len] = 1
            idx += run_len
            val = 1 - val
        return mask_flat.reshape(size[0], size[1])

    # Handle compressed RLE string (pycocotools format)
    try:
        from pycocotools import mask as coco_mask
        decoded = coco_mask.decode({"size": size, "counts": counts.encode() if isinstance(counts, str) else counts})
        return decoded.astype(np.uint8)
    except ImportError:
        return np.zeros(shape, dtype=np.uint8)


class HandsLoader:
    """Load hand and object masks from HD-EPIC Hands-Masks data.

    Directory structure:
        Hands-Masks/contours_cleaned/{video_id}_gt_cleaned_masks.json

    Each JSON file maps frame_number -> {hand_label: rle_mask_dict}.
    Hand labels: 'left', 'right', or object names.
    RLE masks have 'size' [H, W] and 'counts' (run-length encoded).
    """

    def __init__(self, hands_dir: str | Path):
        self.hands_dir = Path(hands_dir)
        self.contours_dir = self.hands_dir / "contours_cleaned"
        self._cache: Dict[str, Dict] = {}

    def _load_masks_file(self, video_id: str) -> Dict:
        if video_id in self._cache:
            return self._cache[video_id]

        path = self.contours_dir / f"{video_id}_gt_cleaned_masks.json"
        if not path.exists():
            raise FileNotFoundError(f"Hands-Masks file not found: {path}")

        with open(path) as f:
            data = json.load(f)
        self._cache[video_id] = data
        return data

    def get_available_frames(self, video_id: str) -> List[int]:
        """Return sorted list of frame numbers with mask data."""
        data = self._load_masks_file(video_id)
        return sorted(int(k) for k in data.keys())

    def get_mask(
        self, video_id: str, frame_number: int, hand: str = "left"
    ) -> Optional[np.ndarray]:
        """Get a single hand mask for a frame.

        Args:
            video_id: Video identifier.
            frame_number: Frame index.
            hand: 'left' or 'right'.

        Returns:
            Binary mask as uint8 array, or None if not available.
        """
        data = self._load_masks_file(video_id)
        frame_key = str(frame_number)
        if frame_key not in data:
            return None
        frame_data = data[frame_key]
        if hand not in frame_data:
            return None
        rle = frame_data[hand]
        shape = tuple(rle.get("size", [1408, 1408]))
        return _rle_decode(rle, shape)

    def get_masks_in_range(
        self,
        video_id: str,
        start_frame: int,
        end_frame: int,
        hand: str = "left",
    ) -> Dict[int, np.ndarray]:
        """Get hand masks for a range of frames.

        Returns dict mapping frame_number -> mask array.
        """
        data = self._load_masks_file(video_id)
        masks = {}
        for frame_key, frame_data in data.items():
            fn = int(frame_key)
            if fn < start_frame or fn > end_frame:
                continue
            if hand in frame_data:
                rle = frame_data[hand]
                shape = tuple(rle.get("size", [1408, 1408]))
                masks[fn] = _rle_decode(rle, shape)
        return masks

    def has_hand_contact(
        self,
        mask: np.ndarray,
        object_bbox: Optional[Tuple[int, int, int, int]] = None,
        iou_threshold: float = 0.05,
    ) -> bool:
        """Check if a hand mask indicates contact with an object.

        If object_bbox is provided, checks overlap between hand mask and
        the object bounding box (x1, y1, x2, y2).

        Otherwise, returns True if the mask has any non-zero pixels.
        """
        if mask is None or mask.sum() == 0:
            return False

        if object_bbox is None:
            return True

        x1, y1, x2, y2 = object_bbox
        h, w = mask.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False

        contact_ratio = region.sum() / region.size
        return contact_ratio > iou_threshold

    def get_all_labels(self, video_id: str, frame_number: int) -> List[str]:
        """Get all available labels (hands + objects) for a frame."""
        data = self._load_masks_file(video_id)
        frame_key = str(frame_number)
        if frame_key not in data:
            return []
        return list(data[frame_key].keys())
