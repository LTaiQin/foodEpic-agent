"""Video loader for HD-EPIC MP4 files."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class VideoLoader:
    """Load video frames and metadata from HD-EPIC MP4 files."""

    def __init__(self, video_dir: str | Path):
        self.video_dir = Path(video_dir)

    def _resolve_video(self, video_id: str) -> Path:
        """Find the MP4 file for a given video ID."""
        participant_id = video_id.split("-")[0]
        path = self.video_dir / participant_id / f"{video_id}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return path

    def get_video_info(self, video_id: str) -> Dict:
        """Return metadata for a video: fps, frame_count, duration, resolution."""
        path = self._resolve_video(video_id)
        cap = cv2.VideoCapture(str(path))
        try:
            info = {
                "path": str(path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
            return info
        finally:
            cap.release()

    def get_frame(self, video_id: str, timestamp: float) -> np.ndarray:
        """Extract a single frame at the given timestamp.

        Args:
            video_id: e.g. "P08-20240620-180825".
            timestamp: Time in seconds.

        Returns:
            numpy array of shape (H, W, 3), dtype uint8, BGR format.
        """
        path = self._resolve_video(video_id)
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read frame at t={timestamp}s from {path}")
            return frame
        finally:
            cap.release()

    def get_frames(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        fps: float = 2.0,
    ) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at regular intervals within a time range.

        Args:
            video_id: Video identifier.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            fps: Sampling rate (frames per second).

        Returns:
            List of (timestamp, frame) tuples.
        """
        path = self._resolve_video(video_id)
        cap = cv2.VideoCapture(str(path))
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                raise RuntimeError(f"Cannot determine FPS for {path}")

            frames = []
            interval = 1.0 / fps
            t = start_time
            while t <= end_time:
                frame_idx = int(t * video_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append((t, frame))
                t += interval
            return frames
        finally:
            cap.release()

    def get_frame_at_index(self, video_id: str, frame_index: int) -> Tuple[float, np.ndarray]:
        """Extract a frame by its index number.

        Returns:
            Tuple of (timestamp_seconds, frame_array).
        """
        path = self._resolve_video(video_id)
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = frame_index / fps if fps > 0 else 0
            return timestamp, frame
        finally:
            cap.release()
