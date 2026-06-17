"""Gaze data loader for HD-EPIC SLAM-and-Gaze files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class GazePoint:
    """A single gaze observation."""
    timestamp_us: int
    timestamp_s: float
    left_yaw: float
    right_yaw: float
    pitch: float
    depth_m: float
    left_eye_pos: Tuple[float, float, float]
    right_eye_pos: Tuple[float, float, float]

    @property
    def avg_yaw(self) -> float:
        return (self.left_yaw + self.right_yaw) / 2.0

    @property
    def pixel_estimate(self) -> Tuple[float, float]:
        """Rough pixel estimate from yaw/pitch (assuming 1920x1080, 90deg FOV)."""
        cx, cy = 960, 540
        fov_h = np.radians(90)
        fov_v = np.radians(60)
        x = cx + (self.avg_yaw / (fov_h / 2)) * cx
        y = cy - (self.pitch / (fov_v / 2)) * cy
        return (x, y)


class GazeLoader:
    """Load gaze data from HD-EPIC SLAM-and-Gaze directory.

    Directory structure:
        SLAM-and-Gaze/{participant_id}/GAZE_HAND/mps_{video_id}_vrs/eye_gaze/general_eye_gaze.csv
    """

    def __init__(self, gaze_dir: str | Path):
        self.gaze_dir = Path(gaze_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    def _resolve_csv(self, participant_id: str, video_id: str) -> Path:
        pattern = f"mps_{video_id}_vrs"
        eye_gaze_path = (
            self.gaze_dir / participant_id / "GAZE_HAND"
            / pattern / "eye_gaze" / "general_eye_gaze.csv"
        )
        if not eye_gaze_path.exists():
            raise FileNotFoundError(f"Gaze CSV not found: {eye_gaze_path}")
        return eye_gaze_path

    def _load_df(self, participant_id: str, video_id: str) -> pd.DataFrame:
        key = f"{participant_id}/{video_id}"
        if key not in self._cache:
            csv_path = self._resolve_csv(participant_id, video_id)
            self._cache[key] = pd.read_csv(csv_path)
        return self._cache[key]

    def _row_to_gaze(self, row) -> GazePoint:
        return GazePoint(
            timestamp_us=int(row["tracking_timestamp_us"]),
            timestamp_s=float(row["tracking_timestamp_us"]) / 1e6,
            left_yaw=float(row["left_yaw_rads_cpf"]),
            right_yaw=float(row["right_yaw_rads_cpf"]),
            pitch=float(row["pitch_rads_cpf"]),
            depth_m=float(row["depth_m"]),
            left_eye_pos=(
                float(row.get("tx_left_eye_cpf", 0)),
                float(row.get("ty_left_eye_cpf", 0)),
                float(row.get("tz_left_eye_cpf", 0)),
            ),
            right_eye_pos=(
                float(row.get("tx_right_eye_cpf", 0)),
                float(row.get("ty_right_eye_cpf", 0)),
                float(row.get("tz_right_eye_cpf", 0)),
            ),
        )

    def get_gaze_at_time(
        self, participant_id: str, video_id: str, timestamp: float
    ) -> Optional[GazePoint]:
        """Get the gaze observation closest to the given timestamp (seconds)."""
        df = self._load_df(participant_id, video_id)
        ts_us = int(timestamp * 1e6)
        idx = np.argmin(np.abs(df["tracking_timestamp_us"].values - ts_us))
        return self._row_to_gaze(df.iloc[idx])

    def get_gaze_trajectory(
        self,
        participant_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[GazePoint]:
        """Get all gaze observations within a time range."""
        df = self._load_df(participant_id, video_id)
        start_us = int(start_time * 1e6)
        end_us = int(end_time * 1e6)
        mask = (df["tracking_timestamp_us"] >= start_us) & (
            df["tracking_timestamp_us"] <= end_us
        )
        return [self._row_to_gaze(row) for _, row in df[mask].iterrows()]

    def get_fixations(
        self,
        participant_id: str,
        video_id: str,
        min_duration: float = 0.2,
    ) -> List[Dict]:
        """Detect fixation periods (gaze stable for min_duration seconds).

        Returns list of dicts with start_time, end_time, duration, mean_yaw, mean_pitch.
        """
        df = self._load_df(participant_id, video_id)
        if df.empty:
            return []

        timestamps_s = df["tracking_timestamp_us"].values / 1e6
        yaws = ((df["left_yaw_rads_cpf"].values + df["right_yaw_rads_cpf"].values) / 2)
        pitches = df["pitch_rads_cpf"].values

        # Sliding window: group consecutive samples with small angular change
        yaw_thresh = np.radians(3)  # 3 degrees
        pitch_thresh = np.radians(3)

        fixations = []
        start_idx = 0
        for i in range(1, len(df)):
            dyaw = abs(yaws[i] - yaws[start_idx])
            dpitch = abs(pitches[i] - pitches[start_idx])
            if dyaw > yaw_thresh or dpitch > pitch_thresh:
                duration = timestamps_s[i - 1] - timestamps_s[start_idx]
                if duration >= min_duration:
                    fixations.append({
                        "start_time": float(timestamps_s[start_idx]),
                        "end_time": float(timestamps_s[i - 1]),
                        "duration": float(duration),
                        "mean_yaw": float(np.mean(yaws[start_idx:i])),
                        "mean_pitch": float(np.mean(pitches[start_idx:i])),
                    })
                start_idx = i

        # Handle last segment
        duration = timestamps_s[-1] - timestamps_s[start_idx]
        if duration >= min_duration:
            fixations.append({
                "start_time": float(timestamps_s[start_idx]),
                "end_time": float(timestamps_s[-1]),
                "duration": float(duration),
                "mean_yaw": float(np.mean(yaws[start_idx:])),
                "mean_pitch": float(np.mean(pitches[start_idx:])),
            })

        return fixations
