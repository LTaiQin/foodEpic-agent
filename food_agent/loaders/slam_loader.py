"""SLAM trajectory loader for HD-EPIC SLAM-and-Gaze files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SLAMPose:
    """A 6DoF SLAM pose."""
    timestamp_us: int
    timestamp_s: float
    position: np.ndarray  # (3,) tx, ty, tz in world frame
    quaternion: np.ndarray  # (4,) qx, qy, qz, qw
    linear_velocity: np.ndarray  # (3,)
    angular_velocity: np.ndarray  # (3,)
    quality_score: float

    @property
    def facing_direction(self) -> np.ndarray:
        """Compute forward-facing direction from quaternion.

        Assumes -Z is the forward direction in the device frame.
        """
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.quaternion)
        return rot.apply([0, 0, -1])


class SLAMLoader:
    """Load SLAM trajectory data from HD-EPIC.

    Directory structure:
        SLAM-and-Gaze/{participant_id}/SLAM/multi/{session_idx}/slam/closed_loop_trajectory.csv
    """

    def __init__(self, slam_dir: str | Path):
        self.slam_dir = Path(slam_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    def _find_trajectory_csv(self, participant_id: str, video_id: str) -> Path:
        """Find the closed_loop_trajectory.csv for a given video."""
        slam_base = self.slam_dir / participant_id / "SLAM" / "multi"
        if not slam_base.exists():
            raise FileNotFoundError(f"SLAM dir not found: {slam_base}")

        # Session indices are 0, 1, 2, ... ; try to match by checking all
        for session_dir in sorted(slam_base.iterdir()):
            csv_path = session_dir / "slam" / "closed_loop_trajectory.csv"
            if csv_path.exists():
                # Check if this session covers the video's time range
                return csv_path

        raise FileNotFoundError(f"No SLAM trajectory found for {participant_id}/{video_id}")

    def _load_df(self, participant_id: str, video_id: str) -> pd.DataFrame:
        key = f"{participant_id}/{video_id}"
        if key not in self._cache:
            csv_path = self._find_trajectory_csv(participant_id, video_id)
            self._cache[key] = pd.read_csv(csv_path)
        return self._cache[key]

    def _row_to_pose(self, row) -> SLAMPose:
        return SLAMPose(
            timestamp_us=int(row["tracking_timestamp_us"]),
            timestamp_s=float(row["tracking_timestamp_us"]) / 1e6,
            position=np.array([
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ]),
            quaternion=np.array([
                float(row["qx_world_device"]),
                float(row["qy_world_device"]),
                float(row["qz_world_device"]),
                float(row["qw_world_device"]),
            ]),
            linear_velocity=np.array([
                float(row.get("device_linear_velocity_x_device", 0)),
                float(row.get("device_linear_velocity_y_device", 0)),
                float(row.get("device_linear_velocity_z_device", 0)),
            ]),
            angular_velocity=np.array([
                float(row.get("angular_velocity_x_device", 0)),
                float(row.get("angular_velocity_y_device", 0)),
                float(row.get("angular_velocity_z_device", 0)),
            ]),
            quality_score=float(row.get("quality_score", 0)),
        )

    def get_pose(
        self, participant_id: str, video_id: str, timestamp: float
    ) -> Optional[SLAMPose]:
        """Get the SLAM pose closest to the given timestamp (seconds)."""
        df = self._load_df(participant_id, video_id)
        ts_us = int(timestamp * 1e6)
        idx = np.argmin(np.abs(df["tracking_timestamp_us"].values - ts_us))
        return self._row_to_pose(df.iloc[idx])

    def get_trajectory(
        self,
        participant_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
    ) -> List[SLAMPose]:
        """Get all SLAM poses within a time range."""
        df = self._load_df(participant_id, video_id)
        start_us = int(start_time * 1e6)
        end_us = int(end_time * 1e6)
        mask = (df["tracking_timestamp_us"] >= start_us) & (
            df["tracking_timestamp_us"] <= end_us
        )
        return [self._row_to_pose(row) for _, row in df[mask].iterrows()]

    def get_position(
        self, participant_id: str, video_id: str, timestamp: float
    ) -> np.ndarray:
        """Get 3D position at a given timestamp."""
        pose = self.get_pose(participant_id, video_id, timestamp)
        return pose.position if pose is not None else np.zeros(3)

    def get_facing_direction(
        self, participant_id: str, video_id: str, timestamp: float
    ) -> np.ndarray:
        """Get facing direction vector at a given timestamp."""
        pose = self.get_pose(participant_id, video_id, timestamp)
        return pose.facing_direction if pose is not None else np.array([0, 0, -1])
