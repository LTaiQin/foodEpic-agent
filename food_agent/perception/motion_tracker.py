"""Motion tracker: object tracking and trajectory analysis."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .evidence import Evidence


class MotionTracker:
    """Track object motion across video frames.

    Uses SAM 2.1 video predictor for object tracking, and SLAM data
    for 3D trajectory reconstruction.
    """

    def __init__(self, sam2_video_predictor=None, slam_loader=None):
        self._sam2_video = sam2_video_predictor
        self._slam = slam_loader

    def track_object(
        self,
        video_path: str,
        first_frame_mask: np.ndarray,
        frame_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Track an object across video frames using SAM 2.1.

        Args:
            video_path: Path to the MP4 file.
            first_frame_mask: Binary mask of the object in the first frame.
            frame_range: (start_frame, end_frame) or None for full video.

        Returns:
            Dict mapping frame_index to mask array.
        """
        if self._sam2_video is None:
            return {}

        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._sam2_video.init_state(video_path)

            # Add the initial mask as a prompt
            self._sam2_video.add_new_mask(
                state, frame_idx=0, obj_id=1, mask=first_frame_mask
            )

            masks_dict = {}
            for frame_idx, obj_ids, masks in self._sam2_video.propagate_in_video(state):
                if frame_range and (frame_idx < frame_range[0] or frame_idx > frame_range[1]):
                    continue
                if masks is not None and len(masks) > 0:
                    masks_dict[frame_idx] = masks[0].cpu().numpy().astype(np.uint8)

            return masks_dict

    def extract_trajectory(
        self, masks_sequence: Dict[int, np.ndarray]
    ) -> List[Tuple[int, float, float]]:
        """Extract object center trajectory from a sequence of masks.

        Args:
            masks_sequence: Dict mapping frame_index to binary mask.

        Returns:
            List of (frame_index, center_x, center_y) tuples.
        """
        trajectory = []
        for frame_idx in sorted(masks_sequence.keys()):
            mask = masks_sequence[frame_idx]
            if mask is None or mask.sum() == 0:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            trajectory.append((frame_idx, cx, cy))

        return trajectory

    def lift_to_3d(
        self,
        trajectory_2d: List[Tuple[int, float, float]],
        slam_poses: List,
        depth_estimate: float = 1.0,
    ) -> List[Dict]:
        """Lift 2D trajectory to 3D using SLAM poses.

        Args:
            trajectory_2d: List of (frame_idx, cx, cy) tuples.
            slam_poses: List of SLAMPose objects aligned to frames.
            depth_estimate: Estimated depth in meters.

        Returns:
            List of dicts with frame_idx, position_3d.
        """
        if not trajectory_2d or not slam_poses:
            return []

        result = []
        for frame_idx, cx, cy in trajectory_2d:
            # Find the closest SLAM pose
            if frame_idx < len(slam_poses):
                pose = slam_poses[frame_idx]
                # Simple projection: use wearer position + direction * depth
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_quat(pose.quaternion)
                forward = rot.apply([0, 0, -1])
                right = rot.apply([1, 0, 0])
                up = rot.apply([0, 1, 0])

                # Normalize pixel coordinates to [-1, 1]
                norm_x = (cx / 704) - 1  # Assuming 1408 width
                norm_y = (cy / 704) - 1

                # 3D position
                pos_3d = (
                    pose.position
                    + forward * depth_estimate
                    + right * norm_x * depth_estimate * 0.5
                    + up * (-norm_y) * depth_estimate * 0.5
                )

                result.append({
                    "frame_idx": frame_idx,
                    "position_3d": pos_3d.tolist(),
                    "wearer_position": pose.position.tolist(),
                })

        return result

    def classify_motion(self, trajectory_3d: List[Dict]) -> Dict:
        """Classify the type of motion from a 3D trajectory.

        Returns:
            Dict with motion_type, speed_avg, displacement.
        """
        if len(trajectory_3d) < 2:
            return {"motion_type": "stationary", "speed_avg": 0, "displacement": 0}

        positions = np.array([p["position_3d"] for p in trajectory_3d])
        displacements = np.diff(positions, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        total_distance = float(distances.sum())

        start_pos = positions[0]
        end_pos = positions[-1]
        net_displacement = float(np.linalg.norm(end_pos - start_pos))

        # Estimate time span (assume ~30fps)
        n_frames = len(trajectory_3d)
        duration = n_frames / 30.0
        speed_avg = total_distance / duration if duration > 0 else 0

        # Classify
        if net_displacement < 0.05:
            motion_type = "stationary"
        elif net_displacement / (total_distance + 1e-8) > 0.8:
            motion_type = "linear"
        elif total_distance > 0.5 and net_displacement < 0.2:
            motion_type = "stirring"
        else:
            motion_type = "displacement"

        return {
            "motion_type": motion_type,
            "speed_avg": round(speed_avg, 4),
            "displacement": round(net_displacement, 4),
            "total_path_length": round(total_distance, 4),
            "duration": round(duration, 2),
        }

    def get_motion_evidence(
        self,
        video_id: str,
        frame_number: int,
        masks_sequence: Optional[Dict[int, np.ndarray]] = None,
    ) -> Evidence:
        """Generate motion evidence from tracked masks.

        Returns:
            Evidence with motion trajectory data.
        """
        timestamp = frame_number / 30.0

        if masks_sequence:
            trajectory = self.extract_trajectory(masks_sequence)
            motion_info = self.classify_motion(
                [{"position_3d": [t[1], t[2], 0]} for t in trajectory]
            )
        else:
            trajectory = []
            motion_info = {"motion_type": "unknown", "speed_avg": 0, "displacement": 0}

        return Evidence(
            source_module="MotionTracker",
            evidence_type="motion",
            time_range={"start": timestamp, "end": timestamp},
            content={
                "trajectory_points": len(trajectory),
                **motion_info,
            },
            confidence=0.7 if trajectory else 0.0,
        )
