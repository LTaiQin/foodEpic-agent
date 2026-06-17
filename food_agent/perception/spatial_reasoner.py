"""Spatial reasoner: 3D spatial relationships using Digital Twin and SLAM."""

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .evidence import Evidence


class SpatialReasoner:
    """Reason about 3D spatial relationships in the kitchen.

    Combines Digital Twin fixture data with SLAM wearer pose to answer
    spatial queries like "where is the sink" or "what is the wearer facing".
    """

    def __init__(self, digital_twin_loader, slam_loader):
        self._dt = digital_twin_loader
        self._slam = slam_loader

    @staticmethod
    def compute_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """Euclidean distance between two 3D points."""
        return float(np.linalg.norm(np.asarray(pos_a) - np.asarray(pos_b)))

    def compute_spatial_relation(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        wearer_facing: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute spatial relation of A relative to B.

        Args:
            pos_a: Position of object A (3D).
            pos_b: Position of object B (3D).
            wearer_facing: Wearer's facing direction (3D), for left/right.

        Returns:
            Dict with 'relation' and 'distance'.
        """
        diff = np.asarray(pos_a) - np.asarray(pos_b)
        distance = float(np.linalg.norm(diff))

        if distance < 1e-6:
            return {"relation": "same_location", "distance": 0.0}

        if wearer_facing is not None and np.linalg.norm(wearer_facing) > 0:
            # Use cross product with facing direction for left/right
            facing_2d = wearer_facing[:2] / (np.linalg.norm(wearer_facing[:2]) + 1e-8)
            diff_2d = diff[:2] / (np.linalg.norm(diff[:2]) + 1e-8)
            cross = np.cross(facing_2d, diff_2d)
            if abs(cross) > 0.3:
                relation = "left" if cross > 0 else "right"
            else:
                dot = np.dot(facing_2d, diff_2d)
                relation = "in_front" if dot > 0 else "behind"
        else:
            # Axis-based
            if abs(diff[2]) > abs(diff[0]) and abs(diff[2]) > abs(diff[1]):
                relation = "above" if diff[2] > 0 else "below"
            elif abs(diff[0]) > abs(diff[1]):
                relation = "right" if diff[0] > 0 else "left"
            else:
                relation = "in_front" if diff[1] > 0 else "behind"

        return {"relation": relation, "distance": distance}

    def get_nearest_fixture(
        self, participant_id: str, position: np.ndarray
    ) -> Dict:
        """Find the nearest kitchen fixture to a 3D position.

        Returns:
            Dict with fixture_id, fixture_type, distance.
        """
        fixtures = self._dt.get_fixtures(participant_id)
        best = None
        best_dist = float("inf")

        for f in fixtures:
            dist = self.compute_distance(position, f.position)
            if dist < best_dist:
                best_dist = dist
                best = f

        if best is None:
            return {"fixture_id": "none", "fixture_type": "unknown", "distance": -1}

        return {
            "fixture_id": best.id,
            "fixture_type": best.fixture_type,
            "distance": best_dist,
        }

    def get_wearer_pose_at_time(
        self, participant_id: str, video_id: str, timestamp: float
    ) -> Dict:
        """Get the wearer's 3D pose at a given time.

        Returns:
            Dict with position, facing, nearest_fixture, distance_to_nearest.
        """
        pose = self._slam.get_pose(participant_id, video_id, timestamp)
        if pose is None:
            return {
                "position": [0, 0, 0],
                "facing": [0, 0, -1],
                "nearest_fixture": "unknown",
                "distance_to_nearest": -1,
            }

        nearest = self.get_nearest_fixture(participant_id, pose.position)
        facing = pose.facing_direction

        return {
            "position": pose.position.tolist(),
            "facing": facing.tolist(),
            "nearest_fixture": nearest["fixture_type"],
            "distance_to_nearest": nearest["distance"],
        }

    def check_visibility(
        self,
        source: np.ndarray,
        target: np.ndarray,
        participant_id: str,
    ) -> Dict:
        """Check if target is visible from source (simple line-of-sight).

        Uses a simplified check: if the nearest fixture to the line between
        source and target is not obstructing.

        Returns:
            Dict with visible (bool), distance, obstruction.
        """
        distance = self.compute_distance(source, target)
        direction = (np.asarray(target) - np.asarray(source))
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

        # Check if any fixture is close to the line of sight
        fixtures = self._dt.get_fixtures(participant_id)
        min_obstruction_dist = float("inf")
        obstruction = None

        for f in fixtures:
            # Project fixture position onto the line
            to_fixture = f.position - np.asarray(source)
            proj_length = np.dot(to_fixture, direction_norm)
            if proj_length < 0 or proj_length > distance:
                continue
            closest_point = np.asarray(source) + proj_length * direction_norm
            perp_dist = np.linalg.norm(f.position - closest_point)
            if perp_dist < min_obstruction_dist:
                min_obstruction_dist = perp_dist
                obstruction = f.fixture_type

        # If a fixture is very close to the line, it might be blocking
        is_visible = min_obstruction_dist > 0.3  # 30cm threshold

        return {
            "visible": is_visible,
            "distance": distance,
            "obstruction": obstruction if not is_visible else None,
        }

    def describe_spatial_layout(
        self, participant_id: str, mimo_client=None
    ) -> Dict:
        """Generate a natural language description of the kitchen layout.

        Returns:
            Dict with fixtures list and spatial_relations.
        """
        fixtures = self._dt.get_fixtures(participant_id)
        fixture_list = [
            {"id": f.id, "type": f.fixture_type, "position": f.position.tolist()}
            for f in fixtures
        ]

        # Compute pairwise spatial relations
        relations = []
        for i, f1 in enumerate(fixtures):
            for f2 in fixtures[i + 1:]:
                rel = self.compute_spatial_relation(f1.position, f2.position)
                relations.append({
                    "from": f1.id,
                    "to": f2.id,
                    "relation": rel["relation"],
                    "distance": rel["distance"],
                })

        return {
            "fixtures": fixture_list,
            "spatial_relations": relations[:20],  # Limit to top 20
        }

    def query_3d(
        self,
        participant_id: str,
        video_id: str,
        timestamp: float,
        query_type: str = "layout",
    ) -> Evidence:
        """Main entry point for 3D spatial queries.

        Args:
            query_type: 'layout', 'wearer_pose', or 'nearest'.

        Returns:
            Evidence with spatial content.
        """
        if query_type == "layout":
            layout = self.describe_spatial_layout(participant_id)
            return Evidence(
                source_module="SpatialReasoner",
                evidence_type="spatial",
                time_range={"start": timestamp, "end": timestamp},
                content=layout,
                confidence=0.9,
            )
        elif query_type == "wearer_pose":
            pose = self.get_wearer_pose_at_time(participant_id, video_id, timestamp)
            return Evidence(
                source_module="SpatialReasoner",
                evidence_type="spatial",
                time_range={"start": timestamp, "end": timestamp},
                content=pose,
                confidence=0.8,
            )
        else:
            pos = self._slam.get_position(participant_id, video_id, timestamp)
            nearest = self.get_nearest_fixture(participant_id, pos)
            return Evidence(
                source_module="SpatialReasoner",
                evidence_type="spatial",
                time_range={"start": timestamp, "end": timestamp},
                content=nearest,
                confidence=0.85,
            )
