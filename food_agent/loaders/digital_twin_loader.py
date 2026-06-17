"""Digital Twin 3D model loader for HD-EPIC."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Fixture:
    """A kitchen fixture (sink, stove, counter, etc.)."""
    id: str
    fixture_type: str
    mesh_path: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vertex_count: int = 0


class DigitalTwinLoader:
    """Load and query Digital Twin 3D kitchen models.

    Directory structure:
        Digital-Twin/meshes/{participant_id}/{participant_id}_{type}.{idx}.obj
    """

    def __init__(self, dt_dir: str | Path):
        self.dt_dir = Path(dt_dir)
        self.meshes_dir = self.dt_dir / "meshes"
        self._fixture_cache: Dict[str, List[Fixture]] = {}
        self._mesh_cache: Dict[str, object] = {}

    def _parse_fixture_info(self, path: Path, participant_id: str) -> Fixture:
        """Parse fixture type and ID from mesh filename."""
        name = path.stem  # e.g. P01_sink.001
        parts = name.split("_", 1)
        if len(parts) >= 2:
            type_and_idx = parts[1]  # sink.001
            type_parts = type_and_idx.split(".")
            fixture_type = type_parts[0]
        else:
            fixture_type = name

        return Fixture(
            id=name,
            fixture_type=fixture_type,
            mesh_path=str(path),
        )

    def get_fixtures(self, participant_id: str) -> List[Fixture]:
        """Return all fixtures for a participant's kitchen."""
        if participant_id in self._fixture_cache:
            return self._fixture_cache[participant_id]

        participant_meshes = self.meshes_dir / participant_id
        if not participant_meshes.exists():
            raise FileNotFoundError(f"Meshes not found: {participant_meshes}")

        fixtures = []
        for obj_path in sorted(participant_meshes.glob("*.obj")):
            fixture = self._parse_fixture_info(obj_path, participant_id)
            # Compute bounding box from mesh
            try:
                mesh = self.load_mesh(str(obj_path))
                if mesh is not None:
                    vertices = np.asarray(mesh.vertices)
                    if len(vertices) > 0:
                        bbox_min = vertices.min(axis=0)
                        bbox_max = vertices.max(axis=0)
                        fixture.position = (bbox_min + bbox_max) / 2
                        fixture.size = bbox_max - bbox_min
                        fixture.vertex_count = len(vertices)
            except Exception:
                pass
            fixtures.append(fixture)

        self._fixture_cache[participant_id] = fixtures
        return fixtures

    def load_mesh(self, mesh_path: str) -> Optional[object]:
        """Load an OBJ mesh file. Returns open3d.geometry.TriangleMesh or None."""
        if mesh_path in self._mesh_cache:
            return self._mesh_cache[mesh_path]

        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            self._mesh_cache[mesh_path] = mesh
            return mesh
        except Exception:
            return None

    def get_fixture_position(
        self, participant_id: str, fixture_id: str
    ) -> Optional[np.ndarray]:
        """Get the 3D position of a fixture by ID."""
        fixtures = self.get_fixtures(participant_id)
        for f in fixtures:
            if f.id == fixture_id:
                return f.position
        return None

    def get_fixture_by_type(
        self, participant_id: str, fixture_type: str
    ) -> List[Fixture]:
        """Get all fixtures of a given type (e.g. 'sink', 'stove')."""
        fixtures = self.get_fixtures(participant_id)
        return [f for f in fixtures if f.fixture_type == fixture_type]

    def get_spatial_relation(
        self,
        participant_id: str,
        fixture_a_id: str,
        fixture_b_id: str,
        wearer_facing: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute spatial relation between two fixtures.

        Returns dict with 'relation' (left/right/above/below/in_front/behind)
        and 'distance' (meters).
        """
        pos_a = self.get_fixture_position(participant_id, fixture_a_id)
        pos_b = self.get_fixture_position(participant_id, fixture_b_id)
        if pos_a is None or pos_b is None:
            return {"relation": "unknown", "distance": -1}

        diff = pos_a - pos_b
        distance = float(np.linalg.norm(diff))

        if wearer_facing is not None and np.linalg.norm(wearer_facing) > 0:
            cross = np.cross(wearer_facing[:2], diff[:2])
            if cross > 0:
                relation = "left"
            else:
                relation = "right"
        else:
            if abs(diff[2]) > abs(diff[0]) and abs(diff[2]) > abs(diff[1]):
                relation = "above" if diff[2] > 0 else "below"
            elif abs(diff[0]) > abs(diff[1]):
                relation = "right" if diff[0] > 0 else "left"
            else:
                relation = "in_front" if diff[1] > 0 else "behind"

        return {"relation": relation, "distance": distance}
