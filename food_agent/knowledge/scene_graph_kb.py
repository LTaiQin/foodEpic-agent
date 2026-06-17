"""Scene graph knowledge base: store and query frame-level scene graphs."""

from typing import Dict, List, Optional


class SceneGraphKB:
    """In-memory knowledge base for scene graphs.

    Stores per-frame scene graphs (objects + relations) and provides
    query interfaces for temporal and object-based lookups.
    """

    def __init__(self):
        self._graphs: Dict[float, Dict] = {}  # timestamp -> graph

    def add_frame_graph(self, timestamp: float, graph: Dict) -> None:
        """Add a scene graph for a specific timestamp.

        Args:
            timestamp: Frame timestamp in seconds.
            graph: Dict with 'objects' list and 'relations' list.
        """
        self._graphs[timestamp] = graph

    def query_objects(self, object_type: str) -> List[Dict]:
        """Find all timestamps where an object type appears.

        Returns:
            List of dicts with timestamp and object info.
        """
        results = []
        for ts, graph in self._graphs.items():
            for obj in graph.get("objects", []):
                name = obj.get("name", "") if isinstance(obj, dict) else str(obj)
                if object_type.lower() in name.lower():
                    results.append({"timestamp": ts, "object": obj})
        return results

    def query_relations(
        self, subject: Optional[str] = None, predicate: Optional[str] = None
    ) -> List[Dict]:
        """Query scene graph relations.

        Args:
            subject: Filter by subject name (partial match).
            predicate: Filter by predicate (partial match).

        Returns:
            List of matching relations with timestamps.
        """
        results = []
        for ts, graph in self._graphs.items():
            for rel in graph.get("relations", []):
                if not isinstance(rel, dict):
                    continue
                subj = rel.get("subject", "")
                pred = rel.get("predicate", "")
                if subject and subject.lower() not in subj.lower():
                    continue
                if predicate and predicate.lower() not in pred.lower():
                    continue
                results.append({"timestamp": ts, **rel})
        return results

    def get_scene_summary(self, start_time: float, end_time: float) -> Dict:
        """Get a summary of scene graphs in a time range.

        Returns:
            Dict with object_counts, relation_counts, time_range.
        """
        object_counts: Dict[str, int] = {}
        relation_counts: Dict[str, int] = {}
        count = 0

        for ts, graph in self._graphs.items():
            if ts < start_time or ts > end_time:
                continue
            count += 1
            for obj in graph.get("objects", []):
                name = obj.get("name", "") if isinstance(obj, dict) else str(obj)
                object_counts[name] = object_counts.get(name, 0) + 1
            for rel in graph.get("relations", []):
                if isinstance(rel, dict):
                    pred = rel.get("predicate", "unknown")
                    relation_counts[pred] = relation_counts.get(pred, 0) + 1

        return {
            "time_range": [start_time, end_time],
            "frame_count": count,
            "object_counts": object_counts,
            "relation_counts": relation_counts,
        }

    def clear(self) -> None:
        self._graphs.clear()
