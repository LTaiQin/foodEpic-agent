"""SQLite-backed graph memory store."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .schema import GraphEdgeRecord, GraphNodeRecord


class GraphMemoryStore:
    """Small SQLite-backed graph store with JSONL mirrors for inspection."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "graph.db"
        self.nodes_path = self.root / "nodes.jsonl"
        self.edges_path = self.root / "edges.jsonl"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path.as_posix())
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    attributes_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    keywords_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    attributes_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_video_type ON nodes(video_id, node_type);
                CREATE INDEX IF NOT EXISTS idx_nodes_time ON nodes(video_id, start_time, end_time);
                CREATE INDEX IF NOT EXISTS idx_edges_video_type ON edges(video_id, edge_type);
                """
            )

    def replace_graph(self, nodes: list[GraphNodeRecord], edges: list[GraphEdgeRecord]) -> None:
        normalized_nodes = [self._normalize_node(node) for node in nodes]
        normalized_edges = [self._normalize_edge(edge) for edge in edges]
        with self._connect() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            self._upsert_nodes(conn, normalized_nodes)
            self._upsert_edges(conn, normalized_edges)
        self._rewrite_jsonl_mirrors()

    def upsert_node(self, node: GraphNodeRecord) -> None:
        normalized = self._normalize_node(node)
        with self._connect() as conn:
            self._upsert_nodes(conn, [normalized])
        self._rewrite_jsonl_mirrors()

    def upsert_edge(self, edge: GraphEdgeRecord) -> None:
        normalized = self._normalize_edge(edge)
        with self._connect() as conn:
            self._upsert_edges(conn, [normalized])
        self._rewrite_jsonl_mirrors()

    def query_nodes(
        self,
        *,
        video_id: str,
        node_types: list[str] | None = None,
        keyword: str | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        sql = [
            "SELECT * FROM nodes WHERE video_id = ?",
        ]
        params: list[Any] = [video_id]
        if node_types:
            placeholders = ",".join("?" for _ in node_types)
            sql.append(f"AND node_type IN ({placeholders})")
            params.extend(node_types)
        if keyword:
            sql.append("AND (LOWER(label) LIKE ? OR LOWER(attributes_json) LIKE ? OR LOWER(keywords_json) LIKE ?)")
            needle = f"%{keyword.lower()}%"
            params.extend([needle, needle, needle])
        if time_start is not None:
            sql.append("AND (end_time IS NULL OR end_time >= ?)")
            params.append(time_start)
        if time_end is not None:
            sql.append("AND (start_time IS NULL OR start_time <= ?)")
            params.append(time_end)
        sql.append("ORDER BY CASE WHEN start_time IS NULL THEN 1 ELSE 0 END, start_time ASC, label ASC LIMIT ?")
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._row_to_node_dict(row) for row in rows]

    def get_neighbors(self, *, node_ids: list[str], edge_types: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        sql = [
            f"SELECT * FROM edges WHERE source_id IN ({','.join('?' for _ in node_ids)})",
        ]
        params: list[Any] = list(node_ids)
        if edge_types:
            sql.append(f"AND edge_type IN ({','.join('?' for _ in edge_types)})")
            params.extend(edge_types)
        sql.append("ORDER BY edge_type ASC, source_id ASC LIMIT ?")
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [self._row_to_edge_dict(row) for row in rows]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return self._row_to_node_dict(row) if row else None

    def list_edges(self, *, video_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE video_id = ? ORDER BY edge_type ASC, source_id ASC LIMIT ?",
                (video_id, limit),
            ).fetchall()
        return [self._row_to_edge_dict(row) for row in rows]

    def _row_to_node_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "label": row["label"],
            "video_id": row["video_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "attributes": json.loads(row["attributes_json"]),
            "evidence_paths": json.loads(row["evidence_json"]),
            "keywords": json.loads(row["keywords_json"]),
        }

    def _row_to_edge_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "edge_id": row["edge_id"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "edge_type": row["edge_type"],
            "video_id": row["video_id"],
            "attributes": json.loads(row["attributes_json"]),
        }

    def _upsert_nodes(self, conn: sqlite3.Connection, nodes: list[GraphNodeRecord]) -> None:
        conn.executemany(
            """
            INSERT OR REPLACE INTO nodes(node_id, node_type, label, video_id, start_time, end_time, attributes_json, evidence_json, keywords_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node.node_id,
                    node.node_type,
                    node.label,
                    node.video_id,
                    node.start_time,
                    node.end_time,
                    json.dumps(node.attributes, ensure_ascii=False),
                    json.dumps(node.evidence_paths, ensure_ascii=False),
                    json.dumps(node.keywords, ensure_ascii=False),
                )
                for node in nodes
            ],
        )

    def _upsert_edges(self, conn: sqlite3.Connection, edges: list[GraphEdgeRecord]) -> None:
        conn.executemany(
            """
            INSERT OR REPLACE INTO edges(edge_id, source_id, target_id, edge_type, video_id, attributes_json)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.video_id,
                    json.dumps(edge.attributes, ensure_ascii=False),
                )
                for edge in edges
            ],
        )

    def _rewrite_jsonl_mirrors(self) -> None:
        with self._connect() as conn:
            node_rows = conn.execute("SELECT * FROM nodes ORDER BY node_type ASC, start_time ASC, node_id ASC").fetchall()
            edge_rows = conn.execute("SELECT * FROM edges ORDER BY edge_type ASC, source_id ASC, edge_id ASC").fetchall()
        nodes = [self._row_to_node_dict(row) for row in node_rows]
        edges = [self._row_to_edge_dict(row) for row in edge_rows]
        self.nodes_path.write_text(
            "\n".join(json.dumps(node, ensure_ascii=False) for node in nodes) + ("\n" if nodes else ""),
            encoding="utf-8",
        )
        self.edges_path.write_text(
            "\n".join(json.dumps(edge, ensure_ascii=False) for edge in edges) + ("\n" if edges else ""),
            encoding="utf-8",
        )

    def _normalize_node(self, node: GraphNodeRecord) -> GraphNodeRecord:
        return GraphNodeRecord(
            node_id=str(node.node_id),
            node_type=str(node.node_type),
            label=str(node.label),
            video_id=str(node.video_id),
            start_time=self._normalize_scalar(node.start_time),
            end_time=self._normalize_scalar(node.end_time),
            attributes=self._normalize_json_value(node.attributes),
            evidence_paths=[str(item) for item in self._normalize_json_value(node.evidence_paths)],
            keywords=[str(item) for item in self._normalize_json_value(node.keywords)],
        )

    def _normalize_edge(self, edge: GraphEdgeRecord) -> GraphEdgeRecord:
        return GraphEdgeRecord(
            edge_id=str(edge.edge_id),
            source_id=str(edge.source_id),
            target_id=str(edge.target_id),
            edge_type=str(edge.edge_type),
            video_id=str(edge.video_id),
            attributes=self._normalize_json_value(edge.attributes),
        )

    def _normalize_scalar(self, value: Any) -> float | None:
        value = self._normalize_json_value(value)
        if value is None:
            return None
        return float(value)

    def _normalize_json_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, dict):
            return {str(key): self._normalize_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._normalize_json_value(item) for item in value]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if hasattr(value, "item") and callable(value.item):
            try:
                return self._normalize_json_value(value.item())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                return self._normalize_json_value(value.tolist())
            except Exception:  # noqa: BLE001
                pass
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
