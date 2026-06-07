"""Path helpers for the local HD-EPIC workspace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


VIDEO_ID_RE = re.compile(r"(P\d{2}-\d{8}-\d{6})")
PARTICIPANT_RE = re.compile(r"\b(P\d{2})\b")


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths and common path-derived metadata helpers."""

    config: ProjectConfig

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        return cls(ProjectConfig.from_env())

    @property
    def project_root(self) -> Path:
        return self.config.project_root

    @property
    def data_root(self) -> Path:
        return self.config.data_root

    @property
    def annotation_root(self) -> Path:
        return self.config.annotation_root

    @property
    def output_root(self) -> Path:
        return self.config.output_root

    @property
    def graph_memory_root(self) -> Path:
        return self.config.graph_memory_root

    @property
    def graph_agent_artifacts_root(self) -> Path:
        return self.config.graph_agent_artifacts_root

    @property
    def graph_agent_sessions_root(self) -> Path:
        return self.config.graph_agent_sessions_root

    @property
    def graph_agent_runs_root(self) -> Path:
        return self.config.graph_agent_runs_root

    def ensure_output_root(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        return self.output_root

    def relative_to_project(self, path: Path) -> str:
        path = path.resolve()
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()


def infer_participant_id(path: str | Path) -> str | None:
    text = Path(path).as_posix()
    match = PARTICIPANT_RE.search(text)
    return match.group(1) if match else None


def infer_video_id(path: str | Path) -> str | None:
    text = Path(path).as_posix()
    match = VIDEO_ID_RE.search(text)
    return match.group(1) if match else None


def infer_domain(path: str | Path) -> str:
    parts = Path(path).parts
    known = {
        "Videos",
        "Audio-HDF5",
        "audio-annotations",
        "high-level",
        "narrations-and-action-segments",
        "scene-and-object-movements",
        "eye-gaze-priming",
        "SLAM-and-Gaze",
        "Digital-Twin",
        "Hands-Masks",
        "vqa-benchmark",
        "AcquisitionGuidelines",
        "Consent Form",
    }
    for part in parts:
        if part in known:
            return part
    return "other"
