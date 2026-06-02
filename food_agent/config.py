"""Project configuration primitives."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime configuration for local HD-EPIC based experiments."""

    project_root: Path
    data_root: Path
    annotation_root: Path
    output_root: Path

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "ProjectConfig":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        data_root = Path(os.environ.get("HD_EPIC_DATA_ROOT", root / "data" / "HD-EPIC")).resolve()
        annotation_root = Path(
            os.environ.get(
                "HD_EPIC_ANNOTATION_ROOT",
                root / "annotations" / "hd-epic-annotations-main",
            )
        ).resolve()
        output_root = Path(os.environ.get("FOOD_AGENT_OUTPUT_ROOT", root / "outputs")).resolve()
        return cls(
            project_root=root,
            data_root=data_root,
            annotation_root=annotation_root,
            output_root=output_root,
        )

