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


@dataclass(frozen=True)
class ModelConfig:
    """OpenAI-compatible model endpoint configuration."""

    model: str
    api_key: str
    base_url: str
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model=os.environ.get("FOOD_AGENT_MODEL", "gpt-5.4"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://www.cctq.ai/v1"),
            max_retries=int(os.environ.get("FOOD_AGENT_MODEL_MAX_RETRIES", "3")),
            retry_backoff_seconds=float(os.environ.get("FOOD_AGENT_MODEL_RETRY_BACKOFF_SECONDS", "2.0")),
        )


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding existing values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
