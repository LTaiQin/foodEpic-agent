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
    graph_memory_root: Path
    graph_agent_artifacts_root: Path
    graph_agent_sessions_root: Path
    graph_agent_runs_root: Path

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
        graph_memory_root = Path(
            os.environ.get("FOOD_AGENT_GRAPH_MEMORY_ROOT", output_root / "graph_memory")
        ).resolve()
        graph_agent_artifacts_root = Path(
            os.environ.get("FOOD_AGENT_ARTIFACTS_ROOT", output_root / "graph_agent_artifacts")
        ).resolve()
        graph_agent_sessions_root = Path(
            os.environ.get("FOOD_AGENT_SESSIONS_ROOT", output_root / "graph_agent_sessions")
        ).resolve()
        graph_agent_runs_root = Path(
            os.environ.get("FOOD_AGENT_RUNS_ROOT", output_root / "graph_agent_runs")
        ).resolve()
        return cls(
            project_root=root,
            data_root=data_root,
            annotation_root=annotation_root,
            output_root=output_root,
            graph_memory_root=graph_memory_root,
            graph_agent_artifacts_root=graph_agent_artifacts_root,
            graph_agent_sessions_root=graph_agent_sessions_root,
            graph_agent_runs_root=graph_agent_runs_root,
        )


@dataclass(frozen=True)
class ModelConfig:
    """OpenAI-compatible model endpoint configuration."""

    model: str
    api_key: str
    base_url: str
    provider_mode: str = "chat_completions"
    vision_provider_mode: str = "auto"
    max_retries: int = 3
    vision_max_retries: int = 0
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: float = 120.0
    vision_timeout_seconds: float = 35.0
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model=os.environ.get("FOOD_AGENT_MODEL", "gpt-5.4"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://icoe.pp.ua/v1"),
            provider_mode=os.environ.get("FOOD_AGENT_PROVIDER_MODE", "chat_completions"),
            vision_provider_mode=os.environ.get("FOOD_AGENT_VISION_PROVIDER_MODE", "auto"),
            max_retries=int(os.environ.get("FOOD_AGENT_MODEL_MAX_RETRIES", "3")),
            vision_max_retries=int(os.environ.get("FOOD_AGENT_VISION_MAX_RETRIES", "0")),
            retry_backoff_seconds=float(os.environ.get("FOOD_AGENT_MODEL_RETRY_BACKOFF_SECONDS", "2.0")),
            request_timeout_seconds=float(os.environ.get("FOOD_AGENT_MODEL_TIMEOUT_SECONDS", "120.0")),
            vision_timeout_seconds=float(os.environ.get("FOOD_AGENT_VISION_TIMEOUT_SECONDS", "35.0")),
            input_cost_per_million_tokens=float(os.environ.get("FOOD_AGENT_INPUT_COST_PER_MILLION_TOKENS", "0.0")),
            output_cost_per_million_tokens=float(os.environ.get("FOOD_AGENT_OUTPUT_COST_PER_MILLION_TOKENS", "0.0")),
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
