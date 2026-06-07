import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())


@pytest.fixture(autouse=True)
def isolate_agent_runtime_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOOD_AGENT_GRAPH_MEMORY_ROOT", (tmp_path / "graph_memory").as_posix())
    monkeypatch.setenv("FOOD_AGENT_ARTIFACTS_ROOT", (tmp_path / "graph_agent_artifacts").as_posix())
    monkeypatch.setenv("FOOD_AGENT_SESSIONS_ROOT", (tmp_path / "graph_agent_sessions").as_posix())
    monkeypatch.setenv("FOOD_AGENT_RUNS_ROOT", (tmp_path / "graph_agent_runs").as_posix())
