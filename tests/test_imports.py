from pathlib import Path

from food_agent import ProjectConfig, ProjectPaths
from food_agent.paths import infer_domain, infer_participant_id, infer_video_id


def test_project_config_defaults() -> None:
    config = ProjectConfig.from_env()
    assert config.project_root.name == "hd-epic"
    assert config.data_root.name == "HD-EPIC"


def test_project_paths_relative() -> None:
    paths = ProjectPaths.from_env()
    rel = paths.relative_to_project(paths.project_root / "food_agent" / "__init__.py")
    assert rel == "food_agent/__init__.py"


def test_project_config_runtime_roots_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FOOD_AGENT_GRAPH_MEMORY_ROOT", "/tmp/food-agent/graph-memory")
    monkeypatch.setenv("FOOD_AGENT_ARTIFACTS_ROOT", "/tmp/food-agent/artifacts")
    monkeypatch.setenv("FOOD_AGENT_SESSIONS_ROOT", "/tmp/food-agent/sessions")
    monkeypatch.setenv("FOOD_AGENT_RUNS_ROOT", "/tmp/food-agent/runs")
    paths = ProjectPaths.from_env()
    assert paths.graph_memory_root == Path("/tmp/food-agent/graph-memory")
    assert paths.graph_agent_artifacts_root == Path("/tmp/food-agent/artifacts")
    assert paths.graph_agent_sessions_root == Path("/tmp/food-agent/sessions")
    assert paths.graph_agent_runs_root == Path("/tmp/food-agent/runs")


def test_path_inference() -> None:
    path = Path("data/HD-EPIC/Videos/P08/P08-20240620-180825.mp4")
    assert infer_domain(path) == "Videos"
    assert infer_participant_id(path) == "P08"
    assert infer_video_id(path) == "P08-20240620-180825"
