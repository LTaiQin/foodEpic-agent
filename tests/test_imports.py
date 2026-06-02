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


def test_path_inference() -> None:
    path = Path("data/HD-EPIC/Videos/P08/P08-20240620-180825.mp4")
    assert infer_domain(path) == "Videos"
    assert infer_participant_id(path) == "P08"
    assert infer_video_id(path) == "P08-20240620-180825"

