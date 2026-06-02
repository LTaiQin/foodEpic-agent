#!/usr/bin/env python3
"""Smoke test LightAgent against the configured OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.config import ModelConfig, load_env_file
from food_agent.lightagent_wrapper import FoodAgentLightWrapper, import_lightagent_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".secrets" / "model.env")
    parser.add_argument("--baseline", default="textonly")
    parser.add_argument("--question", default="请用一句话说明 food agent 的核心价值。")
    parser.add_argument("--use-env-proxy", action="store_true", help="Keep HTTP(S)/ALL proxy environment variables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    if not args.use_env_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key, None)
    cfg = ModelConfig.from_env()
    LightAgent = import_lightagent_class()
    agent = LightAgent(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        auto_discover_skills=False,
    )
    wrapper = FoodAgentLightWrapper(agent)
    result = wrapper.run(args.question, baseline=args.baseline)
    print("content:", result.content)
    print("task_family:", result.task_family)
    print("exposed_tools:", result.exposed_tools)
    print("trace_types:", [event["type"] for event in result.trace])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
