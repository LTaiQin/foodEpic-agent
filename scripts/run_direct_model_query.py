#!/usr/bin/env python3
"""Run a direct OpenAI-compatible model query with the configured model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.config import load_env_file
from food_agent.model_client import OpenAICompatibleModelClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".secrets" / "model.env")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--use-env-proxy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    client = OpenAICompatibleModelClient(use_env_proxy=args.use_env_proxy)
    response = client.complete([{"role": "user", "content": args.prompt}])
    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

