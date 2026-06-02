#!/usr/bin/env python3
"""Smoke test the configured OpenAI-compatible model endpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from openai import OpenAI

from food_agent.config import ModelConfig, load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".secrets" / "model.env")
    parser.add_argument("--prompt", default="请用一句话回答：food agent 的核心价值是什么？")
    parser.add_argument("--use-env-proxy", action="store_true", help="Keep HTTP(S)/ALL proxy environment variables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    if not args.use_env_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(key, None)
    cfg = ModelConfig.from_env()
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0,
    )
    print("model:", cfg.model)
    print("base_url:", cfg.base_url)
    print("answer:", response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
