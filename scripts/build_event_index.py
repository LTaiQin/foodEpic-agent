#!/usr/bin/env python3
"""Build normalized Parquet event-index tables from the manifest and annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.data_index import build_event_index
from food_agent.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=defaults.output_root / "dataset_manifest.parquet")
    parser.add_argument("--annotation-root", type=Path, default=defaults.annotation_root)
    parser.add_argument("--out-dir", type=Path, default=defaults.output_root / "event_index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest)
    tables = build_event_index(manifest, args.annotation_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = args.out_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"{name}: {len(table)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

