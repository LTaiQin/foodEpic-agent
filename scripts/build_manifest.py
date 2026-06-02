#!/usr/bin/env python3
"""Build a lightweight manifest for the local HD-EPIC dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from food_agent.loaders import csv_head_info, hdf5_metadata, json_summary, load_jsonl_head, video_metadata
from food_agent.paths import ProjectPaths, infer_domain, infer_participant_id, infer_video_id


@dataclass
class ManifestRow:
    path: str
    relative_path: str
    domain: str
    participant_id: str | None
    video_id: str | None
    file_type: str
    size_bytes: int
    status: str
    row_count: int | None = None
    notes: str | None = None
    metadata_json: str | None = None


def file_type(path: Path) -> str:
    if path.name.endswith(".csv.gz"):
        return "csv.gz"
    if path.name.endswith(".tar.gz"):
        return "tar.gz"
    return path.suffix.lstrip(".").lower() or "none"


def should_sample(path: Path) -> bool:
    name = path.name
    if name.endswith(".csv.gz") and name.startswith("semidense_"):
        return False
    return True


def inspect_file(path: Path, ftype: str) -> tuple[str, dict[str, Any] | None, str | None]:
    if not should_sample(path):
        return "deferred", None, "large semidense csv.gz; metadata only"
    try:
        if ftype == "mp4":
            meta = video_metadata(path)
            return meta.pop("status", "ok"), meta, None
        if ftype == "hdf5":
            return "ok", hdf5_metadata(path), None
        if ftype == "json":
            return "ok", json_summary(path), None
        if ftype == "jsonl":
            return "ok", {"sample_rows": load_jsonl_head(path, limit=3)}, None
        if ftype in {"csv", "csv.gz"}:
            return "ok", csv_head_info(path, limit=3), None
        return "ok", None, None
    except Exception as exc:  # noqa: BLE001 - manifest should record blocked files instead of failing.
        return "blocked", None, f"{type(exc).__name__}: {exc}"


def iter_files(data_root: Path, annotation_root: Path | None) -> list[Path]:
    roots = [data_root]
    if annotation_root and annotation_root.exists():
        roots.append(annotation_root)
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(files)


def build_manifest(
    data_root: Path,
    annotation_root: Path | None,
    project_paths: ProjectPaths,
    progress_every: int = 100,
) -> pd.DataFrame:
    rows: list[ManifestRow] = []
    files = iter_files(data_root, annotation_root)
    total = len(files)
    for index, path in enumerate(files, start=1):
        if progress_every and (index == 1 or index % progress_every == 0 or index == total):
            print(f"[manifest] inspecting {index}/{total}: {project_paths.relative_to_project(path)}", flush=True)
        ftype = file_type(path)
        status, metadata, notes = inspect_file(path, ftype)
        rows.append(
            ManifestRow(
                path=path.resolve().as_posix(),
                relative_path=project_paths.relative_to_project(path),
                domain=infer_domain(path),
                participant_id=infer_participant_id(path),
                video_id=infer_video_id(path),
                file_type=ftype,
                size_bytes=path.stat().st_size,
                status=status,
                notes=notes,
                metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def write_report(df: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HD-EPIC Data Format Report",
        "",
        f"- total files: {len(df)}",
        f"- total size bytes: {int(df['size_bytes'].sum())}",
        "",
        "## By Domain",
        "",
        df.groupby("domain").size().sort_values(ascending=False).to_markdown(),
        "",
        "## By File Type",
        "",
        df.groupby("file_type").size().sort_values(ascending=False).to_markdown(),
        "",
        "## By Status",
        "",
        df.groupby("status").size().sort_values(ascending=False).to_markdown(),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    defaults = ProjectPaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--annotation-root", type=Path, default=defaults.annotation_root)
    parser.add_argument("--out", type=Path, default=defaults.output_root / "dataset_manifest.parquet")
    parser.add_argument("--report", type=Path, default=defaults.output_root / "data_format_report.md")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_paths = ProjectPaths.from_env()
    df = build_manifest(
        args.data_root.resolve(),
        args.annotation_root.resolve(),
        project_paths,
        progress_every=args.progress_every,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    write_report(df, args.report)
    print(f"manifest rows: {len(df)}")
    print(f"manifest: {args.out}")
    print(f"report: {args.report}")
    print(df.groupby("status").size().sort_values(ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
