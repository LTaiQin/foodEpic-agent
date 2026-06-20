#!/usr/bin/env python3
"""Download VRS files using aria2: 1 file with 16 connections, or 2 files with 16 connections each."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(md5_file: Path) -> list[tuple[str, str, str]]:
    entries = []
    with md5_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            expected_md5 = parts[0]
            rel = Path(" ".join(parts[1:]).strip())
            rel_str = str(rel)
            if rel_str.startswith("./"):
                rel_str = rel_str[2:]
            if not rel_str.startswith("VRS/") or not rel_str.endswith(".vrs"):
                continue
            url = BASE_URL + urllib.parse.quote(str(rel))
            entries.append((expected_md5, rel_str, url))
    return entries


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def download_one(entry, output_root: Path, proxy: str, connections: int, idx: int, total: int) -> tuple:
    expected_md5, rel, url = entry
    dest = output_root / "HD-EPIC" / rel
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return ("skip", rel)

    aria2_cmd = [
        "aria2c",
        f"--all-proxy={proxy}",
        "-x", str(connections),
        "-s", str(connections),
        "-k", "1M",
        "-c",
        "--file-allocation=none",
        "--retry-wait=10",
        "--max-tries=50",
        "--timeout=600",
        "--connect-timeout=120",
        "--lowest-speed-limit=0",
        "--summary-interval=0",
        "--console-log-level=warn",
        "-d", str(part.parent),
        "-o", part.name,
        url,
    ]

    print(f"[{idx}/{total}] START: {rel} ({connections} connections)", flush=True)
    proc = subprocess.run(aria2_cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return ("error", rel, f"aria2c exited {proc.returncode}")

    if not part.exists():
        return ("error", rel, "partial file not found after download")

    part.rename(dest)
    return ("done", rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloader-root", default="/22liushoulong/agent/hd-epic/hd-epic-downloader-main")
    parser.add_argument("--output", default="/22liushoulong/agent/hd-epic/data")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel downloads (1 or 2)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    downloader_root = Path(args.downloader_root).resolve()
    output_root = Path(args.output).resolve()

    entries = load_manifest(downloader_root / "data" / "md5.txt")
    print(f"Found {len(entries)} VRS files in manifest")

    pending = []
    for entry in entries:
        expected_md5, rel, url = entry
        dest = output_root / "HD-EPIC" / rel
        if dest.exists():
            continue
        pending.append(entry)

    print(f"Pending: {len(pending)} files to download")

    if args.limit > 0:
        pending = pending[:args.limit]
        print(f"Limiting to {args.limit} files")

    connections_per_file = 16 // args.parallel
    counts = {"skip": len(entries) - len(pending), "done": 0, "error": 0}

    print(f"Strategy: {args.parallel} files x {connections_per_file} connections each")

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        idx = 0
        while idx < len(pending):
            batch = pending[idx:idx + args.parallel]
            futures = {
                executor.submit(download_one, entry, output_root, args.proxy, connections_per_file, idx + i + 1, len(pending)): entry[1]
                for i, entry in enumerate(batch)
            }
            for future in as_completed(futures):
                result = future.result()
                status = result[0]
                counts[status] = counts.get(status, 0) + 1
                if status == "done":
                    print(f"  DONE: {result[1]}")
                elif status == "error":
                    print(f"  ERROR: {result[1]} - {result[2]}")
                sys.stdout.flush()
            idx += len(batch)

    print(f"\nSummary: {counts}")


if __name__ == "__main__":
    main()
