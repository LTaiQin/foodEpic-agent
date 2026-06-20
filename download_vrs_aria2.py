#!/usr/bin/env python3
"""Download VRS files using aria2 with 32-thread per-file downloads."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.parse
from pathlib import Path

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


def download_one(entry, output_root: Path, proxy: str, threads: int) -> tuple:
    expected_md5, rel, url = entry
    dest = output_root / "HD-EPIC" / rel
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and md5sum(dest) == expected_md5:
        return ("skip", rel)

    if dest.exists():
        dest.rename(part)

    max_conn = min(threads, 16)
    aria2_cmd = [
        "aria2c",
        f"--all-proxy={proxy}",
        "-x", str(max_conn),
        "-s", str(threads),
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

    proc = subprocess.run(aria2_cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return ("error", rel, f"aria2c exited {proc.returncode}; stderr={proc.stderr.strip()[:200]}")

    if not part.exists():
        return ("error", rel, "partial file not found after download")

    actual_md5 = md5sum(part)
    if actual_md5 != expected_md5:
        part.unlink()
        return ("error", rel, f"md5 mismatch: expected {expected_md5}, got {actual_md5}")

    part.rename(dest)
    return ("done", rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloader-root", default="/22liushoulong/agent/hd-epic/hd-epic-downloader-main")
    parser.add_argument("--output", default="/22liushoulong/agent/hd-epic/data")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    downloader_root = Path(args.downloader_root).resolve()
    output_root = Path(args.output).resolve()

    entries = load_manifest(downloader_root / "data" / "md5.txt")
    print(f"Found {len(entries)} VRS files in manifest")

    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"Limiting to {args.limit} files")

    counts = {"skip": 0, "done": 0, "error": 0}
    for idx, entry in enumerate(entries, 1):
        rel = entry[1]
        print(f"[{idx}/{len(entries)}] {rel} ... ", end="", flush=True)
        result = download_one(entry, output_root, args.proxy, args.threads)
        status = result[0]
        counts[status] = counts.get(status, 0) + 1
        if status == "skip":
            print("skip")
        elif status == "done":
            print("done")
        else:
            print(f"error: {result[2]}")
        sys.stdout.flush()

    print(f"\nSummary: {counts}")
    if counts.get("error", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
