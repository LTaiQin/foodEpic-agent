#!/usr/bin/env python3
"""Download VRS files one by one using curl with resume support."""

from __future__ import annotations

import argparse
import hashlib
import re
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


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def download_one(entry, output_root: Path, proxy: str, log_dir: Path, idx: int, total: int) -> tuple:
    expected_md5, rel, url = entry
    dest = output_root / "HD-EPIC" / rel
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", rel) + ".log")

    if dest.exists() and md5sum(dest) == expected_md5:
        return ("skip", rel)

    if dest.exists():
        dest.rename(part)

    cmd = [
        "curl", "-L", "--retry", "30", "--retry-all-errors",
        "--retry-connrefused", "--retry-delay", "5",
        "--retry-max-time", "86400",
        "--connect-timeout", "60",
        "--speed-time", "300", "--speed-limit", "1024",
        "-C", "-", "-o", str(part), url,
    ]
    if proxy:
        cmd[1:1] = ["--proxy", proxy, "--http1.1"]

    print(f"[{idx}/{total}] START: {rel}", flush=True)
    with log_path.open("ab") as log:
        log.write((" ".join(cmd) + "\n").encode("utf-8"))
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        return ("error", rel, f"curl exited {proc.returncode}")

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
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    downloader_root = Path(args.downloader_root).resolve()
    output_root = Path(args.output).resolve()
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(downloader_root / "data" / "md5.txt")
    print(f"Found {len(entries)} VRS files in manifest")

    pending = []
    for entry in entries:
        expected_md5, rel, url = entry
        dest = output_root / "HD-EPIC" / rel
        if dest.exists() and md5sum(dest) == expected_md5:
            continue
        pending.append(entry)

    print(f"Pending: {len(pending)} files to download")

    if args.limit > 0:
        pending = pending[:args.limit]
        print(f"Limiting to {args.limit} files")

    counts = {"skip": len(entries) - len(pending), "done": 0, "error": 0}

    for idx, entry in enumerate(pending, 1):
        rel = entry[1]
        result = download_one(entry, output_root, args.proxy, log_dir, idx, len(pending))
        status = result[0]
        counts[status] = counts.get(status, 0) + 1
        if status == "done":
            print(f"[{idx}/{len(pending)}] DONE: {rel}", flush=True)
        elif status == "error":
            print(f"[{idx}/{len(pending)}] ERROR: {rel} - {result[2]}", flush=True)
        sys.stdout.flush()

    print(f"\nSummary: {counts}")


if __name__ == "__main__":
    main()
