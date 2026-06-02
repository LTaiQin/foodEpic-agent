#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/"
DEFAULT_TYPES = {
    "root",
    "videos",
    "digital-twin",
    "slam-and-gaze",
    "audio-hdf5",
    "hands-masks",
    "consent form",
    "acquisitionguidelines",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: Path) -> str:
    if path == Path(".") or len(path.suffixes) == 0:
        return ""

    what = path
    while str(what.parent) != ".":
        what = what.parent

    return "root" if what == path else what.name.lower()


def load_manifest(md5_file: Path):
    entries = []
    with md5_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            expected_md5 = parts[0]
            rel = Path(" ".join(parts[1:]).strip())
            kind = classify(rel)
            if kind not in DEFAULT_TYPES:
                continue

            rel_str = str(rel)
            if rel_str.startswith("./"):
                rel_str = rel_str[2:]

            if rel_str.lower().endswith(".vrs") or "/VRS/" in f"/{rel_str}":
                raise RuntimeError(f"VRS file unexpectedly selected: {rel_str}")

            url = BASE_URL + urllib.parse.quote(str(rel))
            entries.append((expected_md5, rel_str, url))
    return entries


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def duration_fmt(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def get_remote_size(url: str, proxy: str) -> int:
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, method="HEAD")
    with opener.open(request, timeout=120) as response:
        size = response.headers.get("Content-Length")
        if not size:
            raise RuntimeError(f"missing Content-Length for {url}")
        return int(size)


def load_sizes(entries, cache_path: Path, proxy: str, workers: int):
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    missing = [(rel, url) for _, rel, url in entries if rel not in cache]
    if missing:
        print(f"Fetching Content-Length for {len(missing)} files...")
        sys.stdout.flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 16)) as executor:
            future_to_rel = {
                executor.submit(get_remote_size, url, proxy): rel
                for rel, url in missing
            }
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_rel), 1):
                rel = future_to_rel[future]
                cache[rel] = future.result()
                if idx % 25 == 0 or idx == len(missing):
                    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
                    print(f"  sized {idx}/{len(missing)}")
                    sys.stdout.flush()

    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return {rel: int(cache[rel]) for _, rel, _ in entries}


def local_bytes(output_root: Path, rel: str, expected_size: int) -> int:
    dest = output_root / "HD-EPIC" / rel
    part = dest.with_name(dest.name + ".part")
    size = 0
    if dest.exists():
        size = max(size, dest.stat().st_size)
    if part.exists():
        size = max(size, part.stat().st_size)
    return min(size, expected_size)


def progress_reporter(stop_event, output_root: Path, entries, sizes, interval: int):
    started = time.time()
    initial_done = sum(local_bytes(output_root, rel, sizes[rel]) for _, rel, _ in entries)
    last_time = started
    last_done = initial_done
    total = sum(sizes.values())

    while not stop_event.wait(interval):
        now = time.time()
        done = sum(local_bytes(output_root, rel, sizes[rel]) for _, rel, _ in entries)
        delta = max(done - last_done, 0)
        speed = delta / max(now - last_time, 1e-6)
        overall_speed = max(done - initial_done, 0) / max(now - started, 1e-6)
        eta_speed = speed if speed > 0 else overall_speed
        eta = (total - done) / eta_speed if eta_speed > 0 else float("inf")
        complete_files = sum(
            1
            for _, rel, _ in entries
            if (output_root / "HD-EPIC" / rel).exists()
            and (output_root / "HD-EPIC" / rel).stat().st_size == sizes[rel]
        )
        part_files = sum(1 for _, rel, _ in entries if (output_root / "HD-EPIC" / rel).with_name(Path(rel).name + ".part").exists())
        print(
            f"[progress] {sizeof_fmt(done)}/{sizeof_fmt(total)} "
            f"({done / total * 100:0.2f}%), files {complete_files}/{len(entries)}, "
            f"partial {part_files}, speed {sizeof_fmt(speed)}/s, avg {sizeof_fmt(overall_speed)}/s, "
            f"ETA {duration_fmt(eta)}"
        )
        sys.stdout.flush()
        last_time = now
        last_done = done


def download_one(entry, output_root: Path, proxy: str, log_dir: Path):
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
        "curl",
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "60",
        "--speed-time",
        "300",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(part),
        url,
    ]
    if proxy:
        cmd[1:1] = ["--proxy", proxy, "--http1.1"]

    with log_path.open("ab") as log:
        log.write((" ".join(cmd) + "\n").encode("utf-8"))
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        return ("error", rel, f"curl exited {proc.returncode}; see {log_path}")

    actual_md5 = md5sum(part)
    if actual_md5 != expected_md5:
        return ("error", rel, f"md5 mismatch: expected {expected_md5}, got {actual_md5}")

    part.rename(dest)
    return ("done", rel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloader-root", default="/22liushoulong/agent/hd-epic/hd-epic-downloader-main")
    parser.add_argument("--output", default="/22liushoulong/agent/hd-epic/data")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=5)
    args = parser.parse_args()

    downloader_root = Path(args.downloader_root).resolve()
    output_root = Path(args.output).resolve()
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(downloader_root / "data" / "md5.txt")
    sizes = load_sizes(entries, output_root / "hd_epic_no_vrs_sizes.json", args.proxy, args.workers)
    print(f"Selected {len(entries)} non-VRS files")
    print(f"Output: {output_root / 'HD-EPIC'}")
    print(f"Workers: {args.workers}")
    print(f"Total size: {sizeof_fmt(sum(sizes.values()))}")
    sys.stdout.flush()

    counts = {"skip": 0, "done": 0, "error": 0}
    stop_event = threading.Event()
    reporter = threading.Thread(
        target=progress_reporter,
        args=(stop_event, output_root, entries, sizes, args.progress_interval),
        daemon=True,
    )
    reporter.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_rel = {
                executor.submit(download_one, entry, output_root, args.proxy, log_dir): entry[1]
                for entry in entries
            }
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_rel), 1):
                result = future.result()
                status = result[0]
                counts[status] = counts.get(status, 0) + 1
                print(f"[{idx}/{len(entries)}] {status}: {result[1]}")
                if status == "error":
                    print(f"  {result[2]}")
                sys.stdout.flush()
    finally:
        stop_event.set()

    print(f"Summary: {counts}")
    if counts.get("error", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
