#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path


BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/"
OUTPUT_ROOT = Path("/22liushoulong/agent/hd-epic/data")
DATA_ROOT = OUTPUT_ROOT / "HD-EPIC"
MD5_FILE = Path("/22liushoulong/agent/hd-epic/hd-epic-downloader-main/data/md5.txt")
PROXY = "http://127.0.0.1:7890"
SEGMENT_SIZE = 128 * 1024 * 1024
MAX_WORKERS = 24


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def duration_fmt(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_md5s():
    result = {}
    with MD5_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rel = " ".join(parts[1:]).strip()
            if rel.startswith("./"):
                rel = rel[2:]
            result[rel] = parts[0]
    return result


def local_total(parts):
    total = 0
    for part in parts:
        if part.exists():
            total += part.stat().st_size
    return total


def download_segment(rel: str, start: int, end: int, seg_path: Path):
    expected = end - start + 1
    if seg_path.exists() and seg_path.stat().st_size == expected:
        return ("skip", rel, start, end)
    if seg_path.exists() and seg_path.stat().st_size > expected:
        seg_path.unlink()
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + urllib.parse.quote("./" + rel)
    cmd = [
        "curl",
        "--proxy",
        PROXY,
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "60",
        "--speed-time",
        "180",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-r",
        f"{start}-{end}",
        "-o",
        str(seg_path),
        url,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        return ("error", rel, start, end, proc.returncode)
    if seg_path.stat().st_size != expected:
        return ("error", rel, start, end, f"bad size {seg_path.stat().st_size} expected {expected}")
    return ("done", rel, start, end)


def reporter(stop_event, parts, total):
    start_time = time.time()
    last_time = start_time
    last_bytes = local_total(parts)
    initial = last_bytes
    while not stop_event.wait(3):
        now = time.time()
        done = local_total(parts)
        speed = max(done - last_bytes, 0) / max(now - last_time, 1e-6)
        avg = max(done - initial, 0) / max(now - start_time, 1e-6)
        eta_speed = speed or avg
        eta = (total - done) / eta_speed if eta_speed else float("inf")
        print(
            f"[seg-progress] {sizeof_fmt(done)}/{sizeof_fmt(total)} "
            f"({done / total * 100:.2f}%), speed {sizeof_fmt(speed)}/s, "
            f"avg {sizeof_fmt(avg)}/s, ETA {duration_fmt(eta)}"
        )
        sys.stdout.flush()
        last_time = now
        last_bytes = done


def main():
    md5s = load_md5s()
    sizes = json.loads((OUTPUT_ROOT / "hd_epic_no_vrs_sizes.json").read_text(encoding="utf-8"))
    part_files = sorted(DATA_ROOT.rglob("*.part"))
    if not part_files:
        print("No .part files found.")
        return

    tasks = []
    all_parts = []
    total_remaining_bundle = 0
    for part in part_files:
        rel = str(part.relative_to(DATA_ROOT))[:-5]
        dest = DATA_ROOT / rel
        total_size = int(sizes[rel])
        current_size = part.stat().st_size
        if current_size >= total_size:
            part.rename(dest)
            continue
        seg_dir = part.with_name(part.name + ".segments")
        # Keep the existing .part as segment 0, then download only missing byte ranges.
        all_parts.append(part)
        start = current_size
        while start < total_size:
            end = min(start + SEGMENT_SIZE - 1, total_size - 1)
            seg_path = seg_dir / f"{start}-{end}.part"
            tasks.append((rel, start, end, seg_path))
            all_parts.append(seg_path)
            start = end + 1
        total_remaining_bundle += total_size
        print(f"Queued {rel}: {sizeof_fmt(current_size)}/{sizeof_fmt(total_size)}")

    print(f"Segments queued: {len(tasks)}, workers: {MAX_WORKERS}")
    sys.stdout.flush()

    stop = threading.Event()
    report = threading.Thread(target=reporter, args=(stop, all_parts, total_remaining_bundle), daemon=True)
    report.start()
    errors = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(download_segment, *task) for task in tasks]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                if result[0] == "error":
                    errors.append(result)
                    print(f"[{idx}/{len(tasks)}] error: {result}")
                elif idx % 10 == 0 or idx == len(tasks):
                    print(f"[{idx}/{len(tasks)}] {result[0]}: {result[1]} {result[2]}-{result[3]}")
                sys.stdout.flush()
    finally:
        stop.set()

    if errors:
        print(f"Errors: {len(errors)}")
        sys.exit(1)

    for part in part_files:
        rel = str(part.relative_to(DATA_ROOT))[:-5]
        dest = DATA_ROOT / rel
        total_size = int(sizes[rel])
        seg_dir = part.with_name(part.name + ".segments")
        segment_paths = []
        for seg in seg_dir.glob("*.part"):
            start = int(seg.name.split("-", 1)[0])
            segment_paths.append((start, seg))
        segment_paths.sort()
        with dest.open("wb") as out:
            with part.open("rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            for _, seg in segment_paths:
                with seg.open("rb") as src:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
        if dest.stat().st_size != total_size:
            raise RuntimeError(f"bad merged size for {rel}")
        expected_md5 = md5s[rel]
        actual_md5 = md5sum(dest)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"md5 mismatch for {rel}: {actual_md5} != {expected_md5}")
        part.unlink()
        for _, seg in segment_paths:
            seg.unlink()
        seg_dir.rmdir()
        print(f"Merged and verified: {rel}")

    print("All segmented downloads completed and verified.")


if __name__ == "__main__":
    main()
