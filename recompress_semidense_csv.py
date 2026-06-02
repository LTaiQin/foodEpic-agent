#!/usr/bin/env python3
import argparse
import concurrent.futures
import gzip
import json
import shutil
import sys
import threading
import time
from pathlib import Path


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def recompress_one(csv_path: Path, dry_run: bool):
    gz_path = csv_path.with_suffix(csv_path.suffix + ".gz")
    rel = str(csv_path)
    csv_size = csv_path.stat().st_size

    if dry_run:
        return {
            "rel": rel,
            "csv_size": csv_size,
            "gz_size": 0,
            "seconds": 0.0,
            "status": "dry-run",
        }

    started = time.time()
    bytes_in = 0
    with csv_path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            bytes_in += len(chunk)

    if bytes_in != csv_size:
        raise RuntimeError(f"bytes written mismatch for {csv_path}")

    verify_bytes = 0
    with gzip.open(gz_path, "rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            verify_bytes += len(chunk)
    if verify_bytes != csv_size:
        raise RuntimeError(f"verification mismatch for {gz_path}")

    gz_size = gz_path.stat().st_size
    csv_path.unlink()
    return {
        "rel": rel,
        "csv_size": csv_size,
        "gz_size": gz_size,
        "seconds": round(time.time() - started, 3),
        "deleted_csv": True,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/22liushoulong/agent/hd-epic/data/HD-EPIC")
    parser.add_argument("--state", default="/22liushoulong/agent/hd-epic/recompress_semidense_state.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    state_path = Path(args.state)
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    csv_paths = sorted(
        p for p in root.rglob("*.csv")
        if p.name in {"semidense_points.csv", "semidense_observations.csv"}
    )
    total = len(csv_paths)
    total_bytes = sum(p.stat().st_size for p in csv_paths)
    print(f"Found {total} semidense csv files under {root}")
    print(f"CSV size remaining: {sizeof_fmt(total_bytes)}")
    print(f"Free space: {sizeof_fmt(shutil.disk_usage(root).free)}")
    print(f"Workers: {args.workers}")
    sys.stdout.flush()

    errors = 0
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(recompress_one, csv_path, args.dry_run): csv_path for csv_path in csv_paths}
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            csv_path = future_map[future]
            try:
                result = future.result()
                with lock:
                    state[result["rel"]] = result
                    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
                remaining = total - idx
                ratio = (result["gz_size"] / result["csv_size"] * 100) if result["csv_size"] else 0
                print(
                    f"[{idx}/{total}] {result['status']}: {csv_path} remaining={remaining} "
                    f"before={sizeof_fmt(result['csv_size'])} after={sizeof_fmt(result['gz_size'])} "
                    f"ratio={ratio:0.1f}%"
                )
                print(f"Free space now: {sizeof_fmt(shutil.disk_usage(root).free)}")
                sys.stdout.flush()
            except Exception as exc:
                errors += 1
                print(f"[{idx}/{total}] error: {csv_path}")
                print(f"  {exc}")
                sys.stdout.flush()

    print(f"Finished. errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
