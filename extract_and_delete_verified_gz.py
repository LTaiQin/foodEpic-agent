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


def gunzip_one(gz_path: Path, dry_run: bool):
    out_path = gz_path.with_suffix("")
    rel = str(gz_path)
    compressed_size = gz_path.stat().st_size

    if dry_run:
        return {
            "rel": rel,
            "compressed_size": compressed_size,
            "uncompressed_size": 0,
            "seconds": 0.0,
            "status": "dry-run",
        }

    started = time.time()
    bytes_out = 0
    with gzip.open(gz_path, "rb") as src, out_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            bytes_out += len(chunk)

    if bytes_out == 0:
        raise RuntimeError(f"empty decompressed output: {out_path}")
    if not out_path.exists() or out_path.stat().st_size != bytes_out:
        raise RuntimeError(f"bad output size for {out_path}")

    # Re-read gzip stream to ensure the archive is fully readable and not truncated.
    verify_bytes = 0
    with gzip.open(gz_path, "rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            verify_bytes += len(chunk)
    if verify_bytes != bytes_out:
        raise RuntimeError(f"verification size mismatch for {gz_path}")

    gz_path.unlink()
    return {
        "rel": rel,
        "compressed_size": compressed_size,
        "uncompressed_size": bytes_out,
        "deleted": True,
        "seconds": round(time.time() - started, 3),
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/22liushoulong/agent/hd-epic/data/HD-EPIC")
    parser.add_argument("--state", default="/22liushoulong/agent/hd-epic/extract_gz_state.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    state_path = Path(args.state)
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    gz_paths = sorted(root.rglob("*.gz"))
    total = len(gz_paths)
    total_compressed = sum(p.stat().st_size for p in gz_paths)
    print(f"Found {total} .gz files under {root}")
    print(f"Compressed size remaining: {sizeof_fmt(total_compressed)}")
    print(f"Free space: {sizeof_fmt(shutil.disk_usage(root).free)}")
    print(f"Workers: {args.workers}")
    sys.stdout.flush()

    errors = 0
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(gunzip_one, gz_path, args.dry_run): gz_path for gz_path in gz_paths}
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            gz_path = future_map[future]
            try:
                result = future.result()
                with lock:
                    state[result["rel"]] = result
                    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
                remaining = total - idx
                print(
                    f"[{idx}/{total}] {result['status']}: {gz_path} "
                    f"remaining={remaining} out={sizeof_fmt(result['uncompressed_size'])}"
                )
                print(f"Free space now: {sizeof_fmt(shutil.disk_usage(root).free)}")
                sys.stdout.flush()
            except Exception as exc:
                errors += 1
                print(f"[{idx}/{total}] error: {gz_path}")
                print(f"  {exc}")
                sys.stdout.flush()

    print(f"Finished. errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
