#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def safe_member_path(base: Path, member: str) -> Path:
    target = (base / member).resolve()
    base_resolved = base.resolve()
    if not str(target).startswith(str(base_resolved) + "/") and target != base_resolved:
        raise RuntimeError(f"unsafe zip member path: {member}")
    return target


def zip_uncompressed_size(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        return sum(info.file_size for info in zf.infolist() if not info.is_dir())


def verify_extracted(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = safe_member_path(dest_dir, info.filename)
            if info.is_dir():
                if not target.exists():
                    raise RuntimeError(f"missing extracted dir: {target}")
                continue
            if not target.is_file():
                raise RuntimeError(f"missing extracted file: {target}")
            actual_size = target.stat().st_size
            if actual_size != info.file_size:
                raise RuntimeError(
                    f"size mismatch for {target}: got {actual_size}, expected {info.file_size}"
                )


def extract_one(zip_path: Path, state: dict, dry_run: bool) -> str:
    dest_dir = zip_path.parent
    rel = str(zip_path)
    compressed_size = zip_path.stat().st_size
    uncompressed_size = zip_uncompressed_size(zip_path)

    print(
        f"[extract] {rel} compressed={sizeof_fmt(compressed_size)} "
        f"uncompressed={sizeof_fmt(uncompressed_size)}"
    )
    sys.stdout.flush()

    if dry_run:
        return "dry-run"

    cmd = ["unzip", "-o", "-q", str(zip_path), "-d", str(dest_dir)]
    started = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"unzip failed with exit code {proc.returncode}: {zip_path}")

    verify_extracted(zip_path, dest_dir)
    zip_path.unlink()

    state[rel] = {
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "deleted": True,
        "seconds": round(time.time() - started, 3),
    }
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/22liushoulong/agent/hd-epic/data/HD-EPIC")
    parser.add_argument("--state", default="/22liushoulong/agent/hd-epic/extract_zips_state.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    state_path = Path(args.state)
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    zip_paths = sorted(root.rglob("*.zip"))
    total = len(zip_paths)
    total_compressed = sum(p.stat().st_size for p in zip_paths)
    print(f"Found {total} zip files under {root}")
    print(f"Compressed size remaining: {sizeof_fmt(total_compressed)}")
    print(f"Free space: {sizeof_fmt(shutil.disk_usage(root).free)}")
    sys.stdout.flush()

    errors = 0
    for idx, zip_path in enumerate(zip_paths, 1):
        try:
            status = extract_one(zip_path, state, args.dry_run)
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            remaining = total - idx
            print(f"[{idx}/{total}] {status}: {zip_path} remaining={remaining}")
            print(f"Free space now: {sizeof_fmt(shutil.disk_usage(root).free)}")
            sys.stdout.flush()
        except Exception as exc:
            errors += 1
            print(f"[{idx}/{total}] error: {zip_path}")
            print(f"  {exc}")
            sys.stdout.flush()

    print(f"Finished. errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
