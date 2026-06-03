#!/usr/bin/env python3
"""Build a compact error analysis report from exported comparison errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True, help="Path to exported error JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Path to markdown output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.errors.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_family = Counter(row["task_family"] for row in rows)
    by_failure = defaultdict(Counter)
    for row in rows:
        by_failure[row["task_family"]][row.get("failure_type") or "none"] += 1

    lines = ["# Error Analysis", ""]
    lines.append(f"- total_errors: {len(rows)}")
    lines.append("")
    lines.append("## By Task Family")
    lines.append("")
    for family, count in by_family.most_common():
        lines.append(f"- {family}: {count}")
    lines.append("")
    lines.append("## Failure Types")
    lines.append("")
    for family, counts in by_failure.items():
        summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        lines.append(f"- {family}: {summary}")
    lines.append("")
    lines.append("## Representative Cases")
    lines.append("")
    for family, _count in by_family.most_common():
        for row in rows:
            if row["task_family"] != family:
                continue
            lines.append(f"### {family}")
            lines.append("")
            lines.append(f"- sample_id: {row['sample_id']}")
            lines.append(f"- prediction: {row['prediction']}")
            lines.append(f"- gold: {row['gold']}")
            lines.append(f"- failure_type: {row.get('failure_type')}")
            lines.append(f"- evidence_ids: {row.get('evidence_ids')}")
            lines.append(f"- question: {row['question']}")
            lines.append("")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
