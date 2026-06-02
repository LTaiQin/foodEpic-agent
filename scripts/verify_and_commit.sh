#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"commit message\" [verify command...]" >&2
  exit 2
fi

message="$1"
shift || true

if [[ $# -gt 0 ]]; then
  echo "[verify] $*"
  "$@"
else
  if [[ -f pyproject.toml ]]; then
    echo "[verify] python -m compileall ."
    python -m compileall .
  else
    echo "[verify] no pyproject.toml; skip default verification"
  fi
fi

if git diff --quiet && git diff --cached --quiet; then
  echo "[commit] no changes"
  exit 0
fi

git add -A
git commit -m "$message"
echo "[commit] created: $message"

