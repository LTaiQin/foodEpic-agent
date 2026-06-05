#!/usr/bin/env bash
set -euo pipefail

ROOT="/22liushoulong/agent/hd-epic"
cd "$ROOT"

SESSION_NAME="${1:-graph-agent-batch}"
shift || true

LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="$LOG_DIR/${SESSION_NAME}-${TIMESTAMP}.log"

CMD=("python" "$ROOT/scripts/run_graph_agent_batch.py" "$@")

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT' && PYTHONUNBUFFERED=1 \"${CMD[@]}\" 2>&1 | tee '$LOG_PATH'"

echo "session=$SESSION_NAME"
echo "log=$LOG_PATH"
echo "attach: tmux attach -t $SESSION_NAME"
