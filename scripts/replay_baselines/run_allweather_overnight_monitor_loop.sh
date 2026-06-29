#!/usr/bin/env bash
# Overnight forward paper monitor — updates status JSON every 15 minutes. No strategy changes.
set -euo pipefail
ROOT="/home/mystic/mystic"
PY="$ROOT/venv/bin/python3"
SCRIPT="$ROOT/scripts/replay_baselines/run_allweather_overnight_forward_status.py"
LOG="$ROOT/var/overnight_forward_monitor.log"
INTERVAL_SEC="${OVERNIGHT_MONITOR_INTERVAL_SEC:-900}"

mkdir -p "$ROOT/var"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) overnight monitor started interval=${INTERVAL_SEC}s" >> "$LOG"

while true; do
  if ! "$PY" "$SCRIPT" >> "$LOG" 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) status script failed" >> "$LOG"
  fi
  sleep "$INTERVAL_SEC"
done
