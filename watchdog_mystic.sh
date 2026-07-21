#!/bin/bash
# Mystic core-stack watchdog — checks the 7 canonical processes are alive.
# If any are missing, does a clean full restart via start_mystic.sh core
# (the script itself stops everything first, so a partial-restart here
# would fight it — a full restart is the correct, already-idempotent path).
#
# MAINTENANCE_LOCK: if /tmp/mystic_maintenance.lock exists, skip entirely.
# A human/agent doing manual stop/pull/start work must create this lock
# first, or this watchdog can race a manual restart mid-flight (observed:
# both invocations spawn processes concurrently -> duplicate PIDs per
# service, since only the backend port-listener check enforces
# single-instance; the other launchers just check "is anything matching
# already running").
set -u

REPO="/home/mystic/mystic"
LOG="$REPO/logs/watchdog_mystic.log"
LOCK="/tmp/mystic_watchdog.lock"
MAINTENANCE_LOCK="/tmp/mystic_maintenance.lock"

if [ -e "$MAINTENANCE_LOCK" ]; then
    exit 0
fi

PATTERNS=(
    "venv/bin/python -m uvicorn backend.main:app"
    "start_live_market_data.py"
    "start_ai_signal_generator.py"
    "start_portfolio_engine_integration.py"
    "start_ai_market_context.py"
    "start_ai_learning.py"
    "backend.services.binance_scalp.runner"
)

exec 9>"$LOCK"
if ! flock -n 9; then
    exit 0
fi

missing=0
for p in "${PATTERNS[@]}"; do
    if ! pgrep -f "$p" >/dev/null 2>&1; then
        missing=1
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) MISSING: $p" >> "$LOG"
    fi
done

if [ "$missing" -eq 1 ]; then
    # Re-check the maintenance lock right before acting — closes the race
    # where a lock is created between the loop above and this point.
    if [ -e "$MAINTENANCE_LOCK" ]; then
        exit 0
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) restarting core stack" >> "$LOG"
    cd "$REPO" || exit 1
    ./start_mystic.sh core >> "$LOG" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) restart attempt complete" >> "$LOG"
fi
