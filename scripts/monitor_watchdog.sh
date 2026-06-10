#!/bin/bash
# Waits for current monitor to finish, then starts extended run (does not kill active monitor).
set -euo pipefail
REPO="/home/mystic/mystic"
EXTEND_FILE="/tmp/mystic_monitor_run_until_epoch"
# Extend target: 72h from now (picked up by updated monitor_12h.sh on next loop if same PID gets replaced)
echo $(($(date +%s) + 259200)) > "$EXTEND_FILE"

while pgrep -f "scripts/monitor_12h.sh" >/dev/null 2>&1; do
    sleep 60
done

sleep 2
export MONITOR_DURATION_SEC=259200
export MONITOR_INTERVAL_SEC=300
nohup "$REPO/scripts/monitor_12h.sh" >> /tmp/mystic_monitor.log 2>&1 &
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] MONITOR_WATCHDOG started extended run PID=$!" >> /tmp/mystic_monitor.log
