#!/bin/bash
# Mystic health monitor — logs snapshots every 5 min. Supports extension via run-until file.
set -uo pipefail

REPO="/home/mystic/mystic"
LOG="/tmp/mystic_monitor.log"
INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-300}"
DURATION_SEC="${MONITOR_DURATION_SEC:-43200}"
EXTEND_FILE="/tmp/mystic_monitor_run_until_epoch"
END_EPOCH=$(( $(date +%s) + DURATION_SEC ))
PYTHON="${REPO}/venv/bin/python3"
API="http://localhost:8000"
METRICS_PY="${REPO}/scripts/monitor_metrics.py"

log_line() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"
}

effective_end_epoch() {
    if [ -f "$EXTEND_FILE" ]; then
        local ext
        ext=$(cat "$EXTEND_FILE" 2>/dev/null || echo "")
        if [ -n "$ext" ] && [ "$ext" -gt "$END_EPOCH" ] 2>/dev/null; then
            echo "$ext"
            return
        fi
    fi
    echo "$END_EPOCH"
}

count_procs() {
    local n=0
    pgrep -f "uvicorn backend.main:app" >/dev/null 2>&1 && n=$((n+1))
    pgrep -f "start_live_market_data.py" >/dev/null 2>&1 && n=$((n+1))
    pgrep -f "start_ai_signal_generator.py" >/dev/null 2>&1 && n=$((n+1))
    pgrep -f "start_portfolio_engine_integration.py" >/dev/null 2>&1 && n=$((n+1))
    pgrep -f "start_ai_market_context.py" >/dev/null 2>&1 && n=$((n+1))
    pgrep -f "start_ai_learning.py" >/dev/null 2>&1 && n=$((n+1))
    pgrep -f "backend.services.binance_scalp.runner" >/dev/null 2>&1 && n=$((n+1))
    echo "$n"
}

scan_log_errors() {
    local hits=0
    for f in /tmp/mystic_*.log; do
        [ -f "$f" ] || continue
        local c
        c=$(tail -n 50 "$f" 2>/dev/null | grep -ciE 'traceback|exception:' || true)
        hits=$((hits + c))
    done
    echo "$hits"
}

log_line "MONITOR_START duration=${DURATION_SEC}s interval=${INTERVAL_SEC}s repo=${REPO} git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

while [ "$(date +%s)" -lt "$(effective_end_epoch)" ]; do
    PROCS=$(count_procs)
    MEM=$(free -m 2>/dev/null | awk '/Mem:/ {printf "%d/%dMB", $3, $2}' || echo "?")
    DISK=$(df -h / 2>/dev/null | awk 'NR==2 {print $5 " used"}' || echo "?")
    DASH_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${API}/dashboard/" 2>/dev/null || echo "000")

    METRICS=$("$PYTHON" "$METRICS_PY" "$API" "$PROCS" "$DASH_CODE" "$MEM" "$DISK" 2>/dev/null || echo '{"metrics_error":true}')

    ERRS=$(scan_log_errors)
    ALERT=""
    [ "$PROCS" -lt 7 ] && ALERT="${ALERT} LOW_PROCS($PROCS/7)"
    [ "$DASH_CODE" != "200" ] && ALERT="${ALERT} DASH_$DASH_CODE"
    echo "$METRICS" | grep -qE 'fetch_error|metrics_error' && ALERT="${ALERT} API_FAIL"
    [ "$ERRS" -gt 2 ] && ALERT="${ALERT} LOG_ERRS($ERRS)"

    if [ -n "$ALERT" ]; then
        log_line "SNAPSHOT $METRICS ALERT:${ALERT# }"
    else
        log_line "SNAPSHOT $METRICS ok"
    fi

    sleep "$INTERVAL_SEC"
done

log_line "MONITOR_END — ready for recheck"
