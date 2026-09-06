#!/bin/bash
# Mystic core-stack watchdog — checks the 7 canonical processes are alive.
# If any are missing, does a clean full restart via start_mystic.sh core
# (the script itself stops everything first, so a partial-restart here
# would fight it — a full restart is the correct, already-idempotent path).
#
# DEPLOY LOCK: if /run/mystic/deploy.lock (or the legacy
# /tmp/mystic_maintenance.lock) exists, this watchdog must NOT start or
# restart Mystic. That closes the observed race where cron restarted the
# stack after an operator had stopped it and checked out unverified code.
# start_mystic.sh is intentionally NOT gated — an approved start during a
# deploy must still work while the lock is held.
set -u

REPO="${MYSTIC_WATCHDOG_REPO:-/home/mystic/mystic}"
LOG="${MYSTIC_WATCHDOG_LOG:-$REPO/logs/watchdog_mystic.log}"
LOCK="${MYSTIC_WATCHDOG_FLOCK:-/run/mystic/watchdog.flock}"
DEPLOY_LOCK="${MYSTIC_DEPLOY_LOCK:-/run/mystic/deploy.lock}"
MAINTENANCE_LOCK="${MYSTIC_MAINTENANCE_LOCK:-/tmp/mystic_maintenance.lock}"
START_CMD="${MYSTIC_WATCHDOG_START_CMD:-}"

_utc_now() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

_log() {
    mkdir -p "$(dirname -- "$LOG")" 2>/dev/null || true
    echo "$(_utc_now) $*" >> "$LOG"
}

deploy_lock_held() {
    local path
    for path in "$DEPLOY_LOCK" "$MAINTENANCE_LOCK"; do
        [ -z "$path" ] && continue
        if [ -e "$path" ] || [ -L "$path" ]; then
            if [ ! -f "$path" ]; then
                _log "WATCHDOG_SUPPRESSED_MALFORMED_LOCK path=$path"
                return 0
            fi
            _log "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK path=$path"
            return 0
        fi
    done
    return 1
}

if deploy_lock_held; then
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

# Root crontab (*/3) and mystic crontab (*/5) both run this script. The flock
# must be writable by either, or one user silently disables the other.
if [ ! -d "$(dirname -- "$LOCK")" ]; then
    mkdir -p "$(dirname -- "$LOCK")" 2>/dev/null || LOCK="/tmp/mystic_watchdog.lock"
fi
if [ ! -e "$LOCK" ]; then
    : >"$LOCK" 2>/dev/null || true
    chmod 666 "$LOCK" 2>/dev/null || true
fi
exec 9>"$LOCK"
if ! flock -n 9; then
    exit 0
fi

missing=0
force_missing="${MYSTIC_WATCHDOG_FORCE_MISSING:-}"
if [ "$force_missing" = "1" ]; then
    missing=1
elif [ "$force_missing" = "0" ]; then
    missing=0
else
    for p in "${PATTERNS[@]}"; do
        if ! pgrep -f "$p" >/dev/null 2>&1; then
            missing=1
            _log "MISSING: $p"
        fi
    done
fi

if [ "$missing" -eq 1 ]; then
    # Re-check the deploy lock right before acting — closes the race where a
    # lock is created between the scan above and this point.
    if deploy_lock_held; then
        exit 0
    fi
    _log "restarting core stack"
    if [ -n "$START_CMD" ]; then
        # Test / operator override — never used in production crontab.
        bash -c "$START_CMD" >> "$LOG" 2>&1 || true
    else
        cd "$REPO" || exit 1
        ./start_mystic.sh core >> "$LOG" 2>&1
    fi
    _log "restart attempt complete"
fi
