#!/bin/bash
# MYSTIC startup script
# Modes:
#   ./start_mystic.sh core|full  (canonical 24/7 — SCALP V2 + DAY V2 live via portfolio engine)
#   ./start_mystic.sh backend|live_md|signal|portfolio|learning|ai_context
#
# Retired (exit 1): all, ai, collector, agents, ai_position_tracker, ai_outcome_bridge, scalp

set -u

MODE="${1:-core}"

cd /home/mystic/mystic || exit 1

DEPLOY_LOCK="${MYSTIC_DEPLOY_LOCK:-/run/mystic/deploy.lock}"
MAINTENANCE_LOCK="${MYSTIC_MAINTENANCE_LOCK:-/tmp/mystic_maintenance.lock}"
if [ -e "$DEPLOY_LOCK" ] || [ -e "$MAINTENANCE_LOCK" ]; then
    echo "NOTE: deploy lock is held; starting approved processes while watchdog is suppressed"
fi

set -a
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    source .env
fi
if [ -f "deploy/core_only_local.env" ]; then
    # shellcheck disable=SC1091
    source deploy/core_only_local.env
fi
set +a

PYTHON="${PWD}/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv not found at $PYTHON"
    exit 1
fi

export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export DATABASE_URL="${DATABASE_URL:-sqlite:////home/mystic/mystic/mystic_trading.db}"
export PAPER_TRADING_INITIAL_BALANCE="${PAPER_TRADING_INITIAL_BALANCE:-10000.0}"
export RUN_ID="${RUN_ID:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${PWD}/logs"
mkdir -p "$LOG_DIR"
LIFECYCLE_LOCK="${LOG_DIR}/mystic_lifecycle.lock"

acquire_lifecycle_lock() {
    exec 9>"$LIFECYCLE_LOCK"
    if ! flock -w 120 9; then
        echo "ERROR: another start/stop is holding $LIFECYCLE_LOCK"
        exit 1
    fi
}

list_app_pids() {
    local pattern="$1"
    local pid cmd
    local self_pid=$$
    local parent_pid=${PPID:-}
    while read -r pid; do
        [ -z "$pid" ] && continue
        [ "$pid" = "$self_pid" ] && continue
        [ -n "$parent_pid" ] && [ "$pid" = "$parent_pid" ] && continue
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        [ -z "$cmd" ] && continue
        case "$cmd" in
            *start_mystic.sh*|*stop_mystic.sh*) continue ;;
        esac
        case "$cmd" in
            *bash*|*ssh*|*sudo\ -u*) continue ;;
        esac
        case "$cmd" in
            *python*|*uvicorn*) ;;
            *) continue ;;
        esac
        case "$cmd" in
            *"$pattern"*) echo "$pid" ;;
        esac
    done < <(pgrep -f "$pattern" 2>/dev/null)
}

process_count() {
    local pattern="$1"
    local n
    n="$(list_app_pids "$pattern" | wc -l)"
    echo "${n// /}"
}

refuse_duplicate_or_collapse() {
    local pattern="$1"
    local label="$2"
    local n
    n="$(process_count "$pattern")"
    n="${n:-0}"
    if [ "$n" -eq 1 ]; then
        echo "OK: $label already running — refusing duplicate start"
        return 0
    fi
    if [ "$n" -gt 1 ]; then
        echo "WARN: $label has $n copies — collapsing before launch"
        stop_by_pattern "$pattern"
    fi
    return 1
}

assert_single_after_start() {
    local pattern="$1"
    local label="$2"
    local n
    n="$(process_count "$pattern")"
    n="${n:-0}"
    if [ "$n" -gt 1 ]; then
        echo "WARN: $label count=$n after start — collapsing to one"
        stop_by_pattern "$pattern"
        return 1
    fi
    return 0
}

acquire_lifecycle_lock

LEGACY_PATTERNS=(
    "live_data_collector.py"
    "start_ai_ml_trading.py"
    "start_agent_orchestrator.py"
    "start_ai_position_tracker.py"
    "start_ai_outcome_bridge.py"
    # Retired paper scalp runner (2026-09-22): stopped on every core restart
    # so any lingering instance from before the paper removal is cleaned up.
    "backend.services.binance_scalp.runner"
)

stop_by_pattern() {
    local pattern="$1"
    local pids
    pids="$(list_app_pids "$pattern")"
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "Stopping: $pattern"
    for pid in $pids; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    pids="$(list_app_pids "$pattern")"
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
}

stop_legacy_processes() {
    local pattern
    for pattern in "${LEGACY_PATTERNS[@]}"; do
        stop_by_pattern "$pattern"
    done
}

require_running() {
    local pattern="$1"
    local label="$2"
    local log_path="${3:-}"
    local tries="${4:-12}"
    local sleep_sec="${5:-1}"
    local i
    for ((i=1; i<=tries; i++)); do
        if [ "$(process_count "$pattern")" -ge 1 ]; then
            echo "OK: $label running"
            return 0
        fi
        sleep "$sleep_sec"
    done
    echo "ERROR: $label failed to start"
    if [ -n "$log_path" ] && [ -f "$log_path" ]; then
        echo "Inspect log: $log_path"
    fi
    return 1
}

uvicorn_process_count() {
    pgrep -f 'venv/bin/python -m uvicorn backend.main:app' 2>/dev/null | wc -l
}

port_8000_listener_count() {
    ss -ltnp 2>/dev/null | grep -c ':8000 ' || true
}

port_8000_pids() {
    ss -ltnp 2>/dev/null | grep ':8000 ' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u
}

backend_health_ok() {
    curl -sf --max-time 5 http://127.0.0.1:8000/api/system/health/quick >/dev/null 2>&1
}

stop_backend() {
    stop_conflicting_systemd_uvicorn
    stop_by_pattern "uvicorn backend.main:app"
    local pid
    for pid in $(port_8000_pids); do
        [ -n "$pid" ] || continue
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in $(port_8000_pids); do
        [ -n "$pid" ] || continue
        kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1
    if [ "$(port_8000_listener_count)" -gt 0 ]; then
        echo "WARNING: port 8000 still in use after stop_backend"
        ss -ltnp 2>/dev/null | grep ':8000 ' || true
        return 1
    fi
}

start_backend() {
    stop_conflicting_systemd_uvicorn

    uv_count="$(uvicorn_process_count)"
    uv_count="${uv_count// /}"
    listener_count="$(port_8000_listener_count)"
    listener_count="${listener_count:-0}"

    if [ "$listener_count" -gt 1 ] || [ "$uv_count" -gt 1 ]; then
        echo "WARN: uvicorn processes=$uv_count listeners_on_8000=$listener_count — resetting to single backend"
        stop_backend || return 1
    elif backend_health_ok; then
        echo "OK: Backend API already healthy on :8000 (1 listener)"
        return 0
    elif [ "$uv_count" -ge 1 ] || [ "$listener_count" -ge 1 ]; then
        echo "Stopping stale/unhealthy backend on :8000..."
        stop_backend || return 1
    fi

    echo "Starting Backend API..."
    nohup "$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > /home/mystic/mystic/logs/mystic_backend.log 2>&1 9>&- &
    local i
    for ((i=1; i<=30; i++)); do
        if backend_health_ok; then
            listener_count="$(port_8000_listener_count)"
            uv_count="$(uvicorn_process_count)"
            uv_count="${uv_count// /}"
            if [ "$listener_count" -eq 1 ] && [ "$uv_count" -eq 1 ]; then
                echo "OK: Backend API running (1 process, 1 listener on :8000)"
                return 0
            fi
            echo "WARN: health OK but uvicorn=$uv_count listeners=$listener_count — resetting"
            stop_backend || return 1
            nohup "$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 >> /home/mystic/mystic/logs/mystic_backend.log 2>&1 9>&- &
            sleep 2
            continue
        fi
        sleep 1
    done
    echo "ERROR: Backend API failed health check on :8000"
    if [ -f /home/mystic/mystic/logs/mystic_backend.log ]; then
        tail -15 /home/mystic/mystic/logs/mystic_backend.log
    fi
    return 1
}

start_live_md() {
    if refuse_duplicate_or_collapse "start_live_market_data.py" "Live Market Data"; then
        return 0
    fi
    echo "Starting Live Market Data loops..."
    nohup "$PYTHON" start_live_market_data.py > /home/mystic/mystic/logs/mystic_live_md.log 2>&1 9>&- &
    require_running "start_live_market_data.py" "Live Market Data" "/home/mystic/mystic/logs/mystic_live_md.log" 20 1 || return 1
    if ! assert_single_after_start "start_live_market_data.py" "Live Market Data"; then
        nohup "$PYTHON" start_live_market_data.py >> /home/mystic/mystic/logs/mystic_live_md.log 2>&1 9>&- &
        require_running "start_live_market_data.py" "Live Market Data" "/home/mystic/mystic/logs/mystic_live_md.log" 20 1 || return 1
    fi
}

start_signal() {
    if refuse_duplicate_or_collapse "start_ai_signal_generator.py" "AI Signal Generator"; then
        return 0
    fi
    echo "Starting AI Signal Generator..."
    nohup "$PYTHON" start_ai_signal_generator.py > /home/mystic/mystic/logs/mystic_signal.log 2>&1 9>&- &
    require_running "start_ai_signal_generator.py" "AI Signal Generator" "/home/mystic/mystic/logs/mystic_signal.log" 20 1 || return 1
    if ! assert_single_after_start "start_ai_signal_generator.py" "AI Signal Generator"; then
        nohup "$PYTHON" start_ai_signal_generator.py >> /home/mystic/mystic/logs/mystic_signal.log 2>&1 9>&- &
        require_running "start_ai_signal_generator.py" "AI Signal Generator" "/home/mystic/mystic/logs/mystic_signal.log" 20 1 || return 1
    fi
}

start_portfolio() {
    local log_mode="${1:-append}"
    if refuse_duplicate_or_collapse "start_portfolio_engine_integration.py" "Portfolio Engine Integration"; then
        return 0
    fi
    echo "Starting Portfolio Engine Integration..."
    if [ "$log_mode" = "truncate" ]; then
        nohup "$PYTHON" start_portfolio_engine_integration.py > /home/mystic/mystic/logs/mystic_portfolio.log 2>&1 9>&- &
    else
        nohup "$PYTHON" start_portfolio_engine_integration.py >> /home/mystic/mystic/logs/mystic_portfolio.log 2>&1 9>&- &
    fi
    require_running "start_portfolio_engine_integration.py" "Portfolio Engine Integration" "/home/mystic/mystic/logs/mystic_portfolio.log" 20 1 || return 1
    if ! assert_single_after_start "start_portfolio_engine_integration.py" "Portfolio Engine Integration"; then
        nohup "$PYTHON" start_portfolio_engine_integration.py >> /home/mystic/mystic/logs/mystic_portfolio.log 2>&1 9>&- &
        require_running "start_portfolio_engine_integration.py" "Portfolio Engine Integration" "/home/mystic/mystic/logs/mystic_portfolio.log" 20 1 || return 1
    fi
}

_launch_learning() {
    echo "Starting AI Learning..."
    nice -n 10 nohup env \
        DAY_HISTORICAL_TRAIN_BASES="BTC,ETH,SOL,XRP" \
        DAY_HISTORICAL_TAIL_4H_BARS="480" \
        DAY_HISTORICAL_ANCHOR_STRIDE="2" \
        DAY_HISTORICAL_ROWS_PER_COLLECT="160" \
        "$PYTHON" start_ai_learning.py > /home/mystic/mystic/logs/mystic_learning.log 2>&1 9>&- &
}

start_learning() {
    if refuse_duplicate_or_collapse "start_ai_learning.py" "AI Learning"; then
        return 0
    fi
    _launch_learning
    require_running "start_ai_learning.py" "AI Learning" "/home/mystic/mystic/logs/mystic_learning.log" 20 1 || return 1
    if ! assert_single_after_start "start_ai_learning.py" "AI Learning"; then
        _launch_learning
        require_running "start_ai_learning.py" "AI Learning" "/home/mystic/mystic/logs/mystic_learning.log" 20 1 || return 1
    fi
}

start_ai_context() {
    if refuse_duplicate_or_collapse "start_ai_market_context.py" "AI Market Context"; then
        return 0
    fi
    echo "Starting AI Market Context..."
    nohup "$PYTHON" start_ai_market_context.py > /home/mystic/mystic/logs/mystic_ai_context.log 2>&1 9>&- &
    require_running "start_ai_market_context.py" "AI Market Context" "/home/mystic/mystic/logs/mystic_ai_context.log" 20 1 || return 1
    if ! assert_single_after_start "start_ai_market_context.py" "AI Market Context"; then
        nohup "$PYTHON" start_ai_market_context.py >> /home/mystic/mystic/logs/mystic_ai_context.log 2>&1 9>&- &
        require_running "start_ai_market_context.py" "AI Market Context" "/home/mystic/mystic/logs/mystic_ai_context.log" 20 1 || return 1
    fi
}

# _launch_scalp / start_scalp removed 2026-09-22 — paper runner retired.
# binance_scalp.runner is now in LEGACY_PATTERNS and is stopped by
# stop_legacy_processes on every core restart.

start_checkpoint_monitor() {
    # Read-only 100-trade SCALP V2 checkpoint monitor.
    # PID-locked singleton: a second invocation exits immediately.
    if refuse_duplicate_or_collapse "scalp_v2_checkpoint_monitor.py" "SCALP V2 Checkpoint Monitor"; then
        return 0
    fi
    echo "Starting SCALP V2 Checkpoint Monitor..."
    nohup "$PYTHON" scripts/scalp_v2_checkpoint_monitor.py --poll 120 \
        > /home/mystic/mystic/logs/scalp_v2_monitor.log 2>&1 9>&- &
    # Monitor is read-only and non-critical — do not gate startup on it
    sleep 1
    local n
    n="$(process_count "scalp_v2_checkpoint_monitor.py")"
    if [ "$n" -ge 1 ]; then
        echo "OK: SCALP V2 Checkpoint Monitor started"
    else
        echo "WARN: SCALP V2 Checkpoint Monitor did not start (non-critical, continuing)"
    fi
}

stop_live_md() { stop_by_pattern "start_live_market_data.py"; }
stop_signal() { stop_by_pattern "start_ai_signal_generator.py"; }
stop_portfolio() { stop_by_pattern "start_portfolio_engine_integration.py"; }
stop_learning() { stop_by_pattern "start_ai_learning.py"; }
stop_ai_context() { stop_by_pattern "start_ai_market_context.py"; }

stop_core_stack() {
    stop_backend
    stop_live_md
    stop_signal
    stop_portfolio
    stop_learning
    stop_ai_context
    stop_legacy_processes
}

ensure_redis() {
    if ! pgrep -x "redis-server" >/dev/null; then
        echo "Starting Redis..."
        systemctl start redis-server 2>/dev/null || service redis-server start 2>/dev/null || true
        sleep 2
    fi
}

stop_conflicting_systemd_uvicorn() {
    systemctl --user stop mystic.service 2>/dev/null || true
    systemctl --user stop mystic-uvicorn.service 2>/dev/null || true
    systemctl --user stop mystic.target 2>/dev/null || true
    if systemctl --user is-enabled --quiet mystic.service 2>/dev/null; then
        echo "Disabling mystic.service (use ./start_mystic.sh core — not systemd uvicorn)"
        systemctl --user disable mystic.service 2>/dev/null || true
    fi
    if systemctl --user is-enabled --quiet mystic-uvicorn.service 2>/dev/null; then
        echo "Disabling mystic-uvicorn.service (use ./start_mystic.sh core)"
        systemctl --user disable mystic-uvicorn.service 2>/dev/null || true
    fi
    sleep 1
}

ensure_running_or_start() {
    local pattern="$1"
    local start_fn="$2"
    local label="$3"
    if [ "$(process_count "$pattern")" -ge 1 ]; then
        echo "OK: $label already running"
        return 0
    fi
    "$start_fn" || return 1
}

run_core_stack() {
    local label="$1"
    stop_conflicting_systemd_uvicorn
    stop_core_stack
    sleep 2
    ensure_redis

    start_backend || return 1
    sleep 3
    start_live_md || return 1
    sleep 2
    start_signal || return 1
    sleep 2
    start_portfolio truncate || return 1
    sleep 2
    start_ai_context || return 1
    sleep 1
    start_learning || return 1
    # SCALP V2 live order authority is built into the portfolio engine (execute_buy_fifo).
    # The old paper binance_scalp.runner is fully retired (2026-09-22).
    # 100-trade checkpoint monitor: read-only, PID-locked singleton.
    start_checkpoint_monitor

    echo ""
    echo "=========================================="
    echo "MYSTIC ${label} STACK STARTED (SCALP V2 + DAY V2 live)"
    echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
    echo "Services: Backend + LiveMD + Signal + Portfolio + Context + Learning + Monitor"
    echo "SCALP V2: live order authority via portfolio engine. Paper scalp removed."
    echo "DAY V2:   live entry authority via trailing-buy intent machinery."
    echo "Ensure .env has EXTERNAL_SUPERVISOR_MODE=true"
    echo "=========================================="
}

retired_mode() {
    echo "ERROR: Mode '$1' is retired. Use './start_mystic.sh core'."
    echo "Retired launchers: start_ai_ml_trading.py, live_data_collector.py, start_agent_orchestrator.py,"
    echo "  start_ai_position_tracker.py, start_ai_outcome_bridge.py — see CANONICAL_SYSTEM.md"
    exit 1
}

case "$MODE" in
    core)
        echo "Mode: core (canonical 24/7)"
        run_core_stack "CORE" || exit 1
        ;;
    full)
        echo "Mode: full (alias of core)"
        systemctl --user stop mystic.target 2>/dev/null || true
        run_core_stack "FULL" || exit 1
        ;;
    all|ai|collector|agents|ai_position_tracker|ai_outcome_bridge|scalp)
        retired_mode "$MODE"
        ;;
    ai_context)
        echo "Mode: ai_context"
        stop_ai_context
        sleep 1
        start_ai_context || exit 1
        ;;
    portfolio)
        echo "Mode: portfolio"
        stop_portfolio
        sleep 1
        start_portfolio truncate || exit 1
        ;;
    learning)
        echo "Mode: learning"
        stop_learning
        sleep 1
        start_learning || exit 1
        ;;
    live_md)
        echo "Mode: live_md"
        stop_live_md
        sleep 1
        start_live_md || exit 1
        ;;
    signal)
        echo "Mode: signal"
        stop_signal
        sleep 1
        start_signal || exit 1
        ;;
    backend)
        echo "Mode: backend"
        stop_conflicting_systemd_uvicorn
        stop_backend
        sleep 1
        start_backend || exit 1
        ;;
    # scalp mode was removed here — it is now in the retired_mode case above.
    *)
        echo "Usage: $0 [core|full|backend|live_md|signal|portfolio|learning|ai_context]"
        exit 1
        ;;
esac
