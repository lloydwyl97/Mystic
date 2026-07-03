#!/bin/bash
# MYSTIC startup script
# Modes:
#   ./start_mystic.sh core|full  (canonical 24/7 — DAY top-4 + scalp paper, separate engines)
#   ./start_mystic.sh scalp      (scalp runner only — starts backend/live_md if needed)
#   ./start_mystic.sh backend|live_md|signal|portfolio|learning|ai_context|scalp
#
# Retired (exit 1): all, ai, collector, agents, ai_position_tracker, ai_outcome_bridge

set -u

MODE="${1:-core}"

cd /home/mystic/mystic || exit 1

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

LEGACY_PATTERNS=(
    "live_data_collector.py"
    "start_ai_ml_trading.py"
    "start_agent_orchestrator.py"
    "start_ai_position_tracker.py"
    "start_ai_outcome_bridge.py"
)

stop_by_pattern() {
    local pattern="$1"
    local pids
    pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "Stopping: $pattern"
    for pid in $pids; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
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
        if pgrep -f "$pattern" >/dev/null 2>&1; then
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
    nohup "$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > /tmp/mystic_backend.log 2>&1 &
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
            nohup "$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 >> /tmp/mystic_backend.log 2>&1 &
            sleep 2
            continue
        fi
        sleep 1
    done
    echo "ERROR: Backend API failed health check on :8000"
    if [ -f /tmp/mystic_backend.log ]; then
        tail -15 /tmp/mystic_backend.log
    fi
    return 1
}

start_live_md() {
    echo "Starting Live Market Data loops..."
    nohup "$PYTHON" start_live_market_data.py > /tmp/mystic_live_md.log 2>&1 &
    require_running "start_live_market_data.py" "Live Market Data" "/tmp/mystic_live_md.log" 20 1 || return 1
}

start_signal() {
    echo "Starting AI Signal Generator..."
    nohup "$PYTHON" start_ai_signal_generator.py > /tmp/mystic_signal.log 2>&1 &
    require_running "start_ai_signal_generator.py" "AI Signal Generator" "/tmp/mystic_signal.log" 20 1 || return 1
}

start_portfolio() {
    local log_mode="${1:-append}"
    echo "Starting Portfolio Engine Integration..."
    if [ "$log_mode" = "truncate" ]; then
        nohup "$PYTHON" start_portfolio_engine_integration.py > /tmp/mystic_portfolio.log 2>&1 &
    else
        nohup "$PYTHON" start_portfolio_engine_integration.py >> /tmp/mystic_portfolio.log 2>&1 &
    fi
    require_running "start_portfolio_engine_integration.py" "Portfolio Engine Integration" "/tmp/mystic_portfolio.log" 20 1 || return 1
}

start_learning() {
    echo "Starting AI Learning..."
    nice -n 10 nohup env \
        DAY_HISTORICAL_TRAIN_BASES="BTC,ETH,SOL,XRP" \
        DAY_HISTORICAL_TAIL_4H_BARS="480" \
        DAY_HISTORICAL_ANCHOR_STRIDE="2" \
        DAY_HISTORICAL_ROWS_PER_COLLECT="160" \
        "$PYTHON" start_ai_learning.py > /tmp/mystic_learning.log 2>&1 &
    require_running "start_ai_learning.py" "AI Learning" "/tmp/mystic_learning.log" 20 1 || return 1
}

start_ai_context() {
    echo "Starting AI Market Context..."
    nohup "$PYTHON" start_ai_market_context.py > /tmp/mystic_ai_context.log 2>&1 &
    require_running "start_ai_market_context.py" "AI Market Context" "/tmp/mystic_ai_context.log" 20 1 || return 1
}

start_scalp() {
    local scalp_paper="${SCALP_PAPER_ENABLED:-true}"
    local scalp_auto_arm="${SCALP_PAPER_AUTO_ARM:-true}"
    local scalp_fee="${SCALP_FEE_MODEL_VERIFIED:-true}"
    echo "Starting Scalp Paper Runner (SCALP_PAPER_ENABLED=${scalp_paper} AUTO_ARM=${scalp_auto_arm})..."
    nohup env SCALP_PAPER_ENABLED="${scalp_paper}" SCALP_PAPER_AUTO_ARM="${scalp_auto_arm}" \
        SCALP_FEE_MODEL_VERIFIED="${scalp_fee}" \
        "$PYTHON" -m backend.services.binance_scalp.runner > /tmp/mystic_scalp.log 2>&1 &
    require_running "backend.services.binance_scalp.runner" "Scalp Paper Runner" "/tmp/mystic_scalp.log" 20 1 || return 1
}

stop_live_md() { stop_by_pattern "start_live_market_data.py"; }
stop_signal() { stop_by_pattern "start_ai_signal_generator.py"; }
stop_portfolio() { stop_by_pattern "start_portfolio_engine_integration.py"; }
stop_learning() { stop_by_pattern "start_ai_learning.py"; }
stop_ai_context() { stop_by_pattern "start_ai_market_context.py"; }
stop_scalp() { stop_by_pattern "backend.services.binance_scalp.runner"; }

stop_core_stack() {
    stop_scalp
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
    if pgrep -f "$pattern" >/dev/null 2>&1; then
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
    sleep 1
    start_scalp || return 1

    echo ""
    echo "=========================================="
    echo "MYSTIC ${label} STACK STARTED (DAY top-4 + scalp paper)"
    echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
    echo "Services: Backend + LiveMD + Signal + Portfolio + Context + Learning + Scalp"
    echo "DAY and scalp are separate engines — PnL and scoreboard are not mixed."
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
    all|ai|collector|agents|ai_position_tracker|ai_outcome_bridge)
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
    scalp)
        echo "Mode: scalp (isolated — does not start/stop DAY portfolio stack)"
        stop_scalp
        sleep 1
        ensure_redis
        ensure_running_or_start "uvicorn backend.main:app" start_backend "Backend API" || exit 1
        sleep 2
        ensure_running_or_start "start_live_market_data.py" start_live_md "Live Market Data" || exit 1
        sleep 2
        start_scalp || exit 1
        echo ""
        echo "=========================================="
        echo "MYSTIC SCALP STACK STARTED"
        echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
        echo "API: /api/scalp/status  /api/scalp/strategies"
        echo "Log: /tmp/mystic_scalp.log"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [core|full|scalp|backend|live_md|signal|portfolio|learning|ai_context]"
        exit 1
        ;;
esac
