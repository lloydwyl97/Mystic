#!/bin/bash
# MYSTIC startup script
# Modes:
#   ./start_mystic.sh full      (recommended — DAY core + scalp paper; desktop shortcut)
#   ./start_mystic.sh core      (DAY stack only — external supervisor, no duplicates)
#   ./start_mystic.sh scalp     (scalp paper runner only — uses running backend/live_md if up)
#   ./start_mystic.sh all       (legacy agents/cleanup — do not use for 24/7)
#   ./start_mystic.sh backend|live_md|signal|portfolio|learning|collector|ai_context|...

set -u

MODE="${1:-core}"

cd /home/mystic/mystic || exit 1

# Load .env then layered local-only flags (same order as scripts/phase1_local_core_stack.sh).
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

start_backend() {
    echo "Starting Backend API..."
    nohup "$PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > /tmp/mystic_backend.log 2>&1 &
    require_running "uvicorn backend.main:app" "Backend API" "/tmp/mystic_backend.log" 20 1 || return 1
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

start_ai() {
    echo "Starting AI ML Trading..."
    nohup "$PYTHON" start_ai_ml_trading.py > /tmp/mystic_ai.log 2>&1 &
    require_running "start_ai_ml_trading.py" "AI ML Trading" "/tmp/mystic_ai.log" 20 1 || return 1
}

# For explicit paper-engine test env (MYSTIC_* / MAX_SPREAD_PCT), use:
#   ./scripts/start_portfolio_paper_test.sh
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

start_collector() {
    echo "Starting Data Collector..."
    nohup "$PYTHON" live_data_collector.py > /tmp/mystic_collector.log 2>&1 &
    require_running "live_data_collector.py" "Data Collector" "/tmp/mystic_collector.log" 20 1 || return 1
}

start_agents() {
    echo "Starting Agent Orchestrator..."
    nohup "$PYTHON" start_agent_orchestrator.py > /tmp/mystic_agents.log 2>&1 &
    require_running "start_agent_orchestrator.py" "Agent Orchestrator" "/tmp/mystic_agents.log" 20 1 || return 1
}

start_learning() {
    echo "Starting AI Learning..."
    nice -n 10 nohup "$PYTHON" start_ai_learning.py > /tmp/mystic_learning.log 2>&1 &
    require_running "start_ai_learning.py" "AI Learning" "/tmp/mystic_learning.log" 20 1 || return 1
}

start_ai_context() {
    echo "Starting AI Market Context..."
    nohup "$PYTHON" start_ai_market_context.py > /tmp/mystic_ai_context.log 2>&1 &
    require_running "start_ai_market_context.py" "AI Market Context" "/tmp/mystic_ai_context.log" 20 1 || return 1
}

# DISABLED: backend/services/ai_position_tracker.py is not shipped; not part of DAY engine.
start_ai_position_tracker() {
    echo "SKIP: AI Position Tracker disabled (implementation not present; not part of DAY engine)"
    return 0
}

# DISABLED: backend/services/ai_outcome_bridge.py is not shipped; not part of DAY engine.
start_ai_outcome_bridge() {
    echo "SKIP: AI Outcome Bridge disabled (implementation not present; not part of DAY engine)"
    return 0
}

stop_live_md() {
    stop_by_pattern "start_live_market_data.py"
}

stop_signal() {
    stop_by_pattern "start_ai_signal_generator.py"
}

stop_backend() {
    stop_by_pattern "uvicorn backend.main:app"
}

stop_ai() {
    stop_by_pattern "start_ai_ml_trading.py"
}

stop_portfolio() {
    stop_by_pattern "start_portfolio_engine_integration.py"
}

stop_collector() {
    stop_by_pattern "live_data_collector.py"
}

stop_agents() {
    stop_by_pattern "start_agent_orchestrator.py"
}

stop_learning() {
    stop_by_pattern "start_ai_learning.py"
}

stop_ai_context() {
    stop_by_pattern "start_ai_market_context.py"
}

stop_ai_position_tracker() {
    stop_by_pattern "start_ai_position_tracker.py"
}

stop_ai_outcome_bridge() {
    stop_by_pattern "start_ai_outcome_bridge.py"
}

stop_scalp() {
    stop_by_pattern "backend.services.binance_scalp.runner"
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

start_scalp() {
    # Scalp mode enables paper on the runner child only (.env may keep false for core).
    local scalp_paper="${SCALP_MODE_PAPER_ENABLED:-true}"
    local scalp_fee="${SCALP_FEE_MODEL_VERIFIED:-true}"
    echo "Starting Scalp Paper Runner (SCALP_PAPER_ENABLED=${scalp_paper})..."
    nohup env SCALP_PAPER_ENABLED="${scalp_paper}" SCALP_FEE_MODEL_VERIFIED="${scalp_fee}" \
        "$PYTHON" -m backend.services.binance_scalp.runner > /tmp/mystic_scalp.log 2>&1 &
    require_running "backend.services.binance_scalp.runner" "Scalp Paper Runner" "/tmp/mystic_scalp.log" 20 1 || return 1
}

case "$MODE" in
    core)
        echo "Mode: core (external supervisor — no duplicate embedded services)"
        stop_backend
        stop_live_md
        stop_signal
        stop_collector
        stop_portfolio
        stop_learning
        stop_ai_context
        stop_ai_position_tracker
        sleep 2

        if ! pgrep -x "redis-server" >/dev/null; then
            echo "Starting Redis..."
            sudo service redis-server start || exit 1
            sleep 2
        fi

        start_backend || exit 1
        sleep 3
        start_live_md || exit 1
        sleep 2
        start_signal || exit 1
        sleep 2
        start_portfolio truncate || exit 1
        sleep 2
        start_ai_context || exit 1
        sleep 1
        start_learning || exit 1
        sleep 1

        echo ""
        echo "=========================================="
        echo "MYSTIC CORE STACK STARTED"
        echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
        echo "Services: Backend + LiveMD (incl. feature_ohlcv) + Signal + Portfolio + Context + Learning"
        echo "Ensure .env has EXTERNAL_SUPERVISOR_MODE=true"
        echo "=========================================="
        ;;
    full)
        echo "Mode: full (DAY core + scalp paper)"
        systemctl --user stop mystic.target 2>/dev/null || true
        stop_scalp
        stop_backend
        stop_live_md
        stop_signal
        stop_collector
        stop_portfolio
        stop_learning
        stop_ai_context
        stop_ai_position_tracker
        sleep 2

        if ! pgrep -x "redis-server" >/dev/null; then
            echo "Starting Redis..."
            sudo service redis-server start || exit 1
            sleep 2
        fi

        start_backend || exit 1
        sleep 3
        start_live_md || exit 1
        sleep 2
        start_signal || exit 1
        sleep 2
        start_portfolio truncate || exit 1
        sleep 2
        start_ai_context || exit 1
        sleep 1
        start_learning || exit 1
        sleep 1
        start_scalp || exit 1

        echo ""
        echo "=========================================="
        echo "MYSTIC FULL STACK STARTED (DAY paper + scalp paper)"
        echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
        echo "DAY: Portfolio + Signal + Context + Learning"
        echo "SCALP: paper runner — log /tmp/mystic_scalp.log"
        echo "Ensure .env has EXTERNAL_SUPERVISOR_MODE=true"
        echo "=========================================="
        ;;
    all)
        echo "Mode: all"

        # ── Pre-deployment correctness gate ───────────────────────────────
        # Runs sell-path integration test (isolated SQLite, no live Redis).
        # Aborts startup if trade_state correctness is broken.
        if [ -f "scripts/run_predeploy_checks.sh" ]; then
            bash scripts/run_predeploy_checks.sh || {
                echo "ERROR: Pre-deployment checks failed — startup aborted."
                exit 1
            }
        fi
        # ─────────────────────────────────────────────────────────────────

        stop_backend
        stop_collector
        stop_ai
        stop_agents
        stop_portfolio
        stop_learning
        stop_ai_context
        stop_ai_position_tracker
        stop_ai_outcome_bridge
        sleep 2

        # Run cleanup only for full restarts.
        if [ -f "MANDATORY_CLEANUP.py" ]; then
            echo "Running mandatory cleanup..."
            "$PYTHON" MANDATORY_CLEANUP.py
        fi

        # Redis management only in full mode.
        if ! pgrep -x "redis-server" >/dev/null; then
            echo "Starting Redis..."
            if ! sudo service redis-server start; then
                echo "ERROR: Redis service start command failed"
                exit 1
            fi
            sleep 2
            if ! pgrep -x "redis-server" >/dev/null; then
                echo "ERROR: Redis did not start"
                exit 1
            fi
        fi

        start_backend || exit 1
        sleep 5
        start_collector || exit 1
        sleep 2
        start_ai || exit 1
        sleep 2
        start_agents || exit 1
        sleep 2
        start_portfolio append || exit 1
        sleep 2
        start_learning || exit 1
        sleep 1
        start_ai_context || exit 1
        sleep 1

        # Canonical restart boundary for analysis / audits (LOCAL truth marker)
        MARKER_JSON="${PWD}/.mystic_restart_marker.json"
        TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        GIT_HEAD="$("$PYTHON" -c "import subprocess; r=subprocess.run(['git','rev-parse','--short=12','HEAD'],cwd='${PWD}',capture_output=True,text=True); print((r.stdout or 'unknown').strip())" 2>/dev/null || echo unknown)"
        "$PYTHON" - "$MARKER_JSON" "$TS_UTC" "$GIT_HEAD" <<'PY'
import json, os, subprocess, sys
from pathlib import Path

def pid(pat: str) -> int | None:
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    if r.returncode != 0 or not (r.stdout or "").strip():
        return None
    return int((r.stdout or "").strip().splitlines()[0])

path, ts_utc, git_commit = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "ts_utc": ts_utc,
    "git_commit": git_commit,
    "hostname": os.uname().nodename,
    "stack_mode": "all",
    "pids": {
        "uvicorn": pid("uvicorn backend.main:app"),
        "portfolio_engine_integration": pid("start_portfolio_engine_integration.py"),
        "ai_ml_trading": pid("start_ai_ml_trading.py"),
        "ai_learning": pid("start_ai_learning.py"),
        "ai_market_context": pid("start_ai_market_context.py"),
    },
}
Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
        echo "Restart marker written: ${MARKER_JSON}"

        echo ""
        echo "=========================================="
        echo "MYSTIC STARTED SUCCESSFULLY"
        echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
        echo "Services: Backend + Collector + AI + Agents + Portfolio + Learning + AI Context"
        echo "=========================================="
        ;;
    ai_context)
        echo "Mode: ai_context"
        stop_ai_context
        sleep 1
        start_ai_context || exit 1
        ;;
    ai_position_tracker|ai_outcome_bridge)
        echo "ERROR: Mode '$MODE' is disabled — implementation not present (not part of DAY engine)."
        exit 1
        ;;
    portfolio)
        echo "Mode: portfolio"
        stop_portfolio
        sleep 1
        start_portfolio truncate || exit 1
        ;;
    ai)
        echo "Mode: ai"
        stop_ai
        sleep 1
        start_ai || exit 1
        ;;
    learning)
        echo "Mode: learning"
        stop_learning
        sleep 1
        start_learning || exit 1
        ;;
    collector)
        echo "Mode: collector"
        stop_collector
        sleep 1
        start_collector || exit 1
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
        stop_backend
        sleep 1
        start_backend || exit 1
        ;;
    scalp)
        echo "Mode: scalp (isolated — does not start/stop DAY portfolio stack)"
        stop_scalp
        sleep 1

        if ! pgrep -x "redis-server" >/dev/null; then
            echo "Starting Redis..."
            sudo service redis-server start || exit 1
            sleep 2
        fi

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
        echo "Runner uses SCALP_MODE_PAPER_ENABLED (default true) — .env SCALP_PAPER_ENABLED unchanged"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [full|core|all|scalp|backend|live_md|signal|portfolio|learning|collector|ai_context|ai]"
        exit 1
        ;;
esac
