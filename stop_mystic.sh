#!/bin/bash
# MYSTIC Graceful Shutdown Script — active core stack + legacy zombie patterns

cd /home/mystic/mystic || exit 1

LIFECYCLE_LOCK="${PWD}/logs/mystic_lifecycle.lock"
mkdir -p "${PWD}/logs"
exec 9>"$LIFECYCLE_LOCK"
if ! flock -w 5 9; then
  echo "WARN: lifecycle lock busy — stopping processes anyway"
fi

echo "Stopping Mystic services..."
ACTIVE_PATTERNS=(
  "uvicorn backend.main:app"
  "start_live_market_data.py"
  "start_ai_signal_generator.py"
  "start_portfolio_engine_integration.py"
  "start_ai_learning.py"
  "start_ai_market_context.py"
  "backend.services.binance_scalp.runner"
)
LEGACY_PATTERNS=(
  "live_data_collector.py"
  "start_ai_ml_trading.py"
  "start_agent_orchestrator.py"
  "start_ai_position_tracker.py"
  "start_ai_outcome_bridge.py"
)

# Only real app PIDs — never ssh/bash wrappers, and never this stop script itself.
list_app_pids() {
  local pattern=$1
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
      *stop_mystic.sh*) continue ;;
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

stop_patterns() {
  local pattern pid
  for pattern in "$@"; do
    while read -r pid; do
      [ -z "$pid" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done < <(list_app_pids "$pattern")
  done
}

stop_patterns "${ACTIVE_PATTERNS[@]}" "${LEGACY_PATTERNS[@]}"
echo "Waiting for processes to exit..."
sleep 5

FAILED=0
for pattern in "${ACTIVE_PATTERNS[@]}" "${LEGACY_PATTERNS[@]}"; do
  pids=$(list_app_pids "$pattern")
  if [ -n "$pids" ]; then
    for pid in $pids; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 2
    leftover=$(list_app_pids "$pattern")
    if [ -n "$leftover" ]; then
      echo "WARNING: Process still running after TERM+KILL: $pattern"
      for pid in $leftover; do
        ps -p "$pid" -o pid=,args= 2>/dev/null || true
      done
      FAILED=1
    fi
  fi
done

if [ "$FAILED" -eq 1 ]; then
  echo "Some processes could not be killed. Check output above."
  exit 1
fi
echo "Mystic services stopped."
