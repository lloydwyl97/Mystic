#!/bin/bash
# MYSTIC Graceful Shutdown Script — active core stack + legacy zombie patterns

cd /home/mystic/mystic || exit 1

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

stop_patterns() {
  local pattern
  for pattern in "$@"; do
    while read -r pid; do
      [ -z "$pid" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done < <(pgrep -f "$pattern" 2>/dev/null)
  done
}

stop_patterns "${ACTIVE_PATTERNS[@]}" "${LEGACY_PATTERNS[@]}"
echo "Waiting for processes to exit..."
sleep 5

FAILED=0
for pattern in "${ACTIVE_PATTERNS[@]}" "${LEGACY_PATTERNS[@]}"; do
  pids=$(pgrep -f "$pattern" 2>/dev/null)
  if [ -n "$pids" ]; then
    for pid in $pids; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 2
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      echo "WARNING: Process still running after TERM+KILL: $pattern"
      pgrep -af "$pattern" || true
      FAILED=1
    fi
  fi
done

if [ "$FAILED" -eq 1 ]; then
  echo "Some processes could not be killed. Check output above."
  exit 1
fi
echo "Mystic services stopped."
