#!/bin/bash
# MYSTIC Graceful Shutdown Script
# Stops all Mystic services without starting new ones

cd /home/mystic/mystic || exit 1

# PID-based stop: find PIDs by pattern, SIGTERM, wait, then SIGKILL only if needed
echo "Stopping Mystic services..."
PATTERNS=(
  "uvicorn backend.main:app"
  "start_live_market_data.py"
  "start_ai_signal_generator.py"
  "live_data_collector.py"
  "start_ai_ml_trading.py"
  "start_portfolio_engine_integration.py"
  "start_ai_learning.py"
  "start_agent_orchestrator.py"
  "start_ai_market_context.py"
  "start_ai_position_tracker.py"
  "start_ai_outcome_bridge.py"
  "backend.services.binance_scalp.runner"
)
for pattern in "${PATTERNS[@]}"; do
  while read -r pid; do
    [ -z "$pid" ] && continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done < <(pgrep -f "$pattern" 2>/dev/null)
done
echo "Waiting for processes to exit..."
sleep 5
FAILED=0
for pattern in "${PATTERNS[@]}"; do
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
