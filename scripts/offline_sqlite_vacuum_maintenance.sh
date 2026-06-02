#!/bin/bash
# Offline SQLite maintenance: large-table retention, VACUUM, integrity_check.
# Refuses to run while Mystic processes are active unless --force-offline is passed.

set -euo pipefail

REPO_ROOT="/home/mystic/mystic"
DB_PATH="${DB_PATH:-$REPO_ROOT/mystic_trading.db}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/venv/bin/python3}"
AUTO_MANAGE=0
FORCE_OFFLINE=0
PRUNE_OLD_BACKUPS=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [--auto-manage-services] [--force-offline] [--keep-old-backups]

  --auto-manage-services  Stop Mystic before maintenance and start after (explicit opt-in).
  --force-offline         Skip active-process guard (still run offline; do not use if Mystic is up).
  --keep-old-backups      Do not delete prior mystic_trading.db.backup_* files after success.

Environment:
  DB_PATH       Path to mystic_trading.db (default: $DB_PATH)
  VENV_PYTHON   Python interpreter (default: venv)
EOF
}

for arg in "$@"; do
  case "$arg" in
    --auto-manage-services) AUTO_MANAGE=1 ;;
    --force-offline) FORCE_OFFLINE=1 ;;
    --keep-old-backups) PRUNE_OLD_BACKUPS=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

cd "$REPO_ROOT" || exit 1

PATTERNS=(
  "uvicorn backend.main:app"
  "start_portfolio_engine_integration.py"
  "start_live_market_data.py"
  "start_ai_signal_generator.py"
  "live_data_collector.py"
  "start_ai_ml_trading.py"
  "start_ai_learning.py"
  "start_agent_orchestrator.py"
  "start_ai_market_context.py"
  "start_ai_position_tracker.py"
  "start_ai_outcome_bridge.py"
)

active_pids=""
for pattern in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    active_pids="${active_pids}${pids}"$'\n'
  fi
done

if [ -n "$active_pids" ] && [ "$FORCE_OFFLINE" -eq 0 ]; then
  echo "ERROR: Mystic processes are still running. Stop Mystic first." >&2
  echo "Active:" >&2
  for pattern in "${PATTERNS[@]}"; do
    pgrep -af "$pattern" 2>/dev/null || true
  done
  echo "Use --auto-manage-services to stop/start automatically, or stop manually." >&2
  exit 1
fi

if [ "$AUTO_MANAGE" -eq 1 ]; then
  echo "Stopping Mystic (explicit --auto-manage-services)..."
  "$REPO_ROOT/stop_mystic.sh"
  sleep 3
fi

if [ ! -f "$DB_PATH" ]; then
  echo "ERROR: database not found: $DB_PATH" >&2
  exit 1
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP_PATH="${DB_PATH}.backup_before_offline_vacuum_${TS}"
echo "Backing up to $BACKUP_PATH"
cp -a "$DB_PATH" "$BACKUP_PATH"

SIZE_BEFORE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH")
echo "DB size before: $SIZE_BEFORE bytes ($(awk "BEGIN {printf \"%.2f GB\", $SIZE_BEFORE/1e9}") )"

echo "Running large-table retention (unlimited batches)..."
"$VENV_PYTHON" -m backend.services.sqlite_large_table_retention --db "$DB_PATH" --unlimited

echo "Running VACUUM + integrity_check..."
"$VENV_PYTHON" <<PY
import json
import sys
from pathlib import Path
from backend.services.sqlite_large_table_retention import run_offline_vacuum_and_integrity

db = Path("$DB_PATH")
result = run_offline_vacuum_and_integrity(db)
print(json.dumps(result, indent=2))
if result.get("integrity_check") != "ok":
    sys.exit("integrity_check failed: %r" % result.get("integrity_check"))
if result.get("vacuum") != "ok":
    sys.exit("VACUUM failed")
PY

SIZE_AFTER=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH")
FREED=$((SIZE_BEFORE - SIZE_AFTER))
echo "DB size after:  $SIZE_AFTER bytes ($(awk "BEGIN {printf \"%.2f GB\", $SIZE_AFTER/1e9}") )"
echo "Freed from VACUUM: $FREED bytes ($(awk "BEGIN {printf \"%.2f GB\", $FREED/1e9}") )"
echo "Backup kept: $BACKUP_PATH"

if [ "$PRUNE_OLD_BACKUPS" -eq 1 ]; then
  echo "Pruning stale DB backups (keeping this run's backup only)..."
  PRUNED=0
  PRUNED_BYTES=0
  while IFS= read -r -d '' old; do
    [ "$old" = "$BACKUP_PATH" ] && continue
    case "$old" in
      *.backup_*|*.test_sell_path_*|*.backup_before_*)
        sz=$(stat -c%s "$old" 2>/dev/null || stat -f%z "$old")
        rm -f "$old"
        PRUNED=$((PRUNED + 1))
        PRUNED_BYTES=$((PRUNED_BYTES + sz))
        echo "Removed stale backup: $old ($(awk "BEGIN {printf \"%.2f GB\", $sz/1e9}") )"
        ;;
    esac
  done < <(find "$(dirname "$DB_PATH")" -maxdepth 1 -type f \( -name "$(basename "$DB_PATH").backup_*" -o -name "$(basename "$DB_PATH").test_sell_path_*" \) -print0 2>/dev/null)
  echo "Pruned $PRUNED stale backup file(s), reclaimed $(awk "BEGIN {printf \"%.2f GB\", $PRUNED_BYTES/1e9}")"
fi

if [ "$AUTO_MANAGE" -eq 1 ]; then
  echo "Starting Mystic (explicit --auto-manage-services)..."
  "$REPO_ROOT/start_mystic.sh" core
fi

echo "Offline maintenance complete."
