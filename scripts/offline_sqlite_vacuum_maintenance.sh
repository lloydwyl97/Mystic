#!/bin/bash
# Offline SQLite maintenance: large-table retention, VACUUM, integrity_check.
# Refuses to run while Mystic processes are active unless --force-offline is passed.

set -euo pipefail

REPO_ROOT="/home/mystic/mystic"
DB_PATH="${DB_PATH:-$REPO_ROOT/mystic_trading.db}"
SCALP_DB_PATH="${SCALP_DB_PATH:-$REPO_ROOT/mystic_scalp.db}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/venv/bin/python3}"
AUTO_MANAGE=0
FORCE_OFFLINE=0
PRUNE_OLD_BACKUPS=1
MAINT_LOCK="${MYSTIC_MAINTENANCE_LOCK:-/tmp/mystic_maintenance.lock}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--auto-manage-services] [--force-offline] [--keep-old-backups]

  --auto-manage-services  Stop Mystic before maintenance and start after (explicit opt-in).
  --force-offline         Skip active-process guard (still run offline; do not use if Mystic is up).
  --keep-old-backups      Do not delete prior mystic_trading.db.backup_* files after success.

Environment:
  DB_PATH       Path to mystic_trading.db (default: $DB_PATH)
  SCALP_DB_PATH Path to mystic_scalp.db (default: $SCALP_DB_PATH)
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
  "backend.services.binance_scalp.runner"
)

active_pids=""
for pattern in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    active_pids="${active_pids}${pids}"$'\n'
  fi
done

release_maint_lock() {
  rm -f "$MAINT_LOCK"
}
: > "$MAINT_LOCK"
trap release_maint_lock EXIT

if [ "$AUTO_MANAGE" -eq 1 ]; then
  echo "Stopping Mystic (explicit --auto-manage-services)..."
  "$REPO_ROOT/stop_mystic.sh"
  sleep 3
elif [ -n "$active_pids" ] && [ "$FORCE_OFFLINE" -eq 0 ]; then
  echo "ERROR: Mystic processes are still running. Stop Mystic first." >&2
  echo "Active:" >&2
  for pattern in "${PATTERNS[@]}"; do
    pgrep -af "$pattern" 2>/dev/null || true
  done
  echo "Use --auto-manage-services to stop/start automatically, or stop manually." >&2
  exit 1
fi

maintain_one_db() {
  local db="$1"
  if [ ! -f "$db" ]; then
    echo "SKIP: database not found: $db"
    return 0
  fi

  local ts backup size_before size_after freed
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  backup="${db}.backup_before_offline_vacuum_${ts}"
  echo "Backing up $db to $backup"
  cp -a "$db" "$backup"

  size_before=$(stat -c%s "$db" 2>/dev/null || stat -f%z "$db")
  echo "DB size before: $size_before bytes ($(awk "BEGIN {printf \"%.2f GB\", $size_before/1e9}") )"

  echo "Running large-table retention (unlimited batches) on $db ..."
  "$VENV_PYTHON" -m backend.services.sqlite_large_table_retention --db "$db" --unlimited

  echo "Running VACUUM + integrity_check on $db ..."
  DB_FOR_PY="$db" "$VENV_PYTHON" <<'PY'
import json
import os
import sys
from pathlib import Path
from backend.services.sqlite_large_table_retention import run_offline_vacuum_and_integrity

db = Path(os.environ["DB_FOR_PY"])
result = run_offline_vacuum_and_integrity(db)
print(json.dumps(result, indent=2))
if result.get("integrity_check") != "ok":
    sys.exit("integrity_check failed: %r" % result.get("integrity_check"))
if result.get("vacuum") != "ok":
    sys.exit("VACUUM failed")
PY

  size_after=$(stat -c%s "$db" 2>/dev/null || stat -f%z "$db")
  freed=$((size_before - size_after))
  echo "DB size after:  $size_after bytes ($(awk "BEGIN {printf \"%.2f GB\", $size_after/1e9}") )"
  echo "Freed from VACUUM: $freed bytes ($(awk "BEGIN {printf \"%.2f GB\", $freed/1e9}") )"
  if [ "$PRUNE_OLD_BACKUPS" -eq 1 ]; then
    echo "Pruning DB backups for $(basename "$db") after successful VACUUM..."
    local pruned=0
    local pruned_bytes=0
    local old sz
    while IFS= read -r -d '' old; do
      [ "$old" = "$backup" ] && continue
      case "$old" in
        *.backup_*|*.test_sell_path_*|*.backup_before_*)
          sz=$(stat -c%s "$old" 2>/dev/null || stat -f%z "$old")
          rm -f "$old"
          pruned=$((pruned + 1))
          pruned_bytes=$((pruned_bytes + sz))
          echo "Removed backup: $old ($(awk "BEGIN {printf \"%.2f GB\", $sz/1e9}") )"
          ;;
      esac
    done < <(find "$(dirname "$db")" -maxdepth 1 -type f \( -name "$(basename "$db").backup_*" -o -name "$(basename "$db").test_sell_path_*" \) -print0 2>/dev/null)
    echo "Pruned $pruned backup file(s), reclaimed $(awk "BEGIN {printf \"%.2f GB\", $pruned_bytes/1e9}")"
  else
    echo "Backup kept: $backup"
  fi
}

maintain_one_db "$DB_PATH"
maintain_one_db "$SCALP_DB_PATH"

if [ "$AUTO_MANAGE" -eq 1 ]; then
  echo "Starting Mystic (explicit --auto-manage-services)..."
  "$REPO_ROOT/start_mystic.sh" core
fi

echo "Offline maintenance complete."
