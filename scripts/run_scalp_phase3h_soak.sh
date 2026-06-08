#!/usr/bin/env bash
# Phase 3h — 60 min exit-fix validation soak
set -euo pipefail

REPO=/home/mystic/mystic
VENV="$REPO/venv/bin/python3"
SOAK_DIR=/tmp/scalp_phase3h
DURATION_TARGET_SEC="${SOAK_DURATION_SEC:-3600}"
mkdir -p "$SOAK_DIR"

SOAK_START=$(date -u +"%Y-%m-%d %H:%M:%S")
SOAK_START_EPOCH=$(date +%s)
echo "$SOAK_START" > "$SOAK_DIR/start.txt"
echo "$SOAK_START_EPOCH" >> "$SOAK_DIR/start.txt"
echo "soak_target_sec=$DURATION_TARGET_SEC" > "$SOAK_DIR/.target"

"$VENV" "$REPO/scripts/collect_scalp_soak_metrics.py" baseline > "$SOAK_DIR/baseline.json"

if grep -q '^SCALP_PAPER_ENABLED=' "$REPO/.env"; then
  sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=true/' "$REPO/.env"
else
  echo 'SCALP_PAPER_ENABLED=true' >> "$REPO/.env"
fi

systemctl --user daemon-reload
systemctl --user restart mystic-scalp-paper.service
sleep 5
if ! systemctl --user is-active --quiet mystic-scalp-paper.service; then
  sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=false/' "$REPO/.env"
  exit 2
fi

END_EPOCH=$((SOAK_START_EPOCH + DURATION_TARGET_SEC))
while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  sleep 30
done

SOAK_END_EPOCH=$(date +%s)
DURATION=$((SOAK_END_EPOCH - SOAK_START_EPOCH))

systemctl --user stop mystic-scalp-paper.service
sleep 2
sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=false/' "$REPO/.env"

JOURNAL_SINCE=$(head -1 "$SOAK_DIR/start.txt")
ERRORS=$(journalctl --user -u mystic-scalp-paper.service --since "$JOURNAL_SINCE" --no-pager 2>/dev/null | grep -E 'Traceback|scalp paper tick error|database is locked' || true)
printf '%s' "$ERRORS" > "$SOAK_DIR/journal_errors.txt"

"$VENV" "$REPO/scripts/run_scalp_phase3h_report.py" > "$SOAK_DIR/final_report.json"
echo "DONE duration_sec=$DURATION"
