#!/usr/bin/env bash
# Finish an in-progress Phase 3f soak from existing /tmp/scalp_phase3f/start.txt
set -euo pipefail

REPO=/home/mystic/mystic
VENV="$REPO/venv/bin/python3"
SOAK_DIR=/tmp/scalp_phase3f
DURATION_TARGET_SEC=$((2 * 3600))

if [ ! -f "$SOAK_DIR/start.txt" ]; then
  echo "missing $SOAK_DIR/start.txt" >&2
  exit 1
fi

SOAK_START=$(head -1 "$SOAK_DIR/start.txt")
SOAK_START_EPOCH=$(tail -1 "$SOAK_DIR/start.txt")
END_EPOCH=$((SOAK_START_EPOCH + DURATION_TARGET_SEC))

while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  ELAPSED=$(($(date +%s) - SOAK_START_EPOCH))
  if [ $((ELAPSED % 3600)) -lt 60 ]; then
    "$VENV" -c "
import json, subprocess
from datetime import datetime, timezone
mem={}
with open('/proc/meminfo') as f:
    for line in f:
        if line.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:')):
            k,v=line.split(':'); mem[k.strip()]=int(v.split()[0])
redis=subprocess.check_output(['redis-cli','info','memory'],text=True)
rm={}
for line in redis.splitlines():
    if line.startswith('used_memory_human:'):
        rm['used_memory_human']=line.split(':',1)[1].strip()
print(json.dumps({'ts':datetime.now(timezone.utc).isoformat(),'mem_kb':mem,'redis':rm}))
" >> "$SOAK_DIR/hourly_memory.jsonl" 2>/dev/null || true
  fi
  sleep 30
done

SOAK_END_EPOCH=$(date +%s)
DURATION=$((SOAK_END_EPOCH - SOAK_START_EPOCH))

systemctl --user stop mystic-scalp-paper.service 2>/dev/null || true
sleep 2
sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=false/' "$REPO/.env"

TICKS=$(journalctl --user -u mystic-scalp-paper.service --since "$SOAK_START" --no-pager 2>/dev/null | grep -c 'api/v3/depth' || true)
ERRORS=$(journalctl --user -u mystic-scalp-paper.service --since "$SOAK_START" --no-pager 2>/dev/null | grep -E 'Traceback|scalp paper tick error|database is locked' || true)
ERR_FILE="$SOAK_DIR/journal_errors.txt"
printf '%s' "$ERRORS" > "$ERR_FILE"

"$VENV" "$REPO/scripts/collect_scalp_soak_metrics.py" after "$SOAK_START" > "$SOAK_DIR/after.json"

"$VENV" - <<PY
import json
from pathlib import Path

hourly = []
p = Path("$SOAK_DIR/hourly_memory.jsonl")
if p.exists():
    hourly = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

payload = {
    "soak_start": "$SOAK_START",
    "duration_sec": $DURATION,
    "scan_ticks": int("${TICKS:-0}"),
    "baseline": json.loads(Path("$SOAK_DIR/baseline.json").read_text()),
    "after": json.loads(Path("$SOAK_DIR/after.json").read_text()),
    "hourly_memory": hourly,
    "journal_errors": Path("$ERR_FILE").read_text(),
}
Path("$SOAK_DIR/report_input.json").write_text(json.dumps(payload))
PY

"$VENV" "$REPO/scripts/run_scalp_phase3f_soak_report.py" "$SOAK_DIR/report_input.json" > "$SOAK_DIR/final_report.json"
echo "DONE duration_sec=$DURATION ticks=$TICKS"
