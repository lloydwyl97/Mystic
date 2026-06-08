#!/usr/bin/env bash
# Phase 3f — 8h paper soak (falls back to 4h if memory drops >15% available)
set -euo pipefail

REPO=/home/mystic/mystic
VENV="$REPO/venv/bin/python3"
SOAK_DIR=/tmp/scalp_phase3f
mkdir -p "$SOAK_DIR"

if [ -n "${SOAK_UNTIL_CDT:-}" ]; then
  DURATION_TARGET_SEC=$("$VENV" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
tz = ZoneInfo('America/Chicago')
now = datetime.now(tz)
h, m = map(int, '${SOAK_UNTIL_CDT}'.split(':'))
target = now.replace(hour=h, minute=m, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(int((target - now).total_seconds()))
")
elif [ -n "${SOAK_DURATION_SEC:-}" ]; then
  DURATION_TARGET_SEC="$SOAK_DURATION_SEC"
else
  DURATION_TARGET_SEC=$((2 * 3600))
fi
DURATION_MIN_SEC=$((4 * 3600))
if [ "$DURATION_TARGET_SEC" -lt "$DURATION_MIN_SEC" ]; then
  DURATION_MIN_SEC="$DURATION_TARGET_SEC"
fi
echo "soak_target_sec=$DURATION_TARGET_SEC" > "$SOAK_DIR/.target"

SOAK_START=$(date -u +"%Y-%m-%d %H:%M:%S")
SOAK_START_EPOCH=$(date +%s)
echo "$SOAK_START" > "$SOAK_DIR/start.txt"
echo "$SOAK_START_EPOCH" >> "$SOAK_DIR/start.txt"

# Baseline
"$VENV" "$REPO/scripts/collect_scalp_soak_metrics.py" baseline > "$SOAK_DIR/baseline.json"

# Enable paper only for soak
if grep -q '^SCALP_PAPER_ENABLED=' "$REPO/.env"; then
  sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=true/' "$REPO/.env"
else
  echo 'SCALP_PAPER_ENABLED=true' >> "$REPO/.env"
fi
grep -E '^SCALP_(PAPER_ENABLED|LIVE)=' "$REPO/.env" > "$SOAK_DIR/env_flags.txt"

systemctl --user daemon-reload
systemctl --user restart mystic-scalp-paper.service
sleep 5
if ! systemctl --user is-active --quiet mystic-scalp-paper.service; then
  journalctl --user -u mystic-scalp-paper.service -n 30 --no-pager > "$SOAK_DIR/startup_fail.log"
  sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=false/' "$REPO/.env"
  echo "STARTUP_FAIL" >&2
  exit 2
fi

MEM_BASE=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
: > "$SOAK_DIR/hourly_memory.jsonl"
END_EPOCH=$((SOAK_START_EPOCH + DURATION_TARGET_SEC))
EARLY_STOP=0

while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - SOAK_START_EPOCH))
  if [ "$ELAPSED" -ge "$DURATION_MIN_SEC" ]; then
    MEM_NOW=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$MEM_NOW" -lt $((MEM_BASE * 85 / 100)) ]; then
      EARLY_STOP=1
      break
    fi
  fi
  if [ "$ELAPSED" -ge "$DURATION_TARGET_SEC" ]; then
    break
  fi
  if [ $((ELAPSED % 3600)) -lt 60 ]; then
    "$VENV" -c "
import json, subprocess, time
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
" >> "$SOAK_DIR/hourly_memory.jsonl"
  fi
  sleep 60
done

SOAK_END_EPOCH=$(date +%s)
DURATION=$((SOAK_END_EPOCH - SOAK_START_EPOCH))

systemctl --user stop mystic-scalp-paper.service
sleep 2
sed -i 's/^SCALP_PAPER_ENABLED=.*/SCALP_PAPER_ENABLED=false/' "$REPO/.env"

JOURNAL_SINCE=$(head -1 "$SOAK_DIR/start.txt")
TICKS=$(journalctl --user -u mystic-scalp-paper.service --since "$JOURNAL_SINCE" --no-pager 2>/dev/null | grep -cE 'depth\?symbol=' || true)
ERRORS=$(journalctl --user -u mystic-scalp-paper.service --since "$JOURNAL_SINCE" --no-pager 2>/dev/null | grep -E 'Traceback|scalp paper tick error|database is locked' || true)

"$VENV" "$REPO/scripts/collect_scalp_soak_metrics.py" after "$SOAK_START" > "$SOAK_DIR/after.json"

ERR_FILE="$SOAK_DIR/journal_errors.txt"
printf '%s' "$ERRORS" > "$ERR_FILE"

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
echo "DONE duration_sec=$DURATION target_sec=$DURATION_TARGET_SEC early_stop=$EARLY_STOP ticks=$TICKS"
