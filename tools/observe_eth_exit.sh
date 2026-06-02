#!/bin/bash
set -euo pipefail

DB_PATH="${DB_PATH:-${MYSTIC_DB_PATH:-${DATABASE_PATH:-$(pwd)/mystic_trading.db}}}"
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found at $DB_PATH"
    exit 1
fi

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
OBS_DIR="/home/mystic/mystic/observations/eth_exit_${TIMESTAMP}"
mkdir -p "${OBS_DIR}"

echo "================================================================================"
echo " MYSTIC ETH EXIT OBSERVATION (READ-ONLY)"
echo "================================================================================"
echo " Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Database: ${DB_PATH}"
echo " Output: ${OBS_DIR}"
echo "================================================================================"
echo ""

cd "${OBS_DIR}"

# STEP 1: AUTO-DETECT SYMBOL FORMATS (REDIS + LOGS)
echo "[1/7] Detecting symbol formats..."

SYMBOL_FORMATS=()

# Check Redis keys
if redis-cli --scan --pattern "*ETHUSDT*" 2>/dev/null | head -n 1 | grep -q "ETHUSDT"; then
    SYMBOL_FORMATS+=("ETHUSDT")
fi
if redis-cli --scan --pattern "*ETH/USDT*" 2>/dev/null | head -n 1 | grep -q "ETH/USDT"; then
    SYMBOL_FORMATS+=("ETH/USDT")
fi

# Check logs
if tail -n 200 /tmp/mystic_portfolio.log 2>/dev/null | grep -q "ETHUSDT"; then
    [[ ! " ${SYMBOL_FORMATS[@]} " =~ " ETHUSDT " ]] && SYMBOL_FORMATS+=("ETHUSDT")
fi
if tail -n 200 /tmp/mystic_portfolio.log 2>/dev/null | grep -q "ETH/USDT"; then
    [[ ! " ${SYMBOL_FORMATS[@]} " =~ " ETH/USDT " ]] && SYMBOL_FORMATS+=("ETH/USDT")
fi

if [ ${#SYMBOL_FORMATS[@]} -eq 0 ]; then
    echo "  WARNING: No ETH symbol format detected. Using both."
    SYMBOL_FORMATS=("ETHUSDT" "ETH/USDT")
fi

echo "  Detected formats: ${SYMBOL_FORMATS[*]}"
echo "${SYMBOL_FORMATS[*]}" > symbol_formats.txt

# STEP 2: EXTRACT LOGS (NO HARDCODED TIMESTAMPS)
echo "[2/7] Extracting logs..."

grep -i "ETH" /tmp/mystic_portfolio.log 2>/dev/null | tail -n 5000 > portfolio_eth_all.log || touch portfolio_eth_all.log
grep -iE "ETH.*exit|exit.*ETH|SELL.*ETH|ETH.*SELL|TP1|TP2|TRAIL|STOP|cooldown|pause|realized|pnl" /tmp/mystic_portfolio.log 2>/dev/null | tail -n 2000 > portfolio_eth_focus.log || touch portfolio_eth_focus.log
grep -iE "SELL.*ETH|ETH.*SELL|exit.*ETH|POSITION_CLOSED.*ETH|TP1.*ETH|TP2.*ETH|STOP.*ETH|TRAIL.*ETH" portfolio_eth_all.log > portfolio_eth_exit_keywords.log 2>/dev/null || touch portfolio_eth_exit_keywords.log
grep -i "ETH" /tmp/mystic_ai.log 2>/dev/null | tail -n 2000 > ai_eth_all.log || touch ai_eth_all.log
grep -i "ETH" /tmp/mystic_collector.log 2>/dev/null | tail -n 2000 > collector_eth.log || touch collector_eth.log
tail -n 500 /tmp/mystic_portfolio.log 2>/dev/null > portfolio_tail.log || touch portfolio_tail.log
tail -n 500 /tmp/mystic_ai.log 2>/dev/null > ai_tail.log || touch ai_tail.log
tail -n 500 /tmp/mystic_backend.log 2>/dev/null > backend_tail.log || touch backend_tail.log

echo "  Logs extracted: $(wc -l *.log 2>/dev/null | tail -n 1 | awk '{print $1}') lines"

# STEP 3: PROBE SQLITE SCHEMA
echo "[3/7] Probing SQLite schema..."

python3 << PYEOF > schema_detection.txt
import sqlite3
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
tables = ['portfolio_engine_positions', 'paper_trades', 'pipeline_decisions', 'coin_performance', 'portfolio_engine_ledger', 'portfolio_engine_audit']
for table in tables:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
    result = cursor.fetchone()
    if result:
        print(f"\n{table}: EXISTS")
        cursor.execute(f"PRAGMA table_info({table})")
        for col in cursor.fetchall():
            print(f"  {col[1]}: {col[2]}")
    else:
        print(f"\n{table}: NOT FOUND")
conn.close()
PYEOF

echo "  Schema detection complete"

# STEP 4: QUERY SQLITE (TYPE-SAFE)
echo "[4/7] Querying SQLite..."

python3 << PYEOF > sqlite_position.txt
import sqlite3
import sys
from datetime import datetime
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_positions'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("SELECT * FROM portfolio_engine_positions WHERE symbol LIKE '%ETH%' ORDER BY entry_time DESC LIMIT 5")
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print("\nPosition:")
            for col, val in zip(cols, row):
                if val is not None:
                    if 'time' in col.lower() and isinstance(val, (int, float)):
                        try:
                            val = datetime.fromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            print(f"Warning: timestamp conversion failed: {e}", file=sys.stderr)
                    print(f"  {col}: {val}")
    else:
        print("No ETH positions found")
else:
    print("Table not found")
conn.close()
PYEOF

python3 << PYEOF > sqlite_trades.txt
import sqlite3
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("PRAGMA table_info(paper_trades)")
    cols_info = cursor.fetchall()
    has_mode = any(c[1] == 'mode' for c in cols_info)
    mode_filter = "AND mode = 'live'" if has_mode else ""
    cursor.execute(f"SELECT * FROM paper_trades WHERE symbol LIKE '%ETH%' {mode_filter} AND timestamp > datetime('now', '-48 hours') ORDER BY timestamp ASC")
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for i, row in enumerate(rows, 1):
            print(f"\nTrade #{i}:")
            for col, val in zip(cols, row):
                if val is not None:
                    print(f"  {col}: {val}")
    else:
        print("No ETH trades in last 48h")
else:
    print("Table not found")
conn.close()
PYEOF

python3 << PYEOF > sqlite_pipeline.txt
import sqlite3
import sys
from datetime import datetime
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_decisions'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("SELECT * FROM pipeline_decisions WHERE symbol LIKE '%ETH%' AND timestamp > strftime('%s', 'now', '-14400') ORDER BY timestamp DESC LIMIT 100")
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for i, row in enumerate(rows, 1):
            print(f"\nDecision #{i}:")
            for col, val in zip(cols, row):
                if val is not None:
                    if col == 'timestamp' and isinstance(val, (int, float)):
                        try:
                            val = f"{val} ({datetime.fromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S')})"
                        except Exception as e:
                            print(f"Warning: timestamp conversion failed: {e}", file=sys.stderr)
                    print(f"  {col}: {val}")
    else:
        print("No ETH pipeline decisions in last 4h")
else:
    print("Table not found")
conn.close()
PYEOF

python3 << PYEOF > sqlite_coin_performance.txt
import sqlite3
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coin_performance'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("SELECT * FROM coin_performance WHERE symbol LIKE '%ETH%'")
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print("\nCoin Performance:")
            for col, val in zip(cols, row):
                if val is not None:
                    print(f"  {col}: {val}")
    else:
        print("No ETH coin performance data")
else:
    print("Table not found")
conn.close()
PYEOF

python3 << PYEOF > sqlite_ledger.txt
import sqlite3
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_ledger'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("PRAGMA table_info(portfolio_engine_ledger)")
    cols_info = cursor.fetchall()
    has_symbol = any(c[1] == 'symbol' for c in cols_info)
    if has_symbol:
        cursor.execute("SELECT * FROM portfolio_engine_ledger WHERE symbol LIKE '%ETH%' ORDER BY last_updated DESC LIMIT 5")
    else:
        cursor.execute("SELECT * FROM portfolio_engine_ledger ORDER BY last_updated DESC LIMIT 5")
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for i, row in enumerate(rows, 1):
            print(f"\nLedger #{i}:")
            for col, val in zip(cols, row):
                if val is not None:
                    print(f"  {col}: {val}")
    else:
        print("No ledger entries found")
else:
    print("Table not found")
conn.close()
PYEOF

python3 << PYEOF > sqlite_audit.txt
import sqlite3
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_audit'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("SELECT * FROM portfolio_engine_audit WHERE symbol LIKE '%ETH%' AND ts > datetime('now', '-48 hours') ORDER BY ts ASC")
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    if rows:
        for i, row in enumerate(rows, 1):
            print(f"\nAudit #{i}:")
            for col, val in zip(cols, row):
                if val is not None:
                    print(f"  {col}: {val}")
    else:
        print("No ETH audit entries in last 48h")
else:
    print("Table not found")
conn.close()
PYEOF

echo "  SQLite queries complete"

# STEP 5: PROBE REDIS (TYPE-AWARE)
echo "[5/7] Probing Redis..."

{
    echo "REDIS ETH STATE (TYPE-AWARE):"
    echo ""
    
    # Scan for ETH keys (limit to 200)
    redis-cli --scan --pattern "*ETH*" 2>/dev/null | head -n 200 | while read key; do
        if [ -z "$key" ]; then continue; fi
        
        echo "KEY: $key"
        key_type=$(redis-cli TYPE "$key" 2>/dev/null || echo "error")
        echo "  Type: $key_type"
        
        case "$key_type" in
            string)
                val=$(redis-cli --raw GET "$key" 2>/dev/null | head -c 1000)
                echo "  Value: $val"
                ;;
            hash)
                echo "  Hash fields:"
                redis-cli HGETALL "$key" 2>/dev/null | head -n 80
                ;;
            list)
                echo "  List (last 50):"
                redis-cli LRANGE "$key" -50 -1 2>/dev/null
                ;;
            set)
                echo "  Set members:"
                redis-cli SMEMBERS "$key" 2>/dev/null | head -n 30
                ;;
            zset)
                echo "  ZSet (top 50):"
                redis-cli ZRANGE "$key" 0 49 WITHSCORES 2>/dev/null
                ;;
            none)
                echo "  Status: KEY NOT FOUND"
                ;;
            *)
                echo "  Status: Unknown type or error"
                ;;
        esac
        
        # Get TTL if key exists
        if [ "$key_type" != "none" ]; then
            ttl=$(redis-cli TTL "$key" 2>/dev/null || echo "-999")
            if [ "$ttl" -ge 0 ]; then
                echo "  TTL: ${ttl}s"
            elif [ "$ttl" = "-1" ]; then
                echo "  TTL: never expires"
            fi
        fi
        
        echo "---"
    done
    
    # Check specific expected keys
    echo ""
    echo "SPECIFIC KEY CHECK:"
    for sym in "ETHUSDT" "ETH/USDT"; do
        bus="${sym//\//}"
        for key in "ai_signal:scalp:${bus}" "ai_signal:day:${bus}" "ai_signal_snapshot:${sym}" "price:${sym}" "klines:${sym}:1m"; do
            exists=$(redis-cli EXISTS "$key" 2>/dev/null || echo "0")
            if [ "$exists" = "1" ]; then
                echo "  $key exists"
            fi
        done
    done
    
} > redis_eth_state.txt 2>&1

echo "  Redis snapshot complete"

# STEP 6: DETECT EXIT EVENT
echo "[6/7] Detecting exit event..."

python3 << PYEOF > exit_detection.txt
import sqlite3
import sys
from datetime import datetime
conn = sqlite3.connect("${DB_PATH}")
cursor = conn.cursor()

# Check if paper_trades exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
table_exists = cursor.fetchone()
if table_exists:
    cursor.execute("SELECT timestamp, side, quantity, price, pnl, pnl_pct, trade_id FROM paper_trades WHERE symbol LIKE '%ETH%' AND side = 'SELL' AND timestamp > datetime('now', '-48 hours') ORDER BY timestamp DESC LIMIT 10")
    sells = cursor.fetchall()
    if sells:
        print(f"EXIT DETECTED: {len(sells)} SELL(s) in last 48h")
        for i, row in enumerate(sells, 1):
            pnl = f"\${row[4]:.2f}" if row[4] else "N/A"
            pnl_pct = f"{row[5]:.2f}%" if row[5] else "N/A"
            print(f"\n  Sell #{i}:")
            print(f"    Time: {row[0]}")
            print(f"    Qty: {row[2]:.4f}")
            print(f"    Price: \${row[3]:.2f}")
            print(f"    PnL: {pnl} ({pnl_pct})")
            print(f"    Trade ID: {row[6]}")
        print(f"\n  Most recent exit: {sells[0][0]}")
    else:
        print("NO EXIT DETECTED in last 48h")
        print("Position may still be open")
else:
    print("ERROR: paper_trades table not found")

# Check current position
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_engine_positions'")
pos_table_exists = cursor.fetchone()
if pos_table_exists:
    cursor.execute("SELECT symbol, quantity, entry_price, entry_time FROM portfolio_engine_positions WHERE symbol LIKE '%ETH%' ORDER BY entry_time DESC LIMIT 1")
    pos = cursor.fetchone()
    if pos:
        try:
            entry_dt = datetime.fromtimestamp(pos[3]).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Warning: invalid entry_time: {e}", file=sys.stderr)
            entry_dt = str(pos[3])
        print(f"\nCurrent position: {pos[0]}, qty={pos[1]:.4f}, entry_price=\${pos[2]:.2f}, entry_time={entry_dt}")
    else:
        print("\nNo current ETH position found")

conn.close()
PYEOF

cat exit_detection.txt

# STEP 7: GENERATE SUMMARY
echo ""
echo "[7/7] Generating summary..."

OBS_TIME=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

python3 << PYEOF > summary.txt
import os
import sys

print("="*80)
print("ETH TRADE LIFECYCLE OBSERVATION SUMMARY")
print("="*80)
print(f"Observation Time: ${OBS_TIME}")
print(f"Database: ${DB_PATH}")
print(f"Output Dir: ${OBS_DIR}")
print("="*80)

print("\nSYMBOL FORMATS DETECTED:")
with open('symbol_formats.txt', 'r') as f:
    print(f"  {f.read().strip()}")

print("\nEXIT DETECTION:")
with open('exit_detection.txt', 'r') as f:
    content = f.read()
    if 'EXIT DETECTED' in content:
        print("  EXIT FOUND")
    else:
        print("  NO EXIT (position still open)")
    for line in content.split('\n'):
        if line.strip():
            print(f"  {line}")

print("\nKEY METRICS FROM LOGS:")
tp1_count = 0
tp2_count = 0
stop_count = 0
trail_count = 0
ai_exit_count = 0
time_exit_count = 0

try:
    with open('portfolio_eth_focus.log', 'r') as f:
        log_content = f.read()
        tp1_count = log_content.upper().count('TP1')
        tp2_count = log_content.upper().count('TP2')
        stop_count = log_content.upper().count('STOP')
        trail_count = log_content.upper().count('TRAIL')
        ai_exit_count = log_content.upper().count('AI_EXIT')
        time_exit_count = log_content.upper().count('TIME_EXIT')
except Exception as e:
    print(f"Warning: could not read portfolio_eth_focus.log: {e}", file=sys.stderr)

print(f"  TP1 mentions: {tp1_count}")
print(f"  TP2 mentions: {tp2_count}")
print(f"  STOP mentions: {stop_count}")
print(f"  TRAIL mentions: {trail_count}")
print(f"  AI_EXIT mentions: {ai_exit_count}")
print(f"  TIME_EXIT mentions: {time_exit_count}")

print("\nFILES CAPTURED:")
for fname in sorted(os.listdir('.')):
    if os.path.isfile(fname):
        size = os.path.getsize(fname)
        lines = 0
        if fname.endswith(('.log', '.txt')):
            try:
                with open(fname, 'r') as f:
                    lines = len(f.readlines())
            except Exception as e:
                print(f"Warning: could not read {fname}: {e}", file=sys.stderr)
        print(f"  {fname:<40} {size:>8} bytes  {lines:>5} lines")

print("\n" + "="*80)
print("ANALYSIS CHECKLIST:")
print("="*80)
print("[ ] Review sqlite_position.txt for entry state")
print("[ ] Review sqlite_trades.txt for all BUY/SELL orders")
print("[ ] Review portfolio_eth_exit_keywords.log for trigger")
print("[ ] Review sqlite_coin_performance.txt for metrics")
print("[ ] Review redis_eth_state.txt for live signals")
print("[ ] Search for exit trigger: TP1, TP2, STOP_LOSS, TRAILING, TIME_EXIT, AI_EXIT")
print("[ ] Answer: WHY sold, HOW sold, WHAT PnL, WHAT after")
print("="*80)
PYEOF

echo ""
echo "================================================================================"
echo " OBSERVATION COMPLETE"
echo "================================================================================"
echo " Output: ${OBS_DIR}"
echo " Files: $(ls -1 | wc -l)"
echo ""
ls -lh
echo ""
echo "Review summary.txt for analysis checklist"
echo "================================================================================"
