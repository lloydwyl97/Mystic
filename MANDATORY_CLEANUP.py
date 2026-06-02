#!/usr/bin/env python3
"""
MANDATORY DATA CLEANUP - RUNS BEFORE EVERY STARTUP
Cleans all malformed positions from ALL databases and Redis
"""

import os
import sqlite3
import sys

import redis

print("\n" + "=" * 60)
print("  MANDATORY DATA CLEANUP - REMOVING ALL MALFORMED DATA")
print("=" * 60 + "\n")

# ============================================================================
# STEP 1: CLEAN SQLITE DATABASES
# ============================================================================
print("[1/4] Cleaning SQLite databases...")

databases = [
    "mystic_trading.db",
    "paper_trading.db",
    "trading.db",
]

total_deleted = 0

for db_name in databases:
    if not os.path.exists(db_name):
        print(f"  ⊘ {db_name} not found - skipping")
        continue

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Check if table has a 'symbol' column
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]

            if "symbol" in columns:
                # Delete malformed USDT/USDT positions
                query = f"DELETE FROM {table} WHERE symbol LIKE '%USDT/USDT%' OR symbol LIKE '%USDTUSDT%'"
                cursor.execute(query)
                deleted = cursor.rowcount

                if deleted > 0:
                    print(f"  ✓ {db_name}.{table}: Deleted {deleted} malformed rows")
                    total_deleted += deleted

        conn.commit()
        conn.close()
        print(f"  ✓ {db_name} cleaned")

    except Exception as e:
        print(f"  ✗ {db_name} error: {e}")

print(f"\n  → Total SQLite rows deleted: {total_deleted}\n")

# ============================================================================
# STEP 2: CLEAN REDIS
# ============================================================================
print("[2/4] Cleaning Redis...")

try:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.ping()
    print("  ✓ Redis connected")

    # Delete all malformed position keys
    deleted_count = 0

    # Check paper_trading positions
    paper_keys = list(r.scan_iter(match="paper_trading:position:*", count=500))
    for key in paper_keys:
        val = r.get(key)
        if val and ("USDT/USDT" in val or "USDTUSDT" in val):
            r.delete(key)
            deleted_count += 1
            print(f"  ✓ Deleted Redis key: {key}")

    # Check ai_decision keys
    ai_keys = list(r.scan_iter(match="ai_decision:*", count=500))
    for key in ai_keys:
        if "USDT/USDT" in key or "USDTUSDT" in key:
            r.delete(key)
            deleted_count += 1
            print(f"  ✓ Deleted Redis key: {key}")

    # Check ALL keys for malformed symbols
    all_keys = list(r.scan_iter(match="*", count=1000))
    for key in all_keys:
        if "USDT/USDT" in key or "USDTUSDT" in key:
            r.delete(key)
            deleted_count += 1
            print(f"  ✓ Deleted Redis key: {key}")

    # Check active_trade_ids hash
    active_trades = r.hgetall("paper_trading:active_trade_ids")
    for symbol, _trade_id in active_trades.items():
        if "USDT/USDT" in symbol or "USDTUSDT" in symbol:
            r.hdel("paper_trading:active_trade_ids", symbol)
            deleted_count += 1
            print(f"  ✓ Removed from active_trade_ids: {symbol}")

    print(f"\n  → Total Redis keys deleted: {deleted_count}\n")

except redis.ConnectionError:
    print("  ⊘ Redis not running - skipping")
except Exception as e:
    print(f"  ✗ Redis error: {e}")

# ============================================================================
# STEP 3: VERIFY CLEANUP
# ============================================================================
print("[3/4] Verifying cleanup...")

verify_ok = True

# Verify SQLite
for db_name in databases:
    if not os.path.exists(db_name):
        continue

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]

            if "symbol" in columns:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol LIKE '%USDT/USDT%' OR symbol LIKE '%USDTUSDT%'")
                count = cursor.fetchone()[0]

                if count > 0:
                    print(f"  ✗ {db_name}.{table} still has {count} malformed rows!")
                    verify_ok = False

        conn.close()

    except Exception as e:
        print(f"  ✗ Verification error for {db_name}: {e}")
        verify_ok = False

# Verify Redis
try:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.ping()

    malformed_keys = []
    all_keys = list(r.scan_iter(match="*", count=1000))
    for key in all_keys:
        if "USDT/USDT" in key or "USDTUSDT" in key:
            malformed_keys.append(key)

    if malformed_keys:
        print(f"  ✗ Redis still has {len(malformed_keys)} malformed keys!")
        verify_ok = False

except Exception:
    pass

if verify_ok:
    print("  ✓ Verification passed - no malformed data found\n")
else:
    print("  ✗ Verification FAILED - malformed data still present!\n")

# ============================================================================
# STEP 4: SUMMARY
# ============================================================================
print("[4/4] Cleanup Summary")
print(f"  SQLite rows deleted: {total_deleted}")
print(f"  Verification: {'✓ PASSED' if verify_ok else '✗ FAILED'}")
print()

if verify_ok:
    print("✅ CLEANUP COMPLETE - Safe to start services")
    print("=" * 60 + "\n")
    sys.exit(0)
else:
    print("❌ CLEANUP FAILED - DO NOT START SERVICES")
    print("=" * 60 + "\n")
    sys.exit(1)
