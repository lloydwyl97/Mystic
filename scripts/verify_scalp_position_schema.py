#!/usr/bin/env python3
"""Verify scalp position open-unique migration and dry-run constraints."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import os

    from dotenv import load_dotenv

    from backend.services.binance_scalp.schema import (
        init_scalp_schema,
        verify_open_position_constraints,
    )

    load_dotenv(ROOT / ".env")
    db = os.getenv("DATABASE_PATH", str(ROOT / "mystic_trading.db"))

    applied = init_scalp_schema(db)
    info = verify_open_position_constraints(db)

    results: dict[str, object] = {
        "migration_applied": applied,
        "constraints": info,
        "tests": {},
    }

    with sqlite3.connect(db) as conn:
        # BTC re-entry after closed row should succeed.
        try:
            conn.execute(
                """
                INSERT INTO scalp_paper_positions
                (symbol, exchange, strategy_id, quantity, entry_price, entry_time,
                 entry_time_epoch, trade_id, status)
                VALUES ('BTCUSDT', 'binance_us', 'scalp', 0.0001, 1.0,
                        datetime('now'), 0, 'dryrun_btc_open_test', 'OPEN')
                """
            )
            conn.execute("DELETE FROM scalp_paper_positions WHERE trade_id='dryrun_btc_open_test'")
            results["tests"]["btc_reentry_after_closed"] = "PASS"
        except sqlite3.IntegrityError as exc:
            results["tests"]["btc_reentry_after_closed"] = f"FAIL: {exc}"

        # Duplicate OPEN same symbol must fail.
        try:
            conn.execute(
                """
                INSERT INTO scalp_paper_positions
                (symbol, exchange, strategy_id, quantity, entry_price, entry_time,
                 entry_time_epoch, trade_id, status)
                VALUES ('ETHUSDT', 'binance_us', 'scalp', 0.0001, 1.0,
                        datetime('now'), 0, 'dryrun_eth_dup_test', 'OPEN')
                """
            )
            conn.execute("DELETE FROM scalp_paper_positions WHERE trade_id='dryrun_eth_dup_test'")
            results["tests"]["duplicate_open_same_symbol"] = "FAIL: allowed duplicate OPEN"
        except sqlite3.IntegrityError:
            results["tests"]["duplicate_open_same_symbol"] = "PASS"

        # Second CLOSED BTC row allowed.
        try:
            conn.execute(
                """
                INSERT INTO scalp_paper_positions
                (symbol, exchange, strategy_id, quantity, entry_price, entry_time,
                 entry_time_epoch, trade_id, status)
                VALUES ('BTCUSDT', 'binance_us', 'scalp', 0.0001, 1.0,
                        datetime('now'), 0, 'dryrun_btc_closed2', 'CLOSED')
                """
            )
            conn.execute("DELETE FROM scalp_paper_positions WHERE trade_id='dryrun_btc_closed2'")
            results["tests"]["multiple_closed_same_symbol"] = "PASS"
        except sqlite3.IntegrityError as exc:
            results["tests"]["multiple_closed_same_symbol"] = f"FAIL: {exc}"

        conn.rollback()

    print(json.dumps(results, indent=2))
    tests = results["tests"]
    ok = all(str(v).startswith("PASS") for v in tests.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
