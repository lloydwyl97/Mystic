#!/usr/bin/env python3
"""Idempotent backfill for BTC live recovered close (exchange order 31362412)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from backend.services.live_recovered_close_writer import RecoveredCloseFill, persist_recovered_close


def main() -> int:
    fill = RecoveredCloseFill(
        buy_trade_id="mystic_BTC/USDT_1780519579629",
        symbol="BTC/USDT",
        quantity=0.0003,
        entry_price=65142.79,
        exit_price=65400.66,
        exchange_sell_order_id="31362412",
        closed_at_iso="2026-06-03T21:05:41.418126+00:00",
        closed_at_epoch=1780520741.418126,
        source="periodic_reconcile",
        fill_recovered=True,
        realized_profit_usd=0.07735503000000078,
        fee_usd=0.0,
        entry_time_epoch=1780519579.7120807,
        close_ledger_id=50,
        mode="live",
    )
    result = persist_recovered_close(fill)
    print(result)
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
