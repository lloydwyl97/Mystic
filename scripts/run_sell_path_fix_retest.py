#!/usr/bin/env python3
"""Retest sell-path fixes: negative real-book block + positive stubbed-book sell."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    for raw_line in _env.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

PROD_DB = PROJECT_ROOT / "mystic_trading.db"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
REPORT: dict[str, Any] = {"timestamp_utc": TS}


def baseline(db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db))
    try:
        return {
            "positions": conn.execute("SELECT symbol, quantity FROM portfolio_engine_positions ORDER BY symbol").fetchall(),
            "paper_trades": conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0],
            "paper_sells": conn.execute("SELECT COUNT(*) FROM paper_trades WHERE UPPER(side)='SELL'").fetchone()[0],
            "audit": conn.execute("SELECT COUNT(*) FROM portfolio_engine_audit").fetchone()[0],
            "audit_sells": conn.execute("SELECT COUNT(*) FROM portfolio_engine_audit WHERE UPPER(action)='SELL'").fetchone()[0],
            "close_ledger": conn.execute("SELECT COUNT(*) FROM position_close_ledger").fetchone()[0],
            "learning": conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes").fetchone()[0],
        }
    finally:
        conn.close()


def copy_db(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    s = sqlite3.connect(str(src))
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        s.close()
        d.close()


async def _run_xrp_sell_attempt(
    test_db: Path,
    *,
    stub_book: Any | None,
) -> dict[str, Any]:
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
    from backend.services.day_active_market_bundle import (
        async_fetch_day_active_ohlcv_bundle,
        validate_day_active_bundle,
    )
    from backend.services.portfolio_engine import PortfolioEngine
    from backend.utils.symbols import normalize_symbol

    engine = PortfolioEngine(db_path=str(test_db), test_mode=True)
    engine._live_execution_enabled = False
    engine._live_service = None
    await engine.initialize_from_db()

    ns = normalize_symbol("XRP/USDT")
    pos = engine.open_positions.get(ns)
    if not pos:
        return {"error": "no_xrp_position"}

    entry = float(pos.entry_price)
    float(pos.quantity)
    test_mark = entry * (1.0 + ESTIMATED_ROUNDTRIP_COST + MIN_NET_PROFIT_TO_SELL + 0.001)
    engine._price_cache.set(ns, test_mark)

    hold_bds: dict[str, dict] = {}
    hold_ms: dict[str, list[str]] = {}
    from backend.services.live_market_data import live_market_data_service

    if live_market_data_service:
        bd = await async_fetch_day_active_ohlcv_bundle(live_market_data_service, ns)
        ok, miss = validate_day_active_bundle(bd)
        hold_bds[ns] = bd if isinstance(bd, dict) else {}
        hold_ms[ns] = [] if ok else list(miss)
    else:
        hold_ms[ns] = ["no_live_md"]

    current_bar = int(time.time() / 60) * 60

    async def _go() -> Any:
        return await engine._check_exit_conditions(
            pos,
            test_mark,
            current_bar,
            day_hold_bundle=hold_bds.get(ns),
            day_hold_missing=hold_ms.get(ns),
        )

    if stub_book is not None:
        with patch("backend.services.protected_limit_execution._fetch_order_book", stub_book):
            result = await _go()
    else:
        result = await _go()

    after = baseline(test_db)
    sell_row = None
    conn = sqlite3.connect(str(test_db))
    try:
        sell_row = conn.execute("SELECT trade_id, price, pnl, mode, diagnostics_json FROM paper_trades WHERE UPPER(side)='SELL' ORDER BY rowid DESC LIMIT 1").fetchone()
        learn = conn.execute("SELECT close_reason, net_profit_usd, net_profit_pct, extra_json FROM trade_learning_outcomes ORDER BY id DESC LIMIT 1").fetchone()
        audit = conn.execute("SELECT action, symbol FROM portfolio_engine_audit WHERE UPPER(action)='SELL' ORDER BY id DESC LIMIT 1").fetchone()
        close = conn.execute("SELECT close_reason, realized_profit FROM position_close_ledger ORDER BY id DESC LIMIT 1").fetchone()
        xrp = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE symbol=?", (ns,)).fetchone()
        reject = conn.execute("SELECT reason, filter_name FROM portfolio_engine_rejects ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()

    learn_extra = {}
    if learn and learn[3]:
        try:
            learn_extra = json.loads(learn[3])
        except json.JSONDecodeError:
            pass

    return {
        "test_mark": test_mark,
        "entry": entry,
        "exit_result": result is not None,
        "after": after,
        "xrp_still_open": xrp is not None and float(xrp[0] or 0) > 0,
        "sell_row": sell_row,
        "audit_sell": audit,
        "close_row": close,
        "learning": learn,
        "good_trade": learn_extra.get("good_trade"),
        "bad_trade": learn_extra.get("bad_trade"),
        "last_reject": reject,
    }


async def main() -> None:
    from backend.config.protected_execution import DEPTH_INSUFFICIENT, PRICE_IMPACT_TOO_HIGH
    from backend.services.protected_limit_execution import run_protected_preflight

    REPORT["production_baseline"] = baseline(PROD_DB)

    # --- Negative test: real book, injected profitable mark ---
    neg_db = PROJECT_ROOT / f"mystic_trading.db.test_sell_path_fix_{TS}_neg"
    copy_db(PROD_DB, neg_db)
    neg = await _run_xrp_sell_attempt(neg_db, stub_book=None)
    REPORT["negative_test"] = neg
    REPORT["negative_test"]["expected_block"] = True
    REPORT["negative_test"]["blocked"] = (
        not neg.get("exit_result")
        and neg.get("xrp_still_open")
        and neg["after"]["paper_sells"] == 0
        and neg.get("last_reject") is not None
        and str(neg["last_reject"][0])
        in {
            "EXECUTABLE_NET_PROFIT_BELOW_FLOOR",
            "PROTECTED_FILL_NOT_PROFITABLE",
        }
    )

    # --- Positive test: stubbed profitable book (isolated harness only) ---
    pos_db = PROJECT_ROOT / f"mystic_trading.db.test_sell_path_fix_{TS}_pos"
    copy_db(PROD_DB, pos_db)

    conn = sqlite3.connect(str(pos_db))
    entry = float(conn.execute("SELECT entry_price FROM portfolio_engine_positions WHERE symbol='XRP/USDT'").fetchone()[0])
    qty = float(conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE symbol='XRP/USDT'").fetchone()[0])
    conn.close()

    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL

    profitable_px = entry * (1.0 + ESTIMATED_ROUNDTRIP_COST + MIN_NET_PROFIT_TO_SELL + 0.002)

    async def _stub_book(_sym: str):
        bids = [[profitable_px, qty * 3.0]]
        asks = [[profitable_px * 1.0004, qty * 3.0]]
        return bids, asks, 0.0

    pos = await _run_xrp_sell_attempt(pos_db, stub_book=_stub_book)
    REPORT["positive_test"] = pos
    REPORT["positive_test"]["stub_fill_price"] = profitable_px
    REPORT["positive_test"]["passed"] = (
        pos.get("exit_result")
        and pos["after"]["paper_sells"] == 1
        and pos["after"]["audit_sells"] == 1
        and pos["after"]["close_ledger"] == 1
        and pos["after"]["learning"] == 1
        and pos.get("good_trade") is True
        and pos.get("bad_trade") is False
        and not pos.get("xrp_still_open")
    )

    # --- Oversized reject ---
    pf = await run_protected_preflight(symbol="XRP/USDT", side="SELL", quantity=qty * 5000, reference_price=profitable_px, live_capable=False)
    REPORT["oversized_reject"] = {
        "passed": pf.passed,
        "reason": pf.reject_reason,
        "ok": (not pf.passed and pf.reject_reason in (DEPTH_INSUFFICIENT, PRICE_IMPACT_TOO_HIGH)),
    }

    REPORT["production_after"] = baseline(PROD_DB)
    REPORT["production_unchanged"] = REPORT["production_baseline"] == REPORT["production_after"]

    for path in ["/health", "/api/portfolio-engine/status", "/api/portfolio-engine/positions", "/api/portfolio-engine/execution-protection", "/api/portfolio-engine/learning-status?limit=20"]:
        try:
            with urllib.request.urlopen(f"http://localhost:8000{path}", timeout=10) as r:
                REPORT.setdefault("endpoints", {})[path] = r.status
        except Exception as ex:
            REPORT.setdefault("endpoints", {})[path] = str(ex)


if __name__ == "__main__":
    asyncio.run(main())
    print(json.dumps(REPORT, indent=2, default=str))
