"""SCALP V2 live cutover: identity, same-move reset, accounting, and the loss-hold veto."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


def test_same_move_blocks_until_price_zone_changes(tmp_path: Path):
    from backend.services.scalp_v2.opportunity import arm_opportunity, mark_opportunity

    db = tmp_path / "t.db"
    oid, blocked = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2700.0)
    assert blocked is False
    assert oid
    again, blocked2 = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2700.2)
    assert again == oid
    assert blocked2 is True
    mark_opportunity(db, "ETH/USDT", oid, "CLOSED")
    still, blocked3 = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2701.0)
    assert still == oid
    assert blocked3 is True
    moved, blocked4 = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2900.0)
    assert blocked4 is False
    assert moved != oid


def test_clock_does_not_create_a_new_opportunity():
    from backend.services.scalp_v2.opportunity import ScalpOpportunityId

    a = ScalpOpportunityId.from_intent("BTC/USDT", "BREAK", "2026-09-21T01:00:00+00:00", arm_price=80000)
    b = ScalpOpportunityId.from_intent("BTC/USDT", "BREAK", "2026-09-21T06:00:00+00:00", arm_price=80000)
    assert a.canonical_id == b.canonical_id


def test_reconcile_rows_do_not_double_count_and_trade_ids_persist(tmp_path: Path):
    from backend.services.scalp_v2.accounting_repair import (
        apply_trade_id_backfill,
        exclude_duplicate_realized,
        record_residual,
        record_supplements,
    )

    db = tmp_path / "a.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            trade_id TEXT,
            side TEXT,
            exit_reason TEXT,
            exit_type TEXT,
            pnl REAL,
            pnl_usd_net REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE live_exchange_fills (
            id INTEGER PRIMARY KEY,
            exchange_order_id TEXT,
            venue_trade_ids_json TEXT,
            fill_ids_json TEXT,
            fill_count INTEGER
        )
        """
    )
    conn.execute("INSERT INTO paper_trades VALUES (1,'s1','SELL','NET_PROFIT_EXIT','NET_PROFIT_EXIT',1.0,1.0)")
    conn.execute("INSERT INTO paper_trades VALUES (2,'s2','SELL','EXCHANGE_RECONCILE_CLOSE','EXCHANGE_RECONCILE_CLOSE',0.13,NULL)")
    conn.execute("INSERT INTO paper_trades VALUES (3,'s3','SELL','HUMAN_MANUAL_SELL','HUMAN_MANUAL_SELL',0,0)")
    conn.execute("INSERT INTO live_exchange_fills VALUES (9,'1840254087','[]','[]',1)")
    flagged = exclude_duplicate_realized(conn)
    assert flagged == 2
    realized = conn.execute("SELECT ROUND(SUM(COALESCE(pnl_usd_net,pnl)),2) FROM paper_trades WHERE COALESCE(counts_toward_realized,1)=1").fetchone()[0]
    assert realized == 1.0
    n = apply_trade_id_backfill(
        conn,
        {"1840254087": {"trade_ids": ["31800251"], "order_ids": ["1840254087"], "taker_or_maker": "taker"}},
    )
    assert n == 1
    stored = conn.execute("SELECT venue_trade_ids_json, taker_or_maker FROM live_exchange_fills WHERE id=9").fetchone()
    assert "31800251" in stored[0]
    assert stored[1] == "taker"
    added = record_supplements(
        conn,
        [
            {
                "exchange_order_id": "1840253877",
                "symbol": "BTC/USDT",
                "side": "SELL",
                "trade_ids": ["31800220"],
                "qty": 0.00012,
                "price": 85853.66,
                "fee_amount": 0.00206049,
                "fee_asset": "USDT",
                "taker_or_maker": "taker",
                "parent_order_id": "1840254087",
                "note": "merged sibling",
            }
        ],
    )
    assert added == 1
    assert record_supplements(conn, [{"exchange_order_id": "1840253877", "symbol": "BTC/USDT", "side": "SELL", "trade_ids": ["31800220"], "qty": 0.00012, "price": 1, "fee_amount": 9}]) == 0
    record_residual(conn, "ETH/USDT", 0.01765762, 0.01729654, "exchange minus position lot")
    gap = conn.execute("SELECT residual_qty FROM documented_balance_residuals WHERE symbol='ETH/USDT'").fetchone()[0]
    assert abs(gap - 0.00036108) < 1e-8
    conn.close()


def test_legacy_open_lots_become_exit_only(tmp_path: Path):
    from backend.services.day_v2.migrations import apply_all_migrations

    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE portfolio_engine_positions (symbol TEXT, status TEXT, engine_id TEXT)")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('BTC/USDT','ACTIVE','LEGACY_DAY_LIVE')")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('ETH/USDT','CLOSED','LEGACY_DAY_LIVE')")
    conn.commit()
    conn.close()
    apply_all_migrations(str(db))
    conn = sqlite3.connect(db)
    rows = dict(conn.execute("SELECT symbol, engine_id FROM portfolio_engine_positions"))
    assert rows["BTC/USDT"] == "LEGACY_EXIT_ONLY"
    assert rows["ETH/USDT"] == "LEGACY_DAY_LIVE"
    apply_all_migrations(str(db))
    rows = dict(conn.execute("SELECT symbol, engine_id FROM portfolio_engine_positions"))
    assert rows["BTC/USDT"] == "LEGACY_EXIT_ONLY"


def test_paper_scalp_package_has_no_live_order_call():
    root = Path("backend/services/binance_scalp")
    text = "\n".join(p.read_text() for p in root.glob("*.py"))
    assert "place_order" not in text
    assert "create_order" not in text


def test_core_startup_does_not_launch_paper_scalp():
    text = Path("start_mystic.sh").read_text()
    assert "# start_scalp || return 1  # DISABLED in core mode" in text
    assert "all|ai|collector|agents|ai_position_tracker|ai_outcome_bridge)" in text
    assert 'retired_mode "$MODE"' in text
    watchdog = Path("watchdog_mystic.sh").read_text()
    assert "backend.services.binance_scalp.runner" not in watchdog


@pytest.mark.asyncio
async def test_consecutive_losses_do_not_veto_when_trailing_buy_is_off(monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "off")
    import backend.services.portfolio_engine as pe

    monkeypatch.setattr(pe, "ENABLE_GOVERNANCE_ENFORCEMENT", True)
    monkeypatch.setattr(pe, "governance_risk_governor_shadow_only", lambda: False)
    engine = pe.PortfolioEngine(principal=228.0, test_mode=True)
    engine.cash_balance = 220.0
    engine._available_balance = 220.0
    engine._total_open_risk = 0.0
    engine._get_loss_hold_until = pytest.importorskip("unittest.mock").AsyncMock(return_value=time.time() + 600)
    engine.get_rolling_24h_risk_metrics = pytest.importorskip("unittest.mock").AsyncMock(return_value=(0.0, pe.MAX_CONSEC_LOSSES))
    from unittest.mock import AsyncMock

    engine._get_loss_hold_until = AsyncMock(return_value=time.time() + 600)
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, pe.MAX_CONSEC_LOSSES))
    allowed, reason = await engine._can_open_position("SOL/USDT", 40.0)
    assert allowed is True
    assert reason != "HOLD_CONSEC_LOSSES"
