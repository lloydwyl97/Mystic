"""Exchange dust must stay in equity without becoming a trade or a slot."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from decimal import Decimal
from unittest.mock import AsyncMock

from backend.config.execution_cost_model import TAKER_COMMISSION_PCT
from backend.services.live_exchange_equity import (
    FORWARD_BASELINE_LABEL,
    LIFETIME_CONTRIBUTED_CAPITAL,
    backfill_exchange_reconciled_orders,
    backfill_provable_fill_identities,
    build_exchange_equity,
    cap_qty_coverage_pct,
    dust_trade_id,
    exact_dust_quantity,
    mark_dust_asset,
    persist_current_dust_snapshot,
    reconstruct_forward_baseline_dust,
    should_import_exchange_dust,
)
from backend.services.live_order_identity import fills_for_order, record_exchange_reconciled
from backend.services.live_pnl_reconciliation import _reconcile_symbol
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine


def test_all_nonzero_assets_enter_gross_equity():
    out = build_exchange_equity(
        cash_usdt="223.92463088",
        active_marks=[],
        dust_marks=[
            {"symbol": "BTC/USDT", "asset": "BTC", "quantity": "0.00000997", "executable_bid": "80000"},
            {"symbol": "ETH/USDT", "asset": "ETH", "quantity": "0.00009134", "executable_bid": "2600"},
            {"symbol": "SOL/USDT", "asset": "SOL", "quantity": "0.0008216", "executable_bid": "112"},
            {"symbol": "XRP/USDT", "asset": "XRP", "quantity": "0.0938", "executable_bid": "1.39"},
        ],
    )
    assert Decimal(out["cash_usdt"]) == Decimal("223.92463088")
    assert Decimal(out["dust_market_value"]) > 0
    assert Decimal(out["gross_exchange_equity"]) == Decimal(out["cash_usdt"]) + Decimal(out["dust_market_value"])
    assert Decimal(out["net_liquidatable_equity"]) == Decimal(out["gross_exchange_equity"]) - Decimal(out["estimated_liquidation_cost"])
    assert len(out["dust_by_coin"]) == 4
    assert out["lifetime_contributed_capital"] == LIFETIME_CONTRIBUTED_CAPITAL
    assert "not contributed principal" in out["forward_baseline_label"]


def test_lot_floor_cannot_write_dust_off():
    engine = PortfolioEngine(principal=228.06746265, test_mode=True)
    assert engine._floor_to_step(0.00000997, 0.00001) == 0.0
    assert exact_dust_quantity("0.00000997") == Decimal("0.00000997")
    assert exact_dust_quantity("0.00009134") == Decimal("0.00009134")
    assert exact_dust_quantity("0.0008216") == Decimal("0.0008216")
    assert exact_dust_quantity("0.0938") == Decimal("0.0938")


def test_dust_uses_bid_and_sell_fee():
    marked = mark_dust_asset(symbol="BTC/USDT", asset="BTC", quantity="0.00000997", executable_bid="80000")
    gross = Decimal("0.00000997") * Decimal("80000")
    fee = gross * Decimal(str(TAKER_COMMISSION_PCT))
    assert Decimal(marked["dust_market_value"]) == gross
    assert Decimal(marked["estimated_liquidation_cost"]) == fee
    assert Decimal(marked["net_liquidatable_value"]) == gross - fee


def test_dust_import_is_idempotent():
    assert should_import_exchange_dust(exchange_qty="0.00000997") == "import"
    assert (
        should_import_exchange_dust(
            existing_status="DUST_PENDING",
            existing_trade_id=dust_trade_id("BTC/USDT"),
            existing_qty="0.00000997",
            exchange_qty="0.00000997",
        )
        == "skip"
    )
    assert (
        should_import_exchange_dust(
            existing_status="DUST_PENDING",
            existing_trade_id=dust_trade_id("BTC/USDT"),
            existing_qty="0.00000997",
            exchange_qty="0.00001",
        )
        == "update"
    )
    assert should_import_exchange_dust(existing_status="ACTIVE", existing_qty="0.01", exchange_qty="0.00000997") == "skip"


def test_dust_does_not_consume_slot_or_block_buy():
    engine = PortfolioEngine(principal=228.06746265, test_mode=True)
    engine.cash_balance = 223.92
    engine.open_positions["BTC/USDT"] = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.00000997,
        entry_price=80000.0,
        entry_time=time.time(),
        trade_id=dust_trade_id("BTC/USDT"),
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        status="DUST_PENDING",
    )
    assert engine._count_live_slots() == 0
    assert engine._day_position_blocks_new_entry(engine.open_positions["BTC/USDT"]) is False
    assert engine._day_path_ev_entry_block_reason("BTC/USDT", 4) is None


def test_dust_is_not_realized_pnl():
    engine = PortfolioEngine(principal=228.06746265, test_mode=True)
    engine._realized_pnl = -10.55
    before = engine._realized_pnl
    engine.open_positions["ETH/USDT"] = OpenPosition(
        symbol="ETH/USDT",
        quantity=0.00009134,
        entry_price=2600.0,
        entry_time=time.time(),
        trade_id=dust_trade_id("ETH/USDT"),
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        status="DUST_PENDING",
    )
    assert engine._realized_pnl == before


def test_baseline_is_not_contributed_capital():
    out = build_exchange_equity(cash_usdt="223.92463088", baseline_dust_known=False)
    assert out["lifetime_contributed_capital"] == "UNKNOWN"
    assert out["forward_net_equity_change"] is None
    assert "cash-only" in out["uncertainty"]
    assert FORWARD_BASELINE_LABEL in out["forward_baseline_label"]


def test_order_id_match_backfills_only_provable_ids(tmp_path):
    db = str(tmp_path / "id.db")
    recorded = [{"symbol": "ETH/USDT", "side": "SELL", "order_id": "1587493960", "local_id": 1896, "qty": 0.0215}]
    venue = [
        {
            "symbol": "ETH/USDT",
            "side": "SELL",
            "order": "1587493960",
            "id": "18721599",
            "qty": 0.0215,
            "price": 2592.76,
            "cost": 55.74,
            "fee_cost": 0.01,
            "fee_ccy": "USDT",
            "timestamp": "2026-09-18T17:20:06Z",
        },
        {"symbol": "SOL/USDT", "side": "BUY", "order": "911230782", "id": "11855406", "qty": 0.5, "price": 111.46, "cost": 55.73},
    ]
    written = backfill_provable_fill_identities(db, recorded=recorded, venue_fills=venue)
    assert written == [{"symbol": "ETH/USDT", "side": "SELL", "exchange_order_id": "1587493960", "venue_trade_id": "18721599"}]
    again = backfill_provable_fill_identities(db, recorded=recorded, venue_fills=venue)
    assert again == [{"symbol": "ETH/USDT", "side": "SELL", "exchange_order_id": "1587493960", "venue_trade_id": "18721599"}]
    rows = fills_for_order(db, "1587493960")
    assert len(rows) == 1
    assert "18721599" in str(rows[0].get("venue_trade_ids_json") or rows[0].get("fill_ids_json") or "")
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "paper_trades" not in tables
    conn.close()
    skipped = backfill_provable_fill_identities(
        db,
        recorded=[{"symbol": "BTC/USDT", "side": "BUY", "order_id": "no-such-order", "local_id": 1, "qty": 0.001}],
        venue_fills=[{"symbol": "BTC/USDT", "side": "BUY", "order": "other-order", "id": "invented", "qty": 0.001, "price": 1.0, "cost": 1.0}],
    )
    assert skipped == []


def test_exchange_reconciled_does_not_invent_decision(tmp_path):
    db = str(tmp_path / "recon.db")
    ok = record_exchange_reconciled(
        db,
        symbol="BTC/USDT",
        side="BUY",
        exchange_order_id="1837670272",
        client_order_id="",
        venue_trade_ids=["1"],
        executed_qty=0.001,
        avg_fill_price=80000.0,
        cost_quote=80.0,
        classification="full_buys_lacking_local",
    )
    assert ok is True
    rows = fills_for_order(db, "1837670272")
    assert len(rows) == 1
    assert rows[0]["mystic_trade_id"] in (None, "")
    assert rows[0]["decision_id"] in (None, "")
    assert rows[0]["intent_id"] in (None, "")
    raw = rows[0]["raw_json"]
    assert "EXCHANGE_RECONCILED" in raw
    assert '"decision_id": null' in raw
    again = record_exchange_reconciled(db, symbol="BTC/USDT", side="BUY", exchange_order_id="1837670272", executed_qty=0.001, avg_fill_price=80000.0)
    assert again is False
    written = backfill_exchange_reconciled_orders(
        db,
        unmatched=[
            {
                "source": "venue_fill",
                "symbol": "ETH/USDT",
                "side": "BUY",
                "quantity": 0.02,
                "unmatched_quantity": 0.02,
                "price": 2500.0,
                "exchange_order_id": "1587098371",
                "venue_trade_id": "9",
                "dollar_discrepancy": 50.0,
            }
        ],
        known_order_ids=set(),
    )
    assert written == [{"symbol": "ETH/USDT", "side": "BUY", "exchange_order_id": "1587098371", "source": "EXCHANGE_RECONCILED"}]
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "paper_trades" not in tables
    conn.close()


def test_qty_coverage_cannot_exceed_100():
    assert cap_qty_coverage_pct(matched_qty=100.4, venue_qty=100.0) == 100.0
    rec = _reconcile_symbol(
        "BTC/USDT",
        {("BTC/USDT", "BUY"): [{"qty": 0.002, "price": 1.0, "ts": 1, "order_id": "1", "exit_type": ""}]},
        {"fills": [{"id": "t1", "order": "1", "side": "BUY", "qty": 0.001, "cost": 1.0, "ts": 1, "fee_cost": 0.0, "fee_ccy": "USDT"}]},
    )
    assert rec.qty_coverage_pct <= 100.0
    rec2 = _reconcile_symbol(
        "ETH/USDT",
        {("ETH/USDT", "SELL"): [{"qty": 0.0215, "price": 2592.76, "ts": 1, "order_id": "1587493960", "exit_type": ""}]},
        {
            "fills": [
                {"id": "18721599", "order": "1587493960", "side": "SELL", "qty": 0.0215, "cost": 55.74, "ts": 1, "fee_cost": 0.0, "fee_ccy": "USDT"},
            ]
        },
    )
    assert rec2.id_matched_rows == 1
    assert rec2.unmatched_recorded_rows == 0
    assert rec2.unmatched_venue_fills == 0


def test_retain_dust_restart_is_idempotent():
    engine = PortfolioEngine(principal=228.06746265, test_mode=True)
    engine._live_execution_enabled = True
    engine._executable_bid = AsyncMock(return_value=(80000.0, "executable_bid"))
    engine._persist_position_to_sqlite = AsyncMock()

    async def _run():
        await engine._retain_exchange_dust(symbol="BTC/USDT", asset="BTC", quantity="0.00000997", mark_hint=80000.0)
        first = engine.open_positions["BTC/USDT"].trade_id
        await engine._retain_exchange_dust(symbol="BTC/USDT", asset="BTC", quantity="0.00000997", mark_hint=80000.0)
        return first, engine.open_positions["BTC/USDT"].trade_id, len(engine.open_positions)

    first, second, n = asyncio.run(_run())
    assert first == second == dust_trade_id("BTC/USDT")
    assert n == 1
    assert engine.open_positions["BTC/USDT"].status == "DUST_PENDING"
    assert engine._count_live_slots() == 0


def test_baseline_dust_unknown_without_snapshot(tmp_path):
    db = str(tmp_path / "base.db")
    out = reconstruct_forward_baseline_dust(db)
    assert out["known"] is False
    assert out["net_liquidatable"] is None
    assert "not total-account P&L" in out["uncertainty"]


def test_dust_snapshot_merge_is_idempotent(tmp_path):
    db = str(tmp_path / "snap.db")
    persist_current_dust_snapshot(db, [{"symbol": "BTC/USDT", "quantity": "0.00000997"}])
    persist_current_dust_snapshot(db, [{"symbol": "ETH/USDT", "quantity": "0.00009134"}])
    persist_current_dust_snapshot(db, [{"symbol": "BTC/USDT", "quantity": "0.00000997"}])
    from backend.services.live_account_basis import load_operational_json
    from backend.services.live_exchange_equity import CURRENT_DUST_SNAPSHOT_KEY

    coins = load_operational_json(db, CURRENT_DUST_SNAPSHOT_KEY).get("coins") or []
    symbols = {c["symbol"] for c in coins}
    assert symbols == {"BTC/USDT", "ETH/USDT"}
