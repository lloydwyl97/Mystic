"""Account-wide BUY quote holds. Concurrent entries share one USDT balance."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.services.buy_cash_commitment import (
    buy_failure_keeps_quote,
    committed_buy_quote,
    intent_quote_parts,
)
from backend.services.day_entry_reservations import create_reservation
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.strategy_order_intent import allocate_intent, mark_exchange_order


def _engine(tmp_path, cash: float = 1000.0) -> PortfolioEngine:
    engine = PortfolioEngine(db_path=str(tmp_path / "cash.db"), principal=cash, test_mode=True)
    engine.cash_balance = cash
    engine._available_balance = cash
    return engine


def test_quote_parts_keep_live_remainder_and_known_quote_fee():
    remaining, executed = intent_quote_parts(
        {
            "side": "BUY",
            "status": "PARTIAL",
            "quantity": "0.5",
            "limit_price": "1000",
            "filled_qty": "0.2",
            "filled_cost": "200",
            "filled_cash": "201",
        }
    )
    assert remaining == 300
    assert executed == 201

    _base_remaining, base_executed = intent_quote_parts(
        {
            "side": "BUY",
            "status": "PARTIAL",
            "quantity": "0.5",
            "limit_price": "1000",
            "filled_qty": "0.2",
            "filled_cost": "200",
            "filled_cash": "200",
        }
    )
    assert base_executed == 200


def test_same_engine_symbol_is_one_hold(tmp_path):
    db = tmp_path / "one.db"
    create_reservation(db, decision_id="pre", symbol="BTC/USDT", notional_usd=400, sleeve="DAY_V2", ttl_sec=3600)
    create_reservation(db, decision_id="exec", symbol="BTC/USDT", notional_usd=420, sleeve="ACTIVE", ttl_sec=3600)
    assert committed_buy_quote(str(db)) == 420


def test_working_intent_counts_after_the_reservation_expires(tmp_path):
    db = str(tmp_path / "intent.db")
    create_reservation(db, decision_id="lost", symbol="ETH/USDT", notional_usd=300, sleeve="SCALP_V2", ttl_sec=1)
    allocate_intent(
        db,
        engine_id="SCALP_V2",
        trade_id="t-eth",
        decision_id="lost",
        symbol="ETH/USDT",
        side="BUY",
        quantity=0.3,
        limit_price=1000,
    )
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE day_entry_reservations SET expires_at=? WHERE decision_id='lost'", (time.time() - 5,))
    assert committed_buy_quote(db) == 300


def test_proven_absent_buy_does_not_hold_cash(tmp_path):
    db = str(tmp_path / "absent.db")
    cid = allocate_intent(
        db,
        engine_id="DAY_V2",
        trade_id="t-btc",
        decision_id="gone",
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        limit_price=40000,
    )
    mark_exchange_order(db, cid, "", status="NO_VENUE_ORDER")
    assert committed_buy_quote(db) == 0
    assert buy_failure_keeps_quote("ORDER_UNCONFIRMED") is True
    assert buy_failure_keeps_quote("EXPIRED_UNFILLED") is False
    assert buy_failure_keeps_quote("") is True


@pytest.mark.asyncio
async def test_two_and_eight_buys_cannot_overcommit(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.portfolio_engine.ENABLE_SLEEVE_BLOCKING", False)
    engine = _engine(tmp_path)
    symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")

    async def reserve(engine_id: str, symbol: str, notional: float, decision: str) -> float:
        async with engine._global_cash_lock:
            ok, _reason = engine._try_reserve_entry(
                symbol,
                notional,
                decision_id=decision,
                sleeve=engine_id,
                engine_id=engine_id,
            )
            if not ok:
                return 0.0
            key = f"{engine_id}::{symbol}"
            return float(engine._entry_reservations[key]["notional"])

    pair = await asyncio.gather(
        reserve("DAY_V2", "BTC/USDT", 700, "day-btc"),
        reserve("SCALP_V2", "ETH/USDT", 700, "scalp-eth"),
    )
    assert sum(pair) <= 1000
    assert sum(pair) == 1000

    engine._entry_reservations.clear()
    import sqlite3

    with sqlite3.connect(engine.db_path) as conn:
        conn.execute("UPDATE day_entry_reservations SET status='RELEASED' WHERE status='ACTIVE'")

    eight = await asyncio.gather(*[reserve(engine_id, symbol, 300, f"{engine_id}-{symbol}") for engine_id in ("DAY_V2", "SCALP_V2") for symbol in symbols])
    assert sum(eight) <= 1000
    assert sum(1 for amount in eight if amount > 0) < 8


@pytest.mark.asyncio
async def test_scalp_cash_sees_day_reservation_and_sell_block_remains(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.portfolio_engine.ENABLE_SLEEVE_BLOCKING", False)
    engine = _engine(tmp_path)
    async with engine._global_cash_lock:
        ok, reason = engine._try_reserve_entry("BTC/USDT", 700, decision_id="day-btc", sleeve="DAY_V2", engine_id="DAY_V2")
    assert ok, reason
    assert float(engine._pending_buy_notional()) == 700
    free = float(engine._available_balance) - float(engine._pending_buy_notional())
    assert free == 300

    from backend.services.strategy_order_intent import allocate_intent as _alloc
    from backend.services.strategy_order_intent import mark_exchange_order as _mark
    from backend.services.strategy_order_intent import sell_intent_blocks

    cid = _alloc(engine.db_path, engine_id="DAY_V2", trade_id="sell-btc", decision_id="s", symbol="ETH/USDT", side="SELL", quantity=1, limit_price=1)
    _mark(engine.db_path, cid, "1", status="OPEN")
    assert sell_intent_blocks(engine.db_path, "ETH/USDT") is True
    allowed, block = await engine._can_open_position("ETH/USDT", 25, engine_id="SCALP_V2")
    assert allowed is False
    assert block == "WORKING_SELL_SAME_SYMBOL"
    allowed_other, _other = await engine._can_open_position("SOL/USDT", 25, engine_id="SCALP_V2")
    assert allowed_other is True
