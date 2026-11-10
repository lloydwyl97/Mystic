"""Crash windows for a strategy order whose identity is stored before submit."""

from __future__ import annotations

import sqlite3

import pytest

from backend.services.filled_asset_custody import remember
from backend.services.live_exchange_equity import backfill_exchange_reconciled_orders
from backend.services.live_order_identity import OrderIdentity, fills_for_order, record_fill
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.strategy_order_intent import (
    adopt_venue_fill,
    allocate_intent,
    apply_venue_order,
    client_order_id_for,
    mark_exchange_order,
    recovered_sell_order,
)
from backend.services.trade_learning_writer import TradeLearningRecord, record_trade_outcome

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")


def _buy(db: str, engine: str, symbol: str, trade: str, *, qty: float = 0.01, price: float = 100.0) -> str:
    return allocate_intent(
        db,
        engine_id=engine,
        trade_id=trade,
        decision_id=f"dec-{trade}",
        symbol=symbol,
        side="BUY",
        quantity=qty,
        limit_price=price,
        setup="RANGE_BOUNCE",
        regime="btcup_vollo",
        feature_version="5",
    )


def _paper_qty(db: str, trade: str) -> float | None:
    conn = sqlite3.connect(db)
    try:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='paper_trades'").fetchone()
        if exists is None:
            return None
        row = conn.execute("SELECT quantity FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY'", (trade,)).fetchone()
    finally:
        conn.close()
    return None if row is None else float(row[0])


def test_client_order_id_is_stable_and_fits_the_venue_limit() -> None:
    cid = client_order_id_for(engine_id="DAY_V2", trade_id="mystic_BTC/USDT_1", side="BUY")
    assert cid == client_order_id_for(engine_id="DAY_V2", trade_id="mystic_BTC/USDT_1", side="BUY")
    assert len(cid) == 32
    assert len(f"{cid}-i") <= 36


def test_eight_buy_windows_recover_exactly_once(tmp_path) -> None:
    db = str(tmp_path / "w.db")
    engine = PortfolioEngine(db_path=db, principal=1000.0, test_mode=True)
    engine._ensure_db_schema()
    # 1-3. Identity exists before submit, ack, and fill persistence. No lot yet.
    cid = _buy(db, "DAY_V2", "BTC/USDT", "t-day")
    assert _paper_qty(db, "t-day") is None
    # 4. Venue fill arrives before custody. Client id with the IOC suffix still maps.
    first = adopt_venue_fill(db, client_order_id=f"{cid}-i", exchange_order_id="ord-1", fill_id="f1", quantity=0.004, price=100.0, fee=0.01)
    assert first["adopted"] is True
    assert first["decision_id"] == "dec-t-day"
    assert first["setup"] == "RANGE_BOUNCE"
    assert first["feature_version"] == "5"
    # 7. Partial fill after restart. Same fill id is not applied twice.
    again = adopt_venue_fill(db, client_order_id=cid, exchange_order_id="ord-1", fill_id="f1", quantity=0.004, price=100.0, fee=0.01)
    assert again["adopted"] is False
    second = adopt_venue_fill(db, client_order_id=cid, exchange_order_id="ord-1", fill_id="f2", quantity=0.006, price=110.0, fee=0.02)
    assert second["filled_qty"] == pytest.approx(0.01)
    assert second["vwap"] == pytest.approx(106.0)
    assert _paper_qty(db, "t-day") == pytest.approx(0.01)
    # 5. Custody exists, position commit has not happened.
    remember(db, engine_id="DAY_V2", symbol="BTC/USDT", trade_id="t-day", quantity=0.01, fill_price=106.0, fee=0.03, cash_debit=1.06, exchange_order_id="ord-1")

    async def _restore() -> int:
        return await engine.restore_filled_asset_custody()

    import asyncio

    assert asyncio.run(_restore()) == 1
    # 6. After the lot exists, a second recovery does not duplicate it.
    assert asyncio.run(_restore()) == 0
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM paper_trades WHERE trade_id='t-day' AND side='BUY'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE trade_id='t-day'").fetchone()[0] == 1
        held = conn.execute("SELECT entry_time, entry_decision_id, engine_id, quantity, highest_price, lowest_price FROM portfolio_engine_positions WHERE trade_id='t-day'").fetchone()
        intent_at = conn.execute("SELECT created_at FROM strategy_order_intents WHERE trade_id='t-day'").fetchone()[0]
        explain = conn.execute("SELECT explainability_json FROM paper_trades WHERE trade_id='t-day'").fetchone()
    finally:
        conn.close()
    assert held[0] == pytest.approx(intent_at)
    assert held[1] == "dec-t-day"
    assert held[2] == "DAY_V2"
    assert held[3] == pytest.approx(0.01)
    assert held[4] == pytest.approx(106.0)
    assert held[5] == pytest.approx(106.0)
    assert explain and "RANGE_BOUNCE" in str(explain[0]) and "feature_version" in str(explain[0])


def test_scalp_and_day_and_four_symbols_recover_without_duplicate_slots(tmp_path) -> None:
    db = str(tmp_path / "m.db")
    intents = []
    for i, symbol in enumerate(SYMBOLS):
        intents.append(_buy(db, "DAY_V2", symbol, f"day-{i}", qty=1.0, price=10.0 + i))
        if i < 2:
            intents.append(_buy(db, "SCALP_V2", symbol, f"scalp-{i}", qty=2.0, price=20.0 + i))
    for i, cid in enumerate(intents):
        adopt_venue_fill(db, client_order_id=cid, exchange_order_id=f"o{i}", fill_id=f"fill-{i}", quantity=1.0, price=10.0)
        adopt_venue_fill(db, client_order_id=cid, exchange_order_id=f"o{i}", fill_id=f"fill-{i}", quantity=1.0, price=10.0)
    conn = sqlite3.connect(db)
    try:
        day = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE trade_id LIKE 'day-%'").fetchone()[0]
        scalp = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE trade_id LIKE 'scalp-%'").fetchone()[0]
    finally:
        conn.close()
    assert day == 4
    assert scalp == 2
    assert day <= 4
    assert scalp <= 4
    assert day + scalp <= 8


def test_sell_close_and_learning_are_exactly_once(tmp_path) -> None:
    db = str(tmp_path / "s.db")
    cid = allocate_intent(
        db,
        engine_id="SCALP_V2",
        trade_id="buy-1",
        decision_id="dec-sell",
        symbol="ETH/USDT",
        side="SELL",
        quantity=0.02,
        limit_price=2500.0,
        setup="RANGE_BOUNCE_SCALP",
        regime="btcup_volhi",
        feature_version="5",
    )
    first = adopt_venue_fill(db, client_order_id=f"{cid}-i", exchange_order_id="sell-ord", fill_id="s1", quantity=0.01, price=2500.0, fee=0.05)
    rest = adopt_venue_fill(db, client_order_id=cid, exchange_order_id="sell-ord", fill_id="s2", quantity=0.01, price=2510.0, fee=0.05)
    dup = adopt_venue_fill(db, client_order_id=cid, exchange_order_id="sell-ord", fill_id="s1", quantity=0.01, price=2500.0, fee=0.05)
    assert first["adopted"] and rest["adopted"] and not dup["adopted"]
    assert rest["filled_qty"] == pytest.approx(0.02)
    assert rest["vwap"] == pytest.approx(2505.0)
    record = TradeLearningRecord(
        symbol="ETH/USDT",
        close_reason="NET_PROFIT_EXIT",
        net_profit_pct=0.001,
        exit_price=2505.0,
        entry_price=2500.0,
        quantity=0.02,
        event_key="close-buy-1",
        extra={"setup": "RANGE_BOUNCE_SCALP", "decision_id": "dec-sell"},
    )
    assert record_trade_outcome(record, db_path=db) is True
    assert record_trade_outcome(record, db_path=db) is True
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes WHERE event_key='close-buy-1'").fetchone()[0] == 1
    finally:
        conn.close()
    reused = recovered_sell_order(db, "buy-1", requested_qty=0.02)
    assert reused is not None
    assert reused["id"] == "sell-ord"
    assert reused["filled"] == pytest.approx(0.02)
    assert reused["average"] == pytest.approx(2505.0)
    again_id = allocate_intent(db, engine_id="SCALP_V2", trade_id="buy-1", symbol="ETH/USDT", side="SELL", quantity=0.01)
    assert again_id != cid


def test_reconciliation_adopts_a_strategy_fill_and_leaves_external_orders_alone(tmp_path) -> None:
    db = str(tmp_path / "r.db")
    cid = _buy(db, "DAY_V2", "BTC/USDT", "t-owned")
    mark_exchange_order(db, cid, "ord-owned", status="ACKED")
    owned = backfill_exchange_reconciled_orders(
        db,
        unmatched=[
            {
                "source": "venue_fill",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": 0.01,
                "unmatched_quantity": 0.01,
                "price": 100.0,
                "exchange_order_id": "ord-owned",
                "venue_trade_id": "fill-owned",
            }
        ],
        known_order_ids=set(),
    )
    assert owned[0]["source"] == "STRATEGY_INTENT"
    assert owned[0]["trade_id"] == "t-owned"
    assert fills_for_order(db, "ord-owned") == []
    again = backfill_exchange_reconciled_orders(
        db,
        unmatched=[
            {
                "source": "venue_fill",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": 0.01,
                "unmatched_quantity": 0.01,
                "price": 100.0,
                "exchange_order_id": "ord-owned",
                "venue_trade_id": "fill-owned",
            }
        ],
        known_order_ids=set(),
    )
    assert again[0]["source"] == "STRATEGY_INTENT"
    external = backfill_exchange_reconciled_orders(
        db,
        unmatched=[
            {
                "source": "venue_fill",
                "symbol": "ETH/USDT",
                "side": "BUY",
                "quantity": 0.02,
                "unmatched_quantity": 0.02,
                "price": 2500.0,
                "exchange_order_id": "ord-external",
                "venue_trade_id": "fill-ext",
            }
        ],
        known_order_ids=set(),
    )
    assert external[0]["source"] == "EXCHANGE_RECONCILED"
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM paper_trades WHERE trade_id='t-owned'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM paper_trades WHERE order_id='ord-external'").fetchone()[0] == 0
    finally:
        conn.close()
    assert len(fills_for_order(db, "ord-external")) == 1


def test_recorded_fill_without_custody_restores_one_lot(tmp_path) -> None:
    db = str(tmp_path / "ledger.db")
    engine = PortfolioEngine(db_path=db, principal=1000.0, test_mode=True)
    engine._ensure_db_schema()
    cid = _buy(db, "DAY_V2", "ETH/USDT", "t-ledger", qty=0.02, price=2500.0)
    mark_exchange_order(db, cid, "ord-ledger", status="ACKED")
    assert record_fill(
        db,
        OrderIdentity(
            symbol="ETH/USDT",
            side="BUY",
            exchange_order_id="ord-ledger",
            client_order_id=f"{cid}-i",
            executed_qty=0.02,
            avg_fill_price=2500.0,
            mystic_trade_id="t-ledger",
            venue_trade_ids=["fill-ledger"],
            decision_id="dec-t-ledger",
        ),
    )
    import asyncio

    assert asyncio.run(engine.restore_filled_asset_custody()) == 1
    assert asyncio.run(engine.restore_filled_asset_custody()) == 0
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM paper_trades WHERE trade_id='t-ledger' AND side='BUY'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE trade_id='t-ledger'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM filled_asset_custody WHERE trade_id='t-ledger'").fetchone()[0] == 1
    finally:
        conn.close()


def test_probe_order_and_aggregate_fill_are_applied_once(tmp_path) -> None:
    db = str(tmp_path / "p.db")
    cid = _buy(db, "SCALP_V2", "SOL/USDT", "t-probe", qty=1.0, price=100.0)
    first = apply_venue_order(
        db,
        {"id": "ord-p", "clientOrderId": f"{cid}-i", "filled": 1.0, "average": 101.0, "fee": {"cost": 0.1}},
    )
    assert first["adopted"] is False
    assert first["reason"] == "NEEDS_TRADES"
    second = apply_venue_order(
        db,
        {
            "id": "ord-p",
            "clientOrderId": cid,
            "trades": [{"id": "leg-1", "qty": 0.4, "price": 100.0}, {"id": "leg-2", "qty": 0.6, "price": 102.0}],
        },
    )
    assert second["adopted"] is True
    assert _paper_qty(db, "t-probe") == pytest.approx(1.0)


def test_four_symbols_restore_one_lot_each(tmp_path) -> None:
    db = str(tmp_path / "slots.db")
    for engine, prefix in (("DAY_V2", "day"), ("SCALP_V2", "scalp")):
        for symbol in SYMBOLS:
            cid = _buy(db, engine, symbol, f"{prefix}-{symbol}", qty=1.0, price=10.0)
            adopt_venue_fill(db, client_order_id=cid, exchange_order_id=f"{prefix}-{symbol}", fill_id=f"{prefix}-{symbol}", quantity=1.0, price=10.0)
    engine = PortfolioEngine(db_path=db, principal=1000.0, test_mode=True)
    engine._ensure_db_schema()

    import asyncio

    assert asyncio.run(engine.restore_filled_asset_custody()) == 8
    assert asyncio.run(engine.restore_filled_asset_custody()) == 0
    conn = sqlite3.connect(db)
    try:
        day = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE engine_id='DAY_V2'").fetchone()[0]
        scalp = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE engine_id='SCALP_V2'").fetchone()[0]
        symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM portfolio_engine_positions").fetchone()[0]
        trades = conn.execute("SELECT COUNT(DISTINCT trade_id) FROM portfolio_engine_positions").fetchone()[0]
    finally:
        conn.close()
    assert day == 4
    assert scalp == 4
    assert day + scalp == 8
    assert symbols == 4
    assert trades == 8
