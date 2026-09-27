"""A confirmed SCALP V2 live BUY must land in the canonical live_exchange_fills ledger."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from tests.test_scalp_v2_live_fill_ownership import _arm, _buy, _engine, _one

BUY_ORDER_ID = "1599263273"
BUY_CLIENT_ID = "x-TKT5PX2Fscalpbuy001"
BUY_TRADE_ID = "26770010"


def _buy_order() -> dict:
    return {
        "id": BUY_ORDER_ID,
        "clientOrderId": BUY_CLIENT_ID,
        "status": "closed",
        "filled": 0.0161,
        "average": 2691.12,
        "timestamp": 1790531807000,
        "info": {
            "fills": [
                {
                    "qty": "0.0161",
                    "price": "2691.12",
                    "commission": "0.0000032",
                    "commissionAsset": "ETH",
                    "tradeId": BUY_TRADE_ID,
                }
            ]
        },
    }


def _fills(db_path: str, order_id: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM live_exchange_fills WHERE exchange_order_id=?", (order_id,))]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_successful_buy_writes_canonical_fill_row(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_buy_order()))
    opp_id = _arm(engine)

    result = await _buy(engine, opp_id)

    assert result is not None
    rows = _fills(engine.db_path, BUY_ORDER_ID)
    assert len(rows) == 1
    row = rows[0]
    assert row["side"] == "BUY"
    assert row["symbol"] == "ETH/USDT"
    assert BUY_TRADE_ID in str(row["fill_ids_json"])
    assert row["executed_qty"] == pytest.approx(0.0161)
    assert row["avg_fill_price"] == pytest.approx(2691.12)
    assert row["fee_asset"] == "ETH"
    assert row["client_order_id"] == BUY_CLIENT_ID
    # Paper row, position, reservation and opportunity provenance intact.
    buy = _one(
        engine.db_path,
        "SELECT trade_id, order_id, scalp_opportunity_id FROM paper_trades WHERE side='BUY' ORDER BY id DESC LIMIT 1",
    )
    assert buy[1] == BUY_ORDER_ID
    assert buy[2] == opp_id
    assert row["mystic_trade_id"] == buy[0]
    assert row["decision_id"] == opp_id
    pos = engine.open_positions["SCALP_V2::ETH/USDT"]
    assert pos.engine_id == "SCALP_V2" and pos.quantity > 0
    assert _one(engine.db_path, "SELECT COUNT(*) FROM day_entry_reservations WHERE status='ACTIVE'")[0] == 0


@pytest.mark.asyncio
async def test_fill_ledger_failure_keeps_position_and_reservation(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_buy_order()))
    monkeypatch.setattr(
        "backend.services.live_order_identity.record_fill",
        lambda _db, _ident: (_ for _ in ()).throw(RuntimeError("ledger down")),
    )
    opp_id = _arm(engine)

    result = await _buy(engine, opp_id)

    assert result is not None
    assert engine.open_positions["SCALP_V2::ETH/USDT"].quantity > 0
    assert _one(engine.db_path, "SELECT COUNT(*) FROM day_entry_reservations WHERE status='ACTIVE'")[0] == 0
    assert _one(engine.db_path, "SELECT COUNT(*) FROM paper_trades WHERE side='BUY'")[0] == 1


@pytest.mark.asyncio
async def test_fill_record_is_idempotent(tmp_path, monkeypatch):
    from backend.services.live_order_identity import extract_identity, record_fill

    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_buy_order()))
    opp_id = _arm(engine)
    await _buy(engine, opp_id)
    buy = _one(engine.db_path, "SELECT trade_id FROM paper_trades WHERE side='BUY' ORDER BY id DESC LIMIT 1")

    identity = extract_identity(
        _buy_order(),
        symbol="ETH/USDT",
        side="BUY",
        mystic_trade_id=buy[0],
        decision_id=opp_id,
        client_order_id=BUY_CLIENT_ID,
        fallback_qty=0.0161,
        fallback_price=2691.12,
    )
    assert record_fill(engine.db_path, identity) is True
    assert len(_fills(engine.db_path, BUY_ORDER_ID)) == 1


@pytest.mark.asyncio
async def test_post_fill_failure_path_still_writes_fill_row(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_buy_order()))
    monkeypatch.setattr(
        "backend.services.portfolio_engine.PortfolioEngine._scalp_v2_commit_buy_sync",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("commit down")),
    )
    opp_id = _arm(engine)

    assert await _buy(engine, opp_id) is None
    rows = _fills(engine.db_path, BUY_ORDER_ID)
    assert len(rows) == 1
    assert rows[0]["side"] == "BUY"
