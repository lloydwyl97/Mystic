"""The same venue fills reconstruct one canonical position from every crash window."""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.services.live_fill_economics import canonical_buy_economics, plan_sell_quantity
from backend.services.live_order_identity import OrderIdentity, record_fill
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.protected_limit_execution import lost_ack_may_submit_replacement, summarize_day_candidate_batch_ms
from backend.services.strategy_order_intent import adopt_venue_fill, allocate_intent, apply_venue_order, mark_exchange_order

LEGS = (
    {"id": "leg-1", "qty": 0.004, "price": 100.0, "commission": 0.0000008, "commissionAsset": "BTC"},
    {"id": "leg-2", "qty": 0.006, "price": 110.0, "commission": 0.0000012, "commissionAsset": "BTC"},
)
GROSS = 0.01
BASE_FEE = 0.000002
NET = 0.009998
VWAP = 106.0
CASH = 1.06


def _books(db: str, trade: str) -> tuple[float, float, float]:
    conn = sqlite3.connect(db)
    try:
        paper = conn.execute("SELECT quantity, price FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY'", (trade,)).fetchone()
        intent = conn.execute("SELECT filled_qty, filled_cost, filled_gross, filled_fee, filled_cash FROM strategy_order_intents WHERE trade_id=?", (trade,)).fetchone()
    finally:
        conn.close()
    assert paper is not None and intent is not None
    return paper, intent


def _assert_canonical(db: str, trade: str) -> None:
    paper, intent = _books(db, trade)
    assert paper[0] == pytest.approx(NET)
    assert paper[1] == pytest.approx(VWAP)
    assert intent[0] == pytest.approx(NET)
    assert intent[2] == pytest.approx(GROSS)
    assert intent[1] / intent[2] == pytest.approx(VWAP)
    assert intent[4] == pytest.approx(CASH)
    expected = canonical_buy_economics(gross_qty=GROSS, price=VWAP, base_fee=BASE_FEE)
    assert paper[0] == pytest.approx(expected["net_qty"])
    assert intent[3] == pytest.approx(expected["fee_usd"])


def test_multi_fill_base_commission_is_exactly_once(tmp_path) -> None:
    db = str(tmp_path / "legs.db")
    cid = allocate_intent(db, engine_id="DAY_V2", trade_id="t", decision_id="d", symbol="BTC/USDT", side="BUY", quantity=GROSS, limit_price=VWAP)
    first = apply_venue_order(db, {"id": "ord", "clientOrderId": cid, "trades": [LEGS[0]]})
    again = apply_venue_order(db, {"id": "ord", "clientOrderId": cid, "trades": [LEGS[0]]})
    rest = apply_venue_order(db, {"id": "ord", "clientOrderId": cid, "trades": [LEGS[1]]})
    assert first["adopted"] is True
    assert again["adopted"] is False
    assert rest["filled_qty"] == pytest.approx(NET)
    assert rest["gross_qty"] == pytest.approx(GROSS)
    assert rest["vwap"] == pytest.approx(VWAP)
    _assert_canonical(db, "t")


def test_every_recovery_window_books_the_same_position(tmp_path) -> None:
    def _fresh(name: str) -> tuple[str, str]:
        db = str(tmp_path / f"{name}.db")
        cid = allocate_intent(db, engine_id="DAY_V2", trade_id="t", decision_id="d", symbol="BTC/USDT", side="BUY", quantity=GROSS, limit_price=VWAP, setup="RANGE_BOUNCE", feature_version="5")
        return db, cid

    ack_db, ack_cid = _fresh("ack")
    mark_exchange_order(ack_db, ack_cid, "ord", status="ACKED")
    assert apply_venue_order(ack_db, {"id": "ord", "clientOrderId": ack_cid, "status": "NEW"})["reason"] == "NO_FILL"
    conn = sqlite3.connect(ack_db)
    try:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='paper_trades'").fetchone() is None or conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    finally:
        conn.close()

    snap_db, snap_cid = _fresh("snap")
    refused = apply_venue_order(snap_db, {"id": "ord", "clientOrderId": snap_cid, "filled": GROSS, "average": VWAP})
    assert refused["reason"] == "NEEDS_TRADES"
    apply_venue_order(snap_db, {"id": "ord", "clientOrderId": snap_cid, "trades": list(LEGS)})
    _assert_canonical(snap_db, "t")

    summary_db, summary_cid = _fresh("summary")
    apply_venue_order(
        summary_db,
        {"id": "ord", "clientOrderId": summary_cid, "filled": GROSS, "average": VWAP, "fee": {"cost": BASE_FEE, "currency": "BTC"}},
    )
    _assert_canonical(summary_db, "t")

    quote_db, quote_cid = _fresh("quote")
    quoted = adopt_venue_fill(quote_db, client_order_id=quote_cid, exchange_order_id="ord", fill_id="q1", quantity=1.0, price=100.0, fee=0.25, fee_asset="USDT")
    assert quoted["filled_qty"] == pytest.approx(1.0)
    assert quoted["cash_debit"] == pytest.approx(100.25)

    other_db, other_cid = _fresh("other")
    other = adopt_venue_fill(other_db, client_order_id=other_cid, exchange_order_id="ord", fill_id="o1", quantity=1.0, price=100.0, fee=0.01, fee_asset="BNB")
    assert other["filled_qty"] == pytest.approx(1.0)
    assert other["cash_debit"] == pytest.approx(100.0)

    ledger_db, ledger_cid = _fresh("ledger")
    mark_exchange_order(ledger_db, ledger_cid, "ord", status="ACKED")
    assert record_fill(
        ledger_db,
        OrderIdentity(
            symbol="BTC/USDT",
            side="BUY",
            exchange_order_id="ord",
            client_order_id=ledger_cid,
            executed_qty=GROSS,
            avg_fill_price=VWAP,
            mystic_trade_id="t",
            venue_trade_ids=["leg-1"],
            fee_amount=BASE_FEE,
            fee_asset="BTC",
            fee_items=[{"amount": 0.0000008, "asset": "BTC"}, {"amount": 0.0000012, "asset": "BTC"}],
            fee_from_exchange=True,
        ),
    )
    engine = PortfolioEngine(db_path=ledger_db, principal=1000.0, test_mode=True)
    engine._ensure_db_schema()
    import asyncio

    assert asyncio.run(engine.restore_filled_asset_custody()) == 1
    _assert_canonical(ledger_db, "t")
    conn = sqlite3.connect(ledger_db)
    try:
        position = conn.execute("SELECT quantity, entry_price, entry_fee FROM portfolio_engine_positions WHERE trade_id='t'").fetchone()
    finally:
        conn.close()
    assert position[0] == pytest.approx(NET)
    assert position[1] == pytest.approx(VWAP)
    assert asyncio.run(engine.restore_filled_asset_custody()) == 0


def test_recovered_net_quantity_cannot_oversell(tmp_path) -> None:
    steps = {"BTC": 0.00001, "ETH": 0.0001, "SOL": 0.01, "XRP": 0.1}
    samples = {
        "BTC": (0.00154, 3.1e-7),
        "ETH": (0.1084, 2.168e-5),
        "SOL": (0.521, 0.0001042),
        "XRP": (91.2, 0.01824),
    }
    for asset, (gross, fee) in samples.items():
        net = canonical_buy_economics(gross_qty=gross, price=100.0, base_fee=fee)["net_qty"]
        assert net < gross
        held = plan_sell_quantity(net_active_qty=net, exchange_free_qty=net, protected_dust_qty=0, qty_step=steps[asset])
        mistaken = plan_sell_quantity(net_active_qty=gross, exchange_free_qty=gross, protected_dust_qty=0, qty_step=steps[asset])
        assert float(held.sellable) <= net
        # SOL's 0.01 step is coarser than its 2 bps fee, so the floored gross
        # size is still inside the net balance. The other three are not.
        if asset == "SOL":
            assert float(mistaken.sellable) <= net
        else:
            assert float(mistaken.sellable) > net


def test_lost_ack_submits_again_only_when_the_venue_proves_absence() -> None:
    assert lost_ack_may_submit_replacement({"status": "error", "code": "-2013"}) is True
    assert lost_ack_may_submit_replacement({"status": "success", "order": {"id": "1"}}) is False
    assert lost_ack_may_submit_replacement({"status": "error", "code": "-1000"}) is False
    assert lost_ack_may_submit_replacement(None) is False


def test_day_batch_timing_is_not_stored_until_a_batch_is_stamped(tmp_path) -> None:
    db = str(tmp_path / "day.db")
    empty = summarize_day_candidate_batch_ms(db)
    assert empty["source"] == "not_stored"
    assert empty["samples"] == 0
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE adaptive_candidate_markouts (
            id INTEGER PRIMARY KEY,
            engine_id TEXT,
            economic_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO adaptive_candidate_markouts (engine_id, economic_json) VALUES ('DAY_V2', ?)",
        (json.dumps({"_timing": {"eval_batch_start": 1000.0, "eval_batch_end": 1001.25, "signal_observed": 1000.1}}),),
    )
    conn.commit()
    conn.close()
    measured = summarize_day_candidate_batch_ms(db)
    assert measured["source"] == "adaptive_candidate_markouts.timing"
    assert measured["samples"] == 1
    assert measured["avg"] == pytest.approx(1250.0)
