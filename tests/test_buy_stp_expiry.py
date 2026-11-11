"""A protective sell may expire a resting buy. The buy quote follows the venue result."""

from __future__ import annotations

import inspect
import sqlite3

import pytest

from backend.services.buy_cash_commitment import committed_buy_quote
from backend.services.day_entry_reservations import create_reservation
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.strategy_order_intent import allocate_intent, order_is_working


class _Venue:
    def __init__(self, order: dict):
        self.order = order
        self.calls = 0

    async def fetch_order(self, *_args, **_kwargs):
        self.calls += 1
        return {"status": "success", "order": self.order}

    async def fetch_order_trades(self, *_args, **_kwargs):
        return {"status": "success", "trades": []}


def _engine(db: str) -> PortfolioEngine:
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = db
    engine._entry_reservations = {}
    return engine


def _arm(db: str, *, qty: float, limit_px: float) -> str:
    create_reservation(db, decision_id="buy-1", symbol="BTC/USDT", notional_usd=qty * limit_px, sleeve="SCALP_V2", ttl_sec=3600)
    return allocate_intent(
        db,
        engine_id="SCALP_V2",
        trade_id="scalp-btc",
        decision_id="buy-1",
        symbol="BTC/USDT",
        side="BUY",
        quantity=qty,
        limit_price=limit_px,
    )


@pytest.mark.asyncio
async def test_zero_fill_stp_expiry_releases_the_quote(tmp_path):
    db = str(tmp_path / "zero.db")
    cid = _arm(db, qty=1.0, limit_px=400)
    engine = _engine(db)
    engine._entry_reservations["SCALP_V2::BTC/USDT"] = {"notional": 400.0, "decision_id": "buy-1", "sleeve": "SCALP_V2"}
    engine._live_service = _Venue(
        {
            "id": "55",
            "clientOrderId": cid,
            "status": "expired",
            "filled": 0,
            "amount": 1,
            "side": "buy",
            "info": {"status": "EXPIRED_IN_MATCH", "executedQty": "0"},
        }
    )

    await engine._classify_resting_buys("BTC/USDT")

    assert engine._live_service.calls == 1
    assert committed_buy_quote(db, memory_reservations=engine._entry_reservations) == 0
    assert engine._entry_reservations == {}
    assert engine._buy_stp_expired_count == 1
    with sqlite3.connect(db) as conn:
        status = conn.execute("SELECT status FROM strategy_order_intents WHERE client_order_id=?", (cid,)).fetchone()[0]
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        buys = 0 if "paper_trades" not in tables else conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='BUY'").fetchone()[0]
    assert status == "EXPIRED"
    assert buys == 0


@pytest.mark.asyncio
async def test_partial_stp_expiry_books_the_fill_and_releases_the_rest(tmp_path):
    db = str(tmp_path / "partial.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    engine = _engine(db)
    engine._entry_reservations["SCALP_V2::BTC/USDT"] = {"notional": 100.0, "decision_id": "buy-1", "sleeve": "SCALP_V2"}
    engine._live_service = _Venue(
        {
            "id": "56",
            "clientOrderId": cid,
            "status": "expired",
            "filled": 0.25,
            "average": 100,
            "amount": 1,
            "side": "buy",
            "info": {"status": "EXPIRED_IN_MATCH", "executedQty": "0.25", "cummulativeQuoteQty": "25"},
            "trades": [{"qty": 0.25, "price": 100, "commission": 0.01, "commission_asset": "USDT", "trade_id": "fill-1"}],
        }
    )

    await engine._classify_resting_buys("BTC/USDT")

    assert float(committed_buy_quote(db, memory_reservations=engine._entry_reservations)) == pytest.approx(25.01)
    assert "SCALP_V2::BTC/USDT" not in engine._entry_reservations
    with sqlite3.connect(db) as conn:
        qty, price = conn.execute("SELECT quantity, price FROM paper_trades WHERE side='BUY'").fetchone()
        status = conn.execute("SELECT status, filled_qty FROM strategy_order_intents WHERE client_order_id=?", (cid,)).fetchone()
    assert qty == pytest.approx(0.25)
    assert price == pytest.approx(100)
    assert status[1] == pytest.approx(0.25)
    assert status[0] != "PENDING"


@pytest.mark.asyncio
async def test_open_buy_is_recovered_without_a_sell_or_restart(tmp_path):
    db = str(tmp_path / "open.db")
    cid = _arm(db, qty=1.0, limit_px=400)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_order_intents SET status='OPEN' WHERE client_order_id=?", (cid,))
    engine = _engine(db)
    engine._entry_reservations["SCALP_V2::BTC/USDT"] = {"notional": 400.0, "decision_id": "buy-1", "sleeve": "SCALP_V2"}
    engine._live_service = _Venue(
        {
            "id": "77",
            "clientOrderId": cid,
            "status": "expired",
            "filled": 0,
            "amount": 1,
            "side": "buy",
            "info": {"status": "EXPIRED", "executedQty": "0"},
        }
    )

    await engine._recover_strategy_order_intents()

    assert engine._live_service.calls == 1
    assert committed_buy_quote(db, memory_reservations=engine._entry_reservations) == 0
    with sqlite3.connect(db) as conn:
        status = conn.execute("SELECT status FROM strategy_order_intents WHERE client_order_id=?", (cid,)).fetchone()[0]
    assert status == "EXPIRED"


@pytest.mark.asyncio
async def test_probe_timeout_keeps_the_quote(tmp_path):
    db = str(tmp_path / "timeout.db")
    cid = _arm(db, qty=1.0, limit_px=400)

    class _Down:
        async def fetch_order(self, *_args, **_kwargs):
            raise TimeoutError("venue")

    engine = _engine(db)
    engine._entry_reservations["SCALP_V2::BTC/USDT"] = {"notional": 400.0, "decision_id": "buy-1", "sleeve": "SCALP_V2"}
    engine._live_service = _Down()

    await engine._recover_strategy_order_intents()

    assert float(committed_buy_quote(db, memory_reservations=engine._entry_reservations)) == 400
    with sqlite3.connect(db) as conn:
        status = conn.execute("SELECT status FROM strategy_order_intents WHERE client_order_id=?", (cid,)).fetchone()[0]
    assert status == "PENDING"


def test_unknown_fee_does_not_open_a_position(tmp_path):
    from backend.services.strategy_order_intent import materialize_buy_custody

    db = str(tmp_path / "fee.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE strategy_order_intents SET status='FEE_PENDING', filled_qty=0.25, filled_cost=25 WHERE client_order_id=?",
            (cid,),
        )
    assert materialize_buy_custody(db) == 0


@pytest.mark.asyncio
async def test_ambiguous_buy_is_reread_before_the_two_minute_loop(tmp_path, monkeypatch):
    sleeps: list[float] = []

    async def _sleep(delay):
        sleeps.append(float(delay))

    monkeypatch.setattr("backend.services.portfolio_engine.asyncio.sleep", _sleep)
    db = str(tmp_path / "fast.db")
    cid = _arm(db, qty=1.0, limit_px=400)
    engine = _engine(db)
    engine._entry_reservations["SCALP_V2::BTC/USDT"] = {"notional": 400.0, "decision_id": "buy-1", "sleeve": "SCALP_V2"}
    calls = {"n": 0}

    class _Later(_Venue):
        async def fetch_order(self, *_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                return {"status": "error", "message": "timeout"}
            return {"status": "success", "order": self.order}

    engine._live_service = _Later(
        {
            "id": "90",
            "clientOrderId": cid,
            "status": "expired",
            "filled": 0,
            "amount": 1,
            "side": "buy",
            "info": {"status": "EXPIRED", "executedQty": "0"},
        }
    )

    await engine._fast_recover_ambiguous_buy(cid)

    assert sleeps[:2] == [8.0, 8.0]
    assert max(sleeps) < 120
    assert committed_buy_quote(db, memory_reservations=engine._entry_reservations) == 0


def test_post_fill_range_ignores_a_gap_and_prices_before_the_fill(tmp_path, monkeypatch):
    from backend.services.day_order_flow_tape import TapePrint

    db = str(tmp_path / "range.db")
    engine = _engine(db)
    engine.open_positions = {}
    engine._symbol_lots = lambda _symbol: []
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE portfolio_engine_positions (trade_id TEXT, highest_price REAL, lowest_price REAL)")
        conn.execute("INSERT INTO portfolio_engine_positions VALUES ('scalp-btc', 100, 100)")
    seen: dict[str, float] = {}

    def _tape(_symbol, *, since_ts, until_ts=None):
        seen["since"] = since_ts
        return [
            TapePrint(trade_ts=since_ts + 1, price=97, qty=1, is_buyer_maker=True),
            TapePrint(trade_ts=since_ts + 3, price=103, qty=1, is_buyer_maker=False),
        ]

    monkeypatch.setattr("backend.services.day_order_flow_tape.read_tape", _tape)
    engine._apply_observed_post_fill_range("BTC/USDT", "scalp-btc", 1_000.0, 100.0)
    with sqlite3.connect(db) as conn:
        high, low = conn.execute("SELECT highest_price, lowest_price FROM portfolio_engine_positions").fetchone()
    assert seen["since"] == 1_000.0
    assert high == 103
    assert low == 97

    def _gap(_symbol, *, since_ts, until_ts=None):
        return [TapePrint(trade_ts=since_ts + 30, price=80, qty=1, is_buyer_maker=True)]

    monkeypatch.setattr("backend.services.day_order_flow_tape.read_tape", _gap)
    engine._apply_observed_post_fill_range("BTC/USDT", "scalp-btc", 1_000.0, 100.0)
    with sqlite3.connect(db) as conn:
        high, low = conn.execute("SELECT highest_price, lowest_price FROM portfolio_engine_positions").fetchone()
    assert (high, low) == (103, 97)


def test_raw_stp_status_is_terminal_and_sell_classifies_afterward():
    assert order_is_working({"status": "EXPIRED_IN_MATCH", "filled": 0}) is False
    assert order_is_working({"status": "expired", "info": {"status": "EXPIRED_IN_MATCH"}}) is False
    source = inspect.getsource(PortfolioEngine.execute_sell_fifo)
    assert source.index("_execute_sell_fifo_locked") < source.index("_classify_resting_buys")
