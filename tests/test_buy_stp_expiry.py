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


@pytest.mark.asyncio
async def test_partial_buy_refresh_adds_only_the_new_fill_and_keeps_the_low(tmp_path):
    from types import SimpleNamespace

    db = str(tmp_path / "grow.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET status='PARTIAL', filled_qty=0.25, filled_gross=0.25, filled_cost=25, filled_cash=25, filled_fee=0
            WHERE client_order_id=?
            """,
            (cid,),
        )
        conn.execute(
            """
            CREATE TABLE portfolio_engine_positions (
                engine_id TEXT, trade_id TEXT, quantity REAL, original_position_cost REAL,
                highest_price REAL, lowest_price REAL, entry_price REAL, entry_fee REAL
            )
            """
        )
        conn.execute("INSERT INTO portfolio_engine_positions VALUES ('SCALP_V2', 'scalp-btc', 0.25, 25, 110, 90, 100, 0)")
    from backend.database_schema import create_paper_trades_table

    create_paper_trades_table(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (trade_id, paper_run_id, symbol, side, quantity, price, remaining_position, timestamp)
            VALUES ('scalp-btc', 'test', 'BTC/USDT', 'BUY', 0.25, 100, 0.25, '2026-11-11T00:00:00+00:00')
            """
        )
    engine = _engine(db)
    lot = SimpleNamespace(
        trade_id="scalp-btc",
        engine_id="SCALP_V2",
        symbol="BTC/USDT",
        quantity=0.25,
        entry_price=100.0,
        entry_fee=0.0,
        original_position_cost=25.0,
        highest_price=110.0,
        lowest_price=90.0,
    )
    engine.open_positions = {"SCALP_V2::BTC/USDT": lot}
    engine._live_service = _Venue(
        {
            "id": "91",
            "clientOrderId": cid,
            "status": "open",
            "filled": 0.55,
            "average": 97.8,
            "amount": 1,
            "side": "buy",
            "info": {"status": "PARTIALLY_FILLED", "executedQty": "0.55"},
            "trades": [{"qty": 0.30, "price": 96, "commission": 0, "commission_asset": "USDT", "trade_id": "leg-2"}],
        }
    )

    await engine._refresh_partial_buy_before_sell(lot)

    assert lot.quantity == pytest.approx(0.55)
    assert lot.lowest_price == 90
    assert lot.highest_price == 110
    await engine._refresh_partial_buy_before_sell(lot)
    assert lot.quantity == pytest.approx(0.55)


def _position_table(db: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_engine_positions (
                symbol TEXT, quantity REAL, entry_price REAL, entry_time REAL, trade_id TEXT,
                stop_price REAL, take_profit_1_price REAL, take_profit_2_price REAL,
                highest_price REAL, lowest_price REAL, atr_at_entry REAL,
                entry_bar_timestamp INTEGER, confidence_at_entry REAL, last_updated TEXT,
                engine_id TEXT, entry_fee REAL, original_position_cost REAL, status TEXT
            )
            """
        )


def _paper_buy(db: str, *, remaining: float, qty: float = 0.25, price: float = 100.0) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (
                trade_id, paper_run_id, symbol, side, quantity, price, remaining_position, timestamp
            ) VALUES ('scalp-btc', 'test', 'BTC/USDT', 'BUY', ?, ?, ?, '2026-11-11T00:00:00+00:00')
            """,
            (qty, price, remaining),
        )


def _lot(db: str, *, engine: str, trade: str, qty: float, cost: float, high: float, low: float, entry: float, fee: float = 0.0) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO portfolio_engine_positions (
                symbol, engine_id, trade_id, quantity, original_position_cost,
                highest_price, lowest_price, entry_price, entry_fee, entry_time
            ) VALUES ('BTC/USDT', ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (engine, trade, qty, cost, high, low, entry, fee),
        )


def _ledger(db: str, cash: float = 1000.0) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_engine_ledger (
                id INTEGER PRIMARY KEY, cash_balance REAL, positions_value REAL, total_equity REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO portfolio_engine_ledger (id, cash_balance, positions_value, total_equity) VALUES (1, ?, 0, ?)",
            (cash, cash),
        )


@pytest.mark.asyncio
async def test_later_fill_lowers_the_watermark_without_raising_an_older_one(tmp_path):
    from types import SimpleNamespace

    from backend.database_schema import create_paper_trades_table
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_EXIT_CATASTROPHIC, evaluate_scalp_v2_exit

    db = str(tmp_path / "low.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE strategy_order_intents SET status='PARTIAL', filled_qty=0.25, filled_gross=0.25, filled_cost=25, filled_cash=25, filled_fee=0 WHERE client_order_id=?",
            (cid,),
        )
    create_paper_trades_table(db)
    _position_table(db)
    _paper_buy(db, remaining=0.25)
    _lot(db, engine="SCALP_V2", trade="scalp-btc", qty=0.25, cost=25, high=100, low=100, entry=100)
    _lot(db, engine="DAY_V2", trade="day-btc", qty=0.40, cost=40, high=105, low=99, entry=100)
    engine = _engine(db)
    lot = SimpleNamespace(
        trade_id="scalp-btc",
        engine_id="SCALP_V2",
        symbol="BTC/USDT",
        quantity=0.25,
        entry_price=100.0,
        entry_fee=0.0,
        original_position_cost=25.0,
        highest_price=100.0,
        lowest_price=100.0,
    )
    day = SimpleNamespace(trade_id="day-btc", engine_id="DAY_V2", symbol="BTC/USDT", quantity=0.40, entry_price=100.0, highest_price=105.0, lowest_price=99.0)
    engine.open_positions = {"SCALP_V2::BTC/USDT": lot, "DAY_V2::BTC/USDT": day}
    engine._live_service = _Venue(
        {
            "id": "92",
            "clientOrderId": cid,
            "status": "open",
            "filled": 0.55,
            "average": 97.818181,
            "amount": 1,
            "side": "buy",
            "info": {"status": "PARTIALLY_FILLED", "executedQty": "0.55"},
            "trades": [{"qty": 0.30, "price": 96, "commission": 0, "commission_asset": "USDT", "trade_id": "leg-2"}],
        }
    )

    await engine._refresh_partial_buy_before_sell(lot)

    assert lot.quantity == pytest.approx(0.55)
    assert lot.entry_price == pytest.approx((25 + 28.8) / 0.55)
    assert lot.lowest_price == pytest.approx(96)
    assert lot.highest_price == pytest.approx(100)
    assert day.quantity == pytest.approx(0.40)
    decision = evaluate_scalp_v2_exit(position=lot, current_price=90, net_pnl_pct=-0.1, hold_minutes=1, bar_low=90, symbol="BTC/USDT")
    assert decision["action"] == "sell"
    assert decision["reason"] == SCALP_V2_EXIT_CATASTROPHIC
    with sqlite3.connect(db) as conn:
        day_qty = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE engine_id='DAY_V2'").fetchone()[0]
    assert day_qty == pytest.approx(0.40)
    await engine._refresh_partial_buy_before_sell(lot)
    assert lot.quantity == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_refresh_miss_sells_the_booked_quantity_and_reopen_books_only_the_new_base(tmp_path):
    from types import SimpleNamespace

    from backend.database_schema import create_paper_trades_table

    db = str(tmp_path / "reopen.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET status='PARTIAL', filled_qty=0.25, filled_gross=0.25, filled_cost=25, filled_cash=25,
                filled_fee=0, booked_cash=25, booked_high=110, booked_low=90, fill_ids_json='["leg-1"]'
            WHERE client_order_id=?
            """,
            (cid,),
        )
    create_paper_trades_table(db)
    _position_table(db)
    _ledger(db, 1000)
    _paper_buy(db, remaining=0)
    engine = _engine(db)
    lot = SimpleNamespace(
        trade_id="scalp-btc",
        engine_id="SCALP_V2",
        symbol="BTC/USDT",
        quantity=0.25,
        entry_price=100.0,
        entry_fee=0.0,
        original_position_cost=25.0,
        highest_price=110.0,
        lowest_price=90.0,
    )
    engine.open_positions = {}
    engine._live_service = _Venue(
        {
            "id": "93",
            "clientOrderId": cid,
            "status": "expired",
            "filled": 0.55,
            "average": 97.818181,
            "amount": 1,
            "side": "buy",
            "info": {"status": "EXPIRED_IN_MATCH", "executedQty": "0.55", "cummulativeQuoteQty": "53.8"},
            "trades": [
                {"qty": 0.25, "price": 100, "commission": 0, "commission_asset": "USDT", "trade_id": "leg-1"},
                {"qty": 0.30, "price": 96, "commission": 0, "commission_asset": "USDT", "trade_id": "leg-2"},
            ],
        }
    )

    class _Limited(_Venue):
        async def fetch_order(self, *_args, **_kwargs):
            return {"status": "error", "rate_limited": True}

    held = engine._live_service.order
    engine._live_service = _Limited(held)
    await engine._refresh_partial_buy_before_sell(lot)
    assert lot.quantity == pytest.approx(0.25)

    engine._live_service = _Venue(held)
    await engine._classify_resting_buys("BTC/USDT")

    reopened = engine.open_positions["SCALP_V2::BTC/USDT"]
    assert reopened.quantity == pytest.approx(0.30)
    assert reopened.lowest_price == pytest.approx(90)
    assert reopened.highest_price == pytest.approx(110)
    assert engine._exit_reeval_requested is True
    with sqlite3.connect(db) as conn:
        cash = conn.execute("SELECT cash_balance FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]
        rows = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
        status, filled = conn.execute("SELECT status, filled_qty FROM strategy_order_intents").fetchone()
    assert cash == pytest.approx(1000 - 28.8)
    assert rows == 1
    assert filled == pytest.approx(0.55)
    assert status == "FILLED"
    await engine._classify_resting_buys("BTC/USDT")
    assert engine.open_positions["SCALP_V2::BTC/USDT"].quantity == pytest.approx(0.30)
    with sqlite3.connect(db) as conn:
        cash_again = conn.execute("SELECT cash_balance FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]
    assert cash_again == pytest.approx(cash)


@pytest.mark.asyncio
async def test_incomplete_mytrades_does_not_guess_the_unseen_base(tmp_path):
    from types import SimpleNamespace

    from backend.database_schema import create_paper_trades_table

    db = str(tmp_path / "gap.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET status='PARTIAL', filled_qty=0.25, filled_gross=0.25, filled_cost=25, filled_cash=25,
                filled_fee=0, fill_ids_json='["leg-1"]'
            WHERE client_order_id=?
            """,
            (cid,),
        )
    create_paper_trades_table(db)
    _position_table(db)
    _paper_buy(db, remaining=0.25)
    _lot(db, engine="SCALP_V2", trade="scalp-btc", qty=0.25, cost=25, high=100, low=100, entry=100)
    engine = _engine(db)
    lot = SimpleNamespace(
        trade_id="scalp-btc",
        engine_id="SCALP_V2",
        symbol="BTC/USDT",
        quantity=0.25,
        entry_price=100.0,
        entry_fee=0.0,
        original_position_cost=25.0,
        highest_price=100.0,
        lowest_price=100.0,
    )
    engine.open_positions = {"SCALP_V2::BTC/USDT": lot}
    engine._live_service = _Venue(
        {
            "id": "94",
            "clientOrderId": cid,
            "status": "open",
            "filled": 0.55,
            "average": 97.8,
            "amount": 1,
            "side": "buy",
            "info": {"status": "PARTIALLY_FILLED", "executedQty": "0.55"},
            "trades": [{"qty": 0.25, "price": 100, "commission": 0, "commission_asset": "USDT", "trade_id": "leg-1"}],
        }
    )

    await engine._refresh_partial_buy_before_sell(lot)

    assert lot.quantity == pytest.approx(0.25)
    with sqlite3.connect(db) as conn:
        filled = conn.execute("SELECT filled_qty FROM strategy_order_intents").fetchone()[0]
    assert filled == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_base_fee_on_the_second_leg_reduces_only_that_leg(tmp_path):
    from types import SimpleNamespace

    from backend.database_schema import create_paper_trades_table

    db = str(tmp_path / "fee.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET status='PARTIAL', filled_qty=0.25, filled_gross=0.25, filled_cost=25, filled_cash=25.01, filled_fee=0.01
            WHERE client_order_id=?
            """,
            (cid,),
        )
    create_paper_trades_table(db)
    _position_table(db)
    _paper_buy(db, remaining=0.25)
    _lot(db, engine="SCALP_V2", trade="scalp-btc", qty=0.25, cost=25.01, high=110, low=90, entry=100, fee=0.01)
    engine = _engine(db)
    lot = SimpleNamespace(
        trade_id="scalp-btc",
        engine_id="SCALP_V2",
        symbol="BTC/USDT",
        quantity=0.25,
        entry_price=100.0,
        entry_fee=0.01,
        original_position_cost=25.01,
        highest_price=110.0,
        lowest_price=90.0,
    )
    engine.open_positions = {"SCALP_V2::BTC/USDT": lot}
    engine._live_service = _Venue(
        {
            "id": "95",
            "clientOrderId": cid,
            "status": "open",
            "filled": 0.55,
            "amount": 1,
            "side": "buy",
            "info": {"status": "PARTIALLY_FILLED", "executedQty": "0.55"},
            "trades": [{"qty": 0.30, "price": 96, "commission": 0.001, "commission_asset": "BTC", "trade_id": "leg-2"}],
        }
    )

    await engine._refresh_partial_buy_before_sell(lot)

    assert lot.quantity == pytest.approx(0.25 + 0.30 - 0.001)
    assert lot.lowest_price == pytest.approx(90)
    assert lot.original_position_cost == pytest.approx(25.01 + 28.8)
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    assert count == 1


@pytest.mark.asyncio
async def test_open_zero_fill_is_not_read_before_a_sell_and_partial_exposes_unbooked_qty(tmp_path):
    from types import SimpleNamespace

    from backend.database_schema import create_paper_trades_table
    from backend.services.buy_cash_commitment import buy_cash_view

    db = str(tmp_path / "obs.db")
    cid = _arm(db, qty=1.0, limit_px=100)
    create_paper_trades_table(db)
    _position_table(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_order_intents SET status='OPEN', filled_qty=0 WHERE client_order_id=?", (cid,))
    engine = _engine(db)

    class _Counting:
        def __init__(self):
            self.calls = 0

        async def fetch_order(self, *_args, **_kwargs):
            self.calls += 1
            return {"status": "error", "rate_limited": True}

    engine._live_service = _Counting()
    lot = SimpleNamespace(
        trade_id="scalp-btc",
        engine_id="SCALP_V2",
        symbol="BTC/USDT",
        quantity=0,
        entry_price=0,
        highest_price=0,
        lowest_price=0,
        original_position_cost=0,
        entry_fee=0,
    )
    await engine._refresh_partial_buy_before_sell(lot)
    assert engine._live_service.calls == 0
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            UPDATE strategy_order_intents
            SET status='PARTIAL', filled_qty=0.55, filled_gross=0.55, filled_cost=53.8, filled_cash=53.8, quantity=1
            WHERE client_order_id=?
            """,
            (cid,),
        )
    _paper_buy(db, remaining=0.25)
    _lot(db, engine="SCALP_V2", trade="scalp-btc", qty=0.25, cost=0, high=0, low=0, entry=100)
    row = buy_cash_view(db).working_buys[0]
    assert row["unbooked_fill_qty"] == pytest.approx(0.30)
    assert row["venue_cumulative_filled"] == pytest.approx(0.55)
    assert row["locally_booked_filled"] == pytest.approx(0.25)
    assert row["position_qty"] == pytest.approx(0.25)
    assert row["working_remainder"] == pytest.approx(0.45)


def test_sell_budget_is_read_after_the_partial_refresh():
    source = inspect.getsource(PortfolioEngine._execute_sell_fifo_locked)
    assert source.index("_refresh_partial_buy_before_sell") < source.index("_venue_sell_budget.append")


def test_raw_stp_status_is_terminal_and_sell_classifies_afterward():
    assert order_is_working({"status": "EXPIRED_IN_MATCH", "filled": 0}) is False
    assert order_is_working({"status": "expired", "info": {"status": "EXPIRED_IN_MATCH"}}) is False
    source = inspect.getsource(PortfolioEngine.execute_sell_fifo)
    assert source.index("_execute_sell_fifo_locked") < source.index("_classify_resting_buys")
