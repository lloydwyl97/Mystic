"""Protected inventory follows the exchange balance; DAY V2 winner protection never exits below break-even."""

from __future__ import annotations

import sqlite3
import time

import pytest

from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit
from backend.services.protected_external_inventory import ensure_schema, record_protected, shrink_to_exchange
from tests.test_scalp_v2_live_fill_ownership import _one, _reconcile_engine


def _conn_with(rows: list[tuple[str, float, float]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    for sym, qty, px in rows:
        record_protected(conn, sym, qty, px, source_trade_id="reconcile_import_x")
    return conn


def _qty(conn: sqlite3.Connection, sym: str):
    row = conn.execute("SELECT quantity FROM protected_external_inventory WHERE symbol=?", (sym,)).fetchone()
    return None if row is None else row[0]


def test_sold_balance_deletes_protected_row():
    conn = _conn_with([("XRP/USDT", 44.98088, 2.8)])
    changed = shrink_to_exchange(conn, {}, {})
    assert changed == [("XRP/USDT", 44.98088, 0.0)]
    assert _qty(conn, "XRP/USDT") is None


def test_dust_remainder_deletes_protected_row():
    conn = _conn_with([("BTC/USDT", 0.00031982, 84000.0)])
    shrink_to_exchange(conn, {"BTC": 0.000009}, {})
    assert _qty(conn, "BTC/USDT") is None


def test_partial_sale_caps_quantity():
    conn = _conn_with([("SOL/USDT", 0.6147182, 200.0)])
    shrink_to_exchange(conn, {"SOL": 0.3}, {})
    assert _qty(conn, "SOL/USDT") == pytest.approx(0.3)


def test_strategy_lot_is_not_counted_as_protected():
    conn = _conn_with([("XRP/USDT", 44.98088, 2.8)])
    shrink_to_exchange(conn, {"XRP": 20.0}, {"XRP/USDT": 20.0})
    assert _qty(conn, "XRP/USDT") is None


def test_unchanged_balance_leaves_row_alone():
    conn = _conn_with([("ETH/USDT", 0.00969512, 2600.0)])
    assert shrink_to_exchange(conn, {"ETH": 0.00969512}, {}) == []
    assert _qty(conn, "ETH/USDT") == pytest.approx(0.00969512)


def test_protected_row_never_grows():
    conn = _conn_with([("ETH/USDT", 0.005, 2600.0)])
    shrink_to_exchange(conn, {"ETH": 0.02}, {})
    assert _qty(conn, "ETH/USDT") == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_reconcile_drops_stale_row_so_new_scalp_lot_is_not_vanished(tmp_path, monkeypatch):
    from backend.services.portfolio_engine import OpenPosition

    engine = _reconcile_engine(tmp_path, monkeypatch)
    conn = sqlite3.connect(engine.db_path)
    record_protected(conn, "ETH/USDT", 0.02, 2700.0, source_trade_id="reconcile_import_old")
    conn.commit()
    conn.close()
    engine.open_positions["ETH/USDT"] = OpenPosition(
        symbol="ETH/USDT",
        quantity=0.0095,
        entry_price=2700.0,
        entry_time=time.time(),
        trade_id="scalp_v2_ETHUSDT_2",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        engine_id="SCALP_V2",
    )

    await engine._import_missing_exchange_positions({"ETH": 0.0095}, {"ETH": 0.0095})

    assert _one(engine.db_path, "SELECT COUNT(*) FROM protected_external_inventory")[0] == 0
    assert engine.open_positions["ETH/USDT"].quantity == pytest.approx(0.0095)


def _capture_dispatch(monkeypatch):
    from unittest.mock import MagicMock

    sent = MagicMock()
    monkeypatch.setattr("backend.services.protected_external_inventory._dispatch_unmatched_alert", sent)
    return sent


@pytest.mark.asyncio
async def test_unmatched_balance_alerts_once_not_every_reconcile(tmp_path, monkeypatch):
    sent = _capture_dispatch(monkeypatch)
    engine = _reconcile_engine(tmp_path, monkeypatch)

    await engine._import_missing_exchange_positions({"ETH": 0.02}, {"ETH": 0.02})
    await engine._import_missing_exchange_positions({"ETH": 0.02}, {"ETH": 0.02})

    assert sent.call_count == 1
    assert sent.call_args.args[0] == "ETH/USDT"


@pytest.mark.asyncio
async def test_restored_mystic_fill_does_not_alert(tmp_path, monkeypatch):
    from tests.test_scalp_v2_live_fill_ownership import _seed_buy

    sent = _capture_dispatch(monkeypatch)
    engine = _reconcile_engine(tmp_path, monkeypatch)
    _seed_buy(engine.db_path, remaining=0.0094981)

    await engine._import_missing_exchange_positions({"ETH": 0.0094981}, {"ETH": 0.0094981})

    assert sent.call_count == 0


@pytest.mark.asyncio
async def test_alert_failure_does_not_break_reconcile(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "backend.services.protected_external_inventory._dispatch_unmatched_alert",
        MagicMock(side_effect=RuntimeError("down")),
    )
    engine = _reconcile_engine(tmp_path, monkeypatch)

    await engine._import_missing_exchange_positions({"ETH": 0.02}, {"ETH": 0.02})

    assert _one(engine.db_path, "SELECT quantity FROM protected_external_inventory WHERE symbol='ETH/USDT'")[0] == pytest.approx(0.02)


def _protected_conn() -> sqlite3.Connection:
    from backend.services.protected_external_inventory import ensure_alert_state

    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_alert_state(conn)
    return conn


def test_reclassify_path_alerts_once(monkeypatch):
    from backend.services.protected_external_inventory import (
        handle_unmatched_balance,
        reclassify_unowned_imports,
    )

    sent = _capture_dispatch(monkeypatch)
    conn = _protected_conn()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS portfolio_engine_positions
           (symbol TEXT, quantity REAL, entry_price REAL, trade_id TEXT)"""
    )
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('ETH/USDT', 0.02, 2700.0, 'reconcile_import_ETH_USDT_1')")
    conn.commit()

    assert reclassify_unowned_imports(conn) == ["ETH/USDT"]
    assert sent.call_count == 1
    assert sent.call_args.args[0] == "ETH/USDT"
    # Same generation again: no repeat.
    assert handle_unmatched_balance(conn, "ETH/USDT", 0.02, 2700.0, source="reconcile_import") is False
    assert sent.call_count == 1


def test_balance_clears_then_reappears_alerts_again(monkeypatch):
    from backend.services.protected_external_inventory import handle_unmatched_balance, record_protected, shrink_to_exchange

    sent = _capture_dispatch(monkeypatch)
    conn = _conn_with([("ETH/USDT", 0.02, 2700.0)])

    assert handle_unmatched_balance(conn, "ETH/USDT", 0.02, 2700.0, source="reconcile_import") is True
    assert sent.call_count == 1
    # Balance sold: shrink deletes the row and resets alert state.
    assert shrink_to_exchange(conn, {}, {}) == [("ETH/USDT", 0.02, 0.0)]
    # Same balance returns later: alerts again.
    record_protected(conn, "ETH/USDT", 0.02, 2700.0, source_trade_id="reconcile_import_new")
    assert handle_unmatched_balance(conn, "ETH/USDT", 0.02, 2700.0, source="reconcile_import") is True
    assert sent.call_count == 2


def test_materially_increased_qty_alerts_again(monkeypatch):
    from backend.services.protected_external_inventory import handle_unmatched_balance

    sent = _capture_dispatch(monkeypatch)
    conn = _protected_conn()

    assert handle_unmatched_balance(conn, "ETH/USDT", 0.02, 2700.0, source="reconcile_import") is True
    assert handle_unmatched_balance(conn, "ETH/USDT", 0.02, 2700.0, source="reconcile_import") is False
    assert handle_unmatched_balance(conn, "ETH/USDT", 0.04, 2700.0, source="reconcile_import") is True
    assert sent.call_count == 2


def test_dust_qty_does_not_alert(monkeypatch):
    from backend.services.protected_external_inventory import handle_unmatched_balance

    sent = _capture_dispatch(monkeypatch)
    conn = _protected_conn()

    assert handle_unmatched_balance(conn, "ETH/USDT", 0.0, 2700.0, source="reconcile_import") is False
    assert sent.call_count == 0


@pytest.mark.asyncio
async def test_unmatched_balance_is_never_auto_sold(tmp_path, monkeypatch):
    _capture_dispatch(monkeypatch)
    engine = _reconcile_engine(tmp_path, monkeypatch)

    await engine._import_missing_exchange_positions({"ETH": 0.02}, {"ETH": 0.02})

    assert "ETH/USDT" not in engine.open_positions
    assert _one(engine.db_path, "SELECT COUNT(*) FROM portfolio_engine_positions")[0] == 0
    assert _one(engine.db_path, "SELECT quantity FROM protected_external_inventory WHERE symbol='ETH/USDT'")[0] == pytest.approx(0.02)


def _exit(highest: float, current: float, atr: float = 1.2, cost: float = 0.0006):
    return evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=current,
        bar_low=current,
        highest_price=highest,
        atr_at_entry=atr,
        structural_anchor=0.0,
        target_price=0.0,
        entry_time=time.time() - 3600,
        estimated_roundtrip_cost=cost,
    )


def test_winner_floor_does_not_substitute_break_even():
    # Pre-runner position: 1h ATR = 2.5 x 1.2 = 3.0. At MFE 3.1 the wide trail
    # is 103.1 - 4.5 = 98.6, below break-even. That must not sell.
    assert _exit(highest=103.1, current=100.05) is None
    assert _exit(highest=105.0, current=102.0) is None


def test_winner_floor_holds_above_break_even():
    assert _exit(highest=103.1, current=100.2) is None


def test_small_mfe_is_not_sold_by_a_scalp_trail():
    assert _exit(highest=100.9, current=100.05) is None


def test_wide_trail_on_big_winner_does_not_sell():
    # A runner stop is state. It does not sell unless continuation is worse.
    assert _exit(highest=110.0, current=108.0) is None
    assert _exit(highest=110.0, current=107.5) is None


def test_below_mfe_gate_floor_does_not_apply():
    assert _exit(highest=100.5, current=99.9) is None
