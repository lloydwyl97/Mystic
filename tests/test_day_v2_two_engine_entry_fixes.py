"""DAY V2 entry fixes found on Ocean 2026-09-28: sibling-engine skip, own-reservation
double count, direct-fill frequency counting, and buy-row symbol format."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend.services.day_v2.frequency_guard import check_frequency_limit
from backend.services.portfolio_engine_integration import day_v2_held_symbols
from backend.services.two_engine_capital import check_engine_budget


def _lot(engine_id: str, status: str = "ACTIVE"):
    return SimpleNamespace(engine_id=engine_id, status=status, quantity=0.3, entry_price=120.0, original_position_cost=36.0)


def test_scalp_lot_does_not_block_day_symbol():
    positions = {
        "SCALP_V2::SOL/USDT": _lot("SCALP_V2"),
        "DAY_V2::BTC/USDT": _lot("DAY_V2"),
        "ETH/USDT": _lot("DAY_V2"),
    }
    assert day_v2_held_symbols(positions) == {"BTCUSDT", "ETHUSDT"}


def _reservation_db(tmp_path) -> str:
    db = str(tmp_path / "res.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE day_entry_reservations (reservation_id TEXT, sleeve TEXT, notional_usd REAL, expires_at REAL, status TEXT)")
    conn.execute("INSERT INTO day_entry_reservations VALUES ('own', 'DAY_V2', 80.0, ?, 'ACTIVE')", (time.time() + 3600,))
    conn.commit()
    conn.close()
    return db


def test_budget_excludes_callers_own_reservation(tmp_path, monkeypatch):
    import backend.services.two_engine_capital as tec

    monkeypatch.setattr(tec, "get_capital_shares", lambda: (0.5, 0.5))
    db = _reservation_db(tmp_path)
    ok, reason, _ = check_engine_budget(db, "DAY_V2", 80.0, 300.0, 300.0, {})
    assert (ok, reason) == (False, "ENGINE_BUDGET_EXCEEDED")
    ok, reason, snap = check_engine_budget(db, "DAY_V2", 80.0, 300.0, 300.0, {}, exclude_reservation_id="own")
    assert (ok, reason) == (True, "")
    assert snap.day.committed_reservations == 0.0


def _trades_db(tmp_path, rows) -> str:
    db = str(tmp_path / "freq.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE paper_trades (engine_id TEXT, side TEXT, status TEXT, is_synthetic INTEGER, symbol TEXT, timestamp TEXT)")
    conn.executemany("INSERT INTO paper_trades VALUES (?, ?, 'executed', 0, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def test_direct_fills_count_toward_frequency_cap(tmp_path):
    db = _trades_db(tmp_path, [("DAY_V2", "BUY", "SOLUSDT", _ago(1)), ("DAY_V2", "BUY", "SOL/USDT", _ago(3))])
    ok, reason = check_frequency_limit(db, "SOLUSDT")
    assert not ok and "SYMBOL:SOLUSDT:2/2" in reason


def test_frequency_cap_ignores_scalp_old_and_sell_rows(tmp_path):
    db = _trades_db(
        tmp_path,
        [
            ("SCALP_V2", "BUY", "SOL/USDT", _ago(1)),
            ("DAY_V2", "BUY", "SOL/USDT", _ago(30)),
            ("DAY_V2", "SELL", "SOL/USDT", _ago(1)),
            ("DAY_V2", "BUY", "SOL/USDT", _ago(1)),
        ],
    )
    assert check_frequency_limit(db, "SOLUSDT") == (True, "OK")


@pytest.mark.asyncio
async def test_direct_entry_passes_own_reservation_to_budget(monkeypatch):
    import backend.services.day_v2.live_entry as le
    import backend.services.two_engine_capital as tec

    seen = {}

    def _budget(*_a, **k):
        seen.update(k)
        return False, "ENGINE_BUDGET_EXCEEDED", None

    async def _safe(*_a, **_k):
        return True, ""

    monkeypatch.setattr(le, "DAY_V2_ENABLED", True)
    monkeypatch.setattr(tec, "check_engine_budget", _budget)
    monkeypatch.setattr("backend.services.day_trailing_buy._pre_submit_safety", _safe)
    engine = SimpleNamespace(db_path="x.db", _total_equity=300.0, _available_balance=300.0, open_positions={}, last_buy_outcome="", last_buy_reject_reason="")
    signal = SimpleNamespace(symbol="SOLUSDT", atr=1.0, signal_bar_ts=0, structural_anchor=119.0, setup="S", opportunity_id="o")
    assert await le.submit_day_v2_direct_entry(engine, signal=signal, ask_price=120.0, quantity=0.3, reservation_id="res-1") is None
    assert seen["exclude_reservation_id"] == "res-1"


def test_day_btc_blocks_second_day_btc_while_scalp_btc_is_independent():
    both = {"DAY_V2::BTC/USDT": _lot("DAY_V2"), "SCALP_V2::BTC/USDT": _lot("SCALP_V2")}
    assert "BTCUSDT" in day_v2_held_symbols(both)
    assert "BTCUSDT" not in day_v2_held_symbols({"SCALP_V2::BTC/USDT": _lot("SCALP_V2")})


@pytest.mark.parametrize("equity", [200.0, 320.0, 500.0, 1000.0])
def test_own_reservation_never_double_counts_at_any_equity(tmp_path, monkeypatch, equity):
    import backend.services.two_engine_capital as tec

    monkeypatch.setattr(tec, "get_capital_shares", lambda: (0.5, 0.5))
    notional = round(equity * 0.5 * 0.9, 2)
    db = str(tmp_path / "res.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE day_entry_reservations (reservation_id TEXT, sleeve TEXT, notional_usd REAL, expires_at REAL, status TEXT)")
    conn.execute("INSERT INTO day_entry_reservations VALUES ('own', 'DAY_V2', ?, ?, 'ACTIVE')", (notional, time.time() + 3600))
    conn.commit()
    conn.close()
    assert check_engine_budget(db, "DAY_V2", notional, equity, equity, {})[0] is False
    ok, reason, snap = check_engine_budget(db, "DAY_V2", notional, equity, equity, {}, exclude_reservation_id="own")
    assert (ok, reason) == (True, "")
    assert snap.day.remaining_budget == pytest.approx(equity * 0.5)
    # a DAY reservation never consumes SCALP's half
    ok, _, snap = check_engine_budget(db, "SCALP_V2", notional, equity, equity, {})
    assert ok and snap.scalp.committed_reservations == 0.0


def test_engine_targets_split_strategy_owned_equity_not_protected(tmp_path, monkeypatch):
    import backend.services.two_engine_capital as tec
    from backend.services.protected_external_inventory import record_protected

    monkeypatch.setattr(tec, "get_capital_shares", lambda: (0.5, 0.5))
    db = str(tmp_path / "cap.db")
    conn = sqlite3.connect(db)
    record_protected(conn, symbol="BTC/USDT", quantity=0.0005, cost_price=80000.0, source_trade_id="imp", entry_order_id="1")
    conn.commit()
    conn.close()
    snap = tec.compute_snapshot(db, 300.0, 100.0, {}, prices={"BTC/USDT": 84000.0})
    assert snap.protected_equity == pytest.approx(42.0)
    assert snap.strategy_owned_equity == pytest.approx(258.0)
    assert (snap.day_target, snap.scalp_target) == (pytest.approx(129.0), pytest.approx(129.0))
    d = snap.as_dict()
    assert d["total_account_equity"] == 300.0 and d["strategy_owned_equity"] == pytest.approx(258.0)
    # physical free cash stays final even when the engine budget has room
    ok, reason, _ = tec.check_engine_budget(db, "DAY_V2", 50.0, 300.0, 40.0, {}, prices={"BTC/USDT": 84000.0})
    assert (ok, reason) == (False, tec.PHYSICAL_CASH_UNAVAILABLE)
