"""Proofs for the two live engines: reaper, reasons, claims, candles, FIFO, authority."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_reaper_expires_armed_and_restart_does_not_resurrect(tmp_path):
    from backend.services.scalp_v2.opportunity import (
        SCALP_V2_OPP_EXPIRY_SEC,
        arm_opportunity,
        opportunity_inventory,
        reap_expired_armed,
    )

    db = tmp_path / "opp.db"
    opp_id, blocked = arm_opportunity(db, "XRP/USDT", "SCALP", 2.5)
    assert blocked is False
    released: list[str] = []
    conn = sqlite3.connect(db)
    old = time.time() - SCALP_V2_OPP_EXPIRY_SEC - 10
    conn.execute(
        "UPDATE scalp_v2_opportunities SET created_at=?, tracked_low=1.0, reservation_id='res_old', reservation_released=0",
        (old,),
    )
    conn.commit()
    conn.close()
    reap_expired_armed(db, release_reservation=lambda rid, _sym: released.append(rid))
    conn = sqlite3.connect(db)
    state, low = conn.execute("SELECT state, tracked_low FROM scalp_v2_opportunities").fetchone()
    conn.close()
    assert state == "EXPIRED"
    assert low == 1.0
    assert released == ["res_old"]
    new_id, blocked_again = arm_opportunity(db, "XRP/USDT", "SCALP", 2.5)
    assert blocked_again is False
    assert new_id == opp_id
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT state, tracked_low, version FROM scalp_v2_opportunities ORDER BY id").fetchall()
    conn.close()
    assert rows[0][0] == "EXPIRED"
    assert rows[1][0] == "ARMED"
    assert rows[1][1] is None
    assert rows[1][2] == 2
    inventory = opportunity_inventory(db)
    assert inventory["armed_current"] == 1
    assert inventory["armed_expired"] == 0
    assert inventory["expired"] >= 1


def test_one_current_opportunity_per_price_zone(tmp_path):
    from backend.services.scalp_v2.opportunity import arm_opportunity, reap_expired_armed

    db = tmp_path / "zone.db"
    _first, blocked_first = arm_opportunity(db, "SOL/USDT", "PULLBACK", 100.0)
    _second, blocked_second = arm_opportunity(db, "SOL/USDT", "REBOUND", 100.05)
    assert blocked_first is False
    assert blocked_second is True
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO scalp_v2_opportunities(
            symbol, opportunity_id, setup_family, structural_anchor,
            state, engine_id, created_at, updated_at, version,
            reservation_id, reservation_released, tracked_low
        ) VALUES ('SOL/USDT', 'other-setup', 'REBOUND',
                  (SELECT structural_anchor FROM scalp_v2_opportunities LIMIT 1),
                  'ARMED', 'SCALP_V2', ?, ?, 1, 'res_dup', 0, NULL)
        """,
        (time.time() - 5, time.time() - 5),
    )
    conn.commit()
    conn.close()
    stats = reap_expired_armed(db, release_reservation=lambda _rid, _sym: None)
    armed = sqlite3.connect(db).execute("SELECT COUNT(*) FROM scalp_v2_opportunities WHERE state='ARMED'").fetchone()[0]
    assert armed == 1
    assert stats["collapsed_duplicates"] >= 1


def test_open_row_survives_reaper(tmp_path):
    from backend.services.scalp_v2.opportunity import SCALP_V2_OPP_EXPIRY_SEC, arm_opportunity, reap_expired_armed

    db = tmp_path / "open.db"
    arm_opportunity(db, "BTC/USDT", "SCALP", 80000)
    conn = sqlite3.connect(db)
    old = time.time() - SCALP_V2_OPP_EXPIRY_SEC - 5
    conn.execute("UPDATE scalp_v2_opportunities SET state='OPEN', created_at=?", (old,))
    conn.commit()
    conn.close()
    reap_expired_armed(db)
    state = sqlite3.connect(db).execute("SELECT state FROM scalp_v2_opportunities").fetchone()[0]
    assert state == "OPEN"


def test_scalp_candidate_records_one_waiting_reason(tmp_path):
    from backend.services.scalp_v2.decision_log import classify_scalp_candidate, reason_counts, record_scalp_decision

    result, reason = classify_scalp_candidate({"entry_eligible": False, "hard_block": None, "soft_reason": "NO_PULLBACK_RECOVERY"})
    assert result == "WAITING_FOR_PULLBACK"
    assert reason == "NO_PULLBACK"
    record_scalp_decision(tmp_path / "d.db", "ETH/USDT", result, reason, cycle_ts=10)
    counts = reason_counts(tmp_path / "d.db", since_ts=0)
    assert counts["ETH/USDT"]["WAITING_FOR_PULLBACK"] == 1


def test_blank_candle_is_not_zero_filled(tmp_path):
    db = tmp_path / "bars.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.execute("INSERT INTO feature_ohlcv VALUES ('XRP-USDT','15m','2026-09-24 21:00:00', 1, 1, 1, 1, NULL)")
    conn.execute("INSERT INTO feature_ohlcv VALUES ('XRP-USDT','15m','2026-09-24 21:15:00', 1, 2, 0.5, 1.5, 10)")
    conn.commit()
    conn.close()
    from backend.services.candle_contract import load_closed_bars

    bars = load_closed_bars(str(db), "XRP-USDT", "15m", 10, as_of=time.time())
    assert len(bars) == 1
    assert bars[0]["volume"] == 10.0


def test_required_15m_open_uses_last_closed_bar():
    from backend.services.day_v2.cycle_gate import required_15m_open

    close_23_30 = 1_790_292_600.0
    assert required_15m_open(close_23_30) == close_23_30 - 900
    mid = close_23_30 + 300
    assert required_15m_open(mid) == close_23_30 - 900


def test_cycle_gate_rejects_stale_boundary_candle():
    from backend.services.day_v2.cycle_gate import HARD_MISSING_CANDLE, cycle_decision

    boundary = 1_790_291_700.0
    stale = cycle_decision(
        completed_bar_count=40,
        minimum_bars=32,
        executable_price=1.53,
        book_age_sec=1.0,
        book_stale_sec=30,
        already_evaluated=False,
        retried=True,
        latest_bar_epoch=boundary - 1800,
        required_open_epoch=boundary - 900,
    )
    assert stale == {"action": "reject", "reason": HARD_MISSING_CANDLE}
    fresh = cycle_decision(
        completed_bar_count=40,
        minimum_bars=32,
        executable_price=1.53,
        book_age_sec=1.0,
        book_stale_sec=30,
        already_evaluated=False,
        retried=False,
        latest_bar_epoch=boundary - 900,
        required_open_epoch=boundary - 900,
    )
    assert fresh["action"] == "proceed"


def test_refresh_schedule_does_not_let_weekly_block_15m():
    from backend.services.canonical_candle_pipeline import next_refresh_pair

    pairs = [("XRPUSDT", "15m"), ("XRPUSDT", "1w")]
    now = 1_000_000.0
    symbol, interval, wait = next_refresh_pair(pairs, {"XRPUSDT:15m": now, "XRPUSDT:1w": now}, now)
    assert (symbol, interval, wait) == ("XRPUSDT", "15m", 0.0)
    symbol, interval, wait = next_refresh_pair(pairs, {"XRPUSDT:15m": now + 30, "XRPUSDT:1w": now + 5}, now)
    assert (symbol, interval, wait) == ("XRPUSDT", "1w", 5.0)


def test_cycle_gate_retries_then_hard_rejects_missing_price():
    from backend.services.day_v2.cycle_gate import HARD_MISSING_PRICE, cycle_decision

    first = cycle_decision(
        completed_bar_count=40,
        minimum_bars=32,
        executable_price=0.0,
        book_age_sec=None,
        book_stale_sec=30,
        already_evaluated=False,
        retried=False,
    )
    assert first["action"] == "retry"
    second = cycle_decision(
        completed_bar_count=40,
        minimum_bars=32,
        executable_price=0.0,
        book_age_sec=None,
        book_stale_sec=30,
        already_evaluated=False,
        retried=True,
    )
    assert second == {"action": "reject", "reason": HARD_MISSING_PRICE}


def test_same_symbol_claim_lets_one_engine_win(tmp_path):
    from backend.services.two_engine_claim import SYMBOL_OCCUPIED_BY_OTHER_ENGINE, claim_symbol

    db = tmp_path / "claim.db"
    results = []
    barrier = threading.Barrier(2)

    def _go(engine: str) -> None:
        barrier.wait()
        results.append(claim_symbol(db, "SOL/USDT", engine, f"dec-{engine}", 25.0))

    threads = [threading.Thread(target=_go, args=(engine,)) for engine in ("SCALP_V2", "DAY_V2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    oks = [row[0] for row in results]
    reasons = [row[1] for row in results]
    assert oks.count(True) == 1
    assert SYMBOL_OCCUPIED_BY_OTHER_ENGINE in reasons


def test_fifo_matches_quantity_and_leaves_unmatched_sell(tmp_path):
    db = tmp_path / "fifo.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE live_exchange_fills (
            id INTEGER PRIMARY KEY,
            symbol TEXT, side TEXT, executed_qty REAL, avg_fill_price REAL,
            fee_amount REAL, fee_asset TEXT, exchange_order_id TEXT,
            event_ts_exchange TEXT, event_ts_recorded TEXT, cost_quote REAL
        )
        """
    )
    conn.execute("INSERT INTO live_exchange_fills VALUES (1,'BTC/USDT','BUY',0.002,100000,0.000001,'BTC','b1','t1','t1',200)")
    conn.execute("INSERT INTO live_exchange_fills VALUES (2,'BTC/USDT','SELL',0.003,101000,0.1,'USDT','s1','t2','t2',303)")
    conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, side TEXT, mode TEXT, order_id TEXT, pnl REAL, exit_reason TEXT)")
    conn.execute("INSERT INTO paper_trades VALUES (1,'SELL','paper','',1.0,'MANUAL')")
    conn.commit()
    conn.close()
    from backend.services.live_fifo_performance import fifo_exchange_performance, quarantine_ghost_rows

    report = fifo_exchange_performance(db)
    assert report["matched_qty"] == pytest.approx(0.001999)
    assert report["unmatched_sell_qty"] > 0
    assert report["paper_rows"] == 1
    conn = sqlite3.connect(db)
    changed = quarantine_ghost_rows(conn)
    conn.commit()
    flagged = conn.execute("SELECT counts_toward_realized FROM paper_trades").fetchone()[0]
    conn.close()
    assert changed == 1
    assert flagged == 0


def test_day_no_signal_names_unmet_condition():
    from backend.services.day_v2.live_signal import explain_no_signal

    bars = []
    price = 100.0
    for i in range(40):
        bars.append({"ts": i, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1})
    explained = explain_no_signal("BTCUSDT", bars, bars[-10:], bars[-12:])
    assert explained["closest"]
    assert explained["unmet"]


@pytest.mark.asyncio
async def test_scalp_confirmed_reaches_adapter_once_and_unknown_authority_does_not(tmp_path, monkeypatch):
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.open_positions = {}
    engine._available_balance = 200.0
    engine._global_cash_lock = asyncio.Lock()
    engine.db_path = str(tmp_path / "live.db")
    engine.last_buy_reject_reason = ""
    engine._check_kill_switch_buy = lambda: (True, "")
    engine._day_entry_held_count = lambda: 0
    engine._normalize_order_amount = lambda **_k: (0.01, "ok", 1.0)
    engine._live_execution_enabled = True
    engine._live_service = object()
    engine._now_ms = lambda: 1
    preflight = MagicMock(passed=True, protected_limit_price=100.0, reject_reason="")
    adapter = AsyncMock(return_value=None)
    monkeypatch.setattr("backend.config.live_test_mode.can_place_live_orders_sync", lambda: (True, ""))
    monkeypatch.setattr("backend.services.execution_mode_service.is_live_execution_allowed_sync", lambda: True)
    monkeypatch.setattr("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=preflight))
    monkeypatch.setattr("backend.services.protected_limit_execution.execute_protected_limit_live", adapter)

    missing = await engine.execute_scalp_v2_buy_live("BTC/USDT", 0.01, 100.0, opportunity_id="opp", entry_authority="")
    assert missing is None
    adapter.assert_not_called()

    from backend.services.two_engine_claim import claim_symbol

    ok, _reason, _rid = claim_symbol(engine.db_path, "BTC/USDT", "DAY_V2", "day-holds-btc", 25.0)
    assert ok is True
    blocked = await engine.execute_scalp_v2_buy_live(
        "BTC/USDT",
        0.01,
        100.0,
        opportunity_id="opp-blocked",
        entry_authority="SCALP_V2_CONFIRMED",
    )
    assert blocked is None
    assert engine.last_buy_reject_reason == "SYMBOL_OCCUPIED_BY_OTHER_ENGINE"
    adapter.assert_not_called()

    reached = await engine.execute_scalp_v2_buy_live(
        "ETH/USDT",
        0.01,
        100.0,
        opportunity_id="opp-eth",
        entry_authority="SCALP_V2_CONFIRMED",
    )
    assert reached is None
    assert adapter.await_count == 1


@pytest.mark.asyncio
async def test_http_buy_stays_retired():
    from fastapi import HTTPException

    from backend.endpoints.live_trading_endpoints import execute_live_market_buy

    with pytest.raises(HTTPException) as raised:
        await execute_live_market_buy({})
    assert raised.value.status_code == 410


def test_scalp_exit_does_not_apply_day_structure_or_default_stall():
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    pos = MagicMock()
    pos.engine_id = "SCALP_V2"
    pos.cost_basis = 100.0
    pos.entry_price = 100.0
    pos.highest_price = 100.2
    pos.lowest_price = 99.8
    pos.symbol = "BTC/USDT"
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.05,
        net_pnl_pct=-0.001,
        hold_minutes=30.0,
        bar_low=99.9,
    )
    assert result.get("action") == "hold"
    day = MagicMock()
    day.engine_id = "DAY_V2"
    assert evaluate_scalp_v2_exit(position=day, current_price=100, net_pnl_pct=0.01, hold_minutes=10, bar_low=99) == {}
