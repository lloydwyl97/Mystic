"""DAY authority model, gate registry, shadow, closed-bar, reservations, kill-switch."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.services.day_bar_integrity import (
    drop_forming_candle,
    forming_candle_cannot_influence,
    validate_exchange_bars,
)
from backend.services.day_entry_reservations import (
    create_reservation,
    load_active_reservations,
    release_reservation,
)
from backend.services.day_gate_registry import (
    DAY_GATES,
    day_aw_owner_enabled,
    day_ml_bypass_enabled,
    get_gate,
    registry_snapshot,
)
from backend.services.day_gate_telemetry import (
    attribution_report,
    ensure_day_gate_schema,
    record_day_decision,
    record_gate_event,
    record_shadow_reject,
)
from backend.services.portfolio_engine import KillSwitchMode, PortfolioEngine


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    db = str(tmp_path / "day_auth.db")
    ensure_day_gate_schema(db)
    return db


def _make_hourly_bars(n: int = 220, *, end_ts: int | None = None, interval: int = 3600) -> list[dict]:
    end = end_ts if end_ts is not None else (int(time.time()) // interval) * interval - interval
    out = []
    for i in range(n):
        ts = end - (n - 1 - i) * interval
        out.append({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + i * 0.01})
    return out


# ---------------------------------------------------------------------------
# Authority / registry
# ---------------------------------------------------------------------------


def test_day_aw_owner_enabled_default_true(monkeypatch):
    monkeypatch.delenv("DAY_AW_OWNER_ENABLED", raising=False)
    assert day_aw_owner_enabled() is True


def test_day_ml_bypass_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DAY_ML_BYPASS_ENABLED", raising=False)
    assert day_ml_bypass_enabled() is False


def test_gate_registry_classification_and_ownership():
    snap = registry_snapshot()
    assert snap["decision_policy_version"] == "day_aw_owner_v1"
    assert "AW_NO_SIGNAL" in DAY_GATES
    g = get_gate("AW_NO_SIGNAL")
    assert g is not None
    assert g.layer == "strategy_signal"
    assert g.behavior == "hard_block"
    assert g.reason_code == "AW_NO_SIGNAL"
    assert g.status == "enabled"
    ml = get_gate("ML_RANK_SIZE")
    assert ml is not None
    assert ml.behavior == "rank"


def test_execution_enabled_follows_aw_owner(monkeypatch):
    monkeypatch.setenv("DAY_AW_OWNER_ENABLED", "true")
    monkeypatch.setenv("ALLWEATHER_BREAKOUT_PULLBACK_ENABLED", "false")
    from backend.services import allweather_breakout_pullback_adapter as aw

    assert aw.execution_enabled() is True
    monkeypatch.setenv("DAY_AW_OWNER_ENABLED", "false")
    monkeypatch.setenv("ALLWEATHER_BREAKOUT_PULLBACK_ENABLED", "false")
    assert aw.execution_enabled() is False


# ---------------------------------------------------------------------------
# Gate counters + decision records + shadow
# ---------------------------------------------------------------------------


def test_record_gate_event_and_first_block(tmp_db):
    record_gate_event(tmp_db, gate_id="AW_SETUP_PASS", symbol="BTC/USDT", outcome="passed", setup="BREAKOUT", regime="trend_up")
    record_gate_event(tmp_db, gate_id="NEGATIVE_EV", symbol="BTC/USDT", outcome="hard_blocked", setup="BREAKOUT", regime="trend_up")
    record_gate_event(tmp_db, gate_id="ORDERBOOK_PREFLIGHT", symbol="BTC/USDT", outcome="hard_blocked", setup="BREAKOUT")
    from backend.services.day_gate_telemetry import counters_today

    rows = {r["gate_id"]: r for r in counters_today(tmp_db)}
    assert rows["AW_SETUP_PASS"]["passed"] >= 1
    assert rows["NEGATIVE_EV"]["hard_blocked"] >= 1


def test_decision_record_stores_gates(tmp_db):
    gates = [
        {"gate_id": "AW_SETUP_PASS", "outcome": "passed"},
        {"gate_id": "NEGATIVE_EV", "outcome": "hard_blocked"},
        {"gate_id": "BUY_MARGIN_FLOOR", "outcome": "hard_blocked"},
    ]
    record_day_decision(
        tmp_db,
        decision_id="dec_1",
        symbol="ETH/USDT",
        aw_valid=True,
        setup="BREAKOUT",
        regime="trend_up",
        gates=gates,
        ml_score=0.05,
        ml_rank_adjustment=0.01,
        ml_size_adjustment=0.9,
        requested_size=0.1,
        approved_size=0.0,
        final_decision="reject",
    )
    conn = sqlite3.connect(tmp_db)
    row = conn.execute("SELECT aw_valid, first_hard_block, other_blocking_gates_json, final_decision FROM day_decision_records WHERE decision_id='dec_1'").fetchone()
    conn.close()
    assert row[0] == 1
    assert row[1] == "NEGATIVE_EV"
    assert "BUY_MARGIN_FLOOR" in (row[2] or "")
    assert row[3] == "reject"


def test_shadow_reject_idempotent_and_no_cash(tmp_db):
    cand = SimpleNamespace(
        symbol="SOL/USDT",
        price=150.0,
        decision_id="dec_shadow_1",
        decision_data={"setup_type": "BREAKOUT", "thesis_invalid_level": 145.0, "thesis_target_level": 160.0},
    )
    record_shadow_reject(tmp_db, candidate=cand, gate_id="AW_NO_SIGNAL", bar_timestamp=int(time.time()))
    record_shadow_reject(tmp_db, candidate=cand, gate_id="AW_NO_SIGNAL", bar_timestamp=int(time.time()))
    conn = sqlite3.connect(tmp_db)
    n = conn.execute("SELECT COUNT(*) FROM day_shadow_rejects WHERE decision_id='dec_shadow_1'").fetchone()[0]
    conn.close()
    assert n == 1
    # No reservation / cash tables touched
    assert load_active_reservations(tmp_db) == []


def test_attribution_saved_vs_destroyed_fields(tmp_db):
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00+00:00")
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY, side TEXT, pnl REAL, fees_paid REAL, slippage_cost REAL,
            exit_type TEXT, timestamp TEXT, is_synthetic INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO day_shadow_rejects(created_at, decision_id, symbol, setup, gate_id, entry_price, hyp_net_pnl, hyp_resolved_at) VALUES (?,?,?,?,?,?,?,?)",
        (today, "d1", "BTC/USDT", "BREAKOUT", "NEGATIVE_EV", 100.0, -2.5, today),
    )
    conn.execute(
        "INSERT INTO day_shadow_rejects(created_at, decision_id, symbol, setup, gate_id, entry_price, hyp_net_pnl, hyp_resolved_at) VALUES (?,?,?,?,?,?,?,?)",
        (today, "d2", "ETH/USDT", "BREAKOUT", "NEGATIVE_EV", 200.0, 3.0, today),
    )
    conn.commit()
    conn.close()
    report = attribution_report(tmp_db)
    assert "gate_opportunity" in report
    opp = {r["gate_id"]: r for r in report["gate_opportunity"]}
    assert "NEGATIVE_EV" in opp
    assert "gate_saved_expectancy" in opp["NEGATIVE_EV"]
    assert "gate_destroyed_expectancy" in opp["NEGATIVE_EV"]


# ---------------------------------------------------------------------------
# Closed-bar integrity
# ---------------------------------------------------------------------------


def test_forming_candle_dropped():
    interval = 3600
    closed_ts = (int(time.time()) // interval) * interval - interval
    bars = _make_hourly_bars(10, end_ts=closed_ts, interval=interval)
    forming = dict(bars[-1])
    forming["ts"] = closed_ts + interval
    forming["close"] = 9999.0
    with_forming = [*bars, forming]
    out, dropped, cts = drop_forming_candle(with_forming, interval_sec=interval)
    assert dropped is True
    assert len(out) == len(bars)
    assert out[-1]["close"] != 9999.0
    assert cts == closed_ts


def test_forming_candle_cannot_influence_strategy_decision():
    interval = 3600
    closed_ts = (int(time.time()) // interval) * interval - interval
    bars = _make_hourly_bars(30, end_ts=closed_ts, interval=interval)

    def compute(b):
        return round(sum(x["close"] for x in b[-5:]) / 5.0, 6) if len(b) >= 5 else None

    assert forming_candle_cannot_influence(bars, interval_sec=interval, compute_fn=compute) is True


def test_validate_rejects_future_duplicate_ooo_wrong_boundary():
    interval = 3600
    now = time.time()
    base = int(now // interval) * interval - interval
    good = [{"ts": base - interval, "open": 1, "high": 1, "low": 1, "close": 1}, {"ts": base, "open": 1, "high": 1, "low": 1, "close": 1}]
    assert validate_exchange_bars(good, interval_sec=interval, min_bars=2, now=now).ok

    future = [*good, {"ts": int(now) + 7200, "open": 1, "high": 1, "low": 1, "close": 1}]
    assert validate_exchange_bars(future, interval_sec=interval, min_bars=1, now=now).error_code == "FUTURE_CANDLE"

    dup = [{"ts": base, "open": 1, "high": 1, "low": 1, "close": 1}, {"ts": base, "open": 1, "high": 1, "low": 1, "close": 2}]
    assert validate_exchange_bars(dup, interval_sec=interval, min_bars=1, now=now).error_code == "DUPLICATE_CANDLE"

    ooo = [{"ts": base, "open": 1, "high": 1, "low": 1, "close": 1}, {"ts": base - interval, "open": 1, "high": 1, "low": 1, "close": 1}]
    assert validate_exchange_bars(ooo, interval_sec=interval, min_bars=1, now=now).error_code == "OUT_OF_ORDER"

    bad_bound = [{"ts": base + 1, "open": 1, "high": 1, "low": 1, "close": 1}]
    assert validate_exchange_bars(bad_bound, interval_sec=interval, min_bars=1, now=now).error_code == "WRONG_INTERVAL_BOUNDARY"


# ---------------------------------------------------------------------------
# Atomic reservations + pending exposure + idempotency
# ---------------------------------------------------------------------------


def test_reservation_idempotent_by_decision_id(tmp_db):
    ok1, _reason1, rid1 = create_reservation(tmp_db, decision_id="dA", symbol="BTC/USDT", notional_usd=100.0, risk_usd=5.0, sleeve="day")
    ok2, reason2, rid2 = create_reservation(tmp_db, decision_id="dA", symbol="BTC/USDT", notional_usd=100.0)
    assert ok1 and ok2
    assert reason2 == "IDEMPOTENT_EXISTING"
    assert rid1 == rid2
    assert len(load_active_reservations(tmp_db)) == 1


def test_reservation_blocks_same_symbol_different_decision(tmp_db):
    ok1, _, _ = create_reservation(tmp_db, decision_id="d1", symbol="ETH/USDT", notional_usd=50.0)
    ok2, reason2, _ = create_reservation(tmp_db, decision_id="d2", symbol="ETH/USDT", notional_usd=50.0)
    assert ok1 is True
    assert ok2 is False
    assert reason2 == "SYMBOL_RESERVED"


def test_reservation_release_idempotent(tmp_db):
    ok, _, rid = create_reservation(tmp_db, decision_id="dR", symbol="SOL/USDT", notional_usd=25.0)
    assert ok
    assert release_reservation(tmp_db, reservation_id=rid) is True
    assert release_reservation(tmp_db, reservation_id=rid) is True
    assert load_active_reservations(tmp_db) == []


def test_concurrent_reservations_only_one_wins(tmp_db):
    results = []

    def worker(i: int):
        ok, reason, rid = create_reservation(
            tmp_db,
            decision_id=f"conc_{i}",
            symbol="XRP/USDT",
            notional_usd=10.0,
        )
        results.append((ok, reason, rid))

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(8)))
    winners = [r for r in results if r[0]]
    assert len(winners) == 1
    assert len(load_active_reservations(tmp_db)) == 1


def test_engine_pending_exposure_counts_reservations():
    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng._entry_reservations = {"BTC/USDT": {"notional": 200.0, "risk_usd": 4.0, "decision_id": "d1", "sleeve": "day"}}
    eng._pending_orders = {}
    assert eng._pending_buy_notional() == 200.0
    assert "BTC/USDT" in eng._pending_buy_symbols()


# ---------------------------------------------------------------------------
# Kill switch protective exits
# ---------------------------------------------------------------------------


def test_pause_buys_blocks_entries_allows_sells():
    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
    eng._kill_switch_reason = "test"
    ok_buy, _ = PortfolioEngine._check_kill_switch_buy(eng)
    assert ok_buy is False
    ok_sell, _ = PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="MANUAL")
    assert ok_sell is True


def test_pause_all_blocks_discretionary_allows_protective():
    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng._kill_switch_mode = KillSwitchMode.PAUSE_ALL
    eng._kill_switch_reason = "halt"
    # bind real emergency helper
    eng._is_emergency_sell = lambda *_args, **_kwargs: False
    ok_disc, reason = PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="MANUAL_TRIM")
    assert ok_disc is False
    assert "PAUSE_ALL" in reason
    ok_prot, _ = PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="ALLWEATHER_ATR_STOP_EXIT")
    assert ok_prot is True
    ok_force, _ = PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="x", force_sell=True)
    assert ok_force is True


def test_pause_all_entries_alias_allows_sells():
    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng._kill_switch_mode = KillSwitchMode.PAUSE_ALL_ENTRIES
    eng._kill_switch_reason = "entries"
    assert PortfolioEngine._check_kill_switch_buy(eng)[0] is False
    assert PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="STOP_LOSS")[0] is True


# ---------------------------------------------------------------------------
# ML cannot override AW rejection (unit-level authority)
# ---------------------------------------------------------------------------


def test_prepare_path_aw_no_signal_terminal_policy():
    """Under AW owner, NO_SIGNAL is hard_block in registry — ML rank gate is rank-only."""
    aw = get_gate("AW_NO_SIGNAL")
    ml = get_gate("ML_RANK_SIZE")
    assert aw.behavior == "hard_block"
    assert ml.behavior == "rank"
    assert aw.dependency == "strategy_critical"


def test_ml_bypass_flag_off_means_no_ev_bypass(monkeypatch):
    monkeypatch.setenv("DAY_ML_BYPASS_ENABLED", "false")
    assert day_ml_bypass_enabled() is False
    monkeypatch.setenv("DAY_ML_BYPASS_ENABLED", "true")
    assert day_ml_bypass_enabled() is True


# ---------------------------------------------------------------------------
# Structured log examples (smoke — ensure helpers emit without error)
# ---------------------------------------------------------------------------


def test_structured_decision_examples(tmp_db, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        record_day_decision(
            tmp_db,
            decision_id="ex_exec",
            symbol="BTC/USDT",
            aw_valid=True,
            setup="BREAKOUT",
            regime="trend_up",
            gates=[{"gate_id": "AW_SETUP_PASS", "outcome": "passed"}],
            requested_size=0.01,
            approved_size=0.01,
            final_decision="execute",
            ml_score=0.04,
            ml_size_adjustment=1.0,
        )
        record_day_decision(
            tmp_db,
            decision_id="ex_aw_reject",
            symbol="ETH/USDT",
            aw_valid=False,
            setup="",
            regime="range",
            gates=[{"gate_id": "AW_NO_SIGNAL", "outcome": "hard_blocked"}],
            first_hard_block="AW_NO_SIGNAL",
            final_decision="shadow",
        )
        cand = SimpleNamespace(
            symbol="SOL/USDT",
            price=140.0,
            decision_id="ex_gate_reject",
            decision_data={"setup_type": "TREND_PULLBACK", "thesis_invalid_level": 130.0, "thesis_target_level": 155.0},
        )
        record_shadow_reject(tmp_db, candidate=cand, gate_id="ORDERBOOK_PREFLIGHT")
    assert any("DAY_DECISION" in r.message for r in caplog.records)
