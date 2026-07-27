"""Integration-style DAY authority checks (no live exchange)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from backend.services.day_entry_reservations import (
    create_reservation,
    ensure_reservation_schema,
    load_active_reservations,
    release_reservation,
)
from backend.services.day_gate_telemetry import (
    ensure_day_gate_schema,
    record_day_decision,
    record_gate_event,
    record_shadow_reject,
)
from backend.services.portfolio_engine import KillSwitchMode, PortfolioEngine


@pytest.fixture()
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "day_int.db")
    ensure_day_gate_schema(path)
    ensure_reservation_schema(path)
    return path


def test_two_symbols_compete_for_cash_via_reservations(db):
    # Simulate limited cash: two reservations for different symbols — both can succeed
    # at reservation layer; engine cash check would block the second if notional exceeds free cash.
    ok_a, _, _ = create_reservation(db, decision_id="cash_a", symbol="BTC/USDT", notional_usd=600.0)
    ok_b, _, _ = create_reservation(db, decision_id="cash_b", symbol="ETH/USDT", notional_usd=600.0)
    assert ok_a and ok_b
    active = load_active_reservations(db)
    assert len(active) == 2
    total = sum(float(r["notional_usd"]) for r in active)
    assert total == 1200.0


def test_two_signals_same_symbol_one_slot(db):
    ok1, _, rid = create_reservation(db, decision_id="slot1", symbol="SOL/USDT", notional_usd=100.0)
    ok2, reason, _ = create_reservation(db, decision_id="slot2", symbol="SOL/USDT", notional_usd=100.0)
    assert ok1 is True
    assert ok2 is False
    assert reason == "SYMBOL_RESERVED"
    release_reservation(db, reservation_id=rid, reason="FILLED")
    ok3, _, _ = create_reservation(db, decision_id="slot3", symbol="SOL/USDT", notional_usd=100.0)
    assert ok3 is True


def test_duplicate_decision_id_no_double_reservation(db):
    ok1, _r1, id1 = create_reservation(db, decision_id="same", symbol="XRP/USDT", notional_usd=40.0)
    ok2, r2, id2 = create_reservation(db, decision_id="same", symbol="XRP/USDT", notional_usd=40.0)
    assert ok1 and ok2 and id1 == id2 and r2 == "IDEMPOTENT_EXISTING"
    assert len(load_active_reservations(db)) == 1


def test_exchange_reject_releases_reservation(db):
    ok, _, rid = create_reservation(db, decision_id="exrej", symbol="BTC/USDT", notional_usd=80.0)
    assert ok
    release_reservation(db, reservation_id=rid, reason="EXCHANGE_REJECT")
    assert load_active_reservations(db) == []


def test_restart_recovery_loads_active(db):
    create_reservation(db, decision_id="persist1", symbol="ETH/USDT", notional_usd=55.0, sleeve="day")
    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng.db_path = db
    eng._entry_reservations = {}
    eng._reload_entry_reservations_from_db()
    assert "ETH/USDT" in eng._entry_reservations or any(normalize_like(k) for k in eng._entry_reservations)
    # Symbol may be normalized with slash
    assert any("ETH" in k for k in eng._entry_reservations)
    assert float(next(iter(eng._entry_reservations.values()))["notional"]) == 55.0


def normalize_like(k: str) -> bool:
    return "ETH" in k


def test_kill_switch_entry_pause_while_protective_exit_allowed():
    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
    eng._kill_switch_reason = "daily"
    eng._is_emergency_sell = lambda *_args, **_kwargs: False
    assert PortfolioEngine._check_kill_switch_buy(eng)[0] is False
    assert PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="STOP_LOSS")[0] is True

    eng._kill_switch_mode = KillSwitchMode.PAUSE_ALL
    assert PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="EXTREME_PROTECTION")[0] is True
    assert PortfolioEngine._check_kill_switch_sell(eng, exit_trigger="DISCRETIONARY")[0] is False


def test_rejected_candidate_shadow_pipeline(db):
    from types import SimpleNamespace

    record_gate_event(db, gate_id="AW_SETUP_PASS", symbol="BTC/USDT", outcome="passed", setup="BREAKOUT")
    record_gate_event(db, gate_id="ORDERBOOK_PREFLIGHT", symbol="BTC/USDT", outcome="hard_blocked", setup="BREAKOUT", decision_id="pipe1")
    cand = SimpleNamespace(
        symbol="BTC/USDT",
        price=65000.0,
        decision_id="pipe1",
        decision_data={"setup_type": "BREAKOUT", "thesis_invalid_level": 64000.0, "thesis_target_level": 67000.0, "allweather_regime": "trend_up"},
    )
    record_shadow_reject(db, candidate=cand, gate_id="ORDERBOOK_PREFLIGHT", bar_timestamp=int(time.time()))
    record_day_decision(
        db,
        decision_id="pipe1",
        symbol="BTC/USDT",
        aw_valid=True,
        setup="BREAKOUT",
        regime="trend_up",
        gates=[
            {"gate_id": "AW_SETUP_PASS", "outcome": "passed"},
            {"gate_id": "ORDERBOOK_PREFLIGHT", "outcome": "hard_blocked"},
        ],
        first_hard_block="ORDERBOOK_PREFLIGHT",
        final_decision="shadow",
        ml_score=0.03,
        requested_size=0.002,
        approved_size=0.0,
    )
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM day_shadow_rejects WHERE decision_id='pipe1'").fetchone()[0] == 1
    assert conn.execute("SELECT final_decision FROM day_decision_records WHERE decision_id='pipe1'").fetchone()[0] == "shadow"
    conn.close()
    # Shadow must not create reservations
    assert load_active_reservations(db) == []


def test_scalp_gate_modules_still_importable():
    """Regression: SCALP measurement modules remain importable (shared infra unchanged for SCALP path)."""
    from backend.services import scalp_gate_registry, scalp_gate_telemetry

    assert scalp_gate_registry.THRESHOLD_FREEZE_ACTIVE is True
    assert callable(scalp_gate_telemetry.record_gate_event)
