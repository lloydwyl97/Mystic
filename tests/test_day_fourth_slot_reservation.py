"""Exact four-slot Decimal reservation — reported Ocean cash case."""

from __future__ import annotations

from decimal import Decimal

from backend.services.day_entry_reservations import (
    create_reservation,
    load_active_reservations,
    release_reservation,
)
from backend.services.day_entry_spendable import (
    money,
    plan_reservation,
    remaining_slot_cap,
    reservations_within_cash,
    spendable_quote,
)
from backend.services.portfolio_engine import PortfolioEngine

CASH = Decimal("228.06746265")
SLOT = Decimal("57.0168656625")
THREE = Decimal("171.0505969875")
REMAINING = Decimal("57.0168656625")
FLOAT_RESIDUE = Decimal("57.01686566250001")


def test_four_equal_slots_sum_exactly_to_cash():
    cap = remaining_slot_cap(free_cash=CASH, remaining_new_slots=4)
    assert cap == SLOT
    assert cap * 4 == CASH
    total = SLOT + SLOT + SLOT + SLOT
    assert total == CASH
    assert remaining_slot_cap(free_cash=CASH - THREE, remaining_new_slots=1) == REMAINING


def test_repeated_slot_math_is_deterministic():
    first = [remaining_slot_cap(free_cash=CASH, remaining_new_slots=4) for _ in range(8)]
    assert all(v == SLOT for v in first)
    leftover = CASH
    reserved = []
    for left in (4, 3, 2, 1):
        cap = remaining_slot_cap(free_cash=leftover, remaining_new_slots=left)
        ok, amt, reason = plan_reservation(target=cap, remaining_cash=leftover)
        assert ok, reason
        reserved.append(amt)
        leftover -= amt
    assert reserved == [SLOT, SLOT, SLOT, SLOT]
    assert leftover == Decimal("0")
    assert sum(reserved, Decimal("0")) == CASH


def test_fourth_slot_float_residue_reserves_exact_remaining():
    spendable = spendable_quote(account_cash=CASH, other_reservations=THREE)
    assert spendable == REMAINING
    ok, reserved, reason = plan_reservation(target=FLOAT_RESIDUE, remaining_cash=spendable)
    assert ok, reason
    assert reserved == REMAINING
    assert reservations_within_cash(reservations=THREE + reserved, cash=CASH)


def test_genuine_shortfall_is_terminal():
    ok, reserved, reason = plan_reservation(target=SLOT, remaining_cash=Decimal("0"))
    assert not ok
    assert reserved == Decimal("0")
    assert reason.startswith("INSUFFICIENT_CASH_WITH_PENDING")


def test_engine_fourth_slot_accepts_reported_residue(tmp_path):
    db = tmp_path / "slots.db"
    engine = PortfolioEngine(db_path=str(db), principal=float(CASH), test_mode=True)
    engine._ensure_db_schema()
    engine.cash_balance = float(CASH)
    engine._available_balance = float(CASH)
    engine.max_positions = 4
    for i, sym in enumerate(("SOL/USDT", "ETH/USDT", "XRP/USDT")):
        ok, reason = engine._try_reserve_entry(sym, SLOT, decision_id=f"d{i}")
        assert ok, reason
    ok, reason = engine._try_reserve_entry("BTC/USDT", FLOAT_RESIDUE, decision_id="d3")
    assert ok, reason
    notionals = [money(r["notional"]) for r in engine._entry_reservations.values()]
    assert sum(notionals, Decimal("0")) == CASH
    assert all(n == SLOT for n in notionals)
    assert reservations_within_cash(reservations=sum(notionals, Decimal("0")), cash=CASH)


def test_restart_recovery_preserves_exact_reservations(tmp_path):
    db = tmp_path / "rec.db"
    engine = PortfolioEngine(db_path=str(db), principal=float(CASH), test_mode=True)
    engine._ensure_db_schema()
    engine.cash_balance = float(CASH)
    engine._available_balance = float(CASH)
    for i, sym in enumerate(("SOL/USDT", "ETH/USDT", "XRP/USDT", "BTC/USDT")):
        ok, reason = engine._try_reserve_entry(sym, SLOT, decision_id=f"rec{i}")
        assert ok, reason
    other = PortfolioEngine(db_path=str(db), principal=float(CASH), test_mode=True)
    other._ensure_db_schema()
    other._reload_entry_reservations_from_db()
    recovered = [money(r["notional"]) for r in other._entry_reservations.values()]
    assert sum(recovered, Decimal("0")) == CASH
    rows = load_active_reservations(str(db))
    assert sum((money(r["notional_usd"]) for r in rows), Decimal("0")) == CASH


def test_reservation_release_once(tmp_path):
    db = tmp_path / "rel.db"
    ok, reason, rid = create_reservation(db, decision_id="rel1", symbol="BTC/USDT", notional_usd=SLOT)
    assert ok, reason
    assert release_reservation(db, reservation_id=rid, reason="RELEASED")
    assert release_reservation(db, reservation_id=rid, reason="RELEASED")
    assert load_active_reservations(db) == []
