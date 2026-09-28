"""Two-engine capital/position architecture (DAY_V2 + SCALP_V2).

Contract under test:
- SCALP_MAX_OPEN_POSITIONS = 4, DAY_MAX_OPEN_POSITIONS = 4, combined max = 8.
- Position identity is (engine_id, symbol): SCALP BTC and DAY BTC coexist;
  same-engine duplicates are blocked.
- Capital allocator splits CURRENT equity 50/50 (config-driven, no borrowing,
  grandfathered over-allocation, physical-cash gate).
- Reservations are engine-scoped; migration is idempotent and preserves
  economics.
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import backend.services.portfolio_engine as pe
from backend.services.portfolio_engine import (
    COMBINED_ENGINE_MAX_POSITIONS,
    DAY_MAX_OPEN_POSITIONS,
    OpenPosition,
    PortfolioEngine,
    make_position_key,
    normalize_symbol,
    split_position_key,
)
from backend.services.two_engine_capital import (
    CAPITAL_CONFIG_INVALID,
    ENGINE_BUDGET_EXCEEDED,
    PHYSICAL_CASH_UNAVAILABLE,
    check_engine_budget,
    compute_snapshot,
    get_capital_shares,
)
from backend.services.two_engine_claim import (
    ENGINE_MAX_POSITIONS,
    MAX_COMBINED_POSITIONS,
    SYMBOL_OCCUPIED,
    claim_symbol,
    engine_cap,
    held_slot_count,
)

DAY = "DAY_V2"
SCALP = "SCALP_V2"
TOP4 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lot(symbol: str, engine_id: str, *, qty: float = 0.01, price: float = 100.0, status: str = "ACTIVE", trade_id: str = "") -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time(),
        trade_id=trade_id or f"t-{engine_id}-{symbol.replace('/', '')}",
        stop_price=price * 0.99,
        take_profit_1_price=price * 1.01,
        take_profit_2_price=price * 1.02,
        status=status,
        engine_id=engine_id,
        original_position_cost=qty * price,
    )


def _reset_trade_state() -> None:
    """Clear process-global trade-state (singleton local map + redis keys).

    Sell-path tests earlier in the suite leave BTC/USDT in COOLDOWN; the DAY
    gate reads that state, so capacity tests must start from a clean slate.
    """
    try:
        from backend.services.trade_state import get_trade_state_store

        store = get_trade_state_store()
        try:
            store._local_state.clear()
        except Exception:
            pass
        try:
            rc = getattr(store, "redis_client", None)
            if rc is not None:
                keys = rc.keys("trade_state:*") or []
                if keys:
                    rc.delete(*keys)
        except Exception:
            pass
    except Exception:
        pass


def _engine(tmp_path, cash: float = 1000.0) -> PortfolioEngine:
    _reset_trade_state()
    eng = PortfolioEngine(db_path=str(tmp_path / "two_engine.db"), principal=cash, test_mode=True)
    eng._ensure_db_schema()
    eng.cash_balance = cash
    eng._available_balance = cash
    eng._total_open_risk = 0.0
    eng._get_loss_hold_until = AsyncMock(return_value=None)
    eng.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 999))
    return eng


def _fill_book(eng: PortfolioEngine, engine_id: str, symbols: list[str]) -> None:
    for sym in symbols:
        eng.open_positions[make_position_key(engine_id, sym)] = _lot(sym, engine_id)


# ---------------------------------------------------------------------------
# 0. Constants: 4 + 4 = 8
# ---------------------------------------------------------------------------


def test_engine_caps_are_four_each_eight_combined():
    assert DAY_MAX_OPEN_POSITIONS == 4
    assert pe.SCALP_MAX_OPEN_POSITIONS == 4
    assert COMBINED_ENGINE_MAX_POSITIONS == 8
    assert engine_cap(DAY) == 4
    assert engine_cap(SCALP) == 4


def test_position_key_identity_is_engine_plus_symbol():
    assert make_position_key(SCALP, "BTCUSDT") == "SCALP_V2::BTC/USDT"
    assert make_position_key(DAY, "BTC/USDT") == "DAY_V2::BTC/USDT"
    assert make_position_key(SCALP, "BTC/USDT") != make_position_key(DAY, "BTC/USDT")
    assert split_position_key("SCALP_V2::BTC/USDT") == ("SCALP_V2", "BTC/USDT")
    # Bare legacy keys split to ("", symbol).
    assert split_position_key("BTC/USDT") == ("", "BTC/USDT")
    assert normalize_symbol("BTCUSDT") == "BTC/USDT"


def test_capital_shares_default_fifty_fifty(monkeypatch):
    monkeypatch.delenv("DAY_CAPITAL_SHARE", raising=False)
    monkeypatch.delenv("SCALP_CAPITAL_SHARE", raising=False)
    assert get_capital_shares() == (0.5, 0.5)


def test_capital_shares_reject_over_allocation(monkeypatch):
    monkeypatch.setenv("DAY_CAPITAL_SHARE", "0.6")
    monkeypatch.setenv("SCALP_CAPITAL_SHARE", "0.6")
    with pytest.raises(ValueError):
        get_capital_shares()
    ok, reason, _ = check_engine_budget(":memory:", DAY, 1.0, 320.0, 320.0, {})
    assert ok is False
    assert reason == CAPITAL_CONFIG_INVALID


# ---------------------------------------------------------------------------
# 1. Capacity matrix (A-F) on the real gate
# ---------------------------------------------------------------------------


async def test_capacity_a_empty_book_both_may_enter(tmp_path):
    eng = _engine(tmp_path)
    ok_day, _ = await eng._can_open_position("BTC/USDT", 40.0, engine_id=DAY)
    ok_scalp, _ = await eng._can_open_position("BTC/USDT", 40.0, engine_id=SCALP)
    assert ok_day is True
    assert ok_scalp is True


async def test_capacity_b_scalp_full_day_still_has_four_slots(tmp_path):
    eng = _engine(tmp_path)
    _fill_book(eng, SCALP, TOP4)
    assert eng._engine_held_count(SCALP) == 4
    assert eng._engine_held_count(DAY) == 0
    # DAY may still open — even on a SCALP-held symbol.
    ok, reason = await eng._can_open_position("SOL/USDT", 40.0, engine_id=DAY)
    assert ok is True, reason
    # ... while a 5th SCALP lot is blocked.
    ok5, reason5 = await eng._can_open_position("SOL/USDT", 40.0, engine_id=SCALP)
    assert ok5 is False
    assert reason5 == "ENGINE_MAX_POSITIONS"


async def test_capacity_c_day_full_scalp_still_has_four_slots(tmp_path):
    eng = _engine(tmp_path)
    _fill_book(eng, DAY, TOP4)
    assert eng._engine_held_count(DAY) == 4
    assert eng._engine_held_count(SCALP) == 0
    ok, reason = await eng._can_open_position("BTC/USDT", 40.0, engine_id=SCALP)
    assert ok is True, reason
    ok5, reason5 = await eng._can_open_position("BTC/USDT", 40.0, engine_id=DAY)
    assert ok5 is False
    assert reason5 == "ENGINE_MAX_POSITIONS"


async def test_capacity_d_full_book_of_eight_is_valid(tmp_path):
    eng = _engine(tmp_path)
    _fill_book(eng, SCALP, TOP4)
    _fill_book(eng, DAY, TOP4)
    assert eng._engine_held_count(SCALP) == 4
    assert eng._engine_held_count(DAY) == 4
    assert eng._combined_held_count() == 8
    # Each engine is also at its own cap, so the engine firewall fires first.
    ok, reason = await eng._can_open_position("BTC/USDT", 40.0, engine_id=DAY, decision_id="nine-day")
    assert ok is False
    assert reason == "ENGINE_MAX_POSITIONS"
    # Combined-cap firewall in isolation: 4 SCALP + 3 DAY held + 1 pending
    # reservation elsewhere, with DAY under its own cap.
    eng2 = _engine(tmp_path)
    _fill_book(eng2, SCALP, TOP4)
    _fill_book(eng2, DAY, TOP4[:3])
    eng2._entry_reservations["SCALP_V2::DOGE/USDT"] = {"notional": 40.0, "decision_id": "pend-1", "sleeve": "SCALP_V2"}
    ok2, reason2 = await eng2._can_open_position("XRP/USDT", 40.0, engine_id=DAY, decision_id="eight-day")
    assert ok2 is False
    assert reason2 == "MAX_POSITIONS_REACHED"


async def test_capacity_e_f_fifth_lot_per_engine_blocked(tmp_path):
    eng = _engine(tmp_path)
    others = ["DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT"]
    _fill_book(eng, SCALP, TOP4)
    ok, reason = await eng._can_open_position(others[0], 40.0, engine_id=SCALP)
    assert (ok, reason) == (False, "ENGINE_MAX_POSITIONS")
    _fill_book(eng, DAY, others)
    ok, reason = await eng._can_open_position("BNB/USDT", 40.0, engine_id=DAY)
    assert (ok, reason) == (False, "ENGINE_MAX_POSITIONS")
    # Neither engine's lots counted against the other engine's cap.
    assert eng._engine_held_count(SCALP) == 4
    assert eng._engine_held_count(DAY) == 4


# ---------------------------------------------------------------------------
# 2. Same-symbol dual ownership
# ---------------------------------------------------------------------------


def test_same_symbol_lots_coexist_with_independent_identity(tmp_path):
    eng = _engine(tmp_path)
    scalp_btc = _lot("BTC/USDT", SCALP, qty=0.00040, price=84000.0, trade_id="s1")
    day_btc = _lot("BTC/USDT", DAY, qty=0.00045, price=85000.0, trade_id="d1")
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = scalp_btc
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = day_btc

    lots = eng._symbol_lots("BTC/USDT")
    assert len(lots) == 2
    found_scalp = eng._find_position(SCALP, "BTC/USDT")
    found_day = eng._find_position(DAY, "BTC/USDT")
    assert found_scalp is scalp_btc
    assert found_day is day_btc
    # Independent economics: cost bases never merged.
    assert found_scalp.entry_price == 84000.0
    assert found_day.entry_price == 85000.0
    assert found_scalp.quantity == 0.00040
    assert found_day.quantity == 0.00045


def test_same_engine_duplicate_blocked_cross_engine_allowed(tmp_path):
    eng = _engine(tmp_path)
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = _lot("BTC/USDT", SCALP)
    # DAY-side duplicate logic sees only DAY-side lots.
    assert eng._day_path_ev_entry_block_reason("BTC/USDT", 4) is None
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = _lot("BTC/USDT", DAY)
    assert eng._day_path_ev_entry_block_reason("BTC/USDT", 4) == "DUPLICATE_SAME_SYMBOL"
    # SCALP-side duplicate: find resolves SCALP's own lot.
    assert eng._find_position(SCALP, "BTC/USDT") is not None
    # Legacy bare-symbol DAY lot still blocks DAY (heritage), never SCALP.
    eng2 = _engine(tmp_path)
    eng2.open_positions["ETH/USDT"] = _lot("ETH/USDT", "LEGACY_DAY_LIVE")
    assert eng2._day_path_ev_entry_block_reason("ETH/USDT", 4) == "DUPLICATE_SAME_SYMBOL"
    assert eng2._find_position(SCALP, "ETH/USDT") is None
    assert eng2._find_position(DAY, "ETH/USDT") is not None


def test_step_rounded_lots_stay_separate_sol_xrp(tmp_path):
    eng = _engine(tmp_path)
    # SOL step 0.001, XRP step 0.1 style rounding — separate books per engine.
    eng.open_positions[make_position_key(SCALP, "SOL/USDT")] = _lot("SOL/USDT", SCALP, qty=0.013, price=121.68, trade_id="s-sol")
    eng.open_positions[make_position_key(DAY, "SOL/USDT")] = _lot("SOL/USDT", DAY, qty=0.014, price=122.10, trade_id="d-sol")
    eng.open_positions[make_position_key(SCALP, "XRP/USDT")] = _lot("XRP/USDT", SCALP, qty=14.7, price=1.508, trade_id="s-xrp")
    eng.open_positions[make_position_key(DAY, "XRP/USDT")] = _lot("XRP/USDT", DAY, qty=15.2, price=1.511, trade_id="d-xrp")
    assert len(eng._symbol_lots("SOL/USDT")) == 2
    assert len(eng._symbol_lots("XRP/USDT")) == 2
    assert eng._combined_held_count() == 4
    assert eng._engine_held_count(SCALP) == 2
    assert eng._engine_held_count(DAY) == 2
    # Engine-scoped sell resolution: SCALP's SOL lot is not DAY's.
    scalp_sol = eng._find_position(SCALP, "SOL/USDT")
    assert scalp_sol.trade_id == "s-sol"
    assert scalp_sol.quantity == 0.013


# ---------------------------------------------------------------------------
# 3. Capital allocator matrix
# ---------------------------------------------------------------------------


def _pos(engine_id: str, cost: float) -> SimpleNamespace:
    return SimpleNamespace(engine_id=engine_id, quantity=1.0, entry_price=cost, original_position_cost=cost, status="ACTIVE")


def test_capital_split_three_twenty_committed_150_40(tmp_path, monkeypatch):
    monkeypatch.delenv("DAY_CAPITAL_SHARE", raising=False)
    monkeypatch.delenv("SCALP_CAPITAL_SHARE", raising=False)
    db = str(tmp_path / "cap.db")
    positions = {
        make_position_key(SCALP, "BTC/USDT"): _pos(SCALP, 150.0),
        make_position_key(DAY, "ETH/USDT"): _pos(DAY, 40.0),
    }
    snap = compute_snapshot(db, 320.0, 130.0, positions)
    assert snap.day.target_capital == pytest.approx(160.0)
    assert snap.scalp.target_capital == pytest.approx(160.0)
    assert snap.scalp.committed_total == pytest.approx(150.0)
    assert snap.day.committed_total == pytest.approx(40.0)
    assert snap.scalp.remaining_budget == pytest.approx(10.0)
    assert snap.day.remaining_budget == pytest.approx(120.0)


def test_capital_scales_with_equity(tmp_path, monkeypatch):
    monkeypatch.delenv("DAY_CAPITAL_SHARE", raising=False)
    monkeypatch.delenv("SCALP_CAPITAL_SHARE", raising=False)
    db = str(tmp_path / "cap.db")
    for equity in (200.0, 500.0, 1000.0):
        snap = compute_snapshot(db, equity, equity, {})
        assert snap.day.target_capital == pytest.approx(equity * 0.5)
        assert snap.scalp.target_capital == pytest.approx(equity * 0.5)
        assert snap.day.remaining_budget == pytest.approx(equity * 0.5)


def test_capital_grandfather_no_forced_block_on_committed(tmp_path, monkeypatch):
    """Over-target committed capital only constrains NEW entries (budget gate),
    never rewrites existing lots."""
    monkeypatch.delenv("DAY_CAPITAL_SHARE", raising=False)
    monkeypatch.delenv("SCALP_CAPITAL_SHARE", raising=False)
    db = str(tmp_path / "cap.db")
    positions = {make_position_key(SCALP, "BTC/USDT"): _pos(SCALP, 200.0)}
    ok, reason, snap = check_engine_budget(db, SCALP, 10.0, 320.0, 110.0, positions)
    assert ok is False
    assert reason == ENGINE_BUDGET_EXCEEDED
    assert snap is not None and snap.scalp.remaining_budget == 0.0
    # The grandfathered lot itself is untouched.
    assert positions[make_position_key(SCALP, "BTC/USDT")].original_position_cost == 200.0


def test_capital_physical_cash_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("DAY_CAPITAL_SHARE", raising=False)
    monkeypatch.delenv("SCALP_CAPITAL_SHARE", raising=False)
    db = str(tmp_path / "cap.db")
    # DAY has virtual budget ($160 target, $0 committed) but no real cash.
    ok, reason, _ = check_engine_budget(db, DAY, 40.0, 320.0, 5.0, {})
    assert ok is False
    assert reason == PHYSICAL_CASH_UNAVAILABLE
    # With real cash, the same order passes.
    ok, reason, _ = check_engine_budget(db, DAY, 40.0, 320.0, 200.0, {})
    assert ok is True, reason


def test_capital_reservations_count_toward_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("DAY_CAPITAL_SHARE", raising=False)
    monkeypatch.delenv("SCALP_CAPITAL_SHARE", raising=False)
    from backend.services.day_entry_reservations import create_reservation

    db = str(tmp_path / "cap.db")
    ok, _, _ = create_reservation(db, decision_id="d-day-1", symbol="BTC/USDT", notional_usd=30.0, sleeve=DAY)
    assert ok is True
    ok, _, _ = create_reservation(db, decision_id="d-scalp-1", symbol="ETH/USDT", notional_usd=20.0, sleeve=SCALP)
    assert ok is True
    snap = compute_snapshot(db, 320.0, 270.0, {})
    assert snap.day.committed_reservations == pytest.approx(30.0)
    assert snap.scalp.committed_reservations == pytest.approx(20.0)
    assert snap.day.remaining_budget == pytest.approx(130.0)
    assert snap.scalp.remaining_budget == pytest.approx(140.0)


# ---------------------------------------------------------------------------
# 4. Reservations: coexistence cross-engine, unique within engine
# ---------------------------------------------------------------------------


def test_reservations_coexist_cross_engine_block_same_engine(tmp_path):
    from backend.services.day_entry_reservations import create_reservation

    db = str(tmp_path / "res.db")
    ok, _, rid_day = create_reservation(db, decision_id="d1", symbol="BTC/USDT", notional_usd=40.0, sleeve=DAY)
    assert ok is True and rid_day
    ok, _, rid_scalp = create_reservation(db, decision_id="s1", symbol="BTC/USDT", notional_usd=40.0, sleeve=SCALP)
    assert ok is True and rid_scalp
    # Same engine + same symbol duplicate is blocked.
    ok, reason, _ = create_reservation(db, decision_id="d2", symbol="BTC/USDT", notional_usd=40.0, sleeve=DAY)
    assert ok is False
    assert reason == "SYMBOL_RESERVED"


# ---------------------------------------------------------------------------
# 5. Claim semantics (two_engine_claim)
# ---------------------------------------------------------------------------


def test_claim_cross_engine_allowed_same_engine_blocked(tmp_path):
    db = str(tmp_path / "claim.db")
    positions = {make_position_key(SCALP, "BTC/USDT"): _lot("BTC/USDT", SCALP)}
    # SCALP holding BTC does not stop a DAY claim on BTC.
    ok, reason, rid = claim_symbol(db, "BTC/USDT", DAY, "dec-day-1", 40.0, positions=positions)
    assert ok is True, reason
    assert rid
    # A second SCALP claim on BTC is a same-engine duplicate.
    ok, reason, _ = claim_symbol(db, "BTC/USDT", SCALP, "dec-scalp-2", 40.0, positions=positions)
    assert ok is False
    assert reason == SYMBOL_OCCUPIED


def test_claim_engine_and_combined_caps(tmp_path):
    db = str(tmp_path / "claim.db")
    positions: dict = {}
    for i, sym in enumerate(TOP4):
        positions[f"SCALP_V2::{sym}"] = _lot(sym, SCALP, trade_id=f"s{i}")
    assert held_slot_count(positions, engine_id=SCALP) == 4
    assert held_slot_count(positions) == 4
    ok, reason, _ = claim_symbol(db, "DOGE/USDT", SCALP, "dec-s5", 40.0, positions=positions)
    assert (ok, reason) == (False, ENGINE_MAX_POSITIONS)
    # Combined-cap firewall in isolation: the requesting engine is under its
    # own cap but the book (including DAY-side heritage lots that predate
    # engine stamps) is already at 8.
    heritage: dict = {}
    for i, sym in enumerate(TOP4):
        heritage[f"SCALP_V2::{sym}"] = _lot(sym, SCALP, trade_id=f"s{i}")
    for i, sym in enumerate(["DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT"]):
        lot = _lot(sym, "", trade_id=f"h{i}")
        lot.engine_id = ""
        heritage[sym] = lot
    assert held_slot_count(heritage) == 8
    assert held_slot_count(heritage, engine_id=DAY) == 0
    ok, reason, _ = claim_symbol(db, "BNB/USDT", DAY, "dec-d9", 40.0, positions=heritage)
    assert (ok, reason) == (False, MAX_COMBINED_POSITIONS)


# ---------------------------------------------------------------------------
# 6. Migration: composite PK, idempotent, economics preserved
# ---------------------------------------------------------------------------


def _legacy_row(symbol="BTC/USDT", qty=0.001, entry=84000.0, trade_id="leg-1"):
    return (symbol, qty, entry, 1790500000.0, trade_id, 0.0, 0.0, 0.0, 0.0, 0, entry, 0.0, 0, 0.0, "2026-09-27T00:00:00+00:00")


def test_migration_composite_pk_idempotent_and_preserves_economics(tmp_path):
    from backend.services.portfolio_engine import (
        _ensure_position_composite_pk,
        _ensure_position_engine_identity,
    )

    db = str(tmp_path / "mig.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE portfolio_engine_positions (
            symbol TEXT PRIMARY KEY, quantity REAL, entry_price REAL, entry_time REAL,
            trade_id TEXT, stop_price REAL, take_profit_1_price REAL, take_profit_2_price REAL,
            trailing_stop_price REAL, tp1_hit INTEGER, highest_price REAL, atr_at_entry REAL,
            entry_bar_timestamp INTEGER, confidence_at_entry REAL, last_updated TEXT)"""
    )
    conn.execute("INSERT INTO portfolio_engine_positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", _legacy_row())
    conn.commit()
    # Run twice: idempotent.
    _ensure_position_engine_identity(conn)
    conn.commit()
    _ensure_position_engine_identity(conn)
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
    assert "engine_id" in cols
    pk = {r[1] for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)") if r[5]}
    assert pk == {"engine_id", "symbol"}
    row = conn.execute("SELECT symbol, quantity, entry_price, trade_id, engine_id FROM portfolio_engine_positions").fetchone()
    assert row == ("BTC/USDT", 0.001, 84000.0, "leg-1", "LEGACY_DAY_LIVE")
    assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] == 1
    # Second engine lot on the same symbol now fits the PK.
    conn.execute("INSERT INTO portfolio_engine_positions (symbol, quantity, entry_price, trade_id, engine_id) VALUES ('BTC/USDT', 0.002, 85000.0, 'scalp-1', 'SCALP_V2')")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] == 2
    conn.close()


async def test_load_coexistence_legacy_bare_plus_scalp_composite(tmp_path):
    eng = _engine(tmp_path)
    legacy = _lot("BTC/USDT", "LEGACY_DAY_LIVE", qty=0.001, price=84000.0, trade_id="leg-1")
    await eng._persist_position_to_sqlite(legacy)
    scalp = _lot("BTC/USDT", SCALP, qty=0.002, price=85000.0, trade_id="scalp-1")
    await eng._persist_position_to_sqlite(scalp)

    eng2 = _engine(tmp_path)
    eng2.db_path = eng.db_path
    await eng2._load_positions_from_sqlite(allow_mutations=False)
    # Legacy heritage keeps the bare key; SCALP loads composite. No overwrite.
    assert "BTC/USDT" in eng2.open_positions
    assert make_position_key(SCALP, "BTC/USDT") in eng2.open_positions
    assert eng2.open_positions["BTC/USDT"].trade_id == "leg-1"
    assert eng2.open_positions[make_position_key(SCALP, "BTC/USDT")].trade_id == "scalp-1"
    assert eng2.open_positions["BTC/USDT"].entry_price == 84000.0
    assert eng2.open_positions[make_position_key(SCALP, "BTC/USDT")].entry_price == 85000.0


async def test_engine_scoped_delete_keeps_sibling(tmp_path):
    eng = _engine(tmp_path)
    await eng._persist_position_to_sqlite(_lot("BTC/USDT", SCALP, trade_id="s1"))
    await eng._persist_position_to_sqlite(_lot("BTC/USDT", DAY, trade_id="d1"))
    await eng._delete_position_from_sqlite("BTC/USDT", SCALP)
    conn = sqlite3.connect(str(eng.db_path))
    rows = conn.execute("SELECT trade_id, engine_id FROM portfolio_engine_positions").fetchall()
    conn.close()
    assert rows == [("d1", DAY)]


# ---------------------------------------------------------------------------
# 7. Ocean-shaped post-deploy invariant: SCALP=4 does not consume DAY slots
# ---------------------------------------------------------------------------


async def test_ocean_shape_scalp4_day_slots_open(tmp_path):
    eng = _engine(tmp_path)
    # Mirror Ocean: 3 ACTIVE SCALP + 1 DUST_PENDING SCALP (dust consumes nothing).
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = _lot("BTC/USDT", SCALP)
    eng.open_positions[make_position_key(SCALP, "ETH/USDT")] = _lot("ETH/USDT", SCALP)
    eng.open_positions[make_position_key(SCALP, "XRP/USDT")] = _lot("XRP/USDT", SCALP)
    eng.open_positions[make_position_key(SCALP, "SOL/USDT")] = _lot("SOL/USDT", SCALP, status="DUST_PENDING")
    assert eng._engine_held_count(SCALP) == 3
    assert eng._engine_held_count(DAY) == 0
    assert eng._combined_held_count() == 3
    ok, reason = await eng._can_open_position("BTC/USDT", 40.0, engine_id=DAY)
    assert ok is True, reason


# ---------------------------------------------------------------------------
# 8. Restart safety: composite keys must not read as orphans
# ---------------------------------------------------------------------------


async def test_restart_orphan_sync_keeps_composite_lots(tmp_path):
    """Regression: _detect_corruption compared composite memory keys against
    bare-symbol SQL tables and dropped every live lot from memory on restart
    (Ocean 2026-09-27: 4 SCALP lots dropped, then rewritten by restore with
    reset high-waters and blanked provenance)."""
    from backend.database_schema import initialize_paper_trading_schema

    eng = _engine(tmp_path)
    initialize_paper_trading_schema(str(tmp_path / "two_engine.db"))
    for sym, tid in (("BTC/USDT", "s1"), ("ETH/USDT", "s2")):
        pos = _lot(sym, SCALP, trade_id=tid)
        eng.open_positions[make_position_key(SCALP, sym)] = pos
        await eng._persist_position_to_sqlite(pos)
        with sqlite3.connect(str(tmp_path / "two_engine.db")) as conn:
            conn.execute(
                "INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (tid, "scalp_v2_live", "live", sym, "BUY", 0.01, 100.0, 0.01, "2026-09-27T00:00:00+00:00", "executed"),
            )
            conn.commit()
    corrupted, _ = await eng._detect_corruption()
    assert corrupted is False
    assert sorted(eng.open_positions) == ["SCALP_V2::BTC/USDT", "SCALP_V2::ETH/USDT"]


def test_prune_coin_performance_uses_symbols_not_keys(tmp_path):
    eng = _engine(tmp_path)
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = _lot("BTC/USDT", SCALP)
    assert eng._prune_coin_performance() is None  # must not raise; held coin kept by symbol
    from backend.services.portfolio_engine import CoinPerformance

    eng.coin_performance = {
        "BTC/USDT": CoinPerformance(symbol="BTC/USDT"),
        **{f"C{i}/USDT": CoinPerformance(symbol=f"C{i}/USDT") for i in range(310)},
    }
    eng._prune_coin_performance()
    assert "BTC/USDT" in eng.coin_performance
    assert len(eng.coin_performance) <= 300


# ---------------------------------------------------------------------------
# 9. Exit monitor: the live loop filters by engine keys
# ---------------------------------------------------------------------------


def _monitor_engine(tmp_path, monkeypatch) -> tuple[PortfolioEngine, AsyncMock]:
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._resolve_exit_monitor_mark = AsyncMock(return_value={"mark_used": 101.0})
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {"BTC/USDT": time.time()}
    check = AsyncMock(return_value=None)
    eng._check_exit_conditions = check
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = _lot("BTC/USDT", SCALP, price=100.0)
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = _lot("BTC/USDT", DAY, price=100.0)
    return eng, check


async def test_exit_monitor_evaluates_lots_filtered_by_engine_key(tmp_path, monkeypatch):
    """Regression: the integration exit loop passes composite keys as `symbols`;
    matching them against bare symbols skipped every lot (Ocean 2026-09-27,
    zero EXIT_CHECK_TELEMETRY after 1a867fb)."""
    eng, check = _monitor_engine(tmp_path, monkeypatch)
    keys = set(eng.open_positions)
    bundles = {k: {"engine": k} for k in keys}
    await eng.monitor_all_positions({}, 0, symbols=keys, hold_day_bundles=bundles, hold_day_missing={k: [] for k in keys})

    evaluated = {c.args[0].engine_id: c.kwargs for c in check.await_args_list}
    assert set(evaluated) == {SCALP, DAY}
    assert evaluated[SCALP]["day_hold_bundle"] == {"engine": "SCALP_V2::BTC/USDT"}
    assert evaluated[DAY]["day_hold_bundle"] == {"engine": "DAY_V2::BTC/USDT"}
    assert evaluated[SCALP]["day_hold_missing"] == []
    assert eng.open_positions["SCALP_V2::BTC/USDT"].highest_price == 101.0


async def test_exit_monitor_single_key_filter_and_bare_symbol_filter(tmp_path, monkeypatch):
    eng, check = _monitor_engine(tmp_path, monkeypatch)
    await eng.monitor_all_positions({}, 0, symbols={"SCALP_V2::BTC/USDT"})
    assert [c.args[0].engine_id for c in check.await_args_list] == [SCALP]

    check.reset_mock()
    await eng.monitor_all_positions({}, 0, symbols={"BTC/USDT"})
    assert sorted(c.args[0].engine_id for c in check.await_args_list) == [DAY, SCALP]


def _dust_engine(tmp_path, monkeypatch, sol_total: float, protected: float = 0.0):
    import backend.services.protected_external_inventory as pei

    monkeypatch.setattr(pei, "protected_quantity", lambda *_a, **_k: protected)
    eng = _engine(tmp_path)
    eng._live_execution_enabled = True
    eng._live_service = SimpleNamespace(get_balance=AsyncMock(return_value={"status": "success", "balance": {"total": {"SOL": sol_total}}}))
    eng._ensure_symbol_constraints = AsyncMock()
    eng._dust_check = lambda _s, qty, price: (qty * price < 5.0, 0, 0, 0)
    eng.open_positions[make_position_key(SCALP, "SOL/USDT")] = _lot("SOL/USDT", SCALP, qty=0.0131512, price=121.68, status="DUST_PENDING")
    return eng


async def test_dust_reconcile_reads_bare_symbol_from_engine_key(tmp_path, monkeypatch):
    """Regression: the composite key made the base coin 'SCALP_V2::SOL' (Ocean 2026-09-28)."""
    eng = _dust_engine(tmp_path, monkeypatch, sol_total=0.0131512)
    await eng.run_dust_reconciliation({"SOL/USDT": 121.7})
    lot = eng.open_positions[make_position_key(SCALP, "SOL/USDT")]
    eng._ensure_symbol_constraints.assert_awaited_with("SOL/USDT")
    assert lot.status == "DUST_PENDING"
    assert lot.dust_qty_canonical == pytest.approx(0.0131512)


async def test_dust_restore_excludes_sibling_and_protected_inventory(tmp_path, monkeypatch):
    eng = _dust_engine(tmp_path, monkeypatch, sol_total=0.30, protected=0.05)
    eng.open_positions[make_position_key(DAY, "SOL/USDT")] = _lot("SOL/USDT", DAY, qty=0.15, price=121.68)
    await eng.run_dust_reconciliation({"SOL/USDT": 121.7})
    lot = eng.open_positions[make_position_key(SCALP, "SOL/USDT")]
    assert lot.status == "ACTIVE"
    assert lot.quantity == pytest.approx(0.10)
    assert eng.open_positions[make_position_key(DAY, "SOL/USDT")].quantity == pytest.approx(0.15)
