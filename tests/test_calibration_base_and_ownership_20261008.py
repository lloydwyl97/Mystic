"""Calibration learns the base forecast's residual; one exchange balance has one owner."""

from __future__ import annotations

import inspect
import os
import re
import sqlite3
import time

import pytest

from backend.services import adaptive_learning as al
from backend.services import day_rank_research as drr
from backend.services import economic_state_rebuild as esr
from backend.services.economic_replay import RepairedDayPolicy
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, _place_day_side_lot, make_position_key
from backend.services.protected_external_inventory import list_protected
from backend.services.scalp_v2.executable_edge import scalp_executable_edge

DAY = "DAY_V2"
SCALP = "SCALP_V2"
T0 = 1_700_000_000.0
SETUP, REGIME = "RANGE_BOUNCE__MARKET", "btcdown_vollo"


def _key_state(db: str) -> tuple[float, float] | None:
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()
    return None if row is None else (float(row[0]), float(row[1]))


def _close(db: str, engine: str, econ: dict, realized: float, opp: str, t: float) -> int:
    row_id = al.record_candidate(
        db,
        engine=engine,
        symbol="BTCUSDT",
        setup=SETUP,
        regime=REGIME,
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=t,
        economic=econ,
        opportunity_id=opp,
    )
    assert al.mark_candidate_filled(db, row_id)
    assert al.record_policy_outcome(db, engine=engine, opportunity_id=opp, net_pct=realized, candidate_id=row_id, now=t + 30.0)
    return row_id


def test_additive_calibration_learns_against_the_uncalibrated_entry_forecast(tmp_path):
    for engine in (DAY, SCALP):
        db = str(tmp_path / f"{engine}.db")
        base, applied, realized = 0.0020, -0.0020, -0.0005
        row_id = _close(db, engine, {"uncalibrated_policy_value": base, "policy_calibration": applied, "policy_value": base + applied}, realized, "B1", T0)
        n, ewma = _key_state(db)
        assert n == pytest.approx(1.0)
        assert ewma == pytest.approx(realized - base)
        assert ewma != pytest.approx(realized - (base + applied))
        with sqlite3.connect(db) as conn:
            stored = conn.execute("SELECT calibration_target, calibration_learned FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
        assert stored[0] == pytest.approx(realized - base) and int(stored[1]) == 1


def test_calibration_target_does_not_include_the_calibration_it_produced(tmp_path):
    """Constant base residual: the key's target never moves, whatever calibration was applied."""
    db = str(tmp_path / "fixed_point.db")
    base, realized = 0.0026, -0.0006
    for k in range(12):
        t = T0 + 900.0 * k
        applied = al.policy_calibration(db, DAY, "BTCUSDT", SETUP, REGIME, now=t)["mean"]
        _close(db, DAY, {"uncalibrated_policy_value": base, "policy_calibration": applied, "policy_value": base + applied}, realized, f"FP{k}", t)
        assert _key_state(db)[1] == pytest.approx(realized - base, abs=1e-15)
    shrunk = al.policy_calibration(db, DAY, "BTCUSDT", SETUP, REGIME, now=T0 + 900.0 * 12)["mean"]
    assert shrunk < 0 and abs(shrunk) > abs(realized - base) / 2.0


def test_entries_persist_every_component_of_the_forecast(tmp_path):
    db = str(tmp_path / "components.db")
    econ = al.day_decision(db, "BTCUSDT", SETUP, REGIME)["economic"]
    for key in ("market_alpha", "lifecycle_policy_estimate", "policy_gap", "uncalibrated_policy_value", "policy_calibration", "policy_value"):
        assert isinstance(econ[key], float)
    assert econ["policy_value"] == pytest.approx(econ["uncalibrated_policy_value"] + econ["policy_calibration"])
    assert econ["lifecycle_policy_estimate"] == pytest.approx(econ["market_alpha"] + econ["policy_gap"])
    view = al.scalp_decision(db, "ETHUSDT", "CLAIM", "r")
    view["policy_calibration"] = -0.0007
    scalp = scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.0001, impact_pct=0.0, edge_source="strategy_claim").economic()
    assert scalp["uncalibrated_policy_value"] == pytest.approx(scalp["policy_value"] - scalp["policy_calibration"])
    assert al.entry_base_forecast({"economic": scalp}) == pytest.approx(scalp["uncalibrated_policy_value"])


def test_entry_base_is_read_only_from_the_entry_fields():
    assert al.entry_base_forecast_of({"uncalibrated_policy_value": 0.003, "policy_calibration": -0.001, "policy_value": 0.002}) == pytest.approx(0.003)
    assert al.entry_base_forecast_of({"policy_calibration": -0.001, "policy_value": 0.002}) == pytest.approx(0.003)
    assert al.entry_base_forecast_of({"policy_value": 0.002}) == pytest.approx(0.002)
    assert al.entry_base_forecast_of({"policy_calibration": None, "policy_value": 0.002}) is None
    assert al.entry_base_forecast_of({"policy_value": True}) is None
    assert al.entry_base_forecast_of(None) is None


def test_a_crash_rolls_back_flag_target_and_observation_and_the_retry_writes_one(tmp_path):
    db = str(tmp_path / "crash.db")
    base, realized = 0.002, -0.001
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup=SETUP,
        regime=REGIME,
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=T0,
        economic={"uncalibrated_policy_value": base, "policy_calibration": -0.001, "policy_value": base - 0.001},
        opportunity_id="CRASH",
    )
    assert al.mark_candidate_filled(db, row_id)
    real = al._fold_observation

    def boom(conn, **kwargs):
        del conn, kwargs
        raise RuntimeError("crash")

    al._fold_observation = boom
    try:
        with pytest.raises(RuntimeError):
            al.record_policy_outcome(db, engine=DAY, opportunity_id="CRASH", net_pct=realized, candidate_id=row_id, now=T0)
    finally:
        al._fold_observation = real
    with sqlite3.connect(db) as conn:
        flag, target = conn.execute("SELECT calibration_learned, calibration_target FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
    assert int(flag) == 0 and target is None and _key_state(db) is None
    conn = al._connect(db)
    try:
        row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
        assert al._learn_policy_calibration(conn, db, row, realized, T0) is True
        row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
        assert al._learn_policy_calibration(conn, db, row, realized, T0) is False
    finally:
        conn.close()
    assert _key_state(db) == (pytest.approx(1.0), pytest.approx(realized - base))


def _sell(db: str, sold_at: str, entered_at: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS paper_trades (symbol TEXT, side TEXT, engine_id TEXT, timestamp TEXT, entry_timestamp TEXT, exit_reason TEXT)")
        conn.execute("INSERT INTO paper_trades VALUES (?,?,?,?,?,?)", ("BTC/USDT", "SELL", DAY, sold_at, entered_at, "LEARNED_CONTINUATION_EXIT"))


def test_the_rebuild_replaces_state_learned_against_the_calibrated_forecast(tmp_path):
    from datetime import datetime, timezone

    db = str(tmp_path / "rebuild.db")
    entered = datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc).timestamp()
    base, applied, realized = 0.0030, -0.0015, -0.0006
    proven = al.record_candidate(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup=SETUP,
        regime=REGIME,
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=entered,
        economic={"uncalibrated_policy_value": base, "policy_calibration": applied, "policy_value": base + applied},
        opportunity_id="P",
    )
    unproven = al.record_candidate(
        db,
        engine=DAY,
        symbol="ETHUSDT",
        setup=SETUP,
        regime=REGIME,
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=entered,
        economic={"policy_calibration": None, "policy_value": 0.001},
        opportunity_id="U",
    )
    for row_id in (proven, unproven):
        assert al.mark_candidate_filled(db, row_id)
    version = al.current_strategy_version(DAY)
    assert al.observe(db, engine=DAY, symbol="BTCUSDT", setup=SETUP, regime=REGIME, metric="policy_calibration", value=realized - (base + applied), strategy_version=version, now=entered)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=?, calibration_learned=1 WHERE id IN (?, ?)", (realized, proven, unproven))
        conn.execute("DELETE FROM adaptive_calibration_meta")
    _sell(db, "2026-10-07T20:01:00Z", "2026-10-07T20:00:05Z")
    al._calibration_backfilled.discard(os.path.abspath(db))
    al.backfill_close_calibration(db)
    assert _key_state(db) == (pytest.approx(1.0), pytest.approx(realized - base))
    with sqlite3.connect(db) as conn:
        rows = dict(conn.execute("SELECT id, calibration_learned FROM adaptive_candidate_markouts").fetchall())
        target = conn.execute("SELECT calibration_target FROM adaptive_candidate_markouts WHERE id=?", (proven,)).fetchone()[0]
        marker = conn.execute("SELECT value FROM adaptive_calibration_meta WHERE key='target'").fetchone()[0]
    assert rows[proven] == 1 and rows[unproven] == 0
    assert target == pytest.approx(realized - base) and marker == al.CALIBRATION_TARGET
    al._calibration_backfilled.discard(os.path.abspath(db))
    al.backfill_close_calibration(db)
    assert _key_state(db) == (pytest.approx(1.0), pytest.approx(realized - base))


def test_chronological_rebuild_reproduces_live_calibration(tmp_path):
    from datetime import datetime, timezone

    live = str(tmp_path / "live.db")
    scratch = str(tmp_path / "scratch.db")
    entered = datetime(2026, 10, 7, 16, 0, tzinfo=timezone.utc).timestamp()
    closed = entered + 45.0
    econ = {"uncalibrated_policy_value": 0.0031, "policy_calibration": -0.0016, "policy_value": 0.0015}
    decision = {"setup": SETUP, "regime": REGIME, "economic": econ}
    row_id = al.record_candidate(
        live, engine=DAY, symbol="XRPUSDT", setup=SETUP, regime=REGIME, ref_price=1.4, roundtrip_cost=0.00066, signaled=True, evaluated_at=entered - 15.0, economic=econ, opportunity_id="RB"
    )
    assert al.mark_candidate_filled(live, row_id)
    assert al.learn_from_close(
        live,
        engine=DAY,
        symbol="XRPUSDT",
        setup=SETUP,
        regime=REGIME,
        strategy_version=al.current_strategy_version(DAY),
        net_pct=-0.00067,
        mfe_pct=0.001,
        mae_pct=0.001,
        hold_min=0.75,
        continuation=None,
        version_current=True,
        is_dust=False,
        entered_at=entered,
        now=closed,
        opportunity_id="RB",
        exit_reason="LEARNED_CONTINUATION_EXIT",
        candidate_id=row_id,
    )
    al._connect(scratch).close()
    fill = {
        "symbol": "XRPUSDT",
        "setup": SETUP,
        "regime": REGIME,
        "decision": decision,
        "strategy_version": al.current_strategy_version(DAY),
        "entered_at": entered,
        "closed_at": closed,
        "entry_price": 1.4,
        "net": -0.00067,
        "mfe": 0.001,
        "mae": 0.001,
        "minutes": 0.75,
        "high": 1.41,
        "exit_reason": "LEARNED_CONTINUATION_EXIT",
    }

    class _Store:
        def minute_path(self, _symbol, _start, _end):
            return []

    rebuilt = esr.rebuild_day([], [fill], _Store(), RepairedDayPolicy(scratch), roundtrip_cost=0.00066, now=closed + 10.0)
    assert rebuilt["counts"]["policy_calibrations"] == 1
    assert _key_state(scratch) == (pytest.approx(_key_state(live)[0]), pytest.approx(_key_state(live)[1], abs=1e-15))
    assert _key_state(live)[1] == pytest.approx(-0.00067 - 0.0031)


def _engine(tmp_path) -> PortfolioEngine:
    eng = PortfolioEngine(db_path=str(tmp_path / "own.db"), principal=1000.0, test_mode=True)
    eng._ensure_db_schema()
    return eng


def _lot(symbol: str, qty: float, price: float, *, engine: str, trade_id: str, status: str = "ACTIVE") -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time(),
        trade_id=trade_id,
        stop_price=price * 0.98,
        take_profit_1_price=price * 1.01,
        take_profit_2_price=price * 1.02,
        engine_id=engine,
        status=status,
        original_position_cost=qty * price,
    )


COINS = (("BTC/USDT", 0.0028, 83000.0, 0.00001), ("ETH/USDT", 0.087, 2570.0, 0.0001), ("SOL/USDT", 1.9, 116.0, 0.0007), ("XRP/USDT", 228.05438, 1.4135, 0.05204))


@pytest.mark.parametrize("order", ["live_first", "bucket_first"])
async def test_reload_keeps_the_live_lot_and_the_dust_bucket(tmp_path, order):
    eng = _engine(tmp_path)
    live = _lot("XRP/USDT", 228.05438, 1.4135, engine="DAY_V2", trade_id="mystic_XRP/USDT_1791410415605")
    bucket = _lot("XRP/USDT", 0.05204, 1.4136, engine="LEGACY_DAY_LIVE", trade_id="dust_exchange:XRPUSDT", status="DUST_PENDING")
    for lot in (live, bucket) if order == "live_first" else (bucket, live):
        await eng._persist_position_to_sqlite(lot)
    eng.open_positions = {}
    await eng._load_positions_from_sqlite(allow_mutations=False)
    lots = eng._symbol_lots("XRP/USDT")
    assert sorted(p.trade_id for p in lots) == ["dust_exchange:XRPUSDT", "mystic_XRP/USDT_1791410415605"]
    assert eng.open_positions["XRP/USDT"].trade_id == "mystic_XRP/USDT_1791410415605"
    assert eng.open_positions[make_position_key("", "XRP/USDT")].trade_id == "dust_exchange:XRPUSDT"


@pytest.mark.parametrize(("symbol", "qty", "price", "dust"), COINS)
async def test_reconcile_right_after_a_buy_cannot_protect_the_bought_quantity(tmp_path, symbol, qty, price, dust):
    """The committed lot is not in the in-memory book yet: it still owns its coins."""
    eng = _engine(tmp_path)
    live = _lot(symbol, qty, price, engine="DAY_V2", trade_id=f"mystic_{symbol}_1")
    bucket = _lot(symbol, dust, price, engine="LEGACY_DAY_LIVE", trade_id=f"dust_exchange:{symbol.replace('/', '')}", status="DUST_PENDING")
    await eng._persist_position_to_sqlite(live)
    await eng._persist_position_to_sqlite(bucket)
    eng.open_positions = {make_position_key("", symbol): bucket}
    await eng._sync_protected_remainder(symbol, qty + dust, [bucket], 0.0)
    assert [r for r in list_protected(eng.db_path) if r["symbol"] == symbol] == []


async def test_true_unexplained_balance_is_protected_once_and_only_the_excess(tmp_path):
    eng = _engine(tmp_path)
    live = _lot("XRP/USDT", 228.05438, 1.4135, engine="DAY_V2", trade_id="mystic_XRP/USDT_1")
    await eng._persist_position_to_sqlite(live)
    eng.open_positions = {}
    await eng._sync_protected_remainder("XRP/USDT", 228.05438 + 50.0, [live], 0.0)
    rows = [r for r in list_protected(eng.db_path) if r["symbol"] == "XRP/USDT"]
    assert len(rows) == 1 and float(rows[0]["quantity"]) == pytest.approx(50.0)


async def test_import_waits_for_a_committed_lot_the_book_has_not_loaded(tmp_path):
    eng = _engine(tmp_path)
    live = _lot("XRP/USDT", 228.05438, 1.4135, engine="DAY_V2", trade_id="mystic_XRP/USDT_1")
    await eng._persist_position_to_sqlite(live)
    eng.open_positions = {}
    eng._live_execution_enabled = True
    eng._live_service = object()
    await eng._import_missing_exchange_positions({"XRP": 228.05438, "USDT": 700.0}, {"XRP": 228.05438, "USDT": 700.0})
    assert list_protected(eng.db_path) == []
    assert [p.trade_id for p in eng._symbol_lots("XRP/USDT")] == []


async def test_a_dust_bucket_overlapping_fill_backed_inventory_is_removed(tmp_path):
    eng = _engine(tmp_path)
    live = _lot("XRP/USDT", 228.05438, 1.4135, engine="DAY_V2", trade_id="mystic_XRP/USDT_1")
    bucket = _lot("XRP/USDT", 0.05204, 1.4136, engine="LEGACY_DAY_LIVE", trade_id="dust_exchange:XRPUSDT", status="DUST_PENDING")
    for lot in (live, bucket):
        await eng._persist_position_to_sqlite(lot)
    eng.open_positions = {}
    _place_day_side_lot(eng.open_positions, "XRP/USDT", live)
    _place_day_side_lot(eng.open_positions, "XRP/USDT", bucket)
    await eng._reconcile_dual_engine_lots(symbol="XRP/USDT", lots=[live, bucket], exchange_qty=228.05438, qty_step=0.01, source="test")
    assert [p.trade_id for p in eng._symbol_lots("XRP/USDT")] == ["mystic_XRP/USDT_1"]
    assert live.quantity == pytest.approx(228.05438)
    assert [r[0] for r in eng._committed_lot_rows("XRP/USDT")] == ["mystic_XRP/USDT_1"]


def test_relative_ranking_uses_only_labels_that_are_due():
    rows = []
    for i, t in enumerate((0.0, 900.0)):
        rows.append({"id": 10 * i + 1, "t": t, "symbol": "BTCUSDT", "setup": "S", "regime": "r", "y": 0.01 if t == 0.0 else -0.01})
        rows.append({"id": 10 * i + 2, "t": t, "symbol": "ETHUSDT", "setup": "S", "regime": "r", "y": -0.01 if t == 0.0 else 0.01})
    late = drr.walk_forward(rows, outcome=lambda c: c["y"], label_delay_sec=901.0)
    assert set(late[1].scores.values()) == {0.0}
    on_time = drr.walk_forward(rows, outcome=lambda c: c["y"], label_delay_sec=900.0)
    assert on_time[1].scores[11] > on_time[1].scores[12]
    assert on_time[0].scores == {1: 0.0, 2: 0.0}


def test_no_trade_opinion_was_added():
    import backend.services.portfolio_engine as pe

    src = "".join(
        inspect.getsource(f)
        for f in (
            al.entry_base_forecast_of,
            al.observe_policy_calibration,
            al._learn_policy_calibration,
            al.rebuild_policy_calibration,
            al.policy_calibration,
            pe._place_day_side_lot,
            pe.PortfolioEngine._proven_lot_qty,
            pe.PortfolioEngine._shrink_unproven_dust_bucket,
            drr.walk_forward,
        )
    )
    for word in ("rsi", "atr", "min_trades", "min_hold", "blacklist", "54.7", "regime_gate", "max_hold"):
        assert re.search(rf"\b{re.escape(word)}\b", src, re.IGNORECASE) is None, word
