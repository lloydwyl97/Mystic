"""2026-09-29 economic pass: re-entry dust ownership, SCALP entry context, leak-free research splits."""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from backend.services import engine_strategy_dust as esd
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.scalp_v2.entry_context import CONTEXT_KEY, build_entry_context, persist_entry_context
from scripts.research import day_clean_model_validation as dcv
from scripts.research import scalp_economic_forensics as sef
from tests.test_atomic_execution_and_crash_injection import _commit, _init_engine, _position

# ------------------------------------------------------ re-entry dust ownership


def _scalp_write(engine: PortfolioEngine, conn: sqlite3.Connection, trade_id: str, qty: float) -> None:
    engine._scalp_v2_write_position_row(
        conn,
        symbol="XRP/USDT",
        quantity=qty,
        fill_price=1.55,
        fee=0.01,
        order_id=f"o_{trade_id}",
        atr=0.0,
        opportunity_id=f"opp_{trade_id}",
        decision_id="",
        reservation_id="",
        client_order_id="",
        trade_id=trade_id,
        entry_time=time.time(),
        timestamp="2026-09-29T14:42:38+00:00",
    )


def test_scalp_v2_reentry_moves_own_dust_to_held(tmp_path):
    db = tmp_path / "s.db"
    engine = _init_engine(db)
    with sqlite3.connect(db) as conn:
        _scalp_write(engine, conn, "scalp_v2_XRPUSDT_old", 32.29354)
        conn.execute("UPDATE portfolio_engine_positions SET quantity=0.09354, status='DUST_PENDING' WHERE engine_id='SCALP_V2'")
        _scalp_write(engine, conn, "scalp_v2_XRPUSDT_new", 31.9936)
        row = conn.execute("SELECT trade_id, quantity, status FROM portfolio_engine_positions WHERE engine_id='SCALP_V2' AND symbol='XRP/USDT'").fetchone()
        held = esd.held_lots(conn, "XRP/USDT")
    assert row == ("scalp_v2_XRPUSDT_new", 31.9936, "ACTIVE")
    assert [(h["engine_id"], h["source_trade_id"], h["quantity"]) for h in held] == [("SCALP_V2", "scalp_v2_XRPUSDT_old", 0.09354)]


def test_scalp_v2_rewrite_of_same_active_lot_holds_nothing(tmp_path):
    db = tmp_path / "s2.db"
    engine = _init_engine(db)
    with sqlite3.connect(db) as conn:
        _scalp_write(engine, conn, "scalp_v2_XRPUSDT_a", 10.0)
        _scalp_write(engine, conn, "scalp_v2_XRPUSDT_b", 11.0)
        assert esd.held_lots(conn, "XRP/USDT") == []


def test_day_atomic_reentry_moves_own_dust_to_held(tmp_path):
    db = tmp_path / "d.db"
    engine = _init_engine(db)
    first = _position(trade_id="mystic_XRP/USDT_old", qty=100.0)
    first.engine_id = "DAY_V2"
    _commit(engine, first, cash=9900.0, positions_value=100.0)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE portfolio_engine_positions SET quantity=0.08804, status='DUST_PENDING' WHERE engine_id='DAY_V2' AND symbol='XRP/USDT'")
    second = _position(trade_id="mystic_XRP/USDT_new", qty=50.0)
    second.engine_id = "DAY_V2"
    _commit(engine, second, cash=9850.0, positions_value=50.0)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT trade_id, status FROM portfolio_engine_positions WHERE engine_id='DAY_V2' AND symbol='XRP/USDT'").fetchone()
        held = esd.held_lots(conn, "XRP/USDT")
    assert row == ("mystic_XRP/USDT_new", "ACTIVE")
    assert [(h["engine_id"], h["source_trade_id"], h["quantity"]) for h in held] == [("DAY_V2", "mystic_XRP/USDT_old", 0.08804)]


def test_day_reentry_never_takes_scalp_dust(tmp_path):
    db = tmp_path / "d2.db"
    engine = _init_engine(db)
    with sqlite3.connect(db) as conn:
        _scalp_write(engine, conn, "scalp_v2_XRPUSDT_old", 0.09354)
        conn.execute("UPDATE portfolio_engine_positions SET status='DUST_PENDING' WHERE engine_id='SCALP_V2'")
    day = _position(trade_id="mystic_XRP/USDT_new", qty=50.0)
    day.engine_id = "DAY_V2"
    _commit(engine, day, cash=9950.0, positions_value=50.0)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status, quantity FROM portfolio_engine_positions WHERE engine_id='SCALP_V2'").fetchone() == ("DUST_PENDING", 0.09354)
        assert esd.held_lots(conn, "XRP/USDT") == []


# ------------------------------------------------------ SCALP entry context


def _router_row():
    return {
        "rank_score": 0.0012,
        "strategy_passed": True,
        "entry_eligible": True,
        "best_setup": "range_bounce_scalp",
        "soft_reason": "MTF_5M_NOT_ALIGNED_RANKED",
        "EV_10s": -0.0001,
        "rank_components": {"EV_10s": -0.0001, "selection_version": "v"},
        "rank_meta": {"net_edge_after_costs_pct": 0.00525, "expected_move_pct": 0.0056, "mtf_5m_aligned": False, "regime": "range"},
        "signal": SimpleNamespace(score=0.7, confidence=0.6, spread_pct=0.0001, impact_pct=0.0, expected_move_pct=0.0056, required_target_pct=0.004, depth_sufficient=True),
        "snap": SimpleNamespace(best_bid=2687.9, best_ask=2688.0, mid=2687.95, spread_pct=0.00004),
    }


def test_entry_context_captures_rank_edge_and_book():
    ctx = build_entry_context(_router_row(), cycle_ts=123.0)
    assert ctx["net_edge_after_costs_pct"] == 0.00525
    assert ctx["strategy_passed"] is True
    assert ctx["mtf_5m_aligned"] is False
    assert ctx["signal_impact_pct"] == 0.0
    assert ctx["book_best_ask"] == 2688.0
    assert ctx["rank_components"]["EV_10s"] == -0.0001
    json.dumps(ctx)


def _paper_db(tmp_path) -> str:
    db = str(tmp_path / "p.db")
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, order_id TEXT, engine_id TEXT, side TEXT, explainability_json TEXT)")
        c.execute("INSERT INTO paper_trades VALUES (1,'o1','SCALP_V2','BUY',NULL), (2,'o2','DAY_V2','BUY','{\"setup_type\":\"RANGE_BOUNCE\"}'), (3,'o3','SCALP_V2','BUY','{\"x\":1}')")
    return db


def test_entry_context_persists_once_on_scalp_buy_only(tmp_path):
    db = _paper_db(tmp_path)
    ctx = build_entry_context(_router_row(), cycle_ts=1.0)
    assert persist_entry_context(db, order_id="o1", context=ctx) is True
    assert persist_entry_context(db, order_id="o1", context={"other": 1}) is False
    assert persist_entry_context(db, order_id="o2", context=ctx) is False
    assert persist_entry_context(db, order_id="o3", context=ctx) is True
    with sqlite3.connect(db) as c:
        rows = {r[0]: json.loads(r[1]) for r in c.execute("SELECT id, explainability_json FROM paper_trades")}
    assert rows[1][CONTEXT_KEY]["net_edge_after_costs_pct"] == 0.00525
    assert rows[2] == {"setup_type": "RANGE_BOUNCE"}
    assert rows[3]["x"] == 1 and CONTEXT_KEY in rows[3]
    assert "setup_type" not in rows[1]


# ------------------------------------------------------ leak-free research


def test_scalp_folds_are_chronological_and_disjoint():
    folds = sef.expanding_folds(96)
    assert len(folds) == 4
    for fit, ev in folds:
        assert not set(fit) & set(ev)
        assert max(fit) < min(ev)
    assert folds[-1][1].stop == 96


def test_tie_corrected_auc():
    assert sef.auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    assert sef.auc([2.0, 3.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_scaled_net_respects_min_notional_and_never_upsizes():
    t = {"net": -0.4, "notional": 40.0}
    assert sef.scaled_net(t, 0.5, 1.0) == pytest.approx(-0.2)
    assert sef.scaled_net(t, 1.0, 1.0) == -0.4
    assert sef.scaled_net({"net": -0.4, "notional": 1.5}, 0.5, 1.0) == -0.4


def test_candidate_rules_see_only_fit_rows():
    fit = [{"net": -1.0, "mom_60s": -0.001}] * 6 + [{"net": 1.0, "mom_60s": 0.001}] * 6
    size = sef.CANDIDATES["C2_MOMENTUM_60S_WORSE_SIDE"](fit)
    assert size({"mom_60s": -0.002}) == sef.MIN_SIZE
    assert size({"mom_60s": 0.002}) == 1.0
    assert size({}) == 1.0


def test_day_purged_split_drops_train_rows_closing_inside_validation():
    rows = [
        {"id": 1, "opened": 0, "closed": 5},
        {"id": 2, "opened": 6, "closed": 25},
        {"id": 3, "opened": 20, "closed": 22},
        {"id": 4, "opened": 30, "closed": 31},
    ]
    train, val = dcv.purged_split(rows, 20, 40)
    assert [r["id"] for r in val] == [3, 4]
    assert [r["id"] for r in train] == [1]


def test_day_exclusions_drop_scalp_dust_and_corrections():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE paper_trades (trade_id TEXT, side TEXT, engine_id TEXT)")
    conn.execute("INSERT INTO paper_trades VALUES ('mystic_scalp_like','BUY','SCALP_V2')")

    def row(ctx, comps=None):
        return {"context_json": json.dumps(ctx), "score_components_json": json.dumps(comps or {})}

    assert dcv.exclusion_reason(conn, row({"is_dust": True})) == "dust"
    assert dcv.exclusion_reason(conn, row({"trade_id": "scalp_v2_XRPUSDT_1"})) == "scalp_trade"
    assert dcv.exclusion_reason(conn, row({"trade_id": "mystic_scalp_like"})) == "non_day_engine"
    assert dcv.exclusion_reason(conn, row({"engine_id": "DAY_V2"}, {"close_reason": "MANUAL_UNMATCHED"})) == "manual_or_correction"
    assert dcv.exclusion_reason(conn, row({"engine_id": "DAY_V2"}, {"close_reason": "NET_PROFIT_EXIT"})) is None
