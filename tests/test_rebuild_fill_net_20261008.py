"""A rebuild replays each close at its venue fill net, as the live close learner now does."""

from __future__ import annotations

import json
import sqlite3
import time

import backend.services.adaptive_learning as al
import backend.services.economic_state_rebuild as esr

SCALP = al.SCALP_ENGINE
KEY = ("ETHUSDT", "VWAP_EMA_RECLAIM", "btcdown_vollo")
FLAT_TAUGHT = -0.00066
FILL_NET = -0.0005060746009779401


def _history(db: str, *, with_fill_column: bool) -> None:
    entered = al.anchor_epoch(SCALP) + 86400.0
    closed = entered + 1.0
    extra = json.dumps({"version_current": True, "original_trade_id": "b1"})
    fill_col = ", pnl_pct_net REAL" if with_fill_column else ""
    decision = json.dumps({"setup": KEY[1], "regime": KEY[2]})
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(closed))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, engine_id TEXT, symbol TEXT, setup TEXT, strategy_version TEXT, entry_timestamp REAL, "
            "exit_timestamp REAL, net_profit_pct REAL, close_reason TEXT, hold_seconds REAL, extra_json TEXT, indicators_while_holding_json TEXT)"
        )
        conn.execute(
            "CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, trade_id TEXT, decision_id TEXT, engine_id TEXT, symbol TEXT, side TEXT, timestamp TEXT, "
            f"scalp_opportunity_id TEXT, adaptive_decision_json TEXT{fill_col})"
        )
        conn.execute(
            "INSERT INTO trade_learning_outcomes (engine_id, symbol, setup, strategy_version, entry_timestamp, exit_timestamp, net_profit_pct, close_reason, "
            "hold_seconds, extra_json, indicators_while_holding_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (SCALP, KEY[0], KEY[1], al.current_strategy_version(SCALP), entered, closed, FLAT_TAUGHT, "LEARNED_CONTINUATION_EXIT", 1.0, extra, "{}"),
        )
        conn.execute(
            "INSERT INTO paper_trades (trade_id, decision_id, engine_id, symbol, side, timestamp, scalp_opportunity_id, adaptive_decision_json) VALUES (?,?,?,?,?,?,?,?)",
            ("b1", "d1", SCALP, KEY[0], "BUY", stamp, "opp", decision),
        )
        if with_fill_column:
            conn.execute(
                "INSERT INTO paper_trades (trade_id, decision_id, engine_id, symbol, side, timestamp, scalp_opportunity_id, adaptive_decision_json, pnl_pct_net) VALUES (?,?,?,?,?,?,?,?,?)",
                ("s1", "d1", SCALP, KEY[0], "SELL", stamp, "opp", decision, FILL_NET),
            )
        else:
            conn.execute(
                "INSERT INTO paper_trades (trade_id, decision_id, engine_id, symbol, side, timestamp, scalp_opportunity_id, adaptive_decision_json) VALUES (?,?,?,?,?,?,?,?)",
                ("s1", "d1", SCALP, KEY[0], "SELL", stamp, "opp", decision),
            )


def test_realized_close_replays_the_sell_fill_net(tmp_path):
    db = str(tmp_path / "t.db")
    _history(db, with_fill_column=True)
    closes = esr.realized_closes(db, SCALP)
    assert len(closes) == 1
    assert closes[0]["net"] == FILL_NET


def _calibrated_history(db: str) -> float:
    """One live SCALP claim with a stored base forecast, filled and closed at its fill net."""
    from backend.services.strategy_version import exit_policy_anchor

    t = float(exit_policy_anchor(SCALP)["epoch"]) + 3600.0
    _history(db, with_fill_column=True)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM trade_learning_outcomes")
        conn.execute("DELETE FROM paper_trades")
    econ = {"policy_value": 0.0004, "policy_calibration": 0.0, "uncalibrated_policy_value": 0.0004}
    row = al.record_candidate(
        db,
        engine=SCALP,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        ref_price=100.0,
        roundtrip_cost=0.0006,
        signaled=True,
        evaluated_at=t,
        raw_expected_move=0.002,
        raw_move_source="STRATEGY_CLAIM",
        economic=econ,
    )
    assert al.link_candidate_fill(db, row, "opp")
    closed = t + 2.0
    assert al.learn_from_close(
        db,
        engine=SCALP,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        strategy_version=al.current_strategy_version(SCALP),
        net_pct=FILL_NET,
        mfe_pct=None,
        mae_pct=None,
        hold_min=0.03,
        continuation=None,
        version_current=True,
        is_dust=False,
        entered_at=t + 1.0,
        now=closed,
        opportunity_id="opp",
        exit_reason="LEARNED_CONTINUATION_EXIT",
        candidate_id=row,
    )
    decision = json.dumps({"setup": KEY[1], "regime": KEY[2], "lineage": {"candidate_id": row}})
    extra = json.dumps({"version_current": True, "original_trade_id": "b1"})
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO trade_learning_outcomes (engine_id, symbol, setup, strategy_version, entry_timestamp, exit_timestamp, net_profit_pct, close_reason, "
            "hold_seconds, extra_json, indicators_while_holding_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (SCALP, KEY[0], KEY[1], al.current_strategy_version(SCALP), t + 1.0, closed, FILL_NET, "LEARNED_CONTINUATION_EXIT", 1.0, extra, "{}"),
        )
        for trade_id, side, at in (("b1", "BUY", t + 1.0), ("s1", "SELL", closed)):
            conn.execute(
                "INSERT INTO paper_trades (trade_id, decision_id, engine_id, symbol, side, timestamp, scalp_opportunity_id, adaptive_decision_json, pnl_pct_net) VALUES (?,?,?,?,?,?,?,?,?)",
                (trade_id, "d1", SCALP, KEY[0], side, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(at)), "opp", decision, FILL_NET if side == "SELL" else None),
            )
    return t + 3600.0


def test_regeneration_relearns_policy_calibration(tmp_path):
    """Ocean 2026-10-09 dry run: the replay copied calibration_learned=1 from production
    rows, so every policy_calibration observation was dropped from the rebuilt state."""
    db = str(tmp_path / "t.db")
    now = _calibrated_history(db)
    out = esr.regenerate(db, SCALP, now=now, workdir=str(tmp_path))
    calibration = out["metrics"]["policy_calibration"]
    assert calibration["before_n"] > 0
    assert abs(calibration["before_n"] - calibration["after_n"]) < 1e-9


def test_realized_close_without_a_fill_net_keeps_the_recorded_net(tmp_path):
    db = str(tmp_path / "t.db")
    _history(db, with_fill_column=False)
    closes = esr.realized_closes(db, SCALP)
    assert len(closes) == 1
    assert closes[0]["net"] == FLAT_TAUGHT
