"""DAY live entry follows realized policy net. Account equity counts owned dust once."""

from __future__ import annotations

import json
import sqlite3

import pytest

import backend.services.adaptive_learning as al
from backend.services.engine_strategy_dust import held_inventory_equity

DAY = al.DAY_ENGINE
KEY = ("BTCUSDT", "RANGE_BOUNCE", "btcup_vollo")
VER = al.current_strategy_version(DAY)


def _obs(db, metric, value):
    assert al.observe(db, engine=DAY, symbol=KEY[0], setup=KEY[1], regime=KEY[2], metric=metric, value=value, strategy_version=VER, now=1_800_000_000.0)


def test_lifecycle_and_policy_gap_do_not_fund_a_day_entry(tmp_path):
    db = str(tmp_path / "t.db")
    for _ in range(12):
        _obs(db, "lifecycle_net", 0.02)
        _obs(db, "policy_gap", 0.01)
        _obs(db, "policy_calibration", -0.003)
    decision = al.day_decision(db, *KEY)
    assert decision["expected_net"] == 0.0
    assert decision["economic"]["policy_gap"] > 0.0
    assert decision["economic"]["market_alpha"] > 0.0
    assert decision["economic"]["policy_base"] == "trade_net"


def test_trade_net_sets_the_base_and_its_own_calibration_adjusts_it(tmp_path):
    db = str(tmp_path / "t.db")
    for _ in range(6):
        _obs(db, "trade_net", -0.0004)
    base = al.day_net_expectancy(db, *KEY)
    assert base["uncalibrated_mean"] < 0.0
    assert base["policy_calibration"] == 0.0
    for _ in range(6):
        _obs(db, "trade_net_calibration", -0.0002)
    after = al.day_net_expectancy(db, *KEY)
    assert after["mean"] == pytest.approx(after["uncalibrated_mean"] + after["policy_calibration"])
    assert after["mean"] < after["uncalibrated_mean"] < 0.0


def test_only_a_trade_net_base_trains_the_live_calibration(tmp_path):
    db = str(tmp_path / "t.db")
    moment = 1_800_000_100.0
    legacy = al.record_candidate(
        db,
        engine=DAY,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=moment,
        economic={"uncalibrated_policy_value": 0.004, "policy_value": 0.001, "policy_calibration": -0.003},
    )
    aligned = al.record_candidate(
        db,
        engine=DAY,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=moment + 10,
        economic={"policy_base": "trade_net", "uncalibrated_policy_value": -0.0004, "policy_value": -0.0004, "policy_calibration": 0.0},
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        for row_id in (legacy, aligned):
            conn.execute("UPDATE adaptive_candidate_markouts SET filled=1, realized_net=? WHERE id=?", (-0.0005, row_id))
        conn.commit()
        legacy_row = dict(conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (legacy,)).fetchone())
        aligned_row = dict(conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (aligned,)).fetchone())
    conn = al._connect(db)
    try:
        assert al._learn_policy_calibration(conn, db, legacy_row, -0.0005, moment)
        assert al._learn_policy_calibration(conn, db, aligned_row, -0.0005, moment + 10)
        conn.commit()
    finally:
        conn.close()
    legacy_cal = al.policy_calibration(db, DAY, *KEY, now=moment + 20)["mean"]
    live = al.day_net_expectancy(db, *KEY, now=moment + 20)
    assert legacy_cal != 0.0
    assert live["policy_calibration"] < 0.0
    assert abs(live["policy_calibration"]) < 0.0001
    assert live["policy_gap"] == 0.0


def test_held_dust_is_valued_once_and_an_open_lot_is_not_repeated(tmp_path):
    db = str(tmp_path / "d.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE engine_strategy_dust (
                id INTEGER PRIMARY KEY, engine_id TEXT, symbol TEXT, source_trade_id TEXT,
                quantity REAL, quantity_exact TEXT, entry_price REAL, provenance_json TEXT,
                status TEXT, retired_class TEXT, retired_ref TEXT, created_at TEXT, retired_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO engine_strategy_dust (engine_id, symbol, source_trade_id, quantity, entry_price, status, created_at) VALUES (?,?,?,?,?,?,?)",
            ("DAY_V2", "BTC/USDT", "open-lot", 0.01, 80000.0, "HELD", "2026-10-09T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO engine_strategy_dust (engine_id, symbol, source_trade_id, quantity, entry_price, status, created_at) VALUES (?,?,?,?,?,?,?)",
            ("DAY_V2", "ETH/USDT", "held-only", 0.5, 2400.0, "HELD", "2026-10-09T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO engine_strategy_dust (engine_id, symbol, source_trade_id, quantity, entry_price, status, created_at) VALUES (?,?,?,?,?,?,?)",
            ("DAY_V2", "SOL/USDT", "retired", 2.0, 100.0, "RETIRED", "2026-10-09T00:00:00Z"),
        )
    prices = {"BTC/USDT": 82000.0, "ETH/USDT": 2500.0, "SOL/USDT": 110.0}
    market, cost = held_inventory_equity(db, prices, {"open-lot"})
    assert market == pytest.approx(0.5 * 2500.0)
    assert cost == pytest.approx(0.5 * 2400.0)
    assert "SOL" not in json.dumps({"market": market})
