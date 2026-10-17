"""DAY entry value is the continuation surface's value of the just-filled position.

The entry decision and the first continuation check read one quantity, and every
refused executable DAY opportunity teaches that surface while DAY is flat.
"""

from __future__ import annotations

import datetime
import json
import sqlite3

import pytest

import backend.services.adaptive_learning as al
import backend.services.continuation_backfill as cb
import backend.services.continuation_surface as cs
from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services.day_v2.day_policy_decision import decide_day_continuation
from backend.services.exact_policy_capture import ensure_capture_schema
from tests.day_entry_value_support import install_day_surface, teach_entry

T0 = 1_790_900_100.0
KEY = ("BTCUSDT", "RANGE_BOUNCE", "neutral")


def _bar_ts(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def test_cold_entry_value_is_minus_cost_and_does_not_abstain(tmp_path):
    db = str(tmp_path / "cold.db")
    cost = canonical_roundtrip_cost_pct()
    entry = al.day_entry_value(db, *KEY, cost=cost, now=T0)
    assert entry["value"] == pytest.approx(-cost) and entry["informed"] is False
    d = al.day_decision(db, *KEY, now=T0)
    assert d["expected_net"] == pytest.approx(-cost)
    assert d["abstain"] is False
    assert d["economic"]["policy_base"] == al.DAY_ENTRY_VALUE_BASE == "continuation_value"
    assert d["economic"]["continuation_entry_informed"] is False


def test_entry_value_equals_the_first_continuation_check_at_fill(tmp_path):
    db = str(tmp_path / "coherent.db")
    cost = canonical_roundtrip_cost_pct()
    teach_entry(db, *KEY, 0.004, times=3, now=T0)
    teach_entry(db, "ETHUSDT", "RANGE_BOUNCE", "neutral", -0.006, times=2, now=T0)
    d = al.day_decision(db, *KEY, now=T0)
    check = decide_day_continuation(
        db,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        entry_price=100.0,
        mark=100.0,
        low=100.0,
        high=100.0,
        prev_net=None,
        age_sec=0.0,
        atr_at_entry=0.0,
        structural_anchor=0.0,
        target_price=0.0,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0,
    )
    assert check["net"] == pytest.approx(-cost)
    assert d["expected_net"] == pytest.approx(check["terminal"])
    assert d["economic"]["continuation_entry_advantage"] == pytest.approx(check["advantage"])
    assert d["economic"]["continuation_entry_how"] == cs.installed_aggregator(db, "DAY_V2")
    assert d["economic"]["policy_value"] == pytest.approx(d["expected_net"])


def test_realized_close_posterior_is_a_diagnostic_not_the_entry_value(tmp_path):
    db = str(tmp_path / "diag.db")
    teach_entry(db, *KEY, 0.002, now=T0)
    before = al.day_decision(db, *KEY, now=T0)
    for _ in range(5):
        al.observe(db, engine=al.DAY_ENGINE, symbol=KEY[0], setup=KEY[1], regime=KEY[2], metric="trade_net", value=-0.02, strategy_version=al.current_strategy_version(al.DAY_ENGINE), now=T0)
    after = al.day_decision(db, *KEY, now=T0)
    assert after["expected_net"] == pytest.approx(before["expected_net"])
    assert after["economic"]["trade_net_value"] < before["economic"]["trade_net_value"]


def _opportunity_db(tmp_path) -> str:
    db = str(tmp_path / "opp.db")
    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL)")
    conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, trade_id TEXT, engine_id TEXT, symbol TEXT, side TEXT, entry_timestamp TEXT, timestamp TEXT)")
    rows = [
        (1, "BTCUSDT", "RANGE_BOUNCE", T0, 100.0, 99.98, 0),
        (2, "BTCUSDT", "BREAKOUT_CONTINUATION", T0 + 5, 100.0, 99.98, 0),
        (3, "BTCUSDT", "VWAP_REVERSION", T0 + 900, 100.0, 99.98, 1),
        (4, "BTCUSDT", "HTF_TREND_PULLBACK", T0 + 1800, 100.0, 99.98, None),
        (5, "BTCUSDT", "RANGE_BOUNCE", T0 + 2700, 100.0, 100.5, 0),
    ]
    for cid, symbol, setup, decided, ask, bid, funded in rows:
        conn.execute(
            """
            INSERT INTO policy_episodes (candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask,
                roundtrip_cost, snapshot_json, status, funded, capture_version, entry_bid)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (cid, "COUNTERFACTUAL", "DAY_V2", symbol, setup, "neutral", decided, ask, 0.00066, json.dumps({"atr_15m": 0.4}), "OPEN", funded, "EXACT_POLICY_V1", bid),
        )
    for i in range(240):
        opened = T0 - 60 + i * 60
        conn.execute("INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)", ("BTC-USDT", "1m", _bar_ts(opened), 101.0, 101.0, 101.0, 101.0))
    conn.commit()
    conn.close()
    return db


def test_only_refused_executable_opportunities_become_fresh_positions(tmp_path):
    db = _opportunity_db(tmp_path)
    found = cb.opportunity_positions(db, since=T0 - 1)
    assert sorted(p.setup for p, _share in found) == ["BREAKOUT_CONTINUATION", "RANGE_BOUNCE"]
    for position, share in found:
        assert share == pytest.approx(0.5)
        assert position.trade_id.startswith(cb.OPPORTUNITY_PREFIX)
        assert position.entry_price == 100.0
        (snap,) = position.snapshots
        assert snap.mark == 99.98 and snap.hold_sec == 0.0 and snap.mfe == 0.0


def test_opportunity_labels_fold_once_with_path_weight(tmp_path, monkeypatch):
    db = _opportunity_db(tmp_path)
    install_day_surface(db)
    monkeypatch.setattr(cb, "_LAST_FOLD", 0.0)
    as_of = T0 + 2 * 3600 + 60
    folded = cb.fold_recent_advantages(db, now=as_of)
    matured = [h for h in cs.horizons_for("DAY_V2") if h <= 2 * 3600]
    assert folded == 2 * len(matured)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT trade_id, horizon, advantage, weight FROM continuation_advantage_log ORDER BY trade_id, horizon").fetchall()
    assert len(rows) == folded
    for trade_id, horizon, advantage, weight in rows:
        assert trade_id.startswith(cb.OPPORTUNITY_PREFIX) and horizon in matured
        assert weight == pytest.approx(0.5 / len(cs.horizons_for("DAY_V2")))
        assert advantage > 0.0
    monkeypatch.setattr(cb, "_LAST_FOLD", 0.0)
    assert cb.fold_recent_advantages(db, now=as_of + 120) == 0
    d = al.day_decision(db, *KEY, now=as_of)
    assert d["economic"]["continuation_entry_informed"] is True
    assert d["expected_net"] > -canonical_roundtrip_cost_pct()
