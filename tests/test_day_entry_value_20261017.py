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


def test_fresh_entry_has_zero_excursion_tilt_while_open_position_uses_ridge(tmp_path):
    db = str(tmp_path / "ridge_fresh.db")
    install_day_surface(db)
    cost = canonical_roundtrip_cost_pct()
    for step in range(10):
        aged = cs.state_features(entry=100.0, mark=99.0, net=-0.01 - 0.001 * step, mfe=0.001, mae=0.01 + 0.001 * step, high_water=100.1, prev_net=-0.009, age_sec=600.0 + 60.0 * step)
        for h in cs.DAY_ADVANTAGE_HORIZONS:
            cs.record_advantage(db, engine="DAY_V2", symbol="ETHUSDT", setup="RANGE_BOUNCE", regime="other_reg", horizon=h, features=aged, advantage=-0.03, weight=1.0, now=T0 + step)
    teach_entry(db, "SOLUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", 0.004, times=2, now=T0 + 20)
    fresh = cs.state_features(entry=1.0, mark=1.0, net=-cost, mfe=0.0, mae=0.0, high_water=1.0, prev_net=None, age_sec=0.0)
    trace_fresh = cs.surface_trace(db, "DAY_V2", "SOLUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", fresh, T0 + 20, how="blended")
    assert trace_fresh is not None
    assert all(item["ridge"] == 0.0 for item in trace_fresh["horizons"])
    d = al.day_decision(db, "SOLUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 20)
    assert d["expected_net"] > 0.0


def test_regime_isolation_prevents_other_regime_losses_from_crushing_positive_regime_edge(tmp_path):
    db = str(tmp_path / "reg_iso.db")
    teach_entry(db, "BTCUSDT", "VWAP_REVERSION__MARKET", "btcup_volhi", -0.04, times=20, now=T0)
    teach_entry(db, "ETHUSDT", "RANGE_BOUNCE__MARKET", "btcdown_vollo", -0.04, times=20, now=T0)
    teach_entry(db, "SOLUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", 0.004, times=2, now=T0)
    pos_reg = al.day_decision(db, "SOLUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0)
    neg_reg = al.day_decision(db, "BTCUSDT", "VWAP_REVERSION__MARKET", "btcup_volhi", now=T0)
    assert pos_reg["expected_net"] > 0.0
    assert neg_reg["expected_net"] < 0.0


def test_age_zero_opportunity_rows_do_not_corrupt_intra_hold_ridge(tmp_path):
    db = str(tmp_path / "opp_ridge.db")
    install_day_surface(db)
    for i in range(50):
        opp_feat = cs.state_features(entry=100.0, mark=99.98, net=-0.00066, mfe=0.0, mae=0.0002, high_water=100.0, prev_net=None, age_sec=0.0)
        cs.record_advantage(db, engine="DAY_V2", symbol="SOLUSDT", setup="RANGE_BOUNCE__MARKET", regime="btcflat_volhi", horizon=3600, features=opp_feat, advantage=0.005, weight=1.0, now=T0 + i)
    ridge = cs._load_ridge(db, "DAY_V2")
    assert float((ridge.get("3600") or {}).get("n") or 0.0) == 0.0

    f_fresh = cs.state_features(entry=100.0, mark=99.98, net=-0.00066, mfe=0.0, mae=0.0002, high_water=100.0, prev_net=None, age_sec=0.0)
    f_half = cs.state_features(entry=100.0, mark=99.98, net=-0.00066, mfe=0.0, mae=0.0002, high_water=100.0, prev_net=None, age_sec=1800.0)
    f_elapsed = cs.state_features(entry=100.0, mark=99.98, net=-0.00066, mfe=0.0, mae=0.0002, high_water=100.0, prev_net=None, age_sec=3600.0)
    tr0 = cs.surface_trace(db, "DAY_V2", "SOLUSDT", "RANGE_BOUNCE__MARKET", "btcflat_volhi", f_fresh, T0 + 60)
    tr_half = cs.surface_trace(db, "DAY_V2", "SOLUSDT", "RANGE_BOUNCE__MARKET", "btcflat_volhi", f_half, T0 + 60)
    tr_end = cs.surface_trace(db, "DAY_V2", "SOLUSDT", "RANGE_BOUNCE__MARKET", "btcflat_volhi", f_elapsed, T0 + 60)
    assert tr0["advantage"] > 0.0
    assert all(item["ridge"] == 0.0 for item in tr0["horizons"])
    assert tr_half["advantage"] == pytest.approx(0.5 * tr0["advantage"], rel=1e-6)
    assert tr_end["advantage"] == pytest.approx(0.0, abs=1e-9)


def test_fold_recent_advantages_skips_open_positions_and_bounds_closed_position_weights(tmp_path, monkeypatch):
    db = str(tmp_path / "harm.db")
    install_day_surface(db)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT, engine_id TEXT, symbol TEXT, side TEXT,
            entry_price REAL, price REAL, entry_timestamp TEXT, timestamp TEXT,
            strategy_version TEXT, entry_contract_version TEXT, exit_contract_version TEXT,
            adaptive_decision_json TEXT, atr_at_entry REAL, exit_reason TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ai_position_heartbeats (
            epoch_ms INTEGER, ts_utc TEXT, trade_id TEXT, mark REAL,
            net_unrealized_pct REAL, mfe_pct REAL, mae_pct REAL,
            hold_seconds REAL, highest_since_entry REAL
        )
        """
    )
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL)")
    ver = ("DAY_V2_FIVE_SETUP_RUNNER_V1", "DAY_V2_ENTRY_PRIOR_STRUCTURE_V1", "DAY_V2_STRUCTURE_RUNNER_V1")
    iso0 = datetime.datetime.fromtimestamp(T0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        """
        INSERT INTO paper_trades (
            trade_id, engine_id, symbol, side, entry_price, price, entry_timestamp, timestamp,
            strategy_version, entry_contract_version, exit_contract_version, adaptive_decision_json, atr_at_entry
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        ("open_t1", "DAY_V2", "SOL/USDT", "BUY", 100.0, 100.0, iso0, iso0, *ver, json.dumps({"setup": "VWAP_REVERSION__MARKET", "regime": "btcup_vollo"}), 1.0),
    )
    for k in range(1, 6):
        ts_k = T0 + k * 300.0
        iso_k = datetime.datetime.fromtimestamp(ts_k, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute("INSERT INTO ai_position_heartbeats VALUES (?,?,?,?,?,?,?,?,?)", (int(ts_k * 1000), iso_k, "open_t1", 101.0, 0.009, 0.01, 0.0, k * 300.0, 101.0))
    for i in range(120):
        opened = T0 + i * 60
        conn.execute("INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)", ("SOL-USDT", "1m", _bar_ts(opened), 101.0, 101.0, 101.0, 101.0))
    conn.commit()

    # While open_t1 has no SELL row (not closed), fold_recent_advantages must not fold it
    monkeypatch.setattr(cb, "_LAST_FOLD", 0.0)
    assert cb.fold_recent_advantages(db, now=T0 + 3600.0) == 0

    # Once closed, incremental folds across time must never exceed 1/7 weight on h=900s
    iso_sell = datetime.datetime.fromtimestamp(T0 + 1800.0, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        """
        INSERT INTO paper_trades (
            trade_id, engine_id, symbol, side, entry_price, price, entry_timestamp, timestamp,
            strategy_version, entry_contract_version, exit_contract_version, adaptive_decision_json, atr_at_entry, exit_reason
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        ("sell_t1", "DAY_V2", "SOL/USDT", "SELL", 100.0, 101.0, iso0, iso_sell, *ver, json.dumps({"setup": "VWAP_REVERSION__MARKET", "regime": "btcup_vollo"}), 1.0, "LEARNED_CONTINUATION_EXIT"),
    )
    conn.commit()
    for step_t in (T0 + 2200.0, T0 + 2600.0, T0 + 3600.0):
        monkeypatch.setattr(cb, "_LAST_FOLD", 0.0)
        cb.fold_recent_advantages(db, now=step_t)
    sum_w_900 = conn.execute("SELECT SUM(weight) FROM continuation_advantage_log WHERE trade_id='open_t1' AND horizon=900").fetchone()[0]
    conn.close()
    assert sum_w_900 == pytest.approx(1.0 / len(cs.DAY_ADVANTAGE_HORIZONS), rel=1e-6)
