"""Hold-advantage continuation: incremental labels, no duration reward, no sample gate."""

from __future__ import annotations

import datetime
import inspect
import json
import sqlite3

import pytest

import backend.services.adaptive_learning as al
import backend.services.continuation_backfill as cb
import backend.services.continuation_surface as cs
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.services.day_v2.live_exit_evaluator import catastrophic_threshold_price
from backend.services.scalp_v2.executable_edge import scalp_executable_edge
from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT

T0 = 1_790_900_000.0
DAY = (
    "DAY_V2_FIVE_SETUP_RUNNER_V1",
    "DAY_V2_ENTRY_PRIOR_STRUCTURE_V1",
    "DAY_V2_STRUCTURE_RUNNER_V1",
)
FEATURES = {"net": 0.01, "mfe": 0.02, "mae": 0.0, "giveback": 0.0, "dist_high": 0.0, "slope": 0.0, "age_min": 30.0}


def _iso(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar_ts(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _db(tmp_path) -> str:
    path = str(tmp_path / "hist.db")
    conn = sqlite3.connect(path)
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
    conn.commit()
    conn.close()
    return path


def _trade(path: str, *, trade_id: str, entry: float, exit_px: float, entry_at: float, exit_at: float, atr: float = 0.0) -> None:
    decision = json.dumps({"setup": "RANGE_BOUNCE", "regime": "neutral"})
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades
            (trade_id, engine_id, symbol, side, entry_price, price, entry_timestamp, timestamp,
             strategy_version, entry_contract_version, exit_contract_version, adaptive_decision_json, atr_at_entry, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (trade_id, "DAY_V2", "BTC/USDT", "BUY", entry, entry, _iso(entry_at), _iso(entry_at), *DAY, decision, atr, None),
        )
        conn.execute(
            """
            INSERT INTO paper_trades
            (trade_id, engine_id, symbol, side, entry_price, price, entry_timestamp, timestamp,
             strategy_version, entry_contract_version, exit_contract_version, adaptive_decision_json, atr_at_entry, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (f"sell-{trade_id}", "DAY_V2", "BTC/USDT", "SELL", entry, exit_px, _iso(entry_at), _iso(exit_at), *DAY, decision, atr, "NET_PROFIT_EXIT"),
        )


def _beat(path: str, trade_id: str, entry: float, moment: float, mark: float, entry_at: float) -> None:
    net = (mark - entry) / entry - ESTIMATED_ROUNDTRIP_COST
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO ai_position_heartbeats VALUES (?,?,?,?,?,?,?,?,?)",
            (int(moment * 1000), _iso(moment), trade_id, mark, net, max(0.0, (mark - entry) / entry), 0.0, moment - entry_at, max(entry, mark)),
        )


def _bars(path: str, start: float, minutes: int, close: float, low: float | None = None) -> None:
    floor = close if low is None else low
    with sqlite3.connect(path) as conn:
        for i in range(minutes):
            opened = start + i * 60
            conn.execute(
                "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)",
                ("BTC-USDT", "1m", _bar_ts(opened), close, max(close, floor), floor, close),
            )


def test_label_is_hold_advantage_not_absolute_price_and_ignores_the_old_exit(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="now", entry=100, exit_px=90, entry_at=T0, exit_at=T0 + 1200)
    _beat(path, "now", 100, T0 + 300, 101, T0)
    _bars(path, T0, 200, 110)
    labels, _inv = cb.build_advantage_labels(path, as_of=T0 + 8000)
    label = next(item for item in labels if item.horizon == 3600)
    now_net = (101 - 100) / 100 - ESTIMATED_ROUNDTRIP_COST
    future_net = (110 - 100) / 100 - ESTIMATED_ROUNDTRIP_COST
    assert label.advantage == pytest.approx(future_net - now_net)
    assert label.advantage == pytest.approx(0.09)
    assert label.advantage != pytest.approx(110)
    assert label.advantage != pytest.approx(future_net)
    assert label.features["net"] == pytest.approx(now_net)
    assert label.features["slope"] == pytest.approx(0.0)
    assert label.label_time > label.snapshot_time
    assert "age_min" in label.features
    early, _ = cb.build_advantage_labels(path, as_of=T0 + 400)
    assert all(item.horizon != 900 for item in early)
    assert all(item.label_time <= T0 + 400 + 1e-6 for item in early)


def test_missing_horizon_stays_missing_and_catastrophic_censors_the_recovery(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="gap", entry=100, exit_px=100, entry_at=T0, exit_at=T0 + 600, atr=0.0)
    _beat(path, "gap", 100, T0 + 60, 100, T0)
    labels, inv = cb.build_advantage_labels(path, as_of=T0 + 80000)
    assert labels == []
    assert inv["DAY_V2"]["unusable"]["horizon_unavailable"] > 0
    _trade(path, trade_id="stop", entry=100, exit_px=110, entry_at=T0 + 100000, exit_at=T0 + 101000, atr=1.0)
    _beat(path, "stop", 100, T0 + 100300, 101, T0 + 100000)
    _bars(path, T0 + 100000, 30, 110, low=90)
    censored, _ = cb.build_advantage_labels(path, as_of=T0 + 104000)
    safety = catastrophic_threshold_price(100.0, 1.0, 0.0)
    row = next(item for item in censored if item.trade_id == "stop" and item.horizon == 900)
    assert row.advantage == pytest.approx(cb._net(100.0, safety) - ((101 - 100) / 100 - ESTIMATED_ROUNDTRIP_COST))
    assert row.advantage < 0


def test_overlapping_heartbeats_share_weight_and_there_is_no_duration_term(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="burst", entry=100, exit_px=100, entry_at=T0, exit_at=T0 + 8000)
    for offset in (3600, 3601, 3602, 7200):
        _beat(path, "burst", 100, T0 + offset, 101, T0)
    _bars(path, T0, 900, 101)
    labels, _ = cb.build_advantage_labels(path, as_of=T0 + 54000)
    assert labels
    assert sum(item.weight for item in labels) == pytest.approx(1.0)
    assert max(item.weight for item in labels) < 1.0
    tight = [item.weight for item in labels if abs(item.snapshot_time - (T0 + 3601)) < 0.01 or abs(item.snapshot_time - (T0 + 3602)) < 0.01]
    spaced = [item.weight for item in labels if abs(item.snapshot_time - (T0 + 7200)) < 0.01]
    assert tight and spaced
    assert sum(tight) < sum(spaced)
    assert len({round(item.advantage, 8) for item in labels if item.horizon == 900}) == 1


def test_loss_teaches_exit_and_later_profit_teaches_hold_without_a_sample_gate():
    memory = cs.ContinuationMemory()
    assert memory.advantage("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", FEATURES, T0) == (0.0, None)
    memory.update("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", 900, FEATURES, -0.04, 1.0, T0)
    negative, horizon = memory.advantage("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", FEATURES, T0 + 1)
    assert horizon == 900
    assert negative < 0
    assert al.learned_hold_or_exit(expected_terminal_net=0.01 + negative, unrealized_net=0.01) == "exit"
    for step in range(40):
        memory.update("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", 900, FEATURES, 0.05, 1.0, T0 + 1000 + step)
        memory.update("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", 14400, FEATURES, 0.08, 1.0, T0 + 1000 + step)
    positive, chosen = memory.advantage("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", FEATURES, T0 + 2000, how="best")
    assert positive > 0
    assert chosen == 14400
    assert al.learned_hold_or_exit(expected_terminal_net=0.01 + positive, unrealized_net=0.01) == "hold"
    winner = dict(FEATURES, net=0.03, mfe=0.04)
    loser = dict(FEATURES, net=-0.03, mae=0.04)
    for step in range(30):
        memory.update("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", 900, winner, 0.04, 1.0, T0 + 3000 + step)
        memory.update("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", 900, loser, -0.04, 1.0, T0 + 3000 + step)
    held, _ = memory.advantage("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", winner, T0 + 4000)
    sold, _ = memory.advantage("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", loser, T0 + 4000)
    assert held > sold


def test_rows_do_not_take_authority_until_that_engine_is_installed(tmp_path):
    path = _db(tmp_path)
    mark = 0.01
    assert cs.record_advantage(path, engine="DAY_V2", symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="neutral", horizon=900, features=FEATURES, advantage=-0.2, weight=1.0, now=T0)
    assert al.continuation_terminal(path, "DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", mark, now=T0) == pytest.approx(mark)
    surface = cs.ContinuationMemory()
    surface.update("DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", 900, FEATURES, -0.2, 1.0, T0)
    state = str(tmp_path / "surface.db")
    cb.write_surface(state, surface)
    target = str(tmp_path / "live.db")
    al.observe(target, engine="DAY_V2", symbol="ETHUSDT", setup="RANGE_BOUNCE", regime="neutral", metric="trade_net", value=0.02, strategy_version=al.current_strategy_version("DAY_V2"), now=T0)
    before = sqlite3.connect(target).execute("SELECT ewma FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0]
    installed = cs.install_surface(target, state, {"DAY_V2": {"anchor_commit": "7ab0f13", "anchor_at": "2026-10-01T21:41:15Z", "observations": 1}}, how="blended", engines=("DAY_V2",))
    after = sqlite3.connect(target).execute("SELECT ewma FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0]
    assert installed["DAY_V2"] > 0
    assert "SCALP_V2" not in installed
    assert after == before
    assert cs.advantage_authority(target, "DAY_V2") is True
    assert cs.advantage_authority(target, "SCALP_V2") is False
    assert cs.installed_aggregator(target, "DAY_V2") == "blended"
    terminal = al.continuation_terminal(target, "DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", mark, now=T0, features=FEATURES)
    assert terminal < mark
    assert cb.fold_recent_advantages(path) == 0


def test_replay_exits_a_later_loser_without_a_clock(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="teacher", entry=100, exit_px=90, entry_at=T0, exit_at=T0 + 1800)
    _beat(path, "teacher", 100, T0 + 300, 101, T0)
    _bars(path, T0, 40, 90)
    _trade(path, trade_id="student", entry=100, exit_px=90, entry_at=T0 + 20000, exit_at=T0 + 21800)
    _beat(path, "student", 100, T0 + 20300, 101, T0 + 20000)
    _bars(path, T0 + 20000, 40, 90)
    report = cb.walk_advantage(path, as_of=T0 + 25000)
    day = report["engines"]["DAY_V2"]["best"]
    assert day["learned"]["trades"] == 2
    assert day["learned_reasons"]["LEARNED"] >= 1
    assert cb.continuation_accepts({"net": -0.01, "dd": 0.02, "trades": 3}, {"net": -0.02, "dd": 0.02, "trades": 3}) is True
    assert cb.continuation_accepts({"net": -0.01, "dd": 0.05, "trades": 3}, {"net": -0.02, "dd": 0.02, "trades": 3}) is False
    assert cb.continuation_accepts({"net": -0.03, "dd": 0.01, "trades": 3}, {"net": -0.02, "dd": 0.02, "trades": 3}) is False
    assert cb.continuation_predicts({"rank_correlation": 0.33, "hold_exit_accuracy": 0.75}) is True
    assert cb.continuation_predicts({"rank_correlation": 0.04, "hold_exit_accuracy": 0.14}) is False


def test_no_time_rule_sample_gate_or_entry_change():
    surface = inspect.getsource(cs)
    decision = inspect.getsource(cs.ContinuationMemory.advantage)
    assert "TIME_STOP" not in surface
    assert "MAX_HOLD" not in surface
    assert "n < 8" not in surface
    assert "n < 20" not in surface
    assert "age_min" not in decision
    assert "hold_seconds" not in decision
    assert "hold_adv" not in inspect.getsource(al.day_decision)
    assert "continuation_terminal" not in inspect.getsource(al.day_decision)
    assert "hold_adv" not in inspect.getsource(scalp_executable_edge)
    assert pytest.approx(0.015) == SCALP_V2_CATASTROPHIC_PCT
    assert catastrophic_threshold_price(100.0, 1.0, 0.0) == pytest.approx(97.0)
