"""Continuation backfill: causal labels, version isolation, and both-way hold learning."""

from __future__ import annotations

import inspect
import json
import sqlite3

import pytest

import backend.services.adaptive_learning as al
import backend.services.continuation_backfill as cb
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


def _iso(moment: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(moment, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar_ts(moment: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(moment, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _trade(path: str, *, trade_id: str, entry: float, exit_px: float, entry_at: float, exit_at: float, version=DAY, setup="RANGE_BOUNCE", old: bool = False) -> None:
    decision = json.dumps({"setup": setup, "regime": "neutral"})
    versions = ("OLD", "OLD", "OLD") if old else version
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades
            (trade_id, engine_id, symbol, side, entry_price, price, entry_timestamp, timestamp,
             strategy_version, entry_contract_version, exit_contract_version, adaptive_decision_json, atr_at_entry, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (trade_id, "DAY_V2", "BTC/USDT", "BUY", entry, entry, _iso(entry_at), _iso(entry_at), *versions, decision, 1.0, None),
        )
        conn.execute(
            """
            INSERT INTO paper_trades
            (trade_id, engine_id, symbol, side, entry_price, price, entry_timestamp, timestamp,
             strategy_version, entry_contract_version, exit_contract_version, adaptive_decision_json, atr_at_entry, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (f"sell-{trade_id}", "DAY_V2", "BTC/USDT", "SELL", entry, exit_px, _iso(entry_at), _iso(exit_at), *versions, decision, 1.0, "NET_PROFIT_EXIT"),
        )


def _beat(path: str, trade_id: str, entry: float, moment: float, mark: float, entry_at: float) -> None:
    net = (mark - entry) / entry - ESTIMATED_ROUNDTRIP_COST
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO ai_position_heartbeats VALUES (?,?,?,?,?,?,?,?,?)",
            (int(moment * 1000), _iso(moment), trade_id, mark, net, max(0.0, (mark - entry) / entry), 0.0, moment - entry_at, max(entry, mark)),
        )


def _flat_bars(path: str, start: float, minutes: int, close: float) -> None:
    with sqlite3.connect(path) as conn:
        for i in range(minutes):
            opened = start + i * 60
            conn.execute(
                "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)",
                ("BTC-USDT", "1m", _bar_ts(opened), close, close, close, close),
            )


def test_old_version_and_future_bars_do_not_enter_the_label(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="old", entry=100, exit_px=110, entry_at=T0, exit_at=T0 + 3600, old=True)
    _beat(path, "old", 100, T0 + 300, 101, T0)
    _trade(path, trade_id="now", entry=100, exit_px=90, entry_at=T0, exit_at=T0 + 3600)
    _beat(path, "now", 100, T0 + 300, 101, T0)
    _flat_bars(path, T0, 80, 90)
    early, early_inv = cb.build_observations(path, as_of=T0 + 300)
    assert early == []
    assert early_inv["DAY_V2"]["usable_snapshots"] == 0
    later, inv = cb.build_observations(path, as_of=T0 + 7200)
    assert inv["DAY_V2"]["positions"] == 1
    assert inv["DAY_V2"]["unusable"]["not_current_version"] == 1
    assert later
    assert all(obs.trade_id == "now" for obs in later)
    assert all(obs.snapshot_time <= T0 + 300 for obs in later)
    assert all(obs.label_time > obs.snapshot_time for obs in later)
    assert all(obs.remaining < 0 for obs in later)
    snap = next(obs for obs in later if obs.source == "exit_fill")
    assert snap.unrealized_net == pytest.approx((101 - 100) / 100 - ESTIMATED_ROUNDTRIP_COST)
    assert snap.remaining == pytest.approx(((90 - 100) / 100) - ((101 - 100) / 100))


def test_continuation_moves_both_ways_and_install_leaves_entry_state(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="loss", entry=100, exit_px=90, entry_at=T0, exit_at=T0 + 1800)
    _beat(path, "loss", 100, T0 + 300, 101, T0)
    _flat_bars(path, T0, 40, 90)
    loss_obs, _inv = cb.build_observations(path, as_of=T0 + 7200)
    state = str(tmp_path / "state.db")
    cb.write_observations(state, loss_obs)
    green = (101 - 100) / 100 - ESTIMATED_ROUNDTRIP_COST
    down = al.continuation_terminal(state, "DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", green, now=T0 + 7200)
    assert down < green
    assert al.continuation_terminal(state, "SCALP_V2", "BTCUSDT", "VWAP_EMA_RECLAIM", "neutral", green, now=T0 + 7200) == pytest.approx(green)
    assert al.learned_hold_or_exit(expected_terminal_net=down, unrealized_net=green) == "exit"
    _trade(path, trade_id="win", entry=100, exit_px=110, entry_at=T0 + 8000, exit_at=T0 + 9800)
    _beat(path, "win", 100, T0 + 8300, 101, T0 + 8000)
    _flat_bars(path, T0 + 8000, 40, 110)
    for _ in range(12):
        wins, _ = cb.build_observations(path, as_of=T0 + 20000)
        cb.write_observations(state, [obs for obs in wins if obs.trade_id == "win"])
    back = al.continuation_terminal(state, "DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", green, now=T0 + 20000)
    assert back > down
    assert al.learned_hold_or_exit(expected_terminal_net=back, unrealized_net=green) == "hold"
    empty = str(tmp_path / "empty.db")
    assert al.continuation_terminal(empty, "DAY_V2", "XRPUSDT", "EXHAUSTION_MR", "neutral", -0.01, now=T0) == pytest.approx(-0.01)
    al.observe(path, engine="DAY_V2", symbol="ETHUSDT", setup="RANGE_BOUNCE", regime="neutral", metric="trade_net", value=0.02, strategy_version=al.current_strategy_version("DAY_V2"), now=T0)
    before = sqlite3.connect(path).execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    net_before = sqlite3.connect(path).execute("SELECT ewma FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0]
    cb.install_continuation(path, state, _inv)
    after = sqlite3.connect(path).execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    net_after = sqlite3.connect(path).execute("SELECT ewma FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0]
    assert after == before
    assert net_after == net_before
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric LIKE 'hold_remaining%'").fetchone()[0] > 0


def test_no_clock_or_sample_gate_and_entry_code_is_untouched():
    src = inspect.getsource(cb)
    assert "TIME_STOP" not in src
    assert "effective_n" not in src
    assert "samples <" not in src
    assert "MIN_TRADES" not in src
    assert "hold_remaining" not in inspect.getsource(al.day_decision)
    assert "hold_remaining" not in inspect.getsource(scalp_executable_edge)
    assert pytest.approx(0.015) == SCALP_V2_CATASTROPHIC_PCT
    assert catastrophic_threshold_price(100.0, 1.0, 0.0) == pytest.approx(97.0)


def test_replay_can_exit_before_the_recorded_fill(tmp_path):
    path = _db(tmp_path)
    _trade(path, trade_id="teacher", entry=100, exit_px=98, entry_at=T0, exit_at=T0 + 1800)
    _beat(path, "teacher", 100, T0 + 300, 101, T0)
    _flat_bars(path, T0, 30, 99)
    _trade(path, trade_id="student", entry=100, exit_px=98, entry_at=T0 + 10000, exit_at=T0 + 11800)
    _beat(path, "student", 100, T0 + 10300, 101, T0 + 10000)
    _flat_bars(path, T0 + 10000, 30, 99)
    report = cb.walk_forward(path, as_of=T0 + 14000)
    day = report["engines"]["DAY_V2"]
    assert day["learned"]["trades"] == 2
    assert day["actual"]["trades"] == 2
    assert day["classes"]["EXIT_EARLIER_PROFITABLY"] == 1
    assert day["learned_reasons"]["LEARNED"] == 1
    mem = cb._MemoryPosterior()
    written, _ = cb.build_observations(path, as_of=T0 + 14000)
    state = str(tmp_path / "match.db")
    cb.write_observations(state, written)
    for obs in written:
        mem.observe(obs)
    green = (101 - 100) / 100 - ESTIMATED_ROUNDTRIP_COST
    sqlite_terminal = al.continuation_terminal(state, "DAY_V2", "BTCUSDT", "RANGE_BOUNCE", "neutral", green, now=T0 + 14000)
    assert mem.terminal("DAY_V2", "BTC/USDT", "RANGE_BOUNCE", "neutral", green, T0 + 14000) == pytest.approx(sqlite_terminal)
