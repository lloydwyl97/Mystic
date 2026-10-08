"""Research policy episodes stay out of accounting and keep the decision that was made."""

from __future__ import annotations

import inspect
import sqlite3

import pytest

from backend.services import adaptive_learning as al
from backend.services import policy_episode as pe
from backend.services.adaptive_learning import executable_bid_net

DAY = "DAY_V2"
T0 = 1_700_000_000.0


def _open(db: str, candidate_id: int = 1, ask: float = 100.0) -> int:
    episode = pe.open_episode(
        db,
        candidate_id=candidate_id,
        engine=DAY,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="btcdown_vollo",
        entry_ask=ask,
        roundtrip_cost=0.00066,
        economics={"market_alpha": -0.005, "policy_gap": 0.009, "uncalibrated_policy_value": 0.003, "policy_calibration": -0.002, "policy_value": 0.001},
        rank_position=1,
        size_mult=1.1,
        now=T0,
    )
    assert episode is not None
    return episode


def test_a_missing_ask_opens_nothing_and_writes_no_accounting(tmp_path):
    db = str(tmp_path / "e.db")
    assert pe.open_episode(db, candidate_id=1, engine=DAY, symbol="BTCUSDT", setup="S", regime="r", entry_ask=0.0, roundtrip_cost=0.00066, now=T0) is None
    assert not __import__("os").path.exists(db)


def test_the_same_candidate_is_one_episode(tmp_path):
    db = str(tmp_path / "e.db")
    first = _open(db)
    second = _open(db)
    assert first == second
    with sqlite3.connect(db) as conn:
        kind, n = conn.execute("SELECT kind, COUNT(*) FROM policy_episodes").fetchone()
    assert kind == pe.COUNTERFACTUAL and n == 1


def test_exit_uses_the_bid_and_stores_the_decision_that_was_made(tmp_path, monkeypatch):
    db = str(tmp_path / "e.db")
    _open(db, ask=100.0)

    def _terminal(db_path, engine, symbol, setup, regime, unrealized, now=None, features=None):
        del db_path, engine, symbol, setup, regime, now, features
        return float(unrealized) - 0.01

    monkeypatch.setattr(pe, "continuation_terminal", _terminal)
    stats = pe.advance_episodes(db, lambda _symbol: 99.0, now=T0 + 40.0)
    assert stats["exited"] == 1 and stats["real"] == 0
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT kind, status, exit_bid, net, gross, exit_reason, learned FROM policy_episodes").fetchone()
        check = conn.execute("SELECT action, advantage FROM policy_episode_checks").fetchone()
        positions = conn.execute("SELECT name FROM sqlite_master WHERE name='portfolio_engine_positions'").fetchone()
    assert row[0] == pe.COUNTERFACTUAL and row[1] == pe.CLOSED
    assert row[2] == pytest.approx(99.0)
    assert row[3] == pytest.approx(executable_bid_net(100.0, 99.0, 0.00066))
    assert row[4] == pytest.approx((99.0 - 100.0) / 100.0)
    assert row[5] == pe.EXIT_REASON and row[6] == 1
    assert check[0] == "exit" and check[1] == pytest.approx(-0.01)
    assert positions is None
    assert "direct_policy_net" not in inspect.getsource(al.day_net_expectancy)


def test_a_later_model_does_not_rewrite_the_stored_decision(tmp_path, monkeypatch):
    db = str(tmp_path / "e.db")
    _open(db)

    def _down(_db, _e, _s, _u, _r, unrealized, now=None, features=None):
        del _db, _e, _s, _u, _r, now, features
        return float(unrealized) - 0.02

    monkeypatch.setattr(pe, "continuation_terminal", _down)
    pe.advance_episodes(db, lambda _symbol: 99.5, now=T0 + 30.0)
    def _later(*_a, **_k):
        return 1.0

    monkeypatch.setattr(pe, "continuation_terminal", _later)
    pe.advance_episodes(db, lambda _symbol: 99.5, now=T0 + 90.0)
    with sqlite3.connect(db) as conn:
        advantage = conn.execute("SELECT advantage FROM policy_episode_checks").fetchone()[0]
        n = conn.execute("SELECT n FROM adaptive_metric_state WHERE metric='direct_policy_net'").fetchone()[0]
    assert advantage == pytest.approx(-0.02)
    assert pe.decision_from_check(-0.01, advantage) == "exit"
    assert n == pytest.approx(1.0)


def test_a_real_fill_is_not_a_second_simulated_trade(tmp_path, monkeypatch):
    db = str(tmp_path / "e.db")
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="ETHUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="r",
        ref_price=2000.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": 0.001, "uncalibrated_policy_value": 0.003, "policy_calibration": -0.002},
        opportunity_id="REAL-1",
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=? WHERE id=?", (-0.0004, row_id))
    pe.open_episode(db, candidate_id=row_id, engine=DAY, symbol="ETHUSDT", setup="BREAKOUT_CONTINUATION__MARKET", regime="r", entry_ask=2000.0, roundtrip_cost=0.00066, now=T0)
    def _ignore(*_a, **_k):
        return -1.0

    monkeypatch.setattr(pe, "continuation_terminal", _ignore)
    stats = pe.advance_episodes(db, lambda _symbol: 1.0, now=T0 + 50.0)
    assert stats["real"] == 1 and stats["exited"] == 0
    with sqlite3.connect(db) as conn:
        kind, net, bid = conn.execute("SELECT kind, net, exit_bid FROM policy_episodes").fetchone()
    assert kind == pe.REAL and net == pytest.approx(-0.0004) and bid is None


def test_a_failed_observation_does_not_close_the_episode(tmp_path, monkeypatch):
    db = str(tmp_path / "e.db")
    _open(db)

    def _down(_db, _e, _s, _u, _r, unrealized, now=None, features=None):
        del _db, _e, _s, _u, _r, now, features
        return float(unrealized) - 0.01

    monkeypatch.setattr(pe, "continuation_terminal", _down)

    def _boom(*a, **k):
        del a, k
        raise RuntimeError("disk")

    monkeypatch.setattr(pe, "_fold_observation", _boom)
    with pytest.raises(RuntimeError):
        pe.advance_episodes(db, lambda _symbol: 99.0, now=T0 + 40.0)
    with sqlite3.connect(db) as conn:
        status, learned = conn.execute("SELECT status, learned FROM policy_episodes").fetchone()
        metrics = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='adaptive_metric_state'").fetchone()[0]
    assert status == pe.OPEN and learned == 0
    if metrics:
        with sqlite3.connect(db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric='direct_policy_net'").fetchone()[0]
        assert count == 0


def test_four_symbols_rank_on_policy_net_not_a_fifteen_minute_mark(tmp_path):
    db = str(tmp_path / "e.db")
    nets = {"BTCUSDT": -0.0002, "ETHUSDT": 0.0004, "SOLUSDT": -0.0008, "XRPUSDT": 0.0001}
    for i, (symbol, net) in enumerate(nets.items()):
        pe.open_episode(db, candidate_id=i + 1, engine=DAY, symbol=symbol, setup="S", regime="r", entry_ask=10.0, roundtrip_cost=0.00066, now=T0)
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE policy_episodes SET status=?, net=?, kind=? WHERE candidate_id=?", (pe.CLOSED, net, pe.COUNTERFACTUAL, i + 1))
    board = pe.symbol_outcomes(db)
    assert [row["symbol"] for row in board] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    assert max(board, key=lambda row: row["net"])["symbol"] == "ETHUSDT"
    assert "m15" not in inspect.getsource(pe.advance_episodes)
    assert "feature_ohlcv" not in inspect.getsource(pe.advance_episodes)


def test_no_trade_opinion_and_no_sample_gate():
    import re

    src = inspect.getsource(pe)
    for word in ("rsi", "min_hold", "min_trades", "blacklist"):
        assert re.search(rf"\b{word}\b", src, re.IGNORECASE) is None
