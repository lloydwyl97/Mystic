"""Forward exact-policy capture. No orders, cash, or live authority."""

from __future__ import annotations

import sqlite3

from backend.services.day_v2.day_policy_decision import decide_day_continuation
from backend.services.exact_policy_capture import (
    CAPTURE,
    advance_exact_episodes,
    capture_counts,
    note_disposition,
    note_real_fill_price,
    open_exact_episode,
)
from backend.services.policy_episode import COUNTERFACTUAL, REAL


def _cand(**overrides):
    base = {
        "symbol": "BTCUSDT",
        "ask_price": 100.0,
        "as_of": 1_000_000.0,
        "learned_setup": "BREAKOUT_CONTINUATION__MARKET",
        "regime_tag": "quiet",
        "state_features": {"rsi": 0.4, "atr15_pct": 0.01},
        "adaptive": {
            "expected_net": -0.001,
            "size_mult": 1.0,
            "economic": {"uncalibrated_policy_value": -0.001, "policy_calibration": 0.0, "expected_cost": 0.00066},
        },
        "rank": {"position": 1},
        "signal": type(
            "S",
            (),
            {"setup": "BREAKOUT_CONTINUATION", "atr": 1.0, "atr_1h": 2.0, "structural_anchor": 90.0, "target_price": 110.0, "objective_structural": 110.0},
        )(),
    }
    base.update(overrides)
    return base


def test_four_symbols_and_invalid_candidates(tmp_path):
    db = str(tmp_path / "t.db")
    ids = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        ids.append(open_exact_episode(db, _cand(symbol=symbol), len(ids) + 1))
    assert all(ids)
    assert open_exact_episode(db, _cand(symbol="DOGEUSDT"), 99) is None
    assert open_exact_episode(db, _cand(ask_price=0), 98) is None
    conn = sqlite3.connect(db)
    snaps = conn.execute("SELECT snapshot_json FROM policy_episodes").fetchall()
    assert len(snaps) == 4
    assert all("EXACT_POLICY_V1" in row[0] for row in snaps)


def test_refused_episode_uses_same_decision_and_does_not_touch_cash(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr("backend.services.exact_policy_capture._book", lambda _symbol: (99.0, 99.2))
    def _terminal(*args, **kwargs):
        net = kwargs.get("unrealized_net", args[5] if len(args) > 5 else 0.0)
        return float(net) - 0.01

    monkeypatch.setattr("backend.services.adaptive_learning.continuation_terminal", _terminal)
    episode = open_exact_episode(db, _cand(), 7)
    note_disposition(db, 7, funded=False, reason="NO_EXECUTABLE_NET_EDGE")
    stats = advance_exact_episodes(db, now=1_000_030.0)
    assert stats["exited"] == 1
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT kind, net, label_available_at, train_applied, funded FROM policy_episodes WHERE id=?", (episode,)).fetchone()
    assert row[0] == COUNTERFACTUAL
    assert row[2] == 1_000_030.0
    assert row[3] == 0
    assert row[4] == 0
    check = conn.execute("SELECT bid, ask, action, state_id, mfe FROM policy_episode_checks").fetchone()
    assert check[0] == 99.0 and check[1] == 99.2 and check[2] == "exit" and check[3]
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='portfolio_engine_ledger'").fetchone() is None
    assert conn.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0] == 0


def test_same_state_same_decision():
    kwargs = {
        "symbol": "ETHUSDT",
        "setup": "RANGE_BOUNCE",
        "entry_price": 100.0,
        "mark": 101.0,
        "low": 100.0,
        "high": 101.0,
        "prev_net": 0.0,
        "age_sec": 30.0,
        "atr_at_entry": 1.0,
        "structural_anchor": 90.0,
        "target_price": 110.0,
        "entry_time": 10.0,
        "roundtrip_cost": 0.00066,
        "regime": "MARKET",
        "now": 40.0,
    }
    # No installed surface: terminal is the mark plus a zero prior, so the policy holds.
    left = decide_day_continuation(":memory:", **kwargs)
    right = decide_day_continuation(":memory:", **kwargs)
    assert left["action"] == right["action"] == "hold"
    assert left["features"] == right["features"]


def test_real_outcome_precedes_simulation(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr("backend.services.exact_policy_capture._book", lambda _symbol: (90.0, 90.1))
    open_exact_episode(db, _cand(), 3)
    note_disposition(db, 3, funded=True, reason="FILLED")
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO adaptive_candidate_markouts (
            id, engine_id, symbol, setup, regime, strategy_version, signaled,
            ref_price, roundtrip_cost, evaluated_at, filled, realized_net
        ) VALUES (3, 'DAY_V2', 'BTCUSDT', 'HTF_TREND_PULLBACK', 'MARKET', 't', 1, 100.0, 0.00066, 1, 1, 0.0123)
        """
    )
    conn.commit()
    conn.close()
    stats = advance_exact_episodes(db, now=1_000_100.0)
    assert stats["real"] == 1 and stats["exited"] == 0
    conn = sqlite3.connect(db)
    kind, net, label = conn.execute("SELECT kind, net, label_available_at FROM policy_episodes").fetchone()
    assert kind == REAL and net == 0.0123 and label == 1_000_100.0


def test_downtime_does_not_invent_checks(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr("backend.services.exact_policy_capture._book", lambda _symbol: (100.0, 100.1))
    open_exact_episode(db, _cand(), 4)
    note_disposition(db, 4, funded=False, reason="NO_EXECUTABLE_NET_EDGE")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE policy_episodes SET last_check_at=?", (1_000_000.0,))
    conn.commit()
    conn.close()
    stats = advance_exact_episodes(db, now=1_000_000.0 + 1000.0)
    assert stats["downtime"] == 1
    conn = sqlite3.connect(db)
    status, reason, nchecks = conn.execute(
        "SELECT parity_status, non_parity_reason, (SELECT COUNT(*) FROM policy_episode_checks) FROM policy_episodes"
    ).fetchone()
    assert status == "NON_PARITY" and reason == "DOWNTIME_GAP" and nchecks == 1


def test_missing_book_skips_without_a_fake_price(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr("backend.services.exact_policy_capture._book", lambda _symbol: (None, None))
    open_exact_episode(db, _cand(), 5)
    note_disposition(db, 5, funded=False, reason="NO_EXECUTABLE_NET_EDGE")
    stats = advance_exact_episodes(db, now=1_000_040.0)
    assert stats["skipped"] == 1 and stats["exited"] == 0
    assert capture_counts(db)["open"] == 1


def test_snapshot_is_not_rewritten_by_disposition(tmp_path):
    db = str(tmp_path / "t.db")
    open_exact_episode(db, _cand(), 8)
    conn = sqlite3.connect(db)
    before = conn.execute("SELECT snapshot_json FROM policy_episodes").fetchone()[0]
    conn.close()
    note_disposition(db, 8, funded=False, reason="NO_EXECUTABLE_NET_EDGE")
    conn = sqlite3.connect(db)
    after = conn.execute("SELECT snapshot_json, capture_version FROM policy_episodes").fetchone()
    assert after[0] == before and after[1] == CAPTURE


def test_funded_candidate_links_trade_and_real_fill_is_separate(tmp_path):
    db = str(tmp_path / "t.db")
    open_exact_episode(db, _cand(), 11)
    note_disposition(db, 11, funded=True, reason="FILLED", trade_id="mystic_BTCUSDT_1")
    note_real_fill_price(db, 11, fill_price=101.5, exit_reason="DAY_V2_LEARNED_CONTINUATION")
    conn = sqlite3.connect(db)
    trade_id, funded = conn.execute("SELECT trade_id, funded FROM policy_episodes").fetchone()
    fill, reason = conn.execute("SELECT fill_price, exit_reason_real FROM policy_fill_parity").fetchone()
    assert trade_id == "mystic_BTCUSDT_1" and funded == 1
    assert fill == 101.5 and reason == "DAY_V2_LEARNED_CONTINUATION"


def test_catastrophic_exit_uses_the_live_threshold():
    decision = decide_day_continuation(
        ":memory:",
        symbol="SOLUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="quiet",
        entry_price=100.0,
        mark=80.0,
        low=80.0,
        high=100.0,
        prev_net=0.0,
        age_sec=30.0,
        atr_at_entry=1.0,
        structural_anchor=90.0,
        target_price=110.0,
        entry_time=10.0,
        roundtrip_cost=0.00066,
        now=40.0,
    )
    assert decision["action"] == "exit"
    assert decision["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"


def test_label_stays_unavailable_until_exit_and_restart_keeps_state(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    open_exact_episode(db, _cand(), 12)
    note_disposition(db, 12, funded=False, reason="NO_EXECUTABLE_NET_EDGE")
    prices = iter([(105.0, 105.2), (101.0, 101.2), (101.0, 101.2)])
    monkeypatch.setattr("backend.services.exact_policy_capture._book", lambda _symbol: next(prices))
    advance_exact_episodes(db, now=1_000_040.0)
    conn = sqlite3.connect(db)
    label, mfe = conn.execute("SELECT label_available_at, mfe FROM policy_episodes").fetchone()
    assert label is None and mfe > 0
    conn.close()
    advance_exact_episodes(db, now=1_000_070.0)
    conn = sqlite3.connect(db)
    label, giveback, checks = conn.execute(
        "SELECT label_available_at, (SELECT giveback FROM policy_episode_checks ORDER BY checked_at DESC LIMIT 1), COUNT(*) FROM policy_episodes, policy_episode_checks"
    ).fetchone()
    state_rows = conn.execute("SELECT COUNT(*) FROM exact_policy_learner_state").fetchone()[0]
    assert label is None and giveback > 0 and checks == 2 and state_rows == 1
    conn.close()
    advance_exact_episodes(db, now=1_000_100.0)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM policy_episode_checks").fetchone()[0] == 3
    assert conn.execute("SELECT label_available_at FROM policy_episodes").fetchone()[0] is None


def test_resolved_counterfactual_trains_once_and_does_not_touch_live_posterior(tmp_path, monkeypatch):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr("backend.services.exact_policy_capture._book", lambda _symbol: (90.0, 90.2))

    def _terminal(*args, **kwargs):
        net = kwargs.get("unrealized_net", args[5] if len(args) > 5 else 0.0)
        return float(net) - 0.01

    monkeypatch.setattr("backend.services.adaptive_learning.continuation_terminal", _terminal)
    open_exact_episode(db, _cand(), 13)
    note_disposition(db, 13, funded=False, reason="NO_EXECUTABLE_NET_EDGE")
    first = advance_exact_episodes(db, now=1_000_020.0)
    second = advance_exact_episodes(db, now=1_000_050.0)
    assert first["exited"] == 1 and second["exited"] == 0
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT train_eligible, train_applied, label_available_at, kind FROM policy_episodes").fetchone()
    assert row[0] == 1 and row[1] == 0 and row[2] == 1_000_020.0 and row[3] == COUNTERFACTUAL
    assert conn.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0] == 0


def test_live_and_counterfactual_call_the_same_evaluator():
    from backend.services.exact_policy_capture import live_day_decision

    class _Position:
        entry_price = 100.0
        highest_price = 101.0
        lowest_price = 100.0
        entry_time = 10.0
        symbol = "XRPUSDT"
        trade_id = ""
        atr_at_entry = 1.0
        thesis_invalid_level = 90.0
        thesis_target_level = 110.0
        entry_thesis = "BREAKOUT_CONTINUATION"
        adaptive_decision = {"setup": "BREAKOUT_CONTINUATION__MARKET", "regime": "quiet"}
        day_atr_1h_at_entry = 2.0
        day_objective_structural = 110.0

    live = live_day_decision(":memory:", _Position(), 101.0, 0.00066, now=40.0)
    direct = decide_day_continuation(
        ":memory:",
        symbol="XRPUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        exit_setup="BREAKOUT_CONTINUATION",
        regime="quiet",
        entry_price=100.0,
        mark=101.0,
        low=100.0,
        high=101.0,
        prev_net=None,
        age_sec=30.0,
        atr_at_entry=1.0,
        structural_anchor=90.0,
        target_price=110.0,
        entry_time=10.0,
        roundtrip_cost=0.00066,
        atr_1h_at_entry=2.0,
        objective_structural=110.0,
        now=40.0,
    )
    assert live["action"] == direct["action"] == "hold"
    assert live["reason"] == direct["reason"]
    assert live["features"] == direct["features"]


def test_capture_does_not_touch_orders_scalp_or_other_venues():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    capture = (root / "backend/services/exact_policy_capture.py").read_text()
    decision = (root / "backend/services/day_v2/day_policy_decision.py").read_text()
    scalp = (root / "backend/services/scalp_v2/exit_evaluator.py").read_text()
    for name in ("execute_sell", "submit_order", "cash_balance", "coinbase", "kraken", "policy_research_enabled"):
        assert name not in capture.lower()
        assert name not in decision.lower()
    assert "exact_policy_capture" not in scalp
    assert "DOGE" not in capture
