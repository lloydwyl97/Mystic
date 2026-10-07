"""Live short-horizon marks, two-sided calibration, and trajectory weights."""

from __future__ import annotations

import inspect
import json
import re
import sqlite3

from backend.services import adaptive_learning as al
from backend.services.book_queue_capture import queue_features
from backend.services.continuation_backfill import Position, Snapshot, advantage_labels_for
from backend.services.scalp_v2.executable_edge import scalp_executable_edge
from backend.services.strategy_version import current_lifecycle_label

DAY = "DAY_V2"
SCALP = "SCALP_V2"
T0 = 1_700_000_000.0


def _marks(db: str, row_id: int) -> dict:
    with sqlite3.connect(db) as conn:
        raw = conn.execute("SELECT markouts_json FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()[0]
    return json.loads(raw or "{}")


def _book(db: str, ts: float, bid: float) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS microstructure_feature_snapshots (symbol TEXT, ts_utc REAL, features_json TEXT)")
        conn.execute(
            "INSERT INTO microstructure_feature_snapshots (symbol, ts_utc, features_json) VALUES (?,?,?)",
            ("BTC", ts, json.dumps({"best_bid": bid, "best_ask": bid + 0.1})),
        )


def test_live_30s_and_60s_ignore_a_later_candle_close(tmp_path):
    db = str(tmp_path / "t.db")
    row_id = al.record_candidate(
        db, engine=SCALP, symbol="BTCUSDT", setup="CLAIM", regime="r", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=T0
    )
    al.resolve_markouts(db, lambda _s, _t: 130.0, now=T0 + 200)
    marks = _marks(db, row_id)
    assert marks["30"] is None and marks["60"] is None
    assert marks["120"] is not None and marks["120"] > 0.2


def test_live_30s_and_60s_use_the_aligned_bid(tmp_path):
    db = str(tmp_path / "t.db")
    _book(db, T0 + 28.0, 100.2)
    _book(db, T0 + 58.0, 100.4)
    row_id = al.record_candidate(
        db, engine=SCALP, symbol="BTCUSDT", setup="CLAIM", regime="r", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=T0
    )
    al.resolve_markouts(db, lambda _s, _t: 130.0, now=T0 + 200)
    marks = _marks(db, row_id)
    assert abs(marks["30"] - ((100.2 - 100.0) / 100.0 - 0.0006)) < 1e-12
    assert abs(marks["60"] - ((100.4 - 100.0) / 100.0 - 0.0006)) < 1e-12


def test_bid_mark_does_not_charge_the_exit_half_spread_twice(tmp_path):
    db = str(tmp_path / "t.db")
    _book(db, T0 + 28.0, 99.988)
    # Ask entry. Full spread 1.2 bps is inside the bid. Stored cost adds the 0.6 bps exit half on top of 6 bps of fee and slippage.
    row_id = al.record_candidate(
        db, engine=SCALP, symbol="BTCUSDT", setup="CLAIM", regime="r", ref_price=100.0, roundtrip_cost=0.00066, signaled=True, evaluated_at=T0
    )
    al.resolve_markouts(db, lambda _s, _t: 130.0, now=T0 + 200)
    marks = _marks(db, row_id)
    assert abs(marks["30"] - ((99.988 - 100.0) / 100.0 - 0.0006)) < 1e-12
    assert marks["30"] > (99.988 - 100.0) / 100.0 - 0.00066


def test_calibration_folds_as_a_running_mean(tmp_path):
    db = str(tmp_path / "t.db")
    version = al.current_strategy_version(DAY)
    moment = T0
    assert al.observe(db, engine=DAY, symbol="SOLUSDT", setup="EXHAUSTION_MR", regime="r", metric="policy_calibration", value=-0.01, strategy_version=version, now=moment)
    assert al.observe(db, engine=DAY, symbol="SOLUSDT", setup="EXHAUSTION_MR", regime="r", metric="policy_calibration", value=0.0, strategy_version=version, now=moment)
    with sqlite3.connect(db) as conn:
        ewma = conn.execute("SELECT ewma FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()[0]
    assert abs(ewma - (-0.005)) < 1e-12


def test_a_print_after_the_horizon_stays_missing(tmp_path):
    db = str(tmp_path / "t.db")
    _book(db, T0 + 40.0, 101.0)
    row_id = al.record_candidate(
        db, engine=SCALP, symbol="BTCUSDT", setup="CLAIM", regime="r", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=T0
    )
    al.resolve_markouts(db, lambda _s, _t: 130.0, now=T0 + 200)
    assert _marks(db, row_id)["30"] is None


def test_stored_short_mark_is_rebuilt_from_the_bid(tmp_path):
    db = str(tmp_path / "t.db")
    row_id = al.record_candidate(
        db, engine=SCALP, symbol="BTCUSDT", setup="CLAIM", regime="r", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=T0
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?", (json.dumps({"30": 0.25, "60": 0.25}), row_id))
    _book(db, T0 + 28.0, 100.2)
    _book(db, T0 + 58.0, 100.4)
    al.resolve_markouts(db, lambda _s, _t: 130.0, now=T0 + 200)
    marks = _marks(db, row_id)
    assert abs(marks["30"] - ((100.2 - 100.0) / 100.0 - 0.0006)) < 1e-12
    assert marks["30"] < 0.01


def _day_close(db: str, setup: str, predicted: float, realized: float, opp: str) -> None:
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="SOLUSDT",
        setup=setup,
        regime="btcdown_vollo",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": predicted, "market_alpha": -0.0006, "policy_gap": 0.005},
        opportunity_id=opp,
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?",
            (json.dumps({"lifecycle": {"net": -0.008, "reason": "HORIZON_MARK"}}), row_id),
        )
    assert al.record_policy_outcome(db, engine=DAY, opportunity_id=opp, net_pct=realized, candidate_id=row_id)


def test_a_loss_lowers_the_next_forecast_and_a_win_can_raise_it(tmp_path):
    db = str(tmp_path / "t.db")
    setup = "EXHAUSTION_MR"
    before = al.day_decision(db, "SOLUSDT", setup, "btcdown_vollo")["expected_net"]
    _day_close(db, setup, 0.0044, -0.00107, "OPP-LOSS")
    after_loss = al.day_decision(db, "SOLUSDT", setup, "btcdown_vollo")["expected_net"]
    assert after_loss < before
    _day_close(db, setup, after_loss, after_loss + 0.02, "OPP-WIN")
    after_win = al.day_decision(db, "SOLUSDT", setup, "btcdown_vollo")["expected_net"]
    assert after_win > after_loss


def test_scalp_fee_loss_lowers_the_next_executable_edge(tmp_path):
    db = str(tmp_path / "t.db")
    view = al.scalp_decision(db, "ETHUSDT", "CLAIM", "r")
    before = scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.0001, impact_pct=0.0, edge_source="strategy_claim").final_executable_edge_pct
    row_id = al.record_candidate(
        db,
        engine=SCALP,
        symbol="ETHUSDT",
        setup="CLAIM",
        regime="r",
        ref_price=100.0,
        roundtrip_cost=0.0006,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": before},
        raw_expected_move=0.002,
        raw_move_source="strategy_claim",
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        horizon = conn.execute("SELECT label_horizon FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()[0]
        conn.execute("UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?", (json.dumps({str(int(horizon)): 0.0002}), row_id))
    assert al.record_policy_outcome(db, engine=SCALP, opportunity_id="", net_pct=-0.00129, candidate_id=row_id)
    view = al.scalp_decision(db, "ETHUSDT", "CLAIM", "r")
    after = scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.0001, impact_pct=0.0, edge_source="strategy_claim").final_executable_edge_pct
    assert after < before
    assert view["policy_calibration"] < 0


def test_no_setup_state_has_a_learned_policy_value(tmp_path):
    db = str(tmp_path / "t.db")
    decision = al.day_decision(db, "BTCUSDT", "NO_SETUP", "r")
    assert decision["expected_net"] == decision["economic"]["policy_value"]
    assert decision["economic"]["policy_calibration"] == 0.0


def test_retired_exit_does_not_teach_the_current_policy(tmp_path):
    assert current_lifecycle_label("TIME_STOP_EXIT") is False
    assert current_lifecycle_label("HORIZON_MARK") is True
    db = str(tmp_path / "t.db")
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="RANGE_BOUNCE",
        regime="r",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": 0.004},
        opportunity_id="OLD",
    )
    al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?",
            (json.dumps({"lifecycle": {"net": -0.01, "reason": "TIME_STOP_EXIT"}}), row_id),
        )
    assert al.record_policy_outcome(db, engine=DAY, opportunity_id="OLD", net_pct=-0.02, candidate_id=row_id) is False
    assert al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "r")["economic"]["policy_calibration"] == 0.0


def test_overlapping_continuation_states_share_one_trajectory():
    entry = 1_000.0
    snaps = [
        Snapshot(time=1_100.0, net=0.001, mark=101.0, mfe=0.002, mae=0.0, hold_sec=100.0, high_water=0.002),
        Snapshot(time=1_160.0, net=0.0005, mark=100.5, mfe=0.002, mae=0.001, hold_sec=160.0, high_water=0.002),
    ]
    position = Position(
        engine=DAY,
        trade_id="T1",
        symbol="BTCUSDT",
        setup="EXHAUSTION_MR",
        regime="r",
        entry_price=100.0,
        entry_time=entry,
        exit_time=None,
        exit_price=None,
        exit_reason="",
        atr=0.0,
        anchor=100.0,
        snapshots=snaps,
    )
    bars = [(1_980.0, 100.0, 101.0, 100.0, 100.5), (2_040.0, 100.5, 101.0, 100.0, 100.8)]
    labels, _skipped = advantage_labels_for(position, bars, as_of=3_000.0)
    assert len(labels) >= 2
    assert abs(sum(item.weight for item in labels) - 1.0) < 1e-9
    assert all(item.weight < 1.0 for item in labels)


def test_the_repair_adds_no_trade_opinion():
    src = inspect.getsource(al.executable_bid_at) + inspect.getsource(al.policy_calibration) + inspect.getsource(al._learn_policy_gap)
    for banned in ("rsi", "atr", "min_trades", "min_hold", "blacklist", "54.7"):
        assert re.search(rf"\b{banned}\b", src, re.IGNORECASE) is None


def test_queue_features_are_continuous_and_not_a_gate():
    snaps = [
        {"bids": [(100.0, 5.0), (99.0, 4.0)], "asks": [(101.0, 1.0), (102.0, 1.0)]},
        {"bids": [(100.0, 2.0), (99.0, 4.0)], "asks": [(101.0, 3.0), (102.0, 1.0)]},
    ]
    feats = queue_features(snaps)
    assert feats["imbalance_l1"] != 0.0
    assert feats["cancel_pressure"] > 0.0
    for key in ("imbalance_depth", "absorption", "fragility", "book_recovery", "adverse_selection_bps"):
        assert key in feats
