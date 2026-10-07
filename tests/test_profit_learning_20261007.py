"""Live short-horizon marks, two-sided calibration, and trajectory weights."""

from __future__ import annotations

import inspect
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from backend.services import adaptive_learning as al
from backend.services import economic_state_rebuild as esr
from backend.services.book_queue_capture import queue_features
from backend.services.continuation_backfill import DAY_HORIZONS_SEC, Position, Snapshot, advantage_labels_for, fold_recent_advantages
from backend.services.continuation_surface import record_advantage
from backend.services.economic_replay import RepairedDayPolicy
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
    assert (
        al.learn_from_close(
            db,
            engine=DAY,
            symbol="BTCUSDT",
            setup="RANGE_BOUNCE",
            regime="r",
            strategy_version=al.current_strategy_version(DAY),
            net_pct=-0.02,
            mfe_pct=None,
            mae_pct=None,
            hold_min=1.0,
            continuation=None,
            version_current=True,
            is_dust=False,
            entered_at=al.anchor_epoch(DAY) + 10.0,
            exit_reason="TIME_STOP_EXIT",
            candidate_id=1,
        )
        is False
    )
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
    src = inspect.getsource(al.executable_bid_at) + inspect.getsource(al.policy_calibration) + inspect.getsource(al._learn_policy_gap) + inspect.getsource(al._learn_policy_calibration)
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


def _calibration_row(db: str) -> tuple[float, float]:
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()
    assert row is not None
    return float(row[0]), float(row[1])


def _flags(db: str, row_id: int) -> tuple[int, int]:
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT calibration_learned, policy_learned FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
    return int(row[0]), int(row[1])


def test_calibration_updates_at_close_without_a_lifecycle_label(tmp_path):
    db = str(tmp_path / "t.db")
    setup, regime = "EXHAUSTION_MR__MARKET", "btcdown_vollo"
    predicted, realized = 0.002742, 0.001092
    before = al.day_decision(db, "XRPUSDT", setup, regime)["expected_net"]
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="XRPUSDT",
        setup=setup,
        regime=regime,
        ref_price=1.4,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": predicted, "market_alpha": 0.0, "policy_gap": 0.0},
        opportunity_id="CLOSE-1",
    )
    assert al.mark_candidate_filled(db, row_id)
    assert al.record_policy_outcome(db, engine=DAY, opportunity_id="CLOSE-1", net_pct=realized, candidate_id=row_id, now=T0 + 40.0)
    calibrated, gapped = _flags(db, row_id)
    assert calibrated == 1 and gapped == 0
    n, ewma = _calibration_row(db)
    assert abs(n - 1.0) < 1e-9
    assert abs(ewma - (realized - predicted)) < 1e-12
    after = al.day_decision(db, "XRPUSDT", setup, regime)["expected_net"]
    assert after < before
    predicted_win = after
    realized_win = predicted_win + 0.004
    row_win = al.record_candidate(
        db,
        engine=DAY,
        symbol="XRPUSDT",
        setup=setup,
        regime=regime,
        ref_price=1.4,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0 + 900.0,
        economic={"policy_value": predicted_win},
        opportunity_id="CLOSE-2",
    )
    assert al.mark_candidate_filled(db, row_win)
    assert al.record_policy_outcome(db, engine=DAY, opportunity_id="CLOSE-2", net_pct=realized_win, candidate_id=row_win, now=T0 + 940.0)
    assert al.day_decision(db, "XRPUSDT", setup, regime)["expected_net"] > after


def test_lifecycle_resolution_learns_the_gap_once_and_does_not_recalibrate(tmp_path):
    db = str(tmp_path / "t.db")
    predicted, realized, market = 0.0018, -0.0005, -0.008
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="btcdown_vollo",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": predicted},
        opportunity_id="GAP-1",
    )
    assert al.mark_candidate_filled(db, row_id)
    assert al.record_policy_outcome(db, engine=DAY, opportunity_id="GAP-1", net_pct=realized, candidate_id=row_id, now=T0 + 50.0)
    n0, ewma0 = _calibration_row(db)
    assert _flags(db, row_id) == (1, 0)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?",
            (json.dumps({"lifecycle": {"net": market, "reason": "HORIZON_MARK", "minutes": 240}}), row_id),
        )
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=T0 + 400 * 60)
    n1, ewma1 = _calibration_row(db)
    assert abs(n1 - n0) < 1e-12 and abs(ewma1 - ewma0) < 1e-12
    assert _flags(db, row_id) == (1, 1)
    with sqlite3.connect(db) as conn:
        gap_n, gap = conn.execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='policy_gap'").fetchone()
    assert abs(float(gap_n) - 1.0) < 1e-6
    assert abs(float(gap) - (realized - market)) < 1e-12
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=T0 + 500 * 60)
    n2, ewma2 = _calibration_row(db)
    assert abs(n2 - n1) < 1e-12 and abs(ewma2 - ewma1) < 1e-12
    with sqlite3.connect(db) as conn:
        gap_n2 = conn.execute("SELECT n FROM adaptive_metric_state WHERE metric='policy_gap'").fetchone()[0]
    assert abs(float(gap_n2) - float(gap_n)) < 1e-12


def test_backfill_teaches_a_missed_close_once_and_skips_one_already_learned(tmp_path):
    floor = datetime.fromisoformat(al.CALIBRATION_BACKFILL_FROM.replace("Z", "+00:00")).timestamp()
    db = str(tmp_path / "t.db")
    predicted, realized = 0.001486, -0.000494
    entered = floor + 200.0
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="btcdown_vollo",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=entered - 20.0,
        economic={"policy_value": predicted},
        opportunity_id="MISS",
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=? WHERE id=?", (realized, row_id))
        conn.execute(
            "CREATE TABLE paper_trades (symbol TEXT, side TEXT, engine_id TEXT, timestamp TEXT, entry_timestamp TEXT, exit_reason TEXT)"
        )
        conn.execute(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?)",
            ("BTC/USDT", "SELL", DAY, "2026-10-07T15:40:00Z", "2026-10-07T15:28:45Z", "LEARNED_CONTINUATION_EXIT"),
        )
    assert al.backfill_close_calibration(db) == 1
    assert _flags(db, row_id) == (1, 0)
    _n, ewma = _calibration_row(db)
    assert abs(ewma - (realized - predicted)) < 1e-12
    assert al.backfill_close_calibration(db) == 0
    _n2, ewma2 = _calibration_row(db)
    assert abs(ewma2 - ewma) < 1e-12


def test_backfill_ignores_a_close_before_the_repair_and_a_retired_exit(tmp_path):
    db = str(tmp_path / "t.db")
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="ETHUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="r",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": 0.002},
        opportunity_id="OLD",
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=? WHERE id=?", (-0.001, row_id))
        conn.execute(
            "CREATE TABLE paper_trades (symbol TEXT, side TEXT, engine_id TEXT, timestamp TEXT, entry_timestamp TEXT, exit_reason TEXT)"
        )
        conn.execute(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?)",
            ("ETH/USDT", "SELL", DAY, "2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z", "LEARNED_CONTINUATION_EXIT"),
        )
    assert al.backfill_close_calibration(db) == 0
    assert _flags(db, row_id) == (0, 0)


def test_rows_that_already_learned_the_gap_are_not_calibrated_again(tmp_path):
    db = str(tmp_path / "t.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE adaptive_candidate_markouts (id INTEGER PRIMARY KEY, engine_id TEXT, symbol TEXT, setup TEXT, regime TEXT, "
            "strategy_version TEXT, signaled INTEGER, ref_price REAL, roundtrip_cost REAL, evaluated_at REAL, markouts_json TEXT, "
            "features_json TEXT, learned INTEGER, resolved INTEGER, label_horizon REAL, policy_learned INTEGER, realized_net REAL, economic_json TEXT)"
        )
        conn.execute(
            "INSERT INTO adaptive_candidate_markouts VALUES (1,'DAY_V2','BTCUSDT','S','r','v',1,1,0,1,'{}','{}',0,0,0,1,-0.01,'{\"policy_value\":0.002}')"
        )
    al._connect(db).close()
    with sqlite3.connect(db) as conn:
        flag = conn.execute("SELECT calibration_learned FROM adaptive_candidate_markouts WHERE id=1").fetchone()[0]
        metrics = conn.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()[0]
    assert int(flag) == 1
    assert int(metrics) == 0


def test_rebuild_reproduces_calibration_without_a_lifecycle_label(tmp_path):
    live = str(tmp_path / "live.db")
    scratch = str(tmp_path / "scratch.db")
    entered = datetime(2026, 10, 7, 16, 0, tzinfo=timezone.utc).timestamp()
    closed = entered + 45.0
    predicted, realized = 0.001459, -0.000672
    decision = {"setup": "BREAKOUT_CONTINUATION__MARKET", "regime": "btcdown_vollo", "economic": {"policy_value": predicted}}
    row_id = al.record_candidate(
        live,
        engine=DAY,
        symbol="XRPUSDT",
        setup=decision["setup"],
        regime=decision["regime"],
        ref_price=1.4,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=entered - 15.0,
        economic=decision["economic"],
        opportunity_id="RB",
    )
    assert al.mark_candidate_filled(live, row_id)
    assert al.learn_from_close(
        live,
        engine=DAY,
        symbol="XRPUSDT",
        setup=decision["setup"],
        regime=decision["regime"],
        strategy_version=al.current_strategy_version(DAY),
        net_pct=realized,
        mfe_pct=0.001,
        mae_pct=0.001,
        hold_min=0.75,
        continuation=None,
        version_current=True,
        is_dust=False,
        entered_at=entered,
        now=closed,
        opportunity_id="RB",
        exit_reason="LEARNED_CONTINUATION_EXIT",
        candidate_id=row_id,
    )
    al._connect(scratch).close()
    fill = {
        "symbol": "XRPUSDT",
        "setup": decision["setup"],
        "regime": decision["regime"],
        "decision": decision,
        "strategy_version": al.current_strategy_version(DAY),
        "entered_at": entered,
        "closed_at": closed,
        "entry_price": 1.4,
        "net": realized,
        "mfe": 0.001,
        "mae": 0.001,
        "minutes": 0.75,
        "high": 1.41,
        "exit_reason": "LEARNED_CONTINUATION_EXIT",
    }

    class _Store:
        def minute_path(self, _symbol, _start, _end):
            return []

    rebuilt_day = esr.rebuild_day([], [fill], _Store(), RepairedDayPolicy(scratch), roundtrip_cost=0.002, now=closed + 10.0)
    assert rebuilt_day["counts"]["policy_calibrations"] == 1 and rebuilt_day["counts"]["policy_gaps"] == 0
    live_row = _calibration_row(live)
    scratch_row = _calibration_row(scratch)
    assert abs(live_row[0] - scratch_row[0]) < 1e-9
    assert abs(live_row[1] - scratch_row[1]) < 1e-12
    same = al.day_decision(live, "XRPUSDT", decision["setup"], decision["regime"], now=closed + 10.0)
    rebuilt = al.day_decision(scratch, "XRPUSDT", decision["setup"], decision["regime"], now=closed + 10.0)
    assert abs(same["economic"]["policy_calibration"] - rebuilt["economic"]["policy_calibration"]) < 1e-12
    assert abs(same["economic"]["policy_value"] - rebuilt["economic"]["policy_value"]) < 1e-12


def test_post_exit_horizons_update_the_hold_learner_when_they_resolve(tmp_path):
    db = str(tmp_path / "c.db")
    assert DAY_HORIZONS_SEC == (15 * 60, 30 * 60, 60 * 60, 2 * 60 * 60, 4 * 60 * 60, 6 * 60 * 60)
    integration = inspect.getsource(fold_recent_advantages)
    assert "advantage_labels_for" in integration
    from backend.services import portfolio_engine_integration as integration_mod

    assert "fold_recent_advantages(db_path)" in inspect.getsource(integration_mod)
    entry, exit_t = 1_000_000.0, 1_000_040.0
    position = Position(
        engine=DAY,
        trade_id="T15",
        symbol="XRPUSDT",
        setup="EXHAUSTION_MR__MARKET",
        regime="btcdown_vollo",
        entry_price=1.43,
        entry_time=entry,
        exit_time=exit_t,
        exit_price=1.43,
        exit_reason="LEARNED_CONTINUATION_EXIT",
        atr=0.0,
        anchor=1.4,
        closed=True,
        snapshots=[Snapshot(time=exit_t, net=0.001, mark=1.43, mfe=0.002, mae=0.0, hold_sec=40.0, high_water=0.002)],
    )
    later = 1.43 * 1.004
    early = advantage_labels_for(position, [(exit_t + 900.0 - 20.0, later, later, later, later)], as_of=exit_t + 960.0)[0]
    assert [item.horizon for item in early] == [900]
    assert early[0].advantage > 0
    thirty = advantage_labels_for(
        position,
        [(exit_t + 900.0 - 20.0, later, later, later, later), (exit_t + 1800.0 - 20.0, later, later, later, later)],
        as_of=exit_t + 1860.0,
    )[0]
    assert {item.horizon for item in thirty} == {900, 1800}
    before = al.estimate(db, DAY, "XRPUSDT", "EXHAUSTION_MR__MARKET", "btcdown_vollo", "hold_adv_900")["n"]
    assert record_advantage(
        db,
        engine=DAY,
        symbol="XRPUSDT",
        setup="EXHAUSTION_MR__MARKET",
        regime="btcdown_vollo",
        horizon=900,
        features=early[0].features,
        advantage=early[0].advantage,
        weight=early[0].weight,
        now=exit_t + 900.0,
    )
    after = al.estimate(db, DAY, "XRPUSDT", "EXHAUSTION_MR__MARKET", "btcdown_vollo", "hold_adv_900")
    assert after["n"] > before
    assert re.search(r"\bmin_hold\b", inspect.getsource(advantage_labels_for), re.IGNORECASE) is None


def _calibration_state(db: str) -> tuple[int, float | None, float | None]:
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT COUNT(*), SUM(n), SUM(ewma) FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()
    count = int(row[0] or 0)
    if count == 0:
        return 0, None, None
    return count, float(row[1]), float(row[2])


def test_a_crash_between_the_flag_and_the_observation_learns_once_for_both_engines(tmp_path):
    predicted, realized = 0.002, -0.001
    cases = ((DAY, "BTCUSDT", "EXHAUSTION_MR__MARKET"), (SCALP, "ETHUSDT", "CLAIM"))
    real_fold = al._fold_observation
    for engine, symbol, setup in cases:
        db = str(tmp_path / f"{engine}.db")
        row_id = al.record_candidate(
            db,
            engine=engine,
            symbol=symbol,
            setup=setup,
            regime="btcdown_vollo",
            ref_price=100.0,
            roundtrip_cost=0.002,
            signaled=True,
            evaluated_at=T0,
            economic={"policy_value": predicted},
            opportunity_id=f"CRASH-{engine}",
        )
        assert al.mark_candidate_filled(db, row_id)

        def boom(conn, _db=db, _row_id=row_id, **kwargs):
            del conn, kwargs
            other = sqlite3.connect(_db)
            try:
                flag = other.execute("SELECT calibration_learned FROM adaptive_candidate_markouts WHERE id=?", (_row_id,)).fetchone()[0]
                stored = other.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()[0]
            finally:
                other.close()
            assert int(flag) == 0
            assert int(stored) == 0
            raise RuntimeError("crash between flag and observation")

        al._fold_observation = boom
        try:
            try:
                al.record_policy_outcome(db, engine=engine, opportunity_id=f"CRASH-{engine}", net_pct=realized, now=T0, candidate_id=row_id)
            except RuntimeError as exc:
                assert "crash between flag and observation" in str(exc)
            else:
                raise AssertionError(engine)
        finally:
            al._fold_observation = real_fold
        with sqlite3.connect(db) as conn:
            flag, net = conn.execute("SELECT calibration_learned, realized_net FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
        assert int(flag) == 0
        assert net is not None
        assert _calibration_state(db)[0] == 0
        conn = al._connect(db)
        try:
            row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
            assert al._learn_policy_calibration(conn, db, row, realized, T0) is True
            row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
            assert al._learn_policy_calibration(conn, db, row, realized, T0) is False
        finally:
            conn.close()
        count, sample_n, ewma = _calibration_state(db)
        assert count == 1
        assert sample_n is not None and abs(sample_n - 1.0) < 1e-9
        assert ewma is not None and abs(ewma - (realized - predicted)) < 1e-12
        with sqlite3.connect(db) as conn:
            flag = conn.execute("SELECT calibration_learned FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()[0]
        assert int(flag) == 1


def test_a_flagged_close_without_an_observation_is_folded_once(tmp_path):
    db = str(tmp_path / "repair.db")
    predicted, realized = 0.003, -0.002
    entered = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc).timestamp()
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="RANGE_BOUNCE__MARKET",
        regime="btcflat_volhi",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=entered,
        economic={"policy_value": predicted},
        opportunity_id="REPAIR",
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=?, calibration_learned=1 WHERE id=?", (realized, row_id))
        conn.execute("CREATE TABLE paper_trades (symbol TEXT, side TEXT, engine_id TEXT, timestamp TEXT, entry_timestamp TEXT, exit_reason TEXT)")
        conn.execute(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?)",
            ("BTC/USDT", "SELL", DAY, "2026-10-06T12:05:00Z", "2026-10-06T12:00:00Z", "LEARNED_CONTINUATION_EXIT"),
        )
    al._calibration_backfilled.discard(os.path.abspath(db))
    al.backfill_close_calibration(db)
    count, sample_n, ewma = _calibration_state(db)
    assert count == 1
    assert sample_n is not None and abs(sample_n - 1.0) < 1e-9
    assert ewma is not None and abs(ewma - (realized - predicted)) < 1e-12
    assert _flags(db, row_id) == (1, 0)
    al._calibration_backfilled.discard(os.path.abspath(db))
    assert al.backfill_close_calibration(db) == 0
    count2, sample_n2, ewma2 = _calibration_state(db)
    assert count2 == 1
    assert sample_n2 == sample_n
    assert ewma2 == ewma


def test_a_flag_without_a_forecast_is_cleared_and_an_existing_observation_is_not_duplicated(tmp_path):
    db = str(tmp_path / "clear.db")
    version = al.current_strategy_version(DAY)
    assert al.observe(
        db,
        engine=DAY,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        metric="policy_calibration",
        value=-0.004,
        strategy_version=version,
        now=T0,
    )
    kept = al.record_candidate(
        db,
        engine=DAY,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        economic={"policy_value": 0.001},
        opportunity_id="KEPT",
    )
    assert al.mark_candidate_filled(db, kept)
    empty = al.record_candidate(
        db,
        engine=DAY,
        symbol="SOLUSDT",
        setup="RANGE_BOUNCE__MARKET",
        regime="btcflat_vollo",
        ref_price=100.0,
        roundtrip_cost=0.002,
        signaled=True,
        evaluated_at=T0,
        opportunity_id="EMPTY",
    )
    assert al.mark_candidate_filled(db, empty)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=?, calibration_learned=1 WHERE id IN (?, ?)", (-0.001, kept, empty))
    before = _calibration_state(db)
    al._calibration_backfilled.discard(os.path.abspath(db))
    al.backfill_close_calibration(db)
    assert _calibration_state(db) == before
    assert _flags(db, kept) == (1, 0)
    assert _flags(db, empty) == (0, 0)
