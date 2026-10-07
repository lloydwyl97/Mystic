"""Short-horizon label alignment, trajectory weights, and full-state research rows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from backend.services import counterfactual_continuation as cc
from backend.services import edge_research as er
from backend.services import full_state_research as fs
from backend.services.book_queue_capture import BookQueueCapture, queue_features_asof
from backend.services.horizon_alignment import BAR_CLOSE_SOURCE, align_observation, bar_close_known_at, executable_gross, rebuild_disposition
from backend.services.sqlite_large_table_retention import RETENTION_JUSTIFICATION, RETENTION_POLICIES

ROOT = Path(__file__).resolve().parents[1]


def test_30s_label_cannot_use_a_minute_close():
    decision = 1_791_000_000.0
    target = decision + 30
    late_close = bar_close_known_at(decision, 60.0)
    assert late_close > target
    leaked = align_observation([(late_close, 101.0, BAR_CLOSE_SOURCE)], target, 30)
    assert leaked["status"] == "MISSING"
    exact_bar = align_observation([(target, 101.0, BAR_CLOSE_SOURCE)], target, 30)
    assert exact_bar["status"] == "MISSING"
    book = align_observation([(target - 2.0, 100.5, "book_bid"), (late_close, 999.0, BAR_CLOSE_SOURCE)], target, 30)
    assert book["status"] == "OK" and book["source"] == "book_bid" and book["price"] == 100.5
    assert book["timing_error_sec"] <= 0


def test_60s_label_cannot_use_a_later_bar():
    decision = 1_791_000_000.0
    target = decision + 60
    next_close = bar_close_known_at(decision + 60.0, 60.0)
    assert align_observation([(next_close, 101.0, BAR_CLOSE_SOURCE)], target, 60)["status"] == "MISSING"
    assert align_observation([(target, 101.0, BAR_CLOSE_SOURCE)], target, 60)["status"] == "MISSING"
    late_book = align_observation([(target + 5.0, 101.0, "book_bid")], target, 60)
    assert late_book["status"] == "MISSING"
    ok = align_observation([(target - 3.0, 100.2, "book_bid")], target, 60)
    assert ok["status"] == "OK" and ok["obs_ts"] <= target


def test_missing_horizon_stays_missing():
    target = 5_000.0
    only_later = [(target + 1.0, 100.0, "book_bid")]
    too_early = [(target - 30.0, 100.0, "book_bid")]
    assert align_observation(only_later, target, 30)["status"] == "MISSING"
    assert align_observation(too_early, target, 30)["status"] == "MISSING"
    assert align_observation(only_later + too_early, target, 30)["status"] == "MISSING"
    assert rebuild_disposition(True, "MISSING") == "removed"
    assert rebuild_disposition(True, "OK") == "replaced"
    assert rebuild_disposition(False, "MISSING") == "unchanged"


def test_no_observation_after_the_horizon_is_selected():
    target = 8_000.0
    got = align_observation([(target - 4.0, 10.0, "book_bid"), (target + 0.5, 99.0, "book_bid")], target, 30)
    assert got["obs_ts"] == target - 4.0 and got["price"] == 10.0
    assert got["timing_error_sec"] <= 0


def test_continuation_advantage_does_not_read_a_later_print():
    ts = np.array([100.0, 160.0, 195.0])
    path = cc.Path(ts, np.array([100.0, 100.0, 130.0]))
    rows = cc.states_for(path, engine="SCALP_V2", symbol="BTC", entry_t=100.0, entry_px=100.0, cost=0.0006, step_sec=60.0, max_age_sec=120.0, horizons=(30,), match_tol_sec=10.0)
    assert rows == [] or "30" not in rows[0]["advantage"]


def test_one_trajectory_cannot_dominate_by_row_count():
    keys = ["A"] * 100 + ["B"]
    w = er.trajectory_weights(keys)
    assert abs(float(w[:100].sum()) - 1.0) < 1e-9
    assert abs(float(w[100]) - 1.0) < 1e-9
    assert abs(er.effective_independent_weight(w) - 2.0) < 1e-9
    y = np.array([1.0] * 100 + [-1.0])
    score = np.array([1.0] * 100 + [-1.0])
    assert er.spearman(score, y) == 1.0
    weighted = er.weighted_spearman(score, y, w)
    assert weighted is not None and weighted > 0


def test_full_state_row_exists_without_a_setup_and_label_is_market_only(tmp_path):
    db = str(tmp_path / "s.db")
    fs.record_state(
        db,
        engine=fs.DAY_ENGINE,
        symbol="BTCUSDT",
        decision_ts=1_000.0,
        feature_ts=990.0,
        features={"price": 1.0},
        setup_label=fs.NO_SETUP,
        context={},
        entry_ask=100.0,
        entry_bid=99.9,
        entry_obs_ts=1_000.0,
        entry_source="book_ask",
        cost=0.0006,
    )
    with sqlite3.connect(db) as conn:
        row = conn.execute(f"SELECT setup_label, features_json FROM {fs.STATES}").fetchone()
    assert row[0] == "NO_SETUP"
    assert json.loads(row[1])["price"] == 1.0

    def book(_db, _sym, target, _h):
        return [(target - 1.0, 101.0, "book_bid")]

    assert fs.resolve_market_labels(db, fs.DAY_ENGINE, now=1_000.0 + 900, observe=book, limit=5) == 1
    with sqlite3.connect(db) as conn:
        label = conn.execute(f"SELECT status, gross, kind, source FROM {fs.LABELS}").fetchone()
    assert label[0] == "OK" and label[2] == fs.MARKET_KIND and label[3] == "book_bid"
    assert abs(label[1] - executable_gross(100.0, 101.0)) < 1e-12


def test_short_scalp_label_rejects_bar_close_and_engines_do_not_share_a_queue(tmp_path):
    db = str(tmp_path / "q.db")
    for i in range(6):
        fs.record_state(
            db,
            engine=fs.DAY_ENGINE,
            symbol="ETHUSDT",
            decision_ts=float(i),
            feature_ts=float(i),
            features={},
            setup_label=fs.NO_SETUP,
            context={},
            entry_ask=10.0,
            entry_bid=9.9,
            entry_obs_ts=float(i),
            entry_source="book_ask",
            cost=0.0006,
        )
    fs.record_state(
        db,
        engine=fs.SCALP_ENGINE,
        symbol="ETHUSDT",
        decision_ts=50.0,
        feature_ts=50.0,
        features={},
        setup_label=fs.NO_SETUP,
        context={},
        entry_ask=10.0,
        entry_bid=9.9,
        entry_obs_ts=50.0,
        entry_source="book_ask",
        cost=0.0006,
    )

    def bar_only(_db, _sym, target, _h):
        return [(target, 11.0, BAR_CLOSE_SOURCE)]

    assert fs.resolve_market_labels(db, fs.SCALP_ENGINE, now=50.0 + 30, observe=bar_only, limit=10) == 1
    with sqlite3.connect(db) as conn:
        scalp = conn.execute(f"SELECT status, source FROM {fs.LABELS} l JOIN {fs.STATES} s ON s.id=l.state_id WHERE s.engine=?", (fs.SCALP_ENGINE,)).fetchone()
        day = conn.execute(f"SELECT COUNT(*) FROM {fs.LABELS} l JOIN {fs.STATES} s ON s.id=l.state_id WHERE s.engine=?", (fs.DAY_ENGINE,)).fetchone()[0]
    assert scalp[0] == "MISSING" and scalp[1] is None
    assert day == 0


def test_setup_annotation_does_not_create_the_row(tmp_path):
    db = str(tmp_path / "a.db")
    fs.annotate_setup(db, fs.DAY_ENGINE, "SOLUSDT", 10.0, "RANGE_BOUNCE")
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {fs.STATES}").fetchone()[0] == 0
    fs.record_state(
        db,
        engine=fs.DAY_ENGINE,
        symbol="SOLUSDT",
        decision_ts=10.0,
        feature_ts=10.0,
        features={"a": 1.0},
        setup_label=fs.NO_SETUP,
        context={},
        entry_ask=None,
        entry_bid=None,
        entry_obs_ts=None,
        entry_source=None,
        cost=0.0006,
    )
    fs.annotate_setup(db, fs.DAY_ENGINE, "SOLUSDT", 10.0, "RANGE_BOUNCE")
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT setup_label FROM {fs.STATES}").fetchone()[0] == "RANGE_BOUNCE"


def test_within_decision_rank_is_separate_from_the_time_trend():
    rows = [
        er.Row(t=0, group=0, symbol="A", x=np.zeros(1), label=0.01, label_ts=10),
        er.Row(t=0, group=0, symbol="B", x=np.zeros(1), label=0.00, label_ts=10),
        er.Row(t=100, group=1, symbol="A", x=np.zeros(1), label=1.00, label_ts=110),
        er.Row(t=100, group=1, symbol="B", x=np.zeros(1), label=0.50, label_ts=110),
    ]
    scores = [0.0, 1.0, 2.0, 3.0]
    labels = [r.label for r in rows]
    assert er.spearman(scores, labels) > 0
    within = er.ranking_metrics(rows, scores)
    assert within["chosen"] < within["point_mean"]
    assert within["regret"] > 0
    assert within["best"] > within["chosen"]


def test_scalp_target_is_executable_bid_over_ask():
    assert abs(executable_gross(100.0, 100.1) - 0.001) < 1e-12
    assert executable_gross(100.0, 99.9) < 0


def test_queue_features_ignore_updates_after_the_decision(tmp_path):
    db = str(tmp_path / "b.db")
    cap = BookQueueCapture(db)
    t0 = 1_791_000_000.0
    bids = [[100.0 - i, 1.0] for i in range(5)]
    asks = [[100.5 + i, 1.0] for i in range(5)]
    thin = [[bids[0][0], 0.2], *bids[1:]]
    gone = [[99.0, 5.0], *bids[1:]]
    for k in range(500):
        cap.record("BTC", thin if k == 200 else bids, asks, k + 1, ts=t0 + k * 0.1)
    cap.record("BTC", bids, asks, 600, ts=t0 + 59.0)
    cap.record("BTC", gone, asks, 700, ts=t0 + 70.0)
    cap.record("BTC", gone, asks, 701, ts=t0 + 119.0)
    cap.flush_all()
    cap.drain()
    at = queue_features_asof(db, "BTC", t0 + 50.0, lookback_sec=60.0)
    assert at["bid_remove"] > 0
    assert at["bid_depletions"] == 0


def test_retention_covers_full_state_tables():
    policies = {p.table: p for p in RETENTION_POLICIES}
    assert policies["research_market_states"].ts_column == "decision_ts"
    assert policies["research_market_labels"].keep_days == 7
    assert "research_market_states" in RETENTION_JUSTIFICATION
    plan = fs.storage_plan()
    assert plan["gb_per_day"] < 0.1
    assert plan["scalp_rows_per_hour"] == 480


def test_live_path_has_no_research_count_gate():
    live = (ROOT / "backend/services/portfolio_engine_integration.py").read_text()
    research = (ROOT / "backend/services/full_state_research.py").read_text()
    assert "tick_day_research" in live and "tick_scalp_research" in live
    assert "research_rows" not in research
    assert "disable" not in research
    assert "NO_SETUP" in research
