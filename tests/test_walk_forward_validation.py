"""Item p13: purged/embargoed walk-forward validation with after-cost fold metrics."""

from __future__ import annotations

import sqlite3

import pytest

from backend.services import walk_forward_validation as wfv
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables


def test_splits_are_chronologically_ordered_and_non_overlapping():
    n = 200
    open_times = [float(i) for i in range(n)]
    close_times = [float(i) + 1.0 for i in range(n)]  # instant resolution, no overlap risk
    splits = wfv.purged_walk_forward_splits(open_times, close_times, n_splits=4, embargo_frac=0.0)
    assert len(splits) == 4
    for split in splits:
        assert set(split.train_idx).isdisjoint(set(split.test_idx))
        if split.train_idx:
            assert max(open_times[i] for i in split.train_idx) < min(open_times[i] for i in split.test_idx)


def test_purge_removes_overlapping_training_rows():
    # 100 rows; every row's outcome takes 50 "ticks" to resolve, so many
    # training rows near the train/test boundary overlap into the test fold.
    n = 100
    open_times = [float(i) for i in range(n)]
    close_times = [float(i) + 50.0 for i in range(n)]
    splits_no_purge_equivalent = wfv.purged_walk_forward_splits(open_times, [float(i) for i in range(n)], n_splits=4, embargo_frac=0.0)
    splits_with_overlap = wfv.purged_walk_forward_splits(open_times, close_times, n_splits=4, embargo_frac=0.0)
    total_purged = sum(s.purged_count for s in splits_with_overlap)
    total_purged_baseline = sum(s.purged_count for s in splits_no_purge_equivalent)
    assert total_purged > total_purged_baseline
    assert total_purged > 0


def test_embargo_removes_a_slice_near_the_boundary():
    n = 200
    open_times = [float(i) for i in range(n)]
    close_times = [float(i) + 1.0 for i in range(n)]
    no_embargo = wfv.purged_walk_forward_splits(open_times, close_times, n_splits=4, embargo_frac=0.0)
    with_embargo = wfv.purged_walk_forward_splits(open_times, close_times, n_splits=4, embargo_frac=0.05)
    for a, b in zip(no_embargo, with_embargo, strict=True):
        assert b.embargoed_count >= a.embargoed_count
        assert len(b.train_idx) <= len(a.train_idx)


def test_insufficient_rows_returns_empty():
    splits = wfv.purged_walk_forward_splits([1.0, 2.0], [1.5, 2.5], n_splits=5, embargo_frac=0.0)
    assert splits == []


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        wfv.purged_walk_forward_splits([1.0, 2.0, 3.0], [1.0, 2.0], n_splits=1)


def test_after_cost_fold_metrics_computes_real_pnl_stats():
    rows = [
        {"net_pnl_pct": 0.02, "predicted_label": 1, "outcome_label": 1},
        {"net_pnl_pct": 0.02, "predicted_label": 1, "outcome_label": 1},
        {"net_pnl_pct": -0.01, "predicted_label": 1, "outcome_label": 0},
        {"net_pnl_pct": -0.01, "predicted_label": 0, "outcome_label": 0},
    ]
    metrics = wfv.compute_after_cost_fold_metrics(rows, (0, 1, 2, 3))
    assert metrics["n"] == 4
    assert metrics["win_rate"] == 0.5
    assert metrics["profit_factor"] == pytest.approx(0.04 / 0.02)
    assert metrics["total_net_pnl_pct"] == pytest.approx(0.02)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_after_cost_fold_metrics_handles_empty_idx():
    metrics = wfv.compute_after_cost_fold_metrics([{"net_pnl_pct": 0.01}], ())
    assert metrics["n"] == 0
    assert metrics["accuracy"] is None


def test_max_drawdown_on_declining_equity():
    # +1, +1, -3, +1 -> equity path: 1, 2, -1, 0 -> peak 2, trough -1 -> dd = 3
    dd = wfv._max_drawdown([1.0, 1.0, -3.0, 1.0])
    assert dd == pytest.approx(3.0)


def test_run_purged_walk_forward_report_end_to_end():
    n = 150
    rows = []
    for i in range(n):
        rows.append(
            {
                "opened_at_epoch": float(i * 60),
                "closed_at_epoch": float(i * 60 + 30),
                "net_pnl_pct": 0.01 if i % 3 != 0 else -0.02,
                "predicted_label": 1 if i % 3 != 0 else 0,
                "outcome_label": 1 if i % 3 != 0 else 0,
            }
        )
    report = wfv.run_purged_walk_forward_report(rows, n_splits=4, embargo_frac=0.02)
    assert report.available is True
    assert len(report.folds) == 4
    for fold in report.folds:
        assert fold.n_test > 0
        assert fold.accuracy is not None
    mean_pf = report.mean_across_folds("profit_factor")
    assert mean_pf is not None and mean_pf > 0


def test_run_purged_walk_forward_report_degrades_on_missing_timestamps():
    rows = [{"net_pnl_pct": 0.01}] * 20
    report = wfv.run_purged_walk_forward_report(rows, n_splits=2)
    assert report.available is False
    assert "missing_or_invalid_timestamps" in report.degraded_reason


def test_run_purged_walk_forward_report_degrades_on_too_few_rows():
    rows = [{"opened_at_epoch": 1.0, "closed_at_epoch": 2.0, "net_pnl_pct": 0.01}] * 3
    report = wfv.run_purged_walk_forward_report(rows, n_splits=5)
    assert report.available is False
    assert report.degraded_reason == "insufficient_rows_for_requested_splits"


def test_load_and_report_for_symbol_reads_real_outcome_rows(tmp_path):
    db = tmp_path / "wfv.db"
    ensure_ai_canonical_tables(str(db))
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        for i in range(60):
            row = {
                "symbol": "BTCUSDT",
                "opened_at_utc": f"2026-08-01T{i % 24:02d}:{(i * 7) % 60:02d}:00Z",
                "closed_at_utc": f"2026-08-01T{i % 24:02d}:{(i * 7 + 5) % 60:02d}:00Z",
                "strategy_id": "day",
                "net_pnl_pct": 0.01 if i % 4 != 0 else -0.015,
                "outcome_label": 1 if i % 4 != 0 else 0,
                "ingested_at_utc": "2026-08-01T00:00:00Z",
            }
            use = {k: v for k, v in row.items() if k in cols}
            conn.execute(
                f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                list(use.values()),
            )
        conn.commit()

    report = wfv.load_and_report_for_symbol("day", "BTCUSDT", db_path=str(db), n_splits=3)
    assert report.available is True
    assert report.n_rows == 60
    assert len(report.folds) == 3


def test_load_and_report_for_symbol_degrades_gracefully_with_no_rows(tmp_path):
    db = tmp_path / "empty_wfv.db"
    ensure_ai_canonical_tables(str(db))
    report = wfv.load_and_report_for_symbol("day", "ETHUSDT", db_path=str(db), n_splits=3)
    assert report.available is False


def test_report_to_dict_is_json_safe():
    n = 50
    rows = [{"opened_at_epoch": float(i), "closed_at_epoch": float(i) + 1, "net_pnl_pct": 0.01, "outcome_label": 1, "predicted_label": 1} for i in range(n)]
    report = wfv.run_purged_walk_forward_report(rows, n_splits=2)
    payload = report.to_dict()
    assert payload["available"] is True
    assert isinstance(payload["folds"], list)
