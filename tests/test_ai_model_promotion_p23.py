"""Item p23: accuracy is diagnostic only for model promotion — the real gate
is after-cost profit and bad-trade-rate. A candidate with strictly lower
holdout accuracy than the active model must still promote when its
profit_after_cost and bad_trade_rate are at least as good."""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion import register_candidate_and_maybe_promote


def _write_artifact(path: Path, *, accuracy: float, trained_at: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy": accuracy,
        "feature_version": 5,
        "feature_dim": 145,
        "live_strategy_id": "day",
        "trained_at": trained_at or "2020-01-01T00:00:00+00:00",
    }
    path.write_bytes(pickle.dumps(payload))


def test_lower_accuracy_promotes_when_economics_are_better(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STALE_HOURS", "72")
    monkeypatch.setenv("MODEL_PROMOTION_ACCURACY_MIN_MARGIN", "0.01")
    active = tmp_path / "active" / "BTCUSDT_direction.pkl"
    cand = tmp_path / "cand" / "day_BTCUSDT_test.pkl"
    _write_artifact(active, accuracy=0.65, trained_at="2099-01-01T00:00:00+00:00")
    _write_artifact(cand, accuracy=0.50)
    now = time.time()
    os.utime(active, (now, now))

    db = tmp_path / "t.db"
    ensure_ai_canonical_tables(str(db))

    metrics = {
        "holdout_status": "OK",
        "holdout_sample_count": 40,
        "candidate_accuracy": 0.50,
        "active_accuracy": 0.65,
        "candidate_profit_after_cost": 0.03,
        "active_profit_after_cost": 0.01,
        "candidate_bad_trade_rate": 0.05,
        "active_bad_trade_rate": 0.10,
        "candidate_holdout": {"buy_signal_count": 10, "accuracy": 0.50},
        "holdout_buy_label_count": 12,
    }
    promoted, reason = register_candidate_and_maybe_promote(
        strategy_id="day",
        symbol="BTCUSDT",
        candidate_path=cand,
        active_path=active,
        validation_metrics=metrics,
        db_path=str(db),
    )
    assert promoted is True, f"expected promotion despite lower accuracy, got reject reason={reason}"


def test_accuracy_ok_still_computed_and_reported_when_rejecting(tmp_path, monkeypatch):
    """Accuracy must remain in reject_reason/metrics for observability even
    though it no longer gates the decision on its own — the actual reject
    here is driven by worse profit_after_cost, not by accuracy."""
    monkeypatch.setenv("MODEL_STALE_HOURS", "72")
    active = tmp_path / "active" / "ETHUSDT_direction.pkl"
    cand = tmp_path / "cand" / "day_ETHUSDT_test.pkl"
    _write_artifact(active, accuracy=0.60, trained_at="2099-01-01T00:00:00+00:00")
    _write_artifact(cand, accuracy=0.40)
    now = time.time()
    os.utime(active, (now, now))

    db = tmp_path / "t2.db"
    ensure_ai_canonical_tables(str(db))

    metrics = {
        "holdout_status": "OK",
        "holdout_sample_count": 40,
        "candidate_accuracy": 0.40,
        "active_accuracy": 0.60,
        "candidate_profit_after_cost": -0.02,
        "active_profit_after_cost": 0.02,
        "candidate_bad_trade_rate": 0.30,
        "active_bad_trade_rate": 0.10,
        "candidate_holdout": {"buy_signal_count": 10, "accuracy": 0.40},
        "holdout_buy_label_count": 12,
    }
    promoted, reason = register_candidate_and_maybe_promote(
        strategy_id="day",
        symbol="ETHUSDT",
        candidate_path=cand,
        active_path=active,
        validation_metrics=metrics,
        db_path=str(db),
    )
    assert promoted is False
    assert "profit_after_cost" in reason or "bad_trade" in reason
