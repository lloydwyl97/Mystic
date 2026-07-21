"""AI continuous-improvement wiring: stale-refresh promotion + pattern backfill."""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from pathlib import Path

from backend.services.ai_model_promotion import register_candidate_and_maybe_promote
from backend.services.ai_pattern_memory import backfill_pattern_memory_from_outcomes, build_pattern_vector


def _write_artifact(path: Path, *, accuracy: float, trained_at: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": object(),
        "scaler": None,
        "accuracy": accuracy,
        "feature_version": 5,
        "feature_dim": 145,
        "live_strategy_id": "day",
        "trained_at": trained_at or "2020-01-01T00:00:00+00:00",
    }
    # Minimal stand-in: promotion only reads meta fields via pickle.
    path.write_bytes(pickle.dumps({k: v for k, v in payload.items() if k != "model"}))


def test_stale_tie_promotes_equal_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STALE_HOURS", "1")
    monkeypatch.setenv("MODEL_PROMOTION_ACCURACY_MIN_MARGIN", "0.01")
    active = tmp_path / "active" / "BTCUSDT_direction.pkl"
    cand = tmp_path / "cand" / "day_BTCUSDT_test.pkl"
    _write_artifact(active, accuracy=0.55)
    # Make mtime old
    old = time.time() - 10 * 3600
    Path(active).touch()
    import os

    os.utime(active, (old, old))
    _write_artifact(cand, accuracy=0.55)

    db = tmp_path / "t.db"
    # Ensure promotion tables exist via ensure path
    from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

    ensure_ai_canonical_tables(str(db))

    metrics = {
        "holdout_status": "OK",
        "holdout_sample_count": 40,
        "candidate_accuracy": 0.55,
        "active_accuracy": 0.55,
        "candidate_profit_after_cost": 0.01,
        "active_profit_after_cost": 0.01,
        "candidate_bad_trade_rate": 0.2,
        "active_bad_trade_rate": 0.2,
        "candidate_holdout": {"buy_signal_count": 10, "accuracy": 0.55},
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
    assert promoted is True
    assert reason == "stale_refresh_tie"


def test_fresh_tie_still_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STALE_HOURS", "72")
    active = tmp_path / "active" / "ETHUSDT_direction.pkl"
    cand = tmp_path / "cand" / "day_ETHUSDT_test.pkl"
    _write_artifact(active, accuracy=0.6, trained_at="2099-01-01T00:00:00+00:00")
    _write_artifact(cand, accuracy=0.6)
    # Fresh mtime
    now = time.time()
    import os

    os.utime(active, (now, now))

    db = tmp_path / "t2.db"
    from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

    ensure_ai_canonical_tables(str(db))
    metrics = {
        "holdout_status": "OK",
        "holdout_sample_count": 40,
        "candidate_accuracy": 0.6,
        "active_accuracy": 0.6,
        "candidate_profit_after_cost": 0.02,
        "active_profit_after_cost": 0.02,
        "candidate_bad_trade_rate": 0.1,
        "active_bad_trade_rate": 0.1,
        "candidate_holdout": {"buy_signal_count": 8, "accuracy": 0.6},
        "holdout_buy_label_count": 10,
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
    assert reason == "candidate_not_improved_over_active"


def test_pattern_memory_backfill_from_outcomes(tmp_path):
    db = tmp_path / "mem.db"
    from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

    ensure_ai_canonical_tables(str(db))
    with sqlite3.connect(db) as conn:
        comps = {
            "trend_score": 0.7,
            "chop_score": 0.2,
            "coin_expectancy": 0.1,
            "ai_confidence": 0.8,
        }
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        base = {
            "symbol": "BTC/USDT",
            "strategy_id": "day",
            "closed_at_utc": "2026-07-19T00:00:00Z",
            "opened_at_utc": "2026-07-18T00:00:00Z",
            "hold_seconds": 3600,
            "net_pnl_pct": 0.01,
            "score_components_json": json.dumps(comps),
            "ingested_at_utc": "2026-07-19T00:00:01Z",
            "entry_price": 1.0,
            "exit_price": 1.01,
            "realized_pct": 0.01,
            "outcome_label": "WIN",
            "outcome_class": "good",
        }
        use = {k: v for k, v in base.items() if k in cols}
        conn.execute(
            f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
            list(use.values()),
        )
        base2 = dict(base)
        base2.update(
            {
                "symbol": "ETH/USDT",
                "closed_at_utc": "2026-07-19T01:00:00Z",
                "opened_at_utc": "2026-07-18T01:00:00Z",
                "hold_seconds": 1800,
                "net_pnl_pct": -0.02,
                "realized_pct": -0.02,
                "outcome_label": "LOSS",
                "outcome_class": "bad",
                "ingested_at_utc": "2026-07-19T01:00:01Z",
            }
        )
        use2 = {k: v for k, v in base2.items() if k in cols}
        conn.execute(
            f"INSERT INTO ai_outcome_training_rows ({', '.join(use2)}) VALUES ({', '.join('?' for _ in use2)})",
            list(use2.values()),
        )
        conn.commit()

    stats = backfill_pattern_memory_from_outcomes(db_path=str(db), limit=50)
    assert stats["written_good"] >= 1
    assert stats["written_bad"] >= 1
    vec = build_pattern_vector(chop_score=0.2, coin_edge_score=0.1, trend_score=0.7, confidence=0.8)
    assert vec["trend_score"] == 0.7
