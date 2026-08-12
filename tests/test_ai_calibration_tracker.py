"""Item p12: model calibration validation (Brier score, ECE, reliability buckets)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from backend.services import ai_calibration_tracker as act
from backend.services.ai_calibration_tracker import (
    calibration_confidence_multiplier,
    compute_calibration_for_symbol,
    run_calibration_tracking_cycle,
)
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables


def setup_function(_fn):
    act._MULT_CACHE.clear()


def _row(prob_buy: float, outcome_label: int) -> dict:
    return {
        "outcome_label": outcome_label,
        "net_pnl_pct": None,
        "score_components_json": json.dumps({"prob_buy": prob_buy}),
    }


def test_insufficient_samples_is_neutral_not_degraded():
    rows = [_row(0.6, 1) for _ in range(5)]
    result = compute_calibration_for_symbol("BTCUSDT", rows)
    assert result.available is False
    assert result.degraded is False
    assert result.degraded_reason == "insufficient_samples"


def test_perfectly_calibrated_model_has_low_brier_and_ece():
    rows = []
    # 20 rows at predicted 0.9, 90% actually win (18 wins, 2 losses).
    rows += [_row(0.9, 1) for _ in range(18)]
    rows += [_row(0.9, 0) for _ in range(2)]
    result = compute_calibration_for_symbol("ETHUSDT", rows)
    assert result.available is True
    assert result.degraded is False
    assert result.brier_score < 0.1
    assert result.ece < 0.05
    assert result.sample_count == 20


def test_badly_calibrated_model_is_flagged_degraded():
    # Model claims 90% confidence but only wins 20% of the time.
    rows = [_row(0.9, 1) for _ in range(4)] + [_row(0.9, 0) for _ in range(16)]
    result = compute_calibration_for_symbol("SOLUSDT", rows)
    assert result.available is True
    assert result.degraded is True
    assert result.brier_score > 0.3
    assert "brier" in result.degraded_reason or "ece" in result.degraded_reason


def test_confidence_over_100_scale_is_normalized():
    rows = [{"outcome_label": 1, "score_components_json": json.dumps({"confidence": 90.0})} for _ in range(20)]
    result = compute_calibration_for_symbol("XRPUSDT", rows)
    assert result.available is True
    # avg_predicted in buckets should be ~0.9, not 90.0
    assert all(b.avg_predicted <= 1.0 for b in result.buckets)


def test_run_calibration_tracking_cycle_persists_and_sets_gauges(tmp_path, monkeypatch):
    db = tmp_path / "cal.db"
    ensure_ai_canonical_tables(str(db))
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        base = {
            "symbol": "BTCUSDT",
            "opened_at_utc": "2026-08-01T00:00:00Z",
            "outcome_label": 1,
            "score_components_json": json.dumps({"prob_buy": 0.8}),
            "ingested_at_utc": "2026-08-01T00:00:01Z",
        }
        for i in range(25):
            row = dict(base)
            row["closed_at_utc"] = f"2026-08-01T00:{i:02d}:00Z"
            use = {k: v for k, v in row.items() if k in cols}
            conn.execute(
                f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                list(use.values()),
            )
        conn.commit()

    out = run_calibration_tracking_cycle(db_path=str(db), symbols=["BTCUSDT"], lookback_rows=300)
    assert "BTCUSDT" in out
    assert out["BTCUSDT"]["available"] is True
    assert out["BTCUSDT"]["sample_count"] == 25

    with sqlite3.connect(db) as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM ai_calibration_snapshots WHERE symbol='BTCUSDT'").fetchone()[0]
    assert cnt == 1


def test_calibration_confidence_multiplier_neutral_when_no_snapshot(tmp_path):
    db = tmp_path / "empty.db"
    ensure_ai_canonical_tables(str(db))
    mult, reason = calibration_confidence_multiplier("BTCUSDT", db_path=str(db))
    assert mult == 1.0
    assert reason == "no_calibration_snapshot"


def test_calibration_confidence_multiplier_dampens_when_degraded(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIBRATION_DEGRADED_CONFIDENCE_MULT", "0.7")
    db = tmp_path / "deg.db"
    ensure_ai_canonical_tables(str(db))
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO ai_calibration_snapshots
                (symbol, computed_at_utc, sample_count, brier_score, ece, available, degraded, degraded_reason, buckets_json)
            VALUES ('ETHUSDT', ?, 40, 0.4, 0.2, 1, 1, 'brier=0.4>0.28', '[]')
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
    mult, reason = calibration_confidence_multiplier("ETHUSDT", db_path=str(db))
    assert mult == pytest.approx(0.7)
    assert "calibration_degraded" in reason


def test_calibration_confidence_multiplier_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIBRATION_TRACKING_ENABLED", "false")
    db = tmp_path / "off.db"
    ensure_ai_canonical_tables(str(db))
    mult, reason = calibration_confidence_multiplier("BTCUSDT", db_path=str(db))
    assert mult == 1.0
    assert reason == "calibration_tracking_disabled"
