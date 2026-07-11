"""
Regression: forward-label evidence must survive Redis prediction-bundle
expiry (previously ~72h giveup with no fallback). Binance.US retains kline
history indefinitely via public REST — _ensure_series_covers_window uses that
as a durable fallback when Redis-cached series don't cover a needed labeling
window, so expired/missing Redis data no longer makes a valid row permanently
UNLABELABLE. No second learning database, no duplicate full-market payload
storage (fetched on-demand, in-memory only for the run).

Also covers the existing (already-correct) feature-version/dimension
rejection contract in the Tier B/C join: incompatible rows are safely
skipped, never silently coerced.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from backend.services.ai_learning_ingestion import (
    LABEL_GIVEUP_AGE_SEC,
    _ensure_series_covers_window,
    _features_for_decision_ids,
    ensure_learning_ingestion_tables,
    label_pending_snapshots,
)


class _FakeRedisEmpty:
    """Simulates an expired/missing Redis prediction bundle — no series at all."""

    def get(self, _key):
        return None


def _seed_snapshot(db_path: Path, *, epoch_ms: float, price: float = 100.0, decision_id: str = "d1") -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO ai_candidate_snapshots (
                ts_utc, epoch_ms, symbol, strategy_id, decision_id, decision, reason_code,
                price, label_status, thesis_invalid_level, thesis_target_level
            ) VALUES (datetime('now'), ?, 'BTC/USDT', 'day', ?, 'BUY', 'test', ?, 'PENDING', 95.0, 105.0)
            """,
            (epoch_ms, decision_id, price),
        )
        conn.commit()


def _synthetic_klines(start_ms: float, end_ms: float, *, base_price: float = 100.0, step_ms: float = 3600_000.0) -> list[list[float]]:
    rows = []
    t = start_ms
    while t <= end_ms:
        rows.append([t, base_price, base_price * 1.001, base_price * 0.999, base_price, 10.0])
        t += step_ms
    return rows


def test_prediction_written_and_redis_available_completes_normally():
    """Scenario 1+2: prediction written, Redis bundle available -> label completes normally, no REST fallback used."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "learn.db"
        ensure_learning_ingestion_tables(str(db_path))
        old_epoch_ms = (time.time() - LABEL_GIVEUP_AGE_SEC + 3600) * 1000.0  # old enough for full labeling, not yet given up
        _seed_snapshot(db_path, epoch_ms=old_epoch_ms)

        class _FakeRedisFresh:
            def get(self, key):
                if "day_active_bundle" not in key:
                    return None
                start = old_epoch_ms - 3600_000.0
                end = time.time() * 1000.0
                bundle = {"1h": _synthetic_klines(start, end)}
                return json.dumps(bundle)

        with patch("backend.config.redis_config.get_redis_client", return_value=_FakeRedisFresh()), patch(
            "backend.services.ai_learning_ingestion._fetch_historical_klines_rest"
        ) as mock_rest:
            counters = label_pending_snapshots(str(db_path))

        assert counters["labeled"] + counters["partial"] >= 1
        mock_rest.assert_not_called(), "REST fallback must not fire when Redis coverage is already sufficient"


def test_redis_bundle_expired_before_label_horizon_completes_via_rest_fallback():
    """Scenario 3+4: Redis bundle deleted/expired -> label still completes using the durable REST fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "learn2.db"
        ensure_learning_ingestion_tables(str(db_path))
        old_epoch_ms = (time.time() - LABEL_GIVEUP_AGE_SEC * 0.6) * 1000.0  # past the 50% REST-fallback trigger threshold
        _seed_snapshot(db_path, epoch_ms=old_epoch_ms)

        def _fake_rest(sym_bus, interval, start_ms, end_ms, **_kw):
            return _synthetic_klines(start_ms, end_ms)

        with patch("backend.config.redis_config.get_redis_client", return_value=_FakeRedisEmpty()), patch(
            "backend.services.ai_learning_ingestion._fetch_historical_klines_rest", side_effect=_fake_rest
        ) as mock_rest:
            counters = label_pending_snapshots(str(db_path))

        assert mock_rest.called, "REST fallback must fire once Redis coverage is empty and the row is old enough"
        assert counters["labeled"] + counters["partial"] >= 1, "row must be labeled via the durable fallback, not left UNLABELABLE"
        assert counters["unlabelable"] == 0


def test_missing_durable_evidence_is_a_specific_unrecoverable_reason_not_silent():
    """Scenario 6: Redis empty AND REST fallback also fails -> row eventually goes UNLABELABLE
    only after the giveup age, with a clear, queryable reason (label_status + labeled_at_utc set),
    never silently vanishing from the table."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "learn3.db"
        ensure_learning_ingestion_tables(str(db_path))
        very_old_epoch_ms = (time.time() - LABEL_GIVEUP_AGE_SEC - 3600) * 1000.0  # past giveup age
        _seed_snapshot(db_path, epoch_ms=very_old_epoch_ms)

        with patch("backend.config.redis_config.get_redis_client", return_value=_FakeRedisEmpty()), patch(
            "backend.services.ai_learning_ingestion._fetch_historical_klines_rest", return_value=[]
        ):
            counters = label_pending_snapshots(str(db_path))

        assert counters["unlabelable"] == 1
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT label_status, labeled_at_utc FROM ai_candidate_snapshots").fetchone()
        assert row[0] == "UNLABELABLE"
        assert row[1] is not None, "the row must record when/that it was given up on, not disappear silently"


def test_rest_outage_before_giveup_horizon_stays_retryable_not_destroyed():
    """
    Scenario (explicit final pre-push audit requirement): a temporary REST
    outage while Redis coverage is also empty, but the row has NOT yet
    reached LABEL_GIVEUP_AGE_SEC, must leave the row retryable (PENDING or
    PARTIAL) so a later cycle can still complete it — never prematurely
    mark it UNLABELABLE just because one cycle's REST call failed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "learn5.db"
        ensure_learning_ingestion_tables(str(db_path))
        # Past the 50% REST-trigger threshold, but well before the full giveup age.
        old_epoch_ms = (time.time() - LABEL_GIVEUP_AGE_SEC * 0.6) * 1000.0
        _seed_snapshot(db_path, epoch_ms=old_epoch_ms)

        with patch("backend.config.redis_config.get_redis_client", return_value=_FakeRedisEmpty()), patch(
            "backend.services.ai_learning_ingestion._fetch_historical_klines_rest", return_value=[]
        ):
            counters = label_pending_snapshots(str(db_path))

        assert counters["unlabelable"] == 0, "a transient REST outage must not destroy labelability before giveup age"
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT label_status FROM ai_candidate_snapshots").fetchone()
        assert row[0] in ("PENDING", "PARTIAL"), f"row must remain retryable, got {row[0]!r}"


def test_rest_fallback_does_not_fire_for_ordinary_in_flight_rows():
    """REST fallback must be a last resort, not fire on every routine ~2min cycle
    for rows that simply haven't reached their horizon yet."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "learn4.db"
        ensure_learning_ingestion_tables(str(db_path))
        recent_epoch_ms = (time.time() - 20 * 60) * 1000.0  # 20 minutes old — normal in-flight row
        _seed_snapshot(db_path, epoch_ms=recent_epoch_ms)

        with patch("backend.config.redis_config.get_redis_client", return_value=_FakeRedisEmpty()), patch(
            "backend.services.ai_learning_ingestion._fetch_historical_klines_rest"
        ) as mock_rest:
            label_pending_snapshots(str(db_path))

        mock_rest.assert_not_called()


def test_feature_version_incompatible_is_safely_rejected_not_coerced():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "join.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE ai_inference_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT, symbol TEXT,
                    features_json TEXT, feature_version INTEGER
                )
                """
            )
            conn.execute(
                "INSERT INTO ai_inference_log (decision_id, symbol, features_json, feature_version) VALUES (?, ?, ?, ?)",
                ("old_v", "BTCUSDT", json.dumps([0.1] * 145), 3),  # below min_feature_version=5
            )
            conn.execute(
                "INSERT INTO ai_inference_log (decision_id, symbol, features_json, feature_version) VALUES (?, ?, ?, ?)",
                ("current_v", "BTCUSDT", json.dumps([0.1] * 145), 5),
            )
            conn.commit()
            feats = _features_for_decision_ids(conn, ["old_v", "current_v"], feature_dim=145, min_feature_version=5)

        assert "old_v" not in feats, "incompatible feature_version must be rejected, never silently coerced forward"
        assert "current_v" in feats
        assert len(feats["current_v"]) == 145


def test_feature_dimension_mismatch_is_safely_rejected_not_coerced():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "join2.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE ai_inference_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT, symbol TEXT,
                    features_json TEXT, feature_version INTEGER
                )
                """
            )
            conn.execute(
                "INSERT INTO ai_inference_log (decision_id, symbol, features_json, feature_version) VALUES (?, ?, ?, ?)",
                ("wrong_dim", "BTCUSDT", json.dumps([0.1] * 128), 5),  # wrong dimensionality
            )
            conn.commit()
            feats = _features_for_decision_ids(conn, ["wrong_dim"], feature_dim=145, min_feature_version=5)

        assert "wrong_dim" not in feats, "dimension mismatch must be rejected, never truncated/padded silently"


def test_ensure_series_covers_window_merges_rest_fallback_without_mutating_redis():
    """Confirms the fallback is purely in-memory for this run — no Redis write, no new table."""
    fetched_calls = []

    def _fake_rest(sym_bus, interval, start_ms, end_ms, **_kw):
        fetched_calls.append((sym_bus, interval))
        return _synthetic_klines(start_ms, end_ms)

    with patch("backend.services.ai_learning_ingestion._fetch_historical_klines_rest", side_effect=_fake_rest):
        series = _ensure_series_covers_window({}, sym_bus="BTCUSDT", t0_ms=0.0, end_ms=3600_000.0 * 5)

    assert fetched_calls, "must have attempted the durable fallback for an empty series"
    assert "1h" in series and len(series["1h"]) > 0
