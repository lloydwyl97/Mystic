"""
Regression: heartbeat telemetry schema unification (repair-all continuation,
Phase 3).

record_position_heartbeat() is the sole producer of ai_position_heartbeats
and always stamps HEARTBEAT_CALC_VERSION on every insert — there is exactly
one active writer schema. Historical rows at older versions must be readable
without breaking anything, but must never be counted as *current* runtime
health evidence.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

from backend.services.ai_learning_ingestion import (
    HEARTBEAT_CALC_VERSION,
    _heartbeat_schema_health,
    ensure_learning_ingestion_tables,
    record_position_heartbeat,
)


def test_all_new_heartbeat_writes_use_one_current_version():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hb.db"
        ensure_learning_ingestion_tables(str(db_path))
        for i in range(3):
            record_position_heartbeat(
                symbol="BTC/USDT",
                trade_id=f"t{i}",
                entry_price=100.0,
                mark=100.5,
                entry_time_epoch=time.time() - 60,
                db_path=str(db_path),
            )
        with sqlite3.connect(str(db_path)) as conn:
            versions = {r[0] for r in conn.execute("SELECT DISTINCT heartbeat_calc_version FROM ai_position_heartbeats")}
        assert versions == {HEARTBEAT_CALC_VERSION}


def test_old_v1_rows_do_not_break_reads():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hb2.db"
        ensure_learning_ingestion_tables(str(db_path))
        # Simulate a legacy v1 row written before the v2 columns existed
        # (defaults apply for anything not explicitly set).
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO ai_position_heartbeats (ts_utc, epoch_ms, symbol, trade_id, entry_price, mark, unrealized_pct, heartbeat_calc_version)
                VALUES (datetime('now', '-120 days'), ?, 'ETH/USDT', 'legacy1', 1800.0, 1810.0, 0.0055, 1)
                """,
                (int((time.time() - 120 * 86400) * 1000),),
            )
            conn.commit()
        health = _heartbeat_schema_health(str(db_path))
        assert health["version_counts"]["1"] == 1
        assert health["current_version_row_count"] == 0
        # Reading did not raise, and the row is still visible in the table.
        with sqlite3.connect(str(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM ai_position_heartbeats").fetchone()[0] == 1


def test_active_health_uses_only_current_version_freshness():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hb3.db"
        ensure_learning_ingestion_tables(str(db_path))
        record_position_heartbeat(
            symbol="SOL/USDT",
            trade_id="fresh1",
            entry_price=150.0,
            mark=151.0,
            entry_time_epoch=time.time() - 300,
            db_path=str(db_path),
        )
        health = _heartbeat_schema_health(str(db_path))
        assert health["active_health"] == "FRESH"
        assert health["latest_current_version_age_sec"] is not None
        assert health["latest_current_version_age_sec"] < 60


def test_old_v1_history_cannot_masquerade_as_current_health():
    """Only v1 rows exist (even a recent-looking one) — active health must
    report NO_CURRENT_VERSION_DATA, never silently treat old-schema rows as
    fresh current-version evidence."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hb4.db"
        ensure_learning_ingestion_tables(str(db_path))
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO ai_position_heartbeats (ts_utc, epoch_ms, symbol, trade_id, entry_price, mark, unrealized_pct, heartbeat_calc_version)
                VALUES (datetime('now'), ?, 'XRP/USDT', 'recent_but_v1', 1.0, 1.01, 0.01, 1)
                """,
                (int(time.time() * 1000),),
            )
            conn.commit()
        health = _heartbeat_schema_health(str(db_path))
        assert health["current_version_row_count"] == 0
        assert health["historical_row_count"] == 1
        assert health["active_health"] == "NO_CURRENT_VERSION_DATA"
        assert health["latest_current_version_heartbeat_utc"] is None


def test_stale_current_version_heartbeat_is_reported_stale_not_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hb5.db"
        ensure_learning_ingestion_tables(str(db_path))
        with sqlite3.connect(str(db_path)) as conn:
            stale_epoch_ms = int((time.time() - 3600) * 1000)  # 1h old, current schema
            conn.execute(
                """
                INSERT INTO ai_position_heartbeats (ts_utc, epoch_ms, symbol, trade_id, entry_price, mark, unrealized_pct, heartbeat_calc_version)
                VALUES (datetime('now', '-1 hours'), ?, 'BTC/USDT', 'stale_v2', 100.0, 101.0, 0.01, ?)
                """,
                (stale_epoch_ms, HEARTBEAT_CALC_VERSION),
            )
            conn.commit()
        health = _heartbeat_schema_health(str(db_path))
        assert health["active_health"] == "STALE"


def test_no_data_yet_is_distinct_from_stale_or_no_current_version():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hb6.db"
        ensure_learning_ingestion_tables(str(db_path))
        health = _heartbeat_schema_health(str(db_path))
        assert health["active_health"] == "NO_DATA_YET"
        assert health["current_version_row_count"] == 0
        assert health["historical_row_count"] == 0
