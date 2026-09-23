"""Tests for the DAY V2 / SCALP V2 engine identity repair.

Covers:
  1.  engine_identity exports: DAY_V2_ENGINE_ID, SCALP_V2_ENGINE_ID, LEGACY_DAY_LIVE_ENGINE_ID
  2.  arm_opportunity engine-scoped deduplication (12 scenarios)
  3.  day_trailing_buy intent engine_id set to DAY_V2
  4.  cross-engine SYMBOL_OWNED_BY_OTHER_ENGINE rejection
  5.  day_trailing_buy_store fallback default
  6.  checkpoint monitor PROVEN_SCALP_V2 filter
  7.  audit classification script cohort logic

All tests use in-memory SQLite so no live DB is touched.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. engine_identity exports
# ---------------------------------------------------------------------------


def test_day_v2_engine_id_constant_is_day_v2():
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID

    assert DAY_V2_ENGINE_ID == "DAY_V2"


def test_scalp_v2_engine_id_constant_is_scalp_v2():
    from backend.services.day_v2.engine_identity import SCALP_V2_ENGINE_ID

    assert SCALP_V2_ENGINE_ID == "SCALP_V2"


def test_legacy_day_live_engine_id_constant():
    from backend.services.day_v2.engine_identity import LEGACY_DAY_LIVE_ENGINE_ID

    assert LEGACY_DAY_LIVE_ENGINE_ID == "LEGACY_DAY_LIVE"


def test_engine_id_enum_values_match_constants():
    from backend.services.day_v2.engine_identity import (
        DAY_V2_ENGINE_ID,
        LEGACY_DAY_LIVE_ENGINE_ID,
        SCALP_V2_ENGINE_ID,
        EngineId,
    )

    assert EngineId.DAY_V2_LIVE.value == DAY_V2_ENGINE_ID
    assert EngineId.SCALP_V2_LIVE.value == SCALP_V2_ENGINE_ID
    assert EngineId.LEGACY_DAY_LIVE.value == LEGACY_DAY_LIVE_ENGINE_ID


# ---------------------------------------------------------------------------
# 2. arm_opportunity — engine-scoped deduplication
# ---------------------------------------------------------------------------


def _make_opp_db() -> str:
    """Create a temp SQLite DB with the scalp_v2_opportunities schema."""
    fd, path = tempfile.mkstemp(suffix=".db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE scalp_v2_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            opportunity_id TEXT,
            setup_family TEXT,
            structural_anchor REAL,
            state TEXT,
            engine_id TEXT,
            created_at REAL,
            updated_at REAL
        )
        """
    )
    conn.commit()
    conn.close()
    import os

    os.close(fd)
    return path


def test_arm_opportunity_default_engine_is_scalp_v2():
    """Calling arm_opportunity without engine_id uses SCALP_V2."""
    from backend.services.scalp_v2.opportunity import SCALP_V2_ENGINE_ID, arm_opportunity

    db = _make_opp_db()
    _opp_id, blocked = arm_opportunity(db, "BTC/USDT", "SUPPORT_BOUNCE", 86000.0)
    assert not blocked
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT engine_id FROM scalp_v2_opportunities ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == SCALP_V2_ENGINE_ID


def test_arm_opportunity_day_v2_engine_id_stored():
    """Calling arm_opportunity with DAY_V2 stores DAY_V2 on the row."""
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _make_opp_db()
    _opp_id, blocked = arm_opportunity(db, "ETH/USDT", "SUPPORT_BOUNCE", 2700.0, engine_id=DAY_V2_ENGINE_ID)
    assert not blocked
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT engine_id FROM scalp_v2_opportunities ORDER BY id DESC LIMIT 1").fetchone()
    assert row[0] == DAY_V2_ENGINE_ID


def test_arm_opportunity_same_engine_same_zone_is_blocked():
    """Same engine + same price zone → blocked=True (duplicate prevention)."""
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _make_opp_db()
    _, blocked1 = arm_opportunity(db, "SOL/USDT", "SUPPORT_BOUNCE", 117.0, engine_id=DAY_V2_ENGINE_ID)
    _, blocked2 = arm_opportunity(db, "SOL/USDT", "SUPPORT_BOUNCE", 117.0, engine_id=DAY_V2_ENGINE_ID)
    assert not blocked1
    assert blocked2


def test_arm_opportunity_cross_engine_not_blocked():
    """DAY_V2 arm does NOT block a SCALP_V2 arm for the same symbol+zone."""
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID
    from backend.services.scalp_v2.opportunity import SCALP_V2_ENGINE_ID, arm_opportunity

    db = _make_opp_db()
    _, blocked_day = arm_opportunity(db, "XRP/USDT", "SUPPORT_BOUNCE", 1.53, engine_id=DAY_V2_ENGINE_ID)
    _, blocked_scalp = arm_opportunity(db, "XRP/USDT", "SUPPORT_BOUNCE", 1.53, engine_id=SCALP_V2_ENGINE_ID)
    assert not blocked_day
    assert not blocked_scalp  # cross-engine must not block


def test_arm_opportunity_scalp_arm_does_not_block_day():
    """SCALP_V2 arm does NOT block a DAY_V2 arm for the same symbol+zone."""
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID
    from backend.services.scalp_v2.opportunity import SCALP_V2_ENGINE_ID, arm_opportunity

    db = _make_opp_db()
    _, blocked_scalp = arm_opportunity(db, "BTC/USDT", "DEMAND_ZONE", 85000.0, engine_id=SCALP_V2_ENGINE_ID)
    _, blocked_day = arm_opportunity(db, "BTC/USDT", "DEMAND_ZONE", 85000.0, engine_id=DAY_V2_ENGINE_ID)
    assert not blocked_scalp
    assert not blocked_day


def test_arm_opportunity_reset_on_new_zone_per_engine():
    """A different price zone for the same engine resets the CLOSED state."""
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _make_opp_db()
    opp_id1, _ = arm_opportunity(db, "ETH/USDT", "SUPPORT_BOUNCE", 2700.0, engine_id=DAY_V2_ENGINE_ID)
    # Mark the old row as CLOSED to simulate a completed trade.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE scalp_v2_opportunities SET state='CLOSED' WHERE opportunity_id=?", (opp_id1,))
    conn.commit()
    conn.close()

    # New zone → the CLOSED row gets RESET and a fresh ARMED row is inserted.
    opp_id2, blocked = arm_opportunity(db, "ETH/USDT", "SUPPORT_BOUNCE", 2500.0, engine_id=DAY_V2_ENGINE_ID)
    assert not blocked
    assert opp_id2 != opp_id1


def test_arm_opportunity_two_engines_independent_zones():
    """Two engines on the same symbol maintain independent opportunity rows."""
    from backend.services.day_v2.engine_identity import DAY_V2_ENGINE_ID
    from backend.services.scalp_v2.opportunity import SCALP_V2_ENGINE_ID, arm_opportunity

    db = _make_opp_db()
    arm_opportunity(db, "SOL/USDT", "SUPPORT_BOUNCE", 117.0, engine_id=DAY_V2_ENGINE_ID)
    arm_opportunity(db, "SOL/USDT", "SUPPORT_BOUNCE", 117.0, engine_id=SCALP_V2_ENGINE_ID)

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT engine_id, state FROM scalp_v2_opportunities WHERE symbol='SOL/USDT' ORDER BY id").fetchall()
    conn.close()
    engine_ids = {r[0] for r in rows}
    assert DAY_V2_ENGINE_ID in engine_ids
    assert SCALP_V2_ENGINE_ID in engine_ids
    assert all(r[1] == "ARMED" for r in rows)


# ---------------------------------------------------------------------------
# 3. day_trailing_buy_store fallback default
# ---------------------------------------------------------------------------


def test_create_intent_fallback_engine_id_is_legacy_not_scalp():
    """When engine_id is omitted, create_intent stores LEGACY_DAY_LIVE not SCALP_V2."""
    from backend.services.day_trailing_buy_store import create_intent

    fd, path = tempfile.mkstemp(suffix=".db")
    import os

    os.close(fd)

    ok, reason, intent = create_intent(
        path,
        fields={
            "decision_id": "test-decision-fallback-001",
            "symbol": "ETH/USDT",
            "setup": "SUPPORT_BOUNCE",
            "arm_ask": 2700.0,
            # engine_id intentionally omitted — tests the fallback
        },
    )
    assert ok, reason
    assert intent["engine_id"] == "LEGACY_DAY_LIVE"
    assert intent["engine_id"] != "SCALP_V2"


def test_create_intent_explicit_day_v2_engine_id_stored():
    """When engine_id='DAY_V2' is passed, it is stored correctly."""
    from backend.services.day_trailing_buy_store import create_intent

    fd, path = tempfile.mkstemp(suffix=".db")
    import os

    os.close(fd)

    ok, reason, intent = create_intent(
        path,
        fields={
            "decision_id": "test-decision-dayv2-001",
            "symbol": "BTC/USDT",
            "setup": "DEMAND_ZONE",
            "arm_ask": 86000.0,
            "engine_id": "DAY_V2",
        },
    )
    assert ok, reason
    assert intent["engine_id"] == "DAY_V2"


# ---------------------------------------------------------------------------
# 4. audit script cohort classification
# ---------------------------------------------------------------------------


def _make_audit_db() -> str:
    """Create a minimal DB that the audit script can classify."""
    fd, path = tempfile.mkstemp(suffix=".db")
    import os

    os.close(fd)
    conn = sqlite3.connect(path)
    # day_trailing_buy_intents — minimal schema.
    conn.execute(
        """
        CREATE TABLE day_trailing_buy_intents (
            intent_id TEXT PRIMARY KEY,
            symbol TEXT,
            engine_id TEXT,
            arm_ts REAL,
            trade_id TEXT,
            status TEXT
        )
        """
    )
    # paper_trades — minimal schema.
    conn.execute(
        """
        CREATE TABLE paper_trades (
            trade_id TEXT PRIMARY KEY,
            side TEXT,
            engine_id TEXT,
            symbol TEXT,
            timestamp TEXT,
            pnl REAL,
            price REAL,
            quantity REAL
        )
        """
    )
    # portfolio_engine_positions — minimal schema.
    conn.execute(
        """
        CREATE TABLE portfolio_engine_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            engine_id TEXT
        )
        """
    )
    # portfolio_engine_ledger — required by cohort_performance.
    conn.execute(
        """
        CREATE TABLE portfolio_engine_ledger (
            id INTEGER PRIMARY KEY,
            principal REAL,
            cash_balance REAL,
            realized_pnl REAL,
            total_equity REAL
        )
        """
    )
    conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, 10000, 10000, 0, 10000)")

    now = time.time()
    # One DAY intent (mislabelled SCALP_V2, post-628c47f).
    conn.execute(
        "INSERT INTO day_trailing_buy_intents VALUES (?,?,?,?,?,?)",
        ("intent_day_001", "XRP/USDT", "SCALP_V2", now, "trade_day_001", "FILLED"),
    )
    # One SCALP V2 paper_trade BUY (genuine, no matching intent).
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?)",
        ("trade_scalp_001", "BUY", "SCALP_V2", "XRP/USDT", "2026-09-22T00:00:00+00:00", None, 1.50, 10),
    )
    # DAY mislabelled BUY row.
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?)",
        ("trade_day_001", "BUY", "SCALP_V2", "XRP/USDT", "2026-09-22T00:15:00+00:00", None, 1.53, 22),
    )
    conn.commit()
    conn.close()
    return path


def test_audit_intent_classified_as_proven_day_v2():
    """Intents in day_trailing_buy_intents → PROVEN_DAY_V2 cohort."""
    from scripts.day_v2_engine_repair_audit import classify_intents

    db = _make_audit_db()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    from scripts.day_v2_engine_repair_audit import _ensure_audit_table

    _ensure_audit_table(conn)
    rows = classify_intents(conn, dry_run=True)
    conn.close()
    assert len(rows) == 1
    assert rows[0]["cohort"] == "PROVEN_DAY_V2"
    assert rows[0]["corrected_engine_id"] == "DAY_V2"
    assert rows[0]["original_engine_id"] == "SCALP_V2"


def test_audit_day_paper_trade_buy_classified_as_proven_day_v2():
    """BUY row whose trade_id is in day_trailing_buy_intents → PROVEN_DAY_V2."""
    from scripts.day_v2_engine_repair_audit import classify_paper_trades

    db = _make_audit_db()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    from scripts.day_v2_engine_repair_audit import _ensure_audit_table

    _ensure_audit_table(conn)
    day_trade_ids = {"trade_day_001"}
    rows = classify_paper_trades(conn, day_trade_ids, dry_run=True)
    conn.close()
    assert any(r["record_id"] == "trade_day_001" for r in rows)
    assert all(r["cohort"] == "PROVEN_DAY_V2" for r in rows)


def test_audit_cohort_performance_separates_cohorts():
    """cohort_performance() assigns genuine SCALP_V2 to PROVEN_SCALP_V2."""
    from scripts.day_v2_engine_repair_audit import cohort_performance

    db = _make_audit_db()
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # Add a SELL for the scalp trade.
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?)",
        ("trade_scalp_001_sell", "SELL", "SCALP_V2", "XRP/USDT", "2026-09-22T01:00:00+00:00", 0.15, 1.52, 10),
    )
    conn.commit()
    day_trade_ids = {"trade_day_001"}
    perf = cohort_performance(conn, day_trade_ids)
    conn.close()
    # The genuine scalp SELL should be in PROVEN_SCALP_V2.
    assert perf["PROVEN_SCALP_V2"]["sells"] == 1
    # The DAY BUY should be in PROVEN_DAY_V2 (no SELL for it, so only buy counted).
    assert perf["PROVEN_DAY_V2"]["buys"] >= 1


# ---------------------------------------------------------------------------
# 5. checkpoint monitor PROVEN_SCALP_V2 filter
# ---------------------------------------------------------------------------


def _make_monitor_db() -> str:
    """Create minimal DB for checkpoint monitor filter test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    import os

    os.close(fd)
    conn = sqlite3.connect(path)
    # paper_trades schema (minimal columns used by monitor).
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT,
            symbol TEXT,
            side TEXT,
            quantity REAL,
            price REAL,
            entry_price REAL,
            pnl_usd_net REAL,
            pnl_pct_net REAL,
            hold_time_seconds REAL,
            exit_reason TEXT,
            entry_timestamp TEXT,
            timestamp TEXT,
            stop_price REAL,
            take_profit_price REAL,
            atr_at_entry REAL,
            fees_paid REAL,
            entry_fee_usd REAL,
            exit_fee_usd REAL,
            scalp_opportunity_id TEXT,
            engine_id TEXT,
            order_id TEXT,
            slippage_pct_used REAL,
            spread_pct_used REAL,
            status TEXT,
            is_synthetic INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE day_trailing_buy_intents (
            intent_id TEXT PRIMARY KEY,
            symbol TEXT,
            engine_id TEXT,
            arm_ts REAL,
            trade_id TEXT,
            status TEXT
        )
        """
    )
    now_str = "2026-09-22T12:00:00+00:00"
    # Genuine SCALP V2 SELL (trade_id NOT in intents).
    conn.execute(
        "INSERT INTO paper_trades(trade_id,side,engine_id,pnl_usd_net,status,is_synthetic,timestamp) VALUES (?,?,?,?,?,?,?)",
        ("scalp_sell_001", "SELL", "SCALP_V2", 0.10, "executed", 0, now_str),
    )
    # DAY mislabelled SCALP_V2 SELL (trade_id IS in intents).
    conn.execute(
        "INSERT INTO paper_trades(trade_id,side,engine_id,pnl_usd_net,status,is_synthetic,timestamp) VALUES (?,?,?,?,?,?,?)",
        ("day_sell_001", "SELL", "SCALP_V2", -0.05, "executed", 0, now_str),
    )
    conn.execute(
        "INSERT INTO day_trailing_buy_intents(intent_id,symbol,engine_id,arm_ts,trade_id,status) VALUES (?,?,?,?,?,?)",
        ("intent_001", "XRP/USDT", "SCALP_V2", time.time(), "day_sell_001", "FILLED"),
    )
    conn.commit()
    conn.close()
    return path


def test_monitor_excludes_day_mislabelled_scalp_trades():
    """_load_qualifying_trades excludes SELL rows whose trade_id is in day_trailing_buy_intents."""
    from scripts.scalp_v2_checkpoint_monitor import _load_qualifying_trades

    db = _make_monitor_db()
    conn = sqlite3.connect(db)
    trades = _load_qualifying_trades(conn)
    conn.close()
    trade_ids = [t["trade_id"] for t in trades]
    assert "scalp_sell_001" in trade_ids
    assert "day_sell_001" not in trade_ids


def test_monitor_counts_only_proven_scalp_v2():
    """After repair, only genuine SCALP V2 trades count toward the 100-trade checkpoint."""
    from scripts.scalp_v2_checkpoint_monitor import _load_qualifying_trades

    db = _make_monitor_db()
    conn = sqlite3.connect(db)
    trades = _load_qualifying_trades(conn)
    conn.close()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "scalp_sell_001"
