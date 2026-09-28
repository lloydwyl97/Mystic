"""Restored SCALP loss breaker applied to live SCALP_V2 fills."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend.services.scalp_v2.loss_breaker import check_scalp_loss_breaker

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _db(tmp_path, rows):
    db = str(tmp_path / "t.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, engine_id TEXT, side TEXT, pnl REAL, timestamp TEXT, counts_toward_realized INTEGER, is_synthetic INTEGER)")
    for engine, side, pnl, minutes_ago, *rest in rows:
        synthetic = rest[0] if rest else 0
        c.execute(
            "INSERT INTO paper_trades (engine_id, side, pnl, timestamp, counts_toward_realized, is_synthetic) VALUES (?,?,?,?,1,?)",
            (engine, side, pnl, (NOW - timedelta(minutes=minutes_ago)).isoformat(), synthetic),
        )
    c.commit()
    c.close()
    return db


def _cfg(**kw):
    base = {"daily_loss_limit_pct": 0.05, "max_consecutive_losses": 3, "breaker_recovery_sec": 3600, "circuit_breaker_epoch": ""}
    base.update(kw)
    return SimpleNamespace(**base)


def test_daily_loss_under_limit_keeps_trading(tmp_path):
    db = _db(tmp_path, [("SCALP_V2", "SELL", -3.0, 30), ("SCALP_V2", "SELL", 1.0, 20)])
    assert not check_scalp_loss_breaker(db, _cfg(max_consecutive_losses=0), principal=100.0, now=NOW).halt


def test_daily_loss_limit_halts_entries(tmp_path):
    db = _db(tmp_path, [("SCALP_V2", "SELL", -6.0, 30)])
    res = check_scalp_loss_breaker(db, _cfg(max_consecutive_losses=0), principal=100.0, now=NOW)
    assert res.halt and res.reason == "DAILY_LOSS_LIMIT"


def test_consecutive_losses_trip_persist_and_recover(tmp_path):
    db = _db(tmp_path, [("SCALP_V2", "SELL", 0.5, 50), ("SCALP_V2", "SELL", -0.1, 30), ("SCALP_V2", "SELL", -0.2, 20), ("SCALP_V2", "SELL", -0.3, 10)])
    res = check_scalp_loss_breaker(db, _cfg(), principal=1000.0, now=NOW)
    assert res.halt and res.reason == "CONSECUTIVE_LOSSES_COOLDOWN"
    # restart: persisted cooldown still halts
    assert check_scalp_loss_breaker(db, _cfg(), principal=1000.0, now=NOW + timedelta(minutes=5)).halt
    # after recovery the same streak does not re-trip
    later = NOW + timedelta(hours=2)
    assert not check_scalp_loss_breaker(db, _cfg(), principal=1000.0, now=later).halt
    assert not check_scalp_loss_breaker(db, _cfg(), principal=1000.0, now=later + timedelta(minutes=1)).halt


def test_day_and_synthetic_rows_do_not_count(tmp_path):
    db = _db(
        tmp_path,
        [("DAY_V2", "SELL", -1.0, 30), ("DAY_V2", "SELL", -1.0, 20), ("SCALP_V2", "SELL", -1.0, 15, 1), ("SCALP_V2", "SELL", -0.1, 10), ("SCALP_V2", "BUY", -9.0, 5)],
    )
    assert not check_scalp_loss_breaker(db, _cfg(), principal=1000.0, now=NOW).halt


def test_epoch_floor_excludes_older_losses(tmp_path):
    db = _db(tmp_path, [("SCALP_V2", "SELL", -0.1, 300), ("SCALP_V2", "SELL", -0.1, 290), ("SCALP_V2", "SELL", -0.1, 10)])
    epoch = (NOW - timedelta(minutes=60)).isoformat()
    assert not check_scalp_loss_breaker(db, _cfg(circuit_breaker_epoch=epoch, breaker_recovery_sec=100000), principal=1000.0, now=NOW).halt


def test_unreadable_state_fails_closed(tmp_path):
    db = str(tmp_path / "empty.db")
    sqlite3.connect(db).close()
    res = check_scalp_loss_breaker(db, _cfg(), principal=1000.0, now=NOW)
    assert res.halt and res.reason == "SCALP_BREAKER_STATE_UNAVAILABLE"
