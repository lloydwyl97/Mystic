"""SCALP breaker must fail closed on SQLite lock; recovery design is preserved."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tests.test_scalp_breaker_recovery import _probe, _seed


def test_local_five_losses_trip(tmp_path: Path):
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc)
    _seed(db, [(-0.07, f"2026-08-24 07:5{i}:00") for i in range(5)])
    open_, engine = _probe(db, now=now, max_consec=5, recovery_sec=14400)
    assert open_ is True
    assert engine._last_breaker_reason == "CONSECUTIVE_LOSSES_COOLDOWN"
    assert engine._last_breaker_recovery_until
    with sqlite3.connect(db) as conn:
        tripped, until = conn.execute("SELECT consec_breaker_tripped_at, consec_breaker_recovery_until FROM scalp_meta WHERE id=1").fetchone()
    assert tripped
    assert until


def test_ocean_ten_losses_trip(tmp_path: Path):
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc)
    _seed(db, [(-0.07, f"2026-08-24 06:{i:02d}:00") for i in range(10)])
    open_, engine = _probe(db, now=now, max_consec=10, recovery_sec=14400)
    assert open_ is True
    assert engine._last_breaker_reason == "CONSECUTIVE_LOSSES_COOLDOWN"
    with sqlite3.connect(db) as conn:
        tripped, until = conn.execute("SELECT consec_breaker_tripped_at, consec_breaker_recovery_until FROM scalp_meta WHERE id=1").fetchone()
    assert tripped
    assert until


def test_read_lock_fail_closed_then_recovers(tmp_path: Path, monkeypatch):
    from backend.utils import sqlite_runtime

    monkeypatch.setattr(sqlite_runtime, "_locked_retries", lambda: 2)
    monkeypatch.setattr(sqlite_runtime, "_base_backoff_sec", lambda: 0.01)
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc)
    _seed(db, [(-0.07, f"2026-08-24 07:5{i}:00") for i in range(4)])

    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = SimpleNamespace(
        circuit_breaker_epoch="",
        max_consecutive_losses=5,
        daily_loss_limit_pct=0.05,
        breaker_recovery_sec=14400,
        database_path=str(db),
    )
    engine._ledger = lambda _conn: {"principal": 1000.0}
    engine._utcnow_override = now
    engine._last_breaker_reason = ""
    engine._last_breaker_recovery_until = ""
    engine._last_breaker_eval_after = ""

    def _locked():
        raise sqlite3.OperationalError("database is locked")

    engine._conn = _locked
    open_ = BinanceScalpPaperEngine._check_scalp_circuit_breaker(engine)
    assert open_ is True
    assert engine._last_breaker_reason == "SCALP_BREAKER_STATE_UNAVAILABLE"

    engine._conn = lambda: sqlite3.connect(str(db), timeout=10.0)
    open_ = BinanceScalpPaperEngine._check_scalp_circuit_breaker(engine)
    assert open_ is False
    assert engine._last_breaker_reason != "SCALP_BREAKER_STATE_UNAVAILABLE"


def test_write_lock_fail_closed_then_persists(tmp_path: Path, monkeypatch):
    from backend.utils import sqlite_runtime

    monkeypatch.setattr(sqlite_runtime, "_locked_retries", lambda: 2)
    monkeypatch.setattr(sqlite_runtime, "_base_backoff_sec", lambda: 0.01)
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc)
    _seed(db, [(-0.07, f"2026-08-24 07:5{i}:00") for i in range(5)])

    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = SimpleNamespace(
        circuit_breaker_epoch="",
        max_consecutive_losses=5,
        daily_loss_limit_pct=0.05,
        breaker_recovery_sec=14400,
        database_path=str(db),
    )
    engine._ledger = lambda _conn: {"principal": 1000.0}
    engine._utcnow_override = now
    engine._last_breaker_reason = ""
    engine._last_breaker_recovery_until = ""
    engine._last_breaker_eval_after = ""
    engine._conn = lambda: sqlite3.connect(str(db), timeout=10.0)

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    engine._save_consec_breaker_state = _boom
    open_ = BinanceScalpPaperEngine._check_scalp_circuit_breaker(engine)
    assert open_ is True
    assert engine._last_breaker_reason == "SCALP_BREAKER_STATE_UNAVAILABLE"

    del engine._save_consec_breaker_state
    open_, engine2 = _probe(db, now=now, max_consec=5, recovery_sec=14400)
    assert open_ is True
    assert engine2._last_breaker_recovery_until


def test_four_hour_recovery_and_retrip(tmp_path: Path):
    db = tmp_path / "scalp.db"
    trip = datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc)
    _seed(db, [(-0.07, (trip - timedelta(minutes=4 - i)).strftime("%Y-%m-%d %H:%M:%S")) for i in range(5)])
    open_, first = _probe(db, now=trip, max_consec=5, recovery_sec=14400)
    assert open_ is True
    until = first._last_breaker_recovery_until
    assert until

    during = trip + timedelta(hours=2)
    open_, restarted = _probe(db, now=during, max_consec=5, recovery_sec=14400)
    assert open_ is True
    assert restarted._last_breaker_recovery_until == until

    after = trip + timedelta(hours=4, seconds=1)
    open_, _recovered = _probe(db, now=after, max_consec=5, recovery_sec=14400)
    assert open_ is False

    with sqlite3.connect(db) as conn:
        for idx in range(5):
            conn.execute(
                "INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional, pnl_usd, created_at) VALUES (?,'XRPUSDT','SELL',1,1,1,-0.06,?)",
                (f"fresh{idx}", (after + timedelta(minutes=idx + 1)).strftime("%Y-%m-%d %H:%M:%S")),
            )
        conn.commit()
    later = after + timedelta(minutes=10)
    open_, retrip = _probe(db, now=later, max_consec=5, recovery_sec=14400)
    assert open_ is True
    assert retrip._last_breaker_reason == "CONSECUTIVE_LOSSES_COOLDOWN"
