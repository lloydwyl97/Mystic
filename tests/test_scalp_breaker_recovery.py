"""Autonomous SCALP consecutive-loss breaker recovery — no epoch edit required."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.services.binance_scalp.schema import init_scalp_schema


def _probe(
    db_path: Path,
    *,
    now: datetime,
    max_consec: int = 5,
    recovery_sec: int = 3600,
    epoch: str = "",
    daily_limit_pct: float = 0.05,
):
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = SimpleNamespace(
        circuit_breaker_epoch=epoch,
        max_consecutive_losses=max_consec,
        daily_loss_limit_pct=daily_limit_pct,
        breaker_recovery_sec=recovery_sec,
        database_path=str(db_path),
    )
    engine._conn = lambda: sqlite3.connect(str(db_path), timeout=10.0)
    engine._ledger = lambda _conn: {"principal": 1000.0}
    engine._utcnow_override = now
    engine._last_breaker_reason = ""
    engine._last_breaker_recovery_until = ""
    engine._last_breaker_eval_after = ""
    open_ = BinanceScalpPaperEngine._check_scalp_circuit_breaker(engine)
    return open_, engine


def _seed(db_path: Path, rows: list[tuple[float, str]]) -> None:
    init_scalp_schema(db_path, principal=1000.0)
    with sqlite3.connect(db_path) as conn:
        for idx, (pnl, created_at) in enumerate(rows):
            conn.execute(
                "INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional, pnl_usd, created_at) VALUES (?,'XRPUSDT','SELL',1,1,1,?,?)",
                (f"t{idx}", pnl, created_at),
            )
        conn.commit()


def test_trip_n_consecutive_losses_opens_breaker(tmp_path: Path):
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 22, 22, 10, tzinfo=timezone.utc)
    _seed(db, [(-0.05, f"2026-08-22 22:0{i}:00") for i in range(5)])
    open_, engine = _probe(db, now=now, max_consec=5, recovery_sec=3600)
    assert open_ is True
    assert engine._last_breaker_reason == "CONSECUTIVE_LOSSES_COOLDOWN"
    assert engine._last_breaker_recovery_until


def test_flat_deadlock_does_not_require_impossible_win(tmp_path: Path):
    """OPEN + zero positions must recover after the cooldown without a new win."""
    db = tmp_path / "scalp.db"
    trip = datetime(2026, 8, 22, 22, 2, tzinfo=timezone.utc)
    _seed(db, [(-0.05, (trip - timedelta(minutes=4 - i)).strftime("%Y-%m-%d %H:%M:%S")) for i in range(5)])
    during = trip + timedelta(minutes=10)
    open_, _ = _probe(db, now=during, max_consec=5, recovery_sec=3600)
    assert open_ is True

    after = trip + timedelta(hours=2)
    open_, engine = _probe(db, now=after, max_consec=5, recovery_sec=3600)
    assert open_ is False
    assert engine._last_breaker_reason != "CONSECUTIVE_LOSSES_COOLDOWN"
    with sqlite3.connect(db) as conn:
        sells = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades WHERE upper(side)='SELL'").fetchone()[0]
        eval_after = conn.execute("SELECT consec_breaker_eval_after FROM scalp_meta WHERE id=1").fetchone()[0]
    assert sells == 5
    assert eval_after


def test_recovery_closes_breaker_and_keeps_history(tmp_path: Path):
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 22, 22, 5, tzinfo=timezone.utc)
    _seed(db, [(-0.05, f"2026-08-22 22:0{i}:00") for i in range(5)])
    open_, _ = _probe(db, now=now, max_consec=5, recovery_sec=3600)
    assert open_ is True

    recovered = now + timedelta(hours=2)
    open_, engine = _probe(db, now=recovered, max_consec=5, recovery_sec=3600)
    assert open_ is False
    assert engine._last_breaker_eval_after
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0] == 5
        pnl = conn.execute("SELECT SUM(pnl_usd) FROM scalp_paper_trades").fetchone()[0]
    assert abs(float(pnl) + 0.25) < 1e-9


def test_retrip_after_fresh_loss_sequence(tmp_path: Path):
    db = tmp_path / "scalp.db"
    first = datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc)
    _seed(db, [(-0.05, f"2026-08-22 09:5{i}:00") for i in range(5)])
    open_, _ = _probe(db, now=first, max_consec=5, recovery_sec=3600)
    assert open_ is True
    open_, _ = _probe(db, now=first + timedelta(hours=2), max_consec=5, recovery_sec=3600)
    assert open_ is False

    with sqlite3.connect(db) as conn:
        for idx in range(5):
            conn.execute(
                "INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional, pnl_usd, created_at) VALUES (?,'XRPUSDT','SELL',1,1,1,-0.04,?)",
                (f"new{idx}", f"2026-08-22 12:1{idx}:00"),
            )
        conn.commit()
    retrip = datetime(2026, 8, 22, 12, 20, tzinfo=timezone.utc)
    open_, engine = _probe(db, now=retrip, max_consec=5, recovery_sec=3600)
    assert open_ is True
    assert engine._last_breaker_reason == "CONSECUTIVE_LOSSES_COOLDOWN"


def test_restart_during_cooldown_keeps_protection(tmp_path: Path):
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 22, 22, 5, tzinfo=timezone.utc)
    _seed(db, [(-0.05, f"2026-08-22 22:0{i}:00") for i in range(5)])
    open_, first = _probe(db, now=now, max_consec=5, recovery_sec=3600)
    assert open_ is True
    until = first._last_breaker_recovery_until
    assert until

    later = now + timedelta(minutes=20)
    open_, second = _probe(db, now=later, max_consec=5, recovery_sec=3600)
    assert open_ is True
    assert second._last_breaker_reason == "CONSECUTIVE_LOSSES_COOLDOWN"
    assert second._last_breaker_recovery_until == until


def test_restart_after_recovery_does_not_resurrect_trip(tmp_path: Path):
    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 22, 22, 5, tzinfo=timezone.utc)
    _seed(db, [(-0.05, f"2026-08-22 22:0{i}:00") for i in range(5)])
    assert _probe(db, now=now, max_consec=5, recovery_sec=3600)[0] is True
    recovered = now + timedelta(hours=2)
    assert _probe(db, now=recovered, max_consec=5, recovery_sec=3600)[0] is False
    later = recovered + timedelta(minutes=5)
    open_, engine = _probe(db, now=later, max_consec=5, recovery_sec=3600)
    assert open_ is False
    assert engine._last_breaker_reason != "CONSECUTIVE_LOSSES_COOLDOWN"


def test_recovery_on_tick_connection_while_write_lock_held(tmp_path: Path):
    """Expire must use the tick's BEGIN IMMEDIATE conn; a second writer fail-closes."""
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 24, 13, 40, 23, tzinfo=timezone.utc)
    _seed(db, [(-0.05, f"2026-08-24 13:3{i}:00") for i in range(5)])
    assert _probe(db, now=now, max_consec=5, recovery_sec=14400)[0] is True
    after = now + timedelta(hours=4, seconds=30)

    locker = sqlite3.connect(str(db), timeout=0.1)
    locker.execute("BEGIN IMMEDIATE")
    try:
        locked_engine = object.__new__(BinanceScalpPaperEngine)
        locked_engine.config = SimpleNamespace(
            circuit_breaker_epoch="",
            max_consecutive_losses=5,
            daily_loss_limit_pct=0.05,
            breaker_recovery_sec=14400,
            database_path=str(db),
        )
        locked_engine._ledger = lambda _conn: {"principal": 1000.0}
        locked_engine._utcnow_override = after
        locked_engine._last_breaker_reason = ""
        locked_engine._last_breaker_recovery_until = ""
        locked_engine._last_breaker_eval_after = ""

        def _second_writer():
            c = sqlite3.connect(str(db), timeout=0.05)
            c.execute("BEGIN IMMEDIATE")
            return c

        locked_engine._conn = _second_writer
        assert BinanceScalpPaperEngine._check_scalp_circuit_breaker(locked_engine) is True
        assert locked_engine._last_breaker_reason == "SCALP_BREAKER_STATE_UNAVAILABLE"

        open_ = BinanceScalpPaperEngine._check_scalp_circuit_breaker(locked_engine, locker)
        assert open_ is False
        locker.commit()
    finally:
        locker.close()
    with sqlite3.connect(db) as conn:
        tripped, until = conn.execute("SELECT consec_breaker_tripped_at, consec_breaker_recovery_until FROM scalp_meta WHERE id=1").fetchone()
    assert not tripped
    assert not until


def test_daily_loss_stays_independent_of_consec_recovery(tmp_path: Path):
    db = tmp_path / "scalp.db"
    today = datetime(2026, 8, 23, 13, 0, tzinfo=timezone.utc)
    _seed(db, [(-60.0, "2026-08-23 01:00:00")])
    open_, engine = _probe(db, now=today, max_consec=5, recovery_sec=3600, daily_limit_pct=0.05)
    assert open_ is True
    assert engine._last_breaker_reason == "DAILY_LOSS_LIMIT"
    with sqlite3.connect(db) as conn:
        until = conn.execute("SELECT consec_breaker_recovery_until FROM scalp_meta WHERE id=1").fetchone()[0]
    assert not until
