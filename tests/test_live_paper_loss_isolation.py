"""LIVE loss_heavy isolation — paper history must not pause a live account."""

import sqlite3
from pathlib import Path

from backend.services.live_fill_economics import (
    COIN_PERFORMANCE_FIELD_CLASS,
    live_risk_loss_hits,
    recent_sell_pnls,
    rolling_loss_count,
    sum_realized_pnl_by_mode,
)


def _seed_trades(db: Path, rows: list[tuple[str, str, float, str]]) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                mode TEXT,
                pnl REAL,
                timestamp TEXT,
                is_synthetic INTEGER DEFAULT 0,
                exit_type TEXT DEFAULT ''
            )
            """
        )
        for _i, (symbol, mode, pnl, ts) in enumerate(rows):
            conn.execute(
                "INSERT INTO paper_trades (symbol, side, mode, pnl, timestamp) VALUES (?,?,?,?,?)",
                (symbol, "SELL", mode, pnl, ts),
            )
        conn.commit()


def test_classification_documents_risk_vs_learning():
    assert COIN_PERFORMANCE_FIELD_CLASS["stop_loss_hits_10"] == "B"
    assert COIN_PERFORMANCE_FIELD_CLASS["pause_until"] == "B"
    assert COIN_PERFORMANCE_FIELD_CLASS["sizing_multiplier"] == "A"
    assert COIN_PERFORMANCE_FIELD_CLASS["win_rate_20"] == "A"


def test_paper_losses_do_not_count_as_live_hits(tmp_path: Path):
    db = tmp_path / "t.db"
    _seed_trades(
        db,
        [("BTCUSDT", "paper", -1.0, f"2026-08-2{i}T00:00:00") for i in range(5)] + [("BTCUSDT", "live", 0.53, "2026-08-24T10:14:00")],
    )
    paper_sticky = 10
    live_pnls = recent_sell_pnls(str(db), "BTCUSDT", limit=10, mode="live")
    hits = live_risk_loss_hits(is_live_day=True, sticky_hits=paper_sticky, live_pnls=live_pnls)
    assert hits == 0
    assert hits < 3


def test_live_losses_do_count(tmp_path: Path):
    db = tmp_path / "t.db"
    rows = [("BTCUSDT", "live", -0.2, f"2026-08-24T0{i}:00:00") for i in range(3)]
    _seed_trades(db, rows)
    live_pnls = recent_sell_pnls(str(db), "BTCUSDT", limit=10, mode="live")
    hits = live_risk_loss_hits(is_live_day=True, sticky_hits=0, live_pnls=live_pnls)
    assert hits == 3


def test_rolling_last_n_wins_reduce_count():
    pnls = [-1.0, -1.0, -1.0, 0.5, -0.2]
    assert rolling_loss_count(pnls, 10) == 4
    pnls2 = [0.5, -1.0, -1.0]
    assert rolling_loss_count(pnls2, 10) == 2
    assert rolling_loss_count([0.5], 10) == 0


def test_paper_learning_history_preserved(tmp_path: Path):
    db = tmp_path / "t.db"
    _seed_trades(
        db,
        [
            ("BTCUSDT", "paper", 900.0, "2026-01-01T00:00:00"),
            ("BTCUSDT", "paper", 74.54, "2026-02-01T00:00:00"),
            ("BTCUSDT", "live", 0.53, "2026-08-24T10:14:00"),
        ],
    )
    assert abs(sum_realized_pnl_by_mode(str(db), mode="paper") - 974.54) < 1e-9
    assert abs(sum_realized_pnl_by_mode(str(db), mode="live") - 0.53) < 1e-9
    paper_pnls = recent_sell_pnls(str(db), "BTCUSDT", limit=20, mode="paper")
    assert len(paper_pnls) == 2
