"""Clean acceptance classification and SCALP measurement (not permission) tests."""

from __future__ import annotations

from pathlib import Path

from backend.services.binance_scalp.scalp_setup_measurements import evidence_rank_delta, measure_all_setups
from backend.services.binance_scalp.strategies.base import StrategyMarketContext
from backend.services.validation_cutoff import (
    is_strategy_acceptance_eligible,
    mark_reconciliation_manual_exit,
)


def test_reconciliation_exit_excluded_from_acceptance():
    assert is_strategy_acceptance_eligible(exit_reason="NET_PROFIT_EXIT") is True
    assert is_strategy_acceptance_eligible(exit_reason="RECONCILIATION_MANUAL_EXIT") is False
    assert is_strategy_acceptance_eligible(exit_reason="MANUAL_EXIT", trade_id="983") is False
    assert is_strategy_acceptance_eligible(exit_reason="STALL_EXIT", extra={"acceptance_class": "RECONCILIATION_MANUAL_EXIT"}) is False


def test_mark_reconciliation_keeps_pnl(tmp_path: Path):
    import sqlite3

    db = str(tmp_path / "day.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT,
            exit_reason TEXT, pnl REAL, explainability_json TEXT
        )
        """
    )
    conn.execute("INSERT INTO paper_trades VALUES (983,'t','XRP/USDT','SELL','MANUAL_EXIT',-9.6,'{}')")
    conn.commit()
    conn.close()
    out = mark_reconciliation_manual_exit(db, trade_id=983)
    assert out["updated"] is True
    assert out["pnl"] == -9.6
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT exit_reason, pnl, explainability_json FROM paper_trades WHERE id=983").fetchone()
    conn.close()
    assert row[0] == "RECONCILIATION_MANUAL_EXIT"
    assert row[1] == -9.6
    assert "RECONCILIATION_MANUAL_EXIT" in row[2]


def test_measure_all_setups_returns_nine():
    class Snap:
        mid = 100.0
        spread_pct = 0.0003
        order_book_imbalance = 0.1
        bids = [[99.9, 10.0]]
        asks = [[100.1, 10.0]]

    class Mom:
        mid_change_15s = 0.0002
        mid_change_30s = 0.0003
        mid_change_60s = 0.0001
        bid_change_15s = 0.0002
        bid_change_30s = 0.0002
        bid_change_60s = 0.0001
        realized_volatility_pct = 0.002

    bars = [{"open": 99.8, "high": 100.2, "low": 99.7, "close": 100.0 + i * 0.01, "volume": 10 + i} for i in range(20)]
    ctx = StrategyMarketContext(symbol="BTCUSDT", snap=Snap(), mom=Mom(), bars_1m=bars, econ=None, config=None, notional_usd=50)
    meas = measure_all_setups(ctx)
    assert len(meas) == 9
    assert "reclaim_strength" in meas["vwap_ema_reclaim"]
    assert "momentum_flip_strength" in meas["range_bounce_scalp"]
    delta = evidence_rank_delta(meas)
    assert -0.05 <= delta <= 0.05


def test_opportunity_cycle_writes_under_immediate_lock(tmp_path: Path):
    import sqlite3

    from backend.services.binance_scalp.scalp_opportunity_dataset import record_opportunity_cycle
    from backend.services.binance_scalp.schema import init_scalp_schema

    db = str(tmp_path / "scalp.db")
    init_scalp_schema(db, principal=1000.0)
    holder = sqlite3.connect(db, timeout=10)
    holder.execute("BEGIN IMMEDIATE")
    n = record_opportunity_cycle(
        db,
        rows=[{"symbol": "BTCUSDT", "mid": 100.0, "spread_pct": 0.0002, "rank_score": 0.4, "strategy_passed": False}],
        epoch=1.0,
        conn=holder,
    )
    holder.commit()
    holder.close()
    assert n == 1
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*), symbol FROM scalp_opportunity_snapshots").fetchone()[0] == 1
    conn.close()


def test_exit_manager_columns_repair_when_version_already_3(tmp_path: Path):
    import sqlite3

    from backend.services.binance_scalp.schema import apply_scalp_migrations

    db = str(tmp_path / "scalp.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE scalp_meta (id INTEGER PRIMARY KEY, schema_version INTEGER NOT NULL);
        INSERT INTO scalp_meta VALUES (1, 3);
        CREATE TABLE scalp_paper_positions (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            exchange TEXT,
            strategy_id TEXT,
            quantity REAL,
            entry_price REAL,
            entry_time TEXT,
            entry_time_epoch REAL,
            trade_id TEXT,
            paper_order_id TEXT,
            status TEXT,
            reprice_count INTEGER,
            diagnostics_json TEXT
        );
        """
    )
    conn.commit()
    applied = apply_scalp_migrations(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(scalp_paper_positions)")}
    conn.close()
    assert "last_review_ts" in cols
    assert "state" in cols
    assert any("migrate_exit_manager_v3" in a for a in applied)
