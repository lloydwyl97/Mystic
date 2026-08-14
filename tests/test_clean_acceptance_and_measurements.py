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
    conn.execute(
        "INSERT INTO paper_trades VALUES (983,'t','XRP/USDT','SELL','MANUAL_EXIT',-9.6,'{}')"
    )
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
