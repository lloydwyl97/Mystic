"""Unit tests for the cross-engine MFE/MAE distribution learner (item p5)."""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import mfe_mae_distribution_learner as dist
from backend.services.market_role_outcome_learner import _SCHEMA_SQL


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = str(tmp_path / "test_outcomes.db")
    with sqlite3.connect(p) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    dist._cache.clear()
    monkeypatch.setattr(dist, "MIN_OBS", 5)
    return p


def _insert(db_path, *, symbol, strategy, pnl_pct, mfe_pct, mae_pct, vol_score=0.35, mom_score=0.5, hold_seconds=600):
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO market_role_trade_outcomes
              (trade_id, buy_trade_id, symbol, strategy, realized_pnl_pct,
               hold_seconds, exit_reason, mfe_pct, mae_pct, market_regime,
               volatility_score, momentum_score, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"t_{time.time_ns()}",
                "buy",
                symbol.upper(),
                strategy.lower(),
                pnl_pct,
                hold_seconds,
                "TEST_EXIT",
                mfe_pct,
                mae_pct,
                "range",
                vol_score,
                mom_score,
                now_iso,
            ),
        )
        conn.commit()


def test_insufficient_data_returns_honest_status(db_path):
    for _ in range(2):
        _insert(db_path, symbol="BTCUSDT", strategy="scalp", pnl_pct=0.5, mfe_pct=0.01, mae_pct=0.002)
    result = dist.get_mfe_distribution("BTCUSDT", "scalp", db_path=db_path)
    assert result.confidence_status == "insufficient_data"
    assert result.n_obs == 0  # below MIN_OBS everywhere, including cross-symbol fallback


def test_mfe_uses_winners_only(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETHUSDT", strategy="day", pnl_pct=1.0, mfe_pct=0.02 + i * 0.001, mae_pct=0.001)
    for i in range(6):
        _insert(db_path, symbol="ETHUSDT", strategy="day", pnl_pct=-1.0, mfe_pct=0.5, mae_pct=0.015 + i * 0.001)  # should never enter MFE distribution
    mfe = dist.get_mfe_distribution("ETHUSDT", "day", db_path=db_path)
    assert mfe.confidence_status != "insufficient_data"
    assert mfe.n_obs == 6
    # Winner MFEs are all < 0.03; loser's fake 0.5 must never leak in.
    assert mfe.percentiles["p90"] < 0.03


def test_mae_uses_losers_only(db_path):
    for i in range(6):
        _insert(db_path, symbol="SOLUSDT", strategy="scalp", pnl_pct=1.0, mfe_pct=0.02, mae_pct=0.5)  # winner MAE must not leak in
    for i in range(6):
        _insert(db_path, symbol="SOLUSDT", strategy="scalp", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.01 + i * 0.001)
    mae = dist.get_mae_distribution("SOLUSDT", "scalp", db_path=db_path)
    assert mae.n_obs == 6
    assert mae.percentiles["p90"] < 0.02


def test_cascade_falls_back_when_bucket_too_specific(db_path):
    for i in range(6):
        _insert(db_path, symbol="XRPUSDT", strategy="day", pnl_pct=1.0, mfe_pct=0.01 + i * 0.001, mae_pct=0.001, vol_score=0.35)
    # Asking for a vol bucket with zero matching rows must cascade to the
    # coarser (symbol)-only stratum rather than reporting insufficient_data.
    result = dist.get_mfe_distribution("XRPUSDT", "day", vol_bucket_filter="high_vol", db_path=db_path)
    assert result.n_obs == 6
    assert result.stratum_used == "symbol"
    assert result.fallback_from == "symbol+vol"


def test_cross_symbol_fallback_when_symbol_itself_thin(db_path):
    for i in range(6):
        _insert(db_path, symbol="BTCUSDT", strategy="day", pnl_pct=1.0, mfe_pct=0.01 + i * 0.001, mae_pct=0.001)
    for i in range(2):
        _insert(db_path, symbol="ETHUSDT", strategy="day", pnl_pct=1.0, mfe_pct=0.03, mae_pct=0.001)
    result = dist.get_mfe_distribution("ETHUSDT", "day", db_path=db_path)
    assert result.stratum_used == "strategy_cross_symbol"
    assert result.n_obs == 8  # BTC + ETH pooled


def test_vol_and_momentum_bucket_boundaries():
    assert dist.vol_bucket(0.1) == "low_vol"
    assert dist.vol_bucket(0.4) == "mid_vol"
    assert dist.vol_bucket(0.9) == "high_vol"
    assert dist.vol_bucket(None) is None
    assert dist.momentum_bucket(0.2) == "momentum_down"
    assert dist.momentum_bucket(0.5) == "momentum_flat"
    assert dist.momentum_bucket(0.9) == "momentum_up"


def test_hold_time_bucket_differs_by_strategy():
    assert dist.hold_time_bucket(30, "scalp") == "hold_lt_1m"
    assert dist.hold_time_bucket(30, "day") == "hold_lt_15m"
    assert dist.hold_time_bucket(20000, "day") == "hold_4h_24h"


def test_get_expected_mfe_mae_combines_both(db_path):
    for i in range(6):
        _insert(db_path, symbol="BTCUSDT", strategy="scalp", pnl_pct=1.0, mfe_pct=0.004 + i * 0.0002, mae_pct=0.0005)
    for i in range(6):
        _insert(db_path, symbol="BTCUSDT", strategy="scalp", pnl_pct=-1.0, mfe_pct=0.0005, mae_pct=0.002 + i * 0.0002)
    expected = dist.get_expected_mfe_mae("BTCUSDT", "scalp", db_path=db_path)
    assert expected.expected_mfe_p60 > 0
    assert expected.expected_mae_p60 > 0
    assert expected.mfe_confidence != "insufficient_data"
    assert expected.mae_confidence != "insufficient_data"


def test_never_raises_on_missing_table(tmp_path):
    empty_db = str(tmp_path / "empty.db")
    result = dist.get_mfe_distribution("BTCUSDT", "day", db_path=empty_db)
    assert result.confidence_status == "insufficient_data"
