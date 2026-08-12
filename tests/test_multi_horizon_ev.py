"""Item p11: multi-horizon EV (SCALP 30s-20m, DAY 15m-24h composite weighted EV)."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

from backend.services import mfe_mae_distribution_learner as dist
from backend.services import multi_horizon_ev as mhev
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


def _insert(db_path, *, symbol, strategy, pnl_pct, mfe_pct, mae_pct, hold_seconds):
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
                0.35,
                0.5,
                now_iso,
            ),
        )
        conn.commit()


def test_insufficient_data_reports_unavailable(db_path):
    result = mhev.compute_multi_horizon_ev("BTCUSDT", "scalp", db_path=db_path)
    assert result.available is False
    assert result.degraded_reason == "insufficient_data_all_horizons"


def test_scalp_horizons_use_correct_bucket_set(db_path):
    result = mhev.compute_multi_horizon_ev("BTCUSDT", "scalp", db_path=db_path)
    assert tuple(h.bucket for h in result.horizons) == mhev.SCALP_HORIZON_BUCKETS


def test_day_horizons_use_correct_bucket_set(db_path):
    result = mhev.compute_multi_horizon_ev("BTCUSDT", "day", db_path=db_path)
    assert tuple(h.bucket for h in result.horizons) == mhev.DAY_HORIZON_BUCKETS


def test_composite_ev_reflects_good_short_horizon_and_bad_long_horizon(db_path):
    # hold_lt_1m: consistently strong winners.
    for _ in range(10):
        _insert(db_path, symbol="ETHUSDT", strategy="scalp", pnl_pct=0.01, mfe_pct=0.015, mae_pct=0.002, hold_seconds=30)
    # hold_5m_20m: consistently losers.
    for _ in range(10):
        _insert(db_path, symbol="ETHUSDT", strategy="scalp", pnl_pct=-0.01, mfe_pct=0.002, mae_pct=0.015, hold_seconds=600)

    result = mhev.compute_multi_horizon_ev("ETHUSDT", "scalp", db_path=db_path, cost_pct=0.0)
    assert result.available is True
    short = next(h for h in result.horizons if h.bucket == "hold_lt_1m")
    longer = next(h for h in result.horizons if h.bucket == "hold_5m_20m")
    assert short.net_ev_pct > 0
    assert longer.net_ev_pct < 0
    assert result.composite_ev_pct is not None


def test_disabled_via_env_returns_unavailable(db_path, monkeypatch):
    monkeypatch.setenv("MULTI_HORIZON_EV_ENABLED", "false")
    result = mhev.compute_multi_horizon_ev("BTCUSDT", "day", db_path=db_path)
    assert result.available is False
    assert result.degraded_reason == "disabled"


def test_cost_pct_reduces_composite_ev(db_path):
    for _ in range(10):
        _insert(db_path, symbol="SOLUSDT", strategy="day", pnl_pct=0.02, mfe_pct=0.025, mae_pct=0.003, hold_seconds=600)
    for _ in range(5):
        _insert(db_path, symbol="SOLUSDT", strategy="day", pnl_pct=-0.005, mfe_pct=0.004, mae_pct=0.008, hold_seconds=600)
    zero_cost = mhev.compute_multi_horizon_ev("SOLUSDT", "day", db_path=db_path, cost_pct=0.0)
    with_cost = mhev.compute_multi_horizon_ev("SOLUSDT", "day", db_path=db_path, cost_pct=0.01)
    assert with_cost.composite_ev_pct < zero_cost.composite_ev_pct


def test_horizon_and_result_to_dict_are_json_safe():
    result = mhev.compute_multi_horizon_ev("BTCUSDT", "scalp", db_path=":memory:")
    payload = result.to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert isinstance(payload["horizons"], list)
