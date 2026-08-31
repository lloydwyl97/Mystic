"""Candle-read upgrades: true wick fraction + SCALP MTF confirmation gate."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.scalp_strategy_router import (
    ScalpStrategyRouter,
    _mtf_confirmation_gate_enabled,
)
from backend.services.binance_scalp.strategies.base import StrategyMarketContext
from backend.services.binance_scalp.strategies.range_bounce_scalp import RangeBounceScalpStrategy


class _PermissiveEcon:
    min_projected_surplus_pct = 0.0001
    impact_cap_pct = 0.01
    spread_cap_pct = 0.01
    net_profit_target_pct = 0.0025

    def entry_required_gross_edge_pct(self, spread_pct, impact_pct, extra):
        return 0.0005

    def spread_cap_for_symbol(self, symbol):
        return 0.01


def _ctx_with_bar(*, open_, high, low, close, mid=None):
    mid = mid if mid is not None else close
    econ = _PermissiveEcon()
    config = get_scalp_config()
    snap = SimpleNamespace(
        symbol="BTCUSDT",
        spread_pct=0.0001,
        best_ask=mid + 0.01,
        best_bid=mid - 0.01,
        mid=mid,
        asks=[[mid + 0.01, 100000.0]],
    )
    mom = SimpleNamespace(
        bid_change_15s=0.0005,
        mid_change_15s=0.0005,
        mid_change_30s=0.0005,
        bid_change_60s=0.0,
        momentum_confirmed=True,
    )
    # Support near mid so NOT_NEAR_SUPPORT does not fire
    support = mid * 0.999
    bars = [{"open": support + 0.01, "high": mid + 0.2, "low": support, "close": support + 0.01} for _ in range(14)]
    bars.append({"open": open_, "high": high, "low": low, "close": close})
    return StrategyMarketContext(
        symbol="BTCUSDT",
        snap=snap,
        mom=mom,
        bars_1m=bars,
        econ=econ,
        config=config,
        notional_usd=25.0,
    )


def test_wick_rejection_is_range_normalized():
    """Lower wick = (min(o,c)-low)/(high-low), not (close-low)/close."""
    # Keep support within 0.2% of mid so NOT_NEAR_SUPPORT does not fire.
    # Long lower wick: open=close near high of a tight range.
    mid = 100.0
    low = 99.85
    high = 100.05
    close = 100.0
    ctx = _ctx_with_bar(open_=close, high=high, low=low, close=close, mid=mid)
    sig = RangeBounceScalpStrategy().evaluate(ctx)
    assert sig.passed, sig.reject_reason
    expected = (min(close, close) - low) / (high - low)
    assert abs(sig.setup_context["wick_rejection_pct"] - expected) < 1e-9
    assert abs(sig.setup_context["wick_rejection_range_pct"] - expected) < 1e-9
    legacy = (close - low) / close
    assert abs(sig.setup_context["wick_close_above_low_pct"] - legacy) < 1e-9
    assert abs(legacy - expected) > 1e-4


def test_mtf_gate_env_defaults_on():
    prev = os.environ.pop("SCALP_MTF_CONFIRMATION_GATE_ENABLED", None)
    try:
        assert _mtf_confirmation_gate_enabled() is True
    finally:
        if prev is not None:
            os.environ["SCALP_MTF_CONFIRMATION_GATE_ENABLED"] = prev


def test_mtf_gate_blocks_when_5m_down():
    os.environ["SCALP_MTF_CONFIRMATION_GATE_ENABLED"] = "true"
    os.environ["SCALP_MTF_REQUIRE_15M"] = "false"
    try:
        # Minimal router with mocked klines returning a down 5m trend
        config = get_scalp_config()
        econ = _PermissiveEcon()
        reader = MagicMock()
        reader.read.return_value = SimpleNamespace(
            symbol="BTCUSDT",
            spread_pct=0.0001,
            best_ask=100.01,
            best_bid=99.99,
            mid=100.0,
            asks=[[100.01, 100000.0]],
        )
        momentum = MagicMock()
        momentum.diagnostics.return_value = SimpleNamespace(
            bid_change_15s=0.0005,
            mid_change_15s=0.0005,
            mid_change_30s=0.0005,
            bid_change_60s=0.0,
            momentum_confirmed=True,
        )
        klines = MagicMock()
        # Down 5m: recent mean < prior mean
        down_5m = [{"close": 110.0}] * 3 + [{"close": 100.0}] * 3
        klines.get.return_value = [{"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0} for _ in range(20)]
        klines.get_5m.return_value = down_5m
        klines.get_15m.return_value = [{"close": 100.0}] * 6
        klines.get_1h.return_value = []

        router = ScalpStrategyRouter(config=config, econ=econ, reader=reader, momentum=momentum, klines=klines)
        # Force a passed ranked candidate by stubbing rank path: call confirmation directly
        trend, aligned = router._mtf_trend_confirmation("BTCUSDT", "5m")
        assert aligned is False
        assert trend is not None and trend < 0
    finally:
        os.environ.pop("SCALP_MTF_CONFIRMATION_GATE_ENABLED", None)
        os.environ.pop("SCALP_MTF_REQUIRE_15M", None)
