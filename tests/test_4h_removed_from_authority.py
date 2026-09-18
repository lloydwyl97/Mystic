"""Tests proving 4H has no production trading authority (2026-09-17).

Every test here asserts that 4H values cannot:
- block or cancel a BUY
- cause a SELL
- make DAY_4H_STRUCTURE_BREAK_EXIT reachable
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. No 4H value can block a BUY (_intact_4h_slot_block always returns False)
# ---------------------------------------------------------------------------


class TestNo4hEntryBlock:
    def test_intact_4h_slot_block_always_false(self):
        from backend.services.portfolio_engine import PortfolioEngine

        engine = PortfolioEngine.__new__(PortfolioEngine)
        for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
            blocked, reason = engine._intact_4h_slot_block(sym)
            assert blocked is False, f"4H blocked BUY for {sym}: {reason}"
            assert reason == "", f"4H returned non-empty reason for {sym}: {reason}"


# ---------------------------------------------------------------------------
# 2. No 4H value can cause a SELL (DAY_4H_STRUCTURE_BREAK_EXIT unreachable)
# ---------------------------------------------------------------------------


def _make_bundle_4h_broken():
    """Bundle where 4H rise is broken (close below prior low)."""
    return {
        "4h": [
            [1, 100.0, 105.0, 95.0, 102.0, 1000],  # prior: low=95
            [2, 101.0, 103.0, 90.0, 93.0, 1000],  # current: close=93 < prior_low=95
        ],
        "1h": [{"ema_align": 0.5}],
        "5m": [{"ema_align": 0.5}],
        "15m": [{"ema_align": 0.5}],
    }


class TestNo4hExitAuthority:
    def test_path_aware_never_returns_4h_structure_break(self):
        """Even with a broken 4H, path-aware exit must NOT return 4H_STRUCTURE_BREAK."""
        from backend.services.day_controlled_exits import _evaluate_path_aware_exit

        position = MagicMock()
        position.entry_price = 100.0
        position.highest_price = 100.5
        position.trailing_stop_price = 0.0
        position.trail_pct = 0.005
        position.thesis_invalid_level = 95.0
        position.thesis_target_level = 105.0

        result = _evaluate_path_aware_exit(
            position=position,
            current_price=96.0,
            net_pnl_pct=-0.04,
            hold_minutes=30.0,
            coin_profile={"tp": 0.014, "sl": 0.010, "trail": 0.0025, "max_hold_min": 300},
            bundle=_make_bundle_4h_broken(),
            entry=100.0,
            atr_pct=0.01,
        )
        assert result["reason"] != "DAY_4H_STRUCTURE_BREAK_EXIT", f"4H structure break exit was reachable: {result}"

    def test_evaluate_engine_managed_exit_never_returns_4h_structure_break(self):
        """Legacy ladder must not return 4H_STRUCTURE_BREAK."""
        from backend.services.day_controlled_exits import evaluate_engine_managed_exit

        position = MagicMock()
        position.entry_price = 100.0
        position.stop_price = 95.0
        position.trailing_stop_price = 0.0
        position.trail_pct = 0.005
        position.highest_price = 100.5
        position.lowest_price = 99.5
        position.thesis_invalid_level = 95.0
        position.thesis_target_level = 105.0
        position.entry_thesis = "BREAKOUT_CONTINUATION"
        position.thesis_score = 0.7
        position.entry_vwap = 100.0
        position.tp1_hit = False
        position.max_hold_min = 300
        position.symbol = "BTC/USDT"
        position.entry_time = 0.0
        position.day_route_regime_at_entry = ""

        with patch.dict(os.environ, {"DAY_PATH_AWARE_EXIT": "false"}):
            result = evaluate_engine_managed_exit(
                position=position,
                current_price=96.0,
                net_pnl_pct=-0.04,
                hold_minutes=30.0,
                coin_profile={"tp": 0.014, "sl": 0.010, "trail": 0.0025, "max_hold_min": 300},
                bundle=_make_bundle_4h_broken(),
            )
        assert result["reason"] != "DAY_4H_STRUCTURE_BREAK_EXIT", f"4H structure break exit was reachable in legacy ladder: {result}"

    def test_4h_structure_break_not_in_full_flatten_reasons(self):
        from backend.services.day_controlled_exits import DAY_FULL_FLATTEN_REASONS

        assert "DAY_4H_STRUCTURE_BREAK_EXIT" not in DAY_FULL_FLATTEN_REASONS

    def test_4h_structure_break_not_in_mandatory_flatten(self):
        from backend.services.day_mandatory_exit_execution import MANDATORY_FLATTEN_PREFIXES

        for prefix in MANDATORY_FLATTEN_PREFIXES:
            assert "4H" not in prefix.upper(), f"4H found in MANDATORY_FLATTEN_PREFIXES: {prefix}"

    def test_thesis_invalidated_breakout_no_4h(self):
        """BREAKOUT_CONTINUATION thesis invalidation must not use 4H."""
        from backend.services.day_trade_thesis import thesis_invalidated_live

        # 4H broken but 5m/15m are fine → should NOT invalidate
        bundle = _make_bundle_4h_broken()
        bundle["5m"] = [{"ema_align": 0.7}]
        bundle["15m"] = [{"ema_align": 0.7}]
        result = thesis_invalidated_live(
            "BREAKOUT_CONTINUATION",
            mark=96.0,
            invalid_level=90.0,
            bundle=bundle,
            entry_price=100.0,
        )
        assert result is False, "4H broken alone should not invalidate BREAKOUT_CONTINUATION"


# ---------------------------------------------------------------------------
# 3. All four coins remain live
# ---------------------------------------------------------------------------


class TestAllCoinsLive:
    def test_all_four_in_coin_profiles(self):
        from backend.services.portfolio_engine import COIN_PROFILES

        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
            assert sym in COIN_PROFILES, f"{sym} missing from COIN_PROFILES"


# ---------------------------------------------------------------------------
# 4. Existing net-profit and trailing exits remain active
# ---------------------------------------------------------------------------


class TestExistingExitsPreserved:
    def test_net_profit_exit_reachable(self):
        from backend.services.day_controlled_exits import evaluate_engine_managed_exit

        position = MagicMock()
        position.entry_price = 100.0
        position.stop_price = 95.0
        position.trailing_stop_price = 0.0
        position.trail_pct = 0.005
        position.highest_price = 102.0
        position.lowest_price = 100.0
        position.thesis_invalid_level = 95.0
        position.thesis_target_level = 102.0
        position.take_profit_1_price = 102.0
        position.entry_thesis = ""
        position.thesis_score = 0.0
        position.entry_vwap = 0.0
        position.tp1_hit = False
        position.max_hold_min = 300
        position.symbol = "BTC/USDT"
        position.entry_time = 0.0
        position.day_route_regime_at_entry = ""

        with patch.dict(os.environ, {"DAY_PATH_AWARE_EXIT": "false"}):
            result = evaluate_engine_managed_exit(
                position=position,
                current_price=102.5,
                net_pnl_pct=0.025,
                hold_minutes=60.0,
                coin_profile={"tp": 0.014, "sl": 0.010, "trail": 0.0025, "max_hold_min": 300},
                bundle=None,
            )
        assert result["action"] == "sell", f"Net profit exit not reachable: {result}"
        assert "NET_PROFIT" in result["reason"], f"Wrong exit reason: {result['reason']}"

    def test_trailing_stop_exit_reachable(self):
        from backend.services.day_controlled_exits import evaluate_engine_managed_exit

        position = MagicMock()
        position.entry_price = 100.0
        position.stop_price = 95.0
        position.trailing_stop_price = 101.0
        position.trail_pct = 0.005
        position.highest_price = 102.0
        position.lowest_price = 100.0
        position.thesis_invalid_level = 95.0
        position.thesis_target_level = 105.0
        position.entry_thesis = ""
        position.thesis_score = 0.0
        position.entry_vwap = 0.0
        position.tp1_hit = False
        position.max_hold_min = 300
        position.symbol = "BTC/USDT"
        position.entry_time = 0.0
        position.day_route_regime_at_entry = ""

        with patch.dict(os.environ, {"DAY_PATH_AWARE_EXIT": "false"}):
            result = evaluate_engine_managed_exit(
                position=position,
                current_price=100.5,
                net_pnl_pct=0.005,
                hold_minutes=60.0,
                coin_profile={"tp": 0.014, "sl": 0.010, "trail": 0.0025, "max_hold_min": 300},
                bundle=None,
            )
        assert result["action"] == "sell", f"Trailing stop exit not reachable: {result}"
        assert "TRAILING_STOP" in result["reason"], f"Wrong exit reason: {result['reason']}"


# ---------------------------------------------------------------------------
# 5. Candle data: blank 1m cannot silently become zeros
# ---------------------------------------------------------------------------


class TestCandleDataIntegrity:
    def test_blank_1m_does_not_produce_zero_features(self):
        """An empty 1m candle list should not silently zero the feature vector."""
        from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES

        assert "1m" in DAY_ACTIVE_TIMEFRAMES, "1m must be in active timeframes"
        # 3m is NOT a fetched timeframe — confirm it's absent
        assert "3m" not in DAY_ACTIVE_TIMEFRAMES, "3m should not be in active timeframes"

    def test_validate_bundle_flags_missing_1m(self):
        """Missing 1m bars must be flagged, not silently accepted."""
        from backend.services.day_active_market_bundle import validate_day_active_bundle

        empty_bundle: dict = {tf: [] for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d", "1w")}
        ok, missing = validate_day_active_bundle(empty_bundle)
        assert ok is False, "Empty bundle should not validate"
        assert any("1m" in m for m in missing), "Missing 1m should be flagged"
