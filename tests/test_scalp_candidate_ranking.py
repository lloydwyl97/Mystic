"""Unit tests for scalp candidate ranking repairs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.scalp_candidate_ranking import (
    SOFT_REJECT_SCORE,
    pick_best_global_candidate,
    rank_setup_signal,
)
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext


def _ctx(*, spread: float = 0.0002, mid_change_15s: float = 0.0, confirmed: bool = False) -> StrategyMarketContext:
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    snap = SimpleNamespace(
        symbol="BTCUSDT",
        spread_pct=spread,
        best_ask=100.0,
        best_bid=99.98,
        mid=100.0,
        asks=[[100.0, 1000.0]],
    )
    mom = SimpleNamespace(
        mid_change_15s=mid_change_15s,
        mid_change_30s=mid_change_15s,
        bid_change_15s=mid_change_15s,
        momentum_confirmed=confirmed,
    )
    return StrategyMarketContext(
        symbol="BTCUSDT",
        snap=snap,
        mom=mom,
        bars_1m=[{"low": 99.5, "high": 100.5, "close": 100.0}] * 15,
        econ=econ,
        config=config,
        notional_usd=25.0,
    )


def _soft_sig(reason: str, *, expected: float = 0.0035) -> ScalpSetupSignal:
    return ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=0.0,
        setup_name="range_bounce_scalp",
        confidence=0.4,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=expected,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=False,
        reject_reason=reason,
    )


def test_no_rejection_wick_below_min_tradeable():
    ranked = rank_setup_signal(_soft_sig("NO_REJECTION_WICK"), regime="range", ctx=_ctx())
    assert ranked.rank_score < 1.45
    assert ranked.entry_eligible is False


def test_not_near_support_below_wick_when_flat():
    wick = rank_setup_signal(_soft_sig("NO_REJECTION_WICK"), regime="range", ctx=_ctx())
    support = rank_setup_signal(_soft_sig("NOT_NEAR_SUPPORT"), regime="range", ctx=_ctx())
    assert support.rank_score > wick.rank_score


def test_passed_signal_still_eligible():
    ctx = _ctx()
    passed = ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=2.6,
        setup_name="range_bounce_scalp",
        confidence=0.7,
        entry_reason="support_bounce",
        invalidation_reason="support_break",
        required_target_pct=0.0025,
        expected_move_pct=0.0035,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=True,
        reject_reason=None,
    )
    ranked = rank_setup_signal(passed, regime="range", ctx=ctx)
    assert ranked.entry_eligible is True
    assert ranked.rank_score >= 2.0


def test_weak_global_tie_returns_none():
    rows = [
        {
            "symbol": sym,
            "rank_score": 1.35,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NO_REJECTION_WICK", "regime_native": True},
            "signal": SimpleNamespace(passed=False, spread_pct=0.00005 if sym == "BTCUSDT" else 0.0002),
            "mom": SimpleNamespace(mid_change_15s=0.0, mid_change_30s=0.0),
        }
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    ]
    assert pick_best_global_candidate(rows) is None


def test_global_tie_prefers_non_btc_when_scores_equal():
    rows = [
        {
            "symbol": "BTCUSDT",
            "rank_score": 1.62,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NOT_NEAR_SUPPORT", "regime_native": True, "reachability_surplus": 0.001},
            "signal": SimpleNamespace(passed=False, spread_pct=0.00005),
            "mom": SimpleNamespace(mid_change_15s=0.0001, mid_change_30s=0.0001),
            "intelligence": {"memory_rank_delta": 0.0},
        },
        {
            "symbol": "ETHUSDT",
            "rank_score": 1.62,
            "entry_eligible": True,
            "rank_meta": {"soft_reason": "NOT_NEAR_SUPPORT", "regime_native": True, "reachability_surplus": 0.0012},
            "signal": SimpleNamespace(passed=False, spread_pct=0.0002),
            "mom": SimpleNamespace(mid_change_15s=0.0002, mid_change_30s=0.0002),
            "intelligence": {"memory_rank_delta": 0.01},
        },
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "ETHUSDT"


def test_soft_scores_materially_lower_than_before():
    assert SOFT_REJECT_SCORE["NO_REJECTION_WICK"] < 1.0
    assert SOFT_REJECT_SCORE["NOT_NEAR_SUPPORT"] < 1.45
    assert SOFT_REJECT_SCORE["NOT_NEAR_SUPPORT"] > SOFT_REJECT_SCORE["NO_REJECTION_WICK"]
