"""
Regression test (final pre-push audit item 2): NO_REJECTION_WICK must not be
a hard trade-opinion entry blocker in RangeBounceScalpStrategy.

Prior behavior: `evaluate()` returned a hard `reject_signal(..., "NO_REJECTION_WICK")`
whenever the last 1m bar's lower wick was smaller than a fixed threshold,
regardless of how strong the rest of the setup's real-time evidence (support
proximity, momentum flip, sustained momentum, executable net edge) was. Since
`SOFT_REJECT_SCORE["NO_REJECTION_WICK"] = 0.92` plus the capped momentum boost
(max 0.16) can never reach `_min_tradeable_score()` (1.45), routing this
reject through the "soft" ranking path was a de facto hard block dressed up
as a soft one — a genuine candle-shape trade-opinion gate.

Fix: the strategy no longer inspects wick size as a pass/fail gate. Wick
strength still flows into `score`/`confidence` as a continuous contribution
(matching the user's required remedy: score contribution, not another
pattern gate), but a fully flat/no-wick close no longer removes an otherwise
executable candidate — real bounce evidence is still enforced by the
momentum-flip checks (live 15s/30s/60s order-book momentum), which are
genuine real-time facts, not candle-shape opinion.
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.binance_scalp.strategies.base import StrategyMarketContext
from backend.services.binance_scalp.strategies.range_bounce_scalp import RangeBounceScalpStrategy
from backend.services.binance_scalp.config import get_scalp_config


def _bars(*, low: float, high: float, closes: list[float]) -> list[dict]:
    # Include open so true range-normalized wick fraction can be computed.
    bars = [{"open": low + 0.1, "low": low, "high": high, "close": low + 0.1} for _ in range(14)]
    bars.append({"open": closes[-1], "low": low, "high": high, "close": closes[-1]})
    return bars


class _PermissiveEcon:
    """Economics stub isolating the wick-gate test from unrelated fee/edge tuning."""

    min_projected_surplus_pct = 0.0001
    impact_cap_pct = 0.01
    spread_cap_pct = 0.01
    net_profit_target_pct = 0.0025

    def entry_required_gross_edge_pct(self, spread_pct: float, impact_pct: float, extra: float) -> float:
        return 0.0005

    def spread_cap_for_symbol(self, symbol: str) -> float:
        return 0.01


def _ctx(*, last_close: float, low: float = 99.5, high: float = 100.5, mid: float = 100.3) -> StrategyMarketContext:
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
    return StrategyMarketContext(
        symbol="BTCUSDT",
        snap=snap,
        mom=mom,
        bars_1m=_bars(low=low, high=high, closes=[last_close]),
        econ=econ,
        config=config,
        notional_usd=25.0,
    )


def test_zero_wick_candidate_is_not_hard_rejected():
    """Last bar closes exactly at its low (zero rejection wick) — must still pass."""
    ctx = _ctx(last_close=99.5, low=99.5, high=99.7, mid=99.62)
    sig = RangeBounceScalpStrategy().evaluate(ctx)
    assert sig.passed is True, f"zero-wick candidate must not be hard-rejected, got reject_reason={sig.reject_reason}"
    assert sig.reject_reason != "NO_REJECTION_WICK"
    assert sig.setup_context.get("wick_rejection_pct") == 0.0


def test_small_wick_contributes_to_score_not_a_gate():
    """A tiny (previously sub-threshold) wick lowers score smoothly but does not block."""
    ctx_no_wick = _ctx(last_close=99.62, low=99.5, high=99.7, mid=99.62)
    ctx_small_wick = _ctx(last_close=99.65, low=99.5, high=99.7, mid=99.65)
    sig_no_wick = RangeBounceScalpStrategy().evaluate(ctx_no_wick)
    sig_small_wick = RangeBounceScalpStrategy().evaluate(ctx_small_wick)
    assert sig_no_wick.passed and sig_small_wick.passed
    # Bigger wick -> higher score (continuous contribution), but the smaller
    # one is still a valid, tradeable, passed signal — not rejected.
    assert sig_small_wick.score >= sig_no_wick.score


def test_no_rejection_wick_reason_never_emitted_by_strategy():
    """Sweep a range of wick sizes down to zero; NO_REJECTION_WICK must never appear."""
    for close_offset in (0.0, 0.02, 0.05, 0.1, 0.15):
        low = 99.5
        ctx = _ctx(last_close=low + close_offset, low=low, high=99.7, mid=low + close_offset)
        sig = RangeBounceScalpStrategy().evaluate(ctx)
        assert sig.reject_reason != "NO_REJECTION_WICK"
