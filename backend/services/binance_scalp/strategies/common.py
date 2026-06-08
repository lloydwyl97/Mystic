"""Shared strategy helpers — spread, depth, target reachability."""

from __future__ import annotations

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.orderbook_book import walk_buy_notional
from backend.services.binance_scalp.paper_spread_caps import uses_paper_spread_caps
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext


def spread_cap(econ: ScalpEconomics, config: ScalpConfig, symbol: str) -> float:
    if uses_paper_spread_caps(
        scalp_live=config.scalp_live,
        calibration_mode=config.calibration_mode,
        scalp_paper_enabled=config.scalp_paper_enabled,
    ):
        return econ.spread_cap_for_symbol(symbol)
    return econ.spread_cap_pct


def check_spread(snap: MarketSnapshot, econ: ScalpEconomics, config: ScalpConfig) -> tuple[bool, str | None]:
    cap = spread_cap(econ, config, snap.symbol)
    if snap.spread_pct > cap:
        return False, "SPREAD_TOO_WIDE"
    return True, None


def depth_check(snap: MarketSnapshot, notional: float, econ: ScalpEconomics) -> tuple[bool, float, float]:
    walk = walk_buy_notional(snap.asks, notional, snap.best_ask)
    impact = float(walk.impact_pct)
    ok = walk.depth_sufficient and impact <= econ.impact_cap_pct
    return ok, impact, float(walk.expected_avg_fill or snap.best_ask)


def target_reachable(
    econ: ScalpEconomics,
    *,
    spread_pct: float,
    impact_pct: float,
    expected_move_pct: float,
) -> tuple[bool, float]:
    req = econ.entry_required_gross_edge_pct(spread_pct, impact_pct, 0.0)
    surplus = expected_move_pct - req
    return surplus >= econ.min_projected_surplus_pct, req


def reject_signal(
    ctx: StrategyMarketContext,
    setup_name: str,
    reason: str,
    *,
    expected_move: float = 0.0,
    impact: float = 0.0,
) -> ScalpSetupSignal:
    return ScalpSetupSignal(
        symbol=ctx.symbol,
        side="BUY",
        score=0.0,
        setup_name=setup_name,
        confidence=0.0,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=ctx.econ.net_profit_target_pct,
        expected_move_pct=expected_move,
        spread_pct=ctx.snap.spread_pct,
        impact_pct=impact,
        depth_sufficient=False,
        limit_buy_price=ctx.snap.best_ask,
        passed=False,
        reject_reason=reason,
    )


def pass_signal(
    ctx: StrategyMarketContext,
    setup_name: str,
    *,
    score: float,
    confidence: float,
    entry_reason: str,
    invalidation_reason: str,
    expected_move_pct: float,
    impact_pct: float,
    limit_buy: float,
    setup_context: dict,
) -> ScalpSetupSignal:
    return ScalpSetupSignal(
        symbol=ctx.symbol,
        side="BUY",
        score=score,
        setup_name=setup_name,
        confidence=confidence,
        entry_reason=entry_reason,
        invalidation_reason=invalidation_reason,
        required_target_pct=ctx.econ.net_profit_target_pct,
        expected_move_pct=expected_move_pct,
        spread_pct=ctx.snap.spread_pct,
        impact_pct=impact_pct,
        depth_sufficient=True,
        limit_buy_price=limit_buy,
        passed=True,
        reject_reason=None,
        setup_context=setup_context,
    )
