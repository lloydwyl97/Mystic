"""Phase 3e entry gate — momentum-based projected gross move."""

from __future__ import annotations

from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.momentum_gross_estimate import MomentumGrossEstimate
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics

MOMENTUM_GROSS_BELOW_REQUIRED = "MOMENTUM_GROSS_BELOW_REQUIRED"
PROJECTED_SURPLUS_TOO_SMALL = "PROJECTED_SURPLUS_TOO_SMALL"
SCALP_NO_MOMENTUM_CONFIRMATION = "SCALP_NO_MOMENTUM_CONFIRMATION"
BREAKOUT_NOT_CONFIRMED = "BREAKOUT_NOT_CONFIRMED"
RECENT_RANGE_TOO_SMALL = "RECENT_RANGE_TOO_SMALL"
MOMENTUM_DATA_INSUFFICIENT = "MOMENTUM_DATA_INSUFFICIENT"

# Legacy alias retained for reject table continuity
GROSS_EDGE_BELOW_REQUIRED = MOMENTUM_GROSS_BELOW_REQUIRED


def build_reachability_diagnostics(
    econ: ScalpEconomics,
    *,
    spread_pct: float,
    buy_impact_pct: float,
    sell_impact_pct: float,
    estimate: MomentumGrossEstimate,
    momentum: MomentumDiagnostics | None = None,
    selected_symbol: str | None = None,
    reject_reason: str | None = None,
) -> dict:
    required = econ.entry_required_gross_edge_pct(spread_pct, buy_impact_pct, sell_impact_pct)
    projected = estimate.projected_gross_move_pct
    surplus = projected - required
    reach = {
        "required_gross_move_pct": required,
        "projected_gross_move_pct": projected,
        "projected_edge_pct": projected,
        "projected_edge_minus_required": surplus,
        "projected_surplus_pct": surplus,
        "min_projected_surplus_pct": econ.min_projected_surplus_pct,
        "spread_pct": spread_pct,
        "fee_cost_pct": econ.entry_fee_pct() + econ.exit_fee_pct(),
        "slippage_pct": econ.slippage_buffer_pct,
        "impact_pct": buy_impact_pct + sell_impact_pct,
        "roundtrip_cost_pct": econ.roundtrip_cost_pct(spread_pct, buy_impact_pct, sell_impact_pct),
        "net_profit_target_pct": econ.net_profit_target_pct,
        "entry_edge_buffer_pct": econ.entry_edge_buffer_pct,
        "entry_required_gross_edge_pct_env": econ.entry_required_gross_edge_pct_env,
        **estimate.as_dict(),
    }
    if selected_symbol:
        reach["selected_symbol"] = selected_symbol
    if reject_reason:
        reach["reject_reason"] = reject_reason
    if momentum is not None:
        reach["momentum_confirmed"] = momentum.momentum_confirmed
        reach["flat_regime"] = momentum.flat_regime
        reach.update(momentum.as_dict())
    return reach


def evaluate_buy_entry_gate(
    econ: ScalpEconomics,
    *,
    spread_pct: float,
    buy_impact_pct: float,
    sell_impact_pct: float,
    estimate: MomentumGrossEstimate,
    momentum: MomentumDiagnostics | None,
    apply_entry_gate: bool = True,
    selected_symbol: str | None = None,
) -> tuple[bool, str, dict]:
    """Return (passed, reject_reason, reachability_diagnostics)."""
    required = econ.entry_required_gross_edge_pct(spread_pct, buy_impact_pct, sell_impact_pct)
    projected = estimate.projected_gross_move_pct
    surplus = projected - required

    def _fail(reason: str) -> tuple[bool, str, dict]:
        reach = build_reachability_diagnostics(
            econ,
            spread_pct=spread_pct,
            buy_impact_pct=buy_impact_pct,
            sell_impact_pct=sell_impact_pct,
            estimate=estimate,
            momentum=momentum,
            selected_symbol=selected_symbol,
            reject_reason=reason,
        )
        if not apply_entry_gate:
            return True, "", reach
        return False, reason, reach

    if not estimate.data_sufficient:
        return _fail(MOMENTUM_DATA_INSUFFICIENT)

    if not estimate.range_sufficient:
        return _fail(RECENT_RANGE_TOO_SMALL)

    if not estimate.breakout_confirmed:
        return _fail(BREAKOUT_NOT_CONFIRMED)

    if momentum is None or not momentum.momentum_confirmed:
        return _fail(SCALP_NO_MOMENTUM_CONFIRMATION)

    if projected < required:
        return _fail(MOMENTUM_GROSS_BELOW_REQUIRED)

    if surplus < econ.min_projected_surplus_pct:
        return _fail(PROJECTED_SURPLUS_TOO_SMALL)

    reach = build_reachability_diagnostics(
        econ,
        spread_pct=spread_pct,
        buy_impact_pct=buy_impact_pct,
        sell_impact_pct=sell_impact_pct,
        estimate=estimate,
        momentum=momentum,
        selected_symbol=selected_symbol,
    )
    if not apply_entry_gate:
        return True, "", reach
    return True, "", reach
