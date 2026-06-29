"""Protected preflight for Binance.US scalp — fail-closed, no orders."""

from __future__ import annotations

from dataclasses import dataclass

from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.entry_gate import (
    MOMENTUM_GROSS_BELOW_REQUIRED,
    SCALP_NO_MOMENTUM_CONFIRMATION,
    evaluate_buy_entry_gate,
)
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_gross_estimate import compute_momentum_gross_estimate
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics
from backend.services.binance_scalp.orderbook_book import walk_buy_notional, walk_sell_qty

FEE_MODEL_UNVERIFIED = "FEE_MODEL_UNVERIFIED"
SCALP_PAPER_DISABLED = "SCALP_PAPER_DISABLED"
SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
PRICE_IMPACT_TOO_HIGH = "PRICE_IMPACT_TOO_HIGH"
DEPTH_INSUFFICIENT = "DEPTH_INSUFFICIENT"
NET_EDGE_BELOW_MIN = "NET_EDGE_BELOW_MIN"
NET_PROFIT_TARGET_NOT_MET = "NET_PROFIT_TARGET_NOT_MET"
ORDERBOOK_MISSING = "ORDERBOOK_MISSING"


@dataclass(frozen=True)
class ScalpPreflightResult:
    passed: bool
    reject_reason: str
    symbol: str
    side: str
    spread_pct: float
    buy_impact_pct: float
    sell_impact_pct: float
    expected_net_edge_pct: float
    limit_buy_price: float
    limit_sell_price: float
    depth_sufficient: bool = False
    expected_avg_fill: float = 0.0
    levels_consumed: int = 0
    reachability: dict | None = None

    def as_dict(self) -> dict:
        out = {
            "passed": self.passed,
            "reject_reason": self.reject_reason,
            "symbol": self.symbol,
            "side": self.side,
            "spread_pct": self.spread_pct,
            "buy_impact_pct": self.buy_impact_pct,
            "sell_impact_pct": self.sell_impact_pct,
            "expected_net_edge_pct": self.expected_net_edge_pct,
            "limit_buy_price": self.limit_buy_price,
            "limit_sell_price": self.limit_sell_price,
            "depth_sufficient": self.depth_sufficient,
            "expected_avg_fill": self.expected_avg_fill,
            "levels_consumed": self.levels_consumed,
        }
        if self.reachability is not None:
            out["reachability"] = self.reachability
        return out


def run_scalp_preflight(
    snap: MarketSnapshot,
    econ: ScalpEconomics,
    config: ScalpConfig,
    *,
    side: str,
    notional_usd: float | None = None,
    quantity: float | None = None,
    entry_price: float | None = None,
    entry_buy_impact_pct: float | None = None,
    check_paper_enabled: bool = True,
    momentum: MomentumDiagnostics | None = None,
    apply_entry_gate: bool = True,
) -> ScalpPreflightResult:
    notional = notional_usd or config.max_notional_paper
    spread = snap.spread_pct
    limit_buy = snap.best_ask
    limit_sell = snap.best_bid

    if not econ.is_fee_model_verified():
        return ScalpPreflightResult(
            False,
            FEE_MODEL_UNVERIFIED,
            snap.symbol,
            side.upper(),
            spread,
            0.0,
            0.0,
            0.0,
            limit_buy,
            limit_sell,
        )

    if check_paper_enabled and not config.scalp_paper_enabled:
        return ScalpPreflightResult(
            False,
            SCALP_PAPER_DISABLED,
            snap.symbol,
            side.upper(),
            spread,
            0.0,
            0.0,
            0.0,
            limit_buy,
            limit_sell,
        )

    spread_cap = econ.spread_cap_for_symbol(snap.symbol) if not config.scalp_live and (config.calibration_mode or config.scalp_paper_enabled) else econ.spread_cap_pct
    if spread > spread_cap:
        return ScalpPreflightResult(
            False,
            SPREAD_TOO_WIDE,
            snap.symbol,
            side.upper(),
            spread,
            0.0,
            0.0,
            0.0,
            limit_buy,
            limit_sell,
        )

    buy_walk = walk_buy_notional(snap.asks, notional, snap.best_ask)
    buy_impact = buy_walk.impact_pct
    sell_qty = buy_walk.filled_qty if buy_walk.filled_qty > 0 else (notional / limit_buy if limit_buy > 0 else 0.0)
    sell_walk = walk_sell_qty(snap.bids, sell_qty, snap.best_bid)
    sell_impact = sell_walk.impact_pct

    side_u = side.upper()
    if side_u == "BUY":
        if not buy_walk.depth_sufficient:
            return ScalpPreflightResult(
                False,
                DEPTH_INSUFFICIENT,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sell_impact,
                0.0,
                limit_buy,
                limit_sell,
                depth_sufficient=False,
                expected_avg_fill=buy_walk.expected_avg_fill,
                levels_consumed=buy_walk.levels_consumed,
            )
        if buy_impact > econ.impact_cap_pct:
            return ScalpPreflightResult(
                False,
                PRICE_IMPACT_TOO_HIGH,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sell_impact,
                0.0,
                limit_buy,
                limit_sell,
                depth_sufficient=True,
                expected_avg_fill=buy_walk.expected_avg_fill,
                levels_consumed=buy_walk.levels_consumed,
            )
        estimate = compute_momentum_gross_estimate(snap, momentum, econ)
        projected = estimate.projected_gross_move_pct
        costs = econ.roundtrip_cost_pct(spread, buy_impact, sell_impact)
        expected_net = projected - costs

        gate_ok, gate_reason, reach = evaluate_buy_entry_gate(
            econ,
            spread_pct=spread,
            buy_impact_pct=buy_impact,
            sell_impact_pct=sell_impact,
            estimate=estimate,
            momentum=momentum,
            apply_entry_gate=apply_entry_gate,
            selected_symbol=snap.symbol_bus,
        )
        if not gate_ok:
            return ScalpPreflightResult(
                False,
                gate_reason,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sell_impact,
                expected_net,
                limit_buy,
                limit_sell,
                depth_sufficient=True,
                expected_avg_fill=buy_walk.expected_avg_fill,
                levels_consumed=buy_walk.levels_consumed,
                reachability=reach,
            )

        if not apply_entry_gate and (expected_net <= 0 or expected_net < econ.min_net_edge_pct):
            return ScalpPreflightResult(
                False,
                NET_EDGE_BELOW_MIN,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sell_impact,
                expected_net,
                limit_buy,
                limit_sell,
                depth_sufficient=True,
                expected_avg_fill=buy_walk.expected_avg_fill,
                levels_consumed=buy_walk.levels_consumed,
                reachability=reach,
            )

        return ScalpPreflightResult(
            True,
            "",
            snap.symbol,
            side_u,
            spread,
            buy_impact,
            sell_impact,
            expected_net,
            limit_buy,
            limit_sell,
            depth_sufficient=True,
            expected_avg_fill=buy_walk.expected_avg_fill,
            levels_consumed=buy_walk.levels_consumed,
            reachability=reach,
        )

    if side_u == "SELL":
        qty = quantity or sell_qty
        if qty <= 0:
            return ScalpPreflightResult(
                False,
                DEPTH_INSUFFICIENT,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sell_impact,
                0.0,
                limit_buy,
                limit_sell,
            )
        sw = walk_sell_qty(snap.bids, qty, snap.best_bid)
        if not sw.depth_sufficient:
            return ScalpPreflightResult(
                False,
                DEPTH_INSUFFICIENT,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sw.impact_pct,
                0.0,
                limit_buy,
                limit_sell,
            )
        if sw.impact_pct > econ.impact_cap_pct:
            return ScalpPreflightResult(
                False,
                PRICE_IMPACT_TOO_HIGH,
                snap.symbol,
                side_u,
                spread,
                buy_impact,
                sw.impact_pct,
                0.0,
                limit_buy,
                limit_sell,
            )
        entry_buy_i = float(entry_buy_impact_pct) if entry_buy_impact_pct is not None else buy_impact
        sell_fill = sw.expected_avg_fill if sw.expected_avg_fill > 0 else limit_sell
        expected_net = 0.0
        if entry_price and entry_price > 0:
            expected_net = econ.executable_exit_net_pct(
                entry_price,
                sell_fill,
                entry_buy_impact_pct=entry_buy_i,
                exit_sell_impact_pct=sw.impact_pct,
            )
            if expected_net < econ.net_profit_target_pct:
                return ScalpPreflightResult(
                    False,
                    NET_PROFIT_TARGET_NOT_MET,
                    snap.symbol,
                    side_u,
                    spread,
                    entry_buy_i,
                    sw.impact_pct,
                    expected_net,
                    limit_buy,
                    sell_fill,
                    depth_sufficient=True,
                    expected_avg_fill=sell_fill,
                    levels_consumed=sw.levels_consumed,
                )
        return ScalpPreflightResult(
            True,
            "",
            snap.symbol,
            side_u,
            spread,
            entry_buy_i,
            sw.impact_pct,
            expected_net,
            limit_buy,
            sell_fill,
            depth_sufficient=True,
            expected_avg_fill=sell_fill,
            levels_consumed=sw.levels_consumed,
        )

    return ScalpPreflightResult(
        False,
        ORDERBOOK_MISSING,
        snap.symbol,
        str(side),
        spread,
        buy_impact,
        sell_impact,
        0.0,
        limit_buy,
        limit_sell,
    )


def run_scalp_preflight_default(
    snap: MarketSnapshot,
    *,
    side: str,
    notional_usd: float | None = None,
    check_paper_enabled: bool = False,
) -> ScalpPreflightResult:
    return run_scalp_preflight(
        snap,
        ScalpEconomics.from_env(),
        get_scalp_config(),
        side=side,
        notional_usd=notional_usd,
        check_paper_enabled=check_paper_enabled,
    )
