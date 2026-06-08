"""Per-symbol paper candidate ranking for calibration entries."""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics
from backend.services.binance_scalp.paper_spread_caps import uses_paper_spread_caps


def _spread_cap(econ: ScalpEconomics, config: ScalpConfig, symbol: str) -> float:
    if uses_paper_spread_caps(
        scalp_live=config.scalp_live,
        calibration_mode=config.calibration_mode,
        scalp_paper_enabled=config.scalp_paper_enabled,
    ):
        return econ.spread_cap_for_symbol(symbol)
    return econ.spread_cap_pct


def score_candidate(
    symbol: str,
    snap: MarketSnapshot,
    pf: Any,
    mom: MomentumDiagnostics,
    econ: ScalpEconomics,
    config: ScalpConfig,
) -> dict[str, Any]:
    reach = pf.reachability or {}
    cap = _spread_cap(econ, config, symbol)
    spread_pct = snap.spread_pct
    impact_pct = max(float(pf.buy_impact_pct), float(pf.sell_impact_pct))
    surplus = float(reach.get("projected_surplus_pct") or 0.0)
    projected = float(reach.get("projected_gross_move_pct") or 0.0)
    breakout = bool(reach.get("breakout_confirmed"))
    recent_range = float(reach.get("recent_range_pct") or 0.0)
    expected_net = float(pf.expected_net_edge_pct or 0.0)

    spread_pass = spread_pct <= cap
    impact_pass = impact_pct <= econ.impact_cap_pct
    required = float(reach.get("required_gross_move_pct") or 0.0)
    cap_util = spread_pct / cap if cap > 0 else 1.0
    reachability = projected / required if required > 0 else 0.0
    min_cushion = 0.0005
    barely_over = 0 < surplus < min_cushion

    score = 0.0
    score += min(max(surplus, 0.0) * 12000.0, 5.0)
    score += min(reachability * 2.0, 2.0)
    score += min(recent_range * 500.0, 1.5)
    score += (1.0 - min(cap_util, 1.0)) * 2.0
    if spread_pass:
        score += 0.5
    if impact_pass:
        score += 0.5
    if mom.momentum_confirmed:
        score += 1.0
    if breakout:
        score += 1.0
    score += min(max(expected_net, 0.0) * 600.0, 0.8)
    if barely_over:
        score -= 2.5
    score -= spread_pct * 400.0

    return {
        "symbol": symbol,
        "score": round(score, 4),
        "spread_pass": spread_pass,
        "spread_pct": spread_pct,
        "spread_cap_pct": cap,
        "spread_cap_util": round(cap_util, 4),
        "impact_pass": impact_pass,
        "impact_pct": impact_pct,
        "projected_surplus_pct": surplus,
        "projected_gross_pct": projected,
        "required_gross_pct": required,
        "target_reachability": round(reachability, 4),
        "barely_over_required": barely_over,
        "momentum_confirmed": mom.momentum_confirmed,
        "breakout_confirmed": breakout,
        "recent_range_pct": recent_range,
        "expected_net_edge_pct": expected_net,
    }


def rank_entry_candidates(
    rows: list[tuple[str, MarketSnapshot, Any, MomentumDiagnostics]],
    econ: ScalpEconomics,
    config: ScalpConfig,
) -> tuple[list[tuple[str, MarketSnapshot, Any]], dict[str, Any]]:
    """Return ranked (sym, snap, pf) list and selection metadata for the top pick."""
    scored = [
        (
            score_candidate(sym, snap, pf, mom, econ, config),
            sym,
            snap,
            pf,
        )
        for sym, snap, pf, mom in rows
    ]
    scored.sort(key=lambda row: (-row[0]["score"], row[0]["spread_pct"]))
    ranked = [(sym, snap, pf) for _, sym, snap, pf in scored]
    top = scored[0][0] if scored else {}
    reason = (
        f"ranked {top.get('symbol')}: score={top.get('score')} "
        f"surplus={top.get('projected_surplus_pct', 0)*100:.4f}% reach={top.get('target_reachability')} "
        f"spread={top.get('spread_pct', 0)*100:.4f}% cap_util={top.get('spread_cap_util')} "
        f"range={top.get('recent_range_pct', 0)*100:.4f}% "
        f"momentum={top.get('momentum_confirmed')} breakout={top.get('breakout_confirmed')} "
        f"barely_over={top.get('barely_over_required')}"
        if top
        else "no candidates"
    )
    meta = {
        "selection_reason": reason,
        "ranking": [row[0] for row in scored],
        "selected_symbol": top.get("symbol"),
    }
    return ranked, meta
