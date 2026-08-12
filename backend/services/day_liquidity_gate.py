"""DAY liquidity / spread quality gate.

The existing cost telemetry (`_hydrate_cost_telemetry`) subtracts spread and
slippage from expected value, which naturally lowers a wide-spread
candidate's rank. This module adds two things on top:

1. A dedicated soft rank/size demotion for elevated (but not catastrophic)
   spreads and lopsided order-book depth. Since bandit + EV weighting are
   already indirect, this creates an explicit signal on the persisted
   explainability so post-trade audits can attribute PnL to spread
   quality.
2. A catastrophic hard block for spreads so wide that even the setup's
   thesis target cannot recover them. Per the audit rules, spread /
   liquidity below execution profitability *is* an allowed hard stop —
   this module supplies that gate with symbol-aware defaults.

Feature flag: DAY_LIQUIDITY_GATE_ENABLED (default true).
Hard-block sub-flag: DAY_LIQUIDITY_HARD_BLOCK_ENABLED (default true).
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_TYPICAL_SPREAD_BPS: dict[str, float] = {
    "BTC/USDT": 3.0,
    "ETH/USDT": 3.0,
    "SOL/USDT": 5.0,
    "XRP/USDT": 5.0,
}
FALLBACK_TYPICAL_SPREAD_BPS = 6.0

SIZE_FACTOR_AT_ZERO = 0.35
SIZE_FACTOR_AT_HALF = 0.75
RANK_DELTA_AT_ZERO = -0.12
RANK_DELTA_AT_ONE = 0.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "spread_vs_typical": 0.70,
    "depth_imbalance": 0.30,
}


def liquidity_gate_enabled() -> bool:
    return os.getenv("DAY_LIQUIDITY_GATE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def liquidity_hard_block_enabled() -> bool:
    return os.getenv("DAY_LIQUIDITY_HARD_BLOCK_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _normalize_symbol(sym: str) -> str:
    s = str(sym or "").strip().upper()
    if "/" not in s and s.endswith("USDT") and len(s) > 4:
        s = f"{s[:-4]}/USDT"
    return s


def _typical_spread_bps(symbol: str) -> float:
    s = _normalize_symbol(symbol)
    env_key = f"DAY_LIQUIDITY_TYPICAL_SPREAD_BPS_{s.replace('/', '_')}"
    env_v = os.getenv(env_key)
    if env_v:
        try:
            return float(env_v)
        except (TypeError, ValueError):
            pass
    return DEFAULT_TYPICAL_SPREAD_BPS.get(s, FALLBACK_TYPICAL_SPREAD_BPS)


def _catastrophic_spread_bps(symbol: str) -> float:
    """Absolute spread beyond which the trade is hard-blocked.

    Defaults: 4x symbol typical spread, floor at 30 bps.
    Override via DAY_LIQUIDITY_CATASTROPHIC_SPREAD_BPS_{SYMBOL}.
    """
    s = _normalize_symbol(symbol)
    env_key = f"DAY_LIQUIDITY_CATASTROPHIC_SPREAD_BPS_{s.replace('/', '_')}"
    env_v = os.getenv(env_key)
    if env_v:
        try:
            return float(env_v)
        except (TypeError, ValueError):
            pass
    typical = _typical_spread_bps(s)
    return max(30.0, typical * 4.0)


def _spread_bps(decision_data: dict[str, Any]) -> float:
    """Read spread from decision_data (already normalized by _hydrate_cost_telemetry).

    Accepts either fractional (0.0004) or bps (4.0). If value > 1.0 assume bps.
    Otherwise assume fractional and convert.
    """
    for key in ("spread_pct", "spread_cost_pct", "signal_spread_pct", "entry_spread_pct"):
        raw = decision_data.get(key)
        if raw in (None, ""):
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        if v > 1.0:
            return v  # already bps
        return v * 10000.0
    return 0.0


def _depth_imbalance(decision_data: dict[str, Any]) -> float | None:
    """Order-book depth imbalance if known. None means unknown → neutral credit.

    Accepts either [0, 1] (0.5 = balanced) or [-1, 1] (0 = balanced). Any
    other formats result in None (neutral).
    """
    for key in ("signal_ctx_depth_imbalance", "depth_imbalance", "book_imbalance"):
        raw = decision_data.get(key)
        if raw in (None, ""):
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if -1.5 <= v <= 1.5:
            return v
    return None


def _spread_credit(spread_bps: float, typical_bps: float) -> tuple[float, str]:
    if spread_bps <= 0.0:
        return 0.6, "spread_unknown"
    ratio = spread_bps / max(0.5, typical_bps)
    if ratio <= 1.15:
        return 1.0, "spread_tight"
    if ratio <= 1.6:
        return 0.85, "spread_normal"
    if ratio <= 2.4:
        return 0.6, "spread_elevated"
    if ratio <= 3.5:
        return 0.35, "spread_wide"
    if ratio <= 5.0:
        return 0.15, "spread_very_wide"
    return 0.0, "spread_catastrophic"


def _depth_credit(depth_imbalance: float | None) -> tuple[float, str]:
    if depth_imbalance is None:
        return 0.6, "depth_unknown"
    # Detect format: if the value is in [0, 1], center is 0.5.
    if 0.0 <= depth_imbalance <= 1.0:
        deviation = abs(depth_imbalance - 0.5)
    else:
        # Assume [-1, 1], center 0.0.
        deviation = abs(depth_imbalance) / 2.0
    if deviation <= 0.10:
        return 1.0, "depth_balanced"
    if deviation <= 0.20:
        return 0.8, "depth_soft_skew"
    if deviation <= 0.30:
        return 0.55, "depth_skewed"
    if deviation <= 0.40:
        return 0.3, "depth_heavy_skew"
    return 0.1, "depth_extreme"


def _score_to_rank_delta(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    return RANK_DELTA_AT_ZERO + s * (RANK_DELTA_AT_ONE - RANK_DELTA_AT_ZERO)


def _score_to_size_factor(score: float) -> float:
    s = max(0.0, min(1.0, float(score)))
    if s <= 0.5:
        t = s / 0.5
        return SIZE_FACTOR_AT_ZERO + t * (SIZE_FACTOR_AT_HALF - SIZE_FACTOR_AT_ZERO)
    t = (s - 0.5) / 0.5
    return SIZE_FACTOR_AT_HALF + t * (1.0 - SIZE_FACTOR_AT_HALF)


def compute_liquidity_quality(
    decision_data: dict[str, Any],
    symbol: str,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute per-candidate liquidity quality result."""
    if not liquidity_gate_enabled():
        return {
            "liquidity_gate_enabled": False,
            "liquidity_quality_score": 1.0,
            "liquidity_quality_state": "disabled",
            "liquidity_quality_reasons": "",
            "liquidity_quality_rank_delta": 0.0,
            "liquidity_quality_size_factor": 1.0,
            "liquidity_hard_blocked": False,
            "liquidity_hard_block_reason": "",
            "liquidity_spread_bps": 0.0,
            "liquidity_typical_spread_bps": 0.0,
            "liquidity_components": {},
        }

    dd = dict(decision_data or {})
    w = dict(weights or DEFAULT_WEIGHTS)
    spread_bps = _spread_bps(dd)
    typical_bps = _typical_spread_bps(symbol)
    depth = _depth_imbalance(dd)

    sp_c, sp_r = _spread_credit(spread_bps, typical_bps)
    dp_c, dp_r = _depth_credit(depth)

    components = {
        "spread_vs_typical": {"credit": sp_c, "reason": sp_r, "weight": w["spread_vs_typical"]},
        "depth_imbalance": {"credit": dp_c, "reason": dp_r, "weight": w["depth_imbalance"]},
    }
    total_weight = sum(float(v["weight"]) for v in components.values()) or 1.0
    weighted_sum = sum(float(v["credit"]) * float(v["weight"]) for v in components.values())
    score = max(0.0, min(1.0, weighted_sum / total_weight))

    catastrophic = _catastrophic_spread_bps(symbol)
    hard_blocked = False
    hard_block_reason = ""
    if liquidity_hard_block_enabled() and spread_bps >= catastrophic:
        hard_blocked = True
        hard_block_reason = f"SPREAD_CATASTROPHIC_{spread_bps:.1f}bps>={catastrophic:.1f}bps"

    if score >= 0.75:
        state = "liquidity_good"
    elif score >= 0.55:
        state = "liquidity_ok"
    elif score >= 0.35:
        state = "liquidity_poor"
    else:
        state = "liquidity_bad"
    if hard_blocked:
        state = "liquidity_hard_blocked"

    reasons_joined = ",".join(str(v["reason"]) for v in components.values())

    return {
        "liquidity_gate_enabled": True,
        "liquidity_quality_score": round(score, 5),
        "liquidity_quality_state": state,
        "liquidity_quality_reasons": reasons_joined,
        "liquidity_quality_rank_delta": round(_score_to_rank_delta(score), 5),
        "liquidity_quality_size_factor": round(_score_to_size_factor(score), 5),
        "liquidity_hard_blocked": hard_blocked,
        "liquidity_hard_block_reason": hard_block_reason,
        "liquidity_spread_bps": round(spread_bps, 3),
        "liquidity_typical_spread_bps": round(typical_bps, 3),
        "liquidity_components": components,
    }


def apply_liquidity_gate_to_decision_data(
    decision_data: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    """Stamp liquidity fields onto decision_data and compound into
    thesis_size_factor / thesis_rank_delta. Sets hard_block only when
    spread is catastrophic (allowed hard stop per audit rules).
    """
    result = compute_liquidity_quality(decision_data, symbol)
    dd = dict(decision_data or {})
    for k, v in result.items():
        dd[k] = v
    if result.get("liquidity_gate_enabled"):
        try:
            prev_size = float(dd.get("thesis_size_factor") or 1.0)
        except (TypeError, ValueError):
            prev_size = 1.0
        liq_size = float(result["liquidity_quality_size_factor"])
        dd["thesis_size_factor"] = round(max(SIZE_FACTOR_AT_ZERO, prev_size * liq_size), 5)
        try:
            prev_rank_delta = float(dd.get("thesis_rank_delta") or 0.0)
        except (TypeError, ValueError):
            prev_rank_delta = 0.0
        dd["thesis_rank_delta"] = round(prev_rank_delta + float(result["liquidity_quality_rank_delta"]), 5)

    if result.get("liquidity_hard_blocked"):
        dd["hard_block"] = True
        dd["candidate_eligible"] = False
        # Preserve any pre-existing block reason but prepend this one for clarity.
        existing_reason = str(dd.get("hard_block_reason") or "")
        dd["hard_block_reason"] = str(result["liquidity_hard_block_reason"]) + (f"|{existing_reason}" if existing_reason else "")
    else:
        dd["hard_block"] = bool(dd.get("hard_block") or False)
        dd["candidate_eligible"] = bool(dd.get("candidate_eligible", True) and not dd.get("hard_block"))

    return dd


__all__ = [
    "apply_liquidity_gate_to_decision_data",
    "compute_liquidity_quality",
    "liquidity_gate_enabled",
    "liquidity_hard_block_enabled",
]
