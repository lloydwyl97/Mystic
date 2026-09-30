"""SCALP V2 exit calibration.

Counterfactual on the original position path (1m candles, hard stop, 0.4% net
profit, 0.22% trail, 6h horizon, 6 bps round-trip cost). Giveback and stall
were not assumed to be free money.

Untouched window 2026-09-20T23:14Z onward, n=16:
  existing ladder -0.56, giveback off -0.18, stall off -0.55,
  both off +0.12, 15m-only -0.62, structure-gated -0.69.

Validation window 2026-09-19 through that cut, n=20:
  existing ladder -2.99, both off -4.54 (worst). No policy beat the existing
  ladder on BOTH windows. n=16/20 is not a promotion.

Stall and giveback were the premature exits. They stay off unless an operator
sets SCALP_V2_STALL_EXIT_ENABLED or SCALP_V2_GIVEBACK_EXIT_ENABLED.

The live SCALP ladder is the short-horizon scalp contract: catastrophic stop,
the SCALP net-profit target (SCALP_NET_PROFIT_TARGET_PCT, 0.25%), the SCALP
adverse bound (SCALP_PATH_MAX_ADVERSE_NET_PCT, 0.15%), and the SCALP horizon
(SCALP_HOLD_MAX_MINUTES, 20 min). The earlier 0.4% target / 120-minute
negative-only time stop let scalps sit for hours on a target the entry never
certified. DAY structural invalidation is not on this ladder.
"""

from __future__ import annotations

import os

SCALP_V2_ENGINE_ID = "SCALP_V2"
LEGACY_ENGINE_ID = "LEGACY_DAY_LIVE"


SELECTED_EXIT_POLICY = "target_stop_horizon"


def scalp_v2_stall_exit_enabled() -> bool:
    """Off unless explicitly enabled. The untouched window treated stall as premature."""
    raw = os.getenv("SCALP_V2_STALL_EXIT_ENABLED", "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def scalp_v2_giveback_exit_enabled() -> bool:
    """Off unless explicitly enabled. Giveback is not an active SCALP exit."""
    raw = os.getenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def scalp_v2_min_net_profit_pct(symbol: str = "") -> float:
    """Net profit to take: the SCALP net-profit target entries are admitted against.

    SCALP_V2_MIN_NET_PROFIT_PCT overrides; otherwise SCALP_NET_PROFIT_TARGET_PCT
    (default 0.0025), the same value binance_scalp.economics uses to require
    that the expected move reaches target + costs before entry.
    """
    _ = symbol  # reserved for per-symbol tuning
    return float(os.getenv("SCALP_V2_MIN_NET_PROFIT_PCT") or os.getenv("SCALP_NET_PROFIT_TARGET_PCT") or "0.0025")


def scalp_v2_max_adverse_net_pct(symbol: str = "") -> float:
    """Net loss (positive magnitude) at which a scalp is cut: SCALP_PATH_MAX_ADVERSE_NET_PCT."""
    _ = symbol  # reserved for per-symbol tuning
    try:
        return abs(float(os.getenv("SCALP_PATH_MAX_ADVERSE_NET_PCT", "0.0015")))
    except (TypeError, ValueError):
        return 0.0015


def scalp_v2_trail_pct(symbol: str = "") -> float:
    """Trailing stop width. 0.20-0.25% was too tight (201 trailing exits at avg +$0.026).
    Default: keep existing coin profile; can widen with SCALP_V2_TRAIL_PCT.
    Returns 0.0 to signal 'use coin profile' (no override)."""
    _ = symbol  # reserved for per-symbol tuning
    raw = os.getenv("SCALP_V2_TRAIL_PCT")
    if raw:
        return float(raw)
    return 0.0  # 0 = use existing coin profile (no change)


def is_scalp_v2_engine(engine_id: str) -> bool:
    """Return True iff engine_id identifies a SCALP V2 engine."""
    return str(engine_id or "").strip() == SCALP_V2_ENGINE_ID
