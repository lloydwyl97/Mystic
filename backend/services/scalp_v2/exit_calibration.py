"""SCALP V2 exit calibration.

Counterfactual on the original position path (1m candles, hard stop, 0.4% net
profit, 0.22% trail, 6h horizon, 6 bps round-trip cost). Giveback and stall
were not assumed to be free money.

Untouched window 2026-09-20T23:14Z onward, n=16:
  existing ladder -0.56, giveback off -0.18, stall off -0.55,
  both off +0.12, 15m-only -0.62, structure-gated -0.69.

Validation window 2026-09-19 through that cut, n=20:
  existing ladder -2.99, both off -4.54 (worst). No policy beat the existing
  ladder on both validation and the untouched window. n=16 is not a reliable
  replacement. Selected policy: keep giveback, stall, and the hard stop.
"""

from __future__ import annotations

import os

SCALP_V2_ENGINE_ID = "SCALP_V2"
LEGACY_ENGINE_ID = "LEGACY_DAY_LIVE"


SELECTED_EXIT_POLICY = "preserve_giveback_stall_and_hard_stop"


def scalp_v2_stall_exit_enabled() -> bool:
    """Keep stall. The untouched window liked disabling it; validation did not."""
    raw = os.getenv("SCALP_V2_STALL_EXIT_ENABLED", "true")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def scalp_v2_giveback_exit_enabled() -> bool:
    """Keep giveback. Disabling it was not reliable out of sample."""
    raw = os.getenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", "true")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def scalp_v2_min_net_profit_pct(symbol: str = "") -> float:
    """Minimum net profit to take. Must clear ESTIMATED_ROUNDTRIP_COST.
    Default: 0.004 (0.4%) — same as legacy. Can tune per symbol."""
    _ = symbol  # reserved for per-symbol tuning
    return float(os.getenv("SCALP_V2_MIN_NET_PROFIT_PCT", "0.004"))


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
