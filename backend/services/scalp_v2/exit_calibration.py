"""SCALP V2 exit calibration.

Evidence-based parameter overrides for the SCALP V2 engine.
All values derived from the true candle recovery analysis (Aug 24 - Sep 21).

Evidence summary:
  GIVEBACK_EXIT: 67% recovery rate (14/21), 24% stop rate → 2.8:1 ratio → premature
  STALL_EXIT:    45% recovery rate  (5/11), 18% stop rate → 2.5:1 ratio → premature
  STOP_LOSS_EXIT: 12% recovery rate (1/8),  88% stop rate → correct

DO NOT reuse legacy DAY_ env vars. All SCALP V2 env vars are prefixed SCALP_V2_.
"""

from __future__ import annotations

import os

SCALP_V2_ENGINE_ID = "SCALP_V2"
LEGACY_ENGINE_ID = "LEGACY_DAY_LIVE"


def scalp_v2_stall_exit_enabled() -> bool:
    """STALL EXIT: 45% recovery rate (5/11), 18% stop rate.
    2.5:1 recovery-to-stop ratio supports disabling.
    Override with SCALP_V2_STALL_EXIT_ENABLED=true to re-enable for testing."""
    raw = os.getenv("SCALP_V2_STALL_EXIT_ENABLED", "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def scalp_v2_giveback_exit_enabled() -> bool:
    """GIVEBACK EXIT: 67% recovery rate (14/21), 24% stop rate.
    2.8:1 recovery-to-stop ratio: clearly premature exit.
    Override with SCALP_V2_GIVEBACK_EXIT_ENABLED=true to re-enable."""
    raw = os.getenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", "false")
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
