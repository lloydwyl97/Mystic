"""
Economic assumptions for RF trade-worthiness labels.

Must stay aligned with `backend.config.trading_economics` (TAKER_FEE,
SLIPPAGE_BUFFER) — the same .env keys used by `portfolio_engine` and
`execute_buy_fifo` profitability gates. Do not import portfolio_engine here.

Hold / horizon timing (max hold seconds, 1m bar horizon, traction fraction) lives in
`backend.config.trade_worthiness_timing` — single contract with live day exits.
"""

from __future__ import annotations

import os

from backend.config.trading_economics import SLIPPAGE_BUFFER as SLIPPAGE_PCT
from backend.config.trading_economics import TAKER_FEE


def default_spread_pct() -> float:
    return float(os.getenv("DEFAULT_SPREAD_PCT", "0.0015"))


def profit_edge_buffer_pct() -> float:
    return float(os.getenv("PROFIT_EDGE_BUFFER_PCT", "0.001"))


def required_edge_pct_for_training() -> float:
    """Round-trip fees + slippage + spread + edge buffer (matches execute_buy_fifo G1)."""
    slippage_roundtrip = SLIPPAGE_PCT * 2
    return (2 * TAKER_FEE) + slippage_roundtrip + default_spread_pct() + profit_edge_buffer_pct()


# No-traction check: ~portfolio_engine EXIT REPAIR 2026-04-08 (traction_check_time = max_hold * 0.25)
NO_TRACTION_CHECK_FRAC = float(os.getenv("RF_LABEL_TRACTION_CHECK_FRAC", "0.25"))
NO_TRACTION_MIN_MFE_PCT = float(os.getenv("RF_LABEL_TRACTION_MIN_MFE_PCT", "0.003"))


def ambiguous_max_mfe_pct_for_training() -> float:
    """Skip ambiguous flat paths below this MFE (env RF_LABEL_AMBIGUOUS_MAX_MFE_PCT, default 0.004)."""
    return float(os.getenv("RF_LABEL_AMBIGUOUS_MAX_MFE_PCT", "0.004"))


def required_edge_pct_for_strategy(strategy_id: str) -> float:
    """Edge gate for trade-worthiness labels — **day** may use a different hurdle via env."""
    sid = (strategy_id or "").strip().lower()
    base = required_edge_pct_for_training()
    if sid == "day":
        mult = float(os.getenv("DAY_LABEL_REQUIRED_EDGE_MULT", "1.0"))
        return max(1e-6, base * mult)
    return base


def traction_params_for_strategy(strategy_id: str) -> tuple[float, float]:
    """(traction_check_frac, traction_min_mfe_pct) for binary label path."""
    sid = (strategy_id or "").strip().lower()
    if sid == "day":
        frac = float(os.getenv("DAY_RF_LABEL_TRACTION_CHECK_FRAC", str(NO_TRACTION_CHECK_FRAC)))
        mfe = float(os.getenv("DAY_RF_LABEL_TRACTION_MIN_MFE_PCT", str(NO_TRACTION_MIN_MFE_PCT)))
        return frac, mfe
    return NO_TRACTION_CHECK_FRAC, NO_TRACTION_MIN_MFE_PCT


TRADE_WORTHINESS_LABEL_VERSION = os.getenv("RF_TRADE_WORTHINESS_LABEL_VERSION", "trade_worthiness_v1")
