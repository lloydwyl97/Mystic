"""Named DAY execution-cost fields. Never hide a quality floor inside fees.

Fill evidence (authenticated Binance.US myTrades after 2026-04-21, Ocean
fill_fee_audit, decision_book_tape) is the authority. These numbers are
observability + replay inputs. Live BUY/SELL gates keep reading the existing
production constants until a production-faithful replay authorizes a swap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

# --- Exchange commission (per side, fraction of notional) -------------------
MAKER_COMMISSION_PCT: Final[float] = 0.0
TAKER_COMMISSION_PCT: Final[float] = 0.0002
BNB_TAKER_DISCOUNT: Final[float] = 0.05
# Account currently has no BNB. Do not apply the discount to live estimates.
BNB_FEE_PAYMENT_ACTIVE: Final[bool] = False

# --- Measured symbol-specific full spread (decision_book_tape, n≈980k) ------
RECORDED_FULL_SPREAD_PCT: Final[dict[str, float]] = {
    "BTCUSDT": 0.00005899,
    "ETHUSDT": 0.00007237,
    "SOLUSDT": 0.00021857,
    "XRPUSDT": 0.00020003,
}
FALLBACK_FULL_SPREAD_PCT: Final[float] = 0.00020

# --- Measured one-way exit slippage (60 live Ocean SELL audits) --------------
MEASURED_SLIPPAGE_ONE_WAY_PCT: Final[float] = 0.0000723
# Conservative floor so spread/slippage are never treated as zero.
MIN_SLIPPAGE_ONE_WAY_PCT: Final[float] = 0.00002
MIN_SPREAD_PCT: Final[float] = 0.00001

# Production BUY-veto still subtracts these (do not use as "fees").
LEGACY_BUY_VETO_FEE_PCT: Final[float] = 0.0010
LEGACY_BUY_VETO_SLIP_PCT: Final[float] = 0.0008
LEGACY_BUY_VETO_SPREAD_PCT: Final[float] = 0.0004
LEGACY_BUY_VETO_TOTAL_PCT: Final[float] = LEGACY_BUY_VETO_FEE_PCT + LEGACY_BUY_VETO_SLIP_PCT + LEGACY_BUY_VETO_SPREAD_PCT
LEGACY_SELL_ROUNDTRIP_PCT: Final[float] = 0.0006

# Default labeled expected-move fields stamped when the signal omits them.
DEFAULT_EFE_PCT: Final[float] = 0.012
DEFAULT_EAE_PCT: Final[float] = 0.007

# Arm B named quality floor: same hurdle the 22-bp veto currently implies
# when p_sell=0 (p_buy * DEFAULT_EFE > 22 bp). Stored as min predicted GROSS.
ARM_B_MIN_PREDICTED_GROSS_PCT: Final[float] = LEGACY_BUY_VETO_TOTAL_PCT


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s + "T"
    return s


def expected_exchange_commission_rt_pct(*, taker_entry: bool = True, taker_exit: bool = True) -> float:
    entry = TAKER_COMMISSION_PCT if taker_entry else MAKER_COMMISSION_PCT
    exit_ = TAKER_COMMISSION_PCT if taker_exit else MAKER_COMMISSION_PCT
    rt = entry + exit_
    if BNB_FEE_PAYMENT_ACTIVE:
        rt *= 1.0 - BNB_TAKER_DISCOUNT
    return float(rt)


def expected_spread_pct(symbol: str) -> float:
    return max(MIN_SPREAD_PCT, float(RECORDED_FULL_SPREAD_PCT.get(_api(symbol), FALLBACK_FULL_SPREAD_PCT)))


def expected_slippage_rt_pct() -> float:
    return max(MIN_SLIPPAGE_ONE_WAY_PCT * 2.0, MEASURED_SLIPPAGE_ONE_WAY_PCT * 2.0)


def honest_all_in_rt_pct(symbol: str) -> float:
    return expected_exchange_commission_rt_pct() + expected_spread_pct(symbol) + expected_slippage_rt_pct()


def bnb_savings_usd(*, notional_usd: float, trades: int = 1) -> float:
    """USD saved by paying taker fees in BNB (5% of taker/taker commission only)."""
    commission = float(notional_usd) * expected_exchange_commission_rt_pct() * int(trades)
    return commission * BNB_TAKER_DISCOUNT


@dataclass(frozen=True)
class NamedCostBreakdown:
    expected_exchange_commission: float
    expected_spread: float
    expected_slippage: float
    predicted_gross_trade_value: float
    predicted_net_trade_value: float
    prediction_calibration: str
    min_executable_net_ev: float
    symbol: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_exchange_commission": self.expected_exchange_commission,
            "expected_spread": self.expected_spread,
            "expected_slippage": self.expected_slippage,
            "predicted_gross_trade_value": self.predicted_gross_trade_value,
            "predicted_net_trade_value": self.predicted_net_trade_value,
            "prediction_calibration": self.prediction_calibration,
            "min_executable_net_ev": self.min_executable_net_ev,
            "symbol": self.symbol,
        }


def named_cost_breakdown(
    symbol: str,
    *,
    p_buy: float,
    p_sell: float = 0.0,
    efe: float = DEFAULT_EFE_PCT,
    eae: float = DEFAULT_EAE_PCT,
    calibration: str = "uncalibrated_p_buy_x_efe",
    min_executable_net_ev: float = 0.0,
    predicted_gross: float | None = None,
) -> NamedCostBreakdown:
    comm = expected_exchange_commission_rt_pct()
    spread = expected_spread_pct(symbol)
    slip = expected_slippage_rt_pct()
    if predicted_gross is None:
        predicted_gross = float(p_buy) * float(efe) - float(p_sell) * abs(float(eae))
    predicted_net = float(predicted_gross) - comm - spread - slip
    return NamedCostBreakdown(
        expected_exchange_commission=comm,
        expected_spread=spread,
        expected_slippage=slip,
        predicted_gross_trade_value=float(predicted_gross),
        predicted_net_trade_value=float(predicted_net),
        prediction_calibration=calibration,
        min_executable_net_ev=float(min_executable_net_ev),
        symbol=_api(symbol),
    )


def stamp_named_costs(decision_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Attach named fields. Does not overwrite estimated_fees_pct / veto inputs."""
    dd = decision_data if isinstance(decision_data, dict) else {}
    try:
        p_buy = float(dd.get("prob_buy") or dd.get("winner_probability") or 0.0)
    except (TypeError, ValueError):
        p_buy = 0.0
    try:
        p_sell = float(dd.get("prob_sell") or 0.0)
    except (TypeError, ValueError):
        p_sell = 0.0
    try:
        efe = float(dd.get("estimated_win_pct") or dd.get("expected_favorable_excursion") or DEFAULT_EFE_PCT)
    except (TypeError, ValueError):
        efe = DEFAULT_EFE_PCT
    try:
        eae = float(dd.get("estimated_loss_pct") or dd.get("expected_adverse_excursion") or DEFAULT_EAE_PCT)
    except (TypeError, ValueError):
        eae = DEFAULT_EAE_PCT
    br = named_cost_breakdown(symbol, p_buy=p_buy, p_sell=p_sell, efe=efe, eae=eae)
    dd["expected_exchange_commission"] = br.expected_exchange_commission
    dd["expected_spread"] = br.expected_spread
    dd["expected_slippage"] = br.expected_slippage
    dd["predicted_gross_trade_value"] = br.predicted_gross_trade_value
    dd["predicted_net_trade_value"] = br.predicted_net_trade_value
    dd["prediction_calibration"] = br.prediction_calibration
    dd["min_executable_net_ev"] = br.min_executable_net_ev
    return dd


def shadow_sizing_enabled() -> bool:
    return os.getenv("DAY_SPREAD_SHADOW_MODE", "true").strip().lower() in ("1", "true", "yes", "on")
