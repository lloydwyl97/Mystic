"""
Single source of truth for Mystic trading economics.

Paper and live MUST read the same numbers from this module. There is no
paper-only profit threshold and no live-only profit threshold. The values
exposed here are the inputs to the one DAY trading brain (portfolio_engine)
and any execution adapter (paper or live) that wraps it.

Categories:
  * fee/cost model (TAKER_FEE, SLIPPAGE_BUFFER, ESTIMATED_ROUNDTRIP_COST)
  * sell threshold (MIN_NET_PROFIT_TO_SELL, MIN_PROFIT_AFTER_COSTS_USD)
  * cooldowns    (COOLDOWN_SECONDS_AFTER_SELL,
                  COOLDOWN_SECONDS_AFTER_HUMAN_SELL)

All values are env-overridable but the defaults are the canonical Mystic
DAY-only top-4 profile (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Final

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "TRADING_ECONOMICS env var %s=%r is not a float; using default %s",
            name,
            raw,
            default,
        )
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "TRADING_ECONOMICS env var %s=%r is not an int; using default %s",
            name,
            raw,
            default,
        )
        return int(default)


# -- Fee model (Binance.US Advanced Spot) ------------------------------------
# Apr 2026 universal Advanced Spot: 0% maker, 0.02% taker (all pairs incl. top-four USDT).
# Legacy Tier-0 subset: 0% maker, 0.01% taker — NOT used for Mystic top-four USDT pairs.
# See backend.config.binance_us_fee_schedule for verification sources.
BINANCE_US_MAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_MAKER_FEE_PCT", 0.0)
BINANCE_US_TAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_TAKER_FEE_PCT", 0.0002)
BINANCE_US_TIER0_TAKER_FEE_PCT: Final[float] = _env_float("BINANCE_US_TIER0_TAKER_FEE_PCT", 0.0001)
EXCHANGE_NAME: Final[str] = "Binance.US"
FEE_SCHEDULE_SOURCE_DATE: Final[str] = os.getenv("FEE_SCHEDULE_SOURCE_DATE", "2026-04-21")

MAKER_FEE: Final[float] = _env_float("MAKER_FEE", BINANCE_US_MAKER_FEE_PCT)
TAKER_FEE: Final[float] = _env_float("TAKER_FEE", BINANCE_US_TAKER_FEE_PCT)
# One-sided slippage buffer beyond order-book (Advanced Trading has no platform spread).
SLIPPAGE_BUFFER: Final[float] = _env_float("SLIPPAGE_BUFFER", 0.0001)
# Default order-book half-spread estimate (live measured via bookTicker; override via env).
ORDERBOOK_HALF_SPREAD_ESTIMATE: Final[float] = _env_float("ORDERBOOK_HALF_SPREAD_ESTIMATE", 0.00006)


def canonical_roundtrip_cost_pct(
    *,
    spread_pct: float | None = None,
    buy_impact_pct: float = 0.0,
    sell_impact_pct: float = 0.0,
) -> float:
    """One executable round-trip cost for every edge consumer.

    Price convention: entry is the ask, later marks are trade/mid. That already
    embeds the entry half-spread, so this adds the exit half exactly once —
    half the measured full spread when the book is known, otherwise
    ``ORDERBOOK_HALF_SPREAD_ESTIMATE``. Taker fee and the slippage buffer are
    both sides. Impact is added only when a book walk supplies it. The flat
    estimate and the live spread are never both included.
    """
    fee = 2.0 * TAKER_FEE
    slip = 2.0 * SLIPPAGE_BUFFER
    half = ORDERBOOK_HALF_SPREAD_ESTIMATE if spread_pct is None else max(0.0, float(spread_pct)) / 2.0
    impact = max(0.0, float(buy_impact_pct or 0.0)) + max(0.0, float(sell_impact_pct or 0.0))
    return fee + slip + half + impact


# Flat canonical cost (no live book): 4.0 bps taker + 2.0 bps slippage + 0.6 bps
# exit half-spread = 6.6 bps. A raw ESTIMATED_ROUNDTRIP_COST env value is not
# an alternate formula — if it disagrees, it is ignored so consumers cannot split.
_CANONICAL_FLAT_ROUNDTRIP: Final[float] = canonical_roundtrip_cost_pct()
_env_roundtrip = os.getenv("ESTIMATED_ROUNDTRIP_COST", "")
if _env_roundtrip:
    try:
        _env_roundtrip_f = float(_env_roundtrip)
    except (TypeError, ValueError):
        _env_roundtrip_f = _CANONICAL_FLAT_ROUNDTRIP
    if abs(_env_roundtrip_f - _CANONICAL_FLAT_ROUNDTRIP) > 1e-12:
        logger.warning(
            "ESTIMATED_ROUNDTRIP_COST env %.6f ignored; canonical flat round-trip is %.6f (2*taker + 2*slippage + exit half-spread). Divergent env values omit or double-count spread.",
            _env_roundtrip_f,
            _CANONICAL_FLAT_ROUNDTRIP,
        )
ESTIMATED_ROUNDTRIP_COST: Final[float] = _CANONICAL_FLAT_ROUNDTRIP
ESTIMATED_ROUNDTRIP_COST_PCT: Final[float] = _CANONICAL_FLAT_ROUNDTRIP

# -- Sell thresholds ---------------------------------------------------------
# Real net profit floor (fraction of cost basis) required to take profit.
MIN_NET_PROFIT_TO_SELL: Final[float] = _env_float("MIN_NET_PROFIT_TO_SELL", 0.004)
# Optional absolute floor in USD (must clear both the percent floor and this).
# 0.0 disables; default 0.0 keeps backward compatibility.
MIN_PROFIT_AFTER_COSTS_USD: Final[float] = _env_float("MIN_PROFIT_AFTER_COSTS_USD", 0.0)

# Per-symbol overrides for MIN_NET_PROFIT_TO_SELL. Different coins have
# different achievable MFE distributions in the same market conditions —
# a global 0.4% target may starve wins on low-vol coins while leaving PnL
# on the table for higher-vol coins. Empty env → falls back to the global
# MIN_NET_PROFIT_TO_SELL value above. Format: floats in [0.0005, 0.05].
_PER_COIN_MIN_NET_PROFIT: Final[dict[str, float]] = {
    "BTCUSDT": _env_float("MIN_NET_PROFIT_TO_SELL_BTC", MIN_NET_PROFIT_TO_SELL),
    "ETHUSDT": _env_float("MIN_NET_PROFIT_TO_SELL_ETH", MIN_NET_PROFIT_TO_SELL),
    "SOLUSDT": _env_float("MIN_NET_PROFIT_TO_SELL_SOL", MIN_NET_PROFIT_TO_SELL),
    "XRPUSDT": _env_float("MIN_NET_PROFIT_TO_SELL_XRP", MIN_NET_PROFIT_TO_SELL),
}


def min_net_profit_for_symbol(symbol: str) -> float:
    """Return the per-symbol MIN_NET_PROFIT_TO_SELL, falling back to global.

    Symbol may arrive as "BTC/USDT", "BTCUSDT", or "BTCUSD" — normalized here.
    Callers must use this helper (not the global constant directly) whenever
    they know the trading symbol; otherwise they get the top-level default.
    """
    if not symbol:
        return MIN_NET_PROFIT_TO_SELL
    s = str(symbol).strip().upper().replace("/", "").replace("-", "")
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s + "T"
    return float(_PER_COIN_MIN_NET_PROFIT.get(s, MIN_NET_PROFIT_TO_SELL))


# -- Cooldowns ---------------------------------------------------------------
# Block re-entry on a symbol for this many seconds after Mystic closed it.
COOLDOWN_SECONDS_AFTER_SELL: Final[int] = _env_int(
    "COOLDOWN_SECONDS_AFTER_SELL",
    _env_int("POST_SELL_COOLDOWN_WALL_SEC", 2400),
)
# Block re-entry on a symbol after a HUMAN_MANUAL_SELL detected on the
# exchange. Same cadence as AI-triggered close by default.
COOLDOWN_SECONDS_AFTER_HUMAN_SELL: Final[int] = _env_int(
    "COOLDOWN_SECONDS_AFTER_HUMAN_SELL",
    COOLDOWN_SECONDS_AFTER_SELL,
)

# -- DAY sizing (replay-aligned per-slot notional) -----------------------------
# Baseline replay uses $2,500/slot; 1.5x candidate → $3,750/slot, $15k max (4 slots).
DAY_BASE_NOTIONAL_PER_SLOT_USD: Final[float] = _env_float("DAY_BASE_NOTIONAL_PER_SLOT_USD", 2500.0)
DAY_NOTIONAL_MULT: Final[float] = _env_float("DAY_NOTIONAL_MULT", 1.0)
DAY_TARGET_NOTIONAL_PER_SLOT_USD: Final[float] = _env_float(
    "DAY_TARGET_NOTIONAL_PER_SLOT_USD",
    DAY_BASE_NOTIONAL_PER_SLOT_USD * DAY_NOTIONAL_MULT,
)
DAY_MAX_DEPLOYED_USD: Final[float] = _env_float(
    "DAY_MAX_DEPLOYED_USD",
    DAY_TARGET_NOTIONAL_PER_SLOT_USD * 4.0,
)
DAY_MAX_OPEN_SLOTS: Final[int] = _env_int("DAY_MAX_OPEN_SLOTS", 4)


@dataclass(frozen=True)
class TradingEconomicsSnapshot:
    exchange: str
    maker_fee: float
    taker_fee: float
    slippage_buffer: float
    orderbook_half_spread_estimate: float
    estimated_roundtrip_cost: float
    fee_schedule_source_date: str
    min_net_profit_to_sell: float
    min_profit_after_costs_usd: float
    cooldown_seconds_after_sell: int
    cooldown_seconds_after_human_sell: int
    day_notional_mult: float
    day_target_notional_per_slot_usd: float
    day_max_deployed_usd: float


def get_trading_economics() -> TradingEconomicsSnapshot:
    """Return the canonical economics snapshot shared by paper and live."""
    return TradingEconomicsSnapshot(
        exchange=EXCHANGE_NAME,
        maker_fee=MAKER_FEE,
        taker_fee=TAKER_FEE,
        slippage_buffer=SLIPPAGE_BUFFER,
        orderbook_half_spread_estimate=ORDERBOOK_HALF_SPREAD_ESTIMATE,
        estimated_roundtrip_cost=ESTIMATED_ROUNDTRIP_COST,
        fee_schedule_source_date=FEE_SCHEDULE_SOURCE_DATE,
        min_net_profit_to_sell=MIN_NET_PROFIT_TO_SELL,
        min_profit_after_costs_usd=MIN_PROFIT_AFTER_COSTS_USD,
        cooldown_seconds_after_sell=COOLDOWN_SECONDS_AFTER_SELL,
        cooldown_seconds_after_human_sell=COOLDOWN_SECONDS_AFTER_HUMAN_SELL,
        day_notional_mult=DAY_NOTIONAL_MULT,
        day_target_notional_per_slot_usd=DAY_TARGET_NOTIONAL_PER_SLOT_USD,
        day_max_deployed_usd=DAY_MAX_DEPLOYED_USD,
    )


def _fee_fraction_to_bps(fee: float) -> float:
    """0.02% is 2 bps. Normalize the known 10x / percent-point mis-scales."""
    value = float(fee or 0.0)
    if abs(value - 0.02) < 1e-12 or abs(value - 0.002) < 1e-12:
        value = 0.0002
    return round(value * 10000.0, 2)


def get_trading_economics_display() -> dict[str, Any]:
    """Dashboard/API display payload for fee model."""
    snap = get_trading_economics()
    return {
        "exchange": snap.exchange,
        "maker_fee_pct": snap.maker_fee,
        "taker_fee_pct": snap.taker_fee,
        "maker_fee_bps": _fee_fraction_to_bps(snap.maker_fee),
        "taker_fee_bps": _fee_fraction_to_bps(snap.taker_fee),
        "slippage_buffer_pct": snap.slippage_buffer,
        "orderbook_half_spread_estimate_pct": snap.orderbook_half_spread_estimate,
        "orderbook_full_spread_estimate_pct": snap.orderbook_half_spread_estimate * 2,
        "roundtrip_estimated_cost_pct": snap.estimated_roundtrip_cost,
        "roundtrip_estimated_cost_bps": round(snap.estimated_roundtrip_cost * 10000, 2),
        "fee_schedule_source_date": snap.fee_schedule_source_date,
        "fee_schedule_note": ("Binance.US Advanced Spot: 0% maker / 0.02% taker universal (Apr 2026). No platform spread; order-book spread + slippage buffer only."),
        "min_net_profit_to_sell_pct": snap.min_net_profit_to_sell,
        "day_notional_mult": snap.day_notional_mult,
        "day_base_notional_per_slot_usd": DAY_BASE_NOTIONAL_PER_SLOT_USD,
        "day_target_notional_per_slot_usd": snap.day_target_notional_per_slot_usd,
        "day_max_deployed_usd": snap.day_max_deployed_usd,
        "day_max_open_slots": DAY_MAX_OPEN_SLOTS,
        "baseline_lock_id": os.getenv("DAY_BASELINE_LOCK_ID", "day_baseline_all_pass_v1_size_1_5"),
    }


def is_net_profit_acceptable(
    net_profit_pct: float,
    net_profit_usd: float,
) -> bool:
    """
    Centralized "should AI sell now?" check. Both PAPER and LIVE must call
    this same function to evaluate a candidate sell. Returns True only when
    the net profit clears both the percent floor and the (optional) USD floor.
    """
    if net_profit_pct < MIN_NET_PROFIT_TO_SELL:
        return False
    return not (MIN_PROFIT_AFTER_COSTS_USD > 0.0 and net_profit_usd < MIN_PROFIT_AFTER_COSTS_USD)


def fetch_account_taker_fee(timeout: float = 10.0) -> float | None:
    """Actual account taker commission (as a fraction) from Binance.US, or None.

    This is the authoritative rate the account is charged, independent of any
    fill, read from GET /api/v3/account (``commissionRates.taker``, falling back
    to the integer ``takerCommission`` in 0.0001 units). Best effort: returns
    None on missing credentials, a non-whitelisted IP, or any network error.
    Never raises and never logs secrets.
    """
    import hashlib
    import hmac
    import json as _json
    import time as _time
    import urllib.parse
    import urllib.request

    key = os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCE_API_KEY") or ""
    secret = os.getenv("BINANCE_US_SECRET_KEY") or os.getenv("BINANCE_SECRET") or ""
    if not key or not secret:
        return None
    try:
        params = {"timestamp": int(_time.time() * 1000), "recvWindow": 10000}
        query = urllib.parse.urlencode(params)
        query += "&signature=" + hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"https://api.binance.us/api/v3/account?{query}"
        req = urllib.request.Request(url, headers={"X-MBX-APIKEY": key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.load(resp)
    except Exception:
        return None
    rates = data.get("commissionRates") or {}
    try:
        if rates.get("taker") is not None:
            return float(rates["taker"])
        taker_commission = data.get("takerCommission")
        if taker_commission is not None:
            return float(taker_commission) / 10000.0
    except (TypeError, ValueError):
        return None
    return None


def reconcile_taker_fee(tolerance: float = 0.00005) -> dict[str, Any]:
    """Compare the exchange's actual account taker fee to configured TAKER_FEE.

    Logs a WARNING when they diverge beyond ``tolerance`` (default 0.5 bps) so
    an operator knows the net-edge gate and round-trip cost are calibrated to
    the wrong taker fee. Read-only: never changes config, never blocks trading.
    """
    actual = fetch_account_taker_fee()
    result: dict[str, Any] = {
        "configured_taker_fee": TAKER_FEE,
        "actual_taker_fee": actual,
        "reconciled": actual is not None,
        "diverged": False,
        "delta_bps": None,
        "roundtrip_cost": ESTIMATED_ROUNDTRIP_COST,
    }
    if actual is None:
        logger.info("TAKER_FEE_RECONCILE skipped: no exchange rate available (creds / IP whitelist / network)")
        return result
    delta = actual - TAKER_FEE
    result["delta_bps"] = round(delta * 10000.0, 3)
    if abs(delta) > tolerance:
        result["diverged"] = True
        logger.warning(
            "TAKER_FEE_MISMATCH configured=%.4f%% actual=%.4f%% delta=%+.3f bps — the net-edge gate "
            "and round-trip cost (%.4f%%) are calibrated to the wrong taker fee; update TAKER_FEE / "
            "ESTIMATED_ROUNDTRIP_COST (env) or verify the account fee tier before trusting profit gates",
            TAKER_FEE * 100.0,
            actual * 100.0,
            result["delta_bps"],
            ESTIMATED_ROUNDTRIP_COST * 100.0,
        )
    else:
        logger.info("TAKER_FEE_RECONCILE ok configured=%.4f%% actual=%.4f%%", TAKER_FEE * 100.0, actual * 100.0)
    return result


def log_trading_economics_at_startup() -> TradingEconomicsSnapshot:
    snap = get_trading_economics()
    logger.warning(
        "TRADING_ECONOMICS_RESOLVED exchange=%s maker_fee=%s taker_fee=%s slippage_buffer=%s "
        "orderbook_half_spread=%s roundtrip_cost=%s min_net_profit_to_sell=%s "
        "min_profit_after_costs_usd=%s cooldown_after_sell=%ss cooldown_after_human_sell=%ss fee_source=%s "
        "day_notional_mult=%s day_target_notional_per_slot=%s day_max_deployed=%s",
        snap.exchange,
        snap.maker_fee,
        snap.taker_fee,
        snap.slippage_buffer,
        snap.orderbook_half_spread_estimate,
        snap.estimated_roundtrip_cost,
        snap.min_net_profit_to_sell,
        snap.min_profit_after_costs_usd,
        snap.cooldown_seconds_after_sell,
        snap.cooldown_seconds_after_human_sell,
        snap.fee_schedule_source_date,
        snap.day_notional_mult,
        snap.day_target_notional_per_slot_usd,
        snap.day_max_deployed_usd,
    )
    return snap
