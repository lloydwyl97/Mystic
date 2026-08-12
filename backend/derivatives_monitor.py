import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)


def _http_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        resp = httpx.get(url, params=params or {}, headers=headers or {}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning(f"Unexpected JSON response type: {type(data)}")
            result = {}
        else:
            result = data
    except httpx.HTTPStatusError as e:
        # Handle 451 "Unavailable For Legal Reasons" (Binance Futures blocked in US)
        if e.response.status_code == 451:
            logger.debug(f"API blocked in your region (HTTP 451): {url}")
            return {}
        # Log other HTTP errors as warnings, not exceptions
        logger.warning(f"HTTP {e.response.status_code} error for {url}")
        return {}
    except httpx.RequestError as e:
        logger.warning(f"HTTP request failed for {url}: {e}")
        return {}
    except ValueError as e:
        logger.warning(f"JSON decode failed for {url}: {e}")
        return {}
    else:
        return result


def _reference_feed_enabled() -> bool:
    """Kill switch for the public derivatives reference feed, independent of
    the execution exchange. Default on; set DERIVATIVES_REFERENCE_FEED_ENABLED=0
    to disable without a code change."""
    return os.getenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def fetch_binance_open_interest(symbol: str) -> dict[str, Any]:
    """Fetch open interest from Binance's GLOBAL futures API as a public,
    non-execution REFERENCE feed (item p18).

    This is intentionally decoupled from ``EXCHANGE_ID`` (the actual
    execution venue is Binance US, which has no futures market at all).
    ``/fapi/v1/openInterest`` and ``/fapi/v1/ticker/24hr`` are unauthenticated
    public endpoints — no API key is required or sent. Never used for order
    execution or routing; informational ranking/context input only.
    """
    if not _reference_feed_enabled():
        return {}

    # Normalize symbol (remove USDT and add USDT back for futures)
    normalized = symbol.upper().replace("/", "").replace("-", "")
    if not normalized.endswith("USDT"):
        normalized += "USDT"

    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {
        "symbol": normalized,
    }
    headers = {"User-Agent": "mystic-trading/1.0"}

    data = _http_get(url, params=params, headers=headers)

    if not data or not isinstance(data, dict):
        return {}

    try:
        open_interest = float(data.get("openInterest", 0))
        timestamp = int(data.get("time", 0))
    except (ValueError, TypeError, KeyError):
        return {}

    if open_interest <= 0:
        return {}

    # Get additional market data for context
    market_data = _fetch_futures_ticker_data(normalized)

    # Calculate positioning metrics
    positioning_signals = _analyze_positioning(open_interest, market_data, symbol)

    return {
        "symbol": symbol,
        "open_interest": open_interest,
        "positioning_signals": positioning_signals,
        "timestamp": timestamp,
    }


def _fetch_futures_ticker_data(symbol: str) -> dict[str, Any]:
    """Fetch 24hr futures ticker data (public endpoint, no key needed) for
    additional positioning context."""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    params = {"symbol": symbol}
    headers = {"User-Agent": "mystic-trading/1.0"}

    data = _http_get(url, params=params, headers=headers)
    return data if isinstance(data, dict) else {}


def fetch_binance_funding_and_basis(symbol: str) -> dict[str, Any]:
    """Fetch current funding rate + mark/index price (hence basis) from
    Binance's GLOBAL futures public ``premiumIndex`` endpoint. Same
    non-execution reference-feed scope as ``fetch_binance_open_interest``.

    basis_pct = (mark_price - index_price) / index_price — positive means
    the perpetual is trading rich vs spot/index (positioning-bullish tilt);
    negative means it's trading cheap (positioning-bearish tilt). Purely
    informational; never a gate.
    """
    if not _reference_feed_enabled():
        return {}

    normalized = symbol.upper().replace("/", "").replace("-", "")
    if not normalized.endswith("USDT"):
        normalized += "USDT"

    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    params = {"symbol": normalized}
    headers = {"User-Agent": "mystic-trading/1.0"}

    data = _http_get(url, params=params, headers=headers)
    if not data or not isinstance(data, dict):
        return {}

    try:
        mark_price = float(data.get("markPrice", 0.0))
        index_price = float(data.get("indexPrice", 0.0))
        funding_rate = float(data.get("lastFundingRate", 0.0))
        next_funding_time = int(data.get("nextFundingTime", 0))
    except (TypeError, ValueError):
        return {}

    if mark_price <= 0.0 or index_price <= 0.0:
        return {}

    basis_pct = (mark_price - index_price) / index_price

    return {
        "symbol": symbol,
        "mark_price": mark_price,
        "index_price": index_price,
        "basis_pct": basis_pct,
        "funding_rate": funding_rate,
        "next_funding_time": next_funding_time,
    }


_DERIV_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_DERIV_CACHE_TTL_SEC = 60.0

_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS derivatives_reference_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    open_interest REAL,
    funding_rate REAL,
    basis_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_deriv_hist_symbol_ts ON derivatives_reference_history(symbol, ts_utc);
"""

_HISTORY_LOOKBACK_DAYS = 30
_MIN_HISTORY_SAMPLES_FOR_PERCENTILE = 8


def _ensure_history_schema(db_path: str) -> None:
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.executescript(_HISTORY_SCHEMA)
    except Exception as exc:
        logger.debug("DERIVATIVES_HISTORY_SCHEMA_FAILED: %s", exc)


def _persist_history_row(
    symbol: str,
    *,
    open_interest: float | None,
    funding_rate: float | None,
    basis_pct: float | None,
    db_path: str,
) -> None:
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO derivatives_reference_history
                    (symbol, ts_utc, open_interest, funding_rate, basis_pct)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol.upper(), datetime.now(timezone.utc).isoformat(), open_interest, funding_rate, basis_pct),
            )
    except Exception as exc:
        logger.debug("DERIVATIVES_HISTORY_PERSIST_FAILED symbol=%s: %s", symbol, exc)


def _percentile_rank(value: float, history: list[float]) -> float | None:
    """Fraction of historical samples <= value, in [0, 1]. None if too few samples
    to be a statistically meaningful percentile (honest, not a fabricated 0.5)."""
    if len(history) < _MIN_HISTORY_SAMPLES_FOR_PERCENTILE:
        return None
    return sum(1 for h in history if h <= value) / len(history)


def _zscore(value: float, history: list[float]) -> float | None:
    if len(history) < _MIN_HISTORY_SAMPLES_FOR_PERCENTILE:
        return None
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    std = var**0.5
    if std <= 1e-12:
        return 0.0
    return (value - mean) / std


def _change_percentile_zscore(
    symbol: str,
    *,
    current_oi: float | None,
    current_funding: float | None,
    current_basis: float | None,
    db_path: str,
) -> dict[str, Any]:
    """Item p18 gap-closure: funding change/percentile, OI change, basis
    change/z-score, computed from real persisted history (not fabricated) —
    honestly returns None for any stat that doesn't yet have enough history
    (item p18's own _MIN_HISTORY_SAMPLES_FOR_PERCENTILE floor)."""
    out: dict[str, Any] = {
        "funding_rate_change": None,
        "funding_rate_percentile": None,
        "open_interest_change_pct": None,
        "basis_pct_change": None,
        "basis_pct_zscore": None,
        "history_sample_count": 0,
    }
    try:
        since_iso = datetime.fromtimestamp(time.time() - _HISTORY_LOOKBACK_DAYS * 86400, tz=timezone.utc).isoformat()
        with sqlite3.connect(db_path, timeout=10) as conn:
            rows = conn.execute(
                """
                SELECT open_interest, funding_rate, basis_pct
                FROM derivatives_reference_history
                WHERE symbol = ? AND ts_utc >= ?
                ORDER BY ts_utc ASC
                """,
                (symbol.upper(), since_iso),
            ).fetchall()
    except Exception as exc:
        logger.debug("DERIVATIVES_HISTORY_READ_FAILED symbol=%s: %s", symbol, exc)
        return out

    out["history_sample_count"] = len(rows)
    if not rows:
        return out

    prior_oi = [r[0] for r in rows if r[0] is not None]
    prior_funding = [r[1] for r in rows if r[1] is not None]
    prior_basis = [r[2] for r in rows if r[2] is not None]

    if current_funding is not None:
        if prior_funding:
            out["funding_rate_change"] = current_funding - prior_funding[-1]
        out["funding_rate_percentile"] = _percentile_rank(current_funding, prior_funding)
    if current_oi is not None and prior_oi and prior_oi[-1]:
        out["open_interest_change_pct"] = (current_oi - prior_oi[-1]) / abs(prior_oi[-1])
    if current_basis is not None:
        if prior_basis:
            out["basis_pct_change"] = current_basis - prior_basis[-1]
        out["basis_pct_zscore"] = _zscore(current_basis, prior_basis)
    return out


def derivatives_reference_snapshot(symbol: str, *, db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Combined OI + funding/basis reference snapshot for one symbol, with an
    explicit ``available`` flag and a short in-process TTL cache (this feed
    is polled once per AI-context tick per symbol; 60s cache keeps call
    volume bounded without depending on any external scheduler).

    Always returns a dict with ``available`` set — callers must treat
    ``available: False`` as an honest degraded state (futures data
    unreachable/blocked/rate-limited), never as a neutral market read.

    The OI feed and the funding/basis feed are fetched and reported
    independently (``oi_available`` / ``funding_basis_available``): if only
    one succeeds, the unavailable side's fields are ``None``, never
    zero-filled — a real 0.0 funding rate must never be indistinguishable
    from "feed didn't respond this tick".
    """
    now = time.time()
    cached = _DERIV_CACHE.get(symbol)
    if cached is not None and (now - cached[0]) < _DERIV_CACHE_TTL_SEC:
        return cached[1]

    if not _reference_feed_enabled():
        result = {"available": False, "degraded_reason": "reference_feed_disabled"}
        _DERIV_CACHE[symbol] = (now, result)
        return result

    oi_data = fetch_binance_open_interest(symbol)
    fb_data = fetch_binance_funding_and_basis(symbol)
    oi_available = bool(oi_data)
    fb_available = bool(fb_data)

    if not oi_available and not fb_available:
        result = {"available": False, "degraded_reason": "futures_api_unreachable_or_no_market"}
        _DERIV_CACHE[symbol] = (now, result)
        return result

    signals = derivatives_signal_check(symbol, _data=oi_data) if oi_available else {}

    open_interest = float(oi_data.get("open_interest", 0.0)) if oi_available else None
    funding_rate = float(fb_data.get("funding_rate", 0.0)) if fb_available else None
    basis_pct = float(fb_data.get("basis_pct", 0.0)) if fb_available else None

    _ensure_history_schema(db_path)
    # Compute change/percentile/z-score against PRIOR history before persisting
    # this snapshot, so "change" is never current-vs-itself.
    history_stats = _change_percentile_zscore(symbol, current_oi=open_interest, current_funding=funding_rate, current_basis=basis_pct, db_path=db_path)
    _persist_history_row(symbol, open_interest=open_interest, funding_rate=funding_rate, basis_pct=basis_pct, db_path=db_path)

    result = {
        "available": True,
        "oi_available": oi_available,
        "funding_basis_available": fb_available,
        "degraded_reason": None if (oi_available and fb_available) else ("oi_unavailable" if not oi_available else "funding_basis_unavailable"),
        "open_interest": open_interest,
        "oi_volume_ratio": float(signals.get("oi_volume_ratio", 0.0)) if oi_available else None,
        "positioning_bias": signals.get("positioning_bias", "neutral") if oi_available else None,
        "bias_strength": float(signals.get("bias_strength", 0.0)) if oi_available else None,
        "extreme_positioning": bool(signals.get("extreme_positioning", False)) if oi_available else None,
        "funding_rate": funding_rate,
        "basis_pct": basis_pct,
        **history_stats,
    }
    _DERIV_CACHE[symbol] = (now, result)
    return result


def derivatives_positioning_signal(snapshot: dict[str, Any]) -> float:
    """Item p18 ranking promotion: fold OI positioning-bias and funding-rate
    percentile into one bounded [-1, 1] signal for ``ai_market_context``'s
    ctx_multiplier. Zero whenever the underlying feed is unavailable or a
    given sub-component didn't succeed this tick (honest degrade — never a
    fabricated neutral read standing in for missing data at the ranking
    layer beyond the intentional 0.0 no-op)."""
    if not snapshot or not snapshot.get("available"):
        return 0.0

    signal = 0.0
    if snapshot.get("oi_available") and snapshot.get("positioning_bias") not in (None, "neutral"):
        bias_sign = 1.0 if snapshot["positioning_bias"] == "bullish" else -1.0
        signal += 0.7 * bias_sign * float(snapshot.get("bias_strength") or 0.0)

    funding_pctile = snapshot.get("funding_rate_percentile")
    if snapshot.get("funding_basis_available") and funding_pctile is not None:
        # Contrarian tilt: crowded-long (high funding percentile) skews mildly
        # bearish, crowded-short skews mildly bullish. Only fires once enough
        # funding history exists for a real percentile (never on a fabricated one).
        signal += 0.3 * (-(float(funding_pctile) - 0.5) * 2.0)

    return max(-1.0, min(1.0, signal))


def _analyze_positioning(open_interest: float, market_data: dict[str, Any], _symbol: str) -> dict[str, Any]:
    """Analyze open interest and market data to derive positioning signals."""

    # Get market data values
    try:
        price_change_pct = float(market_data.get("priceChangePercent", 0))
        volume = float(market_data.get("volume", 0))
        count_top_bid = float(market_data.get("countTopBid", 0))
        count_top_ask = float(market_data.get("countTopAsk", 0))
    except (ValueError, TypeError):
        price_change_pct = 0.0
        volume = 0.0
        count_top_bid = 0.0
        count_top_ask = 0.0

    # Calculate positioning metrics
    bid_ask_imbalance = count_top_bid / max(count_top_ask + count_top_bid, 1)

    # Open interest relative to volume (liquidity positioning)
    oi_volume_ratio = open_interest / max(volume, 1)

    # Price momentum vs positioning - GRADUAL SCALING (was binary 1.0/-1.0)
    # Price momentum component: scale from -1 to +1 based on price change (+-5% = full signal)
    price_momentum = max(-1.0, min(1.0, price_change_pct / 5.0))

    # Imbalance component: 0.5 is neutral, scale from -1 to +1
    imbalance_signal = (bid_ask_imbalance - 0.5) * 2.0  # 0.0->-1.0, 0.5->0.0, 1.0->+1.0

    # Combined alignment: average of price momentum and order flow
    if (price_momentum > 0 and imbalance_signal > 0) or (price_momentum < 0 and imbalance_signal < 0):
        # Aligned - bullish or bearish with conviction
        momentum_alignment = (price_momentum + imbalance_signal) / 2.0
    else:
        # Misaligned - reduce confidence, stay closer to neutral
        momentum_alignment = (price_momentum + imbalance_signal) / 4.0

    # Volatility expectation based on open interest changes
    # High OI often precedes volatility
    volatility_expectation = min(oi_volume_ratio / 1000.0, 1.0)  # Normalize

    # Market positioning bias
    if bid_ask_imbalance > 0.6:
        positioning_bias = "bullish"
        bias_strength = min(bid_ask_imbalance, 1.0)
    elif bid_ask_imbalance < 0.4:
        positioning_bias = "bearish"
        bias_strength = min(1.0 - bid_ask_imbalance, 1.0)
    else:
        positioning_bias = "neutral"
        bias_strength = 0.5

    # Extreme positioning warnings
    extreme_positioning = oi_volume_ratio > 2000.0  # Very high OI relative to spot volume

    return {
        "open_interest_volume_ratio": oi_volume_ratio,
        "bid_ask_imbalance": bid_ask_imbalance,
        "momentum_alignment": momentum_alignment,
        "volatility_expectation": volatility_expectation,
        "positioning_bias": positioning_bias,
        "bias_strength": bias_strength,
        "extreme_positioning": extreme_positioning,
        "price_change_pct": price_change_pct,
    }


def derivatives_signal_check(symbol: str, *, _data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get derivatives positioning signals for decision making.

    Pass `_data` (an already-fetched ``fetch_binance_open_interest`` result)
    to avoid a duplicate network round trip when the caller already has it
    (see ``derivatives_reference_snapshot``)."""
    data = _data if _data is not None else fetch_binance_open_interest(symbol)

    if not data:
        return {
            "oi_volume_ratio": 0.0,
            "positioning_volatility": 0.0,
            "momentum_alignment": 0.0,
            "positioning_bias": "neutral",
            "bias_strength": 0.0,
            "extreme_positioning": False,
        }

    signals = data.get("positioning_signals", {})

    # Transform for decision making
    oi_ratio = float(signals.get("open_interest_volume_ratio", 0.0))
    volatility = float(signals.get("volatility_expectation", 0.0))
    alignment = float(signals.get("momentum_alignment", 0.0))
    bias = signals.get("positioning_bias", "neutral")
    strength = float(signals.get("bias_strength", 0.0))
    extreme = bool(signals.get("extreme_positioning", False))

    # Normalize OI ratio (cap at reasonable levels)
    normalized_oi_ratio = min(oi_ratio / 1000.0, 1.0)

    # Calculate positioning risk (high OI + extreme positioning = higher risk)
    positioning_risk = normalized_oi_ratio
    if extreme:
        positioning_risk *= 1.5
    positioning_risk = min(positioning_risk, 1.0)

    # Bias confidence adjustment
    bias_confidence = strength if bias != "neutral" else 0.0

    logger.info(f"Derivatives signals for {symbol}: OI_ratio={normalized_oi_ratio:.3f}, volatility={volatility:.3f}, alignment={alignment:.2f}, bias={bias}({bias_confidence:.2f}), extreme={extreme}")

    return {
        "oi_volume_ratio": normalized_oi_ratio,
        "positioning_volatility": volatility,
        "momentum_alignment": alignment,
        "positioning_bias": bias,
        "bias_strength": bias_confidence,
        "extreme_positioning": extreme,
        "positioning_risk": positioning_risk,
    }
