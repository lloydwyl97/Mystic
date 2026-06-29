"""
Canonical live mark for Mystic — single source for exit monitor, status MTM,
positions API, dashboard, and unrealized PnL.

Always prefers a fresh Binance.US REST quote (bookTicker mid, then 24hr last).
Never uses unbounded in-process ticker caches or legacy redis market:{base} strings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any

from backend.config.mystic_api_schedule import SELL_MARK_MAX_AGE_SECONDS
from backend.utils.symbols import normalize_symbol, to_exchange_symbol

logger = logging.getLogger(__name__)

# Short shared TTL — avoids hammering REST while preventing cross-process stale drift.
_CANONICAL_MARK_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CANONICAL_MARK_CACHE_TTL_SEC = min(5.0, float(SELL_MARK_MAX_AGE_SECONDS))


@dataclass(frozen=True)
class CanonicalMark:
    symbol: str
    symbol_format: str
    mark: float
    bid: float | None
    ask: float | None
    mid: float | None
    last: float | None
    source: str
    timestamp: float
    age_seconds: float
    fresh: bool
    kline_1m_close: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_redis_val(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "ignore")
    return str(raw)


async def _fetch_binance_book_ticker(bus: str) -> dict[str, float] | None:
    from backend.services.canonical_http_client import get_http_client

    client = await get_http_client()
    url = "https://api.binance.us/api/v3/ticker/bookTicker"
    resp = await client.get(url, params={"symbol": bus}, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    bid = float(data.get("bidPrice") or 0.0)
    ask = float(data.get("askPrice") or 0.0)
    if bid <= 0 or ask <= 0:
        return None
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0}


async def _fetch_binance_last_price(bus: str) -> float | None:
    from backend.services.canonical_http_client import get_http_client

    client = await get_http_client()
    url = "https://api.binance.us/api/v3/ticker/24hr"
    resp = await client.get(url, params={"symbol": bus}, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    last_px = float(data.get("lastPrice") or 0.0)
    return last_px if last_px > 0 else None


async def _fetch_binance_1m_close(bus: str) -> float | None:
    from backend.services.canonical_http_client import get_http_client

    client = await get_http_client()
    url = "https://api.binance.us/api/v3/klines"
    resp = await client.get(url, params={"symbol": bus, "interval": "1m", "limit": 1}, timeout=10.0)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    close_px = float(rows[0][4])
    return close_px if close_px > 0 else None


async def _fetch_redis_price_hash(bus: str, now: float) -> CanonicalMark | None:
    """Fresh redis price:{BUS}USDT hash only — never legacy market:{base} string."""
    try:
        from backend.config.redis_config import get_redis_client

        redis_client = get_redis_client()
        if not redis_client:
            return None
        key = f"price:{bus}"
        h = redis_client.hgetall(key) or {}
        if not h:
            return None
        dec = {_decode_redis_val(k): _decode_redis_val(v) for k, v in h.items()}
        px = float(dec.get("v") or dec.get("price") or 0.0)
        ts_raw = dec.get("timestamp") or dec.get("ts")
        ts = float(ts_raw) if ts_raw else None
        if px <= 0 or ts is None:
            return None
        age = max(0.0, now - ts)
        if age > float(SELL_MARK_MAX_AGE_SECONDS):
            return None
        ccxt_sym = normalize_symbol(f"{bus.replace('USDT', '')}/USDT" if "USDT" not in bus else bus)
        if "/" not in ccxt_sym and bus.endswith("USDT"):
            ccxt_sym = normalize_symbol(f"{bus[:-4]}/USDT")
        return CanonicalMark(
            symbol=ccxt_sym,
            symbol_format=key,
            mark=px,
            bid=None,
            ask=None,
            mid=px,
            last=px,
            source="redis_price_hash",
            timestamp=ts,
            age_seconds=age,
            fresh=True,
        )
    except Exception as exc:
        logger.debug("CANONICAL_MARK_REDIS_FALLBACK_FAILED bus=%s %s", bus, exc)
        return None


async def fetch_canonical_mark(symbol: str, *, use_cache: bool = True) -> CanonicalMark | None:
    """
    Return the canonical current mark for *symbol* (CCXT or BUS format).

    Primary: Binance.US bookTicker mid (conservative alignment with executable quotes).
    Fallback: 24hr lastPrice, then fresh redis price:{BUS}USDT hash.
    """
    ccxt_sym = normalize_symbol(symbol)
    bus = to_exchange_symbol(ccxt_sym).replace("/", "").upper()
    now = time.time()

    if use_cache:
        cached = _CANONICAL_MARK_CACHE.get(bus)
        if cached and (now - cached[0]) <= _CANONICAL_MARK_CACHE_TTL_SEC:
            payload = dict(cached[1])
            payload["age_seconds"] = max(0.0, now - float(payload.get("timestamp") or now))
            payload["fresh"] = payload["age_seconds"] <= float(SELL_MARK_MAX_AGE_SECONDS)
            return CanonicalMark(**payload)

    bid = ask = mid = last = kline_close = None
    source = "missing"

    try:
        book = await _fetch_binance_book_ticker(bus)
        if book:
            bid, ask, mid = book["bid"], book["ask"], book["mid"]
            source = "binance_book_ticker_mid"
    except Exception as exc:
        logger.debug("CANONICAL_MARK_BOOK_TICKER_FAILED %s: %s", bus, exc)

    if mid is None or mid <= 0:
        try:
            last = await _fetch_binance_last_price(bus)
            if last and last > 0:
                mid = last
                source = "binance_ticker_24hr_last"
        except Exception as exc:
            logger.debug("CANONICAL_MARK_24HR_FAILED %s: %s", bus, exc)

    if mid is None or mid <= 0:
        redis_mark = await _fetch_redis_price_hash(bus, now)
        if redis_mark is not None:
            if use_cache:
                _CANONICAL_MARK_CACHE[bus] = (now, redis_mark.to_dict())
            return redis_mark
        return None

    try:
        kline_close = await _fetch_binance_1m_close(bus)
    except Exception:
        kline_close = None

    result = CanonicalMark(
        symbol=ccxt_sym,
        symbol_format=bus,
        mark=float(mid),
        bid=bid,
        ask=ask,
        mid=float(mid),
        last=last if last else float(mid),
        source=source,
        timestamp=now,
        age_seconds=0.0,
        fresh=True,
        kline_1m_close=kline_close,
    )
    if use_cache:
        _CANONICAL_MARK_CACHE[bus] = (now, result.to_dict())
    return result


def canonical_mark_to_exit_telemetry_fields(mark: CanonicalMark | None) -> dict[str, Any]:
    if mark is None:
        return {
            "mark_used": None,
            "mark_source": "missing",
            "mark_timestamp": None,
            "mark_age_seconds": None,
            "price_source_stale": True,
            "stale_mark_used": False,
            "bid": None,
            "ask": None,
            "mid": None,
            "last": None,
            "kline_1m_close": None,
            "canonical_source": "missing",
        }
    stale = not mark.fresh or mark.age_seconds > float(SELL_MARK_MAX_AGE_SECONDS)
    return {
        "mark_used": mark.mark,
        "mark_source": mark.source,
        "mark_timestamp": mark.timestamp,
        "mark_age_seconds": mark.age_seconds,
        "price_source_stale": stale,
        "stale_mark_used": stale,
        "bid": mark.bid,
        "ask": mark.ask,
        "mid": mark.mid,
        "last": mark.last,
        "kline_1m_close": mark.kline_1m_close,
        "canonical_source": mark.source,
        "symbol_format": mark.symbol_format,
    }
