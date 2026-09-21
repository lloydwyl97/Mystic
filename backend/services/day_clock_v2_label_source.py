"""V5 3h label market-data authority. Research only.

Does not use feature_ohlcv persistence timestamps as candle identity.
Does not feed live ranking, sizing, or exits.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.services.day_4h_entry_features import HOLD_SYMBOL
from backend.services.day_path_input_validity import parse_bar_ts

LABEL_SOURCE_VERSION = "day_clock_v2_label_source_v2"
INTERVAL = "1m"
INTERVAL_SEC = 60
OHLCV_TOLERANCE = 1e-8
REST_WINDOW_SEC = 10 * 60
REST_TIMEOUT_SEC = 10.0
BINANCE_US_KLINES = "https://api.binance.us/api/v3/klines"

SOURCE_REDIS = "redis_canonical_1m"
SOURCE_REST = "binance_us_rest_1m"

STATUS_PENDING_NOT_MATURE = "PENDING_NOT_MATURE"
STATUS_PENDING_LABEL_SOURCE = "PENDING_LABEL_SOURCE"
STATUS_COMPLETE = "COMPLETE"
STATUS_TERMINAL_INVALID = "TERMINAL_INVALID"

INVALID_NO_BARS = "NO_SOURCE_BARS_AT_HORIZON"
INVALID_MISMATCH = "LABEL_SOURCE_AUTHORITY_MISMATCH"
INVALID_REST_TRANSIENT = "LABEL_SOURCE_REST_TRANSIENT"
INVALID_REDIS_QUALITY = "LABEL_SOURCE_REDIS_QUALITY"
INVALID_FORMING = "LABEL_SOURCE_FORMING_CANDLE"
INVALID_FUTURE = "LABEL_SOURCE_FUTURE_CANDLE"
INVALID_PIT = "LABEL_SOURCE_PIT_VIOLATION"

RETRYABLE_REASONS = frozenset(
    {
        INVALID_NO_BARS,
        INVALID_REST_TRANSIENT,
        INVALID_REDIS_QUALITY,
    }
)
TERMINAL_REASONS = frozenset(
    {
        INVALID_MISMATCH,
        INVALID_FORMING,
        INVALID_FUTURE,
        INVALID_PIT,
    }
)

RestFetch = Callable[[str, int, int], list[Any]]


@dataclass(frozen=True)
class HorizonCandle:
    open_ts: datetime
    close_ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def ohlcv(self) -> tuple[float, float, float, float, float]:
        return (self.open, self.high, self.low, self.close, self.volume)


def _utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def api_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


def candle_close_ts(open_ts: datetime) -> datetime:
    return _utc(open_ts) + timedelta(seconds=INTERVAL_SEC)


def last_closed_open_ts(horizon_at: datetime) -> datetime:
    """Open time of the latest 1m candle whose close_ts <= horizon_at."""
    cutoff = _utc(horizon_at) - timedelta(seconds=INTERVAL_SEC)
    epoch = int(cutoff.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % INTERVAL_SEC), tz=timezone.utc)


def _num(raw: Any) -> float | None:
    try:
        if raw is None or raw == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_open_ts(raw: Any) -> datetime | None:
    return parse_bar_ts(raw)


def candle_from_parts(
    open_raw: Any,
    open_: Any,
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
) -> HorizonCandle | None:
    open_ts = parse_open_ts(open_raw)
    o, h, lo, c, v = _num(open_), _num(high), _num(low), _num(close), _num(volume)
    if open_ts is None or o is None or h is None or lo is None or c is None or v is None:
        return None
    if c <= 0 or h <= 0 or lo <= 0:
        return None
    return HorizonCandle(
        open_ts=_utc(open_ts),
        close_ts=candle_close_ts(open_ts),
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=v,
    )


def parse_redis_rows(raw: Any) -> list[HorizonCandle]:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out: list[HorizonCandle] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 5:
            candle = candle_from_parts(
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                item[5] if len(item) > 5 else 0.0,
            )
        elif isinstance(item, dict):
            candle = candle_from_parts(
                item.get("t", item.get("ts", item.get("open_ts"))),
                item.get("o", item.get("open")),
                item.get("h", item.get("high")),
                item.get("l", item.get("low")),
                item.get("c", item.get("close")),
                item.get("v", item.get("volume")),
            )
        else:
            continue
        if candle is not None:
            out.append(candle)
    return out


def parse_rest_klines(rows: Any) -> list[HorizonCandle]:
    if not isinstance(rows, list):
        return []
    out: list[HorizonCandle] = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        candle = candle_from_parts(item[0], item[1], item[2], item[3], item[4], item[5])
        if candle is not None:
            out.append(candle)
    return out


def redis_history_ok(candles: list[HorizonCandle]) -> bool:
    if not candles:
        return False
    opens = [c.open_ts for c in candles]
    if len(opens) != len(set(opens)):
        return False
    return all(opens[i] < opens[i + 1] for i in range(len(opens) - 1))


def drop_unclosed_or_future(
    candles: list[HorizonCandle],
    *,
    horizon_at: datetime,
    now: datetime,
) -> list[HorizonCandle]:
    horizon = _utc(horizon_at)
    stamp = _utc(now)
    kept: list[HorizonCandle] = []
    for candle in candles:
        if candle.close_ts > stamp:
            continue
        if candle.open_ts > stamp or candle.open_ts > horizon:
            continue
        if candle.close_ts > horizon:
            continue
        kept.append(candle)
    return kept


def select_closed_horizon_candle(
    candles: list[HorizonCandle],
    *,
    horizon_at: datetime,
    now: datetime,
) -> HorizonCandle | None:
    eligible = drop_unclosed_or_future(candles, horizon_at=horizon_at, now=now)
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.close_ts)


def ohlcv_equal(left: HorizonCandle, right: HorizonCandle) -> bool:
    if left.open_ts != right.open_ts or left.close_ts != right.close_ts:
        return False
    return all(abs(a - b) <= OHLCV_TOLERANCE for a, b in zip(left.ohlcv(), right.ohlcv(), strict=True))


def default_rest_fetch(symbol: str, start_ms: int, end_ms: int) -> list[Any]:
    url = f"{BINANCE_US_KLINES}?symbol={api_symbol(symbol)}&interval={INTERVAL}&startTime={int(start_ms)}&endTime={int(end_ms)}&limit=20"
    req = urllib.request.Request(url, headers={"User-Agent": "mystic-clock-v2-label/v2"})
    with urllib.request.urlopen(req, timeout=REST_TIMEOUT_SEC) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise TypeError("binance_us_klines_not_a_list")
    return payload


def _redis_client(existing: Any = None) -> Any:
    if existing is not None:
        return existing
    try:
        import redis

        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        return redis.from_url(url, decode_responses=True)
    except Exception:
        return None


def load_redis_candles(symbol: str, *, redis_client: Any = None) -> list[HorizonCandle] | None:
    if api_symbol(symbol) == HOLD_SYMBOL:
        return []
    client = _redis_client(redis_client)
    if client is None:
        return None
    try:
        raw = client.get(f"klines:{api_symbol(symbol)}:1m")
    except Exception:
        return None
    candles = parse_redis_rows(raw)
    if not redis_history_ok(candles):
        return []
    return candles


def rest_window_ms(horizon_at: datetime) -> tuple[int, int]:
    last_open = last_closed_open_ts(horizon_at)
    start = last_open - timedelta(seconds=REST_WINDOW_SEC)
    return int(start.timestamp() * 1000), int(last_open.timestamp() * 1000)


def resolve_v5_horizon_candle(
    symbol: str,
    horizon_at: datetime,
    *,
    now: datetime | None = None,
    redis_client: Any = None,
    rest_fetch: RestFetch | None = None,
) -> dict[str, Any]:
    """Resolve the last closed 1m candle with close_ts <= horizon_at."""
    stamp = _utc(now or datetime.now(timezone.utc))
    horizon = _utc(horizon_at)
    exchange = api_symbol(symbol)
    base = {
        "ok": False,
        "candle": None,
        "label_source": None,
        "label_source_version": LABEL_SOURCE_VERSION,
        "source_verified": False,
        "source_fetch_timestamp": stamp.isoformat(),
        "exchange_symbol": exchange,
        "interval": INTERVAL,
        "target_bar_open_ts": None,
        "target_bar_close_ts": None,
        "reason": None,
        "status": STATUS_PENDING_LABEL_SOURCE,
        "redis_present": False,
        "rest_present": False,
    }
    if exchange == HOLD_SYMBOL:
        base["ok"] = True
        base["status"] = STATUS_COMPLETE
        base["label_source"] = "hold_reference"
        return base
    if stamp.timestamp() + 1e-9 < horizon.timestamp():
        base["reason"] = "HORIZON_NOT_MATURE"
        base["status"] = STATUS_PENDING_NOT_MATURE
        return base

    redis_rows = load_redis_candles(symbol, redis_client=redis_client)
    redis_candle = None
    if redis_rows:
        redis_candle = select_closed_horizon_candle(redis_rows, horizon_at=horizon, now=stamp)
        base["redis_present"] = redis_candle is not None

    fetch = rest_fetch or default_rest_fetch
    start_ms, end_ms = rest_window_ms(horizon)
    rest_error = None
    rest_candle = None
    try:
        rest_rows = parse_rest_klines(fetch(exchange, start_ms, end_ms))
        rest_candle = select_closed_horizon_candle(rest_rows, horizon_at=horizon, now=stamp)
        base["rest_present"] = rest_candle is not None
    except (urllib.error.URLError, TimeoutError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        rest_error = str(exc) or exc.__class__.__name__

    if rest_error:
        base["reason"] = INVALID_REST_TRANSIENT
        base["status"] = STATUS_PENDING_LABEL_SOURCE
        return base

    if redis_candle is not None and rest_candle is not None:
        if not ohlcv_equal(redis_candle, rest_candle):
            base["reason"] = INVALID_MISMATCH
            base["status"] = STATUS_TERMINAL_INVALID
            return base
        return _ok(base, redis_candle, SOURCE_REDIS, verified=True)
    if redis_candle is not None and rest_candle is None:
        # Redis presence without a REST candle is not a contradiction. REST is
        # often briefly empty; keep the row retryable until REST can confirm or
        # disagree. Do not complete from Redis alone while verification is required.
        base["reason"] = INVALID_REST_TRANSIENT
        base["status"] = STATUS_PENDING_LABEL_SOURCE
        return base
    if rest_candle is not None:
        return _ok(base, rest_candle, SOURCE_REST, verified=False)

    base["reason"] = INVALID_NO_BARS
    base["status"] = STATUS_PENDING_LABEL_SOURCE
    return base


def _ok(base: dict[str, Any], candle: HorizonCandle, source: str, *, verified: bool) -> dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "ok": True,
            "candle": candle,
            "label_source": source,
            "source_verified": verified,
            "target_bar_open_ts": candle.open_ts.isoformat(),
            "target_bar_close_ts": candle.close_ts.isoformat(),
            "reason": None,
            "status": STATUS_COMPLETE,
        }
    )
    return out


def pit_ok(*, decision_ts: datetime, horizon_at: datetime, candle: HorizonCandle) -> bool:
    decision = _utc(decision_ts)
    horizon = _utc(horizon_at)
    if not (decision < horizon):
        return False
    if candle.close_ts > horizon:
        return False
    return not (candle.open_ts > horizon or candle.open_ts >= candle.close_ts)


__all__ = [
    "INTERVAL",
    "INVALID_MISMATCH",
    "INVALID_NO_BARS",
    "INVALID_REST_TRANSIENT",
    "LABEL_SOURCE_VERSION",
    "RETRYABLE_REASONS",
    "SOURCE_REDIS",
    "SOURCE_REST",
    "STATUS_COMPLETE",
    "STATUS_PENDING_LABEL_SOURCE",
    "STATUS_PENDING_NOT_MATURE",
    "STATUS_TERMINAL_INVALID",
    "TERMINAL_REASONS",
    "HorizonCandle",
    "candle_close_ts",
    "last_closed_open_ts",
    "ohlcv_equal",
    "parse_redis_rows",
    "parse_rest_klines",
    "pit_ok",
    "resolve_v5_horizon_candle",
    "select_closed_horizon_candle",
]
