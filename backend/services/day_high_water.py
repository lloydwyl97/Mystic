"""DAY open-position high-water fidelity.

Exit marks stay on the executable book mid. High-water / MFE / trail
activation may also fold the 1m candle high already fetched on that same
live path so a brief post-entry print is not lost between 45s mid samples.

High-water is monotonic: never lowered.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def fold_high_water(previous: float, *candidates: float | None) -> float:
    """Return max of previous and all positive candidates. Never lowers."""
    best = float(previous or 0.0)
    for raw in candidates:
        if raw is None:
            continue
        try:
            px = float(raw)
        except (TypeError, ValueError):
            continue
        if px > best:
            best = px
    return best


def first_full_minute_epoch(entry_epoch: float) -> float:
    """First 1m open strictly after entry — avoids pre-entry prints on the entry minute."""
    if entry_epoch <= 0:
        return 0.0
    return float((int(entry_epoch) // 60 + 1) * 60)


def kline_open_is_post_entry(entry_epoch: float, kline_open_epoch: float | None) -> bool:
    """True only when the 1m candle opened at or after the first full minute after entry.

    The current forming candle high has no trade timestamp. If that candle
    opened before entry, its high can include pre-entry prints. Without
    intraminute stamps we must not use that full high.
    """
    if entry_epoch <= 0:
        return False
    if kline_open_epoch is None:
        return False
    try:
        opened = float(kline_open_epoch)
    except (TypeError, ValueError):
        return False
    if opened <= 0:
        return False
    if opened > 1e12:
        opened = opened / 1000.0
    return opened + 1e-9 >= first_full_minute_epoch(entry_epoch)


def usable_kline_high(
    entry_epoch: float,
    kline_open_epoch: float | None,
    kline_high: float | None,
) -> float | None:
    """Return 1m high only when the candle is proven post-entry. Else None."""
    if kline_high is None:
        return None
    try:
        high = float(kline_high)
    except (TypeError, ValueError):
        return None
    if high <= 0:
        return None
    if not kline_open_is_post_entry(entry_epoch, kline_open_epoch):
        return None
    return high


def _candle_open_epoch(candle: dict[str, Any]) -> float:
    raw = candle.get("open_time", candle.get("ts", candle.get("timestamp")))
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val / 1000.0 if val > 1e12 else val
    text = str(raw).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def max_post_entry_1m_high(entry_epoch: float, candles: list[dict[str, Any]] | None) -> float:
    """Max 1m high from candles that cannot include a pre-entry print.

    Explicit ``open_time``: include if the candle opened at or after the first
    full minute after entry.

    Persist-only ``ts`` (feature_ohlcv writes now()): require persist time at
    or after first_full_minute+60s so an entry-minute bar flushed after the
    minute close cannot leak a pre-entry wick.
    """
    full_min = first_full_minute_epoch(entry_epoch)
    persist_cutoff = full_min + 60.0 if full_min > 0 else 0.0
    best = 0.0
    for candle in candles or []:
        if not isinstance(candle, dict):
            continue
        has_open = candle.get("open_time") is not None
        if has_open:
            ts = _candle_open_epoch({"open_time": candle.get("open_time")})
            cutoff = full_min
        else:
            ts = _candle_open_epoch(candle)
            cutoff = persist_cutoff
        if cutoff > 0 and ts + 1e-9 < cutoff:
            continue
        try:
            high = float(candle.get("high") or 0.0)
        except (TypeError, ValueError):
            continue
        if high > best:
            best = high
    return best


def load_feature_1m_candles(symbol: str, start_epoch: float, end_epoch: float | None = None) -> list[dict[str, Any]]:
    """Read persisted live 1m bars (feature_ohlcv) if present. Never raises to caller."""
    try:
        from backend.services.feature_store import get_ohlcv
        from backend.utils.symbols import to_ccxt_symbol, to_exchange_symbol
    except Exception:
        return []

    try:
        ccxt_sym = to_ccxt_symbol(symbol)
        bus = to_exchange_symbol(ccxt_sym).replace("/", "").upper()
        hyphen = str(ccxt_sym).replace("/", "-")
    except Exception:
        ccxt_sym = str(symbol)
        bus = str(symbol).replace("/", "").upper()
        hyphen = str(symbol).replace("/", "-")

    start = datetime.fromtimestamp(float(start_epoch), tz=timezone.utc) if start_epoch > 0 else None
    end = datetime.fromtimestamp(float(end_epoch), tz=timezone.utc) if end_epoch else None
    for form in (hyphen, ccxt_sym, bus, str(symbol)):
        try:
            rows = get_ohlcv(form, "1m", start=start, end=end, limit=2000)
        except Exception:
            continue
        if rows:
            return rows
    return []
