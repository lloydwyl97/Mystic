"""DAY V2 five-minute confirmation gate.

Checks for a two-bar pattern on the 5m timeframe before allowing a structural-
pullback intent to submit:

  1. A red bar (close < open) anywhere after the opportunity was armed.
  2. A subsequent bar whose close reclaims: close > prior_bar.high OR close > reclaim_level.

Source detection result: NATIVE_5M (feature_ohlcv interval='5m' exists and is live).
If native 5m bars are unavailable, SYNTHETIC_5X1M path is used as fallback (5 x 1m bars
aligned to 300s epoch boundaries).

Public API
----------
check_5m_confirmation(db_path, symbol, reclaim_level, *, opportunity_armed_at, now)
    -> FiveMinConfirmResult

load_synthetic_5m_bars(db_path, symbol, *, since_ts, now)
    -> list[SyntheticBar]
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical symbol converter
# ---------------------------------------------------------------------------

_HYPHEN_FMT = re.compile(r"^([A-Z]{2,10})-USDT$")
_SLASH_FMT = re.compile(r"^([A-Z]{2,10})/USDT$")
_BARE_FMT = re.compile(r"^([A-Z]{2,10})USDT$")


def canonical_db_symbol(symbol: str, db_path: str) -> str | None:
    """Resolve *symbol* to the exact string stored in feature_ohlcv.

    Tries BTC-USDT (hyphen) first (Ocean live format), then BTC/USDT, then
    BTCUSDT.  Returns the first format that has at least one 5m row in the DB,
    or None when no row exists (entry must be rejected — do not bypass gate).

    Never returns a bare format that is known to have zero rows.
    """
    raw = str(symbol or "").strip().upper()
    # Normalise to base asset only
    base: str
    if (m := _HYPHEN_FMT.match(raw)) or (m := _SLASH_FMT.match(raw)) or (m := _BARE_FMT.match(raw)):
        base = m.group(1)
    else:
        base = raw.replace("-", "").replace("/", "").replace("USDT", "")
    if not base:
        return None
    candidates = [f"{base}-USDT", f"{base}/USDT", f"{base}USDT"]
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            for cand in candidates:
                row = conn.execute(
                    "SELECT 1 FROM feature_ohlcv WHERE symbol=? AND interval='5m' LIMIT 1",
                    (cand,),
                ).fetchone()
                if row:
                    return cand
    except Exception as exc:
        logger.warning("canonical_db_symbol lookup failed symbol=%s: %s", symbol, exc)
    return None


# Set at module load time by inspecting the available bar intervals.
# NATIVE_5M  = feature_ohlcv interval='5m' confirmed live (Ocean: 56k rows, up to today).
# SYNTHETIC_5X1M = fall back to building 5m bars from 5 x closed 1m bars.
CONFIRMATION_SOURCE: str = "NATIVE_5M"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticBar:
    """A (possibly synthetic) 5-minute OHLCV bar."""

    ts: int  # Unix epoch seconds — start of the 5m window
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str  # "NATIVE_5M" or "SYNTHETIC_5X1M"


@dataclass(frozen=True)
class FiveMinConfirmResult:
    """Result of the two-bar confirmation check."""

    confirmed: bool
    red_bar_ts: float  # epoch of the first red bar (0.0 if not found)
    reclaim_bar_ts: float  # epoch of the reclaim bar (0.0 if not confirmed)
    reclaim_level: float  # the reclaim_level that was evaluated
    source: str  # NATIVE_5M or SYNTHETIC_5X1M
    reason: str  # CONFIRMED | STALE_1M_DATA | INSUFFICIENT_BARS | NO_PATTERN


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _five_min_boundary(ts: float) -> int:
    """Return the largest 300-second boundary <= ts."""
    return int(ts) - (int(ts) % 300)


def _dt_str_to_epoch(ts_val: object) -> float:
    """Convert a SQLite datetime string or epoch float/int to epoch float (UTC)."""
    if isinstance(ts_val, (int, float)):
        return float(ts_val)
    s = str(ts_val)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


def _epoch_to_dt_str(epoch: float) -> str:
    """Convert epoch float to SQLite datetime string format."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000000")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_native_5m_bars(
    db_path: str,
    symbol: str,
    since_ts: float,
    until_ts: float,
) -> list[dict]:
    """Query feature_ohlcv for closed 5m bars in [since_ts, until_ts)."""
    since_str = _epoch_to_dt_str(since_ts)
    until_str = _epoch_to_dt_str(until_ts)
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM feature_ohlcv
                WHERE symbol=? AND interval='5m'
                  AND ts >= ? AND ts < ?
                ORDER BY ts ASC
                """,
                (symbol, since_str, until_str),
            ).fetchall()
        return [
            {
                "ts": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in rows
        ]
    except Exception:
        logger.debug("_load_native_5m_bars failed symbol=%s", symbol, exc_info=True)
        return []


def _load_1m_bars(
    db_path: str,
    symbol: str,
    since_ts: float,
    until_ts: float,
) -> list[dict]:
    """Query feature_ohlcv for 1m bars in [since_ts, until_ts) — synthetic path."""
    since_str = _epoch_to_dt_str(since_ts)
    until_str = _epoch_to_dt_str(until_ts)
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM feature_ohlcv
                WHERE symbol=? AND interval='1m'
                  AND ts >= ? AND ts < ?
                ORDER BY ts ASC
                """,
                (symbol, since_str, until_str),
            ).fetchall()
        return [
            {
                "ts": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in rows
        ]
    except Exception:
        logger.debug("_load_1m_bars failed symbol=%s", symbol, exc_info=True)
        return []


def _build_synthetic_5m(bars_1m: list[dict], boundary_ts: int) -> SyntheticBar | None:
    """Build one synthetic 5m bar from exactly 5 contiguous 1m bars at boundary_ts."""
    bucket = [b for b in bars_1m if _five_min_boundary(_dt_str_to_epoch(b["ts"])) == boundary_ts]
    if len(bucket) < 5:
        return None
    return SyntheticBar(
        ts=boundary_ts,
        open=float(bucket[0]["open"]),
        high=max(float(b["high"]) for b in bucket),
        low=min(float(b["low"]) for b in bucket),
        close=float(bucket[-1]["close"]),
        volume=sum(float(b["volume"]) for b in bucket),
        source="SYNTHETIC_5X1M",
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_synthetic_5m_bars(
    db_path: str,
    symbol: str,
    *,
    since_ts: float,
    now: float,
) -> list[SyntheticBar]:
    """Return fully-closed 5m bars from since_ts to now.

    Staleness guard: if the data feed appears stale (no recent bar), returns [].
    For NATIVE_5M: stale when latest bar started more than 720s before now.
    For SYNTHETIC_5X1M: stale when latest 1m bar is more than 120s old.
    """
    # A bar that opened at epoch T is closed when T + 300 <= now - 60
    # (60s safety margin to ensure the bar writer has committed it).
    cutoff = now - 60.0

    if CONFIRMATION_SOURCE == "NATIVE_5M":
        raw = _load_native_5m_bars(db_path, symbol, since_ts, cutoff + 300.0)
        if not raw:
            return []
        latest_epoch = _dt_str_to_epoch(raw[-1]["ts"])
        # Stale if feed has not delivered a 5m bar in the last 720s (2.4 bars)
        if now - latest_epoch > 720.0:
            return []
        result: list[SyntheticBar] = []
        for b in raw:
            bar_epoch = _dt_str_to_epoch(b["ts"])
            if bar_epoch + 300.0 <= cutoff:
                result.append(
                    SyntheticBar(
                        ts=int(bar_epoch),
                        open=b["open"],
                        high=b["high"],
                        low=b["low"],
                        close=b["close"],
                        volume=b["volume"],
                        source="NATIVE_5M",
                    )
                )
        return result

    # --- SYNTHETIC_5X1M path ---
    bars_1m = _load_1m_bars(db_path, symbol, since_ts, cutoff)
    if not bars_1m:
        return []
    latest_1m_epoch = _dt_str_to_epoch(bars_1m[-1]["ts"])
    if now - latest_1m_epoch > 120.0:
        return []  # stale 1m feed
    start_boundary = _five_min_boundary(since_ts)
    end_boundary = _five_min_boundary(cutoff)
    result_syn: list[SyntheticBar] = []
    t = start_boundary
    while t < end_boundary:
        bar = _build_synthetic_5m(bars_1m, t)
        if bar is not None:
            result_syn.append(bar)
        t += 300
    return result_syn


# ---------------------------------------------------------------------------
# Confirmation check
# ---------------------------------------------------------------------------


def check_5m_confirmation(
    db_path: str,
    symbol: str,
    reclaim_level: float,
    *,
    opportunity_armed_at: float,
    now: float | None = None,
) -> FiveMinConfirmResult:
    """Check for two-bar 5m confirmation pattern after opportunity_armed_at.

    Pattern (both bars must be fully closed):
      bar[i]:   red  — close < open
      bar[i+1]: reclaim — close > bar[i].high  OR  close > reclaim_level

    Returns FiveMinConfirmResult with confirmed=True when the pattern is found.
    """
    import time as _time

    if now is None:
        now = _time.time()

    bars = load_synthetic_5m_bars(db_path, symbol, since_ts=opportunity_armed_at, now=now)

    if not bars:
        return FiveMinConfirmResult(
            confirmed=False,
            red_bar_ts=0.0,
            reclaim_bar_ts=0.0,
            reclaim_level=reclaim_level,
            source=CONFIRMATION_SOURCE,
            reason="STALE_1M_DATA",
        )

    if len(bars) < 2:
        return FiveMinConfirmResult(
            confirmed=False,
            red_bar_ts=0.0,
            reclaim_bar_ts=0.0,
            reclaim_level=reclaim_level,
            source=CONFIRMATION_SOURCE,
            reason="INSUFFICIENT_BARS",
        )

    # Scan for red-then-reclaim pattern
    for i in range(len(bars) - 1):
        red = bars[i]
        reclaim = bars[i + 1]
        if red.close >= red.open:
            continue  # not a red bar
        if reclaim.close > red.high or reclaim.close > reclaim_level:
            return FiveMinConfirmResult(
                confirmed=True,
                red_bar_ts=float(red.ts),
                reclaim_bar_ts=float(reclaim.ts),
                reclaim_level=reclaim_level,
                source=CONFIRMATION_SOURCE,
                reason="CONFIRMED",
            )

    return FiveMinConfirmResult(
        confirmed=False,
        red_bar_ts=0.0,
        reclaim_bar_ts=0.0,
        reclaim_level=reclaim_level,
        source=CONFIRMATION_SOURCE,
        reason="NO_PATTERN",
    )
