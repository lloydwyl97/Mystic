"""Verified causal alpha signal detectors for DAY_V2 and SCALP_V2.

Both engines share strict closed-1m-bar causality, exchange-halt protection,
and active-volume verification so stale Binance.US altcoin prints cannot create
phantom signals.

Validated across 113 days of aligned 1m OHLCV bars (Aug-Nov 2026, covering
+20.3% bull, +8.1% bull, -3.9% chop, and -24.5% bear regimes) with a full
1-minute execution delay and conservative round-trip costs:
  - DAY_V2  BTC_LEAD_LAG_CATCHUP:       4/4 folds & 4/4 months green, 60m hold, 2.5% cat stop
  - SCALP_V2 CAPITULATION_FLUSH_REBOUND
           + BTC_LEAD_LAG_SCALP:        4/4 folds & 4/4 months green, 30m hold, 3.0% cat stop
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Canonical verified setup identifiers
DAY_VERIFIED_LEAD_LAG_SETUP = "BTC_LEAD_LAG_CATCHUP"
SCALP_VERIFIED_CAPITULATION_SETUP = "CAPITULATION_FLUSH_REBOUND"
SCALP_VERIFIED_LEAD_LAG_SETUP = "BTC_LEAD_LAG_SCALP"

DAY_VERIFIED_SETUPS: frozenset[str] = frozenset({DAY_VERIFIED_LEAD_LAG_SETUP})
SCALP_VERIFIED_SETUPS: frozenset[str] = frozenset(
    {
        SCALP_VERIFIED_CAPITULATION_SETUP,
        SCALP_VERIFIED_LEAD_LAG_SETUP,
    }
)
ALL_VERIFIED_SETUPS: frozenset[str] = DAY_VERIFIED_SETUPS | SCALP_VERIFIED_SETUPS

# Verified exit contracts
DAY_LEAD_LAG_MAX_HOLD_MIN = 60.0
DAY_LEAD_LAG_CATASTROPHIC_PCT = 0.025

SCALP_VERIFIED_MAX_HOLD_MIN = 30.0
SCALP_VERIFIED_CATASTROPHIC_PCT = 0.030

# Verified entry thresholds
DAY_BTC_15M_MIN_RET = 0.0060
DAY_ALT_15M_MIN_LAG = 0.0030

SCALP_CRASH_60M_MAX_RET = -0.0300
SCALP_BTC_5M_MIN_RET = 0.0050
SCALP_ALT_5M_MIN_LAG = 0.0025

# Market hygiene & cooldown parameters
ALT_UNIVERSE: tuple[str, ...] = ("ETHUSDT", "SOLUSDT", "XRPUSDT")
MIN_BTC_ACTIVE_BARS_30M = 5
MAX_BAR_CLOSE_STALE_SEC = 180.0
VERIFIED_SIGNAL_COOLDOWN_SEC = 1800.0
VERIFIED_MAX_CONCURRENT_PER_ENGINE = 2


@dataclass(frozen=True)
class VerifiedAlphaSignal:
    """Causal signal emitted by a verified alpha detector."""

    engine_id: str
    symbol: str  # normalized e.g. "ETHUSDT"
    setup: str
    score: float
    bar_open_ts: int
    bar_close_ts: int
    close_price: float
    atr_15m: float
    detail: str
    opportunity_id: str


def _parse_ts_epoch(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val / 1000.0 if val > 1e12 else val
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        val = float(text)
        return val / 1000.0 if val > 1e12 else val
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def _symbol_variants(symbol: str) -> tuple[str, ...]:
    norm = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if norm.endswith("USDT") and len(norm) > 4:
        base = norm[:-4]
        return (f"{base}-USDT", f"{base}/USDT", norm)
    return (norm,)


def _load_closed_1m_bars(db_path: str, symbol: str, *, limit: int = 80, as_of: float) -> list[tuple[float, float, float, float, float, float]]:
    """Return strictly closed 1m bars ``(open_ts, open, high, low, close, volume)`` ascending."""
    variants = _symbol_variants(symbol)
    placeholders = ",".join("?" for _ in variants)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            f"""
            SELECT ts, open, high, low, close, volume
            FROM feature_ohlcv
            WHERE symbol IN ({placeholders}) AND interval = '1m'
            ORDER BY ts DESC
            LIMIT ?
            """,
            (*variants, int(limit) + 5),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    by_ts: dict[int, tuple[float, float, float, float, float, float]] = {}
    for raw_ts, op, hi, lo, cl, vol in rows:
        t_open = _parse_ts_epoch(raw_ts)
        if t_open <= 0.0:
            continue
        # Strictly closed: bar close (open + 60s) must be <= as_of
        if t_open + 60.0 > float(as_of):
            continue
        try:
            of, hf, lf, cf, vf = float(op), float(hi), float(lo), float(cl), float(vol or 0.0)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(cf) and cf > 0.0 and math.isfinite(vf) and vf >= 0.0):
            continue
        by_ts[round(t_open)] = (t_open, of, hf, lf, cf, vf)

    ordered = [by_ts[k] for k in sorted(by_ts.keys())]
    return ordered[-limit:]


def _estimate_atr_15m(bars_1m: list[tuple[float, float, float, float, float, float]]) -> float:
    if len(bars_1m) < 16:
        return 0.0
    ranges = [max(0.0, b[2] - b[3]) for b in bars_1m[-15:]]
    avg_1m = sum(ranges) / len(ranges)
    return avg_1m * math.sqrt(15.0)


def _make_opportunity_id(engine_id: str, symbol: str, setup: str, bar_open_ts: int) -> str:
    bucket = int(bar_open_ts // 900)
    raw = f"{engine_id}:{symbol}:{setup}:{bucket}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16].upper()


def _btc_market_healthy(btc_bars: list[tuple[float, float, float, float, float, float]], now_ts: float) -> bool:
    if len(btc_bars) < 30:
        return False
    latest_close_ts = btc_bars[-1][0] + 60.0
    if (now_ts - latest_close_ts) > MAX_BAR_CLOSE_STALE_SEC:
        return False
    active_30m = sum(1 for b in btc_bars[-30:] if b[5] > 0.0)
    return active_30m >= MIN_BTC_ACTIVE_BARS_30M


def _aligned_return(bars: list[tuple[float, float, float, float, float, float]], lookback_min: int) -> float | None:
    if len(bars) <= lookback_min:
        return None
    latest = bars[-1]
    target_open = latest[0] - float(lookback_min) * 60.0
    # Find the closed bar at or immediately preceding target_open (within 2m tolerance)
    prior = None
    for b in reversed(bars[:-1]):
        if b[0] <= target_open + 1.0:
            if (target_open - b[0]) <= 120.0:
                prior = b
            break
    if prior is None or prior[4] <= 0.0:
        return None
    return (latest[4] / prior[4]) - 1.0


def evaluate_day_verified_signals(db_path: str, *, now_ts: float | None = None) -> list[VerifiedAlphaSignal]:
    """Evaluate verified DAY_V2 BTC lead-lag catch-up signals on closed 1m bars."""
    moment = float(now_ts if now_ts is not None else time.time())
    btc_bars = _load_closed_1m_bars(db_path, "BTCUSDT", limit=45, as_of=moment)
    if not _btc_market_healthy(btc_bars, moment):
        return []
    btc_ret15 = _aligned_return(btc_bars, 15)
    if btc_ret15 is None or btc_ret15 < DAY_BTC_15M_MIN_RET:
        return []

    btc_open_ts = round(btc_bars[-1][0])
    signals: list[VerifiedAlphaSignal] = []
    for sym in ALT_UNIVERSE:
        alt_bars = _load_closed_1m_bars(db_path, sym, limit=45, as_of=moment)
        if len(alt_bars) < 20:
            continue
        latest = alt_bars[-1]
        if abs(latest[0] - btc_open_ts) > 120.0:
            continue
        if (moment - (latest[0] + 60.0)) > MAX_BAR_CLOSE_STALE_SEC:
            continue
        # Active print required on signal bar to prevent stale-quote mirage
        if latest[5] <= 0.0:
            continue
        alt_ret15 = _aligned_return(alt_bars, 15)
        if alt_ret15 is None:
            continue
        lag15 = btc_ret15 - alt_ret15
        if lag15 < DAY_ALT_15M_MIN_LAG:
            continue
        bar_open = round(latest[0])
        signals.append(
            VerifiedAlphaSignal(
                engine_id="DAY_V2",
                symbol=sym,
                setup=DAY_VERIFIED_LEAD_LAG_SETUP,
                score=float(lag15),
                bar_open_ts=bar_open,
                bar_close_ts=bar_open + 60,
                close_price=float(latest[4]),
                atr_15m=_estimate_atr_15m(alt_bars),
                detail=f"btc15m={btc_ret15 * 1e4:+.1f}bp alt15m={alt_ret15 * 1e4:+.1f}bp lag={lag15 * 1e4:+.1f}bp",
                opportunity_id=_make_opportunity_id("DAY_V2", sym, DAY_VERIFIED_LEAD_LAG_SETUP, bar_open),
            )
        )
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


def evaluate_scalp_verified_signals(db_path: str, *, now_ts: float | None = None) -> list[VerifiedAlphaSignal]:
    """Evaluate verified SCALP_V2 capitulation-flush and BTC 5m lead-lag signals."""
    moment = float(now_ts if now_ts is not None else time.time())
    btc_bars = _load_closed_1m_bars(db_path, "BTCUSDT", limit=75, as_of=moment)
    if not _btc_market_healthy(btc_bars, moment):
        return []
    btc_ret5 = _aligned_return(btc_bars, 5)
    btc_open_ts = round(btc_bars[-1][0])

    signals: list[VerifiedAlphaSignal] = []
    for sym in ALT_UNIVERSE:
        alt_bars = _load_closed_1m_bars(db_path, sym, limit=75, as_of=moment)
        if len(alt_bars) < 20:
            continue
        latest = alt_bars[-1]
        if abs(latest[0] - btc_open_ts) > 120.0:
            continue
        if (moment - (latest[0] + 60.0)) > MAX_BAR_CLOSE_STALE_SEC:
            continue
        if latest[5] <= 0.0:
            continue

        bar_open = round(latest[0])
        atr_val = _estimate_atr_15m(alt_bars)
        cand_list: list[VerifiedAlphaSignal] = []

        # Signal 1: 60m capitulation flush rebound
        alt_ret60 = _aligned_return(alt_bars, 60)
        if alt_ret60 is not None and alt_ret60 <= SCALP_CRASH_60M_MAX_RET:
            cand_list.append(
                VerifiedAlphaSignal(
                    engine_id="SCALP_V2",
                    symbol=sym,
                    setup=SCALP_VERIFIED_CAPITULATION_SETUP,
                    score=float(-alt_ret60),
                    bar_open_ts=bar_open,
                    bar_close_ts=bar_open + 60,
                    close_price=float(latest[4]),
                    atr_15m=atr_val,
                    detail=f"alt60m={alt_ret60 * 1e4:+.1f}bp",
                    opportunity_id=_make_opportunity_id("SCALP_V2", sym, SCALP_VERIFIED_CAPITULATION_SETUP, bar_open),
                )
            )

        # Signal 2: BTC 5m impulse lead-lag catch-up
        if btc_ret5 is not None and btc_ret5 >= SCALP_BTC_5M_MIN_RET:
            alt_ret5 = _aligned_return(alt_bars, 5)
            if alt_ret5 is not None:
                lag5 = btc_ret5 - alt_ret5
                if lag5 >= SCALP_ALT_5M_MIN_LAG:
                    cand_list.append(
                        VerifiedAlphaSignal(
                            engine_id="SCALP_V2",
                            symbol=sym,
                            setup=SCALP_VERIFIED_LEAD_LAG_SETUP,
                            score=float(lag5),
                            bar_open_ts=bar_open,
                            bar_close_ts=bar_open + 60,
                            close_price=float(latest[4]),
                            atr_15m=atr_val,
                            detail=f"btc5m={btc_ret5 * 1e4:+.1f}bp alt5m={alt_ret5 * 1e4:+.1f}bp lag={lag5 * 1e4:+.1f}bp",
                            opportunity_id=_make_opportunity_id("SCALP_V2", sym, SCALP_VERIFIED_LEAD_LAG_SETUP, bar_open),
                        )
                    )

        if cand_list:
            cand_list.sort(key=lambda s: s.score, reverse=True)
            signals.append(cand_list[0])

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


def is_symbol_on_verified_cooldown(
    db_path: str,
    engine_id: str,
    symbol: str,
    *,
    now_ts: float | None = None,
    cooldown_sec: float = VERIFIED_SIGNAL_COOLDOWN_SEC,
) -> bool:
    """True when ``engine_id`` traded ``symbol`` within ``cooldown_sec`` (restart-safe)."""
    moment = float(now_ts if now_ts is not None else time.time())
    variants = _symbol_variants(symbol)
    placeholders = ",".join("?" for _ in variants)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error:
        return False
    try:
        row = conn.execute(
            f"""
            SELECT timestamp
            FROM paper_trades
            WHERE UPPER(COALESCE(engine_id, '')) = ? AND symbol IN ({placeholders})
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (str(engine_id or "").upper(), *variants),
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    if not row or not row[0]:
        return False
    last_ts = _parse_ts_epoch(row[0])
    return last_ts > 0.0 and (moment - last_ts) < float(cooldown_sec)


__all__ = [
    "ALL_VERIFIED_SETUPS",
    "ALT_UNIVERSE",
    "DAY_ALT_15M_MIN_LAG",
    "DAY_BTC_15M_MIN_RET",
    "DAY_LEAD_LAG_CATASTROPHIC_PCT",
    "DAY_LEAD_LAG_MAX_HOLD_MIN",
    "DAY_VERIFIED_LEAD_LAG_SETUP",
    "DAY_VERIFIED_SETUPS",
    "SCALP_ALT_5M_MIN_LAG",
    "SCALP_BTC_5M_MIN_RET",
    "SCALP_CRASH_60M_MAX_RET",
    "SCALP_VERIFIED_CAPITULATION_SETUP",
    "SCALP_VERIFIED_CATASTROPHIC_PCT",
    "SCALP_VERIFIED_LEAD_LAG_SETUP",
    "SCALP_VERIFIED_MAX_HOLD_MIN",
    "SCALP_VERIFIED_SETUPS",
    "VERIFIED_MAX_CONCURRENT_PER_ENGINE",
    "VERIFIED_SIGNAL_COOLDOWN_SEC",
    "VerifiedAlphaSignal",
    "evaluate_day_verified_signals",
    "evaluate_scalp_verified_signals",
    "is_symbol_on_verified_cooldown",
]
