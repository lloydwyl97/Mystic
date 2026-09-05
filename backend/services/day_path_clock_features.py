"""Point-in-time clock-aligned research features. Offline only.

Never participates in live ranking. Missing values stay None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.config.execution_cost_model import (
    expected_exchange_commission_rt_pct,
    expected_slippage_rt_pct,
    expected_spread_pct,
    honest_all_in_rt_pct,
)
from backend.services.day_4h_entry_features import HOLD_SYMBOL
from backend.services.day_path_clock_v2 import CLOCK_LOOKBACKS_SEC, SCHEMA_VERSION, SOURCE_INTERVAL_SEC
from backend.services.day_path_input_validity import MAX_GAP_SEC, MAX_LAST_BAR_AGE_SEC, parse_bar_ts


@dataclass(frozen=True)
class ClockBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def parse_as_of(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return _as_utc(raw)
    return parse_bar_ts(raw)


def normalize_bars(raw_bars: Any) -> list[ClockBar]:
    out: list[ClockBar] = []
    if not isinstance(raw_bars, list):
        return out
    for item in raw_bars:
        if isinstance(item, ClockBar):
            out.append(item)
            continue
        if isinstance(item, dict):
            ts = parse_bar_ts(item.get("ts") if item.get("ts") is not None else item.get("timestamp"))
            try:
                close = float(item.get("close") or 0.0)
                high = float(item.get("high") if item.get("high") is not None else close)
                low = float(item.get("low") if item.get("low") is not None else close)
                opn = float(item.get("open") if item.get("open") is not None else close)
                vol = float(item.get("volume") or 0.0)
            except (TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 5:
            ts = parse_bar_ts(item[0])
            try:
                opn = float(item[1])
                high = float(item[2])
                low = float(item[3])
                close = float(item[4])
                vol = float(item[5]) if len(item) > 5 else 0.0
            except (TypeError, ValueError):
                continue
        else:
            continue
        if ts is None or close <= 0 or high <= 0 or low <= 0:
            continue
        out.append(ClockBar(ts=_as_utc(ts), open=opn, high=high, low=low, close=close, volume=vol))
    out.sort(key=lambda b: b.ts)
    return out


def clip_asof(bars: list[ClockBar], as_of: datetime) -> list[ClockBar]:
    cutoff = _as_utc(as_of)
    return [b for b in bars if b.ts <= cutoff]


def window_quality(bars: list[ClockBar], *, start: datetime, end: datetime) -> dict[str, Any]:
    inside = [b for b in bars if start <= b.ts <= end]
    gaps = [(inside[i].ts - inside[i - 1].ts).total_seconds() for i in range(1, len(inside))]
    last_age = (end - inside[-1].ts).total_seconds() if inside else None
    return {
        "row_count": len(inside),
        "max_gap_seconds": max(gaps) if gaps else None,
        "latest_bar_age_seconds": last_age,
        "valid": bool(inside and last_age is not None and last_age <= MAX_LAST_BAR_AGE_SEC and (not gaps or max(gaps) <= MAX_GAP_SEC)),
    }


def close_at_or_before(bars: list[ClockBar], when: datetime, *, max_age_sec: float = MAX_LAST_BAR_AGE_SEC) -> ClockBar | None:
    cutoff = _as_utc(when)
    chosen: ClockBar | None = None
    for bar in bars:
        if bar.ts <= cutoff:
            chosen = bar
        else:
            break
    if chosen is None:
        return None
    if (cutoff - chosen.ts).total_seconds() > max_age_sec:
        return None
    return chosen


def clock_return(bars: list[ClockBar], as_of: datetime, lookback_sec: int) -> float | None:
    end = close_at_or_before(bars, as_of)
    start = close_at_or_before(bars, _as_utc(as_of) - timedelta(seconds=lookback_sec))
    if end is None or start is None or start.close <= 0:
        return None
    if end.ts <= start.ts:
        return None
    return (end.close - start.close) / start.close


def realized_vol(bars: list[ClockBar], as_of: datetime, lookback_sec: int, *, min_obs: int = 5) -> float | None:
    end = _as_utc(as_of)
    start = end - timedelta(seconds=lookback_sec)
    inside = [b for b in bars if start <= b.ts <= end]
    if len(inside) < min_obs + 1:
        return None
    quality = window_quality(inside, start=start, end=end)
    if not quality["valid"]:
        return None
    rets: list[float] = []
    for i in range(1, len(inside)):
        prev = inside[i - 1].close
        if prev <= 0:
            continue
        rets.append((inside[i].close - prev) / prev)
    if len(rets) < min_obs:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / len(rets)
    return math.sqrt(var)


def drawdown_rebound(bars: list[ClockBar], as_of: datetime, lookback_sec: int) -> tuple[float | None, float | None]:
    end = close_at_or_before(bars, as_of)
    start = close_at_or_before(bars, _as_utc(as_of) - timedelta(seconds=lookback_sec))
    if end is None or start is None or start.close <= 0:
        return None, None
    inside = [b for b in bars if start.ts <= b.ts <= end.ts]
    if len(inside) < 3:
        return None, None
    trough = min(b.low for b in inside)
    if trough <= 0:
        return None, None
    drawdown = (trough - start.close) / start.close
    rebound = (end.close - trough) / trough
    return drawdown, rebound


def rel_volume(bars: list[ClockBar], as_of: datetime, lookback_sec: int) -> float | None:
    end = _as_utc(as_of)
    recent_start = end - timedelta(seconds=lookback_sec)
    baseline_start = end - timedelta(seconds=lookback_sec * 4)
    recent = [b for b in bars if recent_start <= b.ts <= end]
    baseline = [b for b in bars if baseline_start <= b.ts < recent_start]
    if len(recent) < 3 or len(baseline) < 3:
        return None
    if not window_quality(recent, start=recent_start, end=end)["valid"]:
        return None
    recent_vol = sum(b.volume for b in recent)
    base_vol = sum(b.volume for b in baseline) / 3.0
    if base_vol <= 0:
        return None
    return recent_vol / base_vol


def hold_clock_features() -> dict[str, Any]:
    feats = dict.fromkeys(CLOCK_LOOKBACKS_SEC)
    feats.update(
        {
            "schema_version": SCHEMA_VERSION,
            "symbol": HOLD_SYMBOL,
            "feature_available": False,
            "p_buy": None,
            "legacy_path_ev": 0.0,
            "final_rank_score": 0.0,
            "production_4h_break_true_at_decision": None,
            "distance_to_4h_break_bps": None,
            "4h_range_position": None,
            "4h_alignment_state": None,
            "spread_bps": 0.0,
            "expected_slippage_bps": 0.0,
            "estimated_all_in_cost_bps": 0.0,
            "commission_rt_bps": 0.0,
        }
    )
    return feats


def cost_fields(symbol: str, *, quote_spread_bps: float | None = None) -> dict[str, float]:
    spread = float(quote_spread_bps) if quote_spread_bps is not None else expected_spread_pct(symbol) * 1e4
    slip = expected_slippage_rt_pct() * 1e4
    commission = expected_exchange_commission_rt_pct() * 1e4
    return {
        "spread_bps": spread,
        "expected_slippage_bps": slip,
        "commission_rt_bps": commission,
        "estimated_all_in_cost_bps": honest_all_in_rt_pct(symbol) * 1e4 if quote_spread_bps is None else commission + spread + slip,
    }


def build_clock_features(
    bars: Any,
    *,
    as_of: Any,
    symbol: str,
    btc_bars: Any = None,
    p_buy: float | None = None,
    legacy_path_ev: float | None = None,
    final_rank_score: float | None = None,
    structure: dict[str, Any] | None = None,
    quote_spread_bps: float | None = None,
) -> dict[str, Any]:
    if str(symbol or "").upper() == HOLD_SYMBOL:
        return hold_clock_features()
    when = parse_as_of(as_of)
    if when is None:
        out = hold_clock_features()
        out["symbol"] = str(symbol or "")
        return out
    clipped = clip_asof(normalize_bars(bars), when)
    btc = clip_asof(normalize_bars(btc_bars), when) if btc_bars is not None else []
    ret_5m = clock_return(clipped, when, CLOCK_LOOKBACKS_SEC["ret_5m"])
    ret_15m = clock_return(clipped, when, CLOCK_LOOKBACKS_SEC["ret_15m"])
    ret_30m = clock_return(clipped, when, CLOCK_LOOKBACKS_SEC["ret_30m"])
    ret_1h = clock_return(clipped, when, CLOCK_LOOKBACKS_SEC["ret_1h"])
    btc_ret_5m = clock_return(btc, when, CLOCK_LOOKBACKS_SEC["btc_rel_ret_5m"]) if btc else None
    dd, rebound = drawdown_rebound(clipped, when, CLOCK_LOOKBACKS_SEC["drawdown_30m"])
    struct = dict(structure or {})
    costs = cost_fields(symbol, quote_spread_bps=quote_spread_bps)
    available = ret_5m is not None and ret_15m is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": str(symbol or ""),
        "source_interval_seconds": SOURCE_INTERVAL_SEC,
        "decision_timestamp": when.isoformat(),
        "feature_available": available,
        "ret_5m": ret_5m,
        "ret_15m": ret_15m,
        "ret_30m": ret_30m,
        "ret_1h": ret_1h,
        "realized_vol_10m": realized_vol(clipped, when, CLOCK_LOOKBACKS_SEC["realized_vol_10m"]),
        "drawdown_30m": dd,
        "rebound_30m": rebound,
        "rel_volume_15m": rel_volume(clipped, when, CLOCK_LOOKBACKS_SEC["rel_volume_15m"]),
        "btc_ret_5m": btc_ret_5m,
        "btc_rel_ret_5m": (ret_5m - btc_ret_5m) if ret_5m is not None and btc_ret_5m is not None else None,
        "p_buy": p_buy,
        "legacy_path_ev": legacy_path_ev,
        "final_rank_score": final_rank_score,
        "production_4h_break_true_at_decision": struct.get("production_4h_break_true_at_decision", struct.get("production_4h_break_true_now")),
        "distance_to_4h_break_bps": struct.get("distance_to_4h_break_bps"),
        "4h_range_position": struct.get("4h_range_position"),
        "4h_alignment_state": struct.get("4h_alignment_state"),
        **costs,
    }
