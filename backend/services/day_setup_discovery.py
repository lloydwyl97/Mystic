"""DAY setup discovery: early continuation vs structured pullback.

Does not use 24h green/red. Does not submit orders.
"""

from __future__ import annotations

from typing import Any

from backend.config.day_setup_discovery import (
    BREAK_CONFIRM_BARS,
    CONSOLIDATION_BARS,
    COST_PULLBACK_MULT,
    EARLY_ROOM_ATR_MULT,
    EARLY_TREND,
    EARLY_TREND_MIN_BARS,
    EXTENDED_1H_HIGH_ATR,
    EXTENDED_4H_RANGE_PCT,
    MAX_INTENT_AGE_SEC,
    PULLBACK_ATR_MULT,
    RECLAIM_ATR_MULT,
    REJECT_EXTENDED,
    REJECT_NO_SETUP,
    STRUCTURE_LOOKBACK_MINUTES,
    STRUCTURED_PULLBACK,
    VOLUME_CONFIRM_MULT,
    setup_discovery_route,
    setup_validity_enforced,
)
from backend.config.execution_cost_model import honest_all_in_rt_pct


def _api(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def honest_round_trip_cost_bps(symbol: str) -> float:
    return float(honest_all_in_rt_pct(symbol)) * 10000.0


Bar = tuple[int, float, float, float, float, float]


def _bars_asof(bars: list[Bar], ts: int) -> list[Bar]:
    return [b for b in bars if int(b[0]) <= int(ts)]


def _window(bars: list[Bar], ts: int, minutes: int) -> list[Bar]:
    start = int(ts) - int(minutes) * 60
    return [b for b in bars if start <= int(b[0]) <= int(ts)]


def _ret(bars: list[Bar], ts: int, minutes: int) -> float | None:
    w = _window(bars, ts, minutes)
    if len(w) < 2 or w[0][4] <= 0:
        return None
    return float(w[-1][4]) / float(w[0][4]) - 1.0


def _hh_ll(bars: list[Bar]) -> tuple[float, float]:
    if not bars:
        return 0.0, 0.0
    return max(b[2] for b in bars), min(b[3] for b in bars)


def market_structure(bars: list[Bar], *, ts: int, atr: float, ask: float) -> dict[str, float]:
    asof = _bars_asof(bars, ts)
    px = float(ask or (asof[-1][4] if asof else 0.0) or 0.0)
    atr_abs = max(float(atr or 0.0), px * 0.004 if px else 0.0)
    m15 = _window(asof, ts, 15)
    m60 = _window(asof, ts, 60)
    m240 = _window(asof, ts, 240)
    h15, l15 = _hh_ll(m15)
    h60, l60 = _hh_ll(m60)
    h240, l240 = _hh_ll(m240)
    rng4 = max(h240 - l240, atr_abs, 1e-12)
    prior = asof[:-BREAK_CONFIRM_BARS] if len(asof) > BREAK_CONFIRM_BARS else asof
    prior_h240, _prior_l = _hh_ll(_window(prior, ts - BREAK_CONFIRM_BARS * 60, 240) or prior[-240:])
    return {
        "price": px,
        "atr": atr_abs,
        "atr_bps": (atr_abs / px) * 1e4 if px else 0.0,
        "ret_5": float(_ret(asof, ts, 5) or 0.0),
        "ret_15": float(_ret(asof, ts, 15) or 0.0),
        "ret_30": float(_ret(asof, ts, 30) or 0.0),
        "ret_60": float(_ret(asof, ts, 60) or 0.0),
        "ret_240": float(_ret(asof, ts, 240) or 0.0),
        "dist_low_15_bps": ((px / l15) - 1.0) * 1e4 if l15 else 0.0,
        "dist_low_60_bps": ((px / l60) - 1.0) * 1e4 if l60 else 0.0,
        "dist_low_240_bps": ((px / l240) - 1.0) * 1e4 if l240 else 0.0,
        "dist_high_15_bps": ((h15 / px) - 1.0) * 1e4 if h15 and px else 0.0,
        "dist_high_60_bps": ((h60 / px) - 1.0) * 1e4 if h60 and px else 0.0,
        "dist_high_240_bps": ((h240 / px) - 1.0) * 1e4 if h240 and px else 0.0,
        "range_4h_pct": (px - l240) / rng4 if l240 else 0.0,
        "high_15": h15,
        "low_15": l15,
        "high_60": h60,
        "low_60": l60,
        "high_240": h240,
        "low_240": l240,
        "ext_atr_1h": ((h60 - px) / atr_abs) if atr_abs else 0.0,
        "ext_atr_from_4h_low": ((px - l240) / atr_abs) if atr_abs else 0.0,
        "prior_high_240": prior_h240,
        "dist_prior_high_240_bps": ((prior_h240 / px) - 1.0) * 1e4 if prior_h240 and px else 0.0,
    }


def is_extended(ms: dict[str, float]) -> bool:
    """True when price is already in the top of the 4h range (24h-high chase)."""
    if float(ms.get("range_4h_pct") or 0.0) >= EXTENDED_4H_RANGE_PCT:
        return True
    atr_bps = float(ms.get("atr_bps") or 0.0)
    prior = float(ms.get("dist_prior_high_240_bps") or 1e9)
    return prior <= EXTENDED_1H_HIGH_ATR * atr_bps and float(ms.get("range_4h_pct") or 0.0) >= 0.70


def structure_intact(ms: dict[str, float]) -> bool:
    px = float(ms.get("price") or 0.0)
    low4 = float(ms.get("low_240") or 0.0)
    if px <= 0 or low4 <= 0:
        return False
    if px <= low4:
        return False
    return float(ms.get("ret_60") or 0.0) >= 0.0 or float(ms.get("ret_240") or 0.0) > 0.0


def pullback_bps_required(symbol: str, atr: float, price: float) -> float:
    atr_bps = (float(atr) / float(price)) * 1e4 if price else 0.0
    cost = honest_round_trip_cost_bps(symbol)
    return max(PULLBACK_ATR_MULT * atr_bps, COST_PULLBACK_MULT * cost)


def reclaim_bps_required(symbol: str, atr: float, price: float) -> float:
    atr_bps = (float(atr) / float(price)) * 1e4 if price else 0.0
    cost = honest_round_trip_cost_bps(symbol)
    return max(RECLAIM_ATR_MULT * atr_bps, cost)


def _range_break(asof: list[Bar], *, ts: int, atr: float, ask: float) -> dict[str, Any]:
    need = EARLY_TREND_MIN_BARS
    if len(asof) < need:
        return {"ok": False, "reason": "INSUFFICIENT_BARS"}
    consol = asof[-(CONSOLIDATION_BARS + BREAK_CONFIRM_BARS) : -BREAK_CONFIRM_BARS]
    recent = asof[-BREAK_CONFIRM_BARS:]
    ch, cl = _hh_ll(consol)
    if ch <= cl:
        return {"ok": False, "reason": "NO_RANGE"}
    if ask <= ch:
        return {"ok": False, "reason": "NO_BREAK", "range_high": ch, "range_low": cl}
    rising = sum(1 for i in range(1, len(recent)) if recent[i][4] > recent[i - 1][4] and recent[i][3] >= recent[i - 1][3])
    if rising < 2:
        return {"ok": False, "reason": "NO_RISING_STRUCTURE", "range_high": ch, "range_low": cl}
    cvol = [b[5] for b in consol if b[5] > 0]
    rvol = [b[5] for b in recent if b[5] > 0]
    vol_ok = True
    if cvol and rvol:
        mid = sorted(cvol)[len(cvol) // 2]
        vol_ok = (sum(rvol) / len(rvol)) >= VOLUME_CONFIRM_MULT * mid if mid > 0 else True
    if not vol_ok:
        return {"ok": False, "reason": "VOLUME_NOT_CONFIRMING", "range_high": ch, "range_low": cl}
    room = ch and atr > 0
    _ = room
    return {"ok": True, "reason": "RANGE_BREAK", "range_high": ch, "range_low": cl}


def classify_setup(
    bars: list[Bar],
    *,
    symbol: str,
    ts: int,
    atr: float,
    ask: float,
) -> dict[str, Any]:
    asof = _bars_asof(bars, ts)
    ms = market_structure(asof, ts=ts, atr=atr, ask=ask)
    px = float(ms["price"])
    atr_abs = float(ms["atr"])
    cost = honest_all_in_rt_pct(symbol)
    out = {
        "symbol": _api(symbol),
        "setup_class": REJECT_NO_SETUP,
        "reason": "NO_SETUP",
        "shadow_only": True,
        **ms,
        "cost_rt": cost,
        "pullback_bps_req": pullback_bps_required(symbol, atr_abs, px),
        "reclaim_bps_req": reclaim_bps_required(symbol, atr_abs, px),
        "asof_bars": len(asof),
        "early_trend_need_bars": EARLY_TREND_MIN_BARS,
        "structure_need_minutes": STRUCTURE_LOOKBACK_MINUTES,
    }
    if px <= 0 or atr_abs <= 0:
        out["reason"] = "NO_PRICE_OR_ATR"
        return out
    if is_extended(ms):
        out["setup_class"] = REJECT_EXTENDED
        out["reason"] = "EXCESSIVELY_EXTENDED"
        return out
    prior_room = float(ms.get("dist_prior_high_240_bps") or ms["dist_high_240_bps"])
    room_ok = prior_room >= EARLY_ROOM_ATR_MULT * float(ms["atr_bps"])
    br = _range_break(asof, ts=ts, atr=atr_abs, ask=px)
    if br.get("ok") and room_ok and float(ms["ret_15"]) > 0 and cost < max(prior_room, 0.0) / 1e4:
        out["setup_class"] = EARLY_TREND
        out["reason"] = str(br.get("reason") or "RANGE_BREAK")
        out["range_high"] = float(br.get("range_high") or 0.0)
        return out
    pull_req = float(out["pullback_bps_req"])
    pulled = float(ms["dist_high_60_bps"]) >= pull_req
    if pulled and structure_intact(ms) and float(ms["dist_low_15_bps"]) <= float(out["reclaim_bps_req"]) * 1.5:
        # Near the local 15m low after a real 1h pullback, structure intact.
        out["setup_class"] = STRUCTURED_PULLBACK
        out["reason"] = "ATR_PULLBACK"
        return out
    if pulled and structure_intact(ms):
        out["setup_class"] = STRUCTURED_PULLBACK
        out["reason"] = "ATR_PULLBACK_PENDING_RECLAIM"
        return out
    out["reason"] = str(br.get("reason") or "NO_BREAK_OR_PULLBACK")
    return out


def intent_invalid_reason(
    intent: dict[str, Any],
    *,
    ask: float,
    now: float,
    bars: list[Bar] | None = None,
    source_setup_valid: bool = True,
    better_candidate: bool = False,
) -> str:
    arm_ts = float(intent.get("arm_ts") or intent.get("created_at") or 0.0)
    if arm_ts and now - arm_ts > float(MAX_INTENT_AGE_SEC):
        return "STALE_INTENT_MAX_AGE"
    if not source_setup_valid:
        return "SOURCE_SETUP_INVALID"
    if better_candidate:
        return "REPLACED_BY_BETTER_CANDIDATE"
    thesis = float(intent.get("thesis_invalid_level") or 0.0)
    if thesis > 0 and ask > 0 and ask <= thesis:
        return "STRUCTURE_BROKEN"
    atr = float(intent.get("atr") or 0.0)
    symbol = str(intent.get("symbol") or "")
    if bars:
        ms = market_structure(bars, ts=int(now), atr=atr, ask=ask)
        px = float(ms.get("price") or 0.0)
        low4 = float(ms.get("low_240") or 0.0)
        if px > 0 and low4 > 0 and px <= low4:
            return "STRUCTURE_BROKEN"
        ev = float(intent.get("predicted_ev") or 0.0)
        if ev > 0 and ev <= honest_all_in_rt_pct(symbol):
            return "POST_COST_EDGE_GONE"
    return ""


def live_intent_validity(
    intent: dict[str, Any],
    *,
    ask: float,
    now: float,
    bars: list[Bar] | None = None,
    source_setup_valid: bool = True,
    better_candidate: bool = False,
) -> str:
    reason = intent_invalid_reason(
        intent,
        ask=ask,
        now=now,
        bars=bars,
        source_setup_valid=source_setup_valid,
        better_candidate=better_candidate,
    )
    if reason and setup_validity_enforced():
        return reason
    return ""


def may_arm_setup(classification: dict[str, Any]) -> bool:
    if not setup_discovery_route():
        return True
    return str(classification.get("setup_class") or "") in {EARLY_TREND, STRUCTURED_PULLBACK}


def structured_min_dip_bps(symbol: str, atr: float, price: float, rebound_bps: float) -> float:
    return pullback_bps_required(symbol, atr, price) + max(float(rebound_bps or 0.0), reclaim_bps_required(symbol, atr, price))
