"""
Trade thesis classification for DAY ranking/explainability and position management.

Uses existing Mystic signal/context fields only (no new gates, no .pkl changes).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL

# Canonical DAY exit labels (reporting / paper_trades.exit_reason)
EXIT_NET_PROFIT = "NET_PROFIT_EXIT"
EXIT_EXTREME_PROTECTION = "EXTREME_PROTECTION_EXIT"
EXIT_MANUAL = "MANUAL_EXIT"
EXIT_ADMIN_CLEAR = "ADMIN_CLEAR"
EXIT_LEGACY_INVENTORY_CLEANUP = "LEGACY_INVENTORY_CLEANUP_EXIT"
EXIT_THESIS_WARNING = "THESIS_INVALIDATION_WARNING_ONLY"
EXIT_THESIS_INVALIDATION = "THESIS_INVALIDATION_EXIT"
EXIT_STOP_LOSS = "STOP_LOSS_EXIT"
EXIT_TRAILING_STOP = "TRAILING_STOP_EXIT"
EXIT_DAY_4H_STRUCTURE_BREAK = "DAY_4H_STRUCTURE_BREAK_EXIT"
EXIT_DAY_RISK_FLOOR = "DAY_RISK_FLOOR_EXIT"
EXIT_EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"
EXIT_RESTART_FLATTEN = "RESTART_FLATTEN"
DAY_4H_BUNDLE_MISSING = "DAY_4H_BUNDLE_MISSING"

SETUP_HTF_TREND_PULLBACK = "HTF_TREND_PULLBACK"
SETUP_VWAP_REVERSION = "VWAP_REVERSION"
SETUP_BREAKOUT_CONTINUATION = "BREAKOUT_CONTINUATION"
SETUP_NO_CLEAR_THESIS = "NO_CLEAR_THESIS"

# Active production reversal / range setups (paper + future live use same rules).
# These were activated to generate trades and learnable outcomes in non-bull regimes.
# Conservative sizing and ATR brackets apply; promotion still requires replay proof.
SETUP_FAILED_BREAKDOWN_REVERSAL = "FAILED_BREAKDOWN_REVERSAL"
SETUP_RANGE_BOUNCE = "RANGE_BOUNCE"

# Research-only discovery setups (kept for historical replay compatibility).
RESEARCH_FAILED_BREAKDOWN_REVERSAL = "RESEARCH_FAILED_BREAKDOWN_REVERSAL"
RESEARCH_15M_30M_RECLAIM = "RESEARCH_15M_30M_RECLAIM"
RESEARCH_VOLATILITY_EXPANSION = "RESEARCH_VOLATILITY_EXPANSION"
RESEARCH_RANGE_RECLAIM = "RESEARCH_RANGE_RECLAIM"
RESEARCH_TREND_RETEST = "RESEARCH_TREND_RETEST"
RESEARCH_CAPITULATION_BOUNCE = "RESEARCH_CAPITULATION_BOUNCE"
RESEARCH_SHORT_BEAR_CONTINUATION = "RESEARCH_SHORT_BEAR_CONTINUATION"

ALL_SETUP_TYPES = (
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_NO_CLEAR_THESIS,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_RANGE_BOUNCE,
    RESEARCH_FAILED_BREAKDOWN_REVERSAL,
    RESEARCH_15M_30M_RECLAIM,
    RESEARCH_VOLATILITY_EXPANSION,
    RESEARCH_RANGE_RECLAIM,
    RESEARCH_TREND_RETEST,
    RESEARCH_CAPITULATION_BOUNCE,
    RESEARCH_SHORT_BEAR_CONTINUATION,
)

HTF_TFS = ("15m", "30m", "1h", "4h")
LTF_TFS = ("1m", "5m")
BREAKOUT_ALT_SYMBOLS = frozenset({"SOLUSDT", "XRPUSDT", "DOGEUSDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"})

_RANK_DELTA = {
    # Profit bias (Ocean evidence): RANGE/BREAKOUT win; FBR is the bleed bucket.
    SETUP_HTF_TREND_PULLBACK: 0.04,
    SETUP_VWAP_REVERSION: -0.02,
    SETUP_BREAKOUT_CONTINUATION: 0.06,
    SETUP_NO_CLEAR_THESIS: -0.14,
    SETUP_FAILED_BREAKDOWN_REVERSAL: -0.22,  # demote — do not mint FBR for "activity"
    SETUP_RANGE_BOUNCE: 0.08,
    RESEARCH_FAILED_BREAKDOWN_REVERSAL: 0.0,
    RESEARCH_15M_30M_RECLAIM: 0.0,
    RESEARCH_VOLATILITY_EXPANSION: 0.0,
    RESEARCH_RANGE_RECLAIM: 0.0,
    RESEARCH_TREND_RETEST: 0.0,
    RESEARCH_CAPITULATION_BOUNCE: 0.0,
    RESEARCH_SHORT_BEAR_CONTINUATION: 0.0,
}
_RANK_DELTA_SCALP = {
    SETUP_HTF_TREND_PULLBACK: 0.02,
    SETUP_VWAP_REVERSION: 0.06,
    SETUP_BREAKOUT_CONTINUATION: 0.04,
    SETUP_NO_CLEAR_THESIS: -0.14,
    SETUP_FAILED_BREAKDOWN_REVERSAL: 0.02,
    SETUP_RANGE_BOUNCE: 0.01,
    RESEARCH_FAILED_BREAKDOWN_REVERSAL: 0.0,
    RESEARCH_15M_30M_RECLAIM: 0.0,
    RESEARCH_VOLATILITY_EXPANSION: 0.0,
    RESEARCH_RANGE_RECLAIM: 0.0,
    RESEARCH_TREND_RETEST: 0.0,
    RESEARCH_CAPITULATION_BOUNCE: 0.0,
    RESEARCH_SHORT_BEAR_CONTINUATION: 0.0,
}
_EV_FACTOR = {
    SETUP_HTF_TREND_PULLBACK: 1.04,
    SETUP_VWAP_REVERSION: 0.92,
    SETUP_BREAKOUT_CONTINUATION: 1.06,
    SETUP_NO_CLEAR_THESIS: 0.50,
    SETUP_FAILED_BREAKDOWN_REVERSAL: 0.55,
    SETUP_RANGE_BOUNCE: 1.08,
}
_EV_FACTOR_SCALP = {
    SETUP_HTF_TREND_PULLBACK: 1.0,
    SETUP_VWAP_REVERSION: 1.05,
    SETUP_BREAKOUT_CONTINUATION: 1.02,
    SETUP_NO_CLEAR_THESIS: 0.50,
    SETUP_FAILED_BREAKDOWN_REVERSAL: 0.96,
    SETUP_RANGE_BOUNCE: 0.94,
}
_SIZE_FACTOR = {
    SETUP_HTF_TREND_PULLBACK: 1.0,
    SETUP_VWAP_REVERSION: 0.65,
    SETUP_BREAKOUT_CONTINUATION: 0.88,
    SETUP_NO_CLEAR_THESIS: 0.22,
    SETUP_FAILED_BREAKDOWN_REVERSAL: 0.60,  # conservative in reversal (smaller size)
    SETUP_RANGE_BOUNCE: 0.65,
}
_SIZE_FACTOR_SCALP = {
    SETUP_HTF_TREND_PULLBACK: 0.85,
    SETUP_VWAP_REVERSION: 1.0,
    SETUP_BREAKOUT_CONTINUATION: 0.90,
    SETUP_NO_CLEAR_THESIS: 0.22,
    SETUP_FAILED_BREAKDOWN_REVERSAL: 0.55,
    SETUP_RANGE_BOUNCE: 0.60,
}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None or str(raw).strip() == "":
            return default
        v = float(raw)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def parse_mtf_json(decision_data: dict[str, Any]) -> dict[str, Any]:
    raw = decision_data.get("mtf_json") or decision_data.get("mtf")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _tf_align(mtf: dict[str, Any], tf: str) -> float:
    snap = mtf.get(tf)
    if not isinstance(snap, dict):
        return 0.5
    return _safe_float(snap.get("ema_align"), _safe_float(snap.get("trend"), 0.5))


def _mean(vals: list[float], default: float = 0.5) -> float:
    if not vals:
        return default
    return sum(vals) / len(vals)


def enrich_decision_data_for_thesis(
    decision_data: dict[str, Any],
    *,
    symbol: str,
    context_payload: dict[str, Any] | None = None,
    price_structure_regime: str = "unknown",
) -> dict[str, Any]:
    """Merge ai_context + feature hints into decision_data for thesis classification."""
    dd = dict(decision_data or {})
    ctx = context_payload or {}
    if ctx:
        if ctx.get("mtf_json") and not dd.get("mtf_json"):
            dd["mtf_json"] = ctx["mtf_json"]
        for key in (
            "ctx_rs_btc",
            "ctx_rs_eth",
            "ctx_relative_volume",
            "relative_volume",
            "ctx_spread_pct",
            "ctx_depth_imbalance",
            "market_regime",
        ):
            if key in ctx and dd.get(key) in (None, "", 0, 0.0):
                dd[key] = ctx[key]
        if dd.get("relative_volume") in (None, "", 0, 0.0):
            dd["relative_volume"] = _safe_float(ctx.get("ctx_relative_volume"), 0.0)
    dd["price_structure_regime"] = price_structure_regime or dd.get("price_structure_regime") or "unknown"

    # Reversal / range activity indicators for AI features and learning (populated for new active setups)
    aw = str(dd.get("allweather_setup") or dd.get("setup_type") or "").upper()
    rsi = _safe_float(dd.get("rsi"), 50.0)
    mom = _safe_float(dd.get("price_momentum"), 0.0)
    adx = _safe_float(dd.get("adx"), 20.0)
    bb = _safe_float(dd.get("bb_position"), 0.5)
    dd["reversal_reclaim"] = 1.0 if ("FAILED_BREAKDOWN" in aw or (adx > 16 and rsi < 42 and mom > -0.01)) else 0.0
    dd["range_bounce"] = 1.0 if ("RANGE_BOUNCE" in aw or (bb <= 0.35 and rsi < 48 and adx < 26)) else 0.0

    sym_bus = (symbol or dd.get("symbol") or "").replace("/", "").upper()
    if sym_bus and not dd.get("vwap"):
        try:
            from backend.config.redis_config import get_redis_client

            r = get_redis_client()
            if r:
                feat = r.hgetall(f"feature:{sym_bus}") or {}
                for k, v in feat.items():
                    kk = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                    vv = v.decode("utf-8") if isinstance(v, bytes) else v
                    if kk in ("vwap", "bb_position", "volume_ratio", "atr") and dd.get(kk) in (None, "", 0, 0.0):
                        dd[kk] = _safe_float(vv, 0.0)
                # feature:* hashes (binance_ws_hydrator) never carry vwap, so
                # without this fallback VWAP_REVERSION was unreachable for DAY
                # (vwap always 0 -> score 0, entry_vwap 0 -> vwap invalidation dead).
                if dd.get("vwap") in (None, "", 0, 0.0):
                    vwap = _session_vwap_from_klines(r, sym_bus)
                    if vwap > 0:
                        dd["vwap"] = vwap
        except Exception:
            pass
    return dd


def _session_vwap_from_klines(r: Any, sym_bus: str) -> float:
    """Session (UTC day) VWAP from cached 1m klines; rolling 4h fallback."""
    try:
        raw = r.get(f"klines:{sym_bus}:1m")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not raw:
            return 0.0
        rows = json.loads(raw)
        if not isinstance(rows, list) or not rows:
            return 0.0
        import time as _time

        day_start = (int(_time.time()) // 86400) * 86400
        session = [x for x in rows if isinstance(x, (list, tuple)) and len(x) >= 6 and float(x[0]) >= day_start]
        if len(session) < 30:
            session = [x for x in rows if isinstance(x, (list, tuple)) and len(x) >= 6][-240:]
        num = den = 0.0
        for b in session:
            tp = (float(b[2]) + float(b[3]) + float(b[4])) / 3.0
            v = float(b[5])
            if v > 0:
                num += tp * v
                den += v
        return num / den if den > 0 else 0.0
    except Exception:
        return 0.0


def classify_buy_thesis(
    decision_data: dict[str, Any],
    *,
    symbol: str = "",
    current_price: float = 0.0,
    atr: float = 0.0,
    strategy_id: str = "day",
    price_structure_regime: str = "unknown",
) -> dict[str, Any]:
    dd = enrich_decision_data_for_thesis(
        decision_data,
        symbol=symbol,
        price_structure_regime=price_structure_regime,
    )
    sid = (strategy_id or "day").strip().lower()
    is_scalp = sid not in ("", "day")

    mtf = parse_mtf_json(dd)
    ema = _safe_float(dd.get("ema_alignment"), 0.5)
    mom = _safe_float(dd.get("price_momentum"), 0.0)
    adx = _safe_float(dd.get("adx"), 20.0)
    rsi = _safe_float(dd.get("rsi"), 50.0)
    bb = _safe_float(dd.get("bb_position"), 0.5)
    vwap = _safe_float(dd.get("vwap"), 0.0)
    rel_vol = _safe_float(dd.get("relative_volume"), _safe_float(dd.get("volume_expansion"), 1.0))
    vol_ratio = _safe_float(dd.get("volume_ratio"), rel_vol)
    rs = _safe_float(dd.get("ctx_rs_btc"), 0.0)
    depth = _safe_float(dd.get("ctx_depth_imbalance"), 0.0)
    ps_regime = str(dd.get("price_structure_regime") or price_structure_regime or "unknown")

    htf_vals = [_tf_align(mtf, tf) for tf in HTF_TFS if isinstance(mtf.get(tf), dict)]
    ltf_vals = [_tf_align(mtf, tf) for tf in LTF_TFS if isinstance(mtf.get(tf), dict)]
    htf_mean = _mean(htf_vals, 0.5)
    ltf_mean = _mean(ltf_vals, 0.5)

    vwap_dist = 0.0
    if vwap > 0 and current_price > 0:
        vwap_dist = (current_price - vwap) / vwap
        if abs(vwap_dist) > 0.08:
            # Stale/mismatched vwap (e.g. old klines cache): don't classify or
            # set levels off a reference that far from the live mark.
            vwap = 0.0
            vwap_dist = 0.0

    atr_pct = (atr / current_price) if current_price > 0 and atr > 0 else 0.01

    htf_pullback = 0.0
    if htf_mean >= 0.58 or (ema >= 0.62 and adx >= 18):
        pullback_depth = max(0.0, min(1.0, htf_mean - ltf_mean + 0.15))
        rsi_pullback = 1.0 if 32 <= rsi <= 58 else (0.55 if rsi < 65 else 0.25)
        adx_ok = min(1.0, adx / 28.0) if adx >= 16 else 0.35
        htf_pullback = 0.32 * htf_mean + 0.28 * pullback_depth + 0.22 * rsi_pullback + 0.18 * adx_ok
        if bb > 0.78:
            htf_pullback *= 0.55

    vwap_rev = 0.0
    if vwap > 0 and vwap_dist < -0.0008:
        vwap_rev = 0.32 * min(1.0, abs(vwap_dist) * 180.0)
        vwap_rev += 0.28 * max(0.0, 1.0 - bb)
        vwap_rev += 0.20 * min(1.0, max(rel_vol, vol_ratio) / 1.8)
        if ps_regime == "range_bound":
            vwap_rev += 0.12
        if adx < 24:
            vwap_rev += 0.08
        if depth > 0.05:
            vwap_rev += 0.05

    breakout = 0.0
    if mom > 0.08 or ema > 0.68 or (bb > 0.72 and rel_vol > 1.2):
        breakout = 0.28 * min(1.0, max(0.0, mom) * 2.5)
        breakout += 0.24 * ema
        breakout += 0.20 * min(1.0, bb)
        breakout += 0.16 * min(1.0, max(rel_vol, vol_ratio) / 1.6)
        if adx >= 20:
            breakout += 0.07
        if rs > 0.15:
            breakout += 0.08 * min(1.0, rs)
        sym_norm = (symbol or "").replace("/", "").upper()
        if sym_norm in BREAKOUT_ALT_SYMBOLS:
            breakout += 0.06

    # New active reversal / bounce scoring (paper mirrors live, conservative)
    failed_rev = 0.0
    # Trigger if AW stamped the reversal or classic bear-reclaim signals
    aw_setup = str(dd.get("allweather_setup") or dd.get("setup_type") or "").upper()
    if SETUP_FAILED_BREAKDOWN_REVERSAL in aw_setup or "FAILED_BREAKDOWN" in aw_setup or "RECL" in aw_setup:
        # base from low rsi + reclaim momentum + not extreme adx
        failed_rev = 0.48
        if rsi < 42:
            failed_rev += 0.18
        if mom > -0.005:
            failed_rev += 0.12
        if adx >= 16 and adx < 45:
            failed_rev += 0.08
        failed_rev = min(0.92, failed_rev)

    range_b = 0.0
    if SETUP_RANGE_BOUNCE in aw_setup or "RANGE_BOUNCE" in aw_setup or (ps_regime in ("range", "range_bound", "neutral") and bb <= 0.35 and rsi < 48):
        range_b = 0.42
        if bb <= 0.30:
            range_b += 0.15
        if rsi < 40:
            range_b += 0.12
        if rel_vol > 0.9:
            range_b += 0.08
        if adx < 26:
            range_b += 0.06
        range_b = min(0.88, range_b)

    scores = {
        SETUP_HTF_TREND_PULLBACK: min(1.0, htf_pullback),
        SETUP_VWAP_REVERSION: min(1.0, vwap_rev),
        SETUP_BREAKOUT_CONTINUATION: min(1.0, breakout),
        SETUP_FAILED_BREAKDOWN_REVERSAL: min(1.0, failed_rev),
        SETUP_RANGE_BOUNCE: min(1.0, range_b),
    }

    if is_scalp:
        pref = [SETUP_VWAP_REVERSION, SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK, SETUP_FAILED_BREAKDOWN_REVERSAL, SETUP_RANGE_BOUNCE]
        rank_map = _RANK_DELTA_SCALP
        ev_map = _EV_FACTOR_SCALP
        size_map = _SIZE_FACTOR_SCALP
    else:
        # Prefer proven winners first; FBR last (fills also gated off by default).
        pref = [
            SETUP_RANGE_BOUNCE,
            SETUP_BREAKOUT_CONTINUATION,
            SETUP_HTF_TREND_PULLBACK,
            SETUP_VWAP_REVERSION,
            SETUP_FAILED_BREAKDOWN_REVERSAL,
        ]
        rank_map = _RANK_DELTA
        ev_map = _EV_FACTOR
        size_map = _SIZE_FACTOR

    setup_type = SETUP_NO_CLEAR_THESIS
    best_score = 0.0
    clear_min = 0.40
    for st in pref:
        sc = scores.get(st, 0.0)
        if sc >= clear_min and sc > best_score:
            best_score = sc
            setup_type = st

    if setup_type == SETUP_NO_CLEAR_THESIS:
        top_st = max(scores, key=scores.get)
        top_sc = scores[top_st]
        if top_sc >= 0.30:
            setup_type = top_st
            best_score = top_sc * 0.82

    # Research-only thesis discovery (no live promotion).
    # "NO_CLEAR_THESIS" now explicitly means the engine recognized too few setup types.
    # We probe additional patterns for future replay validation only.
    if setup_type == SETUP_NO_CLEAR_THESIS:
        ltf_map = {tf: mtf.get(tf, {}) for tf in LTF_TFS if isinstance(mtf.get(tf), dict)}
        ema_stack = {"trend_up": ema >= 0.62 and adx >= 18, "trend_down": ema <= 0.38 and adx >= 18}
        research_setup, research_score = _research_discover_setups(
            mtf=mtf,
            ltf=ltf_map,
            vwap=vwap,
            current_price=current_price,
            atr_pct=atr_pct,
            rsi_1h=rsi,
            adx_1h=adx,
            ema_stack=ema_stack,
            symbol=symbol,
        )
        if research_setup and research_score > 0.25:
            setup_type = research_setup
            best_score = research_score

    thesis_score = min(1.0, max(0.0, best_score)) if setup_type != SETUP_NO_CLEAR_THESIS else min(0.35, max(0.05, max(scores.values()) * 0.30))

    invalid_level = 0.0
    target_level = 0.0
    trend_tf = ""
    if setup_type == SETUP_HTF_TREND_PULLBACK and current_price > 0:
        if htf_vals:
            trend_tf = max(HTF_TFS, key=lambda tf: _tf_align(mtf, tf) if isinstance(mtf.get(tf), dict) else 0.0)
        invalid_level = current_price * (1.0 - max(0.007, atr_pct * 1.25))
        target_level = current_price * (1.0 + max(0.011, atr_pct * 1.75))
    elif setup_type == SETUP_VWAP_REVERSION:
        invalid_level = vwap * 0.994 if vwap > 0 else (current_price * 0.991 if current_price > 0 else 0.0)
        target_level = vwap * 1.002 if vwap > 0 else (current_price * 1.005 if current_price > 0 else 0.0)
    elif setup_type == SETUP_BREAKOUT_CONTINUATION and current_price > 0:
        invalid_level = current_price * (1.0 - max(0.005, atr_pct * 0.95))
        target_level = current_price * (1.0 + max(0.014, atr_pct * 2.1))
    elif setup_type == SETUP_FAILED_BREAKDOWN_REVERSAL and current_price > 0:
        # Conservative: invalid just below the sweep/reclaim area, target modest reclaim
        invalid_level = current_price * (1.0 - max(0.006, atr_pct * 1.1))
        target_level = current_price * (1.0 + max(0.009, atr_pct * 1.6))
    elif setup_type == SETUP_RANGE_BOUNCE and current_price > 0:
        invalid_level = current_price * (1.0 - max(0.004, atr_pct * 0.9))
        target_level = current_price * (1.0 + max(0.008, atr_pct * 1.5))

    spread_pct = _safe_float(dd.get("spread_pct"), _safe_float(dd.get("spread"), 0.0))
    if current_price > 0:
        invalid_level = floor_invalidation_level(
            current_price,
            invalid_level,
            atr_pct=atr_pct,
            spread_pct=spread_pct,
        )

    return {
        "setup_type": setup_type,
        "entry_thesis": setup_type,
        "thesis_score": round(thesis_score, 4),
        "thesis_invalid_level": round(invalid_level, 8) if invalid_level > 0 else 0.0,
        "thesis_target_level": round(target_level, 8) if target_level > 0 else 0.0,
        "entry_vwap": round(vwap, 8) if vwap > 0 else 0.0,
        "thesis_trend_tf": trend_tf,
        "thesis_rank_delta": rank_map.get(setup_type, -0.14),
        "thesis_ev_factor": ev_map.get(setup_type, 0.5),
        "thesis_size_factor": size_map.get(setup_type, 0.22),
        "thesis_components": {k: round(v, 4) for k, v in scores.items()},
    }


def thesis_levels_for_setup(
    setup_type: str,
    *,
    current_price: float,
    atr: float,
    decision_data: dict[str, Any] | None = None,
    mtf: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute invalid/target/vwap levels for a locked setup type."""
    dd = decision_data or {}
    mtf = mtf if isinstance(mtf, dict) else parse_mtf_json(dd)
    vwap = _safe_float(dd.get("vwap"), 0.0)
    atr_pct = (atr / current_price) if current_price > 0 and atr > 0 else 0.01
    spread_pct = _safe_float(dd.get("spread_pct"), _safe_float(dd.get("spread"), 0.0))
    htf_vals = [_tf_align(mtf, tf) for tf in HTF_TFS if isinstance(mtf.get(tf), dict)]
    invalid_level = 0.0
    target_level = 0.0
    trend_tf = ""
    st = str(setup_type or "")

    if st == SETUP_HTF_TREND_PULLBACK and current_price > 0:
        if htf_vals:
            trend_tf = max(HTF_TFS, key=lambda tf: _tf_align(mtf, tf) if isinstance(mtf.get(tf), dict) else 0.0)
        invalid_level = current_price * (1.0 - max(0.007, atr_pct * 1.25))
        target_level = current_price * (1.0 + max(0.011, atr_pct * 1.75))
    elif st == SETUP_VWAP_REVERSION:
        invalid_level = vwap * 0.994 if vwap > 0 else (current_price * 0.991 if current_price > 0 else 0.0)
        target_level = vwap * 1.002 if vwap > 0 else (current_price * 1.005 if current_price > 0 else 0.0)
    elif st == SETUP_BREAKOUT_CONTINUATION and current_price > 0:
        invalid_level = current_price * (1.0 - max(0.005, atr_pct * 0.95))
        target_level = current_price * (1.0 + max(0.014, atr_pct * 2.1))
    elif st == SETUP_FAILED_BREAKDOWN_REVERSAL and current_price > 0:
        invalid_level = current_price * (1.0 - max(0.006, atr_pct * 1.1))
        target_level = current_price * (1.0 + max(0.009, atr_pct * 1.6))
    elif st == SETUP_RANGE_BOUNCE and current_price > 0:
        invalid_level = current_price * (1.0 - max(0.004, atr_pct * 0.9))
        target_level = current_price * (1.0 + max(0.008, atr_pct * 1.5))

    if current_price > 0:
        invalid_level = floor_invalidation_level(
            current_price,
            invalid_level,
            atr_pct=atr_pct,
            spread_pct=spread_pct,
        )

    return {
        "thesis_invalid_level": round(invalid_level, 8) if invalid_level > 0 else 0.0,
        "thesis_target_level": round(target_level, 8) if target_level > 0 else 0.0,
        "entry_vwap": round(vwap, 8) if vwap > 0 else 0.0,
        "thesis_trend_tf": trend_tf,
    }


def resolve_day_route_regime(decision_data: dict[str, Any]) -> str:
    """Best-effort DAY route regime from router stamp or signal/context fields."""
    dd = decision_data or {}
    explicit = str(dd.get("day_route_regime") or "").strip().lower()
    if explicit in ("bull", "bear", "range", "chop", "neutral"):
        return explicit
    for key in (
        "regime",
        "signal_regime_label",
        "ctx_market_regime",
        "market_regime",
        "adaptive_regime",
    ):
        val = str(dd.get(key) or "").strip().lower()
        if "bear" in val or "down" in val or "fear" in val:
            return "bear"
        if "bull" in val or "up" in val:
            return "bull"
        if "range" in val:
            return "range"
    ps = str(dd.get("price_structure_regime") or "").strip().lower()
    if "range" in ps:
        return "range"
    # No real regime evidence (no explicit stamp, no regime/market_regime field,
    # price_structure_regime not "range"-like). Previously defaulted to "neutral",
    # which apply_ml_locked_setup_override/remap_setup_for_day_regime treats the
    # same as a genuinely classified range/neutral market — silently downgrading
    # an otherwise-correct HTF_TREND_PULLBACK/BREAKOUT_CONTINUATION classification
    # to RANGE_BOUNCE with no supporting evidence. "unknown" intentionally does
    # not match any branch in remap_setup_for_day_regime, so it leaves the
    # classifier's own setup_type unchanged instead of inventing a regime opinion.
    return explicit or "unknown"


def remap_setup_for_day_regime(setup: str, regime: str) -> str:
    """Preserve HTF/FBR identity for ranking — never rewrite them into RANGE.

    Soft rank/EV demotion handles toxic buckets. Remap must not launder HTF/FBR
    into RANGE for fill optics. Unknown/empty setups still get a regime default
    via apply_ml_locked_setup_override.
    """
    st = str(setup or SETUP_NO_CLEAR_THESIS)
    if st == "TREND_PULLBACK":
        return SETUP_HTF_TREND_PULLBACK
    # Honest identity: keep primary DAY setups as classified.
    if st in (
        SETUP_HTF_TREND_PULLBACK,
        SETUP_FAILED_BREAKDOWN_REVERSAL,
        SETUP_RANGE_BOUNCE,
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_VWAP_REVERSION,
        SETUP_NO_CLEAR_THESIS,
    ):
        return st
    reg = str(regime or "").strip().lower()
    if reg in ("range", "neutral", "bear"):
        return SETUP_RANGE_BOUNCE
    if reg == "bull":
        return SETUP_HTF_TREND_PULLBACK
    return st


def apply_ml_locked_setup_override(
    decision_data: dict[str, Any],
    *,
    current_price: float,
    atr: float,
) -> dict[str, Any]:
    """Stamp setup/levels for exits; preserve raw vs locked identity for explainability."""
    dd = dict(decision_data or {})
    route_regime = resolve_day_route_regime(dd)
    dd["day_route_regime"] = route_regime

    prior_setup = str(decision_data.get("setup_type") or decision_data.get("entry_thesis") or decision_data.get("allweather_setup") or "")
    locked = str(dd.get("allweather_setup") or dd.get("setup_type") or dd.get("entry_thesis") or "")
    if not locked or locked == SETUP_NO_CLEAR_THESIS:
        if route_regime == "bear":
            locked = SETUP_RANGE_BOUNCE
        elif route_regime in ("range", "neutral"):
            locked = SETUP_RANGE_BOUNCE
        else:
            locked = SETUP_HTF_TREND_PULLBACK

    locked = remap_setup_for_day_regime(locked, route_regime)

    levels = thesis_levels_for_setup(locked, current_price=current_price, atr=atr, decision_data=dd)
    dd["setup_type"] = locked
    dd["entry_thesis"] = locked
    dd["allweather_setup"] = locked
    dd["setup_type_canonical"] = locked
    dd["setup_type_raw"] = prior_setup or locked
    dd["setup_regime_remapped_from"] = prior_setup if prior_setup and prior_setup != locked else ""
    dd["setup_remap_is_identity_preserving"] = bool(not prior_setup or prior_setup == locked)
    # Keep narrative / adaptive labels aligned with final locked setup identity.
    narr = str(dd.get("candidate_explanation_narrative") or "")
    if prior_setup and prior_setup != locked:
        prefix = f"setup={locked} (remapped from {prior_setup}; regime={route_regime}). "
        if f"setup={locked}" not in narr:
            dd["candidate_explanation_narrative"] = (prefix + narr).strip()
        dd["setup_identity_unified"] = True
    elif narr and f"setup={locked}" not in narr and locked:
        dd["candidate_explanation_narrative"] = f"setup={locked}. {narr}".strip()
        dd["setup_identity_unified"] = True
    if route_regime and route_regime != "unknown":
        # Store route regime only — never "structure::SETUP" hybrids as adaptive_regime.
        dd["day_route_regime"] = route_regime
        dd["adaptive_regime"] = route_regime
    identity = resolve_setup_identity(dd)
    dd.update(identity)
    dd.update(levels)
    return dd


def apply_trade_thesis_to_candidate_fields(
    decision_data: dict[str, Any],
    *,
    symbol: str,
    current_price: float,
    atr: float,
    strategy_id: str,
    price_structure_regime: str = "unknown",
    context_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dd = enrich_decision_data_for_thesis(
        decision_data,
        symbol=symbol,
        context_payload=context_payload,
        price_structure_regime=price_structure_regime,
    )
    thesis = classify_buy_thesis(
        dd,
        symbol=symbol,
        current_price=current_price,
        atr=atr,
        strategy_id=strategy_id,
        price_structure_regime=price_structure_regime,
    )
    bear = bear_regime_entry_adjustment(
        dd,
        setup_type=str(thesis.get("setup_type") or ""),
        context_payload=context_payload,
    )
    thesis["thesis_rank_delta"] = round(
        float(thesis.get("thesis_rank_delta") or 0.0) + float(bear.get("bear_regime_rank_penalty") or 0.0),
        4,
    )
    thesis["thesis_size_factor"] = round(
        float(thesis.get("thesis_size_factor") or 1.0) * float(bear.get("bear_regime_size_factor") or 1.0),
        4,
    )
    thesis.update(bear)
    dd.update(thesis)
    return apply_ml_locked_setup_override(dd, current_price=current_price, atr=atr)


def min_invalidation_distance_pct(atr_pct: float, spread_pct: float = 0.0) -> float:
    """Minimum distance (fraction) below entry before invalidation may count."""
    rt = float(ESTIMATED_ROUNDTRIP_COST)
    spread = max(0.0, float(spread_pct or 0.0))
    slippage_buf = rt * 0.5
    atr_noise = max(0.008, float(atr_pct or 0.01) * 1.5)
    return max(0.012, spread * 2.0 + rt + slippage_buf + atr_noise * 0.85)


def floor_invalidation_level(
    entry_price: float,
    invalid_level: float,
    *,
    atr_pct: float,
    spread_pct: float = 0.0,
) -> float:
    """Push invalidation below spread/fee/slippage/ATR noise band."""
    if entry_price <= 0:
        return invalid_level
    min_dist = min_invalidation_distance_pct(atr_pct, spread_pct)
    floor_level = entry_price * (1.0 - min_dist)
    if invalid_level > 0:
        return min(invalid_level, floor_level)
    return floor_level


def bear_regime_entry_adjustment(
    decision_data: dict[str, Any],
    *,
    setup_type: str,
    context_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Penalize weak LTF bounces against 1h/4h bear unless breakout is strong."""
    dd = decision_data or {}
    mtf = parse_mtf_json(dd)
    if context_payload and isinstance(context_payload.get("mtf"), dict):
        mtf = {**mtf, **context_payload["mtf"]}

    h1 = _tf_align(mtf, "1h") if isinstance(mtf.get("1h"), dict) else None
    h4 = _tf_align(mtf, "4h") if isinstance(mtf.get("4h"), dict) else None
    m5 = _tf_align(mtf, "5m") if isinstance(mtf.get("5m"), dict) else None
    m15 = _tf_align(mtf, "15m") if isinstance(mtf.get("15m"), dict) else None

    rank_pen = 0.0
    size_mult = 1.0
    htf_bear = (h1 is not None and h1 < 0.42) and (h4 is not None and h4 < 0.40)
    ltf_bounce = (m5 is not None and m5 > 0.55) or (m15 is not None and m15 > 0.52)
    strong_breakout = setup_type == SETUP_BREAKOUT_CONTINUATION and _safe_float(dd.get("thesis_score"), 0.0) >= 0.65

    if htf_bear and ltf_bounce and not strong_breakout:
        rank_pen = -0.10
        size_mult = 0.55
        if setup_type == SETUP_VWAP_REVERSION:
            rank_pen = -0.14
            size_mult = 0.40

    return {
        "bear_regime_rank_penalty": round(rank_pen, 4),
        "bear_regime_size_factor": round(size_mult, 4),
    }


# Internal STALL detail preserved for learning (display still uses STALL_EXIT).
EXIT_STALL = "STALL_EXIT"
EXIT_STALL_DEAD_NO_MFE = "STALL_EXIT_DEAD_NO_MFE"
DEAD_TRADE_NO_MFE = "DEAD_NO_MFE"


def canonical_day_exit_reason(exit_trigger: str, *, exit_type_name: str = "") -> str:
    """Map internal triggers to canonical DAY exit_reason labels (display/reporting)."""
    trig = str(exit_trigger or "").strip().upper()
    if trig.startswith("EXTREME_PROTECTION"):
        return EXIT_EXTREME_PROTECTION
    if trig.startswith("LEGACY_INVENTORY_CLEANUP"):
        return EXIT_LEGACY_INVENTORY_CLEANUP
    if trig.startswith(EXIT_NET_PROFIT):
        return EXIT_NET_PROFIT
    if "ADMIN" in trig and "CLEAR" in trig:
        return EXIT_ADMIN_CLEAR
    if trig.startswith(EXIT_THESIS_INVALIDATION) or (trig.startswith("THESIS_INVALIDATION") and not trig.startswith(EXIT_THESIS_WARNING)):
        return EXIT_THESIS_INVALIDATION
    if trig.startswith(EXIT_THESIS_WARNING):
        return EXIT_THESIS_WARNING
    if trig.startswith(EXIT_STOP_LOSS) or trig.startswith("STOP_LOSS"):
        return EXIT_STOP_LOSS
    if trig.startswith("TRAILING_STOP"):
        return EXIT_TRAILING_STOP
    if trig.startswith("TIME_STOP"):
        return "TIME_STOP_EXIT"
    # Display/reporting collapse: all stall variants → STALL_EXIT.
    # Learning must call split_day_exit_reasons() to keep DEAD_NO_MFE detail.
    if trig.startswith("STALL_EXIT") or trig.startswith("STALL"):
        return EXIT_STALL
    if trig.startswith("GIVEBACK_EXIT") or trig.startswith("GIVEBACK"):
        return "GIVEBACK_EXIT"
    if trig.startswith("VOLATILITY_STOP"):
        return "VOLATILITY_STOP_EXIT"
    if trig.startswith("FAILED_RECLAIM"):
        return "FAILED_RECLAIM_EXIT"
    if trig in (
        EXIT_NET_PROFIT,
        EXIT_MANUAL,
        EXIT_ADMIN_CLEAR,
        EXIT_EXTREME_PROTECTION,
        EXIT_THESIS_WARNING,
        EXIT_THESIS_INVALIDATION,
        EXIT_STOP_LOSS,
        EXIT_TRAILING_STOP,
        EXIT_LEGACY_INVENTORY_CLEANUP,
    ):
        return trig
    # AW-specific reasons must be resolved before the generic MANUAL catch-all
    if "ALLWEATHER_ATR_STOP" in trig:
        return "ALLWEATHER_ATR_STOP_EXIT"
    if "ALLWEATHER_ATR_TARGET" in trig:
        return "ALLWEATHER_ATR_TARGET_EXIT"
    if "ALLWEATHER_TIME_STOP" in trig:
        return "ALLWEATHER_TIME_STOP_EXIT"
    if trig.startswith(EXIT_DAY_4H_STRUCTURE_BREAK) or "4H_STRUCTURE_BREAK" in trig:
        return EXIT_DAY_4H_STRUCTURE_BREAK
    if trig.startswith(EXIT_DAY_RISK_FLOOR) or "DAY_RISK_FLOOR" in trig:
        return EXIT_DAY_RISK_FLOOR
    if "EMERGENCY_FLATTEN" in trig:
        return EXIT_EMERGENCY_FLATTEN
    if "RESTART_FLATTEN" in trig:
        return EXIT_RESTART_FLATTEN
    if trig == "MANUAL" or exit_type_name == "MANUAL":
        return EXIT_MANUAL
    return EXIT_NET_PROFIT


def split_day_exit_reasons(exit_trigger: str, *, exit_type_name: str = "") -> dict[str, Any]:
    """
    Split engine exit trigger into raw (learning) vs canonical (display) labels.

    STALL_EXIT_DEAD_NO_MFE must survive for learning/attribution even though
    paper_trades / UI continue to show STALL_EXIT.
    """
    trig = str(exit_trigger or "").strip().upper()
    dead_trade_reason: str | None = None
    if "STALL_EXIT_DEAD_NO_MFE" in trig or "STALL_EXIT_DEAD" in trig:
        raw_exit_reason = EXIT_STALL_DEAD_NO_MFE
        dead_trade_reason = DEAD_TRADE_NO_MFE
    else:
        raw_exit_reason = trig
    canonical_exit_reason = canonical_day_exit_reason(trig, exit_type_name=exit_type_name)
    if not raw_exit_reason:
        raw_exit_reason = canonical_exit_reason
    return {
        "raw_exit_reason": raw_exit_reason,
        "canonical_exit_reason": canonical_exit_reason,
        "dead_trade_reason": dead_trade_reason,
    }


def resolve_setup_identity(decision_data: dict[str, Any] | None) -> dict[str, str]:
    """
    Authoritative setup key for learning/ranking — label consistency only.

    Prefer locked ``setup_type`` / ``entry_thesis`` (post-remap). Keep raw/remap
    source separately so narrative pollution cannot reopen a second bucket.
    """
    dd = dict(decision_data or {})
    canonical = str(dd.get("setup_type_canonical") or dd.get("setup_type") or dd.get("entry_thesis") or dd.get("allweather_setup") or "").strip().upper()
    raw = str(dd.get("setup_type_raw") or dd.get("setup_regime_remapped_from") or "").strip().upper()
    if not raw:
        raw = canonical
    entry_thesis = str(dd.get("entry_thesis") or canonical or "").strip().upper() or canonical
    # Prefer day_route_regime. Ignore adaptive strings that embed a setup
    # (e.g. "trending_up::HTF_TREND_PULLBACK") — those polluted penalty buckets.
    regime = str(dd.get("day_route_regime") or dd.get("regime") or "").strip().lower()
    adaptive = str(dd.get("adaptive_regime") or "").strip()
    if "::" in adaptive:
        adaptive = regime or adaptive.split("::", 1)[0]
    if not regime:
        regime = adaptive.lower() if adaptive and "::" not in adaptive else "neutral"
    return {
        "setup_type_canonical": canonical,
        "setup_type_raw": raw,
        "entry_thesis": entry_thesis or canonical,
        "day_route_regime": regime or "neutral",
        "adaptive_regime": adaptive or (regime or "neutral"),
        # Alias used by ranking/penalty callers
        "setup_type": canonical,
    }


def thesis_min_profit_floor(
    entry_thesis: str,
    thesis_score: float,
    symbol: str | None = None,
) -> float:
    """Setup-family profit floor with per-symbol MIN_NET_PROFIT override.

    The base floor comes from `min_net_profit_for_symbol` (per-coin env override
    or global default). Setup-family adjustments are additive on top so that
    trend-continuation setups still require a wider target than mean-reversion.
    """
    if symbol:
        from backend.config.trading_economics import min_net_profit_for_symbol

        base = float(min_net_profit_for_symbol(symbol))
    else:
        base = float(MIN_NET_PROFIT_TO_SELL)
    score_adj = max(0.0, min(1.0, 1.0 - float(thesis_score or 0.0)))
    if entry_thesis == SETUP_HTF_TREND_PULLBACK:
        return base + 0.0022 + 0.0015 * score_adj
    if entry_thesis == SETUP_BREAKOUT_CONTINUATION:
        return base + 0.0012 + 0.0008 * score_adj
    if entry_thesis == SETUP_VWAP_REVERSION:
        return base + 0.0006
    return base


def _ema_last(closes: list[float], period: int) -> float:
    if len(closes) < period:
        return float(closes[-1]) if closes else 0.0
    k = 2.0 / (period + 1.0)
    ema = float(closes[0])
    for v in closes[1:]:
        ema = float(v) * k + ema * (1.0 - k)
    return ema


def _ohlcv_ema_alignment(rows: list[Any]) -> float | None:
    """EMA-alignment from raw OHLCV rows; same semantics as
    ai_market_context._ema_alignment_score: 1.0 EMA9>EMA21>EMA50 (uptrend),
    0.0 reversed (downtrend), 0.5 chop. None when insufficient bars."""
    try:
        closes = [float(r[4]) for r in rows if isinstance(r, (list, tuple)) and len(r) >= 5]
    except (TypeError, ValueError):
        return None
    if len(closes) < 50:
        return None
    e9 = _ema_last(closes, 9)
    e21 = _ema_last(closes, 21)
    e50 = _ema_last(closes, 50)
    if e9 > e21 > e50:
        return 1.0
    if e9 < e21 < e50:
        return 0.0
    return 0.5


def _bundle_tf_align(bundle: dict[str, Any] | None, tf: str) -> float | None:
    if not isinstance(bundle, dict):
        return None
    snap = bundle.get(tf)
    if isinstance(snap, dict):
        if snap.get("ema_align") not in (None, ""):
            return _safe_float(snap.get("ema_align"), 0.5)
        return _safe_float(snap.get("trend"), 0.5)
    # Day-active hold bundles store raw OHLCV candle lists per timeframe
    # (see day_active_market_bundle). Without this branch every bundle-based
    # thesis invalidation condition was unreachable (align always None).
    if isinstance(snap, list) and snap:
        return _ohlcv_ema_alignment(snap)
    return None


def _ohlcv_ohlc(row: Any) -> tuple[float, float, float, float] | None:
    """Parse CCXT-style [ts, o, h, l, c, v] or dict candle to (o, h, l, c)."""
    try:
        if isinstance(row, dict):
            o = float(row.get("open") or 0)
            h = float(row.get("high") or 0)
            l = float(row.get("low") or 0)
            c = float(row.get("close") or 0)
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
        else:
            return None
    except (TypeError, ValueError):
        return None
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return None
    return o, h, l, c


DAY_4H_MS = 4 * 3600 * 1000


def current_utc_4h_open_ms(now_epoch: float | None = None) -> int:
    """UTC 4H bar open (ms) for 00/04/08/12/16/20."""
    now = time.time() if now_epoch is None else float(now_epoch)
    ms = int(now * 1000.0)
    return (ms // DAY_4H_MS) * DAY_4H_MS


def ohlcv_row_open_ms(row: Any) -> int | None:
    try:
        if isinstance(row, dict):
            ts = float(row.get("timestamp") or row.get("t") or row.get("open_time") or 0.0)
        elif isinstance(row, (list, tuple)) and row:
            ts = float(row[0] or 0.0)
        else:
            return None
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if ts < 1e12:
        ts *= 1000.0
    return int(ts)


def fourh_requires_boundary_refresh(rows: list[Any] | None, now_epoch: float | None = None) -> bool:
    """True when cached last 4H open is not the current UTC 4H bar."""
    if not isinstance(rows, list) or not rows:
        return True
    last = ohlcv_row_open_ms(rows[-1])
    if last is None:
        return True
    return last != current_utc_4h_open_ms(now_epoch)


def _fresh_forming_close(
    bundle: dict[str, Any] | None,
    current_price: float | None,
    now_epoch: float,
    bar_open_ms: int,
) -> tuple[float | None, str]:
    try:
        px = float(current_price or 0.0)
    except (TypeError, ValueError):
        px = 0.0
    if px > 0:
        return px, "canonical_mark"
    if not isinstance(bundle, dict):
        return None, ""
    rows = bundle.get("1m")
    if not isinstance(rows, list) or not rows:
        return None, ""
    now_ms = int(float(now_epoch) * 1000.0)
    last = rows[-1]
    ot = ohlcv_row_open_ms(last)
    if ot is None or ot < bar_open_ms or ot > now_ms:
        return None, ""
    ohlc = _ohlcv_ohlc(last)
    if ohlc is None:
        return None, ""
    return ohlc[3], "1m_close"


def _row_with_forming_close(row: Any, close: float) -> Any:
    ohlc = _ohlcv_ohlc(row)
    if ohlc is None:
        return row
    o, h, l, _c = ohlc
    h = max(h, close)
    l = min(l, close)
    if isinstance(row, dict):
        out = dict(row)
        out["close"] = close
        out["high"] = h
        out["low"] = l
        return out
    new = list(row)
    while len(new) < 5:
        new.append(0.0)
    new[2] = h
    new[3] = l
    new[4] = close
    return new


def resolve_day_4h_structure_bundle(
    bundle: dict[str, Any] | None,
    *,
    current_price: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any] | None:
    """Same 4H identity/prior-low, forming close from live mark or current 1m.

    Drops future 4H bars. If the cached last bar is the previous UTC 4H
    (1800s TTL carried identity across the boundary), appends a synthetic
    forming bar so prior-low rolls immediately.
    """
    if not isinstance(bundle, dict):
        return bundle
    rows = bundle.get("4h")
    if not isinstance(rows, list) or not rows:
        return bundle
    now = time.time() if now_epoch is None else float(now_epoch)
    bar_open = current_utc_4h_open_ms(now)
    kept: list[Any] = []
    for row in rows:
        ot = ohlcv_row_open_ms(row)
        if ot is not None and ot > bar_open:
            continue
        kept.append(row)
    if not kept:
        out = dict(bundle)
        out["4h"] = []
        out["_forming_4h_close_source"] = "dropped_lookahead"
        return out
    last = kept[-1]
    last_ot = ohlcv_row_open_ms(last)
    fresh, src = _fresh_forming_close(bundle, current_price, now, bar_open)
    if last_ot == bar_open and fresh is not None:
        kept[-1] = _row_with_forming_close(last, fresh)
        src_out = src
    elif last_ot == bar_open - DAY_4H_MS and fresh is not None:
        prev = _ohlcv_ohlc(last)
        prev_c = prev[3] if prev is not None else fresh
        kept.append([bar_open, prev_c, max(prev_c, fresh), min(prev_c, fresh), fresh, 0.0])
        src_out = src + "+boundary_synth"
    else:
        src_out = "cached_4h"
    out = dict(bundle)
    out["4h"] = kept
    out["_forming_4h_close_source"] = src_out
    return out


def _4h_recent_ohlc(bundle: dict[str, Any] | None) -> list[tuple[float, float, float, float]]:
    if not isinstance(bundle, dict):
        return []
    rows = bundle.get("4h")
    if not isinstance(rows, list):
        return []
    out: list[tuple[float, float, float, float]] = []
    for row in rows[-8:]:
        parsed = _ohlcv_ohlc(row)
        if parsed is not None:
            out.append(parsed)
    return out


def htf_4h_rise_intact(
    bundle: dict[str, Any] | None,
    *,
    current_price: float | None = None,
    now_epoch: float | None = None,
) -> bool:
    """True while the 4H breakout/rise is still on (green HH or aligned 4H).

    Used to hold DAY through vertical 4H expansions instead of clipping at
    the first 0.4% path-aware print. Missing bundle is not intact.
    """
    bundle = resolve_day_4h_structure_bundle(bundle, current_price=current_price, now_epoch=now_epoch)
    if not isinstance(bundle, dict):
        return False
    align = _bundle_tf_align(bundle, "4h")
    candles = _4h_recent_ohlc(bundle)
    if len(candles) >= 2:
        _o, h, _l, c = candles[-1]
        _po, ph, pl, pc = candles[-2]
        # A close under the prior 4H low ends the rise even if EMAs still stack up.
        if c < pl:
            return False
        if c >= pc and (c >= _o or h >= ph):
            return True
        if c >= pl and align is not None and align >= 0.50:
            return True
    elif align is not None and align >= 0.52:
        return True
    return False


def _floor_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def resolve_day_risk_floor_price(
    *,
    entry_price: float,
    thesis_invalid_level: float = 0.0,
    prior_4h_low: float = 0.0,
    atr_pct: float = 0.01,
) -> float:
    """Lowest price a DAY position may reach before the engine flattens it.

    Placed just under the structural invalidation so the 4H structure exit gets
    first refusal on a normal thesis failure, and hard-capped so adverse
    excursion stays bounded when structure is far away, unknown, or when price
    gaps through it between 4H closes. Returns 0.0 when it cannot be resolved.
    """
    entry = float(entry_price or 0.0)
    if entry <= 0:
        return 0.0

    buffer = _floor_env("DAY_RISK_FLOOR_STRUCTURE_BUFFER_PCT", 0.003)
    max_adverse = _floor_env("DAY_RISK_FLOOR_MAX_ADVERSE_PCT", 0.06)
    min_adverse = _floor_env("DAY_RISK_FLOOR_MIN_ADVERSE_PCT", 0.02)

    levels = [float(x) for x in (thesis_invalid_level, prior_4h_low) if x and 0.0 < float(x) < entry]
    if levels:
        candidate = min(levels) * (1.0 - buffer)
    else:
        # No structure known. Approximate where structure usually sits so the
        # entry-time stamp matches what will actually be enforced once the 4H
        # levels are available, rather than overstating risk.
        atr_mult = _floor_env("DAY_RISK_FLOOR_ATR_MULT", 1.5)
        candidate = entry * (1.0 - max(min_adverse, float(atr_pct or 0.01) * atr_mult))

    hard_cap = entry * (1.0 - max_adverse)  # never tolerate worse than this
    tight_cap = entry * (1.0 - min_adverse)  # never fire tighter than this
    return max(hard_cap, min(candidate, tight_cap))


def day_4h_structure_snapshot(
    bundle: dict[str, Any] | None,
    *,
    current_price: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Explainability for 4H hold vs structure-break exits."""
    resolved = resolve_day_4h_structure_bundle(bundle, current_price=current_price, now_epoch=now_epoch)
    candles = _4h_recent_ohlc(resolved)
    prior_4h_low = None
    current_4h_close = None
    if len(candles) >= 2:
        prior_4h_low = candles[-2][2]
        current_4h_close = candles[-1][3]
    elif len(candles) == 1:
        current_4h_close = candles[-1][3]
    missing = (not isinstance(resolved, dict)) or not isinstance(resolved.get("4h"), list) or len(candles) < 2
    return {
        "htf_4h_rise_intact": htf_4h_rise_intact(resolved),
        "htf_4h_rise_broken": htf_4h_rise_broken(resolved),
        "prior_4h_low": prior_4h_low,
        "current_4h_close": current_4h_close,
        "4h_bundle_present": not missing,
        "4h_bundle_missing": missing,
        "forming_close_source": (resolved or {}).get("_forming_4h_close_source") if isinstance(resolved, dict) else "",
    }


def htf_4h_rise_broken(
    bundle: dict[str, Any] | None,
    *,
    current_price: float | None = None,
    now_epoch: float | None = None,
) -> bool:
    """True when 4H structure has given way — first lower close below prior low."""
    bundle = resolve_day_4h_structure_bundle(bundle, current_price=current_price, now_epoch=now_epoch)
    if not isinstance(bundle, dict):
        return False
    if htf_4h_rise_intact(bundle):
        return False
    align = _bundle_tf_align(bundle, "4h")
    candles = _4h_recent_ohlc(bundle)
    if len(candles) >= 2:
        o, _h, _l, c = candles[-1]
        _po, _ph, pl, pc = candles[-2]
        if c < pl:
            return True
        if c < o and c < pc and (align is None or align < 0.45):
            return True
    if align is not None and align < 0.40:
        return True
    return False


_PROFIT_CLOSE_MARKERS = (
    "NET_PROFIT",
    "PATH_EXECUTABLE_PROFIT",
    "TP1",
    "TAKE_PROFIT",
)


def is_day_profit_close_reason(reason: str | None) -> bool:
    r = str(reason or "").strip().upper()
    return any(m in r for m in _PROFIT_CLOSE_MARKERS)


def htf_4h_rise_intact_for_symbol(symbol: str) -> bool:
    """Read the cached DAY 4H bundle for a symbol (BTC/ETH/SOL/XRP)."""
    try:
        from backend.services.day_active_market_bundle import read_cached_day_active_bundle_sync

        return htf_4h_rise_intact(read_cached_day_active_bundle_sync(symbol))
    except Exception:
        return False


def same_4h_rise_rebuy_signal(
    *,
    last_close_reason: str | None,
    last_close_epoch: float | None,
    bundle: dict[str, Any] | None,
    now_epoch: float,
) -> str:
    """Diagnostic label only. Never a permission to skip ranking."""
    if not is_day_profit_close_reason(last_close_reason):
        return ""
    try:
        closed = float(last_close_epoch or 0.0)
    except (TypeError, ValueError):
        closed = 0.0
    age = (float(now_epoch) - closed) if closed > 0 else float("inf")
    if age >= 14400.0:
        return ""
    if htf_4h_rise_intact(bundle):
        return "SAME_4H_RISE_NO_REBUY"
    if htf_4h_rise_broken(bundle):
        return ""
    return "SAME_4H_BAR_NO_REBUY"


def should_block_rebuy_on_4h_rise(
    *,
    last_close_reason: str | None,
    last_close_epoch: float | None,
    bundle: dict[str, Any] | None,
    now_epoch: float,
) -> tuple[bool, str]:
    """Never a trade-opinion blocker. Same-rise is rank/diagnostic only."""
    return False, ""


def _current_4h_bar_progress(bundle: dict[str, Any] | None, now_epoch: float) -> float | None:
    if not isinstance(bundle, dict):
        return None
    rows = bundle.get("4h")
    if not isinstance(rows, list) or not rows:
        return None
    last = rows[-1]
    try:
        if isinstance(last, dict):
            ts = float(last.get("timestamp") or last.get("t") or 0.0)
        elif isinstance(last, (list, tuple)) and last:
            ts = float(last[0] or 0.0)
        else:
            return None
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if ts > 1e12:
        ts /= 1000.0
    return max(0.0, min(1.0, (float(now_epoch) - ts) / 14400.0))


_LATE_4H_RANK_SIZE = {
    "LATE_4H_RISE_1H_WEAK": (-0.04, 0.85),
    "LATE_4H_RISE_NO_HH": (-0.04, 0.85),
    "LATE_4H_RISE_RED_BAR": (-0.03, 0.90),
    "LATE_4H_RISE_BAR_LATE": (-0.03, 0.90),
}


def late_4h_rise_signal(bundle: dict[str, Any] | None, now_epoch: float) -> str:
    """Late-rise diagnostic. Used for rank/size only — never permission."""
    if not htf_4h_rise_intact(bundle):
        return ""
    try:
        align1_max = float(os.getenv("DAY_LATE_4H_1H_ALIGN_MAX", "0.42"))
    except (TypeError, ValueError):
        align1_max = 0.42
    try:
        bar_late = float(os.getenv("DAY_LATE_4H_BAR_LATE_FRAC", "0.70"))
    except (TypeError, ValueError):
        bar_late = 0.70
    align1 = _bundle_tf_align(bundle, "1h")
    if align1 is not None and align1 < align1_max:
        return "LATE_4H_RISE_1H_WEAK"
    candles = _4h_recent_ohlc(bundle)
    align4 = _bundle_tf_align(bundle, "4h")
    if len(candles) >= 2:
        o, h, _l, c = candles[-1]
        _po, ph, _pl, pc = candles[-2]
        if c < o and (align4 is None or align4 < 0.58):
            return "LATE_4H_RISE_RED_BAR"
        if h < ph and c < pc:
            return "LATE_4H_RISE_NO_HH"
    frac = _current_4h_bar_progress(bundle, now_epoch)
    if frac is not None and frac >= bar_late:
        return "LATE_4H_RISE_BAR_LATE"
    return ""


def late_4h_rank_size_adjust(signal: str) -> tuple[float, float]:
    return _LATE_4H_RANK_SIZE.get(str(signal or ""), (0.0, 1.0))


def apply_late_4h_rank_to_decision_data(decision_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Stamp late-4H as a rank/size adjustment. Never sets hard_block."""
    dd = dict(decision_data or {})
    try:
        from backend.services.day_active_market_bundle import read_cached_day_active_bundle_sync

        signal = late_4h_rise_signal(read_cached_day_active_bundle_sync(symbol), time.time())
    except Exception:
        signal = ""
    rank_d, size_f = late_4h_rank_size_adjust(signal)
    dd["late_4h_rise_signal"] = signal
    dd["late_4h_rank_delta"] = rank_d
    dd["late_4h_size_factor"] = size_f
    if signal:
        try:
            prev_rank = float(dd.get("thesis_rank_delta") or 0.0)
        except (TypeError, ValueError):
            prev_rank = 0.0
        try:
            prev_size = float(dd.get("thesis_size_factor") or 1.0)
        except (TypeError, ValueError):
            prev_size = 1.0
        dd["thesis_rank_delta"] = round(prev_rank + rank_d, 5)
        dd["thesis_size_factor"] = round(max(0.20, prev_size * size_f), 5)
    dd["hard_block"] = bool(dd.get("hard_block") or False)
    if "candidate_eligible" not in dd:
        dd["candidate_eligible"] = True
    return dd


def should_block_late_4h_rise_entry(
    bundle: dict[str, Any] | None,
    now_epoch: float,
) -> tuple[bool, str]:
    """Never a permission block. Late-4H is rank/size/diagnostics only."""
    return False, ""


def intact_4h_slot_blocked(*, open_intact: int, candidate_intact: bool, max_open: int = 2) -> bool:
    """True when another intact-4H name would stack onto the same tape."""
    if not candidate_intact:
        return False
    try:
        cap = int(os.getenv("DAY_INTACT_4H_MAX_POSITIONS", str(max_open)) or max_open)
    except (TypeError, ValueError):
        cap = int(max_open)
    return int(open_intact) >= max(1, cap)


def thesis_invalidated_live(
    entry_thesis: str,
    *,
    mark: float,
    invalid_level: float,
    bundle: dict[str, Any] | None,
    entry_vwap: float = 0.0,
    entry_price: float = 0.0,
    atr_pct: float = 0.01,
    spread_pct: float = 0.0,
) -> bool:
    if mark <= 0:
        return False
    if entry_price > 0 and invalid_level > 0:
        eff_invalid = floor_invalidation_level(
            entry_price,
            invalid_level,
            atr_pct=atr_pct,
            spread_pct=spread_pct,
        )
        if mark < eff_invalid:
            return True
    elif invalid_level > 0 and mark < invalid_level:
        return True
    if not isinstance(bundle, dict):
        return False
    if entry_thesis == SETUP_HTF_TREND_PULLBACK:
        h1 = _bundle_tf_align(bundle, "1h")
        h4 = _bundle_tf_align(bundle, "4h")
        if h1 is not None and h4 is not None and h1 < 0.38 and h4 < 0.40:
            return True
    if entry_thesis == SETUP_VWAP_REVERSION:
        if entry_vwap > 0 and mark < entry_vwap * 0.993:
            return True
        m5 = _bundle_tf_align(bundle, "5m")
        if m5 is not None and m5 < 0.35:
            return True
    if entry_thesis == SETUP_BREAKOUT_CONTINUATION:
        # 4H still rising: a 15m dip is not thesis death on a vertical breakout.
        if htf_4h_rise_intact(bundle):
            return False
        if htf_4h_rise_broken(bundle):
            return True
        m5 = _bundle_tf_align(bundle, "5m")
        m15 = _bundle_tf_align(bundle, "15m")
        if m5 is not None and m15 is not None and m5 < 0.42 and m15 < 0.45:
            return True
    return False


def evaluate_extreme_protection(
    *,
    entry_price: float,
    mark: float,
    net_pnl_pct: float,
    atr_pct: float = 0.01,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Catastrophic protection only — not normal candle/thesis noise."""
    if entry_price <= 0 or mark <= 0:
        return {"action": "hold", "reason": "no_extreme"}

    loss_floor = max(0.025, float(atr_pct or 0.01) * 2.5)
    flash_floor = max(0.04, float(atr_pct or 0.01) * 3.0)
    gross_loss = (entry_price - mark) / entry_price

    h1 = _bundle_tf_align(bundle, "1h") if bundle else None
    h4 = _bundle_tf_align(bundle, "4h") if bundle else None
    htf_collapse = h1 is not None and h4 is not None and h1 < 0.28 and h4 < 0.28

    catastrophic = False
    if (net_pnl_pct <= -loss_floor and htf_collapse) or (gross_loss >= flash_floor and htf_collapse):
        catastrophic = True

    if catastrophic:
        return {
            "action": "sell",
            "reason": EXIT_EXTREME_PROTECTION,
            "net_pnl_pct": net_pnl_pct,
        }
    return {"action": "hold", "reason": "no_extreme", "net_pnl_pct": net_pnl_pct}


def evaluate_thesis_exit(
    *,
    entry_thesis: str,
    thesis_score: float,
    thesis_invalid_level: float,
    thesis_target_level: float,
    entry_vwap: float,
    entry_price: float,
    mark: float,
    bundle: dict[str, Any] | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Thesis-aware hold/sell hint for position management (no new gates).

    `symbol`: optional per-coin profit floor override. When provided,
    both `thesis_min_profit_floor` and the target-hit / floor gates use
    `min_net_profit_for_symbol(symbol)` instead of the global default.
    """
    if not entry_thesis or entry_thesis == SETUP_NO_CLEAR_THESIS:
        return {"action": "default", "reason": "no_thesis"}

    if entry_price <= 0 or mark <= 0:
        return {"action": "hold", "reason": "missing_price"}

    pnl_pct = (mark - entry_price) / entry_price
    net_pnl = pnl_pct - float(ESTIMATED_ROUNDTRIP_COST)
    atr_pct = abs(entry_price - thesis_invalid_level) / entry_price if thesis_invalid_level > 0 and entry_price > 0 else 0.01
    if thesis_invalid_level > 0 and entry_price > 0:
        atr_pct = max(0.008, (entry_price - thesis_invalid_level) / entry_price)

    invalidated = thesis_invalidated_live(
        entry_thesis,
        mark=mark,
        invalid_level=thesis_invalid_level,
        bundle=bundle,
        entry_vwap=entry_vwap,
        entry_price=entry_price,
        atr_pct=atr_pct,
    )

    if invalidated:
        return {
            "action": "warn",
            "reason": f"{EXIT_THESIS_WARNING}_{entry_thesis}",
            "net_pnl_pct": net_pnl,
        }

    if net_pnl < 0:
        return {"action": "hold", "reason": "THESIS_HOLD_NOISE", "net_pnl_pct": net_pnl}

    if symbol:
        from backend.config.trading_economics import min_net_profit_for_symbol

        min_net = float(min_net_profit_for_symbol(symbol))
    else:
        min_net = float(MIN_NET_PROFIT_TO_SELL)

    target_hit = thesis_target_level > 0 and mark >= thesis_target_level * 0.998
    min_floor = thesis_min_profit_floor(entry_thesis, thesis_score, symbol=symbol)
    if target_hit and net_pnl >= min_net * 0.45:
        return {"action": "sell", "reason": EXIT_NET_PROFIT, "net_pnl_pct": net_pnl, "detail": "target_hit"}

    if net_pnl >= min_net:
        if thesis_target_level > 0 and mark < thesis_target_level * 0.992 and net_pnl < min_floor:
            return {"action": "hold", "reason": "THESIS_HOLD_AWAIT_TARGET", "net_pnl_pct": net_pnl}
        return {"action": "sell", "reason": EXIT_NET_PROFIT, "net_pnl_pct": net_pnl, "detail": "profit_floor"}

    return {"action": "hold", "reason": "THESIS_HOLD", "net_pnl_pct": net_pnl}


def scalp_strategy_to_thesis(setup_name: str, setup_context: dict[str, Any] | None) -> dict[str, Any]:
    """Map scalp strategy signal to canonical thesis fields (paper mirrors live)."""
    name = (setup_name or "").strip().lower()
    ctx = setup_context or {}
    vwap = _safe_float(ctx.get("vwap"), 0.0)
    prior_low = _safe_float(ctx.get("prior_low") or ctx.get("sweep_low"), 0.0)
    sweep = _safe_float(ctx.get("sweep_low"), 0.0)

    if "vwap" in name or "range_bounce" in name:
        setup_type = SETUP_VWAP_REVERSION
        thesis_score = 0.72
    elif "failed_breakdown" in name or "reversal" in name:
        setup_type = SETUP_FAILED_BREAKDOWN_REVERSAL
        thesis_score = 0.65
    elif "compression" in name or "volume_impulse" in name:
        setup_type = SETUP_BREAKOUT_CONTINUATION
        thesis_score = 0.60
    elif "pullback_micro" in name or "trend_pullback" in name:
        setup_type = SETUP_HTF_TREND_PULLBACK
        thesis_score = 0.58
    elif "failed_breakout" in name:
        setup_type = SETUP_RANGE_BOUNCE
        thesis_score = 0.55
    elif "breakout" in name or "momentum" in name or "tape" in name:
        setup_type = SETUP_BREAKOUT_CONTINUATION
        thesis_score = 0.68
    else:
        setup_type = SETUP_NO_CLEAR_THESIS
        thesis_score = 0.25

    # Levels tuned per type (conservative, must clear realistic costs)
    if setup_type == SETUP_FAILED_BREAKDOWN_REVERSAL:
        inv = sweep or (prior_low * 0.998 if prior_low > 0 else 0.0) or (vwap * 0.994 if vwap > 0 else 0.0)
        tgt = (vwap * 1.0025 if vwap > 0 else 0.0) or (prior_low * 1.008 if prior_low > 0 else 0.0)
    elif setup_type == SETUP_RANGE_BOUNCE:
        inv = prior_low * 0.996 if prior_low > 0 else (vwap * 0.993 if vwap > 0 else 0.0)
        tgt = vwap * 1.002 if vwap > 0 else (prior_low * 1.006 if prior_low > 0 else 0.0)
    elif setup_type == SETUP_VWAP_REVERSION:
        inv = vwap * 0.994 if vwap > 0 else 0.0
        tgt = vwap * 1.003 if vwap > 0 else 0.0
    else:
        inv = prior_low or (vwap * 0.994 if vwap > 0 else 0.0)
        tgt = vwap * 1.003 if vwap > 0 else 0.0

    rank_d = _RANK_DELTA_SCALP.get(setup_type, _RANK_DELTA_SCALP.get(SETUP_NO_CLEAR_THESIS, -0.14))
    ev_f = _EV_FACTOR_SCALP.get(setup_type, 0.50)
    sz_f = _SIZE_FACTOR_SCALP.get(setup_type, 0.22)

    return {
        "setup_type": setup_type,
        "entry_thesis": setup_type,
        "thesis_score": round(thesis_score, 4),
        "entry_vwap": round(vwap, 8) if vwap > 0 else 0.0,
        "thesis_trend_tf": "",
        "thesis_invalid_level": round(inv, 8) if inv > 0 else 0.0,
        "thesis_target_level": round(tgt, 8) if tgt > 0 else 0.0,
        "thesis_rank_delta": round(rank_d, 4),
        "thesis_ev_factor": round(ev_f, 4),
        "thesis_size_factor": round(sz_f, 4),
    }


def attach_thesis_to_explainability_dict(payload: dict[str, Any], thesis: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for key in (
        "setup_type",
        "entry_thesis",
        "thesis_score",
        "thesis_invalid_level",
        "thesis_target_level",
        "entry_vwap",
        "thesis_trend_tf",
        "thesis_rank_delta",
        "thesis_ev_factor",
        "thesis_size_factor",
        "thesis_components",
    ):
        if key in thesis:
            out[key] = thesis[key]
    return out


def _research_discover_setups(*, mtf: dict, ltf: dict, vwap: float, current_price: float, atr_pct: float, rsi_1h: float, adx_1h: float, ema_stack: dict, symbol: str) -> tuple[str | None, float]:
    """
    Research-only discovery for additional setup types.
    These are NOT used for live promotion. They exist so that NO_CLEAR_THESIS
    means "engine incomplete" rather than "no trade opportunity".
    All results go only to research replays and diagnostics.
    """
    if current_price <= 0 or atr_pct <= 0:
        return None, 0.0

    # Basic regime from ema/rsi/adx already computed upstream
    trend_up = bool(ema_stack.get("trend_up"))
    trend_down = bool(ema_stack.get("trend_down"))
    adx = float(adx_1h or 0)

    # 1. Failed breakdown reversal (bear trap / reclaim after sweep low)
    # Simple proxy: price above recent low, rsi recovering, adx not extreme
    try:
        ltf_1m = (ltf or {}).get("1m", {}) or {}
        low_1m = float(ltf_1m.get("low", current_price))
        if current_price > low_1m * 1.003 and 35 < rsi_1h < 55 and adx < 28 and not trend_down:
            return RESEARCH_FAILED_BREAKDOWN_REVERSAL, 0.31
    except Exception:
        pass

    # 2. 15m/30m reclaim (price reclaims a broken level on ltf)
    try:
        m15 = (mtf or {}).get("15m", {}) or {}
        if m15.get("close", 0) > 0 and current_price > m15.get("close", current_price) * 0.998 and rsi_1h > 48:
            return RESEARCH_15M_30M_RECLAIM, 0.29
    except Exception:
        pass

    # 3. Volatility expansion (high adx + expanding range) - for breakout or short research
    if adx > 30 and atr_pct > 0.012:
        if trend_down:
            return RESEARCH_SHORT_BEAR_CONTINUATION, 0.33
        return RESEARCH_VOLATILITY_EXPANSION, 0.28

    # 4. Range reclaim / capitulation bounce (low adx, rsi oversold then up)
    if adx < 22 and rsi_1h < 40:
        return RESEARCH_CAPITULATION_BOUNCE, 0.27

    # 5. Trend retest (pullback to ema in uptrend)
    if trend_up and rsi_1h > 45 and atr_pct < 0.02:
        return RESEARCH_TREND_RETEST, 0.30

    # 6. Range reclaim
    if 22 < adx < 32 and 40 < rsi_1h < 60:
        return RESEARCH_RANGE_RECLAIM, 0.26

    return None, 0.0
