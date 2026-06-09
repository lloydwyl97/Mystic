"""
Trade thesis classification for DAY ranking/explainability and position management.

Uses existing Mystic signal/context fields only (no new gates, no .pkl changes).
"""

from __future__ import annotations

import json
import math
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL

SETUP_HTF_TREND_PULLBACK = "HTF_TREND_PULLBACK"
SETUP_VWAP_REVERSION = "VWAP_REVERSION"
SETUP_BREAKOUT_CONTINUATION = "BREAKOUT_CONTINUATION"
SETUP_NO_CLEAR_THESIS = "NO_CLEAR_THESIS"

ALL_SETUP_TYPES = (
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_NO_CLEAR_THESIS,
)

HTF_TFS = ("15m", "30m", "1h", "4h")
LTF_TFS = ("1m", "5m")
BREAKOUT_ALT_SYMBOLS = frozenset({"SOLUSDT", "XRPUSDT", "DOGEUSDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"})

_RANK_DELTA = {
    SETUP_HTF_TREND_PULLBACK: 0.07,
    SETUP_VWAP_REVERSION: -0.02,
    SETUP_BREAKOUT_CONTINUATION: 0.04,
    SETUP_NO_CLEAR_THESIS: -0.14,
}
_RANK_DELTA_SCALP = {
    SETUP_HTF_TREND_PULLBACK: 0.02,
    SETUP_VWAP_REVERSION: 0.06,
    SETUP_BREAKOUT_CONTINUATION: 0.04,
    SETUP_NO_CLEAR_THESIS: -0.14,
}
_EV_FACTOR = {
    SETUP_HTF_TREND_PULLBACK: 1.08,
    SETUP_VWAP_REVERSION: 0.92,
    SETUP_BREAKOUT_CONTINUATION: 1.02,
    SETUP_NO_CLEAR_THESIS: 0.50,
}
_EV_FACTOR_SCALP = {
    SETUP_HTF_TREND_PULLBACK: 1.0,
    SETUP_VWAP_REVERSION: 1.05,
    SETUP_BREAKOUT_CONTINUATION: 1.02,
    SETUP_NO_CLEAR_THESIS: 0.50,
}
_SIZE_FACTOR = {
    SETUP_HTF_TREND_PULLBACK: 1.0,
    SETUP_VWAP_REVERSION: 0.65,
    SETUP_BREAKOUT_CONTINUATION: 0.88,
    SETUP_NO_CLEAR_THESIS: 0.22,
}
_SIZE_FACTOR_SCALP = {
    SETUP_HTF_TREND_PULLBACK: 0.85,
    SETUP_VWAP_REVERSION: 1.0,
    SETUP_BREAKOUT_CONTINUATION: 0.90,
    SETUP_NO_CLEAR_THESIS: 0.22,
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
        except Exception:
            pass
    return dd


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

    scores = {
        SETUP_HTF_TREND_PULLBACK: min(1.0, htf_pullback),
        SETUP_VWAP_REVERSION: min(1.0, vwap_rev),
        SETUP_BREAKOUT_CONTINUATION: min(1.0, breakout),
    }

    if is_scalp:
        pref = [SETUP_VWAP_REVERSION, SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK]
        rank_map = _RANK_DELTA_SCALP
        ev_map = _EV_FACTOR_SCALP
        size_map = _SIZE_FACTOR_SCALP
    else:
        pref = [SETUP_HTF_TREND_PULLBACK, SETUP_BREAKOUT_CONTINUATION, SETUP_VWAP_REVERSION]
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

    thesis_score = (
        min(1.0, max(0.0, best_score))
        if setup_type != SETUP_NO_CLEAR_THESIS
        else min(0.35, max(0.05, max(scores.values()) * 0.30))
    )

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
    dd.update(thesis)
    return dd


def thesis_min_profit_floor(entry_thesis: str, thesis_score: float) -> float:
    base = float(MIN_NET_PROFIT_TO_SELL)
    score_adj = max(0.0, min(1.0, 1.0 - float(thesis_score or 0.0)))
    if entry_thesis == SETUP_HTF_TREND_PULLBACK:
        return base + 0.0022 + 0.0015 * score_adj
    if entry_thesis == SETUP_BREAKOUT_CONTINUATION:
        return base + 0.0012 + 0.0008 * score_adj
    if entry_thesis == SETUP_VWAP_REVERSION:
        return base + 0.0006
    return base


def _bundle_tf_align(bundle: dict[str, Any] | None, tf: str) -> float | None:
    if not isinstance(bundle, dict):
        return None
    snap = bundle.get(tf)
    if not isinstance(snap, dict):
        return None
    if snap.get("ema_align") not in (None, ""):
        return _safe_float(snap.get("ema_align"), 0.5)
    return _safe_float(snap.get("trend"), 0.5)


def thesis_invalidated_live(
    entry_thesis: str,
    *,
    mark: float,
    invalid_level: float,
    bundle: dict[str, Any] | None,
    entry_vwap: float = 0.0,
) -> bool:
    if mark <= 0:
        return False
    if invalid_level > 0 and mark < invalid_level:
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
        m5 = _bundle_tf_align(bundle, "5m")
        m15 = _bundle_tf_align(bundle, "15m")
        if m5 is not None and m15 is not None and m5 < 0.42 and m15 < 0.45:
            return True
    return False


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
) -> dict[str, Any]:
    """Thesis-aware hold/sell hint for position management (no new gates)."""
    if not entry_thesis or entry_thesis == SETUP_NO_CLEAR_THESIS:
        return {"action": "default", "reason": "no_thesis"}

    if entry_price <= 0 or mark <= 0:
        return {"action": "hold", "reason": "missing_price"}

    pnl_pct = (mark - entry_price) / entry_price
    net_pnl = pnl_pct - float(ESTIMATED_ROUNDTRIP_COST)
    invalidated = thesis_invalidated_live(
        entry_thesis,
        mark=mark,
        invalid_level=thesis_invalid_level,
        bundle=bundle,
        entry_vwap=entry_vwap,
    )

    if invalidated:
        return {"action": "sell", "reason": f"THESIS_INVALIDATION_{entry_thesis}", "net_pnl_pct": net_pnl}

    if net_pnl < 0:
        return {"action": "hold", "reason": "THESIS_HOLD_NOISE", "net_pnl_pct": net_pnl}

    target_hit = thesis_target_level > 0 and mark >= thesis_target_level * 0.998
    min_floor = thesis_min_profit_floor(entry_thesis, thesis_score)
    if target_hit and net_pnl >= float(MIN_NET_PROFIT_TO_SELL) * 0.45:
        return {"action": "sell", "reason": "THESIS_TARGET_HIT", "net_pnl_pct": net_pnl}

    if net_pnl >= float(MIN_NET_PROFIT_TO_SELL):
        if thesis_target_level > 0 and mark < thesis_target_level * 0.992 and net_pnl < min_floor:
            return {"action": "hold", "reason": "THESIS_HOLD_AWAIT_TARGET", "net_pnl_pct": net_pnl}
        return {"action": "sell", "reason": "THESIS_PROFIT_PROTECTION", "net_pnl_pct": net_pnl}

    return {"action": "hold", "reason": "THESIS_HOLD", "net_pnl_pct": net_pnl}


def scalp_strategy_to_thesis(setup_name: str, setup_context: dict[str, Any] | None) -> dict[str, Any]:
    """Map scalp strategy signal to canonical thesis fields."""
    name = (setup_name or "").strip().lower()
    ctx = setup_context or {}
    vwap = _safe_float(ctx.get("vwap"), 0.0)
    if "vwap" in name or "range_bounce" in name:
        setup_type = SETUP_VWAP_REVERSION
        thesis_score = 0.72
    elif "breakout" in name or "momentum" in name or "tape" in name:
        setup_type = SETUP_BREAKOUT_CONTINUATION
        thesis_score = 0.68
    else:
        setup_type = SETUP_NO_CLEAR_THESIS
        thesis_score = 0.25
    return {
        "setup_type": setup_type,
        "entry_thesis": setup_type,
        "thesis_score": thesis_score,
        "entry_vwap": vwap,
        "thesis_trend_tf": "",
        "thesis_invalid_level": _safe_float(ctx.get("prior_low"), 0.0) or (vwap * 0.994 if vwap > 0 else 0.0),
        "thesis_target_level": vwap * 1.003 if vwap > 0 and setup_type == SETUP_VWAP_REVERSION else 0.0,
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
