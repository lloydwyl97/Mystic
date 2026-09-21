"""Point-in-time 4H entry-structure features. Shadow / research only.

Uses the same production primitives as the DAY 4H-break exit
(``resolve_day_4h_structure_bundle``, ``htf_4h_rise_broken``).
Does not select trades, change EV, rank, HOLD, sizing, or exits.
Never reads bars whose open is after the ranking timestamp.
"""

from __future__ import annotations

import math
from typing import Any

from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_asof_4h import FOURH_SEC, FourHAsOfTracker
from backend.services.day_trade_thesis import (
    _4h_recent_ohlc,
    _bundle_tf_align,
    current_utc_4h_open_ms,
    day_4h_structure_snapshot,
    htf_4h_rise_broken,
    htf_4h_rise_intact,
    ohlcv_row_open_ms,
    resolve_day_4h_structure_bundle,
)

SCHEMA_VERSION = "day_4h_entry_structure_v1"
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HOLD_SYMBOL = "HOLD"

# Predeclared diagnostic labels. Not profit-tuned weights.
STRUCTURE_INTACT = "intact"
STRUCTURE_BROKEN = "broken"
STRUCTURE_UNDECIDED = "undecided"
STRUCTURE_MISSING = "missing"
ALIGN_STRONG = "aligned"
ALIGN_WEAK = "weak"
ALIGN_UNKNOWN = "unknown"


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _bps(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom <= 0:
        return None
    return (float(numer) / float(denom)) * 1e4


def drop_bars_after(rows: list[Any] | None, now_epoch: float) -> list[Any]:
    """Keep only bars whose open timestamp is <= now. No future bars."""
    if not isinstance(rows, list):
        return []
    now_ms = int(float(now_epoch) * 1000.0)
    kept: list[Any] = []
    for row in rows:
        ot = ohlcv_row_open_ms(row)
        if ot is None or ot > now_ms:
            continue
        kept.append(row)
    return kept


def asof_bundle_from_1m(
    bars_1m: list[tuple[int, float, ...]],
    now_epoch: float,
    *,
    seed_4h: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Build a production-shaped 4H bundle using only bars at or before now."""
    now_f = float(now_epoch)
    clipped = [b for b in bars_1m if int(b[0]) <= now_f + 1e-9]
    tracker = FourHAsOfTracker(bars_1m=clipped)
    if seed_4h:
        closed = [r for r in seed_4h if float(r[0]) + FOURH_SEC <= now_f + 1e-9]
        tracker.seed_completed(closed)
    bundle = tracker.advance(now_f)
    tracker.assert_as_of(now_f, bundle)
    return bundle


def hold_4h_entry_features() -> dict[str, Any]:
    """HOLD has no structural 4H fields and zero economic value."""
    return {
        "4h_structure_schema_version": SCHEMA_VERSION,
        "symbol": HOLD_SYMBOL,
        "4h_structure_state": STRUCTURE_MISSING,
        "production_4h_break_true_now": None,
        "production_4h_intact_at_decision": None,
        "prior_completed_4h_low": None,
        "prior_completed_4h_high": None,
        "prior_completed_4h_close": None,
        "current_4h_open": None,
        "current_4h_high_so_far": None,
        "current_4h_low_so_far": None,
        "current_4h_close_or_mark": None,
        "distance_to_prior_4h_low_bps": None,
        "distance_to_4h_break_bps": None,
        "distance_to_prior_4h_high_bps": None,
        "current_4h_range_bps": None,
        "position_within_current_4h_range": None,
        "4h_return_current_bar_bps": None,
        "4h_return_prior_bar_bps": None,
        "4h_lower_low_flag": None,
        "4h_lower_high_flag": None,
        "4h_lower_close_flag": None,
        "4h_alignment_state": ALIGN_UNKNOWN,
        "4h_alignment": None,
        "minutes_into_current_4h_bar": None,
        "time_since_prior_4h_close": None,
        "distance_to_break_vs_expected_cost": None,
        "distance_to_break_vs_path_ev": None,
        "distance_to_break_vs_expected_upside": None,
        "4h_risk_to_reward_ratio": None,
        "4h_break_distance_bps": None,
        "path_ev": 0.0,
        "expected_cost_bps": 0.0,
        "field_authority": "hold_null",
    }


def _alignment_label(align: float | None) -> str:
    if align is None:
        return ALIGN_UNKNOWN
    if align >= 0.50:
        return ALIGN_STRONG
    return ALIGN_WEAK


def _structure_state(snap: dict[str, Any]) -> str:
    if snap.get("4h_bundle_missing"):
        return STRUCTURE_MISSING
    if snap.get("htf_4h_rise_broken"):
        return STRUCTURE_BROKEN
    if snap.get("htf_4h_rise_intact"):
        return STRUCTURE_INTACT
    return STRUCTURE_UNDECIDED


def build_4h_entry_features(
    *,
    bundle: dict[str, Any] | None,
    now_epoch: float,
    current_price: float | None = None,
    path_ev: float | None = None,
    expected_cost_bps: float | None = None,
    expected_upside_bps: float | None = None,
    symbol: str = "",
) -> dict[str, Any]:
    """Compute strictly as-of 4H structure at a ranking timestamp.

    ``now_epoch`` is the only clock. Bars opening after it are dropped.
    """
    if str(symbol or "").upper() == HOLD_SYMBOL:
        return hold_4h_entry_features()
    now_f = float(now_epoch)
    src = dict(bundle) if isinstance(bundle, dict) else {}
    clipped: dict[str, Any] = {}
    for key, value in src.items():
        if isinstance(value, list):
            clipped[key] = drop_bars_after(value, now_f)
        else:
            clipped[key] = value
    resolved = resolve_day_4h_structure_bundle(clipped, current_price=current_price, now_epoch=now_f)
    snap = day_4h_structure_snapshot(resolved, current_price=current_price, now_epoch=now_f)
    candles = _4h_recent_ohlc(resolved)
    align = _bundle_tf_align(resolved, "4h") if isinstance(resolved, dict) else None
    broken = bool(htf_4h_rise_broken(resolved, current_price=current_price, now_epoch=now_f))
    intact = bool(htf_4h_rise_intact(resolved, current_price=current_price, now_epoch=now_f))

    prior_o = prior_h = prior_l = prior_c = None
    cur_o = cur_h = cur_l = cur_c = None
    if len(candles) >= 2:
        prior_o, prior_h, prior_l, prior_c = candles[-2]
        cur_o, cur_h, cur_l, cur_c = candles[-1]
    elif len(candles) == 1:
        cur_o, cur_h, cur_l, cur_c = candles[-1]
    mark = _num(current_price)
    if mark is None:
        mark = cur_c
    if cur_c is None:
        cur_c = mark

    dist_low = _bps((mark - prior_l) if mark is not None and prior_l is not None else None, mark)
    dist_high = _bps((prior_h - mark) if mark is not None and prior_h is not None else None, mark)
    rng = None
    pos = None
    if cur_h is not None and cur_l is not None and (cur_h - cur_l) > 0 and mark is not None:
        rng = _bps(cur_h - cur_l, mark)
        pos = (mark - cur_l) / (cur_h - cur_l)
    ret_cur = _bps((cur_c - cur_o) if cur_c is not None and cur_o is not None else None, cur_o)
    ret_prior = _bps((prior_c - prior_o) if prior_c is not None and prior_o is not None else None, prior_o)

    bar_open_ms = current_utc_4h_open_ms(now_f)
    minutes_into = max(0.0, (now_f - (bar_open_ms / 1000.0)) / 60.0)
    time_since_prior = minutes_into  # prior 4H close is the current bar open

    cost = _num(expected_cost_bps)
    if cost is None and symbol:
        cost = honest_all_in_rt_pct(symbol) * 1e4
    path = _num(path_ev)
    path_bps = (path * 1e4) if path is not None else None
    upside = _num(expected_upside_bps)
    vs_cost = (dist_low / cost) if dist_low is not None and cost not in (None, 0.0) else None
    vs_path = (dist_low / path_bps) if dist_low is not None and path_bps not in (None, 0.0) else None
    vs_up = (dist_low / upside) if dist_low is not None and upside not in (None, 0.0) else None
    rr = (upside / dist_low) if upside is not None and dist_low not in (None, 0.0) else None

    return {
        "4h_structure_schema_version": SCHEMA_VERSION,
        "symbol": str(symbol or ""),
        "4h_structure_state": _structure_state({**snap, "htf_4h_rise_broken": broken, "htf_4h_rise_intact": intact}),
        "production_4h_break_true_now": broken,
        "production_4h_intact_at_decision": intact,
        "htf_4h_rise_intact": intact,
        "prior_completed_4h_low": prior_l,
        "prior_completed_4h_high": prior_h,
        "prior_completed_4h_close": prior_c,
        "current_4h_open": cur_o,
        "current_4h_high_so_far": cur_h,
        "current_4h_low_so_far": cur_l,
        "current_4h_close_or_mark": cur_c,
        "forming_4h_close": cur_c,
        "prior_4h_low": prior_l,
        "distance_to_prior_4h_low_bps": dist_low,
        "distance_to_4h_break_bps": dist_low,
        "4h_break_distance_bps": dist_low,
        "distance_to_prior_4h_high_bps": dist_high,
        "current_4h_range_bps": rng,
        "position_within_current_4h_range": pos,
        "4h_range_position": pos,
        "4h_return_current_bar_bps": ret_cur,
        "4h_return_prior_bar_bps": ret_prior,
        "4h_lower_low_flag": bool(cur_l is not None and prior_l is not None and cur_l < prior_l),
        "4h_lower_high_flag": bool(cur_h is not None and prior_h is not None and cur_h < prior_h),
        "4h_lower_close_flag": bool(cur_c is not None and prior_c is not None and cur_c < prior_c),
        "4h_alignment_state": _alignment_label(align),
        "4h_alignment": align,
        "minutes_into_current_4h_bar": minutes_into,
        "forming_4h_bar_age_min": minutes_into,
        "time_since_prior_4h_close": time_since_prior,
        "time_to_4h_boundary_min": max(0.0, 240.0 - minutes_into),
        "distance_to_break_vs_expected_cost": vs_cost,
        "distance_to_break_vs_path_ev": vs_path,
        "distance_to_break_vs_expected_upside": vs_up,
        "4h_risk_to_reward_ratio": rr,
        "path_ev": path,
        "expected_cost_bps": cost,
        "forming_close_source": snap.get("forming_close_source"),
        "4h_bundle_missing": snap.get("4h_bundle_missing"),
        "field_authority": "reconstructed_from_asof_bundle",
    }


def shadow_4h_structure_score(features: dict[str, Any] | None) -> dict[str, Any]:
    """Transparent diagnostic decomposition. Not a live trading score.

    Predeclared (not fit to profit):
      state = production intact/broken/undecided/missing
      distance = raw bps to prior 4H low
      alignment = raw 4H EMA align
      risk_to_reward = expected_upside / distance_to_break when both exist
    """
    src = dict(features or {})
    return {
        "4h_structure_state": src.get("4h_structure_state") or STRUCTURE_MISSING,
        "4h_break_distance_bps": src.get("distance_to_4h_break_bps"),
        "4h_alignment_state": src.get("4h_alignment_state") or ALIGN_UNKNOWN,
        "4h_risk_to_reward_ratio": src.get("4h_risk_to_reward_ratio"),
        "composite_formula": "no_composite_not_tuned",
        "ranking_only": True,
        "live_gate": False,
    }


def classify_feature_availability() -> list[dict[str, str]]:
    """Static map of production knowledge vs this shadow builder."""
    return [
        {"feature": "prior_4h_low", "available_at_entry": "AVAILABLE_ONLY_IN_EXIT_ENGINE", "stored": "provenance_schema_unpopulated", "used_by_ranker": "no", "used_by_exit": "yes"},
        {"feature": "distance_to_prior_4h_low_bps", "available_at_entry": "AVAILABLE_BUT_NOT_USED_IN_RANKING", "stored": "provenance_schema_unpopulated", "used_by_ranker": "no", "used_by_exit": "no"},
        {"feature": "forming_4h_bar_age_min", "available_at_entry": "AVAILABLE_BUT_NOT_USED_IN_RANKING", "stored": "provenance_schema_unpopulated", "used_by_ranker": "no", "used_by_exit": "no"},
        {"feature": "htf_4h_rise_broken", "available_at_entry": "AVAILABLE_ONLY_IN_EXIT_ENGINE", "stored": "no", "used_by_ranker": "no", "used_by_exit": "yes"},
        {"feature": "htf_4h_rise_intact", "available_at_entry": "AVAILABLE_ONLY_IN_EXIT_ENGINE", "stored": "no", "used_by_ranker": "no", "used_by_exit": "yes"},
        {"feature": "current_4h_close", "available_at_entry": "AVAILABLE_ONLY_IN_EXIT_ENGINE", "stored": "no", "used_by_ranker": "no", "used_by_exit": "yes"},
        {
            "feature": "slope_pct_4h",
            "available_at_entry": "ALREADY_IN_FEATURE_VECTOR",
            "stored": "ai_inference_log.features_json[129]",
            "used_by_ranker": "path_net_145_indirect",
            "used_by_exit": "no",
        },
        {
            "feature": "mean_ema_align_all_tf",
            "available_at_entry": "ALREADY_IN_FEATURE_VECTOR",
            "stored": "ai_inference_log.features_json[136]",
            "used_by_ranker": "path_net_145_indirect",
            "used_by_exit": "no",
        },
        {"feature": "support_1", "available_at_entry": "ALREADY_IN_FEATURE_VECTOR", "stored": "features_json[107]", "used_by_ranker": "1m_pivot_not_4h", "used_by_exit": "no"},
        {"feature": "late_4h_rise_signal", "available_at_entry": "AVAILABLE_BUT_NOT_USED_IN_RANKING", "stored": "decision_data_stamp", "used_by_ranker": "old_rank_delta_only", "used_by_exit": "no"},
        {"feature": "distance_to_4h_break_bps", "available_at_entry": "NOT_CURRENTLY_AVAILABLE_AT_ENTRY", "stored": "no", "used_by_ranker": "no", "used_by_exit": "no"},
        {"feature": "production_4h_break_true_now", "available_at_entry": "AVAILABLE_ONLY_IN_EXIT_ENGINE", "stored": "no", "used_by_ranker": "no", "used_by_exit": "yes"},
    ]


__all__ = [
    "ALIGN_STRONG",
    "ALIGN_UNKNOWN",
    "ALIGN_WEAK",
    "COINS",
    "HOLD_SYMBOL",
    "SCHEMA_VERSION",
    "STRUCTURE_BROKEN",
    "STRUCTURE_INTACT",
    "STRUCTURE_MISSING",
    "STRUCTURE_UNDECIDED",
    "asof_bundle_from_1m",
    "build_4h_entry_features",
    "classify_feature_availability",
    "drop_bars_after",
    "hold_4h_entry_features",
    "shadow_4h_structure_score",
]
