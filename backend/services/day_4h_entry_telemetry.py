"""Fail-open 4H entry-structure telemetry. Persistence only.

Never changes selected_action, path-EV, rank order, HOLD, size, or exits.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_4h_entry_features import (
    COINS,
    HOLD_SYMBOL,
    SCHEMA_VERSION,
    build_4h_entry_features,
    hold_4h_entry_features,
    shadow_4h_structure_score,
)
from backend.services.day_direct_path_ev_authority import HOLD_EV
from backend.services.day_production_lifecycle_replay import parse_epoch

logger = logging.getLogger(__name__)

PROTECTED = {
    "selected_action",
    "selected_symbol",
    "path_ev_winner",
    "selected_ev",
    "why_selected",
    "btc_path_ev",
    "eth_path_ev",
    "sol_path_ev",
    "xrp_path_ev",
    "hold_ev",
}

TELEMETRY_KEYS = (
    "prior_4h_low",
    "prior_4h_high",
    "prior_4h_close",
    "forming_4h_open",
    "forming_4h_high",
    "forming_4h_low",
    "forming_4h_close",
    "distance_to_prior_4h_low_bps",
    "distance_to_4h_break_bps",
    "4h_range_position",
    "4h_alignment_state",
    "production_4h_break_true_at_decision",
    "minutes_into_4h_bar",
    "distance_to_break_vs_cost",
    "distance_to_break_vs_path_ev",
    "4h_structure_schema_version",
)


def _ev_for(decision: dict[str, Any], symbol: str) -> float | None:
    key = {
        "BTCUSDT": "btc_path_ev",
        "ETHUSDT": "eth_path_ev",
        "SOLUSDT": "sol_path_ev",
        "XRPUSDT": "xrp_path_ev",
        HOLD_SYMBOL: "hold_ev",
    }.get(symbol)
    if not key:
        return None
    raw = decision.get(key)
    try:
        return float(raw) if raw not in (None, "") else HOLD_EV
    except (TypeError, ValueError):
        return HOLD_EV


def _now_epoch(decision: dict[str, Any]) -> float:
    ts = decision.get("prediction_timestamp") or decision.get("decision_timestamp")
    ep = parse_epoch(ts)
    if ep is not None:
        return float(ep)
    import time

    return time.time()


def _bundle_for(symbol: str) -> dict[str, Any] | None:
    try:
        from backend.services.day_active_market_bundle import read_cached_day_active_bundle_sync

        return read_cached_day_active_bundle_sync(symbol.replace("USDT", "/USDT") if symbol.endswith("USDT") else symbol)
    except Exception:
        return None


def _mark_for(symbol: str, redis_client: Any = None) -> float | None:
    try:
        from backend.services.decision_book_tape import snapshot_book

        book = snapshot_book(symbol, redis_client)
        mid = book.get("mid")
        if mid not in (None, ""):
            return float(mid)
    except Exception:
        return None
    return None


def _compact(features: dict[str, Any]) -> dict[str, Any]:
    score = shadow_4h_structure_score(features)
    return {
        "prior_4h_low": features.get("prior_completed_4h_low"),
        "prior_4h_high": features.get("prior_completed_4h_high"),
        "prior_4h_close": features.get("prior_completed_4h_close"),
        "forming_4h_open": features.get("current_4h_open"),
        "forming_4h_high": features.get("current_4h_high_so_far"),
        "forming_4h_low": features.get("current_4h_low_so_far"),
        "forming_4h_close": features.get("current_4h_close_or_mark"),
        "distance_to_prior_4h_low_bps": features.get("distance_to_prior_4h_low_bps"),
        "distance_to_4h_break_bps": features.get("distance_to_4h_break_bps"),
        "4h_range_position": features.get("position_within_current_4h_range"),
        "4h_alignment_state": features.get("4h_alignment_state"),
        "production_4h_break_true_at_decision": features.get("production_4h_break_true_now"),
        "production_4h_intact_at_decision": features.get("production_4h_intact_at_decision") if "production_4h_intact_at_decision" in features else features.get("htf_4h_rise_intact"),
        "minutes_into_4h_bar": features.get("minutes_into_current_4h_bar"),
        "distance_to_break_vs_cost": features.get("distance_to_break_vs_expected_cost"),
        "distance_to_break_vs_path_ev": features.get("distance_to_break_vs_path_ev"),
        "4h_structure_schema_version": SCHEMA_VERSION,
        "4h_structure_state": score["4h_structure_state"],
        "4h_break_distance_bps": score["4h_break_distance_bps"],
        "4h_risk_to_reward_ratio": score["4h_risk_to_reward_ratio"],
    }


def collect_4h_entry_telemetry(decision: dict[str, Any] | None, *, redis_client: Any = None) -> dict[str, Any]:
    """Build per-symbol 4H telemetry. Fail-open. Does not mutate decision."""
    dec = dict(decision or {})
    now = _now_epoch(dec)
    by_symbol: dict[str, Any] = {}
    for symbol in COINS:
        try:
            feats = build_4h_entry_features(
                bundle=_bundle_for(symbol),
                now_epoch=now,
                current_price=_mark_for(symbol, redis_client),
                path_ev=_ev_for(dec, symbol),
                expected_cost_bps=honest_all_in_rt_pct(symbol) * 1e4,
                symbol=symbol,
            )
            by_symbol[symbol] = _compact(feats)
        except Exception as exc:
            logger.debug("4h entry telemetry failed %s: %s", symbol, exc)
            by_symbol[symbol] = _compact(hold_4h_entry_features())
            by_symbol[symbol]["field_authority"] = "telemetry_fail_open"
    by_symbol[HOLD_SYMBOL] = _compact(hold_4h_entry_features())
    selected = str(dec.get("selected_symbol") or "")
    if str(dec.get("selected_action") or "").upper() == "HOLD" or not selected:
        selected = HOLD_SYMBOL
    peer = _peer_structure(by_symbol, selected)
    return {
        "4h_structure_schema_version": SCHEMA_VERSION,
        "4h_entry_telemetry": by_symbol,
        "4h_peer_structure": peer,
        "4h_telemetry_live_gate": False,
        **peer,
    }


def _peer_structure(by_symbol: dict[str, Any], selected: str) -> dict[str, Any]:
    """Selected vs peer distances. Analytics only. Never a live gate."""
    dists: list[tuple[str, float, bool | None]] = []
    for symbol in COINS:
        row = by_symbol.get(symbol) or {}
        dist = row.get("distance_to_4h_break_bps")
        broken = row.get("production_4h_break_true_at_decision")
        if dist is None:
            continue
        try:
            dists.append((symbol, float(dist), broken if isinstance(broken, bool) else None))
        except (TypeError, ValueError):
            continue
    sel_row = by_symbol.get(selected) or {}
    sel_dist = None
    try:
        if sel_row.get("distance_to_4h_break_bps") not in (None, ""):
            sel_dist = float(sel_row["distance_to_4h_break_bps"])
    except (TypeError, ValueError):
        sel_dist = None
    sel_broken = sel_row.get("production_4h_break_true_at_decision") if selected != HOLD_SYMBOL else None
    healthiest_symbol = None
    healthiest_dist = None
    if dists:
        healthiest_symbol, healthiest_dist, _brk = max(dists, key=lambda item: item[1])
    peer_minus = None
    if sel_dist is not None and healthiest_dist is not None:
        peer_minus = healthiest_dist - sel_dist
    broken_peers = [item for item in dists if item[2] is True]
    intact_peers = [item for item in dists if item[2] is False]
    return {
        "selected_4h_state": sel_row.get("4h_structure_state") if selected != HOLD_SYMBOL else None,
        "selected_already_broken_at_ranking": bool(sel_broken) if isinstance(sel_broken, bool) else None,
        "selected_distance_to_break_bps": sel_dist,
        "selected_vs_best_peer_distance_bps": peer_minus,
        "healthiest_peer_symbol": healthiest_symbol,
        "healthiest_peer_distance_bps": healthiest_dist,
        "all_four_already_broken": bool(len(broken_peers) == 4) if len(dists) == 4 else None,
        "all_four_already_broken_definition": "production_4h_break_true_at_decision for BTC+ETH+SOL+XRP",
        "all_four_already_broken_version": "v1",
        "selected_broken_peer_intact_flag": bool(sel_broken is True and intact_peers) if isinstance(sel_broken, bool) else None,
        "peer_minus_selected_distance_bps": peer_minus,
    }


def merge_4h_entry_extras(extras: dict[str, Any], decision: dict[str, Any], *, redis_client: Any = None) -> dict[str, Any]:
    """Attach 4H telemetry. Never overwrite ranking / selection keys."""
    out = dict(extras or {})
    try:
        payload = collect_4h_entry_telemetry(decision, redis_client=redis_client)
    except Exception as exc:
        logger.debug("4h entry extras failed: %s", exc)
        return out
    for key, value in payload.items():
        if key in PROTECTED:
            continue
        out[key] = value
    return out


__all__ = [
    "PROTECTED",
    "TELEMETRY_KEYS",
    "collect_4h_entry_telemetry",
    "merge_4h_entry_extras",
]
