"""DAY paper authority: four-coin path-EV vs HOLD(0).

Old 15m rank does not select. No hybrid. HOLD EV is exactly 0.
Does not change SCALP. Does not change DAY exits.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.day_path_net import (
    DAY_PATH_MODEL_VERSION,
    load_accepted_day_artifact,
    resolve_day_path_ev,
)

DAY_AUTHORITY_MODE = "direct_four_coin_path_ev"
DAY_POLICY_ID = "day_path_aware_v1"
HOLD_ACTION = "HOLD"
HOLD_EV = 0.0
OLD_RANK_EXECUTION_AUTHORITY = False

_COIN_KEYS = ("btc", "eth", "sol", "xrp")


def _api_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def _slash_symbol(api: str) -> str:
    s = _api_symbol(api)
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return s


def _coin_key(api: str) -> str:
    s = _api_symbol(api)
    if s.endswith("USDT"):
        s = s[:-4]
    return s.lower()


def _opt_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = payload.get(key)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isnan(val):
            return val
    return None


def post_cost_economics_ev(decision_data: dict[str, Any] | None) -> float | None:
    """Reconstruct post-cost EV from candidate fields. None if not identifiable.

    Used to keep ranking honest: a path-net score cannot beat HOLD when the
    candidate's own expected move after fees/slip/spread is non-positive.
    Does not invent an edge from buy_margin or regime.
    """
    dd = decision_data or {}
    efe = _opt_float(dd, "expected_favorable_excursion", "estimated_mfe", "estimated_win_pct")
    eae = _opt_float(dd, "expected_adverse_excursion", "estimated_mae", "estimated_loss_pct")
    if efe is None or eae is None:
        return None
    p_buy = _opt_float(dd, "prob_buy", "winner_probability")
    p_sell = _opt_float(dd, "prob_sell")
    if p_buy is None:
        p_buy = 0.5
    if p_sell is None:
        p_sell = max(0.0, 1.0 - float(p_buy))
    p_hold = max(0.0, _opt_float(dd, "prob_hold") or 0.0)
    total = float(p_buy) + float(p_sell) + float(p_hold)
    if total > 0:
        p_buy = float(p_buy) / total
        p_sell = float(p_sell) / total
    fees = max(0.0, _opt_float(dd, "estimated_fees_pct") or 0.0)
    slip = max(0.0, _opt_float(dd, "estimated_slippage_pct") or 0.0)
    spread = max(0.0, _opt_float(dd, "spread_cost_pct", "spread_pct") or 0.0)
    if fees == 0.0 and slip == 0.0 and spread == 0.0:
        fees = float(ESTIMATED_ROUNDTRIP_COST)
    return float(p_buy) * float(efe) - float(p_sell) * abs(float(eae)) - fees - slip - spread


def score_four_coins(*, db_path: str = "") -> dict[str, Any]:
    """Score BTC/ETH/SOL/XRP independently. Invalid timing cannot win.

    Live EV keeps legacy btc_ret_5=0. Corrected BTC-relative EV is shadow only.
    """
    art = load_accepted_day_artifact()
    evs: dict[str, float] = {}
    statuses: dict[str, str] = {}
    valid: dict[str, bool] = {}
    by_symbol: dict[str, dict[str, Any]] = {}
    shadow_evs: dict[str, float | None] = {}
    for api in DAY_TRADE_SYMBOLS:
        key = _coin_key(api)
        pred, stamped = resolve_day_path_ev({"path_as_of_now": True}, symbol=api, db_path=db_path)
        ok = bool(stamped.get("path_input_valid")) and str(stamped.get("path_net_status") or "") == "predicted"
        valid[key] = ok
        statuses[key] = "predicted" if ok else str(stamped.get("path_invalid_reason") or stamped.get("path_net_status") or "unavailable_hold")
        evs[key] = float(pred) if ok and pred is not None else HOLD_EV
        shadow_evs[key] = stamped.get("shadow_correct_btc_path_ev")
        by_symbol[_api_symbol(api)] = {
            "path_input_valid": ok,
            "path_invalid_reason": None if ok else statuses[key],
            "path_row_count": stamped.get("path_row_count"),
            "path_first_bar_ts": stamped.get("path_first_bar_ts"),
            "path_last_bar_ts": stamped.get("path_last_bar_ts"),
            "path_actual_lookback_seconds": stamped.get("path_actual_lookback_seconds"),
            "path_max_gap_seconds": stamped.get("path_max_gap_seconds"),
            "path_latest_bar_age_seconds": stamped.get("path_latest_bar_age_seconds"),
            "path_model_version": stamped.get("path_model_version") or (art.version if art is not None else DAY_PATH_MODEL_VERSION),
            "path_feature_schema_version": stamped.get("path_feature_schema_version"),
            "legacy_btc_ret_5": stamped.get("legacy_btc_ret_5"),
            "correct_btc_ret_5": stamped.get("correct_btc_ret_5"),
            "legacy_path_ev": stamped.get("legacy_path_ev"),
            "shadow_correct_btc_path_ev": stamped.get("shadow_correct_btc_path_ev"),
            "path_max_abs_z": stamped.get("path_max_abs_z"),
            "path_ood_feature_count_at_4": stamped.get("path_ood_feature_count_at_4"),
            "path_ood_feature_count_at_6": stamped.get("path_ood_feature_count_at_6"),
            "path_ood_feature_count_at_8": stamped.get("path_ood_feature_count_at_8"),
            "path_outside_training_minmax_count": stamped.get("path_outside_training_minmax_count"),
        }
    any_valid = any(valid.values())
    shadow_pairs = [(k, v) for k, v in shadow_evs.items() if v is not None and valid.get(k)]
    if shadow_pairs:
        shadow_win_key = max(shadow_pairs, key=lambda item: (float(item[1]), 1))[0]
        shadow_winner = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT", "xrp": "XRPUSDT"}[shadow_win_key]
        if float(shadow_evs[shadow_win_key] or 0) <= HOLD_EV:
            shadow_winner = HOLD_ACTION
    else:
        shadow_winner = HOLD_ACTION
    return {
        "btc_path_ev": float(evs.get("btc", HOLD_EV)),
        "eth_path_ev": float(evs.get("eth", HOLD_EV)),
        "sol_path_ev": float(evs.get("sol", HOLD_EV)),
        "xrp_path_ev": float(evs.get("xrp", HOLD_EV)),
        "hold_ev": HOLD_EV,
        "statuses": statuses,
        "valid": valid,
        "path_input_by_symbol": by_symbol,
        "shadow_correct_btc_winner": shadow_winner,
        "path_net_model_id": (art.version if art is not None else DAY_PATH_MODEL_VERSION),
        "model_trained_at": (art.trained_at if art is not None else ""),
        "horizon_minutes": (int(art.primary_horizon_min) if art is not None else None),
        "costs_bps": round(float(ESTIMATED_ROUNDTRIP_COST) * 1e4, 4),
        "path_net_status": "predicted" if any_valid else "path_input_invalid",
        "model_accuracy": None,
    }


def select_action(
    scores: dict[str, Any],
    *,
    old_rank_nominee: str = "",
    old_rank_score: float | None = None,
) -> dict[str, Any]:
    """Highest EV among four coins and HOLD(0). Old rank is telemetry only."""
    hold_ev = HOLD_EV
    valid_map = scores.get("valid") if isinstance(scores.get("valid"), dict) else None
    pairs: list[tuple[str, float]] = []
    for api, key in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol"), ("XRPUSDT", "xrp")):
        if valid_map is not None and not valid_map.get(key, False):
            continue
        pairs.append((api, float(scores.get(f"{key}_path_ev") or HOLD_EV)))
    pairs.append((HOLD_ACTION, hold_ev))
    winner_name, winner_ev = max(pairs, key=lambda p: (p[1], 0 if p[0] == HOLD_ACTION else 1))
    if winner_ev <= hold_ev:
        selected_action = HOLD_ACTION
        selected_symbol = ""
        selected_ev = hold_ev
        if valid_map is not None and not any(valid_map.values()):
            why = "PATH_INPUT_INVALID"
        else:
            why = "HOLD_WINS"
        path_ev_winner = HOLD_ACTION
    else:
        selected_action = f"BUY_{winner_name}"
        selected_symbol = winner_name
        selected_ev = float(winner_ev)
        why = "PATH_NET_BEATS_HOLD"
        path_ev_winner = winner_name
    now = datetime.now(timezone.utc).isoformat()
    return {
        "day_authority_mode": DAY_AUTHORITY_MODE,
        "old_rank_execution_authority": OLD_RANK_EXECUTION_AUTHORITY,
        "old_rank_nominee": _api_symbol(old_rank_nominee) if old_rank_nominee else "",
        "old_rank_score": old_rank_score,
        "btc_path_ev": float(scores.get("btc_path_ev") or HOLD_EV),
        "eth_path_ev": float(scores.get("eth_path_ev") or HOLD_EV),
        "sol_path_ev": float(scores.get("sol_path_ev") or HOLD_EV),
        "xrp_path_ev": float(scores.get("xrp_path_ev") or HOLD_EV),
        "hold_ev": hold_ev,
        "path_ev_winner": path_ev_winner,
        "selected_action": selected_action,
        "selected_symbol": selected_symbol,
        "selected_ev": selected_ev,
        "path_net_model_id": scores.get("path_net_model_id") or DAY_PATH_MODEL_VERSION,
        "path_aware_policy_id": DAY_POLICY_ID,
        "model_trained_at": scores.get("model_trained_at") or "",
        "model_accuracy": scores.get("model_accuracy"),
        "prediction_timestamp": now,
        "costs_bps": scores.get("costs_bps"),
        "horizon_minutes": scores.get("horizon_minutes"),
        "true_safety_reject_reason": None,
        "why_selected": why,
        "path_net_status": scores.get("path_net_status") or "predicted",
        "forward_net_model_version": scores.get("path_net_model_id") or DAY_PATH_MODEL_VERSION,
        "selected_net_expected_value": selected_ev,
        "predicted_net_return": selected_ev,
        "hold_action_ev": hold_ev,
        "valid": valid_map,
        "path_input_by_symbol": scores.get("path_input_by_symbol") or {},
        "legacy_winner": path_ev_winner,
        "shadow_correct_btc_winner": scores.get("shadow_correct_btc_winner"),
        "winner_disagreement": bool(scores.get("shadow_correct_btc_winner") and scores.get("shadow_correct_btc_winner") != path_ev_winner),
    }


def old_rank_telemetry(candidates: list[Any] | None) -> tuple[str, float | None]:
    """Best old-rank nominee for stamps only. Never used to pick."""
    rows = list(candidates or [])
    if not rows:
        return "", None

    def _score(c: Any) -> float:
        dd = getattr(c, "decision_data", None) or {}
        try:
            return float(dd.get("final_selection_score") or dd.get("selection_score") or c.rank_score())
        except Exception:
            return 0.0

    top = max(rows, key=_score)
    return _api_symbol(getattr(top, "symbol", "") or ""), _score(top)


def decide_day_bar(*, db_path: str = "", candidates: list[Any] | None = None) -> dict[str, Any]:
    nominee, score = old_rank_telemetry(candidates)
    scores = score_four_coins(db_path=db_path)
    return select_action(scores, old_rank_nominee=nominee, old_rank_score=score)
