"""DAY paper authority: four-coin path-EV vs HOLD(0).

Old 15m rank does not select. No hybrid. HOLD EV is exactly 0.
Does not change SCALP. Does not change DAY exits.
"""

from __future__ import annotations

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


def score_four_coins(*, db_path: str = "") -> dict[str, Any]:
    """Score BTC/ETH/SOL/XRP independently. Missing prediction is HOLD (0), not invented."""
    art = load_accepted_day_artifact()
    evs: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for api in DAY_TRADE_SYMBOLS:
        key = _coin_key(api)
        pred, stamped = resolve_day_path_ev({}, symbol=api, db_path=db_path)
        if pred is None:
            evs[key] = HOLD_EV
            statuses[key] = "unavailable_hold"
        else:
            evs[key] = float(pred)
            statuses[key] = str(stamped.get("path_net_status") or "predicted")
    return {
        "btc_path_ev": float(evs.get("btc", HOLD_EV)),
        "eth_path_ev": float(evs.get("eth", HOLD_EV)),
        "sol_path_ev": float(evs.get("sol", HOLD_EV)),
        "xrp_path_ev": float(evs.get("xrp", HOLD_EV)),
        "hold_ev": HOLD_EV,
        "statuses": statuses,
        "path_net_model_id": (art.version if art is not None else DAY_PATH_MODEL_VERSION),
        "model_trained_at": (art.trained_at if art is not None else ""),
        "horizon_minutes": (int(art.primary_horizon_min) if art is not None else None),
        "costs_bps": round(float(ESTIMATED_ROUNDTRIP_COST) * 1e4, 4),
        "path_net_status": "predicted" if art is not None else "unavailable_hold",
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
    pairs = [
        ("BTCUSDT", float(scores.get("btc_path_ev") or HOLD_EV)),
        ("ETHUSDT", float(scores.get("eth_path_ev") or HOLD_EV)),
        ("SOLUSDT", float(scores.get("sol_path_ev") or HOLD_EV)),
        ("XRPUSDT", float(scores.get("xrp_path_ev") or HOLD_EV)),
        (HOLD_ACTION, hold_ev),
    ]
    winner_name, winner_ev = max(pairs, key=lambda p: (p[1], 0 if p[0] == HOLD_ACTION else 1))
    if winner_ev <= hold_ev:
        selected_action = HOLD_ACTION
        selected_symbol = ""
        selected_ev = hold_ev
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
