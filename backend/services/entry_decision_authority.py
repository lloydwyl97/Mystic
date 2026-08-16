"""Entry-decision provenance. Persistence and reporting only.

Does not select trades. Does not change EV, rank, or exits.
Stamps what actually decided a BUY so clean-runtime and
model-controlled books can be counted separately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DAY_ACCEPTED_MODEL = "day_path_net_v1"
SCALP_ACCEPTED_MODEL = "scalp_path_net_v1"
DAY_POLICY = "day_path_aware_v1"
SCALP_POLICY = "scalp_path_aware_v1"
LEGACY_DIRECTION = "OLD_DIRECTION_MODEL"
LEGACY_SOFT_RANK = "LEGACY_SOFT_RANK"
HOLD_EV = 0.0

PROVENANCE_KEYS = (
    "entry_policy_version",
    "model_version",
    "prediction_timestamp",
    "predicted_net_return",
    "p_positive_net",
    "predicted_mfe",
    "predicted_mae",
    "predicted_horizon",
    "hold_ev",
    "buy_ev",
    "selected_action",
    "selection_reason",
    "source_opportunity_id",
    "soft_rank_entry",
    "strategy",
    "setup",
    "feature_fingerprint",
    "path_net_status",
    "forward_net_model_version",
    "direction_model_probability",
    "legacy_rank_score",
    "final_decision_function",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_provenance(payload: dict[str, Any] | None) -> dict[str, Any]:
    src = payload if isinstance(payload, dict) else {}
    return {key: src.get(key) for key in PROVENANCE_KEYS}


def copy_entry_provenance(entry: dict[str, Any] | None, dest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy entry stamps onto a SELL/round-trip payload. Does not overwrite old rows."""
    out = dict(dest or {})
    for key, value in extract_provenance(entry).items():
        if value in (None, "") and key in out:
            continue
        if value not in (None, ""):
            out[key] = value
    return out


def is_model_controlled(payload: dict[str, Any] | None, *, engine: str) -> bool:
    """True only when the accepted predictor is stamped as the BUY authority."""
    src = payload if isinstance(payload, dict) else {}
    version = str(src.get("model_version") or src.get("forward_net_model_version") or "")
    policy = str(src.get("entry_policy_version") or "")
    action = str(src.get("selected_action") or "")
    buy_ev = _num(src.get("buy_ev") if src.get("buy_ev") not in (None, "") else src.get("predicted_net_return"))
    hold_ev = _num(src.get("hold_ev"))
    pred_ts = str(src.get("prediction_timestamp") or "")
    if not policy or not version or not action or not pred_ts:
        return False
    if hold_ev is None or buy_ev is None:
        return False
    if not action.upper().startswith("BUY"):
        return False
    if buy_ev <= hold_ev:
        return False
    if engine == "day":
        return (
            version == DAY_ACCEPTED_MODEL
            and policy == DAY_POLICY
            and str(src.get("path_net_status") or "") == "predicted"
        )
    if engine == "scalp":
        return version == SCALP_ACCEPTED_MODEL and policy == SCALP_POLICY
    return False


def build_day_entry_provenance(
    *,
    decision_data: dict[str, Any] | None,
    symbol: str = "",
    decision_id: str = "",
    bar_timestamp: Any = None,
    rank_score: Any = None,
    why_selected: str = "",
    direction_probability: Any = None,
    feature_fingerprint: str = "",
) -> dict[str, Any]:
    dd = dict(decision_data or {})
    status = str(dd.get("path_net_status") or "")
    version = str(dd.get("forward_net_model_version") or "")
    buy_ev = _num(dd.get("selected_net_expected_value") if dd.get("selected_net_expected_value") not in (None, "") else dd.get("predicted_net_return"))
    p_pos = _num(dd.get("p_positive_net") if dd.get("p_positive_net") not in (None, "") else dd.get("predicted_prob_positive_net"))
    if status == "predicted" and version == DAY_ACCEPTED_MODEL and buy_ev is not None:
        policy = DAY_POLICY
        model_version = version
        if buy_ev > HOLD_EV:
            selected_action = "BUY"
            selection_reason = "PATH_NET_BEATS_HOLD"
        else:
            selected_action = "HOLD"
            selection_reason = "HOLD_WINS"
    elif status in ("unavailable_hold", "error_hold") and version == DAY_ACCEPTED_MODEL:
        policy = DAY_POLICY
        model_version = version
        selected_action = "HOLD"
        selection_reason = "PATH_NET_UNAVAILABLE_HOLD"
    else:
        policy = LEGACY_DIRECTION
        model_version = version or LEGACY_DIRECTION
        selected_action = "BUY"
        selection_reason = str(why_selected or "RANK_THEN_POSITIVE_EV_UNPROVEN")
    opp = str(decision_id or "").strip() or f"{bar_timestamp}:{symbol}"
    return {
        "entry_policy_version": policy,
        "model_version": model_version,
        "prediction_timestamp": _now_iso(),
        "predicted_net_return": buy_ev,
        "p_positive_net": p_pos,
        "predicted_mfe": _num(dd.get("predicted_mfe") if dd.get("predicted_mfe") not in (None, "") else dd.get("expected_mfe")),
        "predicted_mae": _num(dd.get("predicted_mae") if dd.get("predicted_mae") not in (None, "") else dd.get("expected_mae")),
        "predicted_horizon": dd.get("day_path_horizon_min") if dd.get("day_path_horizon_min") not in (None, "") else dd.get("predicted_horizon"),
        "hold_ev": HOLD_EV,
        "buy_ev": buy_ev,
        "selected_action": selected_action,
        "selection_reason": selection_reason,
        "source_opportunity_id": opp,
        "soft_rank_entry": False,
        "strategy": str(dd.get("live_ai_strategy") or "day"),
        "setup": str(dd.get("setup_type") or dd.get("entry_thesis") or "day"),
        "feature_fingerprint": feature_fingerprint or str(dd.get("feature_fingerprint") or ""),
        "path_net_status": status,
        "forward_net_model_version": version,
        "direction_model_probability": _num(direction_probability if direction_probability not in (None, "") else dd.get("prob_buy")),
        "legacy_rank_score": _num(rank_score if rank_score not in (None, "") else dd.get("final_selection_score")),
        "final_decision_function": "portfolio_engine.process_bar_candidates",
    }


def build_scalp_entry_provenance(
    *,
    ranking_meta: dict[str, Any] | None,
    symbol: str = "",
    setup_name: str = "",
    strategy_passed: bool = False,
    epoch: Any = None,
    opportunity_id: Any = None,
    feature_fingerprint: str = "",
) -> dict[str, Any]:
    meta = dict(ranking_meta or {})
    version = str(meta.get("forward_net_model_version") or "")
    buy_ev = _num(meta.get("selected_expected_net_ev") if meta.get("selected_expected_net_ev") not in (None, "") else meta.get("expected_net_ev"))
    hold_ev = _num(meta.get("hold_action_ev"))
    if hold_ev is None:
        hold_ev = HOLD_EV
    soft = not bool(strategy_passed)
    if version == SCALP_ACCEPTED_MODEL and buy_ev is not None and buy_ev > hold_ev:
        policy = SCALP_POLICY
        selected_action = f"BUY_{symbol}" if symbol else "BUY"
        selection_reason = "PATH_NET_BEATS_HOLD"
        model_version = version
    elif version == SCALP_ACCEPTED_MODEL:
        policy = SCALP_POLICY
        selected_action = "HOLD"
        selection_reason = "HOLD_WINS_ACTION_RANK"
        model_version = version
    else:
        policy = LEGACY_SOFT_RANK if soft else "scalp_ranking_not_gating_v2"
        selected_action = f"BUY_{symbol}" if symbol else "BUY"
        selection_reason = str(meta.get("selection_reason") or "LEGACY_RANK_SCORE")
        model_version = version or LEGACY_SOFT_RANK
    epoch_ms = ""
    try:
        if epoch not in (None, ""):
            epoch_ms = str(int(float(epoch) * 1000.0))
    except (TypeError, ValueError):
        epoch_ms = str(epoch or "")
    opp = str(opportunity_id or "").strip() or f"{epoch_ms}:{symbol}:{setup_name}"
    return {
        "entry_policy_version": policy,
        "model_version": model_version,
        "prediction_timestamp": _now_iso(),
        "predicted_net_return": buy_ev,
        "p_positive_net": _num(meta.get("selected_predicted_prob_positive_net") if meta.get("selected_predicted_prob_positive_net") not in (None, "") else meta.get("predicted_prob_positive_net")),
        "predicted_mfe": _num(meta.get("selected_expected_mfe") if meta.get("selected_expected_mfe") not in (None, "") else meta.get("expected_mfe")),
        "predicted_mae": _num(meta.get("selected_expected_mae") if meta.get("selected_expected_mae") not in (None, "") else meta.get("expected_mae")),
        "predicted_horizon": meta.get("selected_expected_hold") if meta.get("selected_expected_hold") not in (None, "") else meta.get("expected_hold"),
        "hold_ev": hold_ev,
        "buy_ev": buy_ev,
        "selected_action": selected_action,
        "selection_reason": selection_reason,
        "source_opportunity_id": opp,
        "soft_rank_entry": soft,
        "strategy": setup_name or "scalp",
        "setup": setup_name or "",
        "feature_fingerprint": feature_fingerprint,
        "path_net_status": "predicted" if version == SCALP_ACCEPTED_MODEL and buy_ev is not None else "",
        "forward_net_model_version": version,
        "direction_model_probability": None,
        "legacy_rank_score": _num(meta.get("rank_score")),
        "final_decision_function": "scalp_candidate_ranking.pick_best_global_candidate",
    }


def summarize_book(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    wins = sum(1 for x in rows if float(x.get("pnl") or 0) > 0)
    net = sum(float(x.get("pnl") or 0) for x in rows)
    return {
        "n": n,
        "wins": wins,
        "wr": None if n == 0 else round(wins / n, 4),
        "net": round(net, 4),
        "expectancy": None if n == 0 else round(net / n, 6),
        "rows": rows,
    }
