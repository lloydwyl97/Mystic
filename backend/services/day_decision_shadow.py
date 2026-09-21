"""Shadow DAY decision contract. Persistence only — never changes the action."""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.config.execution_cost_model import named_cost_breakdown
from backend.services.day_direct_path_ev_authority import HOLD_ACTION, HOLD_EV

logger = logging.getLogger(__name__)

SHADOW_SCHEMA_VERSION = "day_shadow_v1"
CHALLENGER_STATUS_NOT_PROMOTED = "not_promoted"


def _f(payload: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def champion_shadow_fields(decision: dict[str, Any]) -> dict[str, Any]:
    """Observability fields derived from an already-chosen DAY decision.

    Does not pick a coin. Does not size. Does not mutate selected_action.
    """
    dec = dict(decision or {})
    selected = str(dec.get("selected_action") or HOLD_ACTION)
    symbol = str(dec.get("selected_symbol") or "")
    ev = _f(dec, "selected_ev", "selected_net_expected_value", "predicted_net_return", default=HOLD_EV) or HOLD_EV
    costs = None
    if symbol:
        costs = named_cost_breakdown(symbol, p_buy=float(dec.get("prob_buy") or 0.0), predicted_gross=ev)
    comm = costs.expected_exchange_commission if costs else 0.0
    spread = costs.expected_spread if costs else 0.0
    slip = costs.expected_slippage if costs else 0.0
    pred_gross = ev + comm + spread + slip if symbol else 0.0
    pred_net = ev if symbol else HOLD_EV
    return {
        "shadow_schema": SHADOW_SCHEMA_VERSION,
        "champion_selected_action": selected,
        "champion_selected_symbol": symbol or HOLD_ACTION,
        "champion_predicted_gross_bps": round(pred_gross * 1e4, 4),
        "champion_expected_commission_bps": round(comm * 1e4, 4),
        "champion_expected_spread_bps": round(spread * 1e4, 4),
        "champion_expected_slippage_bps": round(slip * 1e4, 4),
        "champion_predicted_net_bps": round(pred_net * 1e4, 4),
        "champion_predicted_net_usd": None,
        "champion_downside_bps": None,
        "champion_hold_value": HOLD_EV,
        "challenger_status": CHALLENGER_STATUS_NOT_PROMOTED,
        "challenger_selected_action": None,
        "model_version": dec.get("path_net_model_id") or dec.get("forward_net_model_version"),
        "calibration_version": None,
        "data_freshness": dec.get("prediction_timestamp"),
    }


def merge_shadow_extras(extras: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Add shadow keys. Never overwrite selected_action / path_ev_winner."""
    out = dict(extras or {})
    try:
        shadow = champion_shadow_fields(decision)
    except Exception as exc:
        logger.debug("day shadow fields failed: %s", exc)
        return out
    protected = {
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
    for key, value in shadow.items():
        if key in protected:
            continue
        out[key] = value
    try:
        from backend.services.day_4h_entry_telemetry import merge_4h_entry_extras

        out = merge_4h_entry_extras(out, decision)
    except Exception as exc:
        logger.debug("day 4h entry extras failed: %s", exc)
    return out


def dump_shadow_extras(extras: dict[str, Any], decision: dict[str, Any]) -> str:
    return json.dumps(merge_shadow_extras(extras, decision), default=str)
