"""Causal attribution of the governed DAY ML model to live DAY decisions.

DAY_V2 live entries are chosen by setup detection, the structural objective
edge (``day_v2.ranking``) and ``adaptive_learning.day_decision``. The governed
per-coin artifact publishes ``ai_signal:day:{SYMBOL}`` but none of those inputs
read it, so its contribution to score, rank, size and the selected action is
zero. Each decision records that, with the serving model version and its output
at decision time, so an outcome can only ever be credited to the model when the
decision itself says the model moved it.

Only attributable outcomes may feed model rollback. Everything else is telemetry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ATTRIBUTION_VERSION = "day_ml_attribution_v1"
ATTRIBUTION_KEY = "ml_model_attribution"
DAY_V2_DECISION_INPUTS = ("setup_detection", "structural_objective_edge", "adaptive_learning")
NOT_AN_INPUT_REASON = "DAY_V2_DECISION_DOES_NOT_READ_ML_MODEL"
_OUTPUT_FIELDS = ("prediction", "argmax_action", "prob_buy", "prob_hold", "prob_sell", "confidence", "model_trained_at")


def _bus(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    return s if s.endswith("USDT") else f"{s}USDT"


def serving_model_version(symbol: str, *, strategy_id: str = "day", root: Path | None = None) -> str:
    try:
        from backend.services import ai_model_registry as registry

        return str(registry.read_pointer(strategy_id, _bus(symbol), "ACTIVE", root).get("version") or "")
    except Exception:
        return ""


def serving_model_output(symbol: str) -> dict[str, Any] | None:
    try:
        from backend.config.redis_config import get_shared_redis_sync
        from backend.services.live_strategy_contracts import redis_ai_signal_key

        r = get_shared_redis_sync()
        raw = (r.hgetall(redis_ai_signal_key("day", _bus(symbol))) or {}) if r is not None else {}
    except Exception:
        return None
    decoded = {(k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
    out = {key: decoded[key] for key in _OUTPUT_FIELDS if key in decoded}
    return out or None


def day_v2_decision_attribution(
    symbol: str,
    *,
    model_version: str | None = None,
    model_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attribution stamp for one DAY_V2 live decision."""
    return {
        "attribution_version": ATTRIBUTION_VERSION,
        "model_version": serving_model_version(symbol) if model_version is None else str(model_version),
        "model_output": serving_model_output(symbol) if model_output is None else dict(model_output),
        "model_output_consumed": False,
        "score_contribution": 0.0,
        "rank_contribution": 0.0,
        "size_contribution": 0.0,
        "changed_selected_action": False,
        "attributable": False,
        "decision_inputs": list(DAY_V2_DECISION_INPUTS),
        "reason": NOT_AN_INPUT_REASON,
    }


def _record(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload or "{}")
        except (TypeError, ValueError):
            return {}
    if not isinstance(payload, dict):
        return {}
    rec = payload.get(ATTRIBUTION_KEY, payload)
    if isinstance(rec, str):
        try:
            rec = json.loads(rec)
        except (TypeError, ValueError):
            return {}
    return rec if isinstance(rec, dict) else {}


def attributable_to(payload: Any, model_version: str) -> bool:
    """True only when the decision recorded that ``model_version`` changed it."""
    rec = _record(payload)
    if not rec or rec.get("attributable") is not True or not model_version:
        return False
    if str(rec.get("model_version") or "") != str(model_version):
        return False
    moved = any(abs(float(rec.get(k) or 0.0)) > 0.0 for k in ("score_contribution", "rank_contribution", "size_contribution"))
    return moved or rec.get("changed_selected_action") is True


__all__ = [
    "ATTRIBUTION_KEY",
    "ATTRIBUTION_VERSION",
    "attributable_to",
    "day_v2_decision_attribution",
    "serving_model_output",
    "serving_model_version",
]
