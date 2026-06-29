"""SCALP feature health guards for learning (soft only — no gates)."""

from __future__ import annotations

import json
from typing import Any

from backend.services.scalp_feature_audit import BAD_STATUSES, build_feature_health_sidecar
from backend.services.scalp_feature_contract import SCALP_FEATURE_DIM, _block_for_index

LEARNING_BLOCKED_FEATURE_NAMES: frozenset[str] = frozenset()


def entry_feature_health_pass(intelligence: dict[str, Any] | None) -> bool:
    ex = intelligence or {}
    if str(ex.get("feature_health_pass")).lower() in ("1", "true", "yes"):
        return True
    if ex.get("feature_health_pass") is True:
        return True
    raw = ex.get("feature_health_json")
    if not raw:
        score = float(ex.get("feature_health_score") or ex.get("scalp_feature_health_score") or 0.0)
        return score >= 0.55
    try:
        side = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return bool(side.get("pass")) or float(side.get("health_pct") or 0) >= 55.0
    except Exception:
        return False


def stamp_feature_health(intelligence: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    side = build_feature_health_sidecar(audit)
    out = dict(intelligence or {})
    out["feature_health_json"] = json.dumps(side, separators=(",", ":"))
    out["feature_health_score"] = round(float(side.get("health_pct") or 0) / 100.0, 4)
    out["scalp_feature_health_score"] = out["feature_health_score"]
    out["feature_health_pass"] = side.get("pass", False)
    out["feature_version"] = side.get("feature_version")
    out["feature_dim"] = SCALP_FEATURE_DIM
    out["entry_scalp_vector"] = [
        float(f.get("value") or 0.0) for f in (audit.get("features") or [])[:SCALP_FEATURE_DIM]
    ]
    return out


def learning_allowed_for_feature(row: dict[str, Any]) -> bool:
    st = str(row.get("status") or "")
    if st in BAD_STATUSES or st == "WARMUP":
        return False
    if st == "CALCULATED_PROXY":
        return False
    return bool(row.get("learning_allowed", False))


__all__ = [
    "LEARNING_BLOCKED_FEATURE_NAMES",
    "entry_feature_health_pass",
    "learning_allowed_for_feature",
    "stamp_feature_health",
]
