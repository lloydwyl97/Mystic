"""Feature health metadata helpers for DAY v5 (145-dim). Sidecar only — does not alter model input."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1, AI_FEATURE_DIM_V2, CONTEXT_DIMS_DAY_FULL, FEATURE_VERSION_CURRENT
from backend.services.day_feature_audit import BAD_STATUSES, build_context_provenance, build_symbol_feature_audit
from backend.services.feature_builder import FEATURE_TRUST_SCORES

# volume_delta / order_flow are computed as volume * sign(close - open) — an OHLCV
# proxy for real trade-tape order flow, not the real thing (see feature_builder.py
# comment at their construction site). They carry the same "explicit proxy, not the
# real signal" caveat as put_call_ratio/volatility_smile below, so they're excluded
# from learning eligibility on the same basis, even though CALCULATED_PROXY status
# alone does not fail the entry feature-health gate (that's intentional — DAY has no
# real tape access, so the proxy is still the best available value to trade on).
LEARNING_BLOCKED_FEATURE_NAMES: frozenset[str] = frozenset({"put_call_ratio", "volatility_smile", "volume_delta", "order_flow", "volume_imbalance"})

# Sidecar display aliases — RF index names stay unchanged; audits/UI see the proxy label.
PROXY_DISPLAY_NAMES: dict[str, str] = {
    "volume_delta": "volume_delta_ohlcv_proxy",
    "order_flow": "order_flow_ohlcv_proxy",
    "volume_imbalance": "volume_imbalance_ohlcv_proxy",
}

_BAD_FOR_PASS: frozenset[str] = frozenset({"FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT"})


def _feature_name_at(idx0: int) -> str:
    if idx0 < AI_FEATURE_DIM_V1:
        from backend.services.feature_mapping import get_feature_name

        return get_feature_name(idx0 + 1)
    ctx_i = idx0 - AI_FEATURE_DIM_V1
    if 0 <= ctx_i < len(CONTEXT_DIMS_DAY_FULL):
        return CONTEXT_DIMS_DAY_FULL[ctx_i]
    return f"unknown_{idx0 + 1}"


def _block_for_index(idx0: int) -> str:
    from backend.services.day_feature_audit import _block_for_index as _blk

    return _blk(idx0)


def build_compact_health_sidecar(
    vector: list[float],
    tech_provenance: dict[str, dict[str, Any]],
    ctx_provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compact health sidecar for Redis / explainability (145 rows, no model input change)."""
    rows: list[dict[str, Any]] = []
    bad: list[str] = []
    learning_blocked = sorted(LEARNING_BLOCKED_FEATURE_NAMES)
    live = calc = proxy = 0

    for idx0 in range(min(len(vector), AI_FEATURE_DIM_V2)):
        name = _feature_name_at(idx0)
        meta = dict((ctx_provenance if idx0 >= AI_FEATURE_DIM_V1 else tech_provenance).get(name) or {})
        status = str(meta.get("status") or "MISSING")
        trust = float(meta.get("trust_score") if meta.get("trust_score") is not None else FEATURE_TRUST_SCORES.get(status, 0.0))
        learning = bool(meta.get("learning_allowed", status not in BAD_STATUSES and name not in LEARNING_BLOCKED_FEATURE_NAMES))
        if name in LEARNING_BLOCKED_FEATURE_NAMES:
            learning = False
        if status == "LIVE":
            live += 1
        elif status == "CALCULATED":
            calc += 1
        elif status == "CALCULATED_PROXY":
            proxy += 1
        if status in _BAD_FOR_PASS or (status == "WARMUP" and abs(float(vector[idx0])) < 1e-12):
            bad.append(name)
        display_name = PROXY_DISPLAY_NAMES.get(name, name)
        rows.append(
            {
                "index": idx0 + 1,
                "name": display_name,
                "canonical_name": name,
                "block": _block_for_index(idx0),
                "value": round(float(vector[idx0]), 8),
                "status": status,
                "source": str(meta.get("source") or "")[:120],
                "trust_score": round(trust, 4),
                "learning_allowed": learning,
            }
        )

    total = len(rows)
    health_pct = round(100.0 * (live + calc + proxy) / max(1, total), 2)
    passed = len(bad) == 0 and total == AI_FEATURE_DIM_V2

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION_CURRENT,
        "feature_dim": AI_FEATURE_DIM_V2,
        "pass": passed,
        "health_pct": health_pct,
        "live_count": live,
        "calculated_count": calc,
        "calculated_proxy_count": proxy,
        "bad_count": len(bad),
        "bad_features": bad[:20],
        "learning_blocked_features": learning_blocked,
        "features": rows,
    }


def sidecar_redis_fields(sidecar: dict[str, Any]) -> dict[str, str]:
    """Flatten sidecar to Redis hash string fields."""
    return {
        "feature_health_pass": "1" if sidecar.get("pass") else "0",
        "feature_health_pct": str(sidecar.get("health_pct", 0.0)),
        "feature_health_bad_count": str(int(sidecar.get("bad_count") or 0)),
        "feature_health_json": json.dumps(sidecar, separators=(",", ":")),
    }


async def build_health_metadata_for_symbol(symbol_bus: str) -> list[dict[str, Any]]:
    """Return 145 health metadata rows for one symbol (full audit path)."""
    report = await build_symbol_feature_audit(symbol_bus)
    return list(report.get("features") or [])


def feature_allows_learning(feature_name: str, meta: dict[str, Any] | None = None) -> bool:
    if feature_name in LEARNING_BLOCKED_FEATURE_NAMES:
        return False
    if meta is None:
        return True
    return bool(meta.get("learning_allowed", True))


def zero_learning_blocked_feature_dims(feats: list[float] | tuple[float, ...]) -> list[float]:
    """Zero proxy/unsupported dims so train, serve, and holdout share one contract."""
    out = [float(x) for x in feats]
    for i in range(len(out)):
        try:
            name = _feature_name_at(i)
        except Exception:
            continue
        base = str(name or "").replace("_ohlcv_proxy", "")
        if base in LEARNING_BLOCKED_FEATURE_NAMES or name in LEARNING_BLOCKED_FEATURE_NAMES:
            out[i] = 0.0
    return out


def entry_feature_health_pass(explainability: dict[str, Any] | None) -> bool:
    """True when entry signal carried a passing feature health sidecar."""
    if not explainability:
        return True
    raw = explainability.get("feature_health_pass")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes")


__all__ = [
    "LEARNING_BLOCKED_FEATURE_NAMES",
    "build_compact_health_sidecar",
    "build_context_provenance",
    "build_health_metadata_for_symbol",
    "entry_feature_health_pass",
    "feature_allows_learning",
    "sidecar_redis_fields",
    "zero_learning_blocked_feature_dims",
]
