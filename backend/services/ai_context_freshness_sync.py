"""
Align ai_signal hash / decision_data context freshness with live ai_context Redis.

Fixes split-brain where TTL-preserve refreshes signal timestamp but leaves stale
context_fresh / ctx_ts_utc on the hash, causing ENTRY_CONTEXT_NOT_FRESH rejects
while live ai_context is healthy.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from backend.services.ai_decision_contract import MARKET_CONTEXT_FIELDS, REDIS_KEY_AI_CONTEXT
from backend.services.live_strategy_contracts import live_ai_fail_closed_without_context
from backend.services.strategy_runtime_audit import compute_context_freshness, get_ctx_fresh_max_age_sec

logger = logging.getLogger(__name__)


def _norm_bus(symbol: str) -> str:
    return (symbol or "").replace("/", "").strip().upper()


def read_live_context(symbol_bus: str) -> tuple[dict[str, Any], float, str]:
    """
    Read live ai_context:{BUS} from Redis.

    Returns:
        (payload, age_sec, ts_utc_str) — age_sec is inf when missing/unparseable.
    """
    bus = _norm_bus(symbol_bus)
    if not bus:
        return {}, float("inf"), ""
    try:
        from backend.config.redis_config import get_redis_client

        redis_client = get_redis_client()
        if not redis_client:
            return {}, float("inf"), ""
        raw = redis_client.hgetall(REDIS_KEY_AI_CONTEXT.format(symbol=bus)) or {}
        if not raw:
            return {}, float("inf"), ""
        payload: dict[str, Any] = {}
        for k, v in raw.items():
            kk = k.decode("utf-8", errors="ignore") if isinstance(k, bytes) else str(k)
            vv = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v
            payload[kk] = vv
        ts = str(payload.get("ts_utc") or payload.get("timestamp") or "").strip()
        age_sec = float("inf")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_sec = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds())
            except Exception:
                age_sec = float("inf")
        return payload, age_sec, ts
    except Exception as exc:
        logger.debug("read_live_context %s failed: %s", bus, exc)
        return {}, float("inf"), ""


def build_context_audit_emit(payload: dict[str, Any], *, age_sec: float, ts_str: str) -> str:
    """Mirror ai_signal_generator ctx_defaulted_audit blob."""
    _missing_cf = [f for f in MARKET_CONTEXT_FIELDS if f not in payload]
    age_at_emit = age_sec if math.isfinite(age_sec) else -1.0
    blob: dict[str, Any] = {
        "redis_ai_context_present": bool(payload),
        "redis_ctx_ts_utc": ts_str,
        "ctx_age_sec_at_emit": age_at_emit,
        "payload_field_count": len(payload),
        "live_ai_fail_closed_without_context": live_ai_fail_closed_without_context(),
        "missing_contract_fields": _missing_cf,
    }
    raw = json.dumps(blob, separators=(",", ":"))
    if len(raw) > 1800:
        blob["missing_contract_fields"] = _missing_cf[:16]
        raw = json.dumps(blob, separators=(",", ":"))
    return raw[:1800]


def overlay_live_context_freshness(dd: dict[str, Any], symbol_bus: str) -> list[str]:
    """
    Recompute ctx_ts_utc, ctx_age_sec, context_fresh from live ai_context.

    When feature_version >= 2, also refresh context_audit_emit from live payload.
    Returns patched field names.
    """
    if not dd:
        return []
    bus = _norm_bus(symbol_bus)
    payload, age_sec, ts_str = read_live_context(bus)
    patched: list[str] = []

    prior_fresh = str(dd.get("context_fresh") or dd.get("context_fresh_str") or "").strip()
    prior_ts = str(dd.get("ctx_ts_utc") or "").strip()

    if ts_str:
        dd["ctx_ts_utc"] = ts_str[:64]
        patched.append("ctx_ts_utc")
    if math.isfinite(age_sec):
        dd["ctx_age_sec"] = round(age_sec, 4)
        patched.append("ctx_age_sec")

    ctx_age_for_gate = age_sec if math.isfinite(age_sec) else None
    fresh, _ = compute_context_freshness(ctx_age_for_gate)
    cf = "1" if fresh else "0"
    dd["context_fresh"] = cf
    dd["context_fresh_str"] = cf
    patched.append("context_fresh")

    fv_raw = dd.get("feature_version", 1)
    try:
        fv = int(float(fv_raw)) if fv_raw not in (None, "") else 1
    except (TypeError, ValueError):
        fv = 1
    if fv >= 2 and payload:
        dd["context_audit_emit"] = build_context_audit_emit(payload, age_sec=age_sec, ts_str=ts_str)
        patched.append("context_audit_emit")

    if prior_fresh != cf or (prior_ts and ts_str and prior_ts != ts_str[:64]):
        logger.info(
            "CONTEXT_FRESHNESS_OVERLAY symbol=%s prior_fresh=%s new_fresh=%s prior_ts=%s live_ts=%s age_sec=%.1f limit=%.1f",
            bus,
            prior_fresh or None,
            cf,
            prior_ts[:32] if prior_ts else None,
            ts_str[:32] if ts_str else None,
            age_sec if math.isfinite(age_sec) else -1.0,
            get_ctx_fresh_max_age_sec(),
        )
    return patched


def apply_overlay_to_explainability(explainability: Any, dd: dict[str, Any]) -> None:
    """Copy overlay fields from decision_data dict onto TradeExplainability."""
    if dd.get("ctx_ts_utc"):
        explainability.ctx_ts_utc = str(dd["ctx_ts_utc"])[:64]
    if "ctx_age_sec" in dd:
        try:
            explainability.ctx_age_sec = float(dd["ctx_age_sec"])
        except (TypeError, ValueError):
            pass
    cf = dd.get("context_fresh_str") or dd.get("context_fresh")
    if cf is not None and str(cf).strip() in ("0", "1"):
        explainability.context_fresh_flag = str(cf).strip()
    if dd.get("context_audit_emit"):
        explainability.context_audit_emit = str(dd["context_audit_emit"])


def build_freshness_snapshot(symbols: list[str]) -> dict[str, Any]:
    """Per-symbol gate evaluation for telemetry (observation only)."""
    from backend.config.ai_signal_bus import MAX_SIGNAL_AGE_SEC
    from backend.services.ai_entry_context_gate import evaluate_signal_hash_for_entry
    from backend.services.live_strategy_contracts import redis_ai_signal_key

    rows: list[dict[str, Any]] = []
    try:
        from backend.config.redis_config import get_redis_client

        r = get_redis_client()
    except Exception:
        r = None

    for sym in symbols:
        bus = _norm_bus(sym)
        dd: dict[str, str] = {}
        if r:
            try:
                raw = r.hgetall(redis_ai_signal_key("day", bus)) or {}
                for k, v in raw.items():
                    kk = k.decode() if isinstance(k, bytes) else str(k)
                    vv = v.decode() if isinstance(v, bytes) else str(v)
                    dd[kk] = vv
            except Exception:
                pass
        payload, live_age, live_ts = read_live_context(bus)
        overlay_live_context_freshness(dd, bus)
        ok, reject, detail = evaluate_signal_hash_for_entry(dd) if dd else (False, "NO_SIGNAL_HASH", {})
        rows.append(
            {
                "symbol": bus,
                "gate_ok": ok,
                "reject_code": reject,
                "context_fresh": dd.get("context_fresh"),
                "ctx_age_sec": dd.get("ctx_age_sec"),
                "live_ctx_age_sec": round(live_age, 2) if math.isfinite(live_age) else None,
                "live_ctx_ts_utc": live_ts[:32] if live_ts else None,
                "content_fresh": dd.get("content_fresh"),
                "max_ctx_age_sec": get_ctx_fresh_max_age_sec(),
                "max_signal_age_sec": MAX_SIGNAL_AGE_SEC,
                "detail": {k: detail[k] for k in list(detail.keys())[:8]},
            }
        )
    blocked = [r for r in rows if not r.get("gate_ok")]
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": rows,
        "blocked_count": len(blocked),
        "blocked_codes": {r["reject_code"]: sum(1 for x in blocked if x.get("reject_code") == r["reject_code"]) for r in blocked},
        "telemetry_only": True,
    }


__all__ = [
    "apply_overlay_to_explainability",
    "build_context_audit_emit",
    "build_freshness_snapshot",
    "overlay_live_context_freshness",
    "read_live_context",
]
