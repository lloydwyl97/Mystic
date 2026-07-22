"""
Fail-closed entry gate for AI context (Step 2).

Used at signal consumption (Redis hash) and again at BUY execution time so
context cannot silently age into a fill without a matching audit trail.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.ai_signal_bus import MAX_SIGNAL_AGE_SEC
from backend.services.strategy_runtime_audit import get_ctx_fresh_max_age_sec

logger = logging.getLogger(__name__)


def _coerce_to_epoch(raw: Any) -> float | None:
    """Robustly convert various timestamp representations to epoch seconds (float).

    Supports:
      - numeric epoch (seconds or milliseconds as int/float/str)
      - ISO strings (with Z, +00:00, without tz)
      - common key variants are tried by the caller
    Returns None if unparsable.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # numeric epoch (or ms)
    try:
        f = float(s)
        if f > 1e11:  # milliseconds
            return f / 1000.0
        if f > 0:
            return f
    except (ValueError, TypeError):
        pass
    # ISO family
    for candidate in (s, s.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).timestamp()
        except Exception:
            continue
    return None


def entry_context_gate_enabled() -> bool:
    return os.getenv("ENTRY_CONTEXT_GATE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _parse_audit_blob(raw: str | None) -> dict[str, Any] | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def needs_context_audit_emit_refresh(raw: str | None, feature_version: int | float | str | None) -> bool:
    """
    True when decision_data should be patched from ai_signal Redis or reconstructed
    from live ai_context. Used when the blob is missing, or non-JSON / truncated
    so the entry gate cannot parse it (fv >= 2 only).
    """
    try:
        fv = int(float(feature_version)) if feature_version not in (None, "") else 1
    except (TypeError, ValueError):
        fv = 1
    if fv < 2:
        return False
    s = (raw or "").strip()
    if not s:
        return True
    return _parse_audit_blob(s) is None


def _feature_version(dd: dict[str, str]) -> int:
    raw = dd.get("feature_version")
    try:
        if raw is None or str(raw).strip() == "":
            return 1
        return int(float(raw))
    except (TypeError, ValueError):
        return 1


def evaluate_signal_hash_for_entry(dd: dict[str, str]) -> tuple[bool, str | None, dict[str, Any]]:
    """
    Validate Redis ai_signal hash fields for a new BUY entry.

    Returns:
        (ok, reject_reason_code_or_None, detail_dict)
    """
    detail: dict[str, Any] = {}
    if not entry_context_gate_enabled():
        detail["gate_disabled"] = True
        return True, None, detail

    fv = _feature_version(dd)
    detail["feature_version"] = fv
    limit = get_ctx_fresh_max_age_sec()
    detail["max_age_sec"] = limit

    if (dd.get("signal_content_stale") or "").strip() == "1":
        detail["signal_content_stale"] = True
        detail["content_age_sec"] = dd.get("content_age_sec")
        return False, "SIGNAL_CONTENT_STALE", detail

    if (dd.get("content_fresh") or "").strip() == "0":
        detail["content_fresh"] = "0"
        detail["content_age_sec"] = dd.get("content_age_sec")
        return False, "SIGNAL_CONTENT_NOT_FRESH", detail

    # Robust signal content timestamp parsing (multiple keys + formats).
    # We try common keys that producers actually emit.
    ts_raw = None
    for k in (
        "signal_content_timestamp",
        "timestamp",
        "writer_timestamp",
        "ts",
        "ts_utc",
        "content_timestamp",
        "signal_ts",
        "generated_at",
        "created_at",
        "asof",
    ):
        val = dd.get(k)
        if val is not None and str(val).strip() != "":
            ts_raw = val
            break

    parsed_ts = _coerce_to_epoch(ts_raw) if ts_raw is not None else None
    detail["max_signal_age_sec"] = MAX_SIGNAL_AGE_SEC

    if parsed_ts is not None and parsed_ts > 0:
        sig_age = time.time() - parsed_ts
        detail["signal_content_age_sec"] = sig_age
        if sig_age > float(MAX_SIGNAL_AGE_SEC):
            return False, "SIGNAL_CONTENT_AGE_EXCEEDED", detail
    else:
        # Fallback: if no usable absolute ts, but a direct age field is present and valid, use it.
        # IMPORTANT: a valid in-limit age must continue to context checks — do not fall through
        # to SIGNAL_CONTENT_TIMESTAMP_MISSING (live hashes often carry content_age_sec=0).
        age_ok = False
        age_val = None
        for k in ("signal_content_age_sec", "content_age_sec", "signal_age_sec", "age_sec"):
            v = dd.get(k)
            if v is not None and str(v).strip() != "":
                age_val = v
                break
        if age_val is not None:
            try:
                age = float(age_val)
                detail["signal_content_age_sec"] = age
                if age > float(MAX_SIGNAL_AGE_SEC):
                    return False, "SIGNAL_CONTENT_AGE_EXCEEDED", detail
                if age >= 0.0:
                    age_ok = True
                    detail["timestamp_via"] = "content_age_fallback"
            except (TypeError, ValueError):
                age_ok = False

        if not age_ok:
            # Distinguish missing vs unparsable
            if ts_raw is None or str(ts_raw).strip() == "":
                detail["timestamp_raw"] = None
                return False, "SIGNAL_CONTENT_TIMESTAMP_MISSING", detail
            detail["timestamp_raw"] = ts_raw
            return False, "SIGNAL_CONTENT_TIMESTAMP_PARSE", detail

    cf = (dd.get("context_fresh") or "").strip()
    if cf != "1":
        detail["context_fresh"] = cf or None
        _age_raw = dd.get("ctx_age_sec")
        try:
            _age_f = float(_age_raw) if _age_raw is not None and str(_age_raw).strip() != "" else None
        except (TypeError, ValueError):
            _age_f = None
        logger.info(
            "CONTEXT_AGE_ABOVE_THRESHOLD reason=context_fresh_flag context_fresh=%r ctx_age_sec=%s threshold=%.3f",
            cf or None,
            _age_raw,
            limit,
        )
        return False, "ENTRY_CONTEXT_NOT_FRESH", detail

    try:
        age = float(dd.get("ctx_age_sec") or "")
        if age < 0 or age > limit:
            detail["ctx_age_sec"] = age
            logger.info(
                "CONTEXT_AGE_ABOVE_THRESHOLD ctx_age_sec=%.3f threshold=%.3f",
                age,
                limit,
            )
            return False, "ENTRY_CONTEXT_STALE_AGE", detail
    except (TypeError, ValueError):
        detail["ctx_age_sec_raw"] = dd.get("ctx_age_sec")
        logger.info(
            "CONTEXT_AGE_ABOVE_THRESHOLD reason=parse_error ctx_age_sec_raw=%r threshold=%.3f",
            dd.get("ctx_age_sec"),
            limit,
        )
        return False, "ENTRY_CONTEXT_STALE_AGE_PARSE", detail

    ts = (dd.get("ctx_ts_utc") or "").strip()
    if not ts:
        return False, "ENTRY_CONTEXT_MISSING_TS", detail

    audit = _parse_audit_blob(dd.get("context_audit_emit"))
    detail["audit_present"] = audit is not None

    if fv >= 2:
        if audit is None:
            return False, "ENTRY_CONTEXT_MISSING_AUDIT_PAYLOAD", detail
        if not audit.get("redis_ai_context_present"):
            detail["audit_keys"] = list(audit.keys())[:20]
            return False, "ENTRY_CONTEXT_MISSING_REDIS_AI_CONTEXT", detail
        mf = audit.get("missing_contract_fields") or []
        if isinstance(mf, list) and len(mf) > 0:
            detail["missing_contract_fields"] = mf[:24]
            return False, "ENTRY_CONTEXT_DEFAULTED_REQUIRED_FIELDS", detail

    return True, None, detail


def evaluate_explainability_at_execution(explainability: Any) -> tuple[bool, str | None, dict[str, Any]]:
    """
    Re-evaluate context at BUY execution using wall-clock age vs ctx_ts_utc.

    Expects TradeExplainability-like object with ctx_ts_utc, context_audit_emit,
    context_fresh_flag, feature_version, etc.
    """
    strat = str(getattr(explainability, "live_ai_strategy", "") or "").strip()
    try:
        fv_skip = int(getattr(explainability, "feature_version", 0) or 0)
    except (TypeError, ValueError):
        fv_skip = 0
    if not strat and fv_skip <= 0:
        return True, None, {"skipped": "non_ml_buy_path"}

    dd: dict[str, str] = {}
    try:
        fv = int(getattr(explainability, "feature_version", 0) or 0)
    except (TypeError, ValueError):
        fv = 1
    if fv <= 0:
        fv = 1
    dd["feature_version"] = str(fv)

    ts = str(getattr(explainability, "ctx_ts_utc", "") or "").strip()
    dd["context_audit_emit"] = str(getattr(explainability, "context_audit_emit", "") or "")

    if ts:
        dd["ctx_ts_utc"] = ts
        try:
            tnorm = str(ts).replace("Z", "+00:00")
            t_parse = datetime.fromisoformat(tnorm)
            if t_parse.tzinfo is None:
                t_parse = t_parse.replace(tzinfo=timezone.utc)
            age_now = (datetime.now(timezone.utc) - t_parse.astimezone(timezone.utc)).total_seconds()
            dd["ctx_age_sec"] = str(age_now)
            lim = get_ctx_fresh_max_age_sec()
            dd["context_fresh"] = "1" if 0 <= age_now <= lim else "0"
        except Exception:
            dd["ctx_age_sec"] = str(getattr(explainability, "ctx_age_sec", -1.0))
            dd["context_fresh"] = str(getattr(explainability, "context_fresh_flag", "") or "").strip() or "0"
    else:
        dd["ctx_ts_utc"] = ""
        dd["ctx_age_sec"] = str(getattr(explainability, "ctx_age_sec", -1.0))
        cf = str(getattr(explainability, "context_fresh_flag", "") or "").strip()
        dd["context_fresh"] = cf if cf in ("0", "1") else "0"

    # Signal content freshness (ai_signal hash) — required at execution parity with consume-time gate.
    sig_ts = str(getattr(explainability, "signal_content_timestamp", "") or "").strip()
    if sig_ts:
        dd["timestamp"] = sig_ts
    sig_cf = str(getattr(explainability, "signal_content_fresh", "") or "").strip()
    if sig_cf in ("0", "1"):
        dd["content_fresh"] = sig_cf
    sig_age = str(getattr(explainability, "signal_content_age_sec", "") or "").strip()
    if sig_age:
        dd["content_age_sec"] = sig_age
    sig_stale = str(getattr(explainability, "signal_content_stale", "") or "").strip()
    if sig_stale in ("0", "1"):
        dd["signal_content_stale"] = sig_stale

    return evaluate_signal_hash_for_entry(dd)


__all__ = [
    "entry_context_gate_enabled",
    "evaluate_explainability_at_execution",
    "evaluate_signal_hash_for_entry",
    "needs_context_audit_emit_refresh",
]
