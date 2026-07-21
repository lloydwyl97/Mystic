"""SCALP entry telemetry — genuine-pass / reject histogram for paper monitoring.

Publishes a short-TTL Redis snapshot (scalp:telemetry:entry) so operators can see
why buys are or are not firing without scraping logs. Soft-rank never counts as
genuine-pass. Separate sleeve from DAY.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_TELEMETRY_TTL_SEC = 300


def telemetry_key(prefix: str = "scalp") -> str:
    from backend.services.binance_scalp.redis_keys import assert_key_allowed, normalize_prefix

    key = f"{normalize_prefix(prefix)}:telemetry:entry"
    assert_key_allowed(key, prefix=prefix)
    return key


def build_entry_telemetry(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one evaluate_all() cycle into a compact telemetry dict."""
    reject_counts: Counter[str] = Counter()
    strategy_pass: Counter[str] = Counter()
    strategy_eval: Counter[str] = Counter()
    passed_total = 0
    eligible_total = 0
    regime_blocked = 0
    target_unreachable = 0
    symbols: list[dict[str, Any]] = []

    for row in ranked or []:
        sym = str(row.get("symbol") or "")
        meta = row.get("rank_meta") or {}
        regime = meta.get("regime") or row.get("micro_regime")
        all_sigs = row.get("all_signals") or []
        sym_passed = 0
        for s in all_sigs:
            name = str(s.get("setup_name") or "?")
            strategy_eval[name] += 1
            if s.get("passed"):
                passed_total += 1
                sym_passed += 1
                strategy_pass[name] += 1
            else:
                reason = str(s.get("reject_reason") or "NONE")
                reject_counts[reason] += 1
                if reason == "TARGET_NOT_REACHABLE":
                    target_unreachable += 1
        soft = str(meta.get("soft_reason") or row.get("soft_reason") or "")
        if soft.startswith("REGIME_BLOCKED"):
            regime_blocked += 1
        if row.get("entry_eligible"):
            eligible_total += 1
        symbols.append(
            {
                "symbol": sym,
                "regime": regime,
                "best_setup": row.get("best_setup") or meta.get("best_setup"),
                "rank_score": row.get("rank_score"),
                "entry_eligible": bool(row.get("entry_eligible")),
                "passed_setups": sym_passed,
                "hard_block": row.get("hard_block") or meta.get("hard_block"),
                "soft_reason": soft or None,
            }
        )

    return {
        "updated_at_epoch": time.time(),
        "symbols_scanned": len(ranked or []),
        "genuine_pass_setups": passed_total,
        "entry_eligible_count": eligible_total,
        "regime_blocked_count": regime_blocked,
        "target_unreachable_count": target_unreachable,
        "reject_reasons": dict(reject_counts.most_common(25)),
        "strategy_eval_counts": dict(strategy_eval),
        "strategy_pass_counts": dict(strategy_pass),
        "symbols": symbols,
        "soft_rank_entry_allowed": False,
        "note": "genuine_pass_setups = strategies that returned passed=True this cycle; entry_eligible_count requires passed + score + regime_native (when enabled).",
    }


def publish_entry_telemetry(
    redis_client: Any,
    ranked: list[dict[str, Any]],
    *,
    prefix: str = "scalp",
) -> dict[str, Any] | None:
    """Build + publish telemetry. Returns payload or None on failure."""
    if redis_client is None:
        return None
    try:
        payload = build_entry_telemetry(ranked)
        key = telemetry_key(prefix)
        redis_client.setex(key, _TELEMETRY_TTL_SEC, json.dumps(payload, separators=(",", ":")))
        return payload
    except Exception as exc:
        logger.debug("SCALP entry telemetry publish skipped: %s", exc)
        return None


def read_entry_telemetry(redis_client: Any, *, prefix: str = "scalp") -> dict[str, Any] | None:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(telemetry_key(prefix))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None
