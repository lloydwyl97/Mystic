"""SCALP entry telemetry — genuine-pass / reject / post-pass blocker monitoring.

Publishes:
  - scalp:telemetry:entry         — latest evaluate cycle (short TTL)
  - scalp:telemetry:entry:rolling — rolling window aggregates (longer TTL)

Soft-rank never counts as genuine-pass. Does not change entry eligibility.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_TELEMETRY_TTL_SEC = 300
_ROLLING_TTL_SEC = 86400  # keep rolling window up to 24h
_DEFAULT_ROLLING_CYCLES = 120  # ~10 min at 5s ticks; override via SCALP_TELEMETRY_ROLLING_CYCLES


def telemetry_key(prefix: str = "scalp") -> str:
    from backend.services.binance_scalp.redis_keys import assert_key_allowed, normalize_prefix

    key = f"{normalize_prefix(prefix)}:telemetry:entry"
    assert_key_allowed(key, prefix=prefix)
    return key


def rolling_key(prefix: str = "scalp") -> str:
    from backend.services.binance_scalp.redis_keys import assert_key_allowed, normalize_prefix

    key = f"{normalize_prefix(prefix)}:telemetry:entry:rolling"
    assert_key_allowed(key, prefix=prefix)
    return key


def _rolling_max_cycles() -> int:
    try:
        return max(20, min(2000, int(os.getenv("SCALP_TELEMETRY_ROLLING_CYCLES", str(_DEFAULT_ROLLING_CYCLES)))))
    except (TypeError, ValueError):
        return _DEFAULT_ROLLING_CYCLES


def _sig_field(s: Any, key: str, default: Any = None) -> Any:
    if isinstance(s, dict):
        return s.get(key, default)
    return getattr(s, key, default)


def _classify_post_pass_blocker(ranked_row: dict[str, Any]) -> str | None:
    """
    Why a genuine-pass setup still failed entry_eligible.
    Returns None if eligible or never passed.
    """
    meta = ranked_row.get("rank_meta") or {}
    ranked_list = meta.get("ranked") or []
    # Prefer best ranked entry that passed
    passed_rows = [r for r in ranked_list if isinstance(r, dict) and r.get("passed")]
    if not passed_rows and not any(_sig_field(s, "passed") for s in (ranked_row.get("all_signals") or [])):
        return None
    if ranked_row.get("entry_eligible"):
        return None

    conf = str(meta.get("selection_confidence") or ranked_row.get("selection_confidence") or "")
    soft = str(meta.get("soft_reason") or ranked_row.get("soft_reason") or "")
    hard = ranked_row.get("hard_block") or meta.get("hard_block")

    if hard:
        return f"HARD:{hard}"
    if soft.startswith("REGIME_BLOCKED") or conf == "regime_mismatch":
        return "REGIME_MISMATCH"
    if soft.startswith("MTF_") or conf == "mtf_confirmation_blocked":
        return f"MTF:{soft or conf}"
    if conf == "symbol_stall_risk_blocked" or soft.startswith("SYMBOL_STALL_RISK"):
        return "STALL_RISK_GATE"
    if conf == "below_min" or soft.startswith("RANK_BELOW") or "BELOW_MIN" in soft.upper():
        return "BELOW_MIN_SCORE"
    if soft.startswith("NO_EXECUTABLE") or hard == "NO_EXECUTABLE_NET_EDGE":
        return "NO_EXECUTABLE_NET_EDGE"
    if soft:
        return f"SOFT:{soft.split(':')[0]}"
    if conf:
        return f"CONF:{conf}"
    return "POST_PASS_UNKNOWN"


def build_entry_telemetry(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one evaluate_all() cycle into a compact telemetry dict."""
    reject_counts: Counter[str] = Counter()
    strategy_pass: Counter[str] = Counter()
    strategy_eval: Counter[str] = Counter()
    strategy_eligible: Counter[str] = Counter()
    symbol_setup_pass: Counter[str] = Counter()  # "BTCUSDT:range_bounce"
    symbol_setup_eval: Counter[str] = Counter()
    post_pass_blockers: Counter[str] = Counter()
    regime_native_pass = 0
    regime_mismatch_pass = 0
    passed_total = 0
    eligible_total = 0
    target_unreachable = 0
    hard_block_count = 0
    symbols: list[dict[str, Any]] = []

    for row in ranked or []:
        sym = str(row.get("symbol") or "")
        meta = row.get("rank_meta") or {}
        regime = meta.get("regime") or row.get("micro_regime")
        all_sigs = row.get("all_signals") or []
        ranked_detail = meta.get("ranked") or []
        sym_passed = 0

        for s in all_sigs:
            name = str(_sig_field(s, "setup_name") or "?")
            strategy_eval[name] += 1
            symbol_setup_eval[f"{sym}:{name}"] += 1
            if _sig_field(s, "passed"):
                passed_total += 1
                sym_passed += 1
                strategy_pass[name] += 1
                symbol_setup_pass[f"{sym}:{name}"] += 1
            else:
                reason = str(_sig_field(s, "reject_reason") or "NONE")
                reject_counts[reason] += 1
                if reason == "TARGET_NOT_REACHABLE":
                    target_unreachable += 1

        # Regime alignment from per-setup ranked detail (passed setups only)
        for rd in ranked_detail:
            if not isinstance(rd, dict) or not rd.get("passed"):
                continue
            if rd.get("regime_native"):
                regime_native_pass += 1
            else:
                regime_mismatch_pass += 1
            if rd.get("entry_eligible"):
                strategy_eligible[str(rd.get("setup_name") or "?")] += 1

        soft = str(meta.get("soft_reason") or row.get("soft_reason") or "")
        hard = row.get("hard_block") or meta.get("hard_block")
        if hard:
            hard_block_count += 1

        ppb = _classify_post_pass_blocker(row)
        if ppb:
            post_pass_blockers[ppb] += 1

        if row.get("entry_eligible"):
            eligible_total += 1

        symbols.append(
            {
                "symbol": sym,
                "regime": regime,
                "best_setup": row.get("best_setup") or meta.get("best_setup"),
                "rank_score": row.get("rank_score"),
                "raw_rank_score": (row.get("rank_meta") or {}).get("best_rank_score") or row.get("rank_score"),
                "entry_eligible": bool(row.get("entry_eligible")),
                "passed_setups": sym_passed,
                "hard_block": hard,
                "soft_reason": soft or None,
                "selection_confidence": meta.get("selection_confidence") or row.get("selection_confidence"),
                "post_pass_blocker": ppb,
                "role_live_adj": None,
            }
        )
        with contextlib.suppress(Exception):
            from backend.services.market_role_intelligence import fetch_role_ranking_delta_from_redis

            symbols[-1]["role_live_adj"] = fetch_role_ranking_delta_from_redis(sym)

    # Pass rates this cycle
    strategy_pass_rate = {
        name: round(strategy_pass[name] / strategy_eval[name], 4)
        for name in strategy_eval
        if strategy_eval[name] > 0
    }

    return {
        "updated_at_epoch": time.time(),
        "symbols_scanned": len(ranked or []),
        "genuine_pass_setups": passed_total,
        "entry_eligible_count": eligible_total,
        "hard_block_count": hard_block_count,
        "regime_native_pass_count": regime_native_pass,
        "regime_mismatch_pass_count": regime_mismatch_pass,
        "target_unreachable_count": target_unreachable,
        "reject_reasons": dict(reject_counts.most_common(25)),
        "post_pass_blockers": dict(post_pass_blockers.most_common(20)),
        "strategy_eval_counts": dict(strategy_eval),
        "strategy_pass_counts": dict(strategy_pass),
        "strategy_eligible_counts": dict(strategy_eligible),
        "strategy_pass_rate": strategy_pass_rate,
        "symbol_setup_eval_counts": dict(symbol_setup_eval.most_common(40)),
        "symbol_setup_pass_counts": dict(symbol_setup_pass.most_common(40)),
        "symbols": symbols,
        "soft_rank_entry_allowed": False,
        "note": (
            "genuine_pass_setups = strategies that returned passed=True this cycle. "
            "entry_eligible_count requires passed + min score + no hard block + regime-native (when enabled). "
            "post_pass_blockers = passed setups that still could not enter."
        ),
    }


def _empty_rolling() -> dict[str, Any]:
    return {
        "cycles": 0,
        "window_max_cycles": _rolling_max_cycles(),
        "started_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
        "genuine_pass_setups": 0,
        "entry_eligible_count": 0,
        "hard_block_count": 0,
        "regime_native_pass_count": 0,
        "regime_mismatch_pass_count": 0,
        "cycles_with_any_pass": 0,
        "cycles_with_eligible": 0,
        "reject_reasons": {},
        "post_pass_blockers": {},
        "strategy_eval_counts": {},
        "strategy_pass_counts": {},
        "strategy_eligible_counts": {},
        "recent_cycle_digest": [],  # last N compact digests
    }


def _merge_counter(dst: dict[str, int], src: dict[str, Any]) -> None:
    for k, v in (src or {}).items():
        try:
            dst[str(k)] = int(dst.get(str(k), 0)) + int(v)
        except (TypeError, ValueError):
            continue


def update_rolling_telemetry(redis_client: Any, cycle: dict[str, Any], *, prefix: str = "scalp") -> dict[str, Any] | None:
    """Merge one cycle into the rolling window stored in Redis."""
    if redis_client is None:
        return None
    try:
        key = rolling_key(prefix)
        raw = redis_client.get(key)
        if raw:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            rolling = json.loads(raw)
        else:
            rolling = _empty_rolling()

        rolling["cycles"] = int(rolling.get("cycles") or 0) + 1
        rolling["window_max_cycles"] = _rolling_max_cycles()
        rolling["updated_at_epoch"] = time.time()
        if not rolling.get("started_at_epoch"):
            rolling["started_at_epoch"] = time.time()

        for field in (
            "genuine_pass_setups",
            "entry_eligible_count",
            "hard_block_count",
            "regime_native_pass_count",
            "regime_mismatch_pass_count",
        ):
            rolling[field] = int(rolling.get(field) or 0) + int(cycle.get(field) or 0)

        if int(cycle.get("genuine_pass_setups") or 0) > 0:
            rolling["cycles_with_any_pass"] = int(rolling.get("cycles_with_any_pass") or 0) + 1
        if int(cycle.get("entry_eligible_count") or 0) > 0:
            rolling["cycles_with_eligible"] = int(rolling.get("cycles_with_eligible") or 0) + 1

        for ck in ("reject_reasons", "post_pass_blockers", "strategy_eval_counts", "strategy_pass_counts", "strategy_eligible_counts"):
            bucket = rolling.get(ck) if isinstance(rolling.get(ck), dict) else {}
            _merge_counter(bucket, cycle.get(ck) or {})
            rolling[ck] = bucket

        # Compact digest for recent cycles
        digest = {
            "t": round(float(cycle.get("updated_at_epoch") or time.time()), 1),
            "pass": int(cycle.get("genuine_pass_setups") or 0),
            "elig": int(cycle.get("entry_eligible_count") or 0),
            "ppb": dict(list((cycle.get("post_pass_blockers") or {}).items())[:5]),
            "top_reject": (list((cycle.get("reject_reasons") or {}).items())[:3]),
        }
        recent = list(rolling.get("recent_cycle_digest") or [])
        recent.append(digest)
        max_c = _rolling_max_cycles()
        if len(recent) > max_c:
            # When overflowing, keep last max_c digests; totals stay cumulative
            # until process restart or explicit reset (acceptable for operator view).
            recent = recent[-max_c:]
        rolling["recent_cycle_digest"] = recent

        # Derived rates
        cycles = max(1, int(rolling["cycles"]))
        eval_total = sum(int(v) for v in (rolling.get("strategy_eval_counts") or {}).values()) or 1
        pass_total = int(rolling.get("genuine_pass_setups") or 0)
        rolling["pass_rate_overall"] = round(pass_total / eval_total, 4)
        rolling["pct_cycles_with_pass"] = round(int(rolling.get("cycles_with_any_pass") or 0) / cycles, 4)
        rolling["pct_cycles_with_eligible"] = round(int(rolling.get("cycles_with_eligible") or 0) / cycles, 4)
        rolling["strategy_pass_rate"] = {
            name: round(
                int((rolling.get("strategy_pass_counts") or {}).get(name, 0))
                / max(1, int((rolling.get("strategy_eval_counts") or {}).get(name, 0))),
                4,
            )
            for name in (rolling.get("strategy_eval_counts") or {})
        }
        # Top lists for API/dashboard
        rolling["top_reject_reasons"] = dict(
            sorted((rolling.get("reject_reasons") or {}).items(), key=lambda kv: -int(kv[1]))[:15]
        )
        rolling["top_post_pass_blockers"] = dict(
            sorted((rolling.get("post_pass_blockers") or {}).items(), key=lambda kv: -int(kv[1]))[:15]
        )

        redis_client.setex(key, _ROLLING_TTL_SEC, json.dumps(rolling, separators=(",", ":")))
        return rolling
    except Exception as exc:
        logger.debug("SCALP rolling telemetry update skipped: %s", exc)
        return None


def publish_entry_telemetry(
    redis_client: Any,
    ranked: list[dict[str, Any]],
    *,
    prefix: str = "scalp",
) -> dict[str, Any] | None:
    """Build + publish latest cycle telemetry and update rolling window."""
    if redis_client is None:
        return None
    try:
        payload = build_entry_telemetry(ranked)
        key = telemetry_key(prefix)
        redis_client.setex(key, _TELEMETRY_TTL_SEC, json.dumps(payload, separators=(",", ":")))
        rolling = update_rolling_telemetry(redis_client, payload, prefix=prefix)
        if rolling is not None:
            payload["rolling"] = {
                "cycles": rolling.get("cycles"),
                "pass_rate_overall": rolling.get("pass_rate_overall"),
                "pct_cycles_with_pass": rolling.get("pct_cycles_with_pass"),
                "pct_cycles_with_eligible": rolling.get("pct_cycles_with_eligible"),
                "top_reject_reasons": rolling.get("top_reject_reasons"),
                "top_post_pass_blockers": rolling.get("top_post_pass_blockers"),
                "strategy_pass_rate": rolling.get("strategy_pass_rate"),
                "regime_native_pass_count": rolling.get("regime_native_pass_count"),
                "regime_mismatch_pass_count": rolling.get("regime_mismatch_pass_count"),
                "genuine_pass_setups": rolling.get("genuine_pass_setups"),
                "entry_eligible_count": rolling.get("entry_eligible_count"),
                "updated_at_epoch": rolling.get("updated_at_epoch"),
            }
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
        payload = json.loads(raw)
        # Attach rolling if not already embedded (older publishers)
        if "rolling" not in payload:
            rraw = redis_client.get(rolling_key(prefix))
            if rraw:
                if isinstance(rraw, bytes):
                    rraw = rraw.decode("utf-8")
                rolling = json.loads(rraw)
                payload["rolling"] = {
                    "cycles": rolling.get("cycles"),
                    "pass_rate_overall": rolling.get("pass_rate_overall"),
                    "pct_cycles_with_pass": rolling.get("pct_cycles_with_pass"),
                    "pct_cycles_with_eligible": rolling.get("pct_cycles_with_eligible"),
                    "top_reject_reasons": rolling.get("top_reject_reasons"),
                    "top_post_pass_blockers": rolling.get("top_post_pass_blockers"),
                    "strategy_pass_rate": rolling.get("strategy_pass_rate"),
                    "regime_native_pass_count": rolling.get("regime_native_pass_count"),
                    "regime_mismatch_pass_count": rolling.get("regime_mismatch_pass_count"),
                    "genuine_pass_setups": rolling.get("genuine_pass_setups"),
                    "entry_eligible_count": rolling.get("entry_eligible_count"),
                    "updated_at_epoch": rolling.get("updated_at_epoch"),
                }
        return payload
    except Exception:
        return None


def read_rolling_telemetry(redis_client: Any, *, prefix: str = "scalp") -> dict[str, Any] | None:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(rolling_key(prefix))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


__all__ = [
    "build_entry_telemetry",
    "publish_entry_telemetry",
    "read_entry_telemetry",
    "read_rolling_telemetry",
    "telemetry_key",
    "rolling_key",
    "update_rolling_telemetry",
]
