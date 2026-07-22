"""Per-symbol rolling market memory in Redis (rank/explain inputs only)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "rolling_market_state:"


def _redis_key(symbol: str) -> str:
    sym = (symbol or "").replace("/", "").upper()
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"
    return f"{REDIS_KEY_PREFIX}{sym}"


def _decode(raw: Any) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def load_market_memory_sync(symbol: str) -> dict[str, Any]:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return {}
        raw = r.get(_redis_key(symbol))
        if not raw:
            return {}
        text = _decode(raw)
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


async def load_market_memory(redis_client: Any, symbol: str) -> dict[str, Any]:
    if not redis_client:
        return {}
    try:
        raw = await redis_client.get(_redis_key(symbol))
        if not raw:
            return {}
        text = _decode(raw)
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        logger.debug("load_market_memory %s: %s", symbol, exc)
        return {}


async def save_market_memory(redis_client: Any, symbol: str, state: dict[str, Any], *, ttl_sec: int = 86400) -> None:
    if not redis_client:
        return
    try:
        payload = dict(state or {})
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        await redis_client.set(_redis_key(symbol), json.dumps(payload, separators=(",", ":")), ex=max(3600, int(ttl_sec)))
    except Exception as exc:
        logger.debug("save_market_memory %s: %s", symbol, exc)


def build_memory_patch_from_decision_data(decision_data: dict[str, Any]) -> dict[str, Any]:
    dd = decision_data or {}
    cur = str(dd.get("day_route_regime") or dd.get("regime") or "neutral")
    prev = str(dd.get("previous_regime") or "")
    trans = float(dd.get("regime_transition_score") or 0.0)
    # Candidate polls must not overwrite close-path same-setup counters.
    return {
        "last_30m_state": str(dd.get("market_state_30m") or cur),
        "last_2h_state": str(dd.get("market_state_2h") or cur),
        "last_6h_state": str(dd.get("market_state_6h") or cur),
        "last_24h_state": str(dd.get("market_state_24h") or cur),
        "current_regime": cur,
        "previous_regime": prev or cur,
        "regime_transition_score": trans,
        "volatility_state": str(dd.get("volatility_state") or "normal"),
        "liquidity_state": str(dd.get("liquidity_state") or "normal"),
        "relative_strength_rank": int(float(dd.get("relative_strength_rank") or 0)),
        "last_setup_candidate": str(dd.get("setup_type") or dd.get("entry_thesis") or ""),
    }


async def update_market_memory_on_candidate(redis_client: Any, symbol: str, decision_data: dict[str, Any]) -> dict[str, Any]:
    existing = await load_market_memory(redis_client, symbol)
    patch = build_memory_patch_from_decision_data(decision_data)
    prev_regime = str(existing.get("current_regime") or "")
    if prev_regime and prev_regime != patch.get("current_regime"):
        patch["previous_regime"] = prev_regime
        patch["regime_transition_score"] = max(float(existing.get("regime_transition_score") or 0.0), 0.35)
    merged = {**existing, **patch, "last_seen_ts": time.time()}
    await save_market_memory(redis_client, symbol, merged)
    return merged


def update_market_memory_on_close_sync(
    symbol: str,
    *,
    setup: str,
    net_pnl_pct: float | None,
    close_reason: str,
    outcome_class: str = "",
) -> None:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return
        existing_raw = r.get(_redis_key(symbol))
        existing: dict[str, Any] = {}
        if existing_raw:
            existing = json.loads(_decode(existing_raw))
        pnl = float(net_pnl_pct or 0.0)
        result = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
        setup_u = str(setup or "").upper()
        same_count = int(existing.get("same_setup_today_count") or 0)
        same_pnl = float(existing.get("same_setup_today_net_pnl") or 0.0)
        last_closed = str(existing.get("last_closed_setup") or existing.get("last_setup_seen") or "").upper()
        if last_closed == setup_u:
            same_count += 1
            same_pnl += pnl
        else:
            same_count = 1
            same_pnl = pnl
        patch = {
            "last_closed_setup": setup_u,
            "last_setup_seen": setup_u,
            "last_setup_result": result,
            "last_trade_result": result,
            "last_trade_failure_reason": str(close_reason or outcome_class or ""),
            "same_setup_today_count": same_count,
            "same_setup_today_net_pnl": round(same_pnl, 6),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        merged = {**existing, **patch}
        r.set(_redis_key(symbol), json.dumps(merged, separators=(",", ":")), ex=86400)
    except Exception as exc:
        logger.debug("update_market_memory_on_close_sync %s: %s", symbol, exc)


async def update_market_memory_on_close(
    redis_client: Any,
    symbol: str,
    *,
    setup: str,
    net_pnl_pct: float | None,
    close_reason: str,
    outcome_class: str = "",
) -> None:
    existing = await load_market_memory(redis_client, symbol)
    pnl = float(net_pnl_pct or 0.0)
    result = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
    setup_u = str(setup or "").upper()
    same_count = int(existing.get("same_setup_today_count") or 0)
    same_pnl = float(existing.get("same_setup_today_net_pnl") or 0.0)
    last_closed = str(existing.get("last_closed_setup") or existing.get("last_setup_seen") or "").upper()
    if last_closed == setup_u:
        same_count += 1
        same_pnl += pnl
    else:
        same_count = 1
        same_pnl = pnl
    patch = {
        "last_closed_setup": setup_u,
        "last_setup_seen": setup_u,
        "last_setup_result": result,
        "last_trade_result": result,
        "last_trade_failure_reason": str(close_reason or outcome_class or ""),
        "same_setup_today_count": same_count,
        "same_setup_today_net_pnl": round(same_pnl, 6),
    }
    await save_market_memory(redis_client, symbol, {**existing, **patch})


def memory_rank_delta(memory: dict[str, Any], setup: str) -> float:
    mem = memory or {}
    setup_u = str(setup or "").upper()
    last_closed = str(mem.get("last_closed_setup") or mem.get("last_setup_seen") or "").upper()
    if last_closed != setup_u:
        return 0.0
    same_pnl = float(mem.get("same_setup_today_net_pnl") or 0.0)
    same_count = int(mem.get("same_setup_today_count") or 0)
    if same_count >= 3 and same_pnl < -0.003:
        return -0.03
    if same_count >= 2 and same_pnl > 0.002:
        return 0.02
    last = str(mem.get("last_setup_result") or "")
    if last == "win":
        return 0.01
    if last == "loss":
        return -0.015
    return 0.0


__all__ = [
    "REDIS_KEY_PREFIX",
    "load_market_memory",
    "load_market_memory_sync",
    "memory_rank_delta",
    "save_market_memory",
    "update_market_memory_on_candidate",
    "update_market_memory_on_close",
    "update_market_memory_on_close_sync",
]
