"""SCALP rolling market memory — rank/explain only (separate from DAY)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

REDIS_KEY_TEMPLATE = "rolling_scalp_market_state:{SYMBOL}"


def _key(symbol: str) -> str:
    sym = symbol.upper().replace("/", "")
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"
    return REDIS_KEY_TEMPLATE.replace("{SYMBOL}", sym)


def default_memory() -> dict[str, Any]:
    return {
        "last_1m_state": "",
        "last_5m_state": "",
        "last_15m_state": "",
        "last_30m_state": "",
        "current_micro_regime": "",
        "previous_micro_regime": "",
        "scalp_regime_transition_score": 0.0,
        "micro_volatility_state": "normal",
        "liquidity_state": "normal",
        "spread_state": "normal",
        "orderbook_freshness_state": "unknown",
        "relative_strength_rank": 0,
        "last_scalp_setup_seen": "",
        "last_scalp_setup_result": "",
        "same_scalp_setup_today_count": 0,
        "same_scalp_setup_today_net_pnl": 0.0,
        "recent_scalp_win_rate": 0.0,
        "recent_scalp_avg_hold_time": 0.0,
        "recent_scalp_slippage": 0.0,
        "updated_at_utc": "",
    }


def load_scalp_market_memory_sync(redis_client: Any, symbol: str) -> dict[str, Any]:
    if not redis_client:
        return default_memory()
    try:
        raw = redis_client.get(_key(symbol))
        if not raw:
            return default_memory()
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
        return data if isinstance(data, dict) else default_memory()
    except Exception:
        return default_memory()


def save_scalp_market_memory_sync(redis_client: Any, symbol: str, memory: dict[str, Any], *, ttl_sec: int = 86400) -> None:
    if not redis_client:
        return
    try:
        mem = dict(memory or {})
        mem["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        redis_client.setex(_key(symbol), ttl_sec, json.dumps(mem, separators=(",", ":")))
    except Exception as exc:
        logger.debug("save_scalp_market_memory skipped %s: %s", symbol, exc)


def update_scalp_market_memory_on_candidate(
    redis_client: Any,
    symbol: str,
    intelligence: dict[str, Any],
    *,
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mem = dict(memory or load_scalp_market_memory_sync(redis_client, symbol))
    prev = str(mem.get("current_micro_regime") or "")
    cur = str(intelligence.get("micro_regime") or intelligence.get("current_micro_regime") or "")
    if cur:
        mem["previous_micro_regime"] = prev or cur
        mem["current_micro_regime"] = cur
    setup = str(intelligence.get("scalp_setup") or intelligence.get("setup_name") or "")
    if setup:
        # Candidate polls must not overwrite close-path same_* key.
        mem["last_scalp_setup_candidate"] = setup
    spread = float(intelligence.get("spread_pct") or 0.0)
    mem["spread_state"] = "tight" if spread < 0.0015 else ("wide" if spread > 0.003 else "normal")
    ob_age = intelligence.get("orderbook_age_sec")
    if ob_age is not None:
        mem["orderbook_freshness_state"] = "fresh" if float(ob_age) < 45 else "stale"
    mem["last_1m_state"] = str(intelligence.get("mid_change_15s") or "")
    save_scalp_market_memory_sync(redis_client, symbol, mem)
    return mem


def update_scalp_market_memory_on_close_sync(
    symbol: str,
    *,
    setup: str,
    net_pnl: float,
    hold_seconds: float,
    slippage: float,
    redis_client: Any = None,
) -> None:
    if redis_client is None:
        try:
            import redis

            from backend.services.binance_scalp.config import get_scalp_config

            redis_client = redis.from_url(get_scalp_config().redis_url, decode_responses=True)
        except Exception:
            return
    mem = load_scalp_market_memory_sync(redis_client, symbol)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if mem.get("_memory_day") != day:
        mem["same_scalp_setup_today_count"] = 0
        mem["same_scalp_setup_today_net_pnl"] = 0.0
        mem["_memory_day"] = day
        # Daily window for per-setup expectancy (was unbounded).
        for k in list(mem.keys()):
            if str(k).startswith("setup_pnl::") or str(k).startswith("setup_n::"):
                mem.pop(k, None)
    setup_l = (setup or "").strip().lower()
    if setup_l:
        last_closed = str(mem.get("last_closed_scalp_setup") or mem.get("last_scalp_setup_seen") or "").strip().lower()
        if last_closed == setup_l:
            mem["same_scalp_setup_today_count"] = int(mem.get("same_scalp_setup_today_count") or 0) + 1
            mem["same_scalp_setup_today_net_pnl"] = float(mem.get("same_scalp_setup_today_net_pnl") or 0) + float(net_pnl)
        else:
            mem["same_scalp_setup_today_count"] = 1
            mem["same_scalp_setup_today_net_pnl"] = float(net_pnl)
        mem["last_closed_scalp_setup"] = setup_l
        mem["last_scalp_setup_seen"] = setup_l
    mem["last_scalp_setup_result"] = "win" if net_pnl > 0 else "loss"
    wins = float(mem.get("_recent_wins") or 0)
    total = float(mem.get("_recent_total") or 0) + 1
    if net_pnl > 0:
        wins += 1
    mem["_recent_wins"] = wins
    mem["_recent_total"] = total
    mem["recent_scalp_win_rate"] = round(wins / total, 4) if total else 0.0
    prev_avg = float(mem.get("recent_scalp_avg_hold_time") or 0)
    mem["recent_scalp_avg_hold_time"] = round((prev_avg * (total - 1) + hold_seconds) / total, 1) if total else hold_seconds
    mem["recent_scalp_slippage"] = round(slippage, 6)
    if setup_l:
        pnl_key = f"setup_pnl::{setup_l}"
        n_key = f"setup_n::{setup_l}"
        mem[pnl_key] = float(mem.get(pnl_key) or 0.0) + float(net_pnl)
        mem[n_key] = int(mem.get(n_key) or 0) + 1
    save_scalp_market_memory_sync(redis_client, symbol, mem)


def memory_rank_delta(memory: dict[str, Any], setup: str) -> float:
    if not memory:
        return 0.0
    setup_l = (setup or "").strip().lower()
    same_pnl = float(memory.get("same_scalp_setup_today_net_pnl") or 0.0)
    win_rate = float(memory.get("recent_scalp_win_rate") or 0.0)
    setup_key = f"setup_pnl::{setup_l}"
    setup_pnl = float(memory.get(setup_key) or 0.0)
    setup_n = int(memory.get(f"setup_n::{setup_l}") or 0)
    delta = 0.0
    # same_* only applies when scoring the setup that earned those closes today.
    last_closed = str(memory.get("last_closed_scalp_setup") or memory.get("last_scalp_setup_seen") or "").strip().lower()
    if setup_l and last_closed == setup_l:
        if same_pnl > 0.5:
            delta += 0.025
        elif same_pnl < -0.5:
            delta -= 0.03
    if win_rate > 0.55:
        delta += 0.015
    elif 0 < win_rate < 0.35:
        delta -= 0.02
    # Per-setup expectancy from today's closes (range_bounce was the loss driver).
    if setup_n >= 3:
        if setup_pnl <= -1.0:
            delta -= 0.04
        elif setup_pnl <= -0.3:
            delta -= 0.025
        elif setup_pnl >= 0.8:
            delta += 0.02
    return round(max(-0.06, min(0.05, delta)), 4)


__all__ = [
    "default_memory",
    "load_scalp_market_memory_sync",
    "memory_rank_delta",
    "update_scalp_market_memory_on_candidate",
    "update_scalp_market_memory_on_close_sync",
]
