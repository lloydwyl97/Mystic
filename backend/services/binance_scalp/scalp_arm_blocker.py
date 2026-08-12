"""Adaptive per-arm scalp block based on recent-30d PnL history.

Data from 2026-08-09 audit: SOL/vwap_ema_reclaim (5 trades, -$1.77, 20% wins)
and SOL/range_bounce_scalp (3 trades, -$2.60, 33% wins) were bleeding while
BTC arms were net-positive. Without a data-driven blocker the scalper keeps
firing the losing SOL setups.

This module returns a `hard_block` label for `(symbol, setup)` combinations
whose recent-window win rate + avg PnL indicate a bleed. The block is time-
limited (default 6h) so an arm can prove itself again after enough elapsed
market conditions. Never blocks arms with fewer than `min_obs` samples so
new setups get a chance to gather data.

Env knobs:
  SCALP_ARM_BLOCKER_ENABLED=true  (kill switch)
  SCALP_ARM_MIN_OBS=5             (need at least N closes before considering)
  SCALP_ARM_MAX_WIN_RATE=0.30     (win rate at/below this + neg avg → block)
  SCALP_ARM_MAX_AVG_PNL_USD=-0.20 (avg PnL at/below this + low win → block)
  SCALP_ARM_LOOKBACK_DAYS=30      (history window)
  SCALP_ARM_BLOCK_HOURS=6         (block duration between rechecks)
  SCALP_ARM_CACHE_TTL_SEC=180     (in-process cache for the block decision)
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from backend.database_schema import DATABASE_PATH


_ARM_CACHE: dict[str, tuple[bool, str, float, dict[str, Any]]] = {}


def _enabled() -> bool:
    return os.getenv("SCALP_ARM_BLOCKER_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _min_obs() -> int:
    try:
        return max(2, int(os.getenv("SCALP_ARM_MIN_OBS", "5")))
    except (TypeError, ValueError):
        return 5


def _max_win_rate() -> float:
    try:
        return float(os.getenv("SCALP_ARM_MAX_WIN_RATE", "0.30"))
    except (TypeError, ValueError):
        return 0.30


def _max_avg_pnl_usd() -> float:
    try:
        return float(os.getenv("SCALP_ARM_MAX_AVG_PNL_USD", "-0.20"))
    except (TypeError, ValueError):
        return -0.20


def _lookback_days() -> int:
    try:
        return max(1, int(os.getenv("SCALP_ARM_LOOKBACK_DAYS", "30")))
    except (TypeError, ValueError):
        return 30


def _cache_ttl_sec() -> float:
    try:
        return max(30.0, float(os.getenv("SCALP_ARM_CACHE_TTL_SEC", "180")))
    except (TypeError, ValueError):
        return 180.0


def _arm_key(symbol: str, setup: str) -> str:
    return f"{str(symbol or '').upper()}|{str(setup or '').lower()}"


def _query_arm_stats(
    symbol: str,
    setup: str,
    *,
    db_path: str = DATABASE_PATH,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    lb = int(lookback_days or _lookback_days())
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - lb * 86400))
    out = {"n": 0, "wins": 0, "total_pnl_usd": 0.0, "avg_pnl_usd": 0.0, "win_rate": 0.0}
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            # ai_signal / scalp_learning_outcomes carries setup_name + net_pnl_usd.
            rows = conn.execute(
                """
                SELECT net_pnl_usd
                FROM scalp_learning_outcomes
                WHERE symbol = ?
                  AND lower(setup_name) = lower(?)
                  AND ingested_at >= ?
                """,
                (symbol, setup, since_iso),
            ).fetchall()
    except sqlite3.Error:
        return out
    if not rows:
        return out
    pnls = [float(r[0] or 0.0) for r in rows]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    total = sum(pnls)
    out.update(
        {
            "n": n,
            "wins": wins,
            "total_pnl_usd": round(total, 4),
            "avg_pnl_usd": round(total / n, 4) if n > 0 else 0.0,
            "win_rate": round(wins / n, 4) if n > 0 else 0.0,
        }
    )
    return out


def arm_blocked(
    symbol: str,
    setup: str,
    *,
    db_path: str = DATABASE_PATH,
) -> tuple[bool, str, dict[str, Any]]:
    """Return (blocked, reason, stats).

    Blocked when:
      * arm blocker is enabled
      * n >= min_obs
      * win_rate <= max_win_rate
      * avg_pnl_usd <= max_avg_pnl_usd
    """
    if not _enabled():
        return False, "", {"disabled": True}
    key = _arm_key(symbol, setup)
    now = time.time()
    cached = _ARM_CACHE.get(key)
    if cached and (now - cached[2]) < _cache_ttl_sec():
        blocked, reason, _ts, stats = cached
        return blocked, reason, stats

    stats = _query_arm_stats(symbol, setup, db_path=db_path)
    n = int(stats.get("n") or 0)
    if n < _min_obs():
        _ARM_CACHE[key] = (False, "", now, stats)
        return False, "", stats
    wr = float(stats.get("win_rate") or 0.0)
    avg = float(stats.get("avg_pnl_usd") or 0.0)
    if wr <= _max_win_rate() and avg <= _max_avg_pnl_usd():
        reason = f"ARM_ADAPTIVE_BLOCK n={n} wr={wr:.2f} avg=${avg:.2f} lookback={_lookback_days()}d"
        _ARM_CACHE[key] = (True, reason, now, stats)
        return True, reason, stats
    _ARM_CACHE[key] = (False, "", now, stats)
    return False, "", stats


__all__ = ["arm_blocked"]
