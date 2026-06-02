"""
Autobuy Cooldowns (Redis, asyncio)

BLOCKER-AUDIT NOTE: This class (AutobuyCooldowns) is NOT imported or used anywhere
in the codebase.  It is dead code.  Cooldown authority has been consolidated into
backend.services.trade_state.TradeStateStore (Redis + SQLite, fail-closed).
Do NOT add new callers here; use trade_state instead.

Purpose:
- Enforce per-symbol, per-strategy, and global cooldowns to avoid over-trading.
- Track a daily maximum number of orders.

Quick test checklist:
- No exchange strings in this module; EXCHANGE_ID/_to_ccxt_symbol() not applicable here.
- No unreachable code after returns.
- Logging (if added by caller) should not include non-ASCII characters.
- No references to streamlit, docker, coinbase, coingecko, kraken, or similar.
- Python 3.12 compatible, Windows PowerShell friendly.
"""

from __future__ import annotations

import os
import time
from typing import Any

from backend.config.redis_config import get_redis_client

# ---- Configuration ----
# All Live Data, No Fallback/Hardcoded Data
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    msg = "REDIS_URL environment variable is required - no fallback/hardcoded Redis URL"
    raise RuntimeError(msg)

MIN_SYMBOL_SECONDS = int(os.getenv("MIN_SYMBOL_SECONDS", "90"))
MIN_STRATEGY_SECONDS = int(os.getenv("MIN_STRATEGY_SECONDS", "30"))
MIN_GLOBAL_SECONDS = int(os.getenv("MIN_GLOBAL_SECONDS", "5"))
MAX_DAILY_ORDERS = int(os.getenv("MAX_DAILY_ORDERS", "90"))

_KEY_PREFIX = "cd:"  # centralize key prefix to avoid typos


def _utc_day_key() -> str:
    """Return YYYYMMDD for the current UTC day."""
    return time.strftime("%Y%m%d", time.gmtime())


def _now() -> float:
    """Epoch seconds (UTC)."""
    return time.time()


class AutobuyCooldowns:
    """
    Redis-backed cooldown helper.

    Keys written (all UTC-based):
      - cd:last_global -> float epoch
      - cd:last_symbol:{symbol} -> float epoch
      - cd:last_strategy:{strategy_id} -> float epoch
      - cd:daily_count:{YYYYMMDD} -> int, expires in ~28h
    """

    def __init__(self, r: Any) -> None:
        self.r = r

    @classmethod
    async def create(cls) -> AutobuyCooldowns:
        """
        Create an instance using shared Redis pool.
        """
        r = get_redis_client()
        return cls(r)

    # ---- internals ----

    async def _ok_since(self, key: str, min_secs: int) -> bool:
        """
        True if key is absent or last timestamp is older than min_secs.
        """
        last = await self.r.get(key)
        if not last:
            return True
        try:
            return (_now() - float(last)) >= min_secs
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Fail-closed: block trading if cooldown value is corrupt
            return False

    # ---- API ----

    async def can_trade(self, symbol: str, strategy_id: str) -> bool:
        """
        Check if trading is allowed given current cooldowns and daily order cap.
        """
        if not await self._ok_since(f"{_KEY_PREFIX}last_global", MIN_GLOBAL_SECONDS):
            return False
        if not await self._ok_since(f"{_KEY_PREFIX}last_symbol:{symbol}", MIN_SYMBOL_SECONDS):
            return False
        if not await self._ok_since(f"{_KEY_PREFIX}last_strategy:{strategy_id}", MIN_STRATEGY_SECONDS):
            return False

        cnt_key = f"{_KEY_PREFIX}daily_count:{_utc_day_key()}"
        cnt = await self.r.get(cnt_key)
        try:
            if cnt is not None and int(cnt) >= MAX_DAILY_ORDERS:
                return False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return False  # corrupt counter: deny trading
        return True

    async def set_trade_cooldown(self, symbol: str, strategy_id: str) -> None:
        """
        Record that we traded now: set cooldown timestamps and increment the daily counter.
        """
        now_s = str(_now())
        cnt_key = f"{_KEY_PREFIX}daily_count:{_utc_day_key()}"

        pipe = self.r.pipeline(transaction=True)
        pipe.set(f"{_KEY_PREFIX}last_global", now_s)
        pipe.set(f"{_KEY_PREFIX}last_symbol:{symbol}", now_s)
        pipe.set(f"{_KEY_PREFIX}last_strategy:{strategy_id}", now_s)
        pipe.incr(cnt_key)
        # Expire late next day (~28h) to allow time zone shifts and late processes.
        pipe.expire(cnt_key, 28 * 3600)
        await pipe.execute()

    async def metrics(self) -> dict[str, int]:
        """
        Return current configuration and today's count (for dashboards/health checks).
        """
        cnt = await self.r.get(f"{_KEY_PREFIX}daily_count:{_utc_day_key()}")
        today_count = 0
        try:
            if cnt is not None:
                today_count = int(cnt)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            today_count = 0

        return {
            "min_symbol_seconds": MIN_SYMBOL_SECONDS,
            "min_strategy_seconds": MIN_STRATEGY_SECONDS,
            "min_global_seconds": MIN_GLOBAL_SECONDS,
            "max_daily_orders": MAX_DAILY_ORDERS,
            "today_count": today_count,
        }
