"""
Trade State Store

Manages per-symbol trade state transitions:
- IDLE: No position, can enter
- IN_TRADE: Position open, cannot re-enter same symbol
- COOLDOWN: Recently exited, blocked from re-entry for cooldown period

Uses Redis for fast lookups with SQLite fallback for persistence.
Fails OPEN by default to allow trading when state is unknown.
"""

from __future__ import annotations

import inspect
import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Cooldown period after exit (seconds)
DEFAULT_EXIT_COOLDOWN_SEC = 300  # 5 minutes


def normalize_trade_state_symbol(symbol: str) -> str:
    """Binance-style bus used in Redis keys (ETHUSDT)."""
    s = str(symbol).upper().replace("/", "")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def trade_state_redis_key(symbol: str) -> str:
    """Redis hash key for per-symbol trade state (must match TradeStateStore)."""
    return f"trade_state:{normalize_trade_state_symbol(symbol)}"


def clear_trade_state_redis_sync(symbol: str) -> None:
    """
    Delete trade_state:* for symbol when the portfolio engine has no matching open position.
    Sync client (orphan cleanup / thread paths).
    """
    try:
        from backend.config.redis_config import get_redis_client

        rc = get_redis_client()
        if not rc:
            return
        key = trade_state_redis_key(symbol)
        rc.delete(key)
        logger.info("TRADE_STATE: cleared Redis key %s (sync)", key)
    except Exception as e:
        logger.warning("TRADE_STATE: clear_trade_state_redis_sync failed for %s: %s", symbol, e)


class TradeStateEnum(Enum):
    """Trade state for a symbol"""

    IDLE = "IDLE"
    IN_TRADE = "IN_TRADE"
    COOLDOWN = "COOLDOWN"


class TradeStateStore:
    """
    Manages trade state per symbol.

    State transitions:
    - IDLE -> IN_TRADE (on entry fill)
    - IN_TRADE -> COOLDOWN (on exit)
    - COOLDOWN -> IDLE (after cooldown expires)
    """

    def __init__(self, redis_client: Any = None):
        self.redis_client = redis_client
        self._local_state: dict[str, dict] = {}

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to consistent format"""
        return normalize_trade_state_symbol(symbol)

    def _get_state_key(self, symbol: str) -> str:
        """Redis key for trade state"""
        return f"trade_state:{self._normalize_symbol(symbol)}"

    def _decode_redis_hash(self, data: dict) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            val = v.decode() if isinstance(v, bytes) else str(v)
            out[key] = val
        return out

    async def _fetch_state_data(self, symbol: str) -> dict[str, str]:
        """Read trade-state hash from Redis (sync or async client) with local fallback."""
        if self.redis_client:
            key = self._get_state_key(symbol)
            result = self.redis_client.hgetall(key)
            if inspect.isawaitable(result):
                result = await result
            if result:
                return self._decode_redis_hash(result)
        local = self._local_state.get(symbol)
        if isinstance(local, dict) and local:
            return {str(k): str(v) for k, v in local.items()}
        return {}

    async def allow_new_entry_async(
        self,
        symbol: str,
        side: str,
        now_ts: float,
        metrics: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Check if new entry is allowed for symbol.

        Returns:
            (allowed: bool, reason: str)
        """
        symbol = self._normalize_symbol(symbol)

        # Only block BUY entries (SELL is always allowed for position management)
        if side.lower() != "buy":
            return True, "SELL_ALLOWED"

        try:
            state_data = await self._fetch_state_data(symbol)
            if state_data:
                state = state_data.get("state", "IDLE")

                if state == "IN_TRADE":
                    return False, "ALREADY_IN_TRADE"

                if state == "COOLDOWN":
                    cooldown_until = float(state_data.get("cooldown_until", 0) or 0)
                    if now_ts < cooldown_until:
                        return False, f"COOLDOWN_ACTIVE_UNTIL_{int(cooldown_until)}"

            return True, "ENTRY_ALLOWED"

        except Exception as e:
            logger.warning(
                "allow_new_entry_async error for %s: %s - allowing entry (fail-open)",
                symbol,
                e,
            )
            return True, f"GATE_ERROR_FAIL_OPEN:{type(e).__name__}"

    def on_entry_fill(self, symbol: str, price: float, atr: float = 0.0) -> None:
        """Record entry fill: IDLE -> IN_TRADE"""
        symbol = self._normalize_symbol(symbol)

        try:
            state_data = {
                "state": "IN_TRADE",
                "entry_price": str(price),
                "entry_time": str(time.time()),
                "atr": str(atr),
            }

            if self.redis_client:
                # Use sync hset if available, otherwise store locally
                try:
                    self.redis_client.hset(self._get_state_key(symbol), mapping=state_data)
                except Exception:
                    pass

            self._local_state[symbol] = state_data
            logger.debug("TRADE_STATE: %s -> IN_TRADE @ $%.4f", symbol, price)

        except Exception as e:
            logger.warning("on_entry_fill error for %s: %s", symbol, e)

    def on_exit(
        self,
        symbol: str,
        price: float,
        exit_reason: str,
        metrics: dict | None = None,
    ) -> None:
        """Record exit: IN_TRADE -> COOLDOWN"""
        symbol = self._normalize_symbol(symbol)

        try:
            cooldown_until = time.time() + DEFAULT_EXIT_COOLDOWN_SEC

            state_data = {
                "state": "COOLDOWN",
                "exit_price": str(price),
                "exit_time": str(time.time()),
                "exit_reason": exit_reason,
                "cooldown_until": str(cooldown_until),
            }

            if self.redis_client:
                try:
                    self.redis_client.hset(self._get_state_key(symbol), mapping=state_data)
                    # Set TTL so cooldown state expires
                    self.redis_client.expire(self._get_state_key(symbol), DEFAULT_EXIT_COOLDOWN_SEC + 60)
                except Exception:
                    pass

            self._local_state[symbol] = state_data
            logger.debug("TRADE_STATE: %s -> COOLDOWN until %.0f (reason: %s)", symbol, cooldown_until, exit_reason)

        except Exception as e:
            logger.warning("on_exit error for %s: %s", symbol, e)

    def get_state(self, symbol: str) -> TradeStateEnum:
        """Get current state for symbol"""
        symbol = self._normalize_symbol(symbol)

        try:
            if self.redis_client:
                state_data = self.redis_client.hgetall(self._get_state_key(symbol))
                if state_data:
                    state = state_data.get("state") or state_data.get(b"state")
                    if isinstance(state, bytes):
                        state = state.decode()

                    if state == "IN_TRADE":
                        return TradeStateEnum.IN_TRADE
                    elif state == "COOLDOWN":
                        cooldown_until = state_data.get("cooldown_until") or state_data.get(b"cooldown_until")
                        if cooldown_until:
                            if isinstance(cooldown_until, bytes):
                                cooldown_until = cooldown_until.decode()
                            if time.time() < float(cooldown_until):
                                return TradeStateEnum.COOLDOWN

            return TradeStateEnum.IDLE

        except Exception as e:
            logger.warning("get_state error for %s: %s", symbol, e)
            return TradeStateEnum.IDLE


# Singleton instance
_store_instance: TradeStateStore | None = None


def get_trade_state_store(redis_client: Any = None) -> TradeStateStore:
    """Get or create the trade state store singleton"""
    global _store_instance
    if _store_instance is None:
        _store_instance = TradeStateStore(redis_client)
    elif redis_client is not None and _store_instance.redis_client is None:
        _store_instance.redis_client = redis_client
    return _store_instance


def notify_exit(symbol: str, price: float, exit_reason: str, metrics: dict | None = None) -> None:
    """Convenience function to notify exit"""
    store = get_trade_state_store()
    store.on_exit(symbol, price, exit_reason, metrics)


def assert_exit_cooldown(symbol: str, order_id: str = "", path: str = "") -> None:
    """
    Assert that cooldown was properly set after exit.
    Logs warning if not in COOLDOWN state but doesn't raise.
    """
    store = get_trade_state_store()
    state = store.get_state(symbol)

    if state != TradeStateEnum.COOLDOWN:
        logger.warning(
            "EXIT_COOLDOWN_INVARIANT: %s not in COOLDOWN after exit (state=%s, order_id=%s, path=%s)",
            symbol,
            state.value,
            order_id,
            path,
        )
