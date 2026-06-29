"""REST fallback writer for orderbook:{BASE} Redis keys (top-4 only, throttled)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS

logger = logging.getLogger(__name__)


def _base(symbol_bus: str) -> str:
    s = (symbol_bus or "BTCUSDT").upper().replace("/USDT", "").replace("USDT", "").strip()
    return s or "BTC"


def _ccxt(symbol_bus: str) -> str:
    b = _base(symbol_bus)
    return f"{b}/USDT"


class OrderbookRedisRefresher:
    """Poll exchange L2 when WebSocket orderbook keys are cold."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = [_base(s) for s in (symbols or TRADING_SYMBOLS)]
        self.is_running = False
        self._task: asyncio.Task | None = None
        self._last_fetch: dict[str, float] = {}

    async def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop(), name="orderbook_redis_refresher:loop")
        logger.info("OrderbookRedisRefresher started for %s", self.symbols)
        try:
            from backend.config.redis_config import get_shared_redis_async

            r = await get_shared_redis_async()
            await self.refresh_all(r, force=True)
        except Exception as exc:
            logger.debug("OrderbookRedisRefresher boot refresh failed: %s", exc)

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def refresh_symbol(self, base: str, redis_client: Any) -> bool:
        from backend.services.order_book_service import fetch_order_book_features_live, write_orderbook_redis_async

        ccxt = f"{_base(base)}/USDT"
        feats = await fetch_order_book_features_live(ccxt)
        if not feats or float(feats.get("bid_ask_spread") or 0) <= 0:
            return False
        ok = await write_orderbook_redis_async(_base(base), feats, redis_client, source="rest_fallback")
        if ok:
            self._last_fetch[_base(base)] = time.time()
        return ok

    async def refresh_all(self, redis_client: Any, *, force: bool = False) -> dict[str, bool]:
        results: dict[str, bool] = {}
        min_gap = max(5.0, float(os.getenv("ORDERBOOK_MIN_INTERVAL_SEC", "10")))
        now = time.time()
        for base in self.symbols:
            if not force and now - self._last_fetch.get(base, 0.0) < min_gap:
                continue
            try:
                results[base] = await self.refresh_symbol(base, redis_client)
                await asyncio.sleep(0.25)
            except Exception as exc:
                logger.debug("orderbook refresh failed %s: %s", base, exc)
                results[base] = False
        return results

    async def _loop(self) -> None:
        import contextlib

        from backend.config.redis_config import get_shared_redis_async

        interval = max(15, int(os.getenv("ORDERBOOK_REFRESH_INTERVAL_SEC", "25")))
        while self.is_running:
            t0 = time.time()
            try:
                r = await get_shared_redis_async()
                await self.refresh_all(r)
            except Exception as exc:
                logger.debug("OrderbookRedisRefresher pass failed: %s", exc)
            elapsed = time.time() - t0
            await asyncio.sleep(max(5.0, interval - elapsed))


import contextlib

_refresher: OrderbookRedisRefresher | None = None


def get_orderbook_redis_refresher() -> OrderbookRedisRefresher:
    global _refresher
    if _refresher is None:
        _refresher = OrderbookRedisRefresher()
    return _refresher


__all__ = ["OrderbookRedisRefresher", "get_orderbook_redis_refresher"]
