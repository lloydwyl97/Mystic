"""Aggregated-trade tape collector — feeds real aggressor buy/sell volume
into the microstructure engine (backend.services.microstructure_engine).

Subscribes to Binance.US ``{sym}usdt@aggTrade`` streams for the top-4 traded
symbols. Each print carries ``isBuyerMaker``:
  * ``isBuyerMaker=True``  -> the resting order was a BUY  -> aggressor SOLD.
  * ``isBuyerMaker=False`` -> the resting order was a SELL -> aggressor BOUGHT.

This is real trade-tape data (not an OHLCV sign*volume proxy). Output feeds
ranking/EV only — see microstructure_engine module docstring for the
architecture rule (nothing here can block a trade).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from backend.utils.network_ipv4 import ensure_ipv4_only

ensure_ipv4_only()

import websockets

from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


class AggTradeCollector:
    """Collects real-time aggregated trade prints for aggressor-volume features."""

    def __init__(self) -> None:
        self.is_running = False
        self.ws_url = "wss://stream.binance.us:9443/stream"
        self.symbols: list[str] = []
        self.websocket = None
        self._last_heartbeat_ts = 0.0
        self._ws_task: asyncio.Task | None = None

        symbols_str = os.getenv("TRADING_SYMBOLS", "BTC,ETH,SOL,XRP")
        self.symbols = [s.strip().replace("USDT", "") for s in symbols_str.split(",")]

        self.stats = {
            "messages_received": 0,
            "trades_processed": 0,
            "errors": 0,
            "reconnects": 0,
            "last_error": None,
            "by_symbol": dict.fromkeys(self.symbols, 0),
            "last_ts_by_symbol": dict.fromkeys(self.symbols, 0.0),
        }

        logger.info(f"AggTradeCollector initialized for {len(self.symbols)} symbols")

    async def start(self) -> None:
        self.is_running = True
        self._ws_task = await task_manager.create_task(self._run_websocket(), name="agg_trade_collector:run_websocket")
        logger.info("AggTradeCollector started - connecting to Binance WebSocket")

    async def stop(self) -> None:
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
        logger.info("AggTradeCollector stopped")

    async def _run_websocket(self) -> None:
        while self.is_running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.warning(f"AggTrade WebSocket error: {e}, reconnecting in 5s...")
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
                self.stats["reconnects"] += 1
                await asyncio.sleep(5)

    async def _connect_and_listen(self) -> None:
        streams = [f"{symbol.lower()}usdt@aggTrade" for symbol in self.symbols]
        url = f"{self.ws_url}?streams={'/'.join(streams)}"
        async with websockets.connect(url, open_timeout=30, ping_interval=20, ping_timeout=20) as websocket:
            self.websocket = websocket
            logger.info(f"Connected to {len(streams)} aggTrade combined streams")

            async for message in websocket:
                if not self.is_running:
                    break
                try:
                    await self._process_message(message)
                except Exception as e:
                    logger.debug(f"Error processing aggTrade message: {e}")
                    self.stats["errors"] += 1

    async def _process_message(self, message: str) -> None:
        """
        aggTrade payload (within {"stream":..., "data": {...}}):
        {
          "e": "aggTrade", "E": 123456789, "s": "BTCUSDT", "a": 12345,
          "p": "0.001", "q": "100", "f": 100, "l": 105, "T": 123456785,
          "m": true   # isBuyerMaker
        }
        """
        try:
            data = json.loads(message)
            if "result" in data:
                return
            if isinstance(data.get("data"), dict):
                payload = data["data"]
            elif str(data.get("e") or "") == "aggTrade":
                payload = data
            else:
                return
            symbol = str(payload.get("s") or "").replace("USDT", "")
            qty = float(payload.get("q") or 0.0)
            is_buyer_maker = bool(payload.get("m"))
            trade_ts_ms = payload.get("T")
            ts = (float(trade_ts_ms) / 1000.0) if trade_ts_ms else None

            if not symbol or qty <= 0:
                return

            with_import_ok = True
            try:
                from backend.services.microstructure_engine import record_agg_trade
            except Exception:
                with_import_ok = False
            if with_import_ok:
                record_agg_trade(symbol, qty=qty, is_buyer_maker=is_buyer_maker, ts=ts)
            try:
                from backend.services.binance_scalp.structural_tape import parse_agg_payload, publish_trade_event

                ev = parse_agg_payload(payload, recv_ts=ts)
                if ev is not None:
                    from backend.config.redis_config import get_shared_redis_sync

                    publish_trade_event(get_shared_redis_sync(), ev)
            except Exception:
                pass

            self.stats["messages_received"] += 1
            self.stats["trades_processed"] += 1
            by_sym = self.stats.setdefault("by_symbol", {})
            prev = int(by_sym.get(symbol) or 0)
            by_sym[symbol] = prev + 1
            last_ts = self.stats.setdefault("last_ts_by_symbol", {})
            last_ts[symbol] = float(ts or 0.0)
            await self._heartbeat_throttled(symbol)

            if prev == 0:
                logger.info(f"First aggTrade processed for {symbol}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse aggTrade WebSocket message: {e}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.warning(f"Error processing aggTrade message: {e}")
            self.stats["errors"] += 1

    async def get_stats(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "symbols": self.symbols,
            "stats": self.stats,
        }

    async def _heartbeat_throttled(self, symbol: str) -> None:
        now = asyncio.get_event_loop().time()
        if now - self._last_heartbeat_ts < 10.0:
            return
        self._last_heartbeat_ts = now
        try:
            from backend.config.redis_config import get_shared_redis_async
            from backend.services.task_health_monitor import beat

            await beat(
                "agg_trade_collector:ws_messages",
                get_shared_redis_async(),
                extra={
                    "last_symbol": symbol,
                    "messages_received": self.stats["messages_received"],
                    "by_symbol": self.stats.get("by_symbol") or {},
                },
            )
        except Exception:
            pass


# Singleton instance
agg_trade_collector = AggTradeCollector()
