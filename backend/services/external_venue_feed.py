"""Public Coinbase and Kraken tapes for BTC, ETH, SOL, and XRP.

The feed writes a redis snapshot. It does not place orders and it does not
read or write the trading database. A dead socket retries. It does not stop
the Binance.US market-data loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque

from backend.services.external_venue_tape import VenueTape

logger = logging.getLogger(__name__)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
_COINBASE = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
_KRAKEN = {"XBT/USD": "BTCUSDT", "ETH/USD": "ETHUSDT", "SOL/USD": "SOLUSDT", "XRP/USD": "XRPUSDT"}
REDIS_PREFIX = "external_discovery:"


def discovery_record(
    tapes: dict[str, dict[str, VenueTape]],
    binance_mids: dict[str, deque[tuple[float, float]]],
    symbol: str,
    now: float,
) -> dict[str, float] | None:
    """One symbol's outside-venue state. Both venues must be fresh or the record is absent."""
    mids = binance_mids.get(symbol) or deque()
    binance_mid = None
    binance_ret = None
    prior = None
    for ts, mid in mids:
        if ts <= now - 5.0:
            prior = mid
        if ts <= now:
            binance_mid = mid
    if binance_mid and prior and prior > 0.0:
        binance_ret = binance_mid / prior - 1.0
    venues = tapes.get(symbol) or {}
    coinbase = venues.get("coinbase").features(now, binance_mid, binance_ret) if venues.get("coinbase") else None
    kraken = venues.get("kraken").features(now, binance_mid, binance_ret) if venues.get("kraken") else None
    if coinbase is None or kraken is None:
        return None
    record = {"venues_fresh": 1.0, "as_of": now}
    for prefix, block in (("coinbase", coinbase), ("kraken", kraken)):
        for key, value in block.items():
            record[f"{prefix}_{key}"] = float(value)
    return record


async def external_discovery_loop() -> None:
    tapes = {symbol: {"coinbase": VenueTape(), "kraken": VenueTape()} for symbol in SYMBOLS}
    binance_mids = {symbol: deque(maxlen=400) for symbol in SYMBOLS}
    await asyncio.gather(
        _coinbase(tapes),
        _kraken(tapes),
        _publish(tapes, binance_mids),
    )


async def _publish(tapes: dict[str, dict[str, VenueTape]], binance_mids: dict[str, deque[tuple[float, float]]]) -> None:
    from backend.config.redis_config import get_shared_redis_sync
    from backend.services.spread_book_telemetry import read_market_book

    redis_client = get_shared_redis_sync()
    while True:
        now = time.time()
        for symbol in SYMBOLS:
            try:
                book = read_market_book(redis_client, symbol)
                if book and float(book.get("bid") or 0) > 0 and float(book.get("ask") or 0) > 0:
                    mid = (float(book["bid"]) + float(book["ask"])) / 2.0
                    binance_mids[symbol].append((now, mid))
            except Exception:
                logger.debug("external discovery binance read failed %s", symbol, exc_info=True)
            record = discovery_record(tapes, binance_mids, symbol, now)
            key = REDIS_PREFIX + symbol
            try:
                if record is None:
                    redis_client.delete(key)
                else:
                    redis_client.hset(key, mapping={name: str(value) for name, value in record.items()})
                    redis_client.expire(key, 30)
            except Exception:
                logger.debug("external discovery redis write failed %s", symbol, exc_info=True)
        await asyncio.sleep(0.5)


def _epoch(value: str) -> float | None:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        from datetime import datetime

        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


async def _coinbase(tapes: dict[str, dict[str, VenueTape]]) -> None:
    import websockets

    products = list(_COINBASE.values())
    reverse = {product: symbol for symbol, product in _COINBASE.items()}
    url = "wss://ws-feed.exchange.coinbase.com"
    subscribe = {"type": "subscribe", "product_ids": products, "channels": ["ticker", "matches"]}
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                await ws.send(json.dumps(subscribe))
                async for raw in ws:
                    message = json.loads(raw)
                    symbol = reverse.get(str(message.get("product_id") or ""))
                    if symbol is None:
                        continue
                    tape = tapes[symbol]["coinbase"]
                    kind = message.get("type")
                    if kind == "ticker":
                        ts = _epoch(str(message.get("time") or "")) or time.time()
                        tape.add_book(ts, float(message.get("best_bid") or 0), float(message.get("best_ask") or 0))
                    elif kind in ("match", "last_match"):
                        ts = _epoch(str(message.get("time") or "")) or time.time()
                        tape.add_trade(ts, float(message.get("size") or 0), str(message.get("side") or "") == "buy")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("coinbase discovery socket dropped", exc_info=True)
            await asyncio.sleep(2.0)


async def _kraken(tapes: dict[str, dict[str, VenueTape]]) -> None:
    import websockets

    url = "wss://ws.kraken.com"
    pairs = list(_KRAKEN)
    subscribe = [
        {"event": "subscribe", "pair": pairs, "subscription": {"name": "ticker"}},
        {"event": "subscribe", "pair": pairs, "subscription": {"name": "trade"}},
    ]
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                for item in subscribe:
                    await ws.send(json.dumps(item))
                async for raw in ws:
                    message = json.loads(raw)
                    if not isinstance(message, list) or len(message) < 4:
                        continue
                    symbol = _KRAKEN.get(str(message[-1]))
                    if symbol is None:
                        continue
                    tape = tapes[symbol]["kraken"]
                    channel = str(message[-2])
                    if channel == "ticker" and isinstance(message[1], dict):
                        bid = float((message[1].get("b") or [0])[0] or 0)
                        ask = float((message[1].get("a") or [0])[0] or 0)
                        tape.add_book(time.time(), bid, ask)
                    elif channel == "trade" and isinstance(message[1], list):
                        for print_row in message[1]:
                            if len(print_row) < 4:
                                continue
                            tape.add_trade(float(print_row[2]), float(print_row[1]), str(print_row[3]) == "b")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("kraken discovery socket dropped", exc_info=True)
            await asyncio.sleep(2.0)


def read_discovery(symbol: str) -> dict[str, float] | None:
    """The latest outside-venue snapshot. A stale or absent key stays absent."""
    try:
        from backend.config.redis_config import get_shared_redis_sync

        raw = get_shared_redis_sync().hgetall(REDIS_PREFIX + str(symbol).upper())
    except Exception:
        return None
    if not raw:
        return None
    out: dict[str, float] = {}
    for key, value in raw.items():
        name = key.decode() if isinstance(key, bytes) else str(key)
        text = value.decode() if isinstance(value, bytes) else str(value)
        try:
            out[name] = float(text)
        except ValueError:
            continue
    as_of = out.get("as_of")
    if as_of is None or time.time() - as_of > 5.0 or out.get("venues_fresh") != 1.0:
        return None
    return out
