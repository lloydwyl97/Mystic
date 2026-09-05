"""
Binance.US WS Hydrator - LIVE ONLY
Streams miniTicker + bookTicker for the Top-10 Binance.US symbols (no env overrides).
Publishes prices, microstructure features, and rolls lightweight 1m/5m/15m candles.

Gap policy:
- Authoritative source: Binance.US kline_1m websocket stream (x=True closed bars).
- On each closed kline event, flush the bar directly; do not wait for the next miniTicker.
- On reconnect, backfill any missed closed bars from REST /api/v3/klines.
- Forming bars (x=False) are NOT written to klines:{SYM}:1m.
- No candles are fabricated; do not forward-fill.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import time
from collections import deque
from typing import Any

import httpx
import websockets

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.metrics import (
    ws_disconnects_total,
    ws_inter_message_seconds,
    ws_last_tick_ts,
    ws_messages_total,
    ws_reconnects_total,
)
from backend.services.task_manager import task_manager
from backend.utils.cache_guard import CacheGuard

logger = logging.getLogger(__name__)

BINANCEUS_WS = "wss://stream.binance.us:9443/stream"
# Use TRADING_SYMBOLS from trading_universe (live data) - enforce Top-10 (uppercase tickers, e.g., BTCUSDT)
SYMBOLS: list[str] = sorted(TRADING_SYMBOLS)


def _streams() -> str:
    mini = [f"{s.lower()}@miniTicker" for s in SYMBOLS]
    bt = [f"{s.lower()}@bookTicker" for s in SYMBOLS]
    # Add kline stream for volume data
    kline = [f"{s.lower()}@kline_1m" for s in SYMBOLS]
    return "/".join(mini + bt + kline)


class BinanceWSHydrator:
    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._cg: CacheGuard | None = None
        # Rolling windows per symbol for lightweight features
        self._ticks: dict[str, deque[tuple[float, float]]] = {}
        # In-process 1m candle builder per symbol: [start_ts, o, h, l, c, v]
        self._c1m: dict[str, list[float]] = {}
        # Track last closed bar's open-timestamp (seconds) per symbol; used for gap detection and backfill
        self._last_bar_ts: dict[str, int] = {}
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []

    async def start(self) -> None:
        self._cg = await CacheGuard.create()
        task = await task_manager.create_task(self._run(), name="binance_ws_hydrator:run")
        self._tasks.append(task)

    async def stop(self) -> None:
        self._stop.set()
        # Cancel and await all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tasks.clear()

    async def _run(self) -> None:
        backoff = 2.0
        last_ts_by_symbol: dict[str, float] = {}
        while not self._stop.is_set():
            try:
                url = f"{BINANCEUS_WS}?streams={_streams()}"
                try:
                    connect_timeout = float(os.getenv("BINANCE_WS_CONNECT_TIMEOUT", "15") or "15")
                except (ValueError, TypeError):
                    connect_timeout = 15.0
                ws = None
                try:
                    ws = await asyncio.wait_for(
                        websockets.connect(url, ping_interval=20, ping_timeout=20),
                        timeout=connect_timeout,
                    )
                except asyncio.TimeoutError:
                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        ws_disconnects_total.labels(reason="connect_timeout").inc()
                    await asyncio.sleep(min(backoff, 30.0))
                    backoff = min(backoff * 1.7, 30.0)
                    continue

                async with ws:
                    backoff = 2.0
                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        ws_reconnects_total.inc()
                    # Backfill any closed bars that were missed during the disconnection gap.
                    for _sym in SYMBOLS:
                        with contextlib.suppress(Exception):
                            await self._backfill_missing_bars(_sym)
                    while not self._stop.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(msg)
                        payload = data.get("data") or {}
                        sym = (payload.get("s") or "").upper()
                        if not sym or sym not in SYMBOLS or self._cg is None:
                            continue

                        stream = data.get("stream") or ""

                        # Per-symbol inter-message timing
                        try:
                            now_ts = time.time()
                            prev = last_ts_by_symbol.get(sym)
                            if prev is not None:
                                ws_inter_message_seconds.labels(symbol=sym).observe(max(0.0, now_ts - prev))
                            last_ts_by_symbol[sym] = now_ts
                            ws_last_tick_ts.labels(symbol=sym).set(int(now_ts))
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass

                        if stream.endswith("@miniTicker"):
                            price = payload.get("c")
                            if price:
                                try:
                                    px = float(price)
                                    now = time.time()
                                    # Persist the latest price (authoritative cache handles exchange normalization)
                                    await self._cg.set_price(sym, px)

                                    # CRITICAL: Always update global market data timestamp when we receive ANY price update
                                    # This ensures market_data:last_update stays fresh even if updates come slowly
                                    try:
                                        await self._cg.mark_market_update("ws")
                                    except Exception as mark_err:
                                        # Log but don't fail - price was set successfully
                                        logger.debug(f"Failed to mark market update for {sym}: {mark_err}")

                                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                        ws_messages_total.labels(type="miniTicker", symbol=sym).inc()
                                    # Rolling window for features
                                    dq = self._ticks.get(sym)
                                    if dq is None:
                                        dq = deque(maxlen=600)  # ~10 minutes at ~1 tick/sec
                                        self._ticks[sym] = dq
                                    dq.append((now, px))
                                    # Trim entries older than 5 minutes
                                    cutoff_5m = now - 300.0
                                    while dq and dq[0][0] < cutoff_5m:
                                        dq.pop() if False else dq.popleft()
                                    # Compute + publish features
                                    await self._compute_and_publish_features(sym, dq, now)
                                    # Update 1m candle
                                    await self._update_candles(sym, now, px)
                                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                    pass

                        elif stream.endswith("@bookTicker"):
                            bid = payload.get("b")
                            ask = payload.get("a")
                            bid_qty = payload.get("B")
                            ask_qty = payload.get("A")
                            try:
                                if bid is not None and ask is not None:
                                    b = float(bid)
                                    a = float(ask)
                                    mid = (a + b) / 2.0 if (a > 0 and b > 0) else 0.0
                                    spread_bp = ((a - b) / mid * 10000.0) if mid > 0 else 0.0
                                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                        ws_messages_total.labels(type="bookTicker", symbol=sym).inc()
                                    # Publish microstructure to feature hash
                                    try:
                                        r = self._cg.r  # type: ignore[attr-defined]
                                        key = f"feature:{sym}"
                                        imb = None
                                        dr = None
                                        try:
                                            if bid_qty is not None and ask_qty is not None:
                                                Bq = float(bid_qty)
                                                Aq = float(ask_qty)
                                                tot = Bq + Aq
                                                if tot > 0:
                                                    imb = (Bq - Aq) / tot
                                                    dr = Bq / tot
                                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                            pass
                                        mapping = {
                                            "spread_bp": f"{spread_bp:.4f}",
                                            "mid": f"{mid:.8f}",
                                            "ts": str(int(time.time())),
                                        }
                                        if imb is not None:
                                            mapping["imbalance"] = f"{imb:.6f}"
                                        if dr is not None:
                                            mapping["depth_ratio"] = f"{dr:.6f}"

                                        # Write mapping to Redis inline to avoid spawning thousands of tasks
                                        for k, v in mapping.items():
                                            await r.hset(key, k, v)
                                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                        pass
                            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                pass

                        elif "@kline_" in stream:
                            # Kline stream provides authoritative OHLCV per 1m bar.
                            # When k["x"] == True the bar is closed and must be flushed immediately.
                            # Forming bars (x=False) are used only to keep the in-memory OHLCV
                            # up-to-date; they are NOT written to Redis.
                            k = payload.get("k")
                            if k:
                                try:
                                    volume = float(k.get("v", 0))

                                    # Update volume in current forming candle
                                    cur = self._c1m.get(sym)
                                    if cur and len(cur) == 6:
                                        cur[5] = volume

                                    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                        ws_messages_total.labels(type="kline", symbol=sym).inc()

                                    # Flush the bar to Redis when it is closed (x=True).
                                    # This is the primary candle-write path; miniTicker is the fallback.
                                    if k.get("x"):
                                        task = await task_manager.create_task(
                                            self._flush_closed_kline(sym, k),
                                            name="binance_ws_hydrator:flush_closed_kline",
                                        )
                                        self._tasks.append(task)
                                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                                    pass

            except Exception as exc:  # catch-all to avoid retaining traceback references
                # Increments metric
                with contextlib.suppress(Exception):
                    ws_disconnects_total.labels(reason="exception").inc()

                # Ensure websocket is closed to release StreamReader/Writer
                with contextlib.suppress(Exception):
                    if ws is not None and not ws.closed:
                        await ws.close()

                # Explicitly delete the exception to break traceback reference cycles
                del exc

                # Backoff before reconnecting
                await asyncio.sleep(min(backoff, 30.0) * 1.25)
                backoff = min(backoff * 1.7, 30.0)

    async def _compute_and_publish_features(self, sym: str, dq: deque[tuple[float, float]], now_ts: float) -> None:
        try:
            if self._cg is None or not dq or len(dq) < 15:
                return
            # z-score of price over last 60s
            cutoff_60s = now_ts - 60.0
            last_60 = [p for (t, p) in dq if t >= cutoff_60s]
            z_price_60s = 0.0
            if len(last_60) >= 5:
                mean_p = sum(last_60) / len(last_60)
                var_p = sum((p - mean_p) * (p - mean_p) for p in last_60) / max(1, (len(last_60) - 1))
                std_p = math.sqrt(var_p) if var_p > 0 else 0.0
                z_price_60s = (last_60[-1] - mean_p) / std_p if std_p > 0 else 0.0

            # realized volatility over last 5m (std of log returns)
            prices_5m = [p for (_t, p) in dq]
            realized_vol_5m = 0.0
            if len(prices_5m) >= 20:
                rets = []
                for i in range(1, len(prices_5m)):
                    p0 = prices_5m[i - 1]
                    p1 = prices_5m[i]
                    if p0 > 0 and p1 > 0:
                        rets.append(math.log(p1 / p0))
                if len(rets) >= 5:
                    mean_r = sum(rets) / len(rets)
                    var_r = sum((r - mean_r) * (r - mean_r) for r in rets) / max(1, (len(rets) - 1))
                    realized_vol_5m = math.sqrt(var_r)

            # RSI-14 from last closes
            rsi_14 = 0.0
            closes = [p for (_t, p) in dq][-15:]
            if len(closes) >= 15:
                gains = []
                losses = []
                for i in range(1, len(closes)):
                    ch = closes[i] - closes[i - 1]
                    gains.append(max(ch, 0.0))
                    losses.append(max(-ch, 0.0))
                avg_gain = sum(gains[-14:]) / 14.0
                avg_loss = sum(losses[-14:]) / 14.0
                if avg_loss == 0:
                    rsi_14 = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

            # Publish features to Redis via CacheGuard's client
            try:
                r = self._cg.r  # type: ignore[attr-defined]
                key = f"feature:{sym}"
                # Regime classification by volatility
                regime = "low"
                if realized_vol_5m >= 0.01:
                    regime = "high"
                elif realized_vol_5m >= 0.003:
                    regime = "medium"

                mapping = {
                    "z_price_60s": f"{z_price_60s:.6f}",
                    "realized_vol_5m": f"{realized_vol_5m:.6f}",
                    "rsi_14": f"{rsi_14:.4f}",
                    "regime": regime,
                    "ts": str(int(now_ts)),
                }

                # Fire-and-forget HSET with mapping=
                async def _hset_task():
                    for k, v in mapping.items():
                        await r.hset(key, k, v)

                task = await task_manager.create_task(_hset_task(), name="binance_ws_hydrator:hset_task_2")
                self._tasks.append(task)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pass
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

    async def _update_candles(self, sym: str, now_ts: float, px: float) -> None:
        """Track the forming (in-progress) 1m candle from miniTicker price ticks.

        This method ONLY maintains in-memory state (_c1m).  It does NOT write
        anything to Redis.  Closed bars are persisted exclusively by
        _flush_closed_kline (authoritative @kline_1m x=True events) and
        _backfill_missing_bars (REST catch-up on reconnect).

        Keeping the two responsibilities separate eliminates the dual-write race
        where both the miniTicker and kline paths would call _append_candle for
        the same minute and produce duplicate timestamps in the Redis array.
        """
        try:
            if self._cg is None:
                return
            start_ts = int(now_ts // 60) * 60
            cur = self._c1m.get(sym)
            if not cur or int(cur[0]) != start_ts:
                # New minute: start a fresh forming candle.
                # The previous minute's closed bar will be persisted by the
                # @kline_1m x=True event, not here.
                self._c1m[sym] = [float(start_ts), px, px, px, px, 0.0]
                return
            # Update the forming candle with the latest miniTicker price.
            cur[3] = min(cur[3], px)  # low
            cur[2] = max(cur[2], px)  # high
            cur[4] = px  # close
            # Volume is updated by @kline_1m events; miniTicker carries no volume.
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

    async def _append_candle(self, sym: str, interval: str, candle: list[float]) -> None:
        """Write one candle to the Redis klines history.

        Upserts by open-timestamp: if a row with the same bar_ts already exists it
        is replaced in-place rather than appended.  This prevents the miniTicker
        and kline paths from creating duplicate rows for the same minute even when
        both paths race to call this method.
        """
        try:
            if self._cg is None:
                return
            r = self._cg.r  # type: ignore[attr-defined]
            key = f"klines:{sym}:{interval}"

            raw = await r.get(key)
            arr: list[list[float]] = []
            if raw:
                try:
                    arr = json.loads(raw)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    arr = []
            bar_ts = candle[0]
            row = [candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]]
            # Upsert: replace an existing row for this minute, or append if new.
            existing_idx = next((i for i, r_row in enumerate(arr) if r_row[0] == bar_ts), None)
            if existing_idx is not None:
                arr[existing_idx] = row
            else:
                arr.append(row)
            # Trim to last 600
            if len(arr) > 600:
                arr = arr[-600:]
            await r.set(key, json.dumps(arr), ex=900)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

    async def _flush_closed_kline(self, sym: str, k: dict[str, Any]) -> None:
        """Flush an exchange-authoritative closed 1m kline to Redis.

        Uses OHLCV from the kline event directly.  Does NOT fabricate data.
        Only called when k["x"] is True (bar is closed by the exchange).
        """
        try:
            if self._cg is None:
                return
            bar_ts_ms = int(k.get("t") or 0)
            if bar_ts_ms <= 0:
                return
            bar_ts = bar_ts_ms // 1000  # UTC open-time in seconds
            o = float(k.get("o") or 0)
            h = float(k.get("h") or 0)
            lo = float(k.get("l") or 0)
            c = float(k.get("c") or 0)
            v = float(k.get("v") or 0)
            if c <= 0:
                return
            # Avoid writing a bar older than what we already have
            last = self._last_bar_ts.get(sym, 0)
            if bar_ts <= last:
                return
            candle = [float(bar_ts), o, h, lo, c, v]
            await self._append_candle(sym, "1m", candle)
            self._last_bar_ts[sym] = bar_ts
            # If the in-memory forming candle covers the same minute, discard it
            # so the next miniTicker starts a fresh bar for the next minute.
            cur = self._c1m.get(sym)
            if cur and int(cur[0]) == bar_ts:
                self._c1m.pop(sym, None)
            await self._maybe_rollup(sym, bar_ts)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

    async def _backfill_missing_bars(self, sym: str) -> None:
        """Fetch any closed 1m bars that were missed during a WS disconnection.

        Only fills bars that are definitively closed (>= 60s before now).
        Does NOT fabricate data; does NOT forward-fill; does NOT write forming bars.
        """
        try:
            if self._cg is None:
                return
            last_ts = self._last_bar_ts.get(sym, 0)
            if last_ts <= 0:
                return
            now_ts = int(time.time())
            # Skip if fewer than 2 bars might be missing (at least 120s gap needed)
            if now_ts - last_ts < 120:
                return
            # Backfill window: from the bar after last_ts to 1 full minute ago (closed bars only)
            start_ms = (last_ts + 60) * 1000
            end_ms = ((now_ts // 60) * 60 - 60) * 1000  # latest fully-closed bar
            if end_ms < start_ms:
                return
            n_missing = (end_ms // 1000 - start_ms // 1000) // 60 + 1
            if n_missing <= 0:
                return
            url = "https://api.binance.us/api/v3/klines"
            params: dict[str, Any] = {
                "symbol": sym,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(300, n_missing + 5),
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data: list[Any] = resp.json()
            if not data:
                return
            r = self._cg.r  # type: ignore[attr-defined]
            raw = await r.get(f"klines:{sym}:1m")
            arr: list[Any] = []
            if raw:
                try:
                    arr = json.loads(raw)
                except (ValueError, TypeError):
                    arr = []
            existing_ts: set[int] = {int(row[0]) for row in arr}
            added = 0
            for kline in data:
                bar_ts = int(kline[0]) // 1000
                if bar_ts in existing_ts:
                    continue
                if float(kline[4] or 0) <= 0:
                    continue  # skip zero-close bars
                candle = [
                    float(bar_ts),
                    float(kline[1]),
                    float(kline[2]),
                    float(kline[3]),
                    float(kline[4]),
                    float(kline[5]),
                ]
                arr.append(candle)
                existing_ts.add(bar_ts)
                added += 1
            if added > 0:
                arr.sort(key=lambda x: x[0])
                if len(arr) > 600:
                    arr = arr[-600:]
                await r.set(f"klines:{sym}:1m", json.dumps(arr), ex=900)
                self._last_bar_ts[sym] = max(int(row[0]) for row in arr)
                logger.info(
                    "Backfilled %d missing 1m bars for %s (gap was %ds, last_ts=%d)",
                    added,
                    sym,
                    now_ts - last_ts,
                    last_ts,
                )
        except Exception as exc:
            logger.debug("Backfill failed for %s: %s", sym, exc)

    async def _maybe_rollup(self, sym: str, last_start_ts: int) -> None:
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            task = await task_manager.create_task(self._rollup_async(sym, last_start_ts), name="binance_ws_hydrator:rollup_async")
            self._tasks.append(task)

    async def _rollup_async(self, sym: str, last_start_ts: int) -> None:
        try:
            if self._cg is None:
                return
            r = self._cg.r  # type: ignore[attr-defined]

            # Get 1m series
            raw = await r.get(f"klines:{sym}:1m")
            if not raw:
                return
            kl = json.loads(raw)
            if not isinstance(kl, list) or len(kl) < 5:
                return
            # 5m roll every 5 minutes
            if (last_start_ts % 300) == 240 and len(kl) >= 5:
                chunk = kl[-5:]
                ts0 = chunk[0][0]
                o = float(chunk[0][1])
                h = max(float(x[2]) for x in chunk)
                low = min(float(x[3]) for x in chunk)
                c = float(chunk[-1][4])
                v = sum(float(x[5]) for x in chunk)
                await self._append_candle(sym, "5m", [ts0, o, h, low, c, v])
            # 15m roll every 15 minutes
            if (last_start_ts % 900) == 840 and len(kl) >= 15:
                chunk = kl[-15:]
                ts0 = chunk[0][0]
                o = float(chunk[0][1])
                h = max(float(x[2]) for x in chunk)
                low = min(float(x[3]) for x in chunk)
                c = float(chunk[-1][4])
                v = sum(float(x[5]) for x in chunk)
                await self._append_candle(sym, "15m", [ts0, o, h, low, c, v])
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass


# Factory function for lazy instantiation (called after env setup)
def create_binance_ws_hydrator() -> BinanceWSHydrator:
    """Create WebSocket hydrator instance after environment is configured"""
    return BinanceWSHydrator()


# Global instance - lazy loaded
binance_ws_hydrator: BinanceWSHydrator | None = None
