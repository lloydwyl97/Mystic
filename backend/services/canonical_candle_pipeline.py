"""One canonical Binance.US candle pipeline.

Binance.US REST → normalize → SQLite feature_ohlcv → Redis klines → API/dashboard.

start_live_market_data.py is the only writer process. Closed candles are immutable
except for a verified same-bucket exchange replacement. Forming candles stay in Redis
only. Gaps are fetched from the exchange; they are never fabricated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from backend.config.canonical_candle_intervals import (
    CANONICAL_CANDLE_INTERVALS,
    CANONICAL_SYMBOLS,
    TELEMETRY_ONLY_NO_TRADE_AUTHORITY,
    align_open_ms,
    interval_ms,
    refresh_sec,
    stale_after_sec,
)
from backend.services.canonical_candle_store import (
    api_symbol,
    continuity_report,
    earliest_aligned_1m_open_ms,
    expected_open_ms_range,
    load_aligned_candles,
    merge_redis_rows,
    parse_redis_rows,
    redis_checkpoint_key,
    redis_forming_key,
    redis_integrity_key,
    redis_klines_key,
    refuse_research_table_read,
    row_from_binance_kline,
    upsert_completed_candles,
)

logger = logging.getLogger(__name__)

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"
WRITER_ROLE = "canonical_candle_pipeline"
FetchFn = Callable[..., Awaitable[list[list[Any]]]]


class CanonicalCandlePipeline:
    def __init__(self) -> None:
        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._lock = asyncio.Lock()
        self._errors: dict[str, str] = {}
        self._reconnects = 0
        self._last_backfill: dict[str, float] = {}
        self._source_ts: dict[str, int] = {}
        self._fetch_fn: FetchFn | None = None
        self._cached_start_ms: int | None = None
        self._hydrate_done = False

    def set_fetch_fn(self, fn: FetchFn | None) -> None:
        self._fetch_fn = fn

    def _stream_key(self, symbol: str, interval: str) -> str:
        return f"{api_symbol(symbol)}:{interval}"

    async def fetch_binance(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        if self._fetch_fn is not None:
            return await self._fetch_fn(
                symbol=api_symbol(symbol),
                interval=interval,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=limit,
            )
        params: dict[str, Any] = {
            "symbol": api_symbol(symbol),
            "interval": interval,
            "limit": min(1000, max(1, int(limit))),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        try:
            from backend.utils.binance_weight_limiter import BinanceWeightLimiter

            limiter = await BinanceWeightLimiter.create()
            await limiter.consume("/api/v3/klines", weight=1, wait=True, timeout=12.0)
        except Exception as exc:
            logger.debug("kline limiter unavailable: %s", exc)
        retries = 0
        last_exc: Exception | None = None
        while retries < 4:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.get(BINANCE_KLINES, params=params)
                    if response.status_code == 429:
                        retries += 1
                        await asyncio.sleep(min(8.0, 1.5 * retries))
                        continue
                    response.raise_for_status()
                    data = response.json()
                return data if isinstance(data, list) else []
            except (httpx.HTTPError, OSError) as exc:
                last_exc = exc
                retries += 1
                self._reconnects += 1
                await asyncio.sleep(min(8.0, 1.5 * retries))
        self._errors[self._stream_key(symbol, interval)] = str(last_exc or "fetch_failed")
        return []

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _completed_open_ms(self, interval: str, now_ms: int | None = None) -> int:
        now_ms = int(now_ms if now_ms is not None else self._now_ms())
        width = interval_ms(interval)
        forming = (now_ms // width) * width
        return forming - width

    async def _redis(self) -> Any:
        from backend.config.redis_config import get_shared_redis_async

        return await get_shared_redis_async()

    async def publish_redis(
        self,
        symbol: str,
        interval: str,
        completed: list[dict[str, Any]],
        forming: dict[str, Any] | None,
    ) -> None:
        redis = await self._redis()
        if redis is None:
            return
        key = redis_klines_key(symbol, interval)
        raw = await redis.get(key)
        merged = merge_redis_rows(parse_redis_rows(raw), completed, interval)
        pipe = redis.pipeline()
        ttl = max(stale_after_sec(interval) * 4, 900)
        pipe.set(key, json.dumps(merged), ex=ttl)
        if forming is not None:
            pipe.set(redis_forming_key(symbol, interval), json.dumps(forming), ex=ttl)
        if completed:
            pipe.set(redis_checkpoint_key(symbol, interval), str(completed[-1]["open_ms"]), ex=ttl * 4)
        await pipe.execute()

    async def ingest_klines(
        self,
        symbol: str,
        interval: str,
        klines: list[list[Any]],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        now_ms = self._now_ms()
        parsed = [row_from_binance_kline(k, symbol=symbol, interval=interval, now_ms=now_ms) for k in klines]
        rows = [r for r in parsed if r is not None]
        completed = [r for r in rows if r["completed"]]
        forming = next((r for r in reversed(rows) if r["forming"]), None)
        stats = {"inserted": 0, "updated": 0, "skipped_forming": 0}
        if persist and completed:
            async with self._lock:
                stats = upsert_completed_candles(symbol, interval, completed)
            await self.publish_redis(symbol, interval, completed, forming)
        elif forming is not None:
            await self.publish_redis(symbol, interval, [], forming)
        if rows:
            self._source_ts[self._stream_key(symbol, interval)] = int(rows[-1]["open_ms"])
        return {"completed": len(completed), "forming": 1 if forming else 0, **stats, "forming_candle": forming}

    async def backfill_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> dict[str, Any]:
        width = interval_ms(interval)
        cursor = int(start_ms)
        fetched = 0
        pages = 0
        while cursor <= end_ms:
            page_end = min(end_ms, cursor + width * 999)
            klines = await self.fetch_binance(symbol, interval, start_ms=cursor, end_ms=page_end, limit=1000)
            pages += 1
            if not klines:
                break
            await self.ingest_klines(symbol, interval, klines, persist=True)
            fetched += len(klines)
            last_open = int(klines[-1][0])
            nxt = last_open + width
            if nxt <= cursor:
                break
            cursor = nxt
            if len(klines) < 2:
                break
            await asyncio.sleep(0.15)
        self._last_backfill[self._stream_key(symbol, interval)] = time.time()
        return {"symbol": api_symbol(symbol), "interval": interval, "fetched": fetched, "pages": pages}

    async def repair_gaps(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> dict[str, Any]:
        report = continuity_report(symbol, interval, start_ms=start_ms, end_completed_ms=end_ms)
        missing = list(report.get("missing_timestamps") or [])
        if report["missing_count"] > 50:
            expected = expected_open_ms_range(start_ms, end_ms, interval)
            have = {int(r["open_ms"]) for r in load_aligned_candles(symbol, interval, start_ms=start_ms, end_ms=end_ms)}
            missing = [ts for ts in expected if ts not in have]
        repaired = 0
        if missing:
            # Fetch contiguous missing spans instead of one-bar requests.
            span_start = missing[0]
            prev = missing[0]
            width = interval_ms(interval)
            spans = []
            for ts in missing[1:]:
                if ts == prev + width:
                    prev = ts
                    continue
                spans.append((span_start, prev))
                span_start = ts
                prev = ts
            spans.append((span_start, prev))
            for a, b in spans:
                result = await self.backfill_range(symbol, interval, a, b + width - 1)
                repaired += int(result.get("fetched") or 0)
        after = continuity_report(symbol, interval, start_ms=start_ms, end_completed_ms=end_ms)
        return {"before": report, "after": after, "repaired_fetched": repaired}

    def canonical_start_ms(self) -> int:
        if self._cached_start_ms is not None:
            return int(self._cached_start_ms)
        try:
            earliest = earliest_aligned_1m_open_ms()
        except Exception as exc:
            logger.warning("canonical start lookup failed: %s", exc)
            earliest = None
        self._cached_start_ms = int(earliest) if earliest is not None else align_week_ago()
        return int(self._cached_start_ms)

    def _recent_window_start_ms(self, interval: str) -> int:
        width = interval_ms(interval)
        lookback = min(max(width * 120, 2 * 86_400_000), 7 * 86_400_000)
        recent = self._now_ms() - lookback
        return max(self.canonical_start_ms(), align_open_ms(recent, interval))

    async def startup_hydrate(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Live bars first, then 1m history, then exact HTF aggregates. Never blocks start()."""
        start_ms = self.canonical_start_ms()
        out: dict[str, Any] = {"start_ms": start_ms, "streams": []}
        symbols = symbols or list(CANONICAL_SYMBOLS)
        for symbol in symbols:
            for interval in CANONICAL_CANDLE_INTERVALS:
                try:
                    await self.refresh_live(symbol, interval)
                except Exception as exc:
                    logger.warning("hydrate live refresh failed %s %s: %s", symbol, interval, exc)
        logger.info("canonical candle live streams published; historical identity rebuild continuing")
        from backend.services.canonical_candle_store import delete_unaligned_persist_now_rows

        for symbol in symbols:
            for interval in CANONICAL_CANDLE_INTERVALS:
                try:
                    while True:
                        removed = delete_unaligned_persist_now_rows(symbol, interval)
                        if removed:
                            logger.info("purged persist-now rows symbol=%s interval=%s count=%s", symbol, interval, removed)
                        if removed < 50000:
                            break
                except Exception as exc:
                    logger.warning("persist-now purge failed %s %s: %s", symbol, interval, exc)
        for symbol in symbols:
            end_ms = self._completed_open_ms("1m")
            result = await self.backfill_range(symbol, "1m", start_ms, end_ms)
            out["streams"].append({"backfill": result})
            await asyncio.sleep(0.2)
        from backend.services.canonical_candle_store import aggregate_exact_from_1m, upsert_completed_candles

        for symbol in symbols:
            bars_1m = load_aligned_candles(symbol, "1m", start_ms=start_ms)
            for interval in CANONICAL_CANDLE_INTERVALS:
                if interval == "1m":
                    continue
                aggregated = aggregate_exact_from_1m(bars_1m, interval)
                if aggregated:
                    async with self._lock:
                        upsert_completed_candles(symbol, interval, aggregated)
                    await self.publish_redis(symbol, interval, aggregated[-200:], None)
                out["streams"].append({"symbol": symbol, "interval": interval, "aggregated": len(aggregated)})
                await asyncio.sleep(0.05)
        # Full-history integrity is operator-triggered. Running it inside
        # startup saturates SQLite and blocks the live API.
        self._hydrate_done = True
        return out

    async def full_history_integrity(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """One complete unique-bucket continuity pass from retained start to last closed candle."""
        symbols = symbols or list(CANONICAL_SYMBOLS)
        start_ms = self.canonical_start_ms()
        rows = []
        for symbol in symbols:
            for interval in CANONICAL_CANDLE_INTERVALS:
                end_ms = self._completed_open_ms(interval)
                await self.repair_gaps(symbol, interval, start_ms, end_ms)
                rows.append(await self.write_integrity(symbol, interval, start_ms=start_ms))
                await asyncio.sleep(0.05)
        return rows

    async def refresh_live(self, symbol: str, interval: str) -> dict[str, Any]:
        klines = await self.fetch_binance(symbol, interval, limit=3)
        ingested = await self.ingest_klines(symbol, interval, klines, persist=True)
        end_ms = self._completed_open_ms(interval)
        key = self._stream_key(symbol, interval)
        if self._hydrate_done and time.time() - self._last_backfill.get(key, 0) > refresh_sec(interval) * 8:
            await self.repair_gaps(symbol, interval, self._recent_window_start_ms(interval), end_ms)
        return ingested

    async def write_integrity(self, symbol: str, interval: str, *, start_ms: int | None = None) -> dict[str, Any]:
        start = int(start_ms if start_ms is not None else self.canonical_start_ms())
        end_ms = self._completed_open_ms(interval)
        report = continuity_report(symbol, interval, start_ms=start, end_completed_ms=end_ms)
        now = time.time()
        source_open = self._source_ts.get(self._stream_key(symbol, interval))
        newest = report.get("newest_completed_ms")
        db_age = None
        if newest:
            db_age = max(0.0, now - (int(newest) + interval_ms(interval)) / 1000.0)
        redis_age = None
        forming = None
        redis_count = 0
        try:
            redis = await self._redis()
            if redis is not None:
                raw = await redis.get(redis_klines_key(symbol, interval))
                rows = parse_redis_rows(raw)
                redis_count = len(rows)
                if rows:
                    last_close = int(rows[-1][0]) + interval_sec_safe(interval)
                    redis_age = max(0.0, now - last_close)
                form_raw = await redis.get(redis_forming_key(symbol, interval))
                if form_raw:
                    forming = json.loads(form_raw.decode() if isinstance(form_raw, (bytes, bytearray)) else form_raw)
        except Exception as exc:
            logger.debug("integrity redis read failed %s %s: %s", symbol, interval, exc)
        status = {
            **report,
            "latest_source_timestamp": source_open,
            "latest_completed_timestamp": newest,
            "expected_next_timestamp": (int(newest) + interval_ms(interval)) if newest else None,
            "current_forming_timestamp": forming.get("open_ms") if isinstance(forming, dict) else None,
            "forming": forming,
            "source_age_sec": (now - source_open / 1000.0) if source_open else None,
            "database_age_sec": db_age,
            "redis_age_sec": redis_age,
            "redis_count": redis_count,
            "api_count": report["row_count"],
            "last_successful_backfill": self._last_backfill.get(self._stream_key(symbol, interval)),
            "last_collector_error": self._errors.get(self._stream_key(symbol, interval)),
            "reconnect_count": self._reconnects,
            "stale": bool((db_age or 0) > stale_after_sec(interval) or (redis_age or 0) > stale_after_sec(interval)),
            "writer": WRITER_ROLE,
            "4h_authority": TELEMETRY_ONLY_NO_TRADE_AUTHORITY if interval == "4h" else "n/a",
            "research_table": refuse_research_table_read(),
        }
        try:
            redis = await self._redis()
            if redis is not None:
                await redis.set(redis_integrity_key(symbol, interval), json.dumps(status, default=str), ex=3600)
        except Exception as exc:
            logger.debug("integrity write failed %s %s: %s", symbol, interval, exc)
        return status

    async def status_matrix(self, *, full: bool = False) -> dict[str, Any]:
        """Live status uses the recent window. Full 44-day scans are operator-only."""
        rows = []
        for symbol in CANONICAL_SYMBOLS:
            for interval in CANONICAL_CANDLE_INTERVALS:
                start_ms = self.canonical_start_ms() if full else self._recent_window_start_ms(interval)
                rows.append(await self.write_integrity(symbol, interval, start_ms=start_ms))
        return {
            "writer": WRITER_ROLE,
            "symbols": list(CANONICAL_SYMBOLS),
            "intervals": list(CANONICAL_CANDLE_INTERVALS),
            "full_history": bool(full),
            "streams": rows,
        }

    async def _refresh_loop(self) -> None:
        idx = 0
        pairs = [(s, i) for s in CANONICAL_SYMBOLS for i in CANONICAL_CANDLE_INTERVALS]
        while self._running:
            symbol, interval = pairs[idx % len(pairs)]
            idx += 1
            try:
                await self.refresh_live(symbol, interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._errors[self._stream_key(symbol, interval)] = str(exc)
                logger.warning("canonical refresh failed %s %s: %s", symbol, interval, exc)
            await asyncio.sleep(max(1.0, refresh_sec(interval) / len(CANONICAL_SYMBOLS)))

    async def _integrity_loop(self) -> None:
        while self._running:
            if not self._hydrate_done:
                await asyncio.sleep(5)
                continue
            try:
                for symbol in CANONICAL_SYMBOLS:
                    for interval in CANONICAL_CANDLE_INTERVALS:
                        end_ms = self._completed_open_ms(interval)
                        start_ms = self._recent_window_start_ms(interval)
                        await self.repair_gaps(symbol, interval, start_ms, end_ms)
                        await self.write_integrity(symbol, interval, start_ms=start_ms)
                        await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("canonical integrity loop failed: %s", exc)
            await asyncio.sleep(120)

    async def _hydrate_background(self) -> None:
        try:
            await self.startup_hydrate()
            self._hydrate_done = True
            logger.info("canonical candle hydrate complete")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("canonical startup hydrate failed: %s", exc)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._hydrate_background(), name="canonical_candle:hydrate"),
            asyncio.create_task(self._refresh_loop(), name="canonical_candle:refresh"),
            asyncio.create_task(self._integrity_loop(), name="canonical_candle:integrity"),
        ]
        logger.info("canonical candle pipeline started writer=%s", WRITER_ROLE)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()


def align_week_ago() -> int:
    return int((time.time() - 44 * 86400) * 1000)


def interval_sec_safe(interval: str) -> int:
    return interval_ms(interval) // 1000


canonical_candle_pipeline = CanonicalCandlePipeline()


async def get_canonical_candles(
    symbol: str,
    interval: str,
    *,
    limit: int = 300,
    include_forming: bool = True,
) -> dict[str, Any]:
    if interval not in CANONICAL_CANDLE_INTERVALS:
        return {
            "success": False,
            "error": f"unsupported_interval:{interval}",
            "supported": list(CANONICAL_CANDLE_INTERVALS),
            "candles": [],
        }
    completed = load_aligned_candles(symbol, interval, limit=limit)
    forming = None
    redis_count = 0
    try:
        redis = await canonical_candle_pipeline._redis()
        if redis is not None:
            raw = await redis.get(redis_klines_key(symbol, interval))
            redis_rows = parse_redis_rows(raw)
            redis_count = len(redis_rows)
            if include_forming:
                form_raw = await redis.get(redis_forming_key(symbol, interval))
                if form_raw:
                    forming = json.loads(form_raw.decode() if isinstance(form_raw, (bytes, bytearray)) else form_raw)
    except Exception as exc:
        logger.debug("canonical get redis failed: %s", exc)
    if not completed and redis_count == 0:
        return {
            "success": False,
            "error": "no_canonical_data",
            "symbol": api_symbol(symbol),
            "interval": interval,
            "candles": [],
            "forming": None,
            "source": "canonical",
        }
    candles = list(completed)
    if include_forming and isinstance(forming, dict):
        candles = [c for c in candles if int(c["open_ms"]) != int(forming.get("open_ms") or -1)]
        candles.append(forming)
    freshness = None
    try:
        redis = await canonical_candle_pipeline._redis()
        if redis is not None:
            raw_int = await redis.get(redis_integrity_key(symbol, interval))
            if raw_int:
                freshness = json.loads(raw_int.decode() if isinstance(raw_int, (bytes, bytearray)) else raw_int)
    except Exception as exc:
        logger.debug("canonical freshness cache read failed: %s", exc)
    return {
        "success": True,
        "symbol": api_symbol(symbol),
        "interval": interval,
        "candles": candles,
        "completed_count": len(completed),
        "forming": forming,
        "redis_count": redis_count,
        "source": "canonical",
        "freshness": freshness,
        "4h_authority": TELEMETRY_ONLY_NO_TRADE_AUTHORITY if interval == "4h" else "n/a",
    }


__all__ = [
    "WRITER_ROLE",
    "CanonicalCandlePipeline",
    "canonical_candle_pipeline",
    "get_canonical_candles",
]
