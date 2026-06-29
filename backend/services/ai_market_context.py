"""
AI Market Context — canonical multi-timeframe + market-context AI inputs.

This is the REAL canonical AI input layer for everything that is not the raw
124-feature 1m vector. It runs as its own loop, fetches OHLCV for every
``DAY_ACTIVE_TIMEFRAMES`` interval (see ``backend/config/day_active_timeframes.py``)
for every traded symbol + BTC + ETH, fetches order book depth, and publishes:

    1. Per-symbol AI context to Redis: ai_context:{SYMBOL}
       -- MTF trend / slope / RSI / EMA-alignment per ``DAY_ACTIVE_TIMEFRAMES`` TF
       -- ``month_from_daily`` in ``mtf_json`` (vec4 from stacked native 1d candles only; no synthetic month bars)
       -- 24h change %, 24h notional volume, relative volume vs 7d
       -- Liquidity tier (1..3) from notional volume
       -- Spread % and L10 depth imbalance
       -- Relative strength vs BTC and ETH (24h)
       -- A coarse market regime label ("trending_up"/"trending_down"/"chop")
       -- A combined ctx_multiplier in [1 - CTX_TOTAL_CAP, 1 + CTX_TOTAL_CAP]
          that ai_signal_generator.py applies to winner_probability_raw.

    2. A snapshot row to SQLite ai_context_snapshots so we can join historical
       trades to the context that existed when each entry was decided.

The published context fields are CANONICAL inputs to the AI decision contract.
They are not telemetry, not "future", not optional — ai_signal_generator.py
reads them every loop and applies the multiplier.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import redis.asyncio as redis_async

from backend.config.day_active_timeframes import (
    DAY_ACTIVE_TIMEFRAMES,
)
from backend.config.redis_config import get_shared_redis_async
from backend.config.trading_universe import TRADING_SYMBOLS, get_trading_symbols
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_decision_contract import (
    AI_CONTEXT_LOOP_SEC,
    AI_SENTIMENT_LOOP_SEC,
    CTX_DEPTH_WEIGHT,
    CTX_MTF_ALIGN_WEIGHT,
    CTX_REGIME_WEIGHT,
    CTX_RS_WEIGHT,
    CTX_TOTAL_CAP,
    MARKET_CONTEXT_FIELDS,
    REDIS_KEY_AI_CONTEXT,
    REDIS_KEY_AI_SENTIMENT,
    REDIS_TTL_AI_CONTEXT_SEC,
)
from backend.services.day_active_market_bundle import (
    apply_day_bundle_stagger,
    async_fetch_day_active_ohlcv_bundle,
    async_read_cached_day_active_bundle,
    month_context_four_from_daily,
)
from backend.services.live_market_data import live_market_data_service
from backend.services.market_regime import regime_score

logger = logging.getLogger(__name__)


def _to_ccxt(symbol: str) -> str:
    """Convert 'BTCUSDT' -> 'BTC/USDT' (the canonical CCXT format used elsewhere)."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT") and not symbol.endswith("/USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def _safe_close_array(rows: list[list[float]] | None) -> np.ndarray:
    if not rows:
        return np.array([], dtype=np.float64)
    return np.array([float(r[4]) for r in rows], dtype=np.float64)


def _safe_volume_array(rows: list[list[float]] | None) -> np.ndarray:
    if not rows:
        return np.array([], dtype=np.float64)
    return np.array([float(r[5]) for r in rows], dtype=np.float64)


def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1]) if len(values) else 0.0
    k = 2.0 / (period + 1.0)
    ema = float(values[0])
    for v in values[1:]:
        ema = float(v) * k + ema * (1.0 - k)
    return ema


def _rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    diff = np.diff(values)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _slope_pct(values: np.ndarray, lookback: int = 20) -> float:
    if len(values) < lookback or values[-lookback] == 0:
        return 0.0
    return float((values[-1] - values[-lookback]) / values[-lookback])


def _atr_pct(rows: list[list[float]] | None, period: int = 14) -> float:
    if not rows or len(rows) < period + 1:
        return 0.0
    highs = np.array([float(r[2]) for r in rows], dtype=np.float64)
    lows = np.array([float(r[3]) for r in rows], dtype=np.float64)
    closes = np.array([float(r[4]) for r in rows], dtype=np.float64)
    tr = np.maximum.reduce(
        [
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ]
    )
    atr = float(np.mean(tr[-period:]))
    last_close = float(closes[-1]) or 1.0
    return atr / last_close


def _ema_alignment_score(values: np.ndarray) -> float:
    """Return [0..1]: 1 if EMA9 > EMA21 > EMA50 (uptrend), 0 if reversed (downtrend), 0.5 chop."""
    if len(values) < 50:
        return 0.5
    e9 = _ema(values, 9)
    e21 = _ema(values, 21)
    e50 = _ema(values, 50)
    if e9 > e21 > e50:
        return 1.0
    if e9 < e21 < e50:
        return 0.0
    return 0.5


def _summarize_tf(rows: list[list[float]] | None) -> dict[str, float]:
    closes = _safe_close_array(rows)
    if len(closes) == 0:
        return {"trend": 0.5, "slope": 0.0, "rsi": 50.0, "atr_pct": 0.0, "ema_align": 0.5, "bars": 0}
    return {
        "trend": _ema_alignment_score(closes),
        "slope": _slope_pct(closes, lookback=min(20, len(closes) - 1) or 1),
        "rsi": _rsi(closes, 14),
        "atr_pct": _atr_pct(rows, 14),
        "ema_align": _ema_alignment_score(closes),
        "bars": len(closes),
    }


def _neutral_mtf_pack() -> dict[str, Any]:
    snap = _summarize_tf(None)
    out: dict[str, Any] = {tf: dict(snap) for tf in DAY_ACTIVE_TIMEFRAMES}
    out["month_from_daily"] = {"vec4": None, "ok": False, "reason": "neutral_pack"}
    return out


def _liquidity_tier(volume_usd: float) -> int:
    if volume_usd >= 5e8:
        return 3
    if volume_usd >= 5e7:
        return 2
    if volume_usd > 0:
        return 1
    return 0


def _market_regime_from_btc(btc_mtf: dict[str, dict[str, float]]) -> str:
    h1 = btc_mtf.get("1h", {})
    h4 = btc_mtf.get("4h", {})
    score = 0
    for snap in (h1, h4):
        t = snap.get("trend", 0.5)
        if t > 0.66:
            score += 1
        elif t < 0.33:
            score -= 1
    if score >= 1:
        return "trending_up"
    if score <= -1:
        return "trending_down"
    return "chop"


def _ctx_multiplier(
    *,
    own_mtf: dict[str, dict[str, float]],
    rs_btc: float,
    rs_eth: float,
    depth_imbalance: float,
    market_regime: str,
) -> tuple[float, dict[str, float]]:
    """
    Combine MTF + RS + depth + regime into one multiplier on winner_probability.

    Returns (multiplier, applied_components_for_audit).
    The multiplier is symmetric around 1.0 and clamped to +/- CTX_TOTAL_CAP.
    """
    align_scores: list[float] = []
    for tf in DAY_ACTIVE_TIMEFRAMES:
        snap = own_mtf.get(tf)
        if isinstance(snap, dict) and snap.get("bars", 0) > 5:
            align_scores.append(float(snap.get("ema_align", 0.5)))
    if align_scores:
        mtf_align = float(np.mean(align_scores)) * 2.0 - 1.0
    else:
        mtf_align = 0.0
    mtf_term = mtf_align * CTX_MTF_ALIGN_WEIGHT

    rs_avg = max(-1.0, min(1.0, (rs_btc + rs_eth) / 2.0))
    rs_term = rs_avg * CTX_RS_WEIGHT

    depth_term = max(-1.0, min(1.0, depth_imbalance)) * CTX_DEPTH_WEIGHT

    if market_regime == "trending_up":
        regime_signed = 1.0
    elif market_regime == "trending_down":
        regime_signed = -1.0
    else:
        regime_signed = 0.0
    regime_term = regime_signed * CTX_REGIME_WEIGHT

    total = mtf_term + rs_term + depth_term + regime_term
    total = max(-CTX_TOTAL_CAP, min(CTX_TOTAL_CAP, total))
    multiplier = 1.0 + total

    return multiplier, {
        "mtf_align_signed": float(mtf_align),
        "mtf_term": float(mtf_term),
        "rs_avg": float(rs_avg),
        "rs_term": float(rs_term),
        "depth_term": float(depth_term),
        "regime_term": float(regime_term),
        "total_signed": float(total),
    }


class AIMarketContextService:
    """Build & publish canonical multi-timeframe + market context for every traded symbol."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols: list[str] = list(symbols or get_trading_symbols())
        self.is_running: bool = False
        self.redis: redis_async.Redis | None = None
        self._task: asyncio.Task | None = None
        self._loop_count: int = 0
        self._fairness_rotation: int = 0
        # Cached canonical sentiment (alternative.me Fear/Greed Index, [-1, +1]).
        # Refreshed every AI_SENTIMENT_LOOP_SEC because the FNG API only updates daily.
        self._sentiment_value: float = 0.0
        self._sentiment_last_fetch_ts: float = 0.0
        self._sentiment_collector = None
        self._orderbook_refresher = None
        ensure_ai_canonical_tables()

    async def _refresh_sentiment_if_due(self) -> None:
        now = time.time()
        if (now - self._sentiment_last_fetch_ts) < AI_SENTIMENT_LOOP_SEC and self._sentiment_last_fetch_ts > 0:
            return
        try:
            s = await regime_score()
            self._sentiment_value = float(max(-1.0, min(1.0, s)))
            self._sentiment_last_fetch_ts = now
            if self.redis is not None:
                with contextlib.suppress(Exception):
                    await self.redis.set(REDIS_KEY_AI_SENTIMENT, str(self._sentiment_value), ex=AI_SENTIMENT_LOOP_SEC * 3)
        except Exception as e:  # pragma: no cover
            logger.debug("AI_CONTEXT sentiment refresh failed: %s", e)

    async def start(self) -> None:
        if self.is_running:
            return
        self.redis = await get_shared_redis_async()
        self.is_running = True
        try:
            from backend.services.ai_active_sentiment_collector import get_active_sentiment_collector

            self._sentiment_collector = get_active_sentiment_collector()
            await self._sentiment_collector.start()
        except Exception as exc:
            logger.warning("AI_CONTEXT sentiment collector start failed: %s", exc)
        try:
            from backend.services.orderbook_redis_refresher import get_orderbook_redis_refresher

            self._orderbook_refresher = get_orderbook_redis_refresher()
            await self._orderbook_refresher.start()
        except Exception as exc:
            logger.warning("AI_CONTEXT orderbook refresher start failed: %s", exc)
        self._task = asyncio.create_task(self._loop(), name="ai_market_context:loop")
        logger.info("AI_CONTEXT: started for %d symbols (cadence=%ss)", len(self.symbols), AI_CONTEXT_LOOP_SEC)

    async def stop(self) -> None:
        self.is_running = False
        if self._sentiment_collector is not None:
            with contextlib.suppress(Exception):
                await self._sentiment_collector.stop()
        if self._orderbook_refresher is not None:
            with contextlib.suppress(Exception):
                await self._orderbook_refresher.stop()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def warm_publish_now(self) -> None:
        """One immediate `_tick()` (boot warm-up so `ai_context:*` exists before ML emit)."""
        try:
            await self._tick()
        except Exception as e:  # pragma: no cover
            logger.warning("AI_CONTEXT warm_publish_now failed (loop will retry): %s", e)

    async def _loop(self) -> None:
        await apply_day_bundle_stagger("ai_context")
        while self.is_running:
            t0 = time.time()
            try:
                await self._tick()
            except Exception as e:  # pragma: no cover
                logger.exception("AI_CONTEXT loop error: %s", e)
            elapsed = time.time() - t0
            sleep_s = max(1.0, AI_CONTEXT_LOOP_SEC - elapsed)
            await asyncio.sleep(sleep_s)

    async def _fetch_mtf_for_symbol(self, symbol: str) -> dict[str, Any]:
        """One entry per ``DAY_ACTIVE_TIMEFRAMES`` TF + ``month_from_daily`` from native 1d only."""
        ccxt_sym = _to_ccxt(symbol)
        out: dict[str, Any] = {}
        ohlcv_1d: list[list[float]] | None = None
        bundle: dict[str, list[list]] = {}
        if live_market_data_service:
            with contextlib.suppress(Exception):
                bundle = await async_fetch_day_active_ohlcv_bundle(live_market_data_service, ccxt_sym)
        for tf in DAY_ACTIVE_TIMEFRAMES:
            rows = bundle.get(tf) if isinstance(bundle.get(tf), list) else None
            out[tf] = _summarize_tf(rows)
            if tf == "1d" and isinstance(rows, list):
                ohlcv_1d = rows

        mv, merr = month_context_four_from_daily(ohlcv_1d if isinstance(ohlcv_1d, list) else [])
        if mv is not None:
            out["month_from_daily"] = {"vec4": mv, "ok": True, "source": "native_1d_candles_only"}
        else:
            out["month_from_daily"] = {"vec4": None, "ok": False, "reason": str(merr or "month_context_unavailable")}
        return out

    async def _approx_quote_volume_24h_from_1m(self, ccxt_sym: str) -> float:
        """Sum close*volume over last ~24h of 1m bars (USDT-quote notional proxy)."""
        rows = None
        with contextlib.suppress(Exception):
            cached = await async_read_cached_day_active_bundle(ccxt_sym)
            if cached and isinstance(cached.get("1m"), list) and len(cached["1m"]) >= 30:
                rows = cached["1m"]
        if rows is None or len(rows) < 30:
            return 0.0
        total = 0.0
        for r in rows[-1440:]:
            total += float(r[4]) * float(r[5])
        return max(0.0, total)

    async def _fetch_24h(self, symbol: str) -> dict[str, float]:
        """Get 24h ticker data: change %, quote volume (USD notional when available)."""
        ccxt_sym = _to_ccxt(symbol)
        with contextlib.suppress(Exception):
            t = await live_market_data_service.get_ticker(ccxt_sym)
            if t:
                pct = float(t.get("percentage") or t.get("change_24h") or 0.0)
                # Normalized ticker uses volume_24h (quote preferred); raw ccxt keys differ.
                qvol = float(
                    t.get("volume_24h") or t.get("quoteVolume") or t.get("baseVolume") or 0.0,
                )
                if qvol < 1.0:
                    fb = await self._approx_quote_volume_24h_from_1m(ccxt_sym)
                    if fb > 0.0:
                        qvol = fb
                return {"change_24h_pct": pct / 100.0, "volume_24h_usd": qvol}
        fb2 = await self._approx_quote_volume_24h_from_1m(ccxt_sym)
        if fb2 > 0.0:
            return {"change_24h_pct": 0.0, "volume_24h_usd": fb2}
        return {"change_24h_pct": 0.0, "volume_24h_usd": 0.0}

    async def _fetch_depth(self, symbol: str) -> tuple[float, float]:
        """Return (spread_pct, depth_imbalance) where imbalance in [-1, +1] (+ = bid-heavy)."""
        ccxt_sym = _to_ccxt(symbol)
        with contextlib.suppress(Exception):
            ob = await live_market_data_service.get_order_book(ccxt_sym, limit=10)
            if ob and isinstance(ob, dict):
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                if bids and asks:
                    bb = float(bids[0][0])
                    ba = float(asks[0][0])
                    mid = (bb + ba) / 2.0
                    spread_pct = (ba - bb) / mid if mid > 0 else 0.0
                    bid_qty = sum(float(b[1]) for b in bids[:10])
                    ask_qty = sum(float(a[1]) for a in asks[:10])
                    denom = bid_qty + ask_qty
                    imb = (bid_qty - ask_qty) / denom if denom > 0 else 0.0
                    if self.redis is not None and spread_pct > 0:
                        with contextlib.suppress(Exception):
                            from backend.services.order_book_service import (
                                order_book_features_from_bids_asks,
                                write_orderbook_redis_async,
                            )
                            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                            feats = order_book_features_from_bids_asks(bids, asks, depth_levels=10)
                            await write_orderbook_redis_async(
                                CanonicalSymbolFormatter.to_base(symbol),
                                feats,
                                self.redis,
                                source="rest_depth",
                            )
                    return spread_pct, max(-1.0, min(1.0, imb))
        return 0.0, 0.0

    async def _batch_read_context_ts_utc(self) -> dict[str, str | None]:
        """Read last-published ts_utc from Redis for fairness ordering (pre-pass ages)."""
        out: dict[str, str | None] = dict.fromkeys(self.symbols)
        if self.redis is None:
            return out
        try:
            pipe = self.redis.pipeline()
            for sym in self.symbols:
                pipe.hget(REDIS_KEY_AI_CONTEXT.format(symbol=sym), "ts_utc")
            vals = await pipe.execute()
            for sym, raw in zip(self.symbols, vals, strict=False):
                if raw is None or str(raw).strip() == "":
                    out[sym] = None
                else:
                    out[sym] = str(raw)
        except Exception as e:  # pragma: no cover
            logger.debug("AI_CONTEXT batch ts read failed: %s", e)
        return out

    def _context_age_sec_from_ts(self, ts_raw: str | None, now: datetime) -> float:
        if not ts_raw or str(ts_raw).strip() == "":
            return float("inf")
        try:
            tnorm = str(ts_raw).replace("Z", "+00:00")
            t = datetime.fromisoformat(tnorm)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return max(0.0, (now - t.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return float("inf")

    def _fair_symbol_order(self, ages_sec: dict[str, float]) -> list[str]:
        """
        Stalest-first so tail symbols are not always refreshed last on long passes.
        Tie-break with a rotating base order so BTC is not always refreshed first when ages tie.
        """
        n = len(self.symbols)
        if n == 0:
            return []
        rot = self._fairness_rotation % n
        rotated = self.symbols[rot:] + self.symbols[:rot]
        idx = {s: i for i, s in enumerate(rotated)}
        return sorted(rotated, key=lambda s: (-ages_sec.get(s, float("inf")), idx[s]))

    async def _tick(self) -> None:
        self._loop_count += 1
        pass_t0 = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        # Refresh canonical sentiment input (cheap, cached).
        await self._refresh_sentiment_if_due()

        pre_ts = await self._batch_read_context_ts_utc()
        ages_pre = {s: self._context_age_sec_from_ts(pre_ts.get(s), now_utc) for s in self.symbols}
        ordered_symbols = self._fair_symbol_order(ages_pre)
        self._fairness_rotation = (self._fairness_rotation + 1) % max(1, len(self.symbols))

        # Fetch BTC + ETH context first (used for relative strength + dominance proxy)
        btc_mtf, eth_mtf = await asyncio.gather(
            self._fetch_mtf_for_symbol("BTCUSDT"),
            self._fetch_mtf_for_symbol("ETHUSDT"),
        )
        btc_24h, eth_24h = await asyncio.gather(
            self._fetch_24h("BTCUSDT"),
            self._fetch_24h("ETHUSDT"),
        )
        market_regime = _market_regime_from_btc(btc_mtf)

        # Universe 24h: parallel for remaining symbols (BTC/ETH already fetched)
        all_24h: dict[str, dict[str, float]] = {"BTCUSDT": btc_24h, "ETHUSDT": eth_24h}
        rest_24 = [sym for sym in self.symbols if sym not in all_24h]
        if rest_24:
            chunks = await asyncio.gather(*[self._fetch_24h(s) for s in rest_24], return_exceptions=True)
            for sym, result in zip(rest_24, chunks, strict=False):
                if isinstance(result, dict):
                    all_24h[sym] = result
                else:
                    logger.debug("AI_CONTEXT _fetch_24h %s failed: %s", sym, result)
                    all_24h[sym] = {"change_24h_pct": 0.0, "volume_24h_usd": 0.0}
        total_vol = sum(v.get("volume_24h_usd", 0.0) for v in all_24h.values()) or 1.0
        btc_dom_proxy = all_24h["BTCUSDT"].get("volume_24h_usd", 0.0) / total_vol

        # Compute per-symbol relative volume baseline using BTC/ETH 24h volumes as anchor
        # (relative_volume here = symbol_vol / median_universe_vol)
        vols = sorted(v.get("volume_24h_usd", 0.0) for v in all_24h.values())
        median_vol = vols[len(vols) // 2] if vols else 1.0
        median_vol = median_vol or 1.0

        # Prefetch MTF + depth for all symbols in parallel (bounded) so one pass does not
        # spend ~O(n) network time sequentially; stalest-first ordering then applies to
        # fast Redis publishes, collapsing intra-pass age spread vs sequential MTF work.
        prefetch_t0 = time.perf_counter()
        mtf_by_sym: dict[str, Any] = {
            "BTCUSDT": btc_mtf,
            "ETHUSDT": eth_mtf,
        }
        rest_for_mtf = [s for s in self.symbols if s not in mtf_by_sym]
        conc_mtf = max(1, min(10, int(os.getenv("AI_CONTEXT_PREFETCH_MTF_CONCURRENCY", "4"))))
        sem_mtf = asyncio.Semaphore(conc_mtf)
        mtf_timeout = float(os.getenv("AI_CONTEXT_PREFETCH_MTF_TIMEOUT_SEC", "180"))

        async def _prefetch_mtf(sym: str) -> None:
            async with sem_mtf:
                try:
                    mtf_by_sym[sym] = await asyncio.wait_for(
                        self._fetch_mtf_for_symbol(sym),
                        timeout=mtf_timeout,
                    )
                except TimeoutError:
                    logger.warning("AI_CONTEXT mtf prefetch %s timed out after %.0fs", sym, mtf_timeout)
                    mtf_by_sym[sym] = _neutral_mtf_pack()
                except Exception as e:  # pragma: no cover
                    logger.warning("AI_CONTEXT mtf prefetch %s failed: %s", sym, e)
                    mtf_by_sym[sym] = _neutral_mtf_pack()

        await asyncio.gather(*(_prefetch_mtf(s) for s in rest_for_mtf))

        conc_depth = max(1, min(20, int(os.getenv("AI_CONTEXT_PREFETCH_DEPTH_CONCURRENCY", "4"))))
        sem_depth = asyncio.Semaphore(conc_depth)
        depth_by_sym: dict[str, tuple[float, float]] = {}
        depth_timeout = float(os.getenv("AI_CONTEXT_PREFETCH_DEPTH_TIMEOUT_SEC", "35"))

        async def _prefetch_depth(sym: str) -> None:
            async with sem_depth:
                try:
                    depth_by_sym[sym] = await asyncio.wait_for(
                        self._fetch_depth(sym),
                        timeout=depth_timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        "AI_CONTEXT depth prefetch %s timed out after %.0fs — using zeros",
                        sym,
                        depth_timeout,
                    )
                    depth_by_sym[sym] = (0.0, 0.0)
                except Exception as e:  # pragma: no cover
                    logger.debug("AI_CONTEXT depth prefetch %s failed: %s", sym, e)
                    depth_by_sym[sym] = (0.0, 0.0)

        await asyncio.gather(*(_prefetch_depth(s) for s in self.symbols))
        prefetch_elapsed = time.perf_counter() - prefetch_t0

        published = 0
        per_sym_dt: list[tuple[str, float, float]] = []  # symbol, secs, age_pre

        for symbol in ordered_symbols:
            sym_t0 = time.perf_counter()
            try:
                own_mtf = mtf_by_sym[symbol]
                t24 = all_24h.get(symbol, {"change_24h_pct": 0.0, "volume_24h_usd": 0.0})
                spread_pct, depth_imb = depth_by_sym[symbol]

                rs_btc = t24["change_24h_pct"] - btc_24h["change_24h_pct"]
                rs_eth = t24["change_24h_pct"] - eth_24h["change_24h_pct"]
                rs_btc = max(-1.0, min(1.0, rs_btc * 5.0))  # scale: 20% relative move = full strength
                rs_eth = max(-1.0, min(1.0, rs_eth * 5.0))

                rel_vol = t24["volume_24h_usd"] / median_vol if median_vol > 0 else 0.0
                liq_tier = _liquidity_tier(t24["volume_24h_usd"])

                multiplier, audit = _ctx_multiplier(
                    own_mtf=own_mtf,
                    rs_btc=rs_btc,
                    rs_eth=rs_eth,
                    depth_imbalance=depth_imb,
                    market_regime=market_regime,
                )

                ts_utc = datetime.now(timezone.utc).isoformat()
                payload: dict[str, Any] = {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "ctx_change_24h_pct": float(t24["change_24h_pct"]),
                    "ctx_volume_24h_usd": float(t24["volume_24h_usd"]),
                    "ctx_relative_volume": float(rel_vol),
                    "ctx_liquidity_tier": int(liq_tier),
                    "ctx_spread_pct": float(spread_pct),
                    "ctx_depth_imbalance": float(depth_imb),
                    "ctx_rs_btc": float(rs_btc),
                    "ctx_rs_eth": float(rs_eth),
                    "ctx_btc_dominance_proxy": float(btc_dom_proxy),
                    "ctx_market_regime": str(market_regime),
                    "ctx_sentiment_fear_greed": float(self._sentiment_value),
                    "ctx_multiplier": float(multiplier),
                    "mtf_json": json.dumps(own_mtf, separators=(",", ":")),
                    "ctx_audit_json": json.dumps(audit, separators=(",", ":")),
                }

                if self.redis is not None:
                    key = REDIS_KEY_AI_CONTEXT.format(symbol=symbol)
                    async with self.redis.pipeline(transaction=True) as pipe:
                        pipe.hmset(key, {k: str(v) for k, v in payload.items()})
                        pipe.expire(key, REDIS_TTL_AI_CONTEXT_SEC)
                        await pipe.execute()

                self._persist_snapshot(payload)
                published += 1
                sym_elapsed = time.perf_counter() - sym_t0
                per_sym_dt.append((symbol, sym_elapsed, ages_pre.get(symbol, float("inf"))))
            except Exception as e:
                logger.warning("AI_CONTEXT %s failed: %s", symbol, e)
                continue

        pass_elapsed = time.perf_counter() - pass_t0
        if pass_elapsed > float(AI_CONTEXT_LOOP_SEC):
            logger.warning(
                "AI_CONTEXT_PASS_EXCEEDED_CADENCE elapsed=%.2fs cadence=%ss — next pass may start after short sleep only",
                pass_elapsed,
                AI_CONTEXT_LOOP_SEC,
            )
        if per_sym_dt:
            slow = max(per_sym_dt, key=lambda x: x[1])
            stale_tail = sorted(per_sym_dt, key=lambda x: -x[2])[:3]
            stale_str = ",".join(f"{s}~{a:.0f}s" for s, _, a in stale_tail)
            logger.info(
                "AI_CONTEXT_PASS loop=%d duration=%.2fs prefetch=%.2fs published=%d/%d regime=%s publish_slowest=%s=%.2fs pre_pass_stalest=%s order_head=%s..%s",
                self._loop_count,
                pass_elapsed,
                prefetch_elapsed,
                published,
                len(self.symbols),
                market_regime,
                slow[0],
                slow[1],
                stale_str,
                ordered_symbols[0] if ordered_symbols else "-",
                ordered_symbols[-1] if ordered_symbols else "-",
            )
        elif self._loop_count % 5 == 1:
            logger.info(
                "AI_CONTEXT loop=%s published=%d/%d regime=%s btc_dom~%.2f",
                self._loop_count,
                published,
                len(self.symbols),
                market_regime,
                btc_dom_proxy,
            )

    def _persist_snapshot(self, payload: dict[str, Any]) -> None:
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute(
                    """
                    INSERT INTO ai_context_snapshots (
                        symbol, ts_utc, change_24h_pct, volume_24h_usd, relative_volume,
                        liquidity_tier, spread_pct, depth_imbalance, rs_btc, rs_eth,
                        btc_dominance_proxy, market_regime, sentiment_fear_greed,
                        mtf_json, ctx_multiplier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["symbol"],
                        payload["ts_utc"],
                        payload["ctx_change_24h_pct"],
                        payload["ctx_volume_24h_usd"],
                        payload["ctx_relative_volume"],
                        payload["ctx_liquidity_tier"],
                        payload["ctx_spread_pct"],
                        payload["ctx_depth_imbalance"],
                        payload["ctx_rs_btc"],
                        payload["ctx_rs_eth"],
                        payload["ctx_btc_dominance_proxy"],
                        payload["ctx_market_regime"],
                        payload["ctx_sentiment_fear_greed"],
                        payload["mtf_json"],
                        payload["ctx_multiplier"],
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.debug("ai_context_snapshots insert failed: %s", e)


_singleton: AIMarketContextService | None = None


def get_market_context_service() -> AIMarketContextService:
    global _singleton
    if _singleton is None:
        _singleton = AIMarketContextService()
    return _singleton


_ctx_overlay_lock = asyncio.Lock()
_ctx_univ_ts: float = 0.0
_ctx_univ_all24h: dict[str, dict[str, float]] | None = None
_CTX_OVERLAY_TTL_SEC = 25.0


async def hydrate_ai_context_payload(symbol_pair: str, payload: dict[str, str] | None) -> dict[str, str]:
    """
    Overlay live 24h volume / dominance / relative-liquidity fields when Redis values are stale.

    The context loop normally publishes these; if quote volume from the ticker is zero or the
    daemon is on older code, ``ctx_volume_24h_usd`` / ``ctx_btc_dominance_proxy`` collapse and
    v2 dims 131-133 and 141 go dead. Uses the same REST + 1m-notional fallback as
    ``AIMarketContextService._fetch_24h`` (25s cached universe rollup).
    """
    out: dict[str, str] = dict(payload or {})
    sym = symbol_pair.strip().upper()
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(out.get(key) or default)
        except (TypeError, ValueError):
            return default

    vol_r = _f("ctx_volume_24h_usd")
    rel_r = _f("ctx_relative_volume")
    dom_r = _f("ctx_btc_dominance_proxy")
    if vol_r >= 1.0 and rel_r >= 1e-6 and dom_r >= 1e-6:
        return out

    svc = get_market_context_service()
    global _ctx_univ_ts, _ctx_univ_all24h
    async with _ctx_overlay_lock:
        now = time.monotonic()
        if _ctx_univ_all24h is None or (now - _ctx_univ_ts) >= _CTX_OVERLAY_TTL_SEC:
            syms = get_trading_symbols()
            parts = await asyncio.gather(*[svc._fetch_24h(s) for s in syms], return_exceptions=True)
            merged: dict[str, dict[str, float]] = {}
            for s, p in zip(syms, parts, strict=False):
                if isinstance(p, dict):
                    merged[s] = p
                else:
                    merged[s] = {"change_24h_pct": 0.0, "volume_24h_usd": 0.0}
            _ctx_univ_all24h = merged
            _ctx_univ_ts = now

    all24 = _ctx_univ_all24h or {}
    t_self = all24.get(sym)
    if t_self is None:
        t_self = await svc._fetch_24h(sym)
    btc_blk = all24.get("BTCUSDT") or {"volume_24h_usd": 0.0, "change_24h_pct": 0.0}
    eth_blk = all24.get("ETHUSDT") or {"volume_24h_usd": 0.0, "change_24h_pct": 0.0}

    vols = sorted(max(0.0, float(v.get("volume_24h_usd", 0.0))) for v in all24.values())
    median_vol = vols[len(vols) // 2] if vols else 1.0
    median_vol = median_vol or 1.0

    total_vol = sum(max(0.0, float(v.get("volume_24h_usd", 0.0))) for v in all24.values())
    total_vol = total_vol if total_vol > 0 else 1.0
    btc_raw = max(0.0, float(btc_blk.get("volume_24h_usd", 0.0)))
    btc_dom = btc_raw / total_vol

    sym_vol = max(0.0, float(t_self.get("volume_24h_usd", 0.0)))
    chg = float(t_self.get("change_24h_pct", 0.0))
    rel_vol = sym_vol / median_vol if median_vol > 0 else 0.0
    liq = _liquidity_tier(sym_vol)

    rs_btc = max(-1.0, min(1.0, (chg - float(btc_blk.get("change_24h_pct", 0.0))) * 5.0))
    rs_eth = max(-1.0, min(1.0, (chg - float(eth_blk.get("change_24h_pct", 0.0))) * 5.0))

    out["ctx_change_24h_pct"] = str(chg)
    out["ctx_volume_24h_usd"] = str(sym_vol)
    out["ctx_relative_volume"] = str(rel_vol)
    out["ctx_liquidity_tier"] = str(liq)
    out["ctx_rs_btc"] = str(rs_btc)
    out["ctx_rs_eth"] = str(rs_eth)
    out["ctx_btc_dominance_proxy"] = str(btc_dom)
    return out


__all__ = ["AIMarketContextService", "get_market_context_service", "hydrate_ai_context_payload"]
