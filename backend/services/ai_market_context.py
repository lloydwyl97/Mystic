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
       -- ``ctx_btc_dominance_proxy``: composite market-structure signal — TOP4
          (BTC/ETH/SOL/XRP) green-breadth + real correlation-to-BTC + residual
          volume-share, so a single symbol's context reflects whether the whole
          traded universe agrees, not just BTC's volume share (2026-07-25).
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
from backend.config.trading_universe import TOP4_BASE_COINS, TRADING_SYMBOLS, get_trading_symbols
from backend.database_schema import DATABASE_PATH
from backend.derivatives_monitor import derivatives_positioning_signal, derivatives_reference_snapshot
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_decision_contract import (
    AI_CONTEXT_LOOP_SEC,
    AI_SENTIMENT_LOOP_SEC,
    CTX_BTC_LAG_WEIGHT,
    CTX_CROSS_EXCHANGE_WEIGHT,
    CTX_DEPTH_WEIGHT,
    CTX_DERIVATIVES_WEIGHT,
    CTX_FEATURE_STACK_WEIGHT,
    CTX_MICROSTRUCTURE_WEIGHT,
    CTX_MTF_ALIGN_WEIGHT,
    CTX_REGIME_WEIGHT,
    CTX_RS_WEIGHT,
    CTX_TOTAL_CAP,
    MARKET_CONTEXT_FIELDS,
    REDIS_KEY_AI_CONTEXT,
    REDIS_KEY_AI_SENTIMENT,
    REDIS_TTL_AI_CONTEXT_SEC,
)
from backend.services.ai_multi_target_regressors import predict_multi_target_from_latest_inference
from backend.services.catalyst_provider import get_default_provider
from backend.services.cross_exchange_reference import cross_exchange_dislocation_signal, cross_exchange_snapshot
from backend.services.day_active_market_bundle import (
    apply_day_bundle_stagger,
    async_fetch_day_active_ohlcv_bundle,
    async_read_cached_day_active_bundle,
    month_context_four_from_daily,
)
from backend.services.day_feature_stack_v2 import (
    btc_lag_predictive_signal,
    compute_feature_stack_snapshot,
    momentum_pct,
    momentum_rvol_confirmation_signal,
)
from backend.services.live_market_data import live_market_data_service
from backend.services.market_regime import regime_score
from backend.services.market_role_intelligence import (
    MARKET_ROLES,
    cache_role_context,
    compute_market_role_context,
)
from backend.services.multi_horizon_ev import compute_multi_horizon_ev

logger = logging.getLogger(__name__)


def _ensure_role_intel_column() -> None:
    """Idempotent migration: add role_intel_json column to ai_context_snapshots."""
    from backend.database_schema import DATABASE_PATH

    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(ai_context_snapshots)")]
            if "role_intel_json" not in cols:
                conn.execute("ALTER TABLE ai_context_snapshots ADD COLUMN role_intel_json TEXT")
                conn.commit()
                logger.info("AI_CONTEXT: added role_intel_json column to ai_context_snapshots")
    except Exception as exc:
        logger.debug("_ensure_role_intel_column: %s", exc)


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


def _market_regime_from_mtf(mtf: dict[str, dict[str, float]]) -> str:
    """Classify trend regime (trending_up/trending_down/chop) from a 1h+4h MTF pack.

    Works on ANY symbol's own MTF snapshot — originally BTC-only (hence the old
    name), which meant every symbol inherited BTC's regime label even when its
    own chart was clearly breaking out independently of BTC (e.g. ETH running
    while BTC chops). Now called once per symbol with that symbol's own_mtf so
    each coin is judged on its own trend (see 2026-07-26 "misses breakouts"
    investigation). Still called with btc_mtf for the broad/BTC-wide regime used
    by market-role classification, where a BTC-wide view is the correct input.
    """
    h1 = mtf.get("1h", {})
    h4 = mtf.get("4h", {})
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
    microstructure_signal: float = 0.0,
    feature_stack_signal: float = 0.0,
    btc_lag_signal: float = 0.0,
    derivatives_signal: float = 0.0,
    cross_exchange_signal: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """
    Combine MTF + RS + depth + regime + microstructure + feature-stack
    (momentum x same-symbol RVOL) + BTC-lag + derivatives + cross-exchange
    into one multiplier on winner_probability.

    Returns (multiplier, applied_components_for_audit).
    The multiplier is symmetric around 1.0 and clamped to +/- CTX_TOTAL_CAP.

    ``microstructure_signal`` is the bounded [-1, 1] normalized output of
    ``microstructure_engine.get_microstructure_ranking_delta`` (already
    divided by its own cap) — real order-flow/OFI/microprice evidence, never
    a hard gate. See backend/services/microstructure_engine.py.

    ``feature_stack_signal``/``btc_lag_signal``/``derivatives_signal``/
    ``cross_exchange_signal`` are the p15/p16/p18/p19/p20 ranking promotions
    (see day_feature_stack_v2.py, derivatives_monitor.py,
    cross_exchange_reference.py) — each already bounded [-1, 1] and each
    honestly 0.0 whenever its underlying feed is unavailable/insufficient,
    never a hard gate.
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

    micro_term = max(-1.0, min(1.0, microstructure_signal)) * CTX_MICROSTRUCTURE_WEIGHT
    feature_stack_term = max(-1.0, min(1.0, feature_stack_signal)) * CTX_FEATURE_STACK_WEIGHT
    btc_lag_term = max(-1.0, min(1.0, btc_lag_signal)) * CTX_BTC_LAG_WEIGHT
    derivatives_term = max(-1.0, min(1.0, derivatives_signal)) * CTX_DERIVATIVES_WEIGHT
    cross_exchange_term = max(-1.0, min(1.0, cross_exchange_signal)) * CTX_CROSS_EXCHANGE_WEIGHT

    total = mtf_term + rs_term + depth_term + regime_term + micro_term + feature_stack_term + btc_lag_term + derivatives_term + cross_exchange_term
    total = max(-CTX_TOTAL_CAP, min(CTX_TOTAL_CAP, total))
    multiplier = 1.0 + total

    return multiplier, {
        "mtf_align_signed": float(mtf_align),
        "mtf_term": float(mtf_term),
        "rs_avg": float(rs_avg),
        "rs_term": float(rs_term),
        "depth_term": float(depth_term),
        "regime_term": float(regime_term),
        "microstructure_term": float(micro_term),
        "feature_stack_term": float(feature_stack_term),
        "btc_lag_term": float(btc_lag_term),
        "derivatives_term": float(derivatives_term),
        "cross_exchange_term": float(cross_exchange_term),
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
        self._catalyst_provider = None
        ensure_ai_canonical_tables()
        _ensure_role_intel_column()

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
        self._catalyst_provider = get_default_provider()
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
        # Broad/BTC-wide regime — kept for market-role classification, which needs a
        # market-wide reference point to judge whether a coin is leading/lagging.
        broad_market_regime = _market_regime_from_mtf(btc_mtf)
        market_regime = broad_market_regime  # back-compat local name used below

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
        btc_vol_share = all_24h["BTCUSDT"].get("volume_24h_usd", 0.0) / total_vol

        # TOP4 MARKET BREADTH: fraction of the four DAY-traded coins (BTC/ETH/SOL/XRP)
        # green on the day, in [0,1]. Real cross-coin structure, not a BTC-only proxy —
        # closes the gap where a single BTC-derived regime label got stamped onto every
        # symbol even when e.g. SOL is diverging from BTC (see ai_regime_validation.py /
        # bull-regime tuning audit, 2026-07-25).
        _top4_usdt = [f"{c}USDT" for c in TOP4_BASE_COINS]
        _top4_changes = [all_24h[s]["change_24h_pct"] for s in _top4_usdt if s in all_24h]
        top4_breadth = (sum(1.0 for c in _top4_changes if c > 0) / len(_top4_changes)) if _top4_changes else 0.5

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

        # Pre-extract 1h OHLCV rows for BTC (needed for role correlation/beta)
        btc_bundle_raw: dict = {}
        with contextlib.suppress(Exception):
            ccxt_btc = _to_ccxt("BTCUSDT")
            btc_bundle_raw = await async_read_cached_day_active_bundle(ccxt_btc) or {}
        btc_rows_1h: list | None = btc_bundle_raw.get("1h") if btc_bundle_raw else None
        btc_rows_4h: list | None = btc_bundle_raw.get("4h") if btc_bundle_raw else None

        for symbol in ordered_symbols:
            sym_t0 = time.perf_counter()
            try:
                own_mtf = mtf_by_sym[symbol]
                # Per-symbol regime: this coin's own 1h/4h trend, not a copy of BTC's.
                # BTC's own regime is reused as-is here (own_mtf IS btc_mtf for BTCUSDT),
                # so this is a superset of the old behavior, not a divergence for BTC itself.
                symbol_regime = broad_market_regime if symbol == "BTCUSDT" else _market_regime_from_mtf(own_mtf)
                t24 = all_24h.get(symbol, {"change_24h_pct": 0.0, "volume_24h_usd": 0.0})
                spread_pct, depth_imb = depth_by_sym[symbol]

                rs_btc = t24["change_24h_pct"] - btc_24h["change_24h_pct"]
                rs_eth = t24["change_24h_pct"] - eth_24h["change_24h_pct"]
                rs_btc = max(-1.0, min(1.0, rs_btc * 5.0))  # scale: 20% relative move = full strength
                rs_eth = max(-1.0, min(1.0, rs_eth * 5.0))

                rel_vol = t24["volume_24h_usd"] / median_vol if median_vol > 0 else 0.0
                liq_tier = _liquidity_tier(t24["volume_24h_usd"])

                # TOP4 CORRELATION: real Pearson correlation of this symbol's 1h returns
                # vs BTC's 1h returns over the recent window — genuine co-movement,
                # distinct from relative strength (relative performance, not correlation).
                corr_to_btc = 1.0
                if symbol != "BTCUSDT" and btc_rows_1h and len(btc_rows_1h) >= 20:
                    try:
                        own_bundle_for_corr = await async_read_cached_day_active_bundle(_to_ccxt(symbol)) or {}
                        own_rows_1h = own_bundle_for_corr.get("1h")
                        if own_rows_1h and len(own_rows_1h) >= 20:
                            n = min(len(own_rows_1h), len(btc_rows_1h), 60)
                            own_closes = np.array([float(r[4]) for r in own_rows_1h[-n:]])
                            btc_closes_corr = np.array([float(r[4]) for r in btc_rows_1h[-n:]])
                            own_rets = np.diff(own_closes) / own_closes[:-1]
                            btc_rets = np.diff(btc_closes_corr) / btc_closes_corr[:-1]
                            if len(own_rets) >= 10 and np.std(own_rets) > 1e-12 and np.std(btc_rets) > 1e-12:
                                corr_to_btc = float(np.corrcoef(own_rets, btc_rets)[0, 1])
                                if not math.isfinite(corr_to_btc):
                                    corr_to_btc = 0.0
                    except Exception as corr_exc:
                        logger.debug("AI_CONTEXT corr_to_btc %s failed: %s", symbol, corr_exc)
                        corr_to_btc = 0.0

                # Composite replacing the old pure-volume-share proxy: breadth (cross-coin
                # agreement) + real correlation-to-BTC + a residual volume-share term for
                # continuity. Same [0,1] range and field name as before — no feature-vector
                # schema change, just a materially richer signal in the same slot.
                market_structure_signal = max(
                    0.0,
                    min(1.0, 0.45 * top4_breadth + 0.30 * ((corr_to_btc + 1.0) / 2.0) + 0.25 * btc_vol_share),
                )

                # Real microstructure engine (OFI + aggressor flow + microprice
                # pressure) — bounded [-cap, +cap]; normalized to [-1, 1] here
                # for the shared multiplier formula. Never a gate.
                microstructure_delta = 0.0
                micro_cap = 0.03
                with contextlib.suppress(Exception):
                    from backend.services.microstructure_engine import (
                        _RANKING_DELTA_CAP,
                        get_microstructure_ranking_delta,
                    )

                    microstructure_delta = get_microstructure_ranking_delta(symbol)
                    micro_cap = _RANKING_DELTA_CAP
                microstructure_signed = (microstructure_delta / micro_cap) if micro_cap else 0.0

                # ------------------------------------------------------------------
                # Feature-stack completion (p15 momentum, p16 same-symbol RVOL,
                # p17 volatility stack, p19 BTC lag correlation) — computed here
                # (before _ctx_multiplier) so p15/p16/p19 can feed real ranking
                # terms below, not just diagnostic JSON. Reuses the same cached
                # DAY MTF bundle already fetched above (no extra API load).
                # ------------------------------------------------------------------
                feature_stack_json = "{}"
                fs_snapshot = None
                own_bundle_for_fs: dict = {}
                try:
                    own_bundle_for_fs = await async_read_cached_day_active_bundle(_to_ccxt(symbol)) or {}
                    fs_snapshot = compute_feature_stack_snapshot(
                        symbol,
                        own_bundle_for_fs,
                        btc_bundle=btc_bundle_raw if symbol.upper() != "BTCUSDT" else None,
                    )
                    feature_stack_json = json.dumps(fs_snapshot.to_dict(), separators=(",", ":"), default=str)
                except Exception as fs_exc:
                    logger.debug("AI_CONTEXT feature_stack %s failed: %s", symbol, fs_exc)

                feature_stack_signed = 0.0
                btc_lag_signed = 0.0
                if fs_snapshot is not None:
                    with contextlib.suppress(Exception):
                        feature_stack_signed = momentum_rvol_confirmation_signal(fs_snapshot.momentum, fs_snapshot.rvol)
                    with contextlib.suppress(Exception):
                        btc_recent_return = 0.0
                        btc_lag = fs_snapshot.btc_lag
                        if btc_lag is not None and btc_lag.confidence == "confident" and btc_lag.best_lag_bars < 0:
                            btc_rows_1m = btc_bundle_raw.get("1m") if btc_bundle_raw else None
                            lag_n = abs(btc_lag.best_lag_bars)
                            if btc_rows_1m and len(btc_rows_1m) > lag_n:
                                btc_recent_return = momentum_pct(btc_rows_1m, lag_n)
                        btc_lag_signed = btc_lag_predictive_signal(btc_lag, btc_recent_return)

                # ------------------------------------------------------------------
                # Derivatives reference feed (p18) — public, non-execution GLOBAL
                # Binance futures OI/funding/basis. Decoupled from EXCHANGE_ID
                # (execution venue has no futures market). Honest degraded state
                # (available=False) when unreachable; never a gate.
                # ------------------------------------------------------------------
                derivatives_json = '{"available": false, "degraded_reason": "not_attempted"}'
                deriv_signed = 0.0
                try:
                    deriv_snapshot = await asyncio.to_thread(derivatives_reference_snapshot, symbol)
                    derivatives_json = json.dumps(deriv_snapshot, separators=(",", ":"), default=str)
                    deriv_signed = derivatives_positioning_signal(deriv_snapshot)
                except Exception as deriv_exc:
                    logger.debug("AI_CONTEXT derivatives_reference %s failed: %s", symbol, deriv_exc)

                # ------------------------------------------------------------------
                # Cross-exchange informational layer (p20) — public Coinbase feed,
                # price dislocation + coarse volume ratio vs the execution venue.
                # Informational only; never changes the execution venue.
                # ------------------------------------------------------------------
                cross_exchange_json = '{"available": false, "degraded_reason": "not_attempted"}'
                cross_exchange_signed = 0.0
                try:
                    own_last_price = 0.0
                    with contextlib.suppress(Exception):
                        rows_1m_ce = (own_bundle_for_fs.get("1m") if own_bundle_for_fs else None) or []
                        if rows_1m_ce:
                            own_last_price = float(rows_1m_ce[-1][4])
                    if own_last_price > 0:
                        ce_snapshot = await asyncio.to_thread(
                            cross_exchange_snapshot,
                            symbol,
                            own_price=own_last_price,
                            own_volume_24h=float(t24.get("volume_24h_usd", 0.0)),
                        )
                        cross_exchange_json = json.dumps(ce_snapshot, separators=(",", ":"), default=str)
                        cross_exchange_signed = cross_exchange_dislocation_signal(ce_snapshot)
                except Exception as ce_exc:
                    logger.debug("AI_CONTEXT cross_exchange %s failed: %s", symbol, ce_exc)

                multiplier, audit = _ctx_multiplier(
                    own_mtf=own_mtf,
                    rs_btc=rs_btc,
                    rs_eth=rs_eth,
                    depth_imbalance=depth_imb,
                    market_regime=symbol_regime,
                    microstructure_signal=microstructure_signed,
                    feature_stack_signal=feature_stack_signed,
                    btc_lag_signal=btc_lag_signed,
                    derivatives_signal=deriv_signed,
                    cross_exchange_signal=cross_exchange_signed,
                )

                # ------------------------------------------------------------------
                # Market-role intelligence (new — appended to payload, no gates)
                # ------------------------------------------------------------------
                role_intel_json = "{}"
                role_ranking_delta = 0.0
                if symbol in MARKET_ROLES or symbol.upper() in MARKET_ROLES:
                    try:
                        # Extract 1h / 4h rows for this symbol from cache
                        sym_bundle_raw: dict = {}
                        with contextlib.suppress(Exception):
                            sym_bundle_raw = await async_read_cached_day_active_bundle(_to_ccxt(symbol)) or {}
                        sym_rows_1h = sym_bundle_raw.get("1h") if sym_bundle_raw else None
                        sym_rows_4h = sym_bundle_raw.get("4h") if sym_bundle_raw else None

                        # Approximate 2h notional volume from last ~120 1m bars
                        vol_2h = 0.0
                        rows_1m = sym_bundle_raw.get("1m") if sym_bundle_raw else None
                        if rows_1m and len(rows_1m) >= 30:
                            for r in rows_1m[-120:]:
                                vol_2h += float(r[4]) * float(r[5])

                        role_ctx = await compute_market_role_context(
                            symbol,
                            btc_rows_1h=btc_rows_1h,
                            sym_rows_1h=sym_rows_1h,
                            btc_rows_4h=btc_rows_4h,
                            sym_rows_4h=sym_rows_4h,
                            mtf_data=own_mtf,
                            market_regime=market_regime,
                            volume_24h_usd=float(t24.get("volume_24h_usd", 0.0)),
                            volume_2h_usd=vol_2h,
                            catalyst_provider=self._catalyst_provider,
                        )
                        cache_role_context(role_ctx)
                        role_intel_json = json.dumps(role_ctx.to_dict(), separators=(",", ":"), default=str)
                        role_ranking_delta = role_ctx.live_ranking_delta()
                    except Exception as role_exc:
                        logger.debug("AI_CONTEXT role_intel %s failed: %s", symbol, role_exc)

                # (feature_stack_json / derivatives_json / cross_exchange_json were
                # computed above, before _ctx_multiplier, so p15/p16/p18/p19/p20 can
                # feed real ranking terms rather than diagnostic-only JSON.)

                # ------------------------------------------------------------------
                # Multi-horizon EV (p11) — composite EV across DAY's realistic
                # 15m-24h holding horizons, each estimated from ONLY the
                # historical trades whose own realized hold_seconds fell in
                # that bucket (mfe_mae_distribution_learner strata). Additive
                # diagnostic/ranking evidence; never a gate.
                # ------------------------------------------------------------------
                multi_horizon_ev_json = '{"available": false, "degraded_reason": "not_attempted"}'
                try:
                    mhev_result = await asyncio.to_thread(compute_multi_horizon_ev, symbol, "day")
                    multi_horizon_ev_json = json.dumps(mhev_result.to_dict(), separators=(",", ":"), default=str)
                except Exception as mhev_exc:
                    logger.debug("AI_CONTEXT multi_horizon_ev %s failed: %s", symbol, mhev_exc)

                # ------------------------------------------------------------------
                # Multi-target ML (p10) — expected_return/MFE/MAE/time-to-target
                # regression heads, reusing the exact feature vector already
                # logged for this symbol's most recent live decision
                # (ai_inference_log.features_json). Additive diagnostic
                # evidence; never a gate.
                # ------------------------------------------------------------------
                multi_target_ml_json = '{"available": false, "degraded_reason": "not_attempted"}'
                try:
                    mtml_pred = await asyncio.to_thread(predict_multi_target_from_latest_inference, "day", symbol)
                    multi_target_ml_json = json.dumps(mtml_pred.to_dict(), separators=(",", ":"), default=str)
                except Exception as mtml_exc:
                    logger.debug("AI_CONTEXT multi_target_ml %s failed: %s", symbol, mtml_exc)

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
                    "ctx_btc_dominance_proxy": float(market_structure_signal),
                    "ctx_top4_breadth": float(top4_breadth),
                    "ctx_corr_to_btc": float(corr_to_btc),
                    "ctx_market_regime": str(symbol_regime),
                    "ctx_broad_market_regime": str(broad_market_regime),
                    "ctx_sentiment_fear_greed": float(self._sentiment_value),
                    "ctx_multiplier": float(multiplier),
                    "mtf_json": json.dumps(own_mtf, separators=(",", ":")),
                    "ctx_audit_json": json.dumps(audit, separators=(",", ":")),
                    # New role-intelligence fields (append-only; existing consumers unaffected)
                    "ctx_role_intel_json": role_intel_json,
                    "ctx_role_ranking_delta": float(role_ranking_delta),
                    # Real microstructure engine fields (append-only). Ranking/EV
                    # input only — see backend/services/microstructure_engine.py.
                    "ctx_microstructure_ranking_delta": float(microstructure_delta),
                    # Feature-stack completion (append-only): multi-horizon momentum,
                    # same-symbol RVOL, ATR7/14/28 + realized vol + vol percentile,
                    # BTC lag correlation — see day_feature_stack_v2.py.
                    "ctx_feature_stack_json": feature_stack_json,
                    # Derivatives reference feed (append-only): OI/funding/basis
                    # from GLOBAL Binance futures — see backend/derivatives_monitor.py.
                    "ctx_derivatives_json": derivatives_json,
                    # Cross-exchange informational layer (append-only): Coinbase
                    # public-feed price dislocation + volume ratio — see
                    # backend/services/cross_exchange_reference.py.
                    "ctx_cross_exchange_json": cross_exchange_json,
                    # Multi-horizon EV (append-only): composite EV across
                    # DAY's 15m-24h realistic holding horizons — see
                    # backend/services/multi_horizon_ev.py.
                    "ctx_multi_horizon_ev_json": multi_horizon_ev_json,
                    # Multi-target ML (append-only): expected_return/MFE/MAE/
                    # time-to-target regression heads — see
                    # backend/services/ai_multi_target_regressors.py.
                    "ctx_multi_target_ml_json": multi_target_ml_json,
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
            # btc-dominance proxy: use last iteration's value if available.
            _btc_dom_val = float(locals().get("market_structure_signal", 0.0) or 0.0)
            logger.info(
                "AI_CONTEXT loop=%s published=%d/%d regime=%s btc_dom~%.2f",
                self._loop_count,
                published,
                len(self.symbols),
                market_regime,
                _btc_dom_val,
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
                        mtf_json, ctx_multiplier, role_intel_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        payload.get("ctx_role_intel_json", "{}"),
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
