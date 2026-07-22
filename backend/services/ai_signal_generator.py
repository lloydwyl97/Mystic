#!/usr/bin/env python3
"""
Real-Time AI Signal Generation System (Production - DAY top-4 Binance.US).
"""

import asyncio
import contextlib
import json
import logging
import math
import os
import pickle
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import redis.asyncio as redis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backend.config.ai_signal_bus import AI_SIGNAL_REDIS_TTL_SEC, MAX_SIGNAL_AGE_SEC
from backend.config.redis_config import get_shared_redis_async
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_decision_contract import (
    AI_FEATURE_DIM_V1,
    CTX_TOTAL_CAP,
    REDIS_KEY_AI_CONTEXT,
    REDIS_KEY_AI_SENTIMENT,
)
from backend.services.confidence_normalizer import ConfidenceNormalizer
from backend.services.gate_reason_codes import GateReason
from backend.services.live_market_data import live_market_data_service
from backend.services.live_strategy_contracts import (
    contract_for,
    live_ai_fail_closed_without_context,
    live_ai_min_feature_version,
    live_ai_min_feature_versions_map,
    live_ai_strict_startup,
    parse_enabled_live_strategies,
    per_coin_artifact_file,
    redis_ai_signal_key,
)
from backend.utils.redis_helpers import WRITER_ROLES, WriterLock, create_writer_payload

try:
    import talib
except ImportError:
    talib = None
    # Note: TA-Lib optional, will use numpy-based alternatives

# UPGRADED: Import Fear/Greed Index for sentiment
try:
    from backend.services.market_regime import get_regime_snapshot_for_signal, regime_score

    REGIME_SCORE_AVAILABLE = True
except ImportError:
    REGIME_SCORE_AVAILABLE = False
    regime_score = None
    get_regime_snapshot_for_signal = None  # type: ignore[misc, assignment]

from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy
from backend.config.buy_admission import (
    buy_margin_threshold_active,
    buy_margin_threshold_core,
    compute_buy_margin,
)
from backend.config.rf_production_standard import rf_live_artifact_production_grade
from backend.config.settings import settings
from backend.config.trade_worthiness_timing import day_label_grid_seconds
from backend.services.day_active_market_bundle import (
    apply_day_bundle_stagger,
    async_fetch_day_active_ohlcv_bundle,
    validate_day_active_bundle,
)
from backend.services.history_context_gates import (
    evaluate_multi_timeframe_coverage,
    feature_store_ohlcv_fallback_enabled,
    feature_store_rows_to_ohlcv,
    min_ohlcv_bars_for_signal,
    min_primary_bars_for_strategy,
    mtf_fetch_limit,
    mtf_history_gate_enabled,
    mtf_min_bars_per_timeframe,
    mtf_required_ok_count,
    ohlcv_1m_fetch_limit_for_primary,
    ohlcv_fetch_limit_1m,
)
from backend.services.portfolio_engine import Sleeve, assign_sleeve
from backend.services.strategy_runtime_audit import (
    EVT_SIGNAL_EMITTED,
    compute_context_freshness,
    insert_audit_row_async,
    sha256_file,
    validate_loaded_slots,
)
from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)


def _finalize_prediction_buy_margin(
    symbol_bus: str,
    prediction_pre_margin: str,
    model_probs: dict[str, float],
    winner_raw: float,
) -> tuple[str, float, float]:
    """
    Compute buy_margin and normalize confidence for telemetry.
    Model direction is PRESERVED — no sleeve or margin gate alters prediction.
    """
    buy_margin = compute_buy_margin(model_probs)
    winner_probability = ConfidenceNormalizer.normalize(float(winner_raw))
    if prediction_pre_margin == "BUY":
        try:
            ccxt_sym = CanonicalSymbolFormatter.to_ccxt(symbol_bus)
        except Exception:
            base = symbol_bus.replace("USDT", "").replace("/", "")
            ccxt_sym = f"{base}/USDT"
        sleeve_asg = assign_sleeve(ccxt_sym, winner_probability, None)
        thr = buy_margin_threshold_core() if sleeve_asg == Sleeve.CORE.value else buy_margin_threshold_active()
        if buy_margin < thr:
            logger.info(
                "BUY_MARGIN_TELEMETRY: %s buy_margin=%.4f < thr=%.4f sleeve=%s (model direction preserved, not gating)",
                symbol_bus,
                buy_margin,
                thr,
                sleeve_asg,
            )
    return prediction_pre_margin, winner_probability, buy_margin


# Module-level flag: only CREATE TABLE once per process
_signal_table_created = False


def _to_ccxt_symbol(sym: str) -> str:
    """Convert symbol to CCXT format using canonical formatter"""
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        return CanonicalSymbolFormatter.to_ccxt(sym)
    except Exception as ex:
        logger.debug("CanonicalSymbolFormatter unavailable, using fallback: %s", ex)
        # Fallback with fix for double USDT
        normalized = sym.replace("-", "/").replace("_", "/")
        if "/" not in normalized:
            # Check if already ends with USDT to avoid double USDT
            if normalized.endswith("USDT"):
                # Already has USDT, convert: BTCUSDT -> BTC/USDT
                base = normalized[:-4]
                return f"{base}/USDT"
            return f"{normalized}/USDT"
        return normalized


class RealTimeAISignalGenerator:
    """Real-time AI signal generation for trading using trained ML models"""

    def __init__(self) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            redis_host = os.getenv("REDIS_HOST")
            if not redis_host:
                msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis URL"
                raise RuntimeError(msg)
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_db = os.getenv("REDIS_DB", "0")
            redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.redis_url = redis_url
        self.redis: redis.Redis | None = None
        self.writer_lock: WriterLock | None = None

        self.symbols = settings.trading_symbols
        self.enabled_strategies: tuple[str, ...] = parse_enabled_live_strategies()
        # Slot key = "<strategy_id>:<SYMBOL>" — two-strategy live path only (no generic artifact).
        self.models: dict[str, RandomForestClassifier] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.model_artifact_paths: dict[str, str] = {}
        self.model_label_versions: dict[str, str] = {}
        self.model_label_horizons: dict[str, int] = {}
        self.model_feature_versions: dict[str, int] = {}
        self.model_feature_dims: dict[str, int] = {}
        self.model_artifact_sha256: dict[str, str] = {}
        self.feature_history: dict[str, list[list[float]]] = {}

        self.models_dir = Path("models/active")
        self.scalers_dir = Path("models/scalers")

        self.is_running = False
        self._tasks: list[asyncio.Task[Any]] = []

        self._model_reload_check_interval = 300.0
        self._last_model_check_time = 0.0
        self._last_model_mtime = 0.0

        self._strategy_last_tick: dict[str, float] = {}

        self.feature_count = 124
        self.lookback_periods = [5, 10, 20, 50, 100, 200]

        self.fear_greed_enabled = REGIME_SCORE_AVAILABLE
        self.last_fear_greed_score = 0.0

        logger.info(
            "Live AI strategies (canonical): %s — artifacts under %s/<strategy>/<SYMBOL>_direction.pkl",
            list(self.enabled_strategies),
            self.models_dir,
        )

    @staticmethod
    def _slot(strategy_id: str, symbol: str) -> str:
        return f"{strategy_id.strip().lower()}:{symbol.strip().upper()}"

    async def start(self) -> None:
        """Start the signal generator"""
        if self.is_running:
            logger.warning("Signal generator already running")
            return

        self.redis = get_shared_redis_async()
        if self.redis is None:
            logger.error("Shared Redis client unavailable; cannot start signal generator")
            return

        # CRITICAL: Acquire writer lock for ai_signal:* keyspace
        self.writer_lock = WriterLock(WRITER_ROLES["AI_SIGNALS"], self.redis)
        success = await self.writer_lock.acquire()
        if not success:
            logger.error("Failed to acquire AI signals writer lock - another signal generator is running")
            return

        try:
            await self._initialize_models()
        except Exception as e:
            logger.exception(f"Failed to start signal generator: {e}")
            if self.writer_lock:
                await self.writer_lock.release()
            raise

        # Start signal generation loop
        from backend.services.task_manager import task_manager

        task = await task_manager.create_task(self._signal_generation_loop(), name="ai_signal_generator:signal_generation_loop")
        self._tasks.append(task)

        self.is_running = True
        logger.info("Real-time AI signal generator started successfully with trained models.")

    async def stop(self) -> None:
        """Stop the signal generator"""
        self.is_running = False

        for task in self._tasks:
            if not task.done():
                task.cancel()

        self._tasks.clear()

        # Release writer lock
        if self.writer_lock:
            await self.writer_lock.release()
            self.writer_lock = None

        if self.redis:
            await self.redis.aclose()  # type: ignore[attr-defined]
            self.redis = None

        logger.info("AI signal generator stopped")

    async def _initialize_models(self) -> None:
        """Load per-strategy per-coin RF artifacts from models/active/<strategy>/<SYM>_direction.pkl."""
        logger.info("Loading per-strategy per-coin AI models from disk...")
        self.models.clear()
        self.scalers.clear()
        self.model_artifact_paths.clear()
        self.model_label_versions.clear()
        self.model_label_horizons.clear()
        self.model_feature_versions.clear()
        self.model_feature_dims.clear()
        self.model_artifact_sha256.clear()

        min_fv_global = live_ai_min_feature_version()
        min_by_strat = live_ai_min_feature_versions_map(self.enabled_strategies)
        loaded_slots = 0
        missing_required: list[str] = []

        for strategy_id in self.enabled_strategies:
            min_fv = min_by_strat.get(strategy_id.strip().lower(), min_fv_global)
            for symbol in self.symbols:
                slot = self._slot(strategy_id, symbol)
                model_path = per_coin_artifact_file(self.models_dir, strategy_id, symbol)
                if not model_path.exists():
                    msg = f"{strategy_id}/{symbol}"
                    logger.warning("PER_COIN_MODEL_MISSING: %s — no artifact at %s", msg, model_path)
                    missing_required.append(msg)
                    continue
                try:
                    with model_path.open("rb") as f:
                        model_data = pickle.load(f)
                    if not isinstance(model_data, dict) or model_data.get("model") is None:
                        logger.warning("PER_COIN_MODEL_INVALID: %s — no 'model' key in %s", slot, model_path)
                        continue
                    coin_model = model_data["model"]
                    coin_scaler = model_data.get("scaler") or model_data.get("global_scaler")
                    per_scalers = model_data.get("scalers")
                    if isinstance(per_scalers, dict) and symbol in per_scalers:
                        coin_scaler = per_scalers[symbol]
                    if coin_scaler is None:
                        logger.warning("PER_COIN_MODEL_NO_SCALER: %s — skipping", slot)
                        continue
                    fv = model_data.get("feature_version")
                    fd = model_data.get("feature_dim")
                    if isinstance(fd, int) and fd in (124, 145):
                        self.model_feature_dims[slot] = fd
                        if isinstance(fv, int) and fv in (1, 2, 3, 4, 5):
                            self.model_feature_versions[slot] = fv
                        else:
                            self.model_feature_versions[slot] = 2 if fd == 145 else 1
                    elif isinstance(fv, int) and fv in (1, 2, 3, 4, 5):
                        self.model_feature_versions[slot] = fv
                        self.model_feature_dims[slot] = 145 if fv >= 2 else 124
                    else:
                        inferred_dim = 124
                        try:
                            inferred_dim = int(coin_scaler.mean_.shape[0])
                        except (AttributeError, IndexError, TypeError):
                            pass
                        if inferred_dim not in (124, 145):
                            inferred_dim = 124
                        self.model_feature_dims[slot] = inferred_dim
                        self.model_feature_versions[slot] = 2 if inferred_dim == 145 else 1

                    if self.model_feature_versions[slot] < min_fv:
                        logger.error(
                            "ARTIFACT_REJECTED: %s feature_version=%d < LIVE_AI_MIN_FEATURE_VERSION=%d — not loaded",
                            slot,
                            self.model_feature_versions[slot],
                            min_fv,
                        )
                        continue

                    self.models[slot] = coin_model
                    self.scalers[slot] = coin_scaler
                    self.model_artifact_paths[slot] = str(model_path.resolve())
                    self.model_artifact_sha256[slot] = sha256_file(model_path)
                    self.model_label_versions[slot] = str(model_data.get("label_version", ""))
                    try:
                        self.model_label_horizons[slot] = int(
                            model_data.get("label_lookahead_bars") or model_data.get("label_lookahead") or 0,
                        )
                    except (TypeError, ValueError):
                        self.model_label_horizons[slot] = 0
                    loaded_slots += 1
                    logger.info(
                        "PER_COIN_MODEL_LOADED: %s strategy=%s accuracy=%.3f fv=%d dim=%d path=%s",
                        symbol,
                        strategy_id,
                        float(model_data.get("accuracy", 0.0)),
                        self.model_feature_versions[slot],
                        self.model_feature_dims[slot],
                        model_path,
                    )
                except Exception as e:
                    logger.exception("PER_COIN_MODEL_LOAD_FAILED: %s error=%s", slot, e)

        need = len(self.enabled_strategies) * len(self.symbols)
        logger.info("Per-strategy models loaded: %d/%d slots", loaded_slots, need)
        if live_ai_strict_startup() and loaded_slots < need:
            msg = f"LIVE_AI_STRICT: expected artifacts for every enabled strategy x symbol ({need} slots), loaded {loaded_slots}. Missing includes: {missing_required[:12]}..."
            raise RuntimeError(msg)
        if loaded_slots == 0:
            logger.warning("NO_PER_COIN_MODELS: no strategy artifacts loaded — no signals will emit")

        validate_loaded_slots(
            models=self.models,
            model_feature_versions=self.model_feature_versions,
            model_feature_dims=self.model_feature_dims,
            model_artifact_paths=self.model_artifact_paths,
            model_artifact_sha256=self.model_artifact_sha256,
            enabled_strategies=self.enabled_strategies,
            min_feature_version=min_fv_global,
            min_feature_version_by_strategy=min_by_strat,
        )

    async def _signal_generation_loop(self) -> None:
        """Tick loop: each live strategy has its own signal clock (see live_strategy_contracts)."""
        logger.info("Starting AI signal generation loop (per-strategy clocks)...")
        await apply_day_bundle_stagger("signal")
        loop_count = 0
        tick_sec = 5.0
        while self.is_running:
            try:
                loop_count += 1
                now = time.time()
                for strategy_id in self.enabled_strategies:
                    c = contract_for(strategy_id)
                    last = self._strategy_last_tick.get(strategy_id, 0.0)
                    if now - last < c.redis_signal_loop_seconds:
                        continue
                    self._strategy_last_tick[strategy_id] = now
                    logger.info(
                        " AI Signal tick #%d strategy=%s — %d symbols",
                        loop_count,
                        strategy_id,
                        len(self.symbols),
                    )
                    for symbol in self.symbols:
                        await self._generate_signal_for_symbol(strategy_id, symbol)

                if now - self._last_model_check_time >= self._model_reload_check_interval:
                    self._last_model_check_time = now
                    try:
                        mtimes: list[float] = []
                        if self.models_dir.exists():
                            for sid in self.enabled_strategies:
                                for sym in self.symbols:
                                    p = per_coin_artifact_file(self.models_dir, sid, sym)
                                    if p.exists():
                                        with contextlib.suppress(OSError):
                                            mtimes.append(p.stat().st_mtime)
                        current_mtime = max(mtimes) if mtimes else 0.0
                        if current_mtime > self._last_model_mtime and self._last_model_mtime > 0:
                            await self._initialize_models()
                            logger.info("MODEL RELOADED")
                        self._last_model_mtime = current_mtime if current_mtime > 0 else self._last_model_mtime
                    except Exception as reload_e:
                        logger.debug("Model reload check skipped: %s", reload_e)

                await asyncio.sleep(tick_sec)
            except Exception as e:
                logger.exception(f"Error in signal generation loop: {e}")
                await asyncio.sleep(5)

    async def _preserve_existing_signal_ttl(self, strategy_id: str, symbol: str, *, skip_reason: str = "") -> None:
        """
        On generation skip: never extend TTL indefinitely while hash timestamp stays old.

        Short grace may refresh Redis TTL but always stamps content staleness on the hash.
        Beyond MAX_SIGNAL_AGE_SEC the key is deleted so TTL alone cannot masquerade as fresh.
        """
        if not self.redis:
            return
        key = redis_ai_signal_key(strategy_id, symbol)
        try:
            raw = await self.redis.hgetall(key)
            if not raw:
                return

            dd: dict[str, str] = {}
            for k, v in raw.items():
                kk = k.decode() if isinstance(k, bytes) else str(k)
                vv = v.decode() if isinstance(v, bytes) else str(v)
                dd[kk] = vv

            content_age: float | None = None
            for field in ("timestamp", "writer_timestamp"):
                raw_ts = (dd.get(field) or "").strip()
                if not raw_ts:
                    continue
                try:
                    content_age = max(0.0, time.time() - float(raw_ts))
                    break
                except (TypeError, ValueError):
                    continue

            if content_age is not None and content_age > float(MAX_SIGNAL_AGE_SEC):
                await self.redis.delete(key)
                logger.warning(
                    "SIGNAL_TTL_PRESERVE_DELETE key=%s content_age_sec=%.1f max=%s reason=%s",
                    key,
                    content_age,
                    MAX_SIGNAL_AGE_SEC,
                    skip_reason,
                )
                return

            # Content still within MAX_SIGNAL_AGE: refresh writer timestamp so entry gates
            # (SIGNAL_CONTENT_AGE_EXCEEDED / content_fresh) stay healthy on generation skips.
            now_ts = time.time()
            fresh_patch: dict[str, str] = {
                "timestamp": str(now_ts),
                "writer_timestamp": str(now_ts),
                "content_fresh": "1",
                "signal_content_stale": "0",
                "content_age_sec": "0",
                "ttl_preserve_skip_reason": (skip_reason or "")[:240],
            }
            # Keep context_fresh aligned with live ai_context (avoid ENTRY_CONTEXT_NOT_FRESH split-brain).
            try:
                from backend.services.ai_context_freshness_sync import overlay_live_context_freshness

                overlay_dd: dict[str, str] = dict(dd)
                try:
                    overlay_dd["feature_version"] = str(int(float(dd.get("feature_version") or "1")))
                except (TypeError, ValueError):
                    overlay_dd["feature_version"] = "1"
                overlay_live_context_freshness(overlay_dd, symbol)
                for field in ("ctx_ts_utc", "ctx_age_sec", "context_fresh", "context_audit_emit"):
                    if field in overlay_dd and overlay_dd[field] not in (None, ""):
                        fresh_patch[field] = str(overlay_dd[field])
            except Exception as ctx_exc:
                logger.debug("SIGNAL_TTL_PRESERVE context overlay skipped %s: %s", key, ctx_exc)
            await self.redis.hset(key, mapping=fresh_patch)
            await self.redis.expire(key, AI_SIGNAL_REDIS_TTL_SEC)
            logger.debug(
                "SIGNAL_TTL_PRESERVE_REFRESH key=%s prior_content_age_sec=%s reason=%s",
                key,
                content_age,
                skip_reason,
            )
            return
        except Exception as exc:
            logger.debug("SIGNAL_TTL_PRESERVE skipped %s: %s", key, exc)

    async def _assemble_day_live_features(
        self,
        ccxt_symbol: str,
        symbol: str,
        bundle: dict[str, list],
        ctx_h: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145
        from backend.services.ai_feature_fundamentals import merge_canonical_sentiment_payload
        from backend.services.day_feature_audit import build_context_provenance
        from backend.services.day_feature_health import build_compact_health_sidecar
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        base = CanonicalSymbolFormatter.to_base(symbol)
        orderbook = None
        vp = None
        ob_age: float | None = None
        if self.redis:
            with contextlib.suppress(Exception):
                raw_ob = await self.redis.hgetall(f"orderbook:{base}")
                from backend.services.order_book_service import parse_orderbook_redis_hash

                orderbook, ob_age = parse_orderbook_redis_hash(raw_ob)
            with contextlib.suppress(Exception):
                raw_vp = await self.redis.hgetall(f"volume_profile:{base}")
                if raw_vp:
                    vp = {(k.decode() if isinstance(k, bytes) else k): float(v.decode() if isinstance(v, bytes) else v) for k, v in raw_vp.items()}
        ob_live = orderbook
        if ob_live is None or float((ob_live or {}).get("bid_ask_spread") or 0.0) <= 0.0:
            with contextlib.suppress(Exception):
                from backend.services.order_book_service import fetch_order_book_features_live

                lo = await fetch_order_book_features_live(ccxt_symbol)
                if lo and float(lo.get("bid_ask_spread") or 0) > 0:
                    ob_live = lo
                    ob_age = 0.0

        sentiment: dict[str, Any] | None = None
        if self.redis:
            with contextlib.suppress(Exception):
                raw_s = await self.redis.get(REDIS_KEY_AI_SENTIMENT)
                if raw_s:
                    sdec = raw_s.decode() if isinstance(raw_s, bytes) else raw_s
                    sentiment = {"fear_greed_index": float(sdec)}
        sentiment = await merge_canonical_sentiment_payload(
            base_symbol=base,
            pair_symbol=symbol,
            ctx_for_overlay=ctx_h if isinstance(ctx_h, dict) else None,
            redis_client=self.redis,
            ohlcv_1m=bundle.get("1m") or [],
            existing=sentiment,
        )

        ctx_age: float | None = None
        if isinstance(ctx_h, dict):
            ctx_ts = ctx_h.get("ctx_ts_utc") or ctx_h.get("updated_at_utc")
            if ctx_ts:
                with contextlib.suppress(Exception):
                    from datetime import datetime, timezone

                    t_parse = datetime.fromisoformat(str(ctx_ts).replace("Z", "+00:00"))
                    ctx_age = max(0.0, (datetime.now(timezone.utc) - t_parse.astimezone(timezone.utc)).total_seconds())

        tech_prov: dict[str, dict[str, Any]] = {}
        vector = build_day_htf_feature_vector_145(
            symbol_ccxt=ccxt_symbol,
            day_bundle=bundle,
            volume_profile=vp,
            orderbook=ob_live,
            sentiment=sentiment,
            ai_context=ctx_h,
            tech_provenance=tech_prov,
            orderbook_age_sec=ob_age,
        )
        ctx_prov = build_context_provenance(
            {
                "ctx_h": ctx_h,
                "ctx_age_sec": ctx_age,
                "bundle_age_sec": 0.0,
                "orderbook_age_sec": ob_age,
            },
            bundle,
        )
        sidecar = build_compact_health_sidecar(vector, tech_prov, ctx_prov)
        return vector, sidecar

    async def _generate_signal_for_symbol(self, strategy_id: str, symbol: str) -> None:
        """Generate AI signal for (live strategy, symbol) using that strategy's per-coin RF artifact."""
        try:
            slot = self._slot(strategy_id, symbol)
            c = contract_for(strategy_id)
            logger.info(" %s Starting signal generation for %s", c.attribution_log_prefix, symbol)

            if slot not in self.models:
                logger.warning("PER_COIN_MODEL_SKIP: %s — no artifact loaded, fail-closed", slot)
                await self._preserve_existing_signal_ttl(strategy_id, symbol)
                return

            sid0 = (strategy_id or "").strip().lower()
            day_tf_audit: dict[str, int] = {}
            market_primary: list[list] = []
            market_1m_exec: list[list] = []
            ranking_source: list[list] = []
            bundle: dict[str, list] = {}
            # DAY reads native 1m for the RF feature vector but ranking indicators
            # (ADX/RSI/ema_alignment/momentum -> trend_score/chop_score) come from a
            # separate, coarser timeframe. Track which TF actually fed ranking so this
            # dual-clock isn't silently assumed downstream (explainability, audits).
            ranking_tf_label = "1m"

            if sid0 == "day":
                from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES

                ccxt_sym = _to_ccxt_symbol(symbol)
                if not live_market_data_service:
                    logger.warning("DAY_ACTIVE_SKIP_NO_LIVE: %s", symbol)
                    await self._preserve_existing_signal_ttl(strategy_id, symbol, skip_reason="DAY_ACTIVE_SKIP_NO_LIVE")
                    return
                bundle = await async_fetch_day_active_ohlcv_bundle(
                    live_market_data_service,
                    ccxt_sym,
                    force_refresh=True,
                )
                bundle_ok, miss = validate_day_active_bundle(bundle)
                if not bundle_ok:
                    counts = {tf: len(bundle.get(tf) or []) for tf in DAY_ACTIVE_TIMEFRAMES if isinstance(bundle.get(tf), list)}
                    logger.warning(
                        "DAY_ACTIVE_CONTRACT_FAIL: %s missing=%s counts=%s",
                        symbol,
                        miss,
                        counts,
                    )
                    await self._preserve_existing_signal_ttl(
                        strategy_id,
                        symbol,
                        skip_reason=f"DAY_ACTIVE_CONTRACT_FAIL:{','.join(miss[:4])}",
                    )
                    return
                market_primary = bundle.get("4h") or bundle.get("1h") or bundle.get("1m") or []
                ranking_source = bundle.get("4h") or market_primary
                market_1m_exec = bundle.get("1m") or []
                day_tf_audit = {
                    tf: len(bundle.get(tf) or [])
                    for tf in DAY_ACTIVE_TIMEFRAMES
                    if isinstance(bundle.get(tf), list) or bundle.get(tf) is None
                }
                if bundle.get("4h"):
                    ranking_tf_label = "4h"
                elif bundle.get("1h"):
                    ranking_tf_label = "1h"
                else:
                    ranking_tf_label = "1m"
                logger.info(
                    "DAY_ACTIVE_OHLCV %s bars=%s (1m_rows=%s)",
                    symbol,
                    day_tf_audit,
                    len(market_1m_exec),
                )
            else:
                market_primary, market_1m_exec = await self._get_market_data(symbol, strategy_id)
                min_bars = min_primary_bars_for_strategy(strategy_id)
                if not market_primary or len(market_primary) < min_bars:
                    logger.warning(
                        "INSUFFICIENT_PRIMARY_OHLCV: %s strat=%s bars=%s need>=%s — skip AI signal",
                        symbol,
                        strategy_id,
                        len(market_primary or []),
                        min_bars,
                    )
                    await self._preserve_existing_signal_ttl(strategy_id, symbol)
                    return

                if mtf_history_gate_enabled():
                    ok_tf, total_tf, tf_counts = await self._evaluate_multi_timeframe_coverage(symbol)
                    need_tf = mtf_required_ok_count(total_tf)
                    if ok_tf < need_tf:
                        logger.warning(
                            "INSUFFICIENT_MTF_HISTORY: %s ok_tf=%s need_tf=%s counts=%s — skip AI signal",
                            symbol,
                            ok_tf,
                            need_tf,
                            tf_counts,
                        )
                        await self._preserve_existing_signal_ttl(strategy_id, symbol)
                        return

                logger.info(
                    " Got %s primary bars (%ss) + %s 1m exec tail for %s",
                    len(market_primary),
                    primary_bar_seconds_for_strategy(strategy_id),
                    len(market_1m_exec or []),
                    symbol,
                )
                ranking_source = market_primary
                ranking_tf_label = f"{int(primary_bar_seconds_for_strategy(strategy_id))}s_primary"

            # CANONICAL: read ai_context BEFORE feature building so v2 artifacts
            # can incorporate context dims directly into the model input.
            ctx_payload, ctx_multiplier, ctx_audit = await self._read_ai_context(symbol)
            ctx_ts_str = (ctx_payload or {}).get("ts_utc", "") or ""
            ctx_age_sec = -1.0
            if ctx_ts_str:
                try:
                    tnorm = str(ctx_ts_str).replace("Z", "+00:00")
                    t_parse = datetime.fromisoformat(tnorm)
                    if t_parse.tzinfo is None:
                        t_parse = t_parse.replace(tzinfo=timezone.utc)
                    ctx_age_sec = (datetime.now(timezone.utc) - t_parse.astimezone(timezone.utc)).total_seconds()
                except Exception:
                    ctx_age_sec = -1.0
            ctx_age_for_audit = ctx_age_sec if ctx_age_sec >= 0 else None
            ctx_fresh, ctx_age_eff = compute_context_freshness(ctx_age_for_audit)
            from backend.services.ai_decision_contract import MARKET_CONTEXT_FIELDS

            _cp = ctx_payload if isinstance(ctx_payload, dict) else {}
            _missing_cf = [f for f in MARKET_CONTEXT_FIELDS if f not in _cp]
            ctx_defaulted_audit: dict[str, Any] = {
                "redis_ai_context_present": bool(ctx_payload),
                "redis_ctx_ts_utc": ctx_ts_str,
                "ctx_age_sec_at_emit": ctx_age_sec,
                "payload_field_count": len(ctx_payload or {}),
                "live_ai_fail_closed_without_context": live_ai_fail_closed_without_context(),
                "missing_contract_fields": _missing_cf,
            }

            from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1
            from backend.services.ai_feature_v2 import context_vector_from_ai_context

            feat_version = self.model_feature_versions.get(slot, 1)
            if feat_version >= 2 and live_ai_fail_closed_without_context() and not ctx_payload:
                logger.warning(
                    "AI_CONTEXT_FAIL_CLOSED: %s v%d model needs ai_context:%s — skip emit",
                    slot,
                    int(feat_version),
                    symbol,
                )
                await self._preserve_existing_signal_ttl(strategy_id, symbol)
                return

            expected_dim = self.model_feature_dims.get(slot, AI_FEATURE_DIM_V1)
            if sid0 == "day":
                from backend.services.ai_market_context import hydrate_ai_context_payload

                ctx_payload = await hydrate_ai_context_payload(symbol, ctx_payload)
                features, feature_health_sidecar = await self._assemble_day_live_features(
                    _to_ccxt_symbol(symbol),
                    symbol,
                    bundle,
                    ctx_payload,
                )
            else:
                feature_health_sidecar = {}
                base_features = await self._calculate_features(
                    market_primary,
                    market_1m_exec,
                    symbol,
                    strategy_id,
                )
                if feat_version >= 2:
                    from backend.services.ai_market_context import hydrate_ai_context_payload

                    ctx_payload = await hydrate_ai_context_payload(symbol, ctx_payload)
                    ctx_part = context_vector_from_ai_context(ctx_payload)
                    features = list(base_features)[:AI_FEATURE_DIM_V1] + list(ctx_part)
                else:
                    features = list(base_features)[:AI_FEATURE_DIM_V1]

            # NEW: retain a bounded history to avoid unbounded RAM growth
            hist = self.feature_history.setdefault(slot, [])
            hist.append(features)
            if len(hist) > 500:
                del hist[:-500]

            if len(features) != expected_dim:
                logger.error(
                    "FEATURE_DIM_MISMATCH: %s got=%d expected=%d (artifact_version=%d) — skip",
                    symbol,
                    len(features),
                    expected_dim,
                    feat_version,
                )
                await self._preserve_existing_signal_ttl(strategy_id, symbol)
                return

            logger.info(
                " Calculated %d features for %s (feature_version=%d)",
                len(features),
                symbol,
                feat_version,
            )

            # Scale features
            try:
                from backend.services.day_feature_health import zero_learning_blocked_feature_dims

                features = zero_learning_blocked_feature_dims(features)
            except Exception:
                pass
            features_scaled = self.scalers[slot].transform([features])

            # Make prediction
            prediction_proba = self.models[slot].predict_proba(features_scaled)[0]

            # FIXED: Handle both binary (2-class) and multi-class (3-class) models
            num_classes = len(prediction_proba)
            if num_classes == 2:
                rf_probs = {"sell": 0.0, "hold": float(prediction_proba[0]), "buy": float(prediction_proba[1])}
            else:
                rf_probs = {"sell": float(prediction_proba[0]), "hold": float(prediction_proba[1]), "buy": float(prediction_proba[2])}

            # Per-coin model is the sole directional authority. No ensemble override.
            # CANONICAL DECISION: ensemble (LSTM/Transformer/Chart/FearGreedAgent) was
            # permanently removed from the live decision path on v2 contract finalization.
            # Sentiment is now an input dim (ctx_sentiment_fear_greed) inside feature_v2,
            # not a separate ensemble vote. See AI_MODEL_REGISTRY in ai_decision_contract.
            model_probs = rf_probs

            # Pick prediction from per-coin model probs (authoritative)
            probs_list = [model_probs.get("sell", 0), model_probs.get("hold", 0), model_probs.get("buy", 0)]
            prediction_idx = int(np.argmax(probs_list))
            prediction = ("SELL", "HOLD", "BUY")[prediction_idx]
            confidence = float(probs_list[prediction_idx])

            signal_decision_id = f"{strategy_id}_{symbol}_{int(time.time() * 1000)}"

            # buy_margin computed for telemetry only — does NOT gate direction
            buy_margin = compute_buy_margin(model_probs)
            winner_probability_raw = ConfidenceNormalizer.normalize(float(confidence))
            prediction_pre_margin = prediction

            # CANONICAL AI INPUT: ai_market_context multiplier (POST-MODEL SANITY NUDGE).
            # On v2 the bulk of MTF + market context is INSIDE the model input vector
            # (CONTEXT_DIMS_V2). The remaining post-model multiplier is a small capped
            # nudge (CTX_TOTAL_CAP=0.10) for v1 backward-compat and as a redundancy
            # check on v2. Already fetched above for feature build.
            winner_probability = max(0.0, min(1.0, winner_probability_raw * ctx_multiplier))

            logger.info(
                " Model prediction for %s: %s (winner_prob=%.3f buy_margin=%.4f argmax_was=%s)",
                symbol,
                prediction,
                winner_probability,
                buy_margin,
                prediction_pre_margin,
            )

            # MODEL row must reflect emitted signal (post-normalize + admission floor), not pre-threshold argmax
            model_decision = {
                "decision_id": signal_decision_id,
                "symbol": symbol,
                "timestamp": time.time(),
                "stage": "MODEL",
                "model_class": prediction,
                "model_probs": model_probs,
                "gate_result": None,
                "gate_reason": GateReason.MODEL,
                "execution_result": None,
                "execution_reason": None,
            }
            await self._store_pipeline_decision(model_decision)

            # Extract ranking indicators for portfolio engine (trend_score, chop_score, vol_penalty)
            _rank_in = ranking_source if len(ranking_source) >= 20 else market_primary
            ranking = self._extract_ranking_indicators(_rank_in)

            # Live candle-shape from native 1m (separate clock from ranking_tf ADX/RSI).
            # Upper/lower wick fractions feed explainability + decision_data so DAY
            # actually reads bar shape on the live path (not only in diagnostics HTF helpers).
            candle_upper_wick_pct = 0.0
            candle_lower_wick_pct = 0.0
            candle_body_pct = 0.0
            candle_shape_tf = "1m" if sid0 == "day" else ranking_tf_label
            _shape_bars = market_1m_exec if sid0 == "day" and market_1m_exec else (_rank_in or market_primary)
            try:
                if _shape_bars and len(_shape_bars) >= 1:
                    _b = _shape_bars[-1]
                    # row = [ts, o, h, l, c, v]
                    _o = float(_b[1]); _h = float(_b[2]); _l = float(_b[3]); _c = float(_b[4])
                    _rng = _h - _l
                    if _rng > 0:
                        candle_upper_wick_pct = (_h - max(_c, _o)) / _rng
                        candle_lower_wick_pct = (min(_c, _o) - _l) / _rng
                        candle_body_pct = abs(_c - _o) / _rng
            except Exception:
                candle_upper_wick_pct = candle_lower_wick_pct = candle_body_pct = 0.0

            # Compute spread_penalty from live bid/ask data in Redis
            spread_pct_raw = 0.0
            try:
                if self.redis:
                    from backend.services.ai_decision_contract import REDIS_KEY_AI_CONTEXT
                    from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                    base = CanonicalSymbolFormatter.to_base(symbol)
                    mkt_json = await self.redis.get(f"market:{base}")
                    if mkt_json:
                        mkt = json.loads(mkt_json)
                        sp = mkt.get("spread_pct")
                        if sp is not None:
                            spread_pct_raw = float(sp)
                    if spread_pct_raw == 0.0:
                        ctx_key = REDIS_KEY_AI_CONTEXT.format(symbol=symbol)
                        ctx_data = await self.redis.hgetall(ctx_key)
                        if ctx_data:
                            ctx_sp = ctx_data.get("ctx_spread_pct") or ctx_data.get(b"ctx_spread_pct")
                            if ctx_sp is not None:
                                spread_pct_raw = float(ctx_sp if isinstance(ctx_sp, str) else ctx_sp.decode())
                    if spread_pct_raw == 0.0:
                        feat_data = await self.redis.hgetall(f"feature:{symbol}")
                        if feat_data:
                            sbp = feat_data.get("spread_bp") or feat_data.get(b"spread_bp")
                            if sbp:
                                spread_pct_raw = float(sbp if isinstance(sbp, str) else sbp.decode()) / 100.0
            except Exception:
                pass
            ranking["spread_penalty"] = min(spread_pct_raw / 0.25, 1.0) if spread_pct_raw > 0 else 0.0

            regime_label, regime_score_val = "sideways", 0.0
            regime_snapshot_label, regime_snapshot_score = "", 0.0
            if REGIME_SCORE_AVAILABLE and get_regime_snapshot_for_signal is not None:
                try:
                    snap = await get_regime_snapshot_for_signal(self.redis)
                    regime_snapshot_label = str(snap.get("regime_label", "sideways"))
                    regime_snapshot_score = float(snap.get("regime_score", 0.0))
                    regime_label = regime_snapshot_label
                    regime_score_val = regime_snapshot_score
                except Exception as re_err:
                    logger.debug("Regime snapshot for signal failed: %s", re_err)
            # Prefer per-coin AI market context so Redis signal regime matches
            # what the model already consumed in the 145-dim feature vector.
            ctx_regime = str(ctx_payload.get("ctx_market_regime") or "").strip().lower()
            if ctx_regime and ctx_regime not in ("unknown", "none", ""):
                if ctx_regime in ("trending_up", "bull", "uptrend"):
                    regime_label, regime_score_val = "bull", 0.5
                elif ctx_regime in ("trending_down", "bear", "downtrend"):
                    regime_label, regime_score_val = "bear", -0.5
                elif ctx_regime in ("chop", "ranging", "range", "sideways"):
                    regime_label, regime_score_val = "sideways", 0.0
                else:
                    regime_label = ctx_regime

            # Store signal in Redis
            # Canonical "side" field - use signal_action for single source of truth
            from backend.services.signal_action import normalize_signal_action

            canonical_side = normalize_signal_action(prediction)
            pb = float(model_probs.get("buy", 0.0))
            ph = float(model_probs.get("hold", 0.0))
            ps = float(model_probs.get("sell", 0.0))
            artifact_path = self.model_artifact_paths.get(slot, "unknown")
            artifact_sha = str(self.model_artifact_sha256.get(slot, "") or "").strip()
            if artifact_path and artifact_path not in ("unknown", "none"):
                try:
                    latest_sha = str(sha256_file(Path(artifact_path)) or "").strip()
                    if latest_sha and latest_sha != artifact_sha:
                        old_sha = artifact_sha
                        self.model_artifact_sha256[slot] = latest_sha
                        artifact_sha = latest_sha
                        logger.info(
                            "ARTIFACT_HASH_REFRESH: slot=%s symbol=%s old=%s new=%s",
                            slot,
                            symbol,
                            old_sha[:12] if old_sha else "",
                            latest_sha[:12],
                        )
                except Exception as hash_e:
                    logger.debug("ARTIFACT_HASH_REFRESH skipped for %s: %s", slot, hash_e)

            signal_data = {
                "symbol": symbol,
                "confidence": str(winner_probability),
                "winner_probability": str(winner_probability),
                "winner_probability_raw": str(winner_probability_raw),
                "buy_margin": str(buy_margin),
                "argmax_action": str(prediction_pre_margin),
                "timestamp": time.time(),
                "side": canonical_side,
                "prediction": prediction,
                "model_used": True,
                "model_artifact_path": artifact_path,
                "live_ai_strategy": strategy_id,
                "signal_source": "per_coin_ml",
                "features_count": len(features),
                "decision_id": signal_decision_id,
                "prob_buy": str(pb),
                "prob_hold": str(ph),
                "prob_sell": str(ps),
                "atr": ranking["atr"],
                "rsi": ranking["rsi"],
                "adx": ranking["adx"],
                "ema_alignment": ranking["ema_alignment"],
                "price_momentum": ranking["price_momentum"],
                "spread_penalty": ranking["spread_penalty"],
                "spread_pct": spread_pct_raw,
                "regime": regime_label,
                "regime_label": regime_label,
                "regime_score": regime_score_val,
                "regime_snapshot_label": regime_snapshot_label,
                "regime_snapshot_score": str(regime_snapshot_score),
                # Canonical AI market context inputs (real, not telemetry)
                "ctx_multiplier": str(float(ctx_multiplier)),
                "ctx_change_24h_pct": ctx_payload.get("ctx_change_24h_pct", "0.0"),
                "ctx_volume_24h_usd": ctx_payload.get("ctx_volume_24h_usd", "0.0"),
                "ctx_relative_volume": ctx_payload.get("ctx_relative_volume", "0.0"),
                "ctx_liquidity_tier": ctx_payload.get("ctx_liquidity_tier", "0"),
                "ctx_depth_imbalance": ctx_payload.get("ctx_depth_imbalance", "0.0"),
                "ctx_rs_btc": ctx_payload.get("ctx_rs_btc", "0.0"),
                "ctx_rs_eth": ctx_payload.get("ctx_rs_eth", "0.0"),
                "ctx_btc_dominance_proxy": ctx_payload.get("ctx_btc_dominance_proxy", "0.0"),
                "ctx_market_regime": ctx_payload.get("ctx_market_regime", "unknown"),
                "ctx_sentiment_fear_greed": ctx_payload.get("ctx_sentiment_fear_greed", "0.0"),
                "feature_version": str(feat_version),
                "primary_signal_bar_seconds": str(int(day_label_grid_seconds()) if sid0 == "day" else int(primary_bar_seconds_for_strategy(strategy_id))),
                # RF features (adx/rsi/ema_alignment/momentum below) come from ranking_tf,
                # NOT necessarily the same clock as the model's native feature vector (DAY
                # feature_builder runs on native 1m). Kept explicit so explainability/audits
                # never have to assume which candle timeframe produced these numbers.
                "ranking_tf": ranking_tf_label,
                "candle_shape_tf": candle_shape_tf,
                "candle_upper_wick_pct": str(round(candle_upper_wick_pct, 6)),
                "candle_lower_wick_pct": str(round(candle_lower_wick_pct, 6)),
                "candle_body_pct": str(round(candle_body_pct, 6)),
                "ai_clock_contract": ("day_htf_v5_1m_ctx10tf" if sid0 == "day" else ("v3" if int(feat_version) >= 3 else "v2")),
                "day_htf_contract": "1m_native+ctx_1m_5m_15m_30m_1h_4h_8h_12h_1d_1w" if sid0 == "day" else "",
                "day_htf_bars_json": (json.dumps(day_tf_audit, separators=(",", ":")) if sid0 == "day" and day_tf_audit else ""),
                "day_label_grid_seconds": (str(int(day_label_grid_seconds())) if sid0 == "day" else ""),
                "artifact_sha256": artifact_sha,
                "feature_dim": str(expected_dim),
                "ctx_ts_utc": ctx_ts_str,
                "ctx_age_sec": str(round(ctx_age_sec, 4)),
                "context_fresh": "1" if ctx_fresh else "0",
                "content_fresh": "1",
                "signal_content_stale": "0",
                "content_age_sec": "0",
                "ttl_preserve_skip_reason": "",
                "context_audit_emit": json.dumps(ctx_defaulted_audit, separators=(",", ":")),
            }
            if sid0 == "day" and feature_health_sidecar:
                from backend.services.day_feature_health import sidecar_redis_fields

                signal_data.update(sidecar_redis_fields(feature_health_sidecar))
            try:
                await self._log_inference(
                    decision_id=signal_decision_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    prediction=prediction,
                    argmax_action=prediction_pre_margin,
                    pb=pb,
                    ph=ph,
                    ps=ps,
                    winner_raw=float(winner_probability_raw),
                    winner_final=float(winner_probability),
                    buy_margin=float(buy_margin),
                    ctx_multiplier=float(ctx_multiplier),
                    ctx_audit=ctx_audit,
                    feature_version=int(feat_version),
                    features=features,
                    model_slot=slot,
                )
            except Exception as log_e:
                logger.debug("AI_INFER_LOG: %s skipped: %s", symbol, log_e)

            if self.redis:
                key = redis_ai_signal_key(strategy_id, symbol)
                # Add writer metadata for single-writer enforcement
                enhanced_signal_data = create_writer_payload(WRITER_ROLES["AI_SIGNALS"], signal_data)
                # Atomic pipeline: hmset + expire in one round-trip
                async with self.redis.pipeline(transaction=True) as pipe:
                    pipe.hmset(key, {field: str(value) for field, value in enhanced_signal_data.items()})
                    pipe.expire(key, AI_SIGNAL_REDIS_TTL_SEC)
                    await pipe.execute()

                if sid0 == "day":
                    logger.info(
                        "DAY_HTF_CANONICAL_REDIS fv=%s dim=%s bars_json=%s clock=%s",
                        signal_data.get("feature_version"),
                        signal_data.get("feature_dim"),
                        signal_data.get("day_htf_bars_json"),
                        signal_data.get("ai_clock_contract"),
                    )

                logger.info(
                    " %s SIGNAL GENERATED: %s -> %s (side: %s, winner_prob: %.3f, buy_margin: %.4f, model_used: %s)",
                    c.attribution_log_prefix,
                    symbol,
                    prediction,
                    canonical_side,
                    winner_probability,
                    buy_margin,
                    signal_data["model_used"],
                )

                # SAVE TO DATABASE FOR HISTORICAL ANALYSIS - LIVE PRODUCTION PERSISTENCE
                try:
                    await self._save_signal_to_database(symbol, canonical_side, winner_probability, signal_data)
                except Exception as db_e:
                    logger.exception(f"DATABASE SAVE FAILED: {symbol} - {db_e}")

                try:
                    await insert_audit_row_async(
                        event_type=EVT_SIGNAL_EMITTED,
                        decision_id=signal_decision_id,
                        strategy_id=strategy_id,
                        symbol=symbol,
                        redis_signal_key=redis_ai_signal_key(strategy_id, symbol),
                        artifact_path=artifact_path,
                        artifact_sha256=artifact_sha,
                        feature_version=int(feat_version),
                        feature_dim=int(expected_dim),
                        context_fresh=ctx_fresh,
                        context_age_sec=float(ctx_age_eff),
                        context_defaulted_json=ctx_defaulted_audit,
                        extra_json={"ctx_audit_components": ctx_audit} if ctx_audit else None,
                    )
                except Exception as aud_e:
                    logger.debug("STRATEGY_RUNTIME_AUDIT emit row skipped: %s", aud_e)

        except Exception as e:
            logger.exception(f"Error generating AI signal for {symbol}: {e}")
            await self._preserve_existing_signal_ttl(strategy_id, symbol)

    async def _store_pipeline_decision(self, decision: dict[str, Any]) -> None:
        """Store pipeline decision for verification tracking"""
        try:
            import asyncio

            def _db_operation():
                from backend.database_schema import DATABASE_PATH

                conn = connect_rw(DATABASE_PATH)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    cursor = conn.cursor()

                    # Table should already exist from startup initialization

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO pipeline_decisions
                        (decision_id, symbol, timestamp, stage, model_class, model_probs_json,
                         gate_result, gate_reason, execution_result, execution_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            decision["decision_id"],
                            decision["symbol"],
                            decision["timestamp"],
                            decision["stage"],
                            decision["model_class"],
                            json.dumps(decision["model_probs"]),
                            decision["gate_result"],
                            decision["gate_reason"],
                            decision["execution_result"],
                            decision["execution_reason"],
                        ),
                    )

                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(run_locked_retry, _db_operation)

        except Exception as e:
            logger.warning(f"Failed to store pipeline decision: {e}")

    async def _save_signal_to_database(self, symbol: str, side: str, confidence: float, _signal_data: dict[str, Any]) -> None:
        """Save AI signal to database for historical analysis - LIVE PRODUCTION"""
        try:
            logger.info(f"DB SAVE ATTEMPT: {symbol} -> {side.upper()} (confidence: {confidence:.3f})")

            # Use asyncio.to_thread for Python 3.9+ (simpler than run_in_executor)
            import asyncio

            def _db_operation():
                global _signal_table_created
                from backend.database_schema import DATABASE_PATH

                conn = connect_rw(DATABASE_PATH)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    cursor = conn.cursor()

                    # Create table only once per process
                    if not _signal_table_created:
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS ai_live_signals (
                                id INTEGER PRIMARY KEY,
                                symbol VARCHAR(50),
                                side VARCHAR(8) NOT NULL CHECK (side IN ('buy','sell','hold')),
                                reason VARCHAR(512),
                                price REAL,
                                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                consumed BOOLEAN DEFAULT 0
                            )
                        """)
                        cursor.execute("CREATE INDEX IF NOT EXISTS ix_ai_live_signals_symbol ON ai_live_signals(symbol)")
                        cursor.execute("CREATE INDEX IF NOT EXISTS ix_ai_live_signals_created_at ON ai_live_signals(created_at)")
                        _signal_table_created = True

                    # Insert signal into database
                    created_at = datetime.now(timezone.utc).isoformat()
                    # Every generator/fallback row includes decision_id; persist canonical tail for DB fallback.
                    if isinstance(_signal_data, dict) and _signal_data.get("decision_id"):
                        amx = str(_signal_data.get("argmax_action") or "")
                        bm = str(_signal_data.get("buy_margin") or "")
                        cf = float(confidence)
                        reason = f"AI | argmax={amx} buy_margin={bm} winner_prob={cf:.4f} Confidence: {cf:.4f}"
                    else:
                        reason = f"AI Model Prediction - Confidence: {float(confidence):.4f}"

                    cursor.execute(
                        """
                        INSERT INTO ai_live_signals (symbol, side, reason, price, created_at, consumed)
                        VALUES (?, ?, ?, ?, ?, 0)
                    """,
                        (
                            symbol,
                            side,
                            reason,
                            None,  # price for now
                            created_at,
                        ),
                    )

                    conn.commit()
                    logger.info(f"SUCCESS: Saved signal to DB: {symbol} -> {side.upper()}")
                except Exception as e:
                    logger.exception(f"DB ERROR in thread: {e}")
                    return False
                else:
                    return True
                finally:
                    conn.close()

            # Run in thread
            result = await asyncio.to_thread(run_locked_retry, _db_operation)

            if result:
                logger.info(f"SIGNAL SAVED TO DB: {symbol} -> {side.upper()} (confidence: {confidence:.3f})")
            else:
                logger.error(f"Failed to save signal to database: {symbol}")

        except Exception as e:
            logger.exception(f"Error saving signal to database: {e}")

    async def _ensure_signal_table_exists(self) -> None:
        """Ensure ai_live_signals table exists - LIVE PRODUCTION"""
        # Table creation now handled in _save_signal_to_database to avoid async issues
        pass

    async def _generate_fallback_signal(self, symbol: str) -> None:
        """Hard-disabled (non-canonical). Only per-strategy RF artifacts emit signals."""
        logger.warning("FALLBACK_DISABLED: %s — no momentum/generic fallback", symbol)

    async def _get_market_data(self, symbol: str, strategy_id: str) -> tuple[list[list], list[list]]:
        """
        Load 1m OHLCV (ingest + execution tail), resample to the strategy **primary** clock
        (15m day). Returns ``(ohlcv_primary, ohlcv_1m_exec_tail)``.
        """
        from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy
        from backend.services.ohlcv_resample import resample_ohlcv_to_seconds

        strat = (strategy_id or "day").strip().lower()
        min_bars = min_ohlcv_bars_for_signal()
        fetch_limit = ohlcv_1m_fetch_limit_for_primary(strat)
        ccxt_symbol = _to_ccxt_symbol(symbol)

        exchange_bars: list[list] = []
        try:
            if live_market_data_service:
                ex = await live_market_data_service.get_ohlcv(ccxt_symbol, "1m", fetch_limit)
                if isinstance(ex, list) and ex:
                    exchange_bars = ex
        except Exception:
            logger.debug("Live OHLCV unavailable for %s", symbol, exc_info=True)

        if len(exchange_bars) >= min_bars:
            psec = primary_bar_seconds_for_strategy(strat)
            prim = resample_ohlcv_to_seconds(exchange_bars, psec)
            return prim, exchange_bars[-min(200, len(exchange_bars)) :]

        store_bars: list[list] = []
        if feature_store_ohlcv_fallback_enabled():
            try:
                from backend.services.feature_store import get_ohlcv_recent
                from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                dash = f"{CanonicalSymbolFormatter.to_base(symbol)}-USDT"
                rows = await asyncio.to_thread(get_ohlcv_recent, dash, "1m", fetch_limit)
                store_bars = feature_store_rows_to_ohlcv(rows)
            except Exception:
                logger.debug("Feature-store OHLCV fallback failed for %s", symbol, exc_info=True)

        if len(store_bars) >= min_bars:
            psec = primary_bar_seconds_for_strategy(strat)
            prim = resample_ohlcv_to_seconds(store_bars, psec)
            return prim, store_bars[-min(200, len(store_bars)) :]

        best = exchange_bars if len(exchange_bars) >= len(store_bars) else store_bars

        if not best:
            logger.warning(
                "OHLCV_MISSING_FAIL_CLOSED: no live market data available for %s (exchange+store empty); refusing to synthesize candles",
                symbol,
            )
            return [], []
        psec = primary_bar_seconds_for_strategy(strat)
        prim = resample_ohlcv_to_seconds(best, psec)
        return prim, best[-min(200, len(best)) :]

    def _resolve_model_bus_symbol(self, symbol: str) -> str | None:
        """Map symbol to BUS form if a per-coin slot exists for this symbol (any enabled strategy)."""
        try:
            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

            bus = CanonicalSymbolFormatter.to_exchange(symbol)
        except Exception:
            u = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
            bus = u if u.endswith("USDT") else f"{u}USDT"
        for sid in self.enabled_strategies:
            if self._slot(sid, bus) in self.models:
                return bus
        return None

    async def ensure_ml_inference_ready(self) -> bool:
        """Load models from disk if needed (no writer lock; safe for read-only inference hooks)."""
        if self.models:
            return True
        if self.redis is None:
            self.redis = get_shared_redis_async()
        try:
            await self._initialize_models()
        except Exception as e:
            logger.warning("ensure_ml_inference_ready: model init failed: %s", e)
            return False
        return bool(self.models)

    async def predict_rf_probs_124(self, symbol: str, strategy_id: str | None = None) -> dict[str, float] | None:
        """
        RF-only class probabilities using the **same** feature vector shape as the loaded artifact
        (124 v1 or 145 v2 — matches per-slot scaler). Does not run chart/LSTM ensemble.
        """
        if not await self.ensure_ml_inference_ready():
            return None
        sid = (strategy_id or (self.enabled_strategies[0] if self.enabled_strategies else "day")).strip().lower()
        try:
            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

            bus = CanonicalSymbolFormatter.to_exchange(symbol)
        except Exception:
            u = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
            bus = u if u.endswith("USDT") else f"{u}USDT"
        slot = self._slot(sid, bus)
        model = self.models.get(slot)
        scaler = self.scalers.get(slot)
        if model is None or scaler is None:
            return None
        feat_ver = int(self.model_feature_versions.get(slot, 1))
        expected_dim = int(self.model_feature_dims.get(slot, self.feature_count))
        if sid == "day":
            ccxt_sym = _to_ccxt_symbol(bus)
            if not live_market_data_service:
                return None
            bundle = await async_fetch_day_active_ohlcv_bundle(
                live_market_data_service,
                ccxt_sym,
                force_refresh=True,
            )
            ok_bundle, missing_b = validate_day_active_bundle(bundle)
            if not ok_bundle:
                logger.debug(
                    "predict_rf_probs_124: insufficient day ACTIVE OHLCV for %s (%s)",
                    bus,
                    missing_b,
                )
                return None
            from backend.services.ai_market_context import hydrate_ai_context_payload

            ctx_payload, _, _ = await self._read_ai_context(bus)
            ctx_payload = await hydrate_ai_context_payload(bus, ctx_payload)
            features, _health = await self._assemble_day_live_features(ccxt_sym, bus, bundle, ctx_payload)
        else:
            m_primary, m_1m = await self._get_market_data(bus, sid)
            min_b = min_primary_bars_for_strategy(sid)
            if not m_primary or len(m_primary) < min_b:
                logger.debug("predict_rf_probs_124: insufficient primary OHLCV for %s", bus)
                return None
            base = await self._calculate_features(m_primary, m_1m, bus, sid)
            if len(base) != self.feature_count:
                logger.debug("predict_rf_probs_124: base feature len mismatch for %s", bus)
                return None
            if feat_ver >= 2 and expected_dim == 145:
                from backend.services.ai_feature_v2 import context_vector_from_ai_context
                from backend.services.ai_market_context import hydrate_ai_context_payload

                ctx_payload, _, _ = await self._read_ai_context(bus)
                ctx_payload = await hydrate_ai_context_payload(bus, ctx_payload)
                ctx_part = context_vector_from_ai_context(ctx_payload)
                features = list(base)[:AI_FEATURE_DIM_V1] + list(ctx_part)
            else:
                features = list(base)[:AI_FEATURE_DIM_V1]
        if len(features) != expected_dim:
            logger.debug("predict_rf_probs_124: dim mismatch for %s got=%s want=%s", bus, len(features), expected_dim)
            return None
        try:
            X = scaler.transform([features])
            proba = model.predict_proba(X)[0]
        except Exception as e:
            logger.debug("predict_rf_probs_124: predict failed for %s: %s", bus, e)
            return None
        n = len(proba)
        if n == 2:
            return {"sell": 0.0, "hold": float(proba[0]), "buy": float(proba[1])}
        return {"sell": float(proba[0]), "hold": float(proba[1]), "buy": float(proba[2])}

    async def _calculate_features(
        self,
        ohlcv_primary: list[list],
        ohlcv_exec_1m: list[list],
        symbol: str,
        strategy_id: str,
    ) -> list[float]:
        """124-dim base on **primary-clock** OHLCV (v3); 1m tail reserved for execution-side overlays."""
        from backend.services.feature_builder import build_feature_vector_124

        min_b = min_primary_bars_for_strategy(strategy_id)
        if not ohlcv_primary or len(ohlcv_primary) < min_b:
            return [0.0] * self.feature_count

        try:
            # Orderbook features from Redis
            orderbook = None
            try:
                from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                base = CanonicalSymbolFormatter.to_base(symbol)
                ob_key = f"orderbook:{base}"
            except Exception:
                ob_key = f"orderbook:{symbol}"

            if self.redis:
                raw_ob = await self.redis.hgetall(ob_key)
                if raw_ob:
                    from backend.services.order_book_service import parse_orderbook_redis_hash

                    orderbook, _ob_age = parse_orderbook_redis_hash(raw_ob)

            sentiment: dict[str, Any] | None = None
            ctx_for_overlay: dict[str, Any] = {}
            if self.redis:
                with contextlib.suppress(TypeError, ValueError, AttributeError):
                    raw_ctx_h = await self.redis.hgetall(REDIS_KEY_AI_CONTEXT.format(symbol=symbol))
                    if raw_ctx_h:
                        ctx_for_overlay = {k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else v) for k, v in raw_ctx_h.items()}
                with contextlib.suppress(TypeError, ValueError, AttributeError):
                    raw_s = await self.redis.get(REDIS_KEY_AI_SENTIMENT)
                    if raw_s:
                        sdec = raw_s.decode() if isinstance(raw_s, bytes) else raw_s
                        sentiment = {"fear_greed_index": float(sdec)}
                # Fallback: canonical ctx hash carries ctx_sentiment_fear_greed when dedicated key unset.
                if sentiment is None or sentiment.get("fear_greed_index", 0.0) == 0.0:
                    raw_fg = ctx_for_overlay.get("ctx_sentiment_fear_greed")
                    if raw_fg is not None and str(raw_fg).strip() != "":
                        sentiment = sentiment or {}
                        sentiment["fear_greed_index"] = float(raw_fg)

            # Canonical fundamentals (81-90 / env / Redis news+social / CoinGecko / ctx / ATR proxy)
            try:
                from backend.services.ai_feature_fundamentals import merge_canonical_sentiment_payload
                from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                sentiment = await merge_canonical_sentiment_payload(
                    base_symbol=CanonicalSymbolFormatter.to_base(symbol),
                    pair_symbol=symbol,
                    ctx_for_overlay=ctx_for_overlay if isinstance(ctx_for_overlay, dict) else None,
                    redis_client=self.redis,
                    ohlcv_1m=ohlcv_primary,
                    existing=sentiment,
                )
            except Exception as fund_e:
                logger.debug("merge_canonical_sentiment_payload skipped: %s", fund_e)

            use_live_ob = orderbook is None or (isinstance(orderbook, dict) and not orderbook)
            if isinstance(orderbook, dict) and orderbook:
                sp = float(orderbook.get("bid_ask_spread", 0.0) or 0.0)
                md = float(orderbook.get("market_depth", 0.0) or 0.0)
                if sp < 1e-12 and md < 1e-12:
                    use_live_ob = True
            if use_live_ob:
                with contextlib.suppress(Exception):
                    from backend.services.order_book_service import fetch_order_book_features_live

                    live_ob = await fetch_order_book_features_live(_to_ccxt_symbol(symbol))
                    if live_ob:
                        orderbook = live_ob

            # Volume profile features from Redis
            volume_profile = None
            try:
                from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                base = CanonicalSymbolFormatter.to_base(symbol)
                vp_key = f"volume_profile:{base}"
            except Exception:
                vp_key = f"volume_profile:{symbol}"

            if self.redis:
                raw_vp = await self.redis.hgetall(vp_key)
                if raw_vp:
                    volume_profile = {k.decode() if isinstance(k, bytes) else k: float(v.decode() if isinstance(v, bytes) else v) for k, v in raw_vp.items()}

            # Sentiment block (124 slots 81-90): merged via ai_feature_fundamentals + Redis F&G above.

            # Convert symbol to ccxt format
            ccxt_symbol = _to_ccxt_symbol(symbol)

            ohlcv_1d: list[list] | None = None
            with contextlib.suppress(Exception):
                from backend.services.day_active_market_bundle import async_read_cached_day_active_bundle

                cached = await async_read_cached_day_active_bundle(ccxt_symbol)
                if isinstance(cached, dict):
                    d1_rows = cached.get("1d")
                    if isinstance(d1_rows, list) and len(d1_rows) >= 2:
                        ohlcv_1d = d1_rows
            if ohlcv_1d is None and live_market_data_service:
                with contextlib.suppress(Exception):
                    d1 = await live_market_data_service.get_ohlcv(ccxt_symbol, "1d", 40)
                    if isinstance(d1, list) and len(d1) >= 2:
                        ohlcv_1d = d1

            if len(ohlcv_exec_1m or []) < 5:
                logger.debug("Short 1m execution tail for %s strat=%s (non-fatal)", symbol, strategy_id)
            return build_feature_vector_124(
                symbol_ccxt=ccxt_symbol,
                ohlcv=ohlcv_primary,
                volume_profile=volume_profile,
                orderbook=orderbook,
                ohlcv_1d=ohlcv_1d,
                sentiment=sentiment,
            )

        except Exception as e:
            logger.exception(f"Error calculating features: {e}")
            return [0.0] * self.feature_count

    def _calculate_change(self, prices: np.ndarray, periods: int) -> float:
        """Calculate price change over periods"""
        if len(prices) < periods + 1:
            return 0.0
        return (prices[-1] - prices[-periods - 1]) / prices[-periods - 1]

    @staticmethod
    def _np_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """True Range ATR using Wilder smoothing (numpy, no talib)."""
        if len(close) < period + 1:
            return 0.0
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - close[:-1])
        lc = np.abs(low[1:] - close[:-1])
        tr = np.maximum(hl, np.maximum(hc, lc))
        atr = float(np.mean(tr[:period]))
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + float(tr[i])) / period
        return atr

    @staticmethod
    def _np_rsi(close: np.ndarray, period: int = 14) -> float:
        """Wilder-smoothed RSI (numpy, no talib)."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.maximum(deltas, 0.0)
        losses = np.maximum(-deltas, 0.0)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
        if avg_loss < 1e-15:
            return 100.0 if avg_gain > 1e-15 else 50.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _np_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Simplified ADX using numpy. Returns 0-100 scale."""
        n = len(close)
        if n < period * 2 + 1:
            return 25.0
        up = np.diff(high)
        down = -np.diff(low)
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - close[:-1])
        lc = np.abs(low[1:] - close[:-1])
        tr = np.maximum(hl, np.maximum(hc, lc))
        atr_s = float(np.mean(tr[:period]))
        plus_s = float(np.mean(plus_dm[:period]))
        minus_s = float(np.mean(minus_dm[:period]))
        dx_vals = []
        for i in range(period, len(tr)):
            atr_s = (atr_s * (period - 1) + float(tr[i])) / period
            plus_s = (plus_s * (period - 1) + float(plus_dm[i])) / period
            minus_s = (minus_s * (period - 1) + float(minus_dm[i])) / period
            if atr_s > 1e-15:
                plus_di = 100.0 * plus_s / atr_s
                minus_di = 100.0 * minus_s / atr_s
                di_sum = plus_di + minus_di
                dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 1e-15 else 0.0
                dx_vals.append(dx)
        if len(dx_vals) < period:
            return 25.0
        adx = float(np.mean(dx_vals[:period]))
        for i in range(period, len(dx_vals)):
            adx = (adx * (period - 1) + dx_vals[i]) / period
        return float(np.clip(adx, 0.0, 100.0))

    def _extract_ranking_indicators(self, market_data: list[list]) -> dict[str, float]:
        """Extract atr, rsi, adx, ema_alignment, price_momentum for portfolio engine ranking."""
        out = {"atr": 0.0, "rsi": 50.0, "adx": 25.0, "ema_alignment": 0.5, "price_momentum": 0.0, "spread_penalty": 0.0}
        if len(market_data) < 20:
            return out
        try:
            df = pd.DataFrame(market_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            close = df["close"].to_numpy(dtype=np.float64)
            high = df["high"].to_numpy(dtype=np.float64)
            low = df["low"].to_numpy(dtype=np.float64)
            if talib is not None:
                atr_arr = talib.ATR(high, low, close, timeperiod=14)
                atr_v = float(atr_arr[-1]) if atr_arr is not None and len(atr_arr) > 0 else 0.0
                out["atr"] = atr_v if math.isfinite(atr_v) else 0.0
                rsi_arr = talib.RSI(close, timeperiod=14)
                rsi_v = float(rsi_arr[-1]) if rsi_arr is not None and len(rsi_arr) > 0 else 50.0
                out["rsi"] = rsi_v if math.isfinite(rsi_v) else 50.0
                adx_arr = talib.ADX(high, low, close, timeperiod=14)
                adx_v = float(adx_arr[-1]) if adx_arr is not None and len(adx_arr) > 0 else 25.0
                out["adx"] = adx_v if math.isfinite(adx_v) else 25.0
            else:
                out["atr"] = self._np_atr(high, low, close, 14)
                out["rsi"] = self._np_rsi(close, 14)
                out["adx"] = self._np_adx(high, low, close, 14)
            if len(close) >= 20:
                ema = talib.EMA(close, timeperiod=20)[-1] if talib is not None else float(np.mean(close[-20:]))
                if not math.isfinite(ema):
                    ema = float(np.mean(close[-20:]))
                out["ema_alignment"] = float(np.clip(0.5 + (close[-1] - ema) / (ema + 1e-9) * 10, 0, 1))
            if len(close) >= 6:
                out["price_momentum"] = float(self._calculate_change(close, 5) * 100)
        except Exception as e:
            logger.debug("Ranking indicators extract failed: %s", e)
        return out

    def _calculate_technical_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> list[float]:
        """Calculate technical indicators using numpy (no external dependencies)"""
        try:
            features = []

            # Moving averages
            for period in self.lookback_periods[:6]:  # 5, 10, 20, 50, 100, 200
                if len(close) >= period:
                    features.append(np.mean(close[-period:]))
                else:
                    features.append(close[-1])

            # Simple EMA approximations
            for period in [12, 26, 50]:
                if len(close) >= period:
                    # Simple exponential moving average approximation
                    alpha = 2.0 / (period + 1)
                    ema = close[-period]
                    for price in close[-period + 1 :]:
                        ema = alpha * price + (1 - alpha) * ema
                    features.append(ema)
                else:
                    features.append(close[-1])

            # RSI approximation
            if len(close) >= 14:
                gains = np.maximum(np.diff(close[-14:]), 0)
                losses = np.maximum(-np.diff(close[-14:]), 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi = 100 - (100 / (1 + rs))
                features.extend([rsi, rsi])  # RSI and RSI_14
            else:
                features.extend([50.0, 50.0])

            # Stochastic approximation
            if len(close) >= 14:
                highest = np.max(high[-14:])
                lowest = np.min(low[-14:])
                k = 100 * (close[-1] - lowest) / (highest - lowest) if highest != lowest else 50
                features.extend([k, k])  # %K and %D approximation
            else:
                features.extend([50.0, 50.0])

            # Williams %R approximation
            if len(close) >= 14:
                highest = np.max(high[-14:])
                lowest = np.min(low[-14:])
                williams_r = -100 * (highest - close[-1]) / (highest - lowest) if highest != lowest else -50
                features.append(williams_r)
            else:
                features.append(-50.0)

            # CCI approximation
            if len(close) >= 20:
                typical_price = (high[-20:] + low[-20:] + close[-20:]) / 3
                sma = np.mean(typical_price)
                mad = np.mean(np.abs(typical_price - sma))
                cci = (typical_price[-1] - sma) / (0.015 * mad) if mad != 0 else 0
                features.append(cci)
            else:
                features.append(0.0)

            # MACD approximation
            if len(close) >= 26:
                ema12 = np.mean(close[-12:])  # Simple approximation
                ema26 = np.mean(close[-26:])
                macd = ema12 - ema26
                signal = np.mean([macd])  # Simple signal line
                histogram = macd - signal
                features.extend([macd, signal, histogram])
            else:
                features.extend([0.0, 0.0, 0.0])

            # Bollinger Bands approximation
            if len(close) >= 20:
                sma = np.mean(close[-20:])
                std = np.std(close[-20:])
                upper = sma + 2 * std
                lower = sma - 2 * std
                position = (close[-1] - lower) / (upper - lower) if upper != lower else 0.5
                width = (upper - lower) / sma if sma != 0 else 0.1
                features.extend([upper, sma, lower, position, width])
            else:
                features.extend([close[-1], close[-1], close[-1], 0.5, 0.1])

            # Volume-based features (simplified)
            if len(close) >= 14:
                # OBV approximation
                price_changes = np.sign(np.diff(close[-14:]))
                obv_changes = price_changes * volume[-13:]
                obv = np.sum(obv_changes)
                features.append(obv)

                # AD approximation (Accumulation/Distribution)
                hl_diff = high[-14:] - low[-14:]
                hl_diff = np.where(hl_diff == 0, 1, hl_diff)  # Avoid division by zero
                ad = np.sum(((close[-14:] - low[-14:]) - (high[-14:] - close[-14:])) / hl_diff * volume[-14:])
                features.append(ad)

                # CMF approximation
                cmf = ad / np.sum(volume[-14:]) if np.sum(volume[-14:]) != 0 else 0
                features.append(cmf)

                # MFI approximation
                typical_prices = (high[-14:] + low[-14:] + close[-14:]) / 3
                money_flow = typical_prices * volume[-14:]
                positive_flow = np.sum(money_flow[typical_prices > np.roll(typical_prices, 1)][1:])
                negative_flow = np.sum(money_flow[typical_prices < np.roll(typical_prices, 1)][1:])
                mfr = positive_flow / negative_flow if negative_flow != 0 else 100
                mfi = 100 - (100 / (1 + mfr))
                features.append(mfi)
            else:
                features.extend([volume[-1] if len(volume) > 0 else 0, 0.0, 0.0, 50.0])

        except Exception:
            logger.exception("Error calculating technical indicators")
            return [0.0] * 24
        else:
            return features

    def _calculate_volatility_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> list[float]:
        """Calculate volatility indicators"""
        if talib is None:
            return [0.01] * 11
        try:
            features = []

            # Basic volatility
            if len(close) >= 14:
                features.append(np.std(close[-14:]) / np.mean(close[-14:]))  # volatility
            else:
                features.append(0.01)

            # ATR
            if len(close) >= 14:
                features.extend(
                    [
                        talib.ATR(high, low, close, timeperiod=14)[-1],
                        talib.NATR(high, low, close, timeperiod=14)[-1],
                    ]
                )
            else:
                features.extend([0.01, 0.01])

            # Keltner Channels (approximation)
            if len(close) >= 20:
                ema = talib.EMA(close, timeperiod=20)
                atr = talib.ATR(high, low, close, timeperiod=14)
                features.extend(
                    [
                        ema[-1] + (atr[-1] * 2),  # upper
                        ema[-1] - (atr[-1] * 2),  # lower
                    ]
                )
            else:
                features.extend([close[-1] * 1.01, close[-1] * 0.99])

            # Donchian Channels
            if len(high) >= 20:
                features.extend(
                    [
                        np.max(high[-20:]),  # upper
                        np.min(low[-20:]),  # lower
                    ]
                )
            else:
                features.extend([high[-1], low[-1]])

            # More volatility features (placeholder)
            features.extend([0.02, 0.015, 0.01, 0.005])  # ATR variations, etc.

        except Exception:
            logger.exception("Error calculating volatility indicators")
            return [0.01] * 11
        else:
            return features

    def _calculate_trend_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> list[float]:
        """Calculate trend indicators"""
        if talib is None:
            return [25.0] * 10
        try:
            features = []

            # ADX, DI+-, Aroon
            if len(close) >= 14:
                adx = talib.ADX(high, low, close, timeperiod=14)
                di_plus = talib.PLUS_DI(high, low, close, timeperiod=14)
                di_minus = talib.MINUS_DI(high, low, close, timeperiod=14)
                aroon_osc = talib.AROONOSC(high, low, timeperiod=14)

                features.extend(
                    [
                        adx[-1],
                        di_plus[-1],
                        di_minus[-1],
                        aroon_osc[-1],
                    ]
                )
            else:
                features.extend([25.0, 20.0, 18.0, 0.0])

            # Ichimoku (simplified)
            if len(close) >= 52:
                tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2
                kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2
                senkou_a = (tenkan + kijun) / 2
                senkou_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2

                features.extend([tenkan, kijun, senkou_a, senkou_b])
            else:
                features.extend([close[-1]] * 4)

            # Parabolic SAR
            if len(close) >= 2:
                psar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
                features.append(psar[-1])
            else:
                features.append(close[-1])

            # Trend strength (placeholder)
            features.append(0.5)

        except Exception:
            logger.exception("Error calculating trend indicators")
            return [25.0] * 10
        else:
            return features

    def _calculate_volume_features(self, volume: np.ndarray, close: np.ndarray) -> list[float]:
        """Calculate volume-based features"""
        if talib is None:
            return [0.0] * 8
        try:
            features = []

            # Volume MAs
            for period in [5, 10, 20]:
                if len(volume) >= period:
                    features.append(talib.SMA(volume.astype(float), timeperiod=period)[-1])
                else:
                    features.append(volume[-1])

            # Volume ratio and VPT
            if len(volume) >= 2 and len(close) >= 2:
                features.append(volume[-1] / volume[-2] if volume[-2] != 0 else 1.0)  # volume ratio
                vpt = (close[-1] - close[-2]) / close[-2] * volume[-1] if close[-2] != 0 else 0
                features.append(vpt)
            else:
                features.extend([1.0, 0.0])

            # NVI, PVI (simplified)
            features.extend([1.0, 1.0])  # Negative/PVI placeholders

            # Volume weighted price
            if len(volume) > 0 and len(close) > 0:
                vwp = np.average(close[-10:], weights=volume[-10:]) if len(close) >= 10 and np.sum(volume[-10:]) > 0 else close[-1]
                features.append(vwp)
            else:
                features.append(close[-1] if len(close) > 0 else 0.0)

        except Exception:
            logger.exception("Error calculating volume features")
            return [0.0] * 8
        else:
            return features

    def _calculate_time_features(self) -> list[float]:
        """Calculate time-based features"""
        now = datetime.now(timezone.utc)

        return [
            now.hour,  # hour
            now.weekday(),  # day_of_week
            now.day,  # day_of_month
            now.month,  # month
            now.isoweekday(),  # iso_weekday
            now.timetuple().tm_yday,  # day_of_year
            now.hour % 12,  # hour_12h
            now.minute,  # minute
            now.second,  # second
            now.hour * 3600 + now.minute * 60 + now.second,  # seconds_since_midnight
        ]

    def _calculate_advanced_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> list[float]:
        """Calculate advanced technical features"""
        try:
            features = []

            # Fibonacci retracements (simplified)
            if len(close) >= 20:
                high_20 = np.max(high[-20:])
                low_20 = np.min(low[-20:])
                range_20 = high_20 - low_20

                features.extend(
                    [
                        low_20 + range_20 * 0.236,  # fib 23.6
                        low_20 + range_20 * 0.382,  # fib 38.2
                        low_20 + range_20 * 0.618,  # fib 61.8
                    ]
                )
            else:
                features.extend([close[-1]] * 3)

            # Pivot points
            if len(close) >= 1:
                pivot = (high[-1] + low[-1] + close[-1]) / 3
                features.extend(
                    [
                        pivot,  # pivot point
                        pivot + (high[-1] - low[-1]) * 0.382,  # r1
                        pivot + (high[-1] - low[-1]) * 0.618,  # r2
                        pivot - (high[-1] - low[-1]) * 0.382,  # s1
                        pivot - (high[-1] - low[-1]) * 0.618,  # s2
                    ]
                )
            else:
                features.extend([close[-1]] * 5)

        except Exception:
            logger.exception("Error calculating advanced features")
            return [0.0] * 8
        else:
            return features

    # =========================================================================
    # OPTIMIZED: Multi-Timeframe Analysis for Better Signal Confirmation
    # =========================================================================
    async def _evaluate_multi_timeframe_coverage(self, symbol: str) -> tuple[int, int, dict[str, int]]:
        """Delegates to shared gate (same rules as portfolio-facing paths)."""
        return await evaluate_multi_timeframe_coverage(_to_ccxt_symbol(symbol))

    # =========================================================================
    # CANONICAL: AI market context integration
    # =========================================================================
    async def _read_ai_context(self, symbol: str) -> tuple[dict[str, str], float, dict[str, Any]]:
        """
        Read the canonical ai_context:{symbol} hash published by
        backend.services.ai_market_context.AIMarketContextService.

        Returns: (raw_payload_str_dict, ctx_multiplier, ctx_audit_dict)
        Falls back to multiplier=1.0 (no adjustment) when context is unavailable.
        """
        if self.redis is None:
            return {}, 1.0, {}
        try:
            raw = await self.redis.hgetall(REDIS_KEY_AI_CONTEXT.format(symbol=symbol))
            if not raw:
                return {}, 1.0, {}
            payload: dict[str, str] = {}
            for k, v in raw.items():
                ks = k.decode() if isinstance(k, bytes) else k
                vs = v.decode() if isinstance(v, bytes) else v
                payload[ks] = vs
            try:
                mult = float(payload.get("ctx_multiplier", "1.0"))
            except (TypeError, ValueError):
                mult = 1.0
            mult = max(1.0 - CTX_TOTAL_CAP, min(1.0 + CTX_TOTAL_CAP, mult))
            try:
                audit = json.loads(payload.get("ctx_audit_json", "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                audit = {}
            return payload, mult, audit
        except Exception as e:
            logger.debug("AI_CONTEXT read for %s failed: %s", symbol, e)
            return {}, 1.0, {}

    async def _log_inference(
        self,
        *,
        decision_id: str,
        strategy_id: str,
        symbol: str,
        prediction: str,
        argmax_action: str,
        pb: float,
        ph: float,
        ps: float,
        winner_raw: float,
        winner_final: float,
        buy_margin: float,
        ctx_multiplier: float,
        ctx_audit: dict[str, Any],
        feature_version: int = 1,
        features: list[float] | None = None,
        model_slot: str,
    ) -> None:
        """Persist one canonical AI inference row to ai_inference_log."""
        ensure_ai_canonical_tables()
        feats_json = json.dumps([float(x) for x in features], separators=(",", ":")) if features is not None else None

        def _do() -> None:
            from backend.database_schema import DATABASE_PATH as _DBP

            with connect_rw(_DBP) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO ai_inference_log (
                        decision_id, strategy_id, symbol, ts_utc, prediction, argmax_action,
                        prob_buy, prob_hold, prob_sell, winner_prob_raw, confidence,
                        buy_margin, ctx_multiplier, ctx_json, feature_version, feature_dim,
                        features_json, model_artifact, label_version, label_horizon_bars
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        strategy_id,
                        symbol,
                        datetime.now(timezone.utc).isoformat(),
                        prediction,
                        argmax_action,
                        float(pb),
                        float(ph),
                        float(ps),
                        float(winner_raw),
                        float(winner_final),
                        float(buy_margin),
                        float(ctx_multiplier),
                        json.dumps(ctx_audit, separators=(",", ":")),
                        int(feature_version),
                        int(len(features)) if features else (145 if int(feature_version) >= 2 else 124),
                        feats_json,
                        self.model_artifact_paths.get(model_slot, "unknown"),
                        str(self.model_label_versions.get(model_slot, "")),
                        int(self.model_label_horizons.get(model_slot, 0)),
                    ),
                )
                conn.commit()

        await asyncio.to_thread(run_locked_retry, _do)


# Singleton pattern for global access
_SignalGeneratorSingleton: RealTimeAISignalGenerator | None = None


class SignalGeneratorSingleton:
    """Singleton wrapper for RealTimeAISignalGenerator"""

    @classmethod
    def get_instance(cls) -> RealTimeAISignalGenerator:
        """Get or create singleton instance"""
        global _SignalGeneratorSingleton
        if _SignalGeneratorSingleton is None:
            _SignalGeneratorSingleton = RealTimeAISignalGenerator()
        return _SignalGeneratorSingleton


def get_signal_generator() -> RealTimeAISignalGenerator:
    """Get the global signal generator instance"""
    return SignalGeneratorSingleton.get_instance()
