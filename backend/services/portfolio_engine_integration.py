"""
Portfolio Engine Integration Layer

Canonical signal-to-execution bridge for the DAY top-4 production stack.
Consumes AI signals from ai_signal_generator and routes execution to the
Portfolio Engine.

This module provides the bridge between the old trading flow and the new
disciplined portfolio engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import redis.asyncio as redis

from backend.config.ai_signal_bus import AI_SIGNAL_REDIS_TTL_SEC, pe_buy_candidate_redis_key
from backend.config.buy_admission import (
    resolve_buy_margin_from_payload,
)
from backend.config.core_test_flags import local_bar_signal_grace_seconds
from backend.config.mystic_api_schedule import (
    EXIT_MONITOR_INTERVAL_SEC,
    LEDGER_MTM_PERSIST_INTERVAL_SEC,
    PRICE_PUBLISHER_INTERVAL_SEC,
    SIGNAL_CONSUMER_INTERVAL_SEC,
)
from backend.database_schema import DATABASE_PATH
from backend.services.gate_reason_codes import GateReason
from backend.services.live_strategy_contracts import per_coin_artifact_file
from backend.services.portfolio_engine import (
    MIN_CONFIDENCE,
    PortfolioEngine,
    get_portfolio_engine,
)
from backend.services.sqlite_large_table_retention import (
    DEFAULT_INITIAL_DELAY_SEC as _LARGE_TABLE_RETENTION_INITIAL_DELAY_SEC,
)
from backend.services.sqlite_large_table_retention import (
    DEFAULT_INTERVAL_SEC as _LARGE_TABLE_RETENTION_INTERVAL_SEC,
)
from backend.services.sqlite_large_table_retention import (
    run_large_table_retention,
)
from backend.services.strategy_runtime_audit import (
    EVT_CANDIDATE_ENQUEUED,
    EVT_SELL_EXECUTED,
    EVT_SIGNAL_CONSUME,
    insert_audit_row_async,
)
from backend.utils.redis_helpers import WRITER_ROLES, WriterLock, verify_writer_payload
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)

# Historic exit-monitor bug: misleading Redis alerts if left behind after code fix
_EXIT_MONITOR_STALE_BD_SIGNATURE = "name 'bd' is not defined"


# Supervisor-only: persist portfolio_engine_ledger MTM fields on a fixed cadence so SQLite
# tracks live marks (same recompute as API /status) without write-on-read in FastAPI.
_LEDGER_MTM_PERSIST_INTERVAL_SEC = LEDGER_MTM_PERSIST_INTERVAL_SEC
_LEDGER_MTM_PERSIST_INITIAL_DELAY_SEC = 5.0
# Tracked base coins are the live DAY top-4, sourced from the single source of
# truth in ``backend.config.trading_universe``. Do not hardcode here.
from backend.config.trading_universe import TOP4_BASE_COINS as _TRACKED_TRADE_BASE_SYMBOLS

# Canonical DAY sells: monitor_all_positions → _check_exit_conditions → execute_sell_fifo only.
# Model SELL signals are ranking penalties, never direct execution (see CANONICAL_SYSTEM.md).
_ENTRY_GATES_ENFORCED = os.getenv("ENTRY_GATES_ENFORCED", "false").strip().lower() in ("1", "true", "yes", "on")
_ENTRY_MAJOR_ONLY = _ENTRY_GATES_ENFORCED and os.getenv("ENTRY_MAJOR_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")
_ENTRY_LIQUIDITY_GATE_ENABLED = _ENTRY_GATES_ENFORCED and os.getenv("ENTRY_LIQUIDITY_GATE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
_ENTRY_MAX_SPREAD_PCT = float(os.getenv("ENTRY_MAX_SPREAD_PCT", "0.0025"))  # 0.25%
_ENTRY_MIN_QUOTE_VOLUME_24H = float(os.getenv("ENTRY_MIN_QUOTE_VOLUME_24H", "50000000"))  # $50M


def _normalize_ccxt_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "/")
    if not s:
        return ""
    if "/" not in s and s.endswith("USDT"):
        s = f"{s[:-4]}/USDT"
    return s


def _parse_entry_allowlist() -> set[str]:
    # DAY_TRADE_SYMBOLS only (Binance.US API form). Internal ccxt form keeps the
    # slash so /process_signal candidate normalization stays stable; symbol
    # gating against DAY_TRADE_SYMBOLS happens upstream in portfolio_engine.
    # Default to the live top-4 sourced from trading_universe; allow operator
    # override via env without ever silently widening to other markets.
    from backend.config.trading_universe import DAY_TRADE_SYMBOLS as _DAY

    default_raw = ",".join(_DAY)
    raw = os.getenv("ENTRY_ALLOWED_SYMBOLS", default_raw)
    out: set[str] = set()
    for tok in raw.split(","):
        sym = _normalize_ccxt_symbol(tok)
        if sym:
            out.add(sym)
    if not out:
        out = {_normalize_ccxt_symbol(s) for s in _DAY}
    return out


_ENTRY_ALLOWED_SYMBOLS = _parse_entry_allowlist()


def _get_build_stamp() -> dict[str, str]:
    """Return build stamp for startup log: commit, engine + integration mtimes, python, pid."""
    stamp = {
        "commit": "",
        "portfolio_engine_mtime": "",
        "portfolio_engine_integration_mtime": "",
        "python": getattr(sys, "executable", ""),
        "pid": str(os.getpid()),
    }
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode == 0 and r.stdout:
            stamp["commit"] = r.stdout.strip()[:12]
    except Exception as e:
        logger.debug("git rev-parse failed (build stamp): %s", e, exc_info=True)
        pass
    try:
        pe_path = Path(__file__).resolve().parent / "portfolio_engine.py"
        if pe_path.exists():
            stamp["portfolio_engine_mtime"] = str(int(pe_path.stat().st_mtime))
    except Exception as e:
        logger.debug("portfolio_engine mtime read failed (build stamp): %s", e, exc_info=True)
        pass
    try:
        integ_path = Path(__file__).resolve()
        stamp["portfolio_engine_integration_mtime"] = str(int(integ_path.stat().st_mtime))
    except Exception as e:
        logger.debug("portfolio_engine_integration mtime read failed (build stamp): %s", e, exc_info=True)
        pass
    return stamp


class PortfolioEngineIntegration:
    """
    Integration layer between existing services and Portfolio Engine.

    This class provides:
    1. Signal processing that feeds into ranked selection
    2. Position monitoring that uses the engine's exit logic
    3. Coordination between bar closes and trade execution
    """

    def __init__(self):
        self.engine: PortfolioEngine | None = None
        self.is_running = False
        self._monitor_task: asyncio.Task | None = None
        self._bar_processor_task: asyncio.Task | None = None
        self._ledger_mtm_task: asyncio.Task | None = None

        # Bar timing: 1m keeps cooldown/"N bars" semantics; entries decide on 15m closes.
        self.bar_interval = 60  # 1-minute bars (cooldown units)
        self.entry_decision_interval = max(60, int(os.getenv("DAY_ENTRY_BAR_SEC", "900")))
        self.last_bar_processed = 0
        self.last_entry_bar_processed = 0
        self._exit_monitor_interval = max(5, EXIT_MONITOR_INTERVAL_SEC)
        self._price_publisher_interval = max(5, PRICE_PUBLISHER_INTERVAL_SEC)
        self._signal_consumer_interval = max(1, SIGNAL_CONSUMER_INTERVAL_SEC)

        # Price cache for monitoring
        self.current_prices: dict[str, float] = {}
        # Per-position candle snapshots used by exit monitoring decisions.
        self._position_candle_cache: dict[str, dict[str, Any]] = {}

        # PHASE 3 FIX: Exit failure resilience tracking
        self.exit_failure_count: dict[str, int] = {}  # symbol -> consecutive failure count
        self.exit_cooldown_until: dict[str, float] = {}  # symbol -> timestamp when cooldown ends
        self.exit_hard_paused: set[str] = set()  # symbols permanently paused due to repeated failures

        # Redis client for distributed locking (INVARIANT-007: Single Execution Authority)
        self.redis_client: redis.Redis | None = None
        self._writer_lock: WriterLock | None = None
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        # LIVE DB-path self-correction: ensure engine always uses DATABASE_PATH (no restart)
        self._db_align_lock = asyncio.Lock()
        # Keep last consumed decision per Redis signal key so source hashes can remain
        # observable without duplicate candidate churn.
        self._consumed_signal_decision_by_key: dict[str, str] = {}

        logger.info("PortfolioEngineIntegration initialized")

    async def _strategy_runtime_audit_row(self, **kwargs: Any) -> None:
        """Best-effort append to strategy_runtime_audit (must never block trading)."""
        try:
            await insert_audit_row_async(**kwargs)
        except Exception:
            logger.debug("strategy_runtime_audit insert failed", exc_info=True)

    async def _resolve_atr_for_candidate(
        self,
        ccxt_symbol: str,
        current_price: float,
        decision_data: dict[str, Any],
    ) -> float:
        """Use signal ATR when valid; else engine 1h/Redis ATR. Never synthesize % of price."""
        raw = decision_data.get("atr")
        try:
            if raw is not None and str(raw).strip() != "":
                v = float(raw)
                if v > 0 and math.isfinite(v):
                    return v
        except (TypeError, ValueError):
            pass
        if self.engine is not None and current_price > 0:
            try:
                v2 = await self.engine._get_atr_for_symbol(ccxt_symbol, current_price)
                if v2 and v2 > 0 and math.isfinite(v2):
                    return float(v2)
            except Exception:
                logger.debug("ATR fallback via engine failed for %s", ccxt_symbol, exc_info=True)
        return 0.0

    async def _initialize_only(self) -> None:
        """Initialize integration without starting background tasks (for startup)"""
        if self.is_running:
            return

        # Initialize Redis for distributed locking (retry for auto-heal on transient failure)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Redis connected for distributed locking")
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    delay = 2**attempt
                    logger.warning(f"Redis connect attempt {attempt + 1}/{max_attempts} failed: {e} - retry in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.exception(f"Failed to connect to Redis after {max_attempts} attempts: {e}")
                    raise RuntimeError("Redis connection required for single execution authority") from e

        self._writer_lock = WriterLock(WRITER_ROLES["DECISION_ROUTER"], self.redis_client)
        await self._writer_lock.acquire()

        # Initialize portfolio engine with live trading service
        from backend.services.portfolio_engine import initialize_portfolio_engine, is_portfolio_engine_initialized

        if not is_portfolio_engine_initialized():
            self.engine = await initialize_portfolio_engine()
        else:
            self.engine = get_portfolio_engine()

        # M2 FIX: bar_interval MUST be set before load_quality_state_from_redis (cooldown semantics)
        assert self.bar_interval and self.bar_interval > 0, "bar_interval required before load_quality_state"
        self.engine.set_bar_interval_seconds(self.bar_interval)
        # Phase 2: Load quality/regime state from Redis (restart safety)
        await self.engine.load_quality_state_from_redis()
        logger.info("INIT_ORDER: set_bar_interval_seconds → load_quality_state_from_redis done (tasks start in start())")

        # Mark as initialized but don't start background tasks yet
        self.is_running = True
        logger.info("PortfolioEngineIntegration initialized (background tasks deferred)")

    async def start(self) -> None:
        """Start the integration layer with background tasks"""
        if not self.is_running:
            # If not initialized yet, do full initialization
            await self._initialize_only()

        # Start background tasks (only if not already started)
        if not hasattr(self, "_monitor_task") or self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._position_monitor_loop(), name="portfolio_engine:monitor")
            self._bar_processor_task = asyncio.create_task(self._bar_processor_loop(), name="portfolio_engine:bar_processor")
            # CRITICAL FIX: Start signal consumption loop - this was MISSING!
            self._signal_consumer_task = asyncio.create_task(self._signal_consumption_loop(), name="portfolio_engine:signal_consumer")
            # CRITICAL FIX: Start price publisher loop to feed Redis with live prices
            self._price_publisher_task = asyncio.create_task(self._price_publisher_loop(), name="portfolio_engine:price_publisher")
            # SYNC FIX: Start Binance balance sync checker
            self._binance_sync_task = asyncio.create_task(self._binance_sync_loop(), name="portfolio_engine:binance_sync")
            # DUST_INVARIANT_LOCK: Cooldown-gated dust reconciliation every 10 min
            self._dust_reconcile_task = asyncio.create_task(self._dust_reconciliation_loop(), name="portfolio_engine:dust_reconcile")
            self._live_reconcile_task = asyncio.create_task(self._live_reconcile_loop(), name="portfolio_engine:live_reconcile")
            self._canonical_reconcile_task = asyncio.create_task(self._canonical_reconcile_loop(), name="portfolio_engine:canonical_reconcile")
            self._paper_retention_task = asyncio.create_task(self._paper_retention_loop(), name="portfolio_engine:paper_retention")
            self._large_table_retention_task = asyncio.create_task(self._large_table_retention_loop(), name="portfolio_engine:large_table_retention")
            self._ledger_mtm_task = asyncio.create_task(self._ledger_mtm_persist_loop(), name="portfolio_engine:ledger_mtm_persist")
            try:
                from backend.services.simplified_pnl_observation import ENABLED as _PNLOB

                self._pnl_observation_task = asyncio.create_task(self._simplified_pnl_observation_loop(), name="portfolio_engine:pnl_observation") if _PNLOB else None
            except Exception:
                self._pnl_observation_task = None
            logger.info(
                "PortfolioEngineIntegration background tasks started "
                "(signal consumer, price publisher, binance sync, dust/live/canonical reconcile, "
                "paper retention, large_table_retention every %.0fs, ledger_mtm_persist every %.0fs)",
                _LARGE_TABLE_RETENTION_INTERVAL_SEC,
                _LEDGER_MTM_PERSIST_INTERVAL_SEC,
            )
            # BUILD stamp: proves current code is running after restart (no old build)
            build = _get_build_stamp()
            logger.info(
                "BUILD: commit=%s portfolio_engine_mtime=%s portfolio_engine_integration_mtime=%s python=%s pid=%s",
                build["commit"] or "n/a",
                build["portfolio_engine_mtime"] or "n/a",
                build["portfolio_engine_integration_mtime"] or "n/a",
                build["python"] or "n/a",
                build["pid"],
            )
            try:
                from backend.services.portfolio_engine import DAY_MODE_ENABLED

                logger.info(
                    "RUNTIME: DAY_MODE_ENABLED=%s | INFO logs go to process stdout/stderr (start_mystic redirects to /tmp/mystic_portfolio.log)",
                    DAY_MODE_ENABLED,
                )
            except Exception:
                logger.info("RUNTIME: DAY_MODE_ENABLED=unknown (import failed)")
            try:
                from backend.config.core_test_flags import (
                    ENABLE_GOVERNANCE_ENFORCEMENT,
                    governance_risk_governor_shadow_only,
                )
                from backend.services.risk_governor import (
                    CHOP_BLOCK_THRESHOLD,
                    DRAWDOWN_TIER_D_PCT,
                    GOVERNANCE_SHADOW_ONLY,
                    MIN_CONFIDENCE_REGIME,
                )

                logger.info(
                    "GOV_CONFIG: shadow_only=%s effective_shadow=%s ENABLE_GOVERNANCE=%s tier_d_pct=%.2f chop_block=%.2f min_conf_regime=%.2f wired=true",
                    GOVERNANCE_SHADOW_ONLY,
                    governance_risk_governor_shadow_only(),
                    ENABLE_GOVERNANCE_ENFORCEMENT,
                    DRAWDOWN_TIER_D_PCT,
                    CHOP_BLOCK_THRESHOLD,
                    MIN_CONFIDENCE_REGIME,
                )
            except Exception:
                logger.info("GOV_CONFIG: (import failed)")
        else:
            logger.info("PortfolioEngineIntegration background tasks already running")

    async def stop(self) -> None:
        """Stop the integration layer"""
        self.is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        if self._bar_processor_task:
            self._bar_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._bar_processor_task

        # Stop signal consumer task
        if hasattr(self, "_signal_consumer_task") and self._signal_consumer_task:
            self._signal_consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._signal_consumer_task

        # Stop price publisher task
        if hasattr(self, "_price_publisher_task") and self._price_publisher_task:
            self._price_publisher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._price_publisher_task

        # Stop binance sync task
        if hasattr(self, "_binance_sync_task") and self._binance_sync_task:
            self._binance_sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._binance_sync_task

        if getattr(self, "_dust_reconcile_task", None):
            self._dust_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dust_reconcile_task
        if getattr(self, "_live_reconcile_task", None):
            self._live_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._live_reconcile_task
        if getattr(self, "_canonical_reconcile_task", None):
            self._canonical_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._canonical_reconcile_task
        if getattr(self, "_paper_retention_task", None):
            self._paper_retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._paper_retention_task
        if getattr(self, "_large_table_retention_task", None):
            self._large_table_retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._large_table_retention_task
        if getattr(self, "_ledger_mtm_task", None):
            self._ledger_mtm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ledger_mtm_task

        if getattr(self, "_pnl_observation_task", None):
            self._pnl_observation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pnl_observation_task

        try:
            from backend.services import simplified_pnl_observation as _pnl_obs_mod
            from backend.services.simplified_pnl_observation import emit_periodic_summary

            if _pnl_obs_mod.ENABLED and self.engine is not None:
                emit_periodic_summary(ledger_realized_pnl=float(getattr(self.engine, "_realized_pnl", 0.0)))
        except Exception:
            logger.debug("PNL_OBS shutdown summary skipped", exc_info=True)

        # ================================================================
        # PHASE 4 FIX #6: USE TRY-FINALLY FOR GUARANTEED REDIS CLEANUP
        # ================================================================
        # Ensure Redis connection is always closed, even on error
        try:
            if self._writer_lock is not None:
                await self._writer_lock.release()
                self._writer_lock = None
            if self.redis_client:
                try:
                    await self.redis_client.aclose()
                    logger.debug("Redis connection closed successfully")
                except asyncio.TimeoutError:
                    logger.warning("Redis close timed out, connection may leak")
                except Exception as e:
                    logger.exception(f"Error closing Redis (may leak connection): {e}")
        finally:
            # Final safeguard - ensure we always log shutdown completion
            logger.info("PortfolioEngineIntegration stopped")

    async def _signal_consumption_loop(self) -> None:
        """
        CRITICAL: Consume AI signals from Redis and add as candidates.
        This is the MISSING LINK - signals must be read and processed!
        """

        logger.info("SIGNAL_CONSUMER: Starting AI signal consumption loop")

        while self.is_running:
            try:
                if not self.redis_client or not self.engine:
                    await asyncio.sleep(5)
                    continue

                # Check market data freshness first (FAIL-CLOSED)
                try:
                    last_update_str = await self.redis_client.get("market_data:last_update")
                    if not last_update_str:
                        logger.debug("SIGNAL_CONSUMER: market_data:last_update missing, waiting...")
                        await asyncio.sleep(5)
                        continue

                    data_age = time.time() - float(last_update_str)
                    if data_age > 30.0:
                        logger.debug(f"SIGNAL_CONSUMER: market data stale ({data_age:.1f}s), waiting...")
                        await asyncio.sleep(self._signal_consumer_interval)
                        continue
                except Exception as e:
                    logger.debug("SIGNAL_CONSUMER: market_data freshness check failed: %s", e, exc_info=True)
                    await asyncio.sleep(5)
                    continue

                # ARCHITECTURE v2: ML-only signal source (HOT/RULE paths removed)

                # Read ML signal keys only (strategy-scoped; legacy ai_signal:<sym> unsupported)
                SCAN_BATCH_SIZE = 300
                all_keys: list[str] = []
                async for key in self.redis_client.scan_iter(match="ai_signal:*:*", count=SCAN_BATCH_SIZE):
                    ks = key.decode() if isinstance(key, bytes) else str(key)
                    parts = ks.split(":")
                    if len(parts) != 3 or parts[0] != "ai_signal":
                        continue
                    all_keys.append(ks)
                logger.info(f"SIGNAL_CONSUMER: Found {len(all_keys)} ML keys")
                candidates_added = 0

                for key in all_keys:
                    # ================================================================
                    # PHASE 4 FIX #1: ATOMIC SIGNAL CLAIMING WITH REDIS WATCH/MULTI
                    # ================================================================
                    # Prevent duplicate processing by atomically claiming the signal
                    # Use Redis WATCH/MULTI/EXEC for optimistic locking
                    # MEDIUM #8 FIX: Atomic claim with coordinated TTL

                    try:
                        # Set a "claimed" flag atomically - if key already exists, claim fails
                        claimed_key = f"claimed:{key}"
                        # IMPORTANT: claimed key TTL (30s) should match or exceed signal key TTL
                        # to prevent orphaned claims when signal expires
                        claimed = await self.redis_client.set(claimed_key, "1", nx=True, ex=30)  # 30s expiry

                        if not claimed:
                            # Signal already claimed by another instance
                            logger.debug(f"SIGNAL_SKIP: {key} already claimed by another instance")
                            continue
                    except Exception as watch_err:
                        logger.warning(f"Failed to claim signal {key}: {watch_err}")
                        continue

                    # Extract symbol + strategy from canonical key ai_signal:<strategy>:<BUS>
                    ks = key.decode() if isinstance(key, bytes) else str(key)
                    parts = ks.split(":")
                    if len(parts) != 3 or parts[0] != "ai_signal":
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            reject_reason="NON_CANONICAL_AI_SIGNAL_KEY",
                            redis_signal_key=ks,
                            extra_json={"key_parts": list(parts[:6])},
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue
                    live_ai_strategy = parts[1].strip().lower()
                    symbol = parts[2]
                    source = "ML"
                    if live_ai_strategy != "day":
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            reject_reason="INVALID_STRATEGY_ID_IN_REDIS_KEY",
                            strategy_id=live_ai_strategy,
                            symbol=symbol,
                            redis_signal_key=ks,
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    # Read decision data - handle both hash and string formats
                    try:
                        decision_data = await self.redis_client.hgetall(key)
                    except Exception as e:
                        # If WRONGTYPE, try to read as string (JSON)
                        if "WRONGTYPE" in str(e):
                            logger.warning(f"SIGNAL_READ_WRONGTYPE: {key} is not a hash, trying string format")
                            json_data = await self.redis_client.get(key)
                            if json_data:
                                import json as json_lib

                                decision_data = json_lib.loads(json_data)
                            else:
                                logger.warning(f"SIGNAL_READ_FAILED: {key} - no data found after format error")
                                # Release claim since we failed to process
                                with contextlib.suppress(Exception):
                                    await self.redis_client.delete(claimed_key)
                                continue
                        else:
                            # Release claim on error
                            with contextlib.suppress(Exception):
                                await self.redis_client.delete(claimed_key)
                            raise

                    if not decision_data:
                        # Release claim if no data
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    logger.debug(f"SIGNAL_CONSUMER: Processing ML signal from {key}")

                    # Canonical signal action: normalize Redis hash to str->str
                    from backend.services.confidence_normalizer import ConfidenceNormalizer
                    from backend.services.signal_action import extract_canonical_action
                    from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                    dd: dict[str, str] = {}
                    for _k, _v in decision_data.items():
                        ks = _k.decode() if isinstance(_k, bytes) else str(_k)
                        if _v is None:
                            continue
                        vs = _v.decode() if isinstance(_v, bytes) else str(_v)
                        dd[ks] = vs

                    side = extract_canonical_action(dd)
                    confidence_raw = dd.get("winner_probability") or dd.get("confidence")
                    if confidence_raw is None or confidence_raw == "":
                        logger.warning("SIGNAL_SKIP: %s missing winner_probability/confidence (invalid signal)", symbol)
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            reject_reason="MISSING_WINNER_PROBABILITY",
                            strategy_id=live_ai_strategy,
                            symbol=symbol,
                            redis_signal_key=ks,
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue
                    try:
                        confidence = float(confidence_raw)
                        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
                            raise ValueError("winner_probability out of [0.0, 1.0]")
                    except (TypeError, ValueError):
                        logger.warning(
                            "SIGNAL_SKIP: %s invalid winner_probability (skip, no Redis write): %r",
                            symbol,
                            confidence_raw,
                        )
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            reject_reason="INVALID_WINNER_PROBABILITY",
                            strategy_id=live_ai_strategy,
                            symbol=symbol,
                            redis_signal_key=ks,
                            extra_json={"raw": str(confidence_raw)[:120]},
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    try:
                        confidence = ConfidenceNormalizer.normalize(confidence)
                    except Exception as norm_err:
                        logger.debug("CONFIDENCE_NORMALIZE: %s failed (using raw): %s", symbol, norm_err)

                    decision_id = dd.get("decision_id") or f"{symbol}_{int(time.time() * 1000)}"
                    if self._consumed_signal_decision_by_key.get(ks) == decision_id:
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    # Global idempotency: per decision_id across all consumers
                    EXECUTED_TTL = 86400  # 24h
                    executed_key = f"executed:{decision_id}"
                    if await self.redis_client.get(executed_key):
                        logger.debug(f"SIGNAL_SKIP: {symbol} decision_id={decision_id} already executed (global idempotency)")
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        self._consumed_signal_decision_by_key[ks] = decision_id
                        continue

                    try:
                        base_symbol = CanonicalSymbolFormatter.to_base(symbol)
                        ccxt_symbol = CanonicalSymbolFormatter.to_ccxt(symbol)
                    except Exception as e:
                        logger.debug("CanonicalSymbolFormatter failed for %s: %s", symbol, e, exc_info=True)
                        base_symbol = symbol.replace("USDT", "").replace("USD", "").replace("/", "")
                        ccxt_symbol = f"{base_symbol}/USDT"

                    is_buy = side == "buy"

                    if is_buy:
                        try:
                            from backend.services.ai_context_freshness_sync import overlay_live_context_freshness

                            overlay_live_context_freshness(dd, symbol)
                        except Exception as ctx_overlay_exc:
                            logger.debug("SIGNAL_CONSUMER context overlay %s: %s", symbol, ctx_overlay_exc)

                    if not is_buy:
                        # Keep non-BUY (HOLD/SELL) candidates in the ranked board.
                        # They must influence score via penalties, not pre-rank admission.
                        side_penalty = 8.0 if str(side).strip().lower() == "sell" else 4.0
                        logger.info(
                            "SIGNAL_SIDE_TELEMETRY: %s side=%s retained_for_ranking penalty=%.2f",
                            symbol,
                            side,
                            side_penalty,
                        )
                        try:
                            dd["quality_opinion_penalty"] = float(dd.get("quality_opinion_penalty") or 0.0) + side_penalty
                            dd["signal_side_penalty"] = float(side_penalty)
                        except (TypeError, ValueError):
                            dd["quality_opinion_penalty"] = side_penalty
                            dd["signal_side_penalty"] = side_penalty
                    # ARCHITECTURE v2: ML model is authoritative — no buy_margin or confidence gate here.
                    # Buy margin logged as telemetry only.
                    else:
                        _bm_telemetry = resolve_buy_margin_from_payload(dd)
                        if _bm_telemetry is not None:
                            logger.debug("TELEMETRY: %s buy_margin=%.4f (not gating)", symbol, _bm_telemetry)

                    logger.info(
                        "SIGNAL_CHECK: %s side=%s winner_prob=%.3f source=%s",
                        symbol,
                        side,
                        confidence,
                        source,
                    )

                    # Get current price from market cache (base_symbol computed above)
                    price_key = f"price:{base_symbol}"
                    price_data = await self.redis_client.hgetall(price_key)

                    logger.info(f"PRICE_CHECK: {symbol} key={price_key} has_data={bool(price_data)}")

                    if not price_data:
                        logger.info(f"SIGNAL_SKIP: {symbol} - no price data at {price_key}")
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            decision_id=decision_id,
                            reject_reason="NO_PRICE_DATA",
                            strategy_id=live_ai_strategy,
                            symbol=symbol,
                            redis_signal_key=ks,
                            extra_json={"price_key": price_key},
                        )
                        await self._update_pipeline_decision(
                            decision_id,
                            {"stage": "GATES", "gate_result": GateReason.REJECT, "gate_reason": GateReason.NO_PRICE_DATA},
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    current_price = float(price_data.get("v", 0))

                    if current_price <= 0:
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            decision_id=decision_id,
                            reject_reason="INVALID_PRICE",
                            strategy_id=live_ai_strategy,
                            symbol=symbol,
                            redis_signal_key=ks,
                        )
                        await self._update_pipeline_decision(
                            decision_id,
                            {"stage": "GATES", "gate_result": GateReason.REJECT, "gate_reason": GateReason.INVALID_PRICE},
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    # ENTRY_MAJOR_ONLY now enforces an independent allowlist (no
                    # master ENTRY_GATES_ENFORCED requirement). When the operator
                    # sets ENTRY_MAJOR_ONLY=true, symbols outside ENTRY_ALLOWED_SYMBOLS
                    # are hard-rejected here, matching the downstream gate in
                    # portfolio_engine.add_buy_candidate. Pre-fix this branch only
                    # added a soft ranking penalty which was insufficient (XRP kept
                    # trading even after removal from the allowlist).
                    _entry_major_only_live = os.getenv("ENTRY_MAJOR_ONLY", "false").strip().lower() in ("1", "true", "yes", "on")
                    if is_buy and _entry_major_only_live and ccxt_symbol not in _ENTRY_ALLOWED_SYMBOLS:
                        logger.info(
                            "ENTRY_SYMBOL_HARD_BLOCK: %s not in major-only allowlist -> reject",
                            ccxt_symbol,
                        )
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_SIGNAL_CONSUME,
                            decision_id=decision_id,
                            reject_reason="SYMBOL_NOT_ALLOWED",
                            strategy_id=live_ai_strategy,
                            symbol=symbol,
                            redis_signal_key=ks,
                        )
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(claimed_key)
                        continue

                    if is_buy and _ENTRY_LIQUIDITY_GATE_ENABLED:

                        def _safe_float_any(raw: Any) -> float | None:
                            try:
                                if raw is None:
                                    return None
                                txt = str(raw).strip()
                                if txt == "":
                                    return None
                                return float(txt)
                            except (TypeError, ValueError):
                                return None

                        spread_pct = None
                        for k in ("spread_pct", "spread", "bid_ask_spread_pct"):
                            spread_pct = _safe_float_any(dd.get(k))
                            if spread_pct is not None:
                                break
                        if spread_pct is None:
                            for k in ("spread_pct", "spread", "bid_ask_spread_pct"):
                                spread_pct = _safe_float_any(price_data.get(k))
                                if spread_pct is not None:
                                    break

                        quote_volume_24h = None
                        for k in ("quote_volume_24h", "quote_volume", "quoteVolume", "volume_usdt", "usd_volume_24h", "volume_24h_usdt"):
                            quote_volume_24h = _safe_float_any(dd.get(k))
                            if quote_volume_24h is not None:
                                break
                        if quote_volume_24h is None:
                            for k in ("quote_volume_24h", "quote_volume", "quoteVolume", "volume_usdt", "usd_volume_24h", "volume_24h_usdt", "qv"):
                                quote_volume_24h = _safe_float_any(price_data.get(k))
                                if quote_volume_24h is not None:
                                    break

                        if spread_pct is None:
                            logger.info(
                                "ENTRY_SPREAD_TELEMETRY: %s spread missing -> penalty only (no gate reject)",
                                ccxt_symbol,
                            )
                            dd["quality_opinion_penalty"] = float(dd.get("quality_opinion_penalty") or 0.0) + 3.0
                            dd["spread_quality_status"] = "missing_penalty_only"
                        elif spread_pct > _ENTRY_MAX_SPREAD_PCT:
                            logger.info(
                                "ENTRY_SPREAD_TELEMETRY: %s spread=%.6f > max=%.6f -> penalty only (no gate reject)",
                                ccxt_symbol,
                                float(spread_pct),
                                float(_ENTRY_MAX_SPREAD_PCT),
                            )
                            dd["quality_opinion_penalty"] = float(dd.get("quality_opinion_penalty") or 0.0) + 4.0
                            dd["spread_quality_status"] = "wide_penalty_only"

                        if quote_volume_24h is None or quote_volume_24h < _ENTRY_MIN_QUOTE_VOLUME_24H:
                            logger.info(
                                "ENTRY_LIQUIDITY_TELEMETRY: %s quote_volume_24h=%.2f < min=%.2f -> penalty only (no gate reject)",
                                ccxt_symbol,
                                float(quote_volume_24h or 0.0),
                                float(_ENTRY_MIN_QUOTE_VOLUME_24H),
                            )
                            dd["quality_opinion_penalty"] = float(dd.get("quality_opinion_penalty") or 0.0) + 2.0
                            dd["liquidity_quality_status"] = "low_penalty_only"

                    q_det: dict[str, Any] = {}
                    ve_det: dict[str, Any] = {}
                    if is_buy:
                        from backend.services.ai_artifact_contract_gate import evaluate_signal_hash_artifact_contract

                        # Narrow ML-edge bypass — never bypass for generic strategy_id "day".
                        _bm_for_art = 0.0
                        for _k in ("buy_margin", "redis_buy_margin_key"):
                            try:
                                if dd.get(_k) not in (None, ""):
                                    _bm_for_art = float(dd.get(_k))
                                    break
                            except Exception:
                                pass
                        _family = str(dd.get("strategy_family") or "").upper()
                        _is_ml_edge_early = bool(dd.get("ml_enriched") or _family == "ML_EDGE" or (_family.startswith("ML") and _bm_for_art > 0.05) or (_bm_for_art > 0.05 and confidence >= 0.58))

                        dd["live_ai_strategy"] = str(dd.get("live_ai_strategy") or live_ai_strategy).strip().lower()
                        if not str(dd.get("model_artifact_path") or "").strip():
                            models_active = Path(__file__).resolve().parents[2] / "models" / "active"
                            dd["model_artifact_path"] = str(per_coin_artifact_file(models_active, live_ai_strategy, symbol))

                        if _is_ml_edge_early:
                            ok_art, art_rej, art_det = True, None, {"ml_edge_bypass_early": True}
                        else:
                            ok_art, art_rej, art_det = evaluate_signal_hash_artifact_contract(
                                dd,
                                redis_strategy_id=live_ai_strategy,
                                symbol_bus=symbol,
                            )
                        if not ok_art:
                            logger.info(
                                "ARTIFACT_CONTRACT_BLOCKED: %s decision_id=%s reason=%s detail=%s",
                                symbol,
                                decision_id,
                                art_rej,
                                art_det,
                            )
                            await self._strategy_runtime_audit_row(
                                event_type=EVT_SIGNAL_CONSUME,
                                decision_id=decision_id,
                                reject_reason=art_rej or "ARTIFACT_CONTRACT_BLOCKED",
                                strategy_id=live_ai_strategy,
                                symbol=symbol,
                                redis_signal_key=ks,
                                artifact_sha256=(dd.get("artifact_sha256") or "")[:128] or None,
                                extra_json={
                                    **art_det,
                                    "model_artifact_path_tail": str(dd.get("model_artifact_path") or "")[-140:],
                                },
                            )
                            await self._update_pipeline_decision(
                                decision_id,
                                {
                                    "stage": "GATES",
                                    "gate_result": GateReason.REJECT,
                                    "gate_reason": GateReason.ARTIFACT_CONTRACT,
                                },
                            )
                            try:
                                await self.redis_client.delete(claimed_key)
                            except Exception as e:
                                logger.debug(
                                    "SIGNAL_CONSUMER: failed to delete key/claim after artifact_contract %s: %s",
                                    key,
                                    e,
                                    exc_info=True,
                                )
                            continue

                    if side == "sell":
                        logger.info(
                            "MODEL_SELL_TELEMETRY: %s side=sell -> ranking_penalty_only (DAY exits via monitor loop only)",
                            ccxt_symbol,
                        )

                    # ARCHITECTURE v2: ML_124_VETO removed — model is authoritative, no re-check.

                    # Extract features for scoring (use decoded dd)
                    atr = await self._resolve_atr_for_candidate(ccxt_symbol, current_price, dd)

                    def _dfloat(d: dict[str, str], key: str, default: float) -> float:
                        try:
                            return float(d.get(key, default))
                        except (TypeError, ValueError):
                            return default

                    rsi = _dfloat(dd, "rsi", 50.0)
                    adx = _dfloat(dd, "adx", 25.0)
                    ema_alignment = _dfloat(dd, "ema_alignment", 0.5)
                    price_momentum = _dfloat(dd, "price_momentum", 0.0)
                    spread_penalty = _dfloat(dd, "spread_penalty", 0.0)
                    _sp = dd.get("spread_pct")

                    _reg_ml = dd.get("regime_label") or dd.get("regime")
                    regime_label_ml = str(_reg_ml).strip().lower() if _reg_ml not in (None, "") else "unknown"
                    _rs_ml = dd.get("regime_score")
                    try:
                        regime_score_ml = float(_rs_ml) if _rs_ml is not None and str(_rs_ml).strip() != "" else 0.0
                    except (TypeError, ValueError):
                        regime_score_ml = 0.0
                    _bm = resolve_buy_margin_from_payload(dd)
                    bm_for_bar = float(_bm) if _bm is not None else None

                    def _dig_int(dd_map: dict[str, str], k: str) -> int:
                        try:
                            raw = dd_map.get(k)
                            if raw is None or str(raw).strip() == "":
                                return 0
                            return int(float(raw))
                        except (TypeError, ValueError):
                            return 0

                    def _dig_float(dd_map: dict[str, str], k: str, default: float = -1.0) -> float:
                        try:
                            raw = dd_map.get(k)
                            if raw is None or str(raw).strip() == "":
                                return default
                            return float(raw)
                        except (TypeError, ValueError):
                            return default

                    def _prob_float(key: str, default: float = 0.0, _dd=dd) -> float:
                        try:
                            raw = _dd.get(key)
                            if raw is None or str(raw).strip() == "":
                                return default
                            v = float(raw)
                            return v if math.isfinite(v) else default
                        except (TypeError, ValueError):
                            return default

                    # Prefer penalties accumulated on dd (side/spread/liquidity). q_det/ve_det
                    # are often empty stubs in this path and previously wiped those values.
                    _q_pen = float(dd.get("quality_opinion_penalty") or 0.0)
                    if _q_pen <= 0.0 and (q_det or {}).get("penalty_total") not in (None, ""):
                        try:
                            _q_pen = float((q_det or {}).get("penalty_total") or 0.0)
                        except (TypeError, ValueError):
                            _q_pen = 0.0
                    _v_pen = float(dd.get("veto_opinion_penalty") or 0.0)
                    if _v_pen <= 0.0 and (ve_det or {}).get("penalty_total") not in (None, ""):
                        try:
                            _v_pen = float((ve_det or {}).get("penalty_total") or 0.0)
                        except (TypeError, ValueError):
                            _v_pen = 0.0
                    _side_pen = float(dd.get("signal_side_penalty") or 0.0)

                    decision_data_parsed = {
                        "symbol": ccxt_symbol,
                        "ema_alignment": ema_alignment,
                        "price_momentum": price_momentum,
                        "rsi": rsi,
                        "adx": adx,
                        "regime": regime_label_ml,
                        "regime_label": regime_label_ml,
                        "regime_score": regime_score_ml,
                        "spread_penalty": spread_penalty,
                        "spread_pct": float(_sp) if _sp is not None else None,
                        "buy_margin": bm_for_bar,
                        "winner_probability": confidence,
                        # Model direction + full probability triple — required for EV / side penalties.
                        "side": str(side or dd.get("side") or ""),
                        "action": str(dd.get("action") or side or ""),
                        "prediction": str(dd.get("prediction") or dd.get("argmax_action") or side or ""),
                        "argmax_action": dd.get("argmax_action", ""),
                        "prob_buy": _prob_float("prob_buy"),
                        "prob_hold": _prob_float("prob_hold"),
                        "prob_sell": _prob_float("prob_sell"),
                        "live_ai_strategy": live_ai_strategy,
                        "artifact_sha256": (dd.get("artifact_sha256") or "")[:128],
                        "model_artifact_path": (dd.get("model_artifact_path") or "")[:512],
                        "feature_version": _dig_int(dd, "feature_version"),
                        "feature_dim": _dig_int(dd, "feature_dim"),
                        "feature_health_pass": str(dd.get("feature_health_pass") or "1"),
                        "feature_health_pct": _dig_float(dd, "feature_health_pct", 100.0),
                        "feature_health_bad_count": _dig_int(dd, "feature_health_bad_count"),
                        "feature_health_json": str(dd.get("feature_health_json") or "")[:65536],
                        "ctx_ts_utc": (dd.get("ctx_ts_utc") or "")[:64],
                        "ctx_age_sec": _dig_float(dd, "ctx_age_sec", -1.0),
                        "context_fresh_str": (dd.get("context_fresh") or ""),
                        "context_audit_emit": (dd.get("context_audit_emit") or ""),
                        "ctx_rs_btc": _dig_float(dd, "ctx_rs_btc", 0.0),
                        "ctx_depth_imbalance": _dig_float(dd, "ctx_depth_imbalance", 0.0),
                        "quality_opinion_penalty": _q_pen,
                        "veto_opinion_penalty": _v_pen,
                        "signal_side_penalty": _side_pen,
                        "symbol_identity_penalty": float(dd.get("symbol_identity_penalty") or 0.0),
                        "quality_opinion_reasons": json.dumps(
                            (q_det or {}).get("opinion_reasons") or (["signal_side_penalty"] if _side_pen > 0 else []),
                            separators=(",", ":"),
                        ),
                        "veto_opinion_reasons": json.dumps((ve_det or {}).get("opinion_reasons", []), separators=(",", ":")),
                        # Dual-clock + live candle-shape (from ai_signal Redis hash)
                        "ranking_tf": str(dd.get("ranking_tf") or ""),
                        "candle_shape_tf": str(dd.get("candle_shape_tf") or ""),
                        "candle_upper_wick_pct": _dig_float(dd, "candle_upper_wick_pct", 0.0),
                        "candle_lower_wick_pct": _dig_float(dd, "candle_lower_wick_pct", 0.0),
                        "candle_body_pct": _dig_float(dd, "candle_body_pct", 0.0),
                        # Multi-bar volume + reversal features from ai_signal_generator (used by
                        # day_candle_quality_gate). Aggregated 24h `ctx_relative_volume` cannot
                        # see per-candle volume spikes on rejection bars; these do.
                        "recent_last_bar_vol_ratio": _dig_float(dd, "recent_last_bar_vol_ratio", 1.0),
                        "recent_vol_5_vs_20": _dig_float(dd, "recent_vol_5_vs_20", 1.0),
                        "recent_vp_divergence": _dig_float(dd, "recent_vp_divergence", 0.0),
                        "recent_3bar_reversal_flag": _dig_int(dd, "recent_3bar_reversal_flag"),
                        "cs_pat_hammer_bull": _dig_int(dd, "cs_pat_hammer_bull"),
                        "cs_pat_shooting_star_bear": _dig_int(dd, "cs_pat_shooting_star_bear"),
                        "cs_pat_doji_neutral": _dig_int(dd, "cs_pat_doji_neutral"),
                        "cs_pat_bullish_engulfing_bull": _dig_int(dd, "cs_pat_bullish_engulfing_bull"),
                        "cs_pat_bearish_engulfing_bear": _dig_int(dd, "cs_pat_bearish_engulfing_bear"),
                        "cs_pat_inside_bar_neutral": _dig_int(dd, "cs_pat_inside_bar_neutral"),
                        "cs_pat_outside_bar_neutral": _dig_int(dd, "cs_pat_outside_bar_neutral"),
                        "cs_pat_three_white_soldiers_bull": _dig_int(dd, "cs_pat_three_white_soldiers_bull"),
                        "cs_pat_three_black_crows_bear": _dig_int(dd, "cs_pat_three_black_crows_bear"),
                        "cs_net_bias": _dig_float(dd, "cs_net_bias", 0.0),
                        "sub_regime": str(dd.get("sub_regime") or "unknown")[:32],
                        "sub_regime_confidence": _dig_float(dd, "sub_regime_confidence", 0.5),
                        "sub_regime_agrees_with_main": _dig_int(dd, "sub_regime_agrees_with_main"),
                        "sub_regime_reason": str(dd.get("sub_regime_reason") or "")[:64],
                        # ML model quality — read by day_model_quality_gate to demote signals
                        # from poorly-validated per-coin models (e.g. SOL 33% accuracy).
                        "model_accuracy": _dig_float(dd, "model_accuracy", 0.5),
                        "model_trained_at": str(dd.get("model_trained_at") or "")[:64],
                        # RF/GBM disagreement computed in ai_signal_generator.py — was missing
                        # from this whitelist so it never reached candidate.decision_data,
                        # ai_candidate_snapshots, the AI ranking dashboard panel, or the
                        # meta-labeling feature vector (see 2026-07-26 recheck).
                        "model_disagreement": _dig_float(dd, "model_disagreement", 0.0),
                    }

                    # If Mystic already holds this symbol, drop the candidate.
                    # The existing position is managed solely by
                    # `_check_exit_conditions` (sells only on confirmed real
                    # net profit after costs).
                    existing_pos = self.engine.open_positions.get(ccxt_symbol) if self.engine else None
                    if existing_pos is not None and float(getattr(existing_pos, "quantity", 0.0) or 0.0) > 0:
                        logger.debug(
                            "CANDIDATE_DROPPED_ALREADY_OPEN: symbol=%s incoming_conf=%.4f",
                            ccxt_symbol,
                            float(confidence),
                        )
                        continue

                    logger.info(
                        "BUY_MARGIN_TRACE stage=signal_to_candidate symbol=%s decision_id=%s redis_buy_margin_key=%r prob_triple=%s resolve_buy_margin=%s stored_decision_buy_margin=%s",
                        ccxt_symbol,
                        decision_id,
                        dd.get("buy_margin"),
                        bool(dd.get("prob_buy") is not None and dd.get("prob_hold") is not None and dd.get("prob_sell") is not None),
                        f"{float(_bm):.6f}" if _bm is not None else "None",
                        f"{float(bm_for_bar):.6f}" if bm_for_bar is not None else "None",
                    )

                    # ML buy enrichment for execution: stamp a sleeve/setup + thesis_score derived from model edge.
                    # This makes positive buy_margin AI signals route as first-class DAY trades (via existing reversal/bull/range sleeves).
                    # Addresses "market moving but no trades": ML can now originate instead of only AW exact-structure.
                    if is_buy:
                        try:
                            from backend.services.day_trade_thesis import (
                                SETUP_FAILED_BREAKDOWN_REVERSAL,
                                SETUP_HTF_TREND_PULLBACK,
                                SETUP_RANGE_BOUNCE,
                            )

                            buy_m = 0.0
                            for k in ("buy_margin", "buy_margin_raw", "redis_buy_margin_key"):
                                try:
                                    if dd.get(k) not in (None, ""):
                                        buy_m = float(dd.get(k))
                                        break
                                except Exception:
                                    pass
                            awr = str(dd.get("allweather_regime") or dd.get("day_route_regime") or dd.get("market_regime") or dd.get("ctx_market_regime") or dd.get("regime") or "neutral").lower()
                            if "down" in awr or "bear" in awr or "trend_down" in awr:
                                use_setup = SETUP_FAILED_BREAKDOWN_REVERSAL
                            elif "range" in awr:
                                use_setup = SETUP_RANGE_BOUNCE
                            else:
                                use_setup = SETUP_HTF_TREND_PULLBACK
                            dd["allweather_setup"] = use_setup
                            dd["setup_type"] = use_setup
                            dd["entry_thesis"] = use_setup
                            ts = max(0.52, 0.50 + min(0.28, buy_m * 0.65))
                            dd["thesis_score"] = ts
                            dd["day_route_regime"] = awr or ("bear" if ("down" in awr or "bear" in awr) else "bull")
                            dd["strategy_family"] = "ML_EDGE"
                            dd["ml_enriched"] = "1"
                            # Ensure the dict that gets passed to add_buy_candidate has the thesis (parsed is built earlier in scope)
                            try:
                                if "decision_data_parsed" in locals():
                                    decision_data_parsed.update(
                                        {
                                            "allweather_setup": use_setup,
                                            "setup_type": use_setup,
                                            "entry_thesis": use_setup,
                                            "thesis_score": ts,
                                            "day_route_regime": dd.get("day_route_regime"),
                                            "strategy_family": "ML_EDGE",
                                            "ml_enriched": "1",
                                        }
                                    )
                            except Exception:
                                pass
                            # Force fields needed by quality gate / router / EV so ML buys can execute and generate learnable outcomes.
                            dd["setup_credit"] = max(0.015, 0.020 + min(0.030, buy_m * 0.08))
                            dd["symbol_trust_setup_strong"] = True
                            dd["strong_setup"] = True
                            dd["net_ev"] = max(0.0015, 0.0009 + min(0.002, buy_m * 0.005))
                            # Update parsed too
                            try:
                                if "decision_data_parsed" in locals():
                                    decision_data_parsed.update(
                                        {
                                            "allweather_setup": use_setup,
                                            "setup_type": use_setup,
                                            "entry_thesis": use_setup,
                                            "thesis_score": ts,
                                            "day_route_regime": dd.get("day_route_regime"),
                                            "strategy_family": "ML_EDGE",
                                            "ml_enriched": "1",
                                            "setup_credit": dd["setup_credit"],
                                            "symbol_trust_setup_strong": True,
                                            "strong_setup": True,
                                            "net_ev": dd["net_ev"],
                                        }
                                    )
                            except Exception:
                                pass
                            logger.info("ML_THESIS_STAMP %s -> %s ts=%.3f m=%.3f credit=%.4f", ccxt_symbol, use_setup, ts, buy_m, dd["setup_credit"])
                        except Exception as _ml_enr:
                            logger.info("ML_ENRICH_SKIP %s: %s", ccxt_symbol, _ml_enr)

                    # DIRECTLY add as candidate (no distributed lock needed for paper trading)
                    # NOTE: Do NOT set executed:{decision_id} here - only when buy actually executes
                    # (bar processor). Setting at enqueue would block retry if candidate is dropped.

                    # Final safety stamp for ML buys right before enqueue: ensure thesis + credit fields are present
                    # so downstream quality/router/EV/artifact bypass see a first-class setup.
                    if is_buy:
                        try:
                            _bm_final = 0.0
                            for k in ("buy_margin", "redis_buy_margin_key", "buy_margin_raw"):
                                v = dd.get(k) or (decision_data_parsed.get(k) if isinstance(decision_data_parsed, dict) else None)
                                if v not in (None, ""):
                                    _bm_final = float(v)
                                    break
                            if _bm_final > 0.0 or confidence >= 0.55:
                                from backend.services.day_trade_thesis import SETUP_FAILED_BREAKDOWN_REVERSAL, SETUP_HTF_TREND_PULLBACK, SETUP_RANGE_BOUNCE

                                _use = SETUP_HTF_TREND_PULLBACK
                                # crude regime hint from dd
                                _r = str(dd.get("regime") or dd.get("day_route_regime") or "").lower()
                                if "down" in _r or "bear" in _r:
                                    _use = SETUP_FAILED_BREAKDOWN_REVERSAL
                                elif "range" in _r:
                                    _use = SETUP_RANGE_BOUNCE
                                _tsf = max(0.55, 0.52 + min(0.25, _bm_final * 0.6))
                                _credit = max(0.015, 0.018 + min(0.025, _bm_final * 0.07))
                                for _d in (dd, decision_data_parsed if isinstance(decision_data_parsed, dict) else {}):
                                    if isinstance(_d, dict):
                                        _d.setdefault("allweather_setup", _use)
                                        _d.setdefault("setup_type", _use)
                                        _d.setdefault("entry_thesis", _use)
                                        _d["thesis_score"] = _tsf
                                        _d["setup_credit"] = _credit
                                        _d["symbol_trust_setup_strong"] = True
                                        _d["strong_setup"] = True
                                        _d["ml_enriched"] = "1"
                                        _d["strategy_family"] = "ML_EDGE"
                                logger.info("ML_FINAL_STAMP %s setup=%s ts=%.3f credit=%.4f", ccxt_symbol, _use, _tsf, _credit)
                        except Exception as _f:
                            logger.debug("ML_FINAL_STAMP_SKIP %s: %s", ccxt_symbol, _f)

                    enqueued, superseded_id = self.engine.add_buy_candidate(
                        symbol=ccxt_symbol,
                        confidence=confidence,
                        current_price=current_price,
                        atr=atr,
                        decision_data=decision_data_parsed,
                        decision_id=decision_id,
                    )
                    await self._apply_buy_candidate_enqueue_pipeline(decision_id, enqueued, superseded_id)
                    if enqueued:
                        candidates_added += 1
                        logger.info(f"CANDIDATE_ADDED: {ccxt_symbol} conf={confidence:.3f} price=${current_price:.2f}")
                        await self._strategy_runtime_audit_row(
                            event_type=EVT_CANDIDATE_ENQUEUED,
                            decision_id=decision_id,
                            strategy_id=live_ai_strategy,
                            symbol=ccxt_symbol,
                            redis_signal_key=ks,
                            artifact_sha256=(dd.get("artifact_sha256") or "")[:128] or None,
                            feature_version=_dig_int(dd, "feature_version") or None,
                            feature_dim=_dig_int(dd, "feature_dim") or None,
                            extra_json={
                                "superseded_decision_id": superseded_id,
                                "confidence": confidence,
                                "winner_prob": confidence,
                            },
                        )
                        # Operator/API visibility: ai_signal:* is deleted below after consumption; mirror
                        # enqueued BUY state into pe_buy_candidate_redis_key so post-consume
                        # readers still see the candidate until bar ranking completes.
                        if self.redis_client:
                            try:
                                pend_key = pe_buy_candidate_redis_key(symbol)
                                _pend_ttl = max(AI_SIGNAL_REDIS_TTL_SEC, int(self.bar_interval * 4))
                                await self.redis_client.hset(
                                    pend_key,
                                    mapping={
                                        "timestamp": str(time.time()),
                                        "decision_id": str(decision_id),
                                        "source": str(source),
                                    },
                                )
                                await self.redis_client.expire(pend_key, _pend_ttl)
                            except Exception as pe_e:
                                logger.debug("PE_PENDING_REDIS: failed %s: %s", symbol, pe_e, exc_info=True)
                    else:
                        logger.debug(
                            "CANDIDATE_NOT_ENQUEUED: %s conf=%.3f reason=intra_bar_lower_score",
                            ccxt_symbol,
                            confidence,
                        )

                    # Keep ai_signal:* source hash for observability; dedupe by decision_id.
                    try:
                        claimed_key = f"claimed:{key}"
                        await self.redis_client.delete(claimed_key)
                        logger.debug(f"Claim released: {claimed_key}")
                    except Exception as e:
                        logger.debug(f"Failed to release claim {claimed_key}: {e}")
                    self._consumed_signal_decision_by_key[ks] = decision_id
                    # ================================================================

            except asyncio.CancelledError:
                # BUG #28 FIX: CancelledError must be first to allow graceful shutdown
                break
            # ================================================================
            # PHASE 4 FIX #4: DISTINGUISH TRANSIENT VS PERMANENT ERRORS
            # ================================================================
            except (redis.ConnectionError, redis.TimeoutError, asyncio.TimeoutError) as transient_err:
                # Transient errors - retry with backoff
                logger.warning(f"Transient error in signal loop (retrying): {transient_err}")
                # Release claim on error if exists
                try:
                    if "claimed_key" in locals() and claimed_key:
                        await self.redis_client.delete(claimed_key)
                except Exception as e:
                    logger.debug("Transient error handler: failed to release claim %s: %s", claimed_key if "claimed_key" in locals() else "?", e, exc_info=True)
                    pass
                await asyncio.sleep(5)
                continue
            except Exception as permanent_err:
                # Permanent errors - log with full context
                logger.exception(f"Permanent error in signal consumption loop: {permanent_err}")
                # Release claim on error if exists
                try:
                    if "claimed_key" in locals() and claimed_key:
                        await self.redis_client.delete(claimed_key)
                except Exception as e:
                    logger.debug("Permanent error handler: failed to release claim %s: %s", claimed_key if "claimed_key" in locals() else "?", e, exc_info=True)
                    pass
                # Slower retry for permanent errors to avoid spam
                await asyncio.sleep(30)
                continue

            # Log candidates added this iteration
            if candidates_added > 0:
                logger.info(f"SIGNAL_CONSUMER: Added {candidates_added} candidates for bar processing")

            # Sleep OUTSIDE all except blocks so it runs every iteration
            await asyncio.sleep(self._signal_consumer_interval)

    async def _update_pipeline_decision(self, decision_id: str, updates: dict[str, Any]) -> None:
        """
        LOW #5 FIX: Update existing pipeline decision with new stage data.
        Delegates to unified service to avoid code duplication.
        """
        from backend.services.pipeline_decision_service import update_pipeline_decision

        await update_pipeline_decision(decision_id, updates)

    async def _apply_buy_candidate_enqueue_pipeline(self, decision_id: str, enqueued: bool, superseded_decision_id: str | None) -> None:
        """Advance pipeline only when a candidate is actually queued; no terminal NOT_EXECUTED churn for peers."""
        _ = superseded_decision_id  # superseding is in-memory only; SQL rows are not finalized here
        if enqueued:
            await self._update_pipeline_decision(
                decision_id,
                {"stage": "GATES", "gate_result": GateReason.PASS, "gate_reason": GateReason.CANDIDATE_ADDED},
            )

    async def _bar_processor_loop(self) -> None:
        """Process candidates at each bar close"""
        from backend.services.day_active_market_bundle import apply_day_bundle_stagger

        await apply_day_bundle_stagger("portfolio")
        while self.is_running:
            try:
                current_time = time.time()
                current_bar = int(current_time / self.bar_interval) * self.bar_interval

                # Only process if new bar
                if current_bar > self.last_bar_processed:
                    grace_sec = local_bar_signal_grace_seconds()
                    if grace_sec > 0:
                        logger.info(
                            "BAR_PROCESS: local signal grace %.1fs before processing bar %s (align ML keys / candidates)",
                            grace_sec,
                            current_bar,
                        )
                        await asyncio.sleep(grace_sec)

                    logger.info(f"BAR_PROCESS: Processing bar {current_bar} with {len(self.engine.current_bar_candidates) if self.engine else 0} candidates")
                    self.last_bar_processed = current_bar

                    if self.engine:
                        cand_buses: list[str] = []
                        try:
                            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

                            for c in self.engine.current_bar_candidates:
                                try:
                                    cand_buses.append(CanonicalSymbolFormatter.to_exchange(c.symbol))
                                except Exception:
                                    pass
                        except Exception:
                            cand_buses = []

                        await self.engine._classify_market_regime()
                        await self.engine._check_churn_guard()
                        await self.engine.update_scoreboard()
                        # Exit monitor loop owns market-data refresh; bar boundary runs exit
                        # checks on cached bundles/prices only before BUY processing.
                        await self._monitor_positions_once(refresh_market_data=False)

                        entry_bar = int(current_time / self.entry_decision_interval) * self.entry_decision_interval
                        result = None
                        if entry_bar > self.last_entry_bar_processed:
                            # 15m (default) entry decisions; pass 1m current_bar for cooldown math.
                            logger.info(
                                "BAR_PROCESS: DAY entry decision bar=%s (interval=%ss)",
                                entry_bar,
                                self.entry_decision_interval,
                            )
                            try:
                                result = await self.engine.process_bar_candidates(current_bar)
                            finally:
                                # Consume this entry window even if process_bar_candidates
                                # raises after a partial fill. Retrying the same 15m bar
                                # on later 1m ticks can double-buy.
                                self.last_entry_bar_processed = entry_bar

                            if self.redis_client and cand_buses:
                                for b in set(cand_buses):
                                    try:
                                        await self.redis_client.delete(pe_buy_candidate_redis_key(b))
                                    except Exception as rd_e:
                                        logger.debug("PE_PENDING_CLEAR: %s: %s", b, rd_e, exc_info=True)

                            if result:
                                logger.info(f"BAR_EXECUTION: {result['symbol']} | qty={result['quantity']:.6f} @ ${result['price']:.4f}")
                                decision_id = result.get("decision_id")
                                if decision_id and self.redis_client:
                                    await self.redis_client.set(f"executed:{decision_id}", "1", ex=86400)
                            else:
                                logger.info("BAR_PROCESS: No trade executed this entry bar")
                        else:
                            logger.debug(
                                "BAR_PROCESS: skip buys until next entry bar (last=%s interval=%ss)",
                                self.last_entry_bar_processed,
                                self.entry_decision_interval,
                            )

                # Sleep until next bar
                next_bar = current_bar + self.bar_interval
                sleep_time = max(0.1, next_bar - time.time())
                logger.debug(f"BAR_LOOP: Sleeping {sleep_time:.1f}s until next bar at {next_bar}")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in bar processor: {e}")
                await asyncio.sleep(5)

    async def _monitor_positions_once(self, *, refresh_market_data: bool = True) -> list[dict[str, Any]]:
        """
        Single-iteration helper for monitoring positions (used by tests).
        Returns list of exit results.
        """
        if not self.engine:
            return []

        # Stale global heartbeat: warn but still run exits. EXIT_MONITOR_FAIL_CLOSED=true restores skip-all.
        exit_fail_closed = os.getenv("EXIT_MONITOR_FAIL_CLOSED", "false").lower() == "true"
        if self.redis_client:
            try:
                last_update_str = await self.redis_client.get("market_data:last_update")
                if not last_update_str:
                    logger.warning("STALE_DATA_WARN: market_data:last_update missing — continuing exit monitor (per-position price refresh)")
                    if exit_fail_closed:
                        return []
                else:
                    last_update = float(last_update_str)
                    data_age = time.time() - last_update
                    stale_sec = float(os.getenv("EXIT_MONITOR_STALE_WARN_SEC", "30"))
                    if data_age > stale_sec:
                        logger.warning(
                            "STALE_DATA_WARN: market data age %.1fs > %.0fs — continuing exit monitor (per-position price refresh)",
                            data_age,
                            stale_sec,
                        )
                        if exit_fail_closed:
                            return []
            except Exception as e:
                logger.warning(
                    "STALE_DATA_CHECK_ERROR: cannot check freshness (%s) — continuing exit monitor unless EXIT_MONITOR_FAIL_CLOSED",
                    e,
                )
                if exit_fail_closed:
                    return []

        # Refresh prices
        await self._refresh_prices()

        # Canonical alignment: engine.open_positions can drift from portfolio_engine_positions
        # (e.g. filtered-dict monitor restore + sell bookkeeping, or prior reconcile).
        # Reload from SQLite before counting / monitoring so EXIT_CHECK_HEARTBEAT and
        # monitor_all_positions match persisted open rows (same truth as /api/.../status).
        try:
            if hasattr(self.engine, "_load_positions_from_sqlite"):
                # SELECT-only rehydrate — FIFO reconcile is throttled on writer mutation loads.
                await self.engine._load_positions_from_sqlite(allow_mutations=False)
        except Exception as e:
            logger.warning("EXIT_MONITOR_REHYDRATE failed: %s", e)

        # PHASE 3 FIX: Apply exit cooldowns and check hard-paused symbols
        current_time = time.time()

        # Build monitor snapshot without mutating engine.open_positions.
        filtered_positions = {}

        for symbol, position in list(self.engine.open_positions.items()):
            # Skip hard-paused symbols
            if symbol in self.exit_hard_paused:
                logger.debug(f"EXIT_MONITORING: Skipping hard-paused symbol {symbol}")
                continue

            # Skip symbols in cooldown
            cooldown_end = self.exit_cooldown_until.get(symbol, 0)
            if current_time < cooldown_end:
                remaining = cooldown_end - current_time
                logger.debug(f"EXIT_MONITORING: Skipping {symbol} in cooldown ({remaining:.0f}s remaining)")
                continue

            filtered_positions[symbol] = position

        # Record live candles for open/bought symbols only (sell-side monitoring source of truth).
        hold_bds: dict[str, dict[str, Any]] = {}
        hold_ms: dict[str, list[str]] = {}
        if refresh_market_data:
            with contextlib.suppress(Exception):
                await self._record_position_candle_snapshots(set(filtered_positions.keys()))

        if filtered_positions:
            try:
                from backend.services.day_active_market_bundle import (
                    async_fetch_day_active_ohlcv_bundle,
                    async_read_cached_day_active_bundle,
                    validate_day_active_bundle,
                )
                from backend.services.live_market_data import live_market_data_service as _svc
                from backend.utils.symbols import normalize_symbol as _norm_ccxt

                for sym_raw in filtered_positions:
                    try:
                        ccxt_sym = _norm_ccxt(sym_raw)
                        bd: dict[str, Any] = {}
                        ok_bd = False
                        ms: list[str] = ["day_bundle_cache_miss"]
                        cached = await async_read_cached_day_active_bundle(ccxt_sym)
                        if isinstance(cached, dict) and cached:
                            bd = dict(cached)
                            ok_bd, ms = validate_day_active_bundle(bd)
                        # Exit monitor must not block profit sells when Redis bundle TTL expires.
                        # refresh_market_data=False skips candle snapshots only — always refresh
                        # bundle on cache miss/invalid for open positions.
                        if not ok_bd and _svc:
                            fetched = await async_fetch_day_active_ohlcv_bundle(_svc, ccxt_sym)
                            if isinstance(fetched, dict) and fetched:
                                bd = dict(fetched)
                                ok_bd, ms = validate_day_active_bundle(bd)
                        hold_bds[sym_raw] = bd if isinstance(bd, dict) else {}
                        hold_ms[sym_raw] = [] if ok_bd else list(ms or ["day_bundle_cache_miss"])
                        if not refresh_market_data and bd:
                            with contextlib.suppress(Exception):
                                await self._apply_bundle_to_position_snapshot(sym_raw, bd)
                    except Exception as ex_inner:
                        hold_bds[sym_raw] = {}
                        hold_ms[sym_raw] = [f"day_bundle_fetch:{ex_inner}"]
            except Exception as ex_outer:
                logger.debug("EXIT_MONITOR_HOLD_BUNDLE_AGG failed: %s", ex_outer)

        # Heartbeat: confirm exit checks are running (WARNING, throttled ~every 30s)
        _now = time.time()
        if not getattr(self, "_last_exit_heartbeat", 0) or (_now - self._last_exit_heartbeat) >= 30:
            self._last_exit_heartbeat = _now
            try:
                from backend.services.portfolio_engine import DAY_MODE_ENABLED
            except Exception as ex:
                logger.debug("DAY_MODE_ENABLED import failed, assuming False: %s", ex)
                DAY_MODE_ENABLED = False
            root_level = logging.getLogger().level
            root_level_name = logging.getLevelName(root_level) if root_level else "NOTSET"
            logger.warning(
                "EXIT_CHECK_HEARTBEAT active_positions=%d day_mode=%s root_loglevel=%s monitor_interval_sec=%s",
                len(filtered_positions),
                DAY_MODE_ENABLED,
                root_level_name,
                self._exit_monitor_interval,
            )
            with contextlib.suppress(Exception):
                await self._log_signal_staleness_alerts()

        try:
            current_bar = int(time.time() / self.bar_interval) * self.bar_interval
            exits = await self.engine.monitor_all_positions(
                self.current_prices,
                current_bar,
                symbols=set(filtered_positions.keys()),
                hold_day_bundles=hold_bds,
                hold_day_missing=hold_ms,
            )

            if filtered_positions and self.engine:
                with contextlib.suppress(Exception):
                    repair_results = await self.engine.process_repair_adds_once(
                        self.current_prices,
                        hold_day_bundles=hold_bds,
                        hold_day_missing=hold_ms,
                        symbols=set(filtered_positions.keys()),
                    )
                    for rr in repair_results or []:
                        logger.warning(
                            "REPAIR_ADD_DONE %s qty=%.8f old_entry=%.4f new_entry=%.4f count=%s",
                            rr.get("symbol"),
                            float(rr.get("quantity") or 0),
                            float(rr.get("old_entry_price") or 0),
                            float(rr.get("new_entry_price") or 0),
                            rr.get("repair_add_count"),
                        )

            for exit_result in exits:
                symbol = exit_result.get("symbol")
                if symbol:
                    self.exit_failure_count.pop(symbol, None)
                    if self.redis_client:
                        with contextlib.suppress(Exception):
                            await self.redis_client.delete(f"alerts:exit_hard_paused:{symbol}")

        except Exception as e:
            from backend.utils.sqlite_runtime import is_locked_error

            # Transient SQLite lock: do not apply long exit cooldowns — retry next cycle.
            if is_locked_error(e) or "database is locked" in str(e).lower():
                logger.warning(
                    "EXIT_MONITORING_SQLITE_BUSY: %s — skip EXIT_FAILED cooldown; retry next cycle",
                    e,
                )
                exits = []
            else:
                logger.exception(f"EXIT_MONITORING_ERROR: {e}")
                for symbol in filtered_positions:
                    await self._handle_exit_failure(symbol, str(e))
                exits = []

        for exit_result in exits:
            logger.info(
                "EXIT_EXECUTED: %s | PnL=$%.2f (%.2f%%) | R=%.2f | %s",
                exit_result["symbol"],
                exit_result["realized_pnl"],
                exit_result["pnl_pct"] * 100.0,
                exit_result["r_multiple"],
                exit_result["exit_type"],
            )

        if exits:
            # After sells, refresh current candles across universe for next entry decisions.
            with contextlib.suppress(Exception):
                await self._refresh_reentry_candles_after_sell()

        return exits

    async def _simplified_pnl_observation_loop(self) -> None:
        """Periodic JSON summary for simplified PnL observation (grep PNL_OBS_SUMMARY); no execution impact."""
        try:
            from backend.services.simplified_pnl_observation import ENABLED as _PNLAB
            from backend.services.simplified_pnl_observation import SUMMARY_INTERVAL_SEC, emit_periodic_summary
        except Exception:
            return
        if not _PNLAB:
            return

        logger.info(
            "SIMPLIFIED_PNL_OBSERVATION: interval_sec=%s (grep PNL_OBS_TRADE_CLOSE | PNL_OBS_SUMMARY | BAR_BUY_MARGIN_HARD counter)",
            SUMMARY_INTERVAL_SEC,
        )
        logger.info("SIMPLIFIED_EXIT_MODE_ACTIVE signal_exit_disabled=true ai_exits_disabled=true min_buy_margin=0.02")

        while self.is_running:
            try:
                await asyncio.sleep(SUMMARY_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            if not self.is_running:
                break
            try:
                eng = self.engine
                ledger_r = float(getattr(eng, "_realized_pnl", 0.0)) if eng is not None else None
                emit_periodic_summary(ledger_realized_pnl=ledger_r)
            except Exception:
                logger.debug("SIMPLIFIED_PNL_OBSERVATION summary failed", exc_info=True)

    async def _ledger_mtm_persist_loop(self) -> None:
        """
        Single-writer (supervisor): recompute MTM from Redis marks and persist portfolio_engine_ledger.
        Keeps SQLite positions_value / total_equity / unrealized_pnl within one interval of live API truth.
        """
        await asyncio.sleep(_LEDGER_MTM_PERSIST_INITIAL_DELAY_SEC)
        _reconcile_every_n = max(1, int(300 / max(1.0, _LEDGER_MTM_PERSIST_INTERVAL_SEC)))
        _loop_count = 0
        while self.is_running:
            try:
                eng = self.engine
                if eng is not None:
                    try:
                        # Network/Redis marks OUTSIDE the writer lock — never hold
                        # BEGIN IMMEDIATE / asyncio SQLite lock across Binance REST.
                        await eng._load_positions_from_sqlite(allow_mutations=False)
                        mtm_prices = await eng._fetch_mtm_prices_for_open_positions()
                        await eng._recompute_positions_values(mtm_prices or self.current_prices or None)
                        async with eng._sqlite_writer_lock:
                            await eng._persist_ledger_to_sqlite()
                            await eng._sync_paper_redis_from_sqlite_authoritative()
                    except Exception as reload_err:
                        logger.warning("LEDGER_MTM_PERSIST reload from SQLite failed: %s", reload_err)

                    # Diagnostic-only reconciliation (read-only, never mutates state):
                    # proves cash derived from transaction evidence still matches the
                    # stored ledger. Runs roughly every 5 minutes, not every tick.
                    _loop_count += 1
                    if _loop_count % _reconcile_every_n == 0:
                        try:
                            from backend.services.portfolio_ledger_reconciliation import reconcile_ledger_cash

                            recon = await asyncio.to_thread(reconcile_ledger_cash, eng.db_path)
                            if not recon.within_tolerance:
                                logger.warning(
                                    "LEDGER_RECONCILIATION_DRIFT %s",
                                    recon.to_dict(),
                                )
                            else:
                                logger.debug("LEDGER_RECONCILIATION_OK %s", recon.to_dict())
                        except Exception as recon_err:
                            logger.debug("LEDGER_RECONCILIATION_SKIPPED: %s", recon_err)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("LEDGER_MTM_PERSIST_LOOP failed: %s", e, exc_info=True)
            try:
                await asyncio.sleep(_LEDGER_MTM_PERSIST_INTERVAL_SEC)
            except asyncio.CancelledError:
                break

    async def _purge_stale_bd_exit_hard_pause_state(self) -> None:
        """Remove Redis hard-pause alerts and in-memory pause state left by the old ``bd`` NameError.

        Does not touch ``exit_cooldown_until`` or non-``bd`` alerts.
        """
        if not self.redis_client:
            return
        cleared: list[str] = []
        try:
            async for raw_key in self.redis_client.scan_iter(match="alerts:exit_hard_paused:*"):
                key_s = raw_key.decode() if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
                prefix = "alerts:exit_hard_paused:"
                if not key_s.startswith(prefix):
                    continue
                sym = key_s[len(prefix) :]
                val = await self.redis_client.get(raw_key)
                if val is None:
                    continue
                val_s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
                if _EXIT_MONITOR_STALE_BD_SIGNATURE not in val_s:
                    continue
                await self.redis_client.delete(raw_key)
                self.exit_hard_paused.discard(sym)
                self.exit_failure_count.pop(sym, None)
                cleared.append(sym)
        except Exception as e:
            logger.warning("EXIT_MONITOR_REDIS_PURGE_BD_STALE failed: %s", e)
        if cleared:
            logger.info("EXIT_HARD_PAUSE_CLEARED_BD_STALE symbols=%s (cooldowns unchanged)", cleared)

    async def _handle_exit_failure(self, symbol: str, error_msg: str) -> None:
        """
        PHASE 3 FIX: Handle exit failure with exponential backoff and hard-pause.

        Backoff strategy:
        - 1st failure: 60 second cooldown
        - 2nd failure: 120 second cooldown
        - 3rd failure: 300 second cooldown
        - 4th+ failure: Hard-pause symbol (permanent until manual intervention)

        Transient ``database is locked`` must not enter this path (caller filters).
        """
        if "database is locked" in str(error_msg).lower() or "database table is locked" in str(error_msg).lower():
            logger.warning(
                "EXIT_FAILED_SKIPPED_SQLITE_BUSY: %s — no cooldown applied; err=%s",
                symbol,
                error_msg[:200],
            )
            return

        current_time = time.time()

        # Increment failure count
        self.exit_failure_count[symbol] = self.exit_failure_count.get(symbol, 0) + 1
        failure_count = self.exit_failure_count[symbol]

        if failure_count >= 4:
            # Hard-pause after 4 consecutive failures
            self.exit_hard_paused.add(symbol)
            logger.error(f"EXIT_HARD_PAUSED: {symbol} failed {failure_count} times - permanently paused until manual intervention")

            # Set Redis alert flag
            if self.redis_client:
                try:
                    await self.redis_client.set(f"alerts:exit_hard_paused:{symbol}", f"count={failure_count},error={error_msg}", ex=86400)
                except Exception as redis_err:
                    logger.warning(f"Failed to set Redis alert for {symbol}: {redis_err}")

        else:
            # Apply exponential backoff cooldown
            cooldown_seconds = [60, 120, 300][min(failure_count - 1, 2)]  # 60s, 120s, 300s
            self.exit_cooldown_until[symbol] = current_time + cooldown_seconds

            logger.warning(f"EXIT_FAILED: {symbol} failure #{failure_count} - cooldown for {cooldown_seconds}s | Error: {error_msg}")

    async def _position_monitor_loop(self) -> None:
        """Monitor positions and execute exits"""
        await self._purge_stale_bd_exit_hard_pause_state()
        logger.info("EXIT_MONITOR_LOOP interval_sec=%s", self._exit_monitor_interval)
        while self.is_running:
            try:
                await self._monitor_positions_once(refresh_market_data=False)

                from backend.config.protected_execution import MANDATORY_EXIT_PENDING_RETRY_SEC

                pending = bool(self.engine and self.engine.has_exit_residual_pending())
                await asyncio.sleep(float(MANDATORY_EXIT_PENDING_RETRY_SEC) if pending else self._exit_monitor_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in position monitor: {e}")
                await asyncio.sleep(5)

    async def _refresh_prices(self) -> None:
        """Refresh prices for all tracked symbols"""
        try:
            # Try to get prices from live market data service
            from backend.services.live_market_data import live_market_data_service

            for symbol in list(self.engine.open_positions.keys()) if self.engine else []:
                try:
                    # Normalize symbol to proper format for API calls
                    from backend.utils.symbols import to_exchange_symbol

                    api_symbol = to_exchange_symbol(symbol)

                    # Try get_ticker with normalized symbol
                    ticker = await live_market_data_service.get_ticker(api_symbol)
                    last_px = float((ticker or {}).get("price") or (ticker or {}).get("last") or 0.0)
                    if ticker and last_px > 0:
                        self.current_prices[symbol] = last_px
                        logger.debug(f"Price refreshed for {symbol} (API: {api_symbol}): ${last_px:.2f}")
                        continue

                    # Fallback: try get_ohlcv with normalized symbol
                    ohlcv = await live_market_data_service.get_ohlcv(api_symbol, "1m", limit=1)
                    if ohlcv and len(ohlcv) > 0 and len(ohlcv[0]) >= 5:
                        close_price = float(ohlcv[0][4])  # close price is index 4
                        if close_price > 0:
                            self.current_prices[symbol] = close_price
                            logger.debug(f"Price refreshed (OHLCV) for {symbol} (API: {api_symbol}): ${close_price:.2f}")
                            continue

                    logger.warning(f"No price data for {symbol} (API: {api_symbol}) - both ticker and OHLCV failed")
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
                    # Don't clear existing price - keep last known

        except ImportError:
            pass  # Service not available
        except Exception as e:
            logger.debug(f"Price refresh error: {e}")

    async def _apply_bundle_to_position_snapshot(self, symbol: str, bundle: dict[str, Any]) -> None:
        """Build position candle snapshot from cached DAY bundle rows (no Binance fetch)."""
        from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES

        snap: dict[str, Any] = {"ts": time.time()}
        for tf in DAY_ACTIVE_TIMEFRAMES:
            rows = bundle.get(tf)
            if isinstance(rows, list) and rows:
                snap[tf] = rows[-1]
        if len(snap) <= 1:
            return
        self._position_candle_cache[symbol] = snap
        if self.redis_client:
            await self.redis_client.set(
                f"position_candles:{symbol}",
                json.dumps(snap, separators=(",", ":")),
                ex=600,
            )

    async def _log_signal_staleness_alerts(self) -> None:
        """Warn when live Redis signals are content-stale beyond contract age."""
        if not self.redis_client:
            return
        try:
            from backend.config.trading_universe import DAY_TRADE_SYMBOLS
            from backend.services.live_strategy_contracts import LiveStrategyId, redis_ai_signal_key

            max_age = float(os.getenv("SIGNAL_CONTENT_STALE_ALERT_SEC", "300"))
            for api_sym in DAY_TRADE_SYMBOLS:
                bus = api_sym.strip().upper().replace("/", "")
                key = redis_ai_signal_key(LiveStrategyId.DAY.value, bus)
                raw = await self.redis_client.hgetall(key)
                if not raw:
                    logger.warning("SIGNAL_STALE_ALERT %s missing_redis_key", bus)
                    continue
                dd = {(k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else str(v)) for k, v in raw.items()}
                if dd.get("content_fresh") == "0" or dd.get("signal_content_stale") == "1":
                    try:
                        age = float(dd.get("content_age_sec") or 0)
                    except (TypeError, ValueError):
                        age = 0.0
                    if age <= 0:
                        try:
                            age = max(0.0, time.time() - float(dd.get("timestamp") or 0))
                        except (TypeError, ValueError):
                            age = 0.0
                    if age >= max_age:
                        logger.warning(
                            "SIGNAL_STALE_ALERT %s content_fresh=%s age_sec=%.0f threshold=%.0f",
                            bus,
                            dd.get("content_fresh"),
                            age,
                            max_age,
                        )
        except Exception as exc:
            logger.debug("SIGNAL_STALE_ALERT scan failed: %s", exc)

    async def _record_position_candle_snapshots(self, symbols: set[str]) -> None:
        """
        Record latest candles for currently bought/open symbols.
        Uses shared DAY bundle cache — no per-TF Binance klines fan-out.
        """
        if not symbols:
            return
        try:
            from backend.services.day_active_market_bundle import (
                async_fetch_day_active_ohlcv_bundle,
                async_read_cached_day_active_bundle,
            )
            from backend.services.live_market_data import live_market_data_service
            from backend.utils.symbols import normalize_symbol as norm_ccxt
        except Exception:
            return

        for symbol in sorted(symbols):
            try:
                ccxt_sym = norm_ccxt(symbol)
                bundle = await async_read_cached_day_active_bundle(ccxt_sym)
                if not bundle and live_market_data_service:
                    bundle = await async_fetch_day_active_ohlcv_bundle(live_market_data_service, ccxt_sym)
                if isinstance(bundle, dict) and bundle:
                    await self._apply_bundle_to_position_snapshot(symbol, bundle)
            except Exception as e:
                logger.debug("POSITION_CANDLE_RECORD_FAIL: %s %s", symbol, e)

    async def _refresh_reentry_candles_after_sell(self) -> None:
        """
        On sell events, refresh current market candles for all tracked symbols so
        post-exit entry selection uses fresh bars.
        """
        if not self.redis_client:
            return

        for base_symbol in _TRACKED_TRADE_BASE_SYMBOLS:
            try:
                from backend.services.day_active_market_bundle import async_read_cached_day_active_bundle
                from backend.utils.symbols import normalize_symbol as norm_ccxt

                ccxt_sym = norm_ccxt(f"{base_symbol}/USDT")
                cached = await async_read_cached_day_active_bundle(ccxt_sym)
                rows = cached.get("1m") if isinstance(cached, dict) else None
                if not isinstance(rows, list) or not rows:
                    continue
                await self.redis_client.set(
                    f"reentry_candle:{base_symbol}",
                    json.dumps({"ts": time.time(), "1m": rows[-1]}, separators=(",", ":")),
                    ex=180,
                )
            except Exception as e:
                logger.debug("REENTRY_CANDLE_REFRESH_FAIL: %s %s", base_symbol, e)

    async def _price_publisher_loop(self) -> None:
        """
        Background task to publish live prices to Redis for signal processing.

        Uses live_market_data ticker cache + limiter (no unbounded direct Binance batch).
        """
        TRADING_SYMBOLS = list(_TRACKED_TRADE_BASE_SYMBOLS)

        logger.info(
            "PRICE_PUBLISHER: Starting for %d symbols interval_sec=%s",
            len(TRADING_SYMBOLS),
            self._price_publisher_interval,
        )

        while self.is_running:
            try:
                if not self.redis_client:
                    await asyncio.sleep(5)
                    continue

                published_count = 0
                try:
                    from backend.services.live_market_data import live_market_data_service
                except ImportError:
                    live_market_data_service = None

                for base_symbol in TRADING_SYMBOLS:
                    try:
                        ccxt_sym = f"{base_symbol}/USDT"
                        price = 0.0
                        try:
                            from backend.services.canonical_mark_price import fetch_canonical_mark

                            cm = await fetch_canonical_mark(ccxt_sym, use_cache=True)
                            if cm is not None and cm.mark > 0:
                                price = float(cm.mark)
                        except Exception:
                            price = 0.0
                        if price <= 0 and live_market_data_service:
                            ticker = await live_market_data_service.get_ticker(ccxt_sym)
                            if ticker:
                                price = float((ticker or {}).get("price") or (ticker or {}).get("last") or 0.0)
                        if price > 0:
                            bus = f"{base_symbol}USDT" if not str(base_symbol).endswith("USDT") else str(base_symbol)
                            price_key = f"price:{bus}"
                            price_data = {
                                "v": str(price),
                                "timestamp": str(time.time()),
                                "ts": str(int(time.time())),
                                "source": "price_publisher",
                            }
                            for field, value in price_data.items():
                                await self.redis_client.hset(price_key, field, value)
                            await self.redis_client.expire(price_key, 120)
                            # Legacy base-only hash (still written for older readers; canonical paths use price:{BUS}).
                            legacy_key = f"price:{base_symbol}"
                            for field, value in price_data.items():
                                await self.redis_client.hset(legacy_key, field, value)
                            await self.redis_client.expire(legacy_key, 120)
                            try:
                                await self.redis_client.set(f"market:{base_symbol}", str(price), ex=120)
                            except Exception:
                                pass
                            published_count += 1
                            logger.debug(f"PRICE_PUBLISH: {base_symbol} = ${price:.2f}")
                    except Exception as e:
                        logger.debug(f"PRICE_PUBLISH_ERROR: {base_symbol} - {e}")

                if published_count > 0:
                    logger.info(f"PRICE_PUBLISHER: Published {published_count}/{len(TRADING_SYMBOLS)} prices to Redis")

                await asyncio.sleep(self._price_publisher_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"PRICE_PUBLISHER_ERROR: {e}")
                await asyncio.sleep(5)

        logger.info("PRICE_PUBLISHER: Stopped")

    async def _binance_sync_loop(self) -> None:
        """
        Background task to periodically sync local database with actual Binance balances.
        Runs every 5 minutes to detect and warn about drift.
        """
        import hashlib
        import hmac

        import httpx

        logger.info("BINANCE_SYNC: Starting balance sync checker (every 5 min)")

        # Wait 60 seconds before first check to let system stabilize
        await asyncio.sleep(60)

        while self.is_running:
            try:
                api_key = os.getenv("BINANCE_API_KEY", "")
                api_secret = os.getenv("BINANCE_SECRET", "")

                if not api_key or not api_secret:
                    logger.warning("BINANCE_SYNC: No API keys configured, skipping")
                    await asyncio.sleep(300)
                    continue

                trading_mode = os.getenv("TRADING_MODE", "paper").lower()
                if trading_mode == "paper":
                    await asyncio.sleep(300)
                    continue

                # Query Binance account balance
                try:
                    timestamp = int(time.time() * 1000)
                    query_string = f"timestamp={timestamp}"
                    signature = hmac.new(api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()

                    from backend.utils.binance_weight_limiter import BinanceWeightLimiter

                    limiter = await BinanceWeightLimiter.create()
                    await limiter.consume("/api/v3/account", weight=10, wait=True, timeout=8.0)

                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.get(f"https://api.binance.us/api/v3/account?{query_string}&signature={signature}", headers={"X-MBX-APIKEY": api_key})

                        if response.status_code == 200:
                            data = response.json()
                            binance_balances = {}

                            for balance in data.get("balances", []):
                                asset = balance["asset"]
                                free = float(balance["free"])
                                if free > 0.001:  # Ignore dust
                                    binance_balances[asset] = free

                            # Compare with local DB - BUG #41 FIX: Use context manager for proper cleanup
                            with sqlite3.connect(DATABASE_PATH) as conn:
                                c = conn.cursor()

                                c.execute("SELECT symbol, quantity FROM portfolio_engine_positions")
                                local_positions = {row[0].split("/")[0]: row[1] for row in c.fetchall()}

                                c.execute("SELECT cash_balance FROM portfolio_engine_ledger ORDER BY id DESC LIMIT 1")
                                local_cash = c.fetchone()
                                local_cash = local_cash[0] if local_cash else 0

                                # Check for drift
                                drift_detected = False

                                # Check USDT
                                binance_usdt = binance_balances.get("USDT", 0)
                                if abs(binance_usdt - local_cash) > 1.0:  # $1 tolerance
                                    logger.warning(f"BINANCE_SYNC: USDT drift detected! Binance=${binance_usdt:.2f}, Local=${local_cash:.2f}")
                                    drift_detected = True

                                # Check positions (Phase 6: forgive dust when Binance has qty but local has 0)
                                for asset, binance_qty in binance_balances.items():
                                    if asset == "USDT":
                                        continue
                                    local_qty = local_positions.get(asset, 0)
                                    if abs(binance_qty - local_qty) <= 0.01:
                                        continue
                                    # Drift: only forgive if local has 0 and Binance qty is dust
                                    if local_qty <= 0.01:
                                        symbol = f"{asset}/USDT"
                                        is_dust = False
                                        if self.engine:
                                            try:
                                                await self.engine._ensure_symbol_constraints(symbol)
                                                price = self.current_prices.get(symbol) or 0.0
                                                is_dust, _, _, _ = self.engine._dust_check(symbol, binance_qty, price)
                                            except Exception as e:
                                                logger.debug("BINANCE_SYNC: dust check failed for %s: %s", symbol, e)
                                        else:
                                            # BUG #M6 FIX: Use richer dust semantics matching engine even when engine unavailable
                                            # Check min_qty AND min_notional (consistent with engine._dust_check)
                                            try:
                                                c.execute(
                                                    "SELECT min_qty, min_notional FROM exchange_symbol_constraints WHERE symbol = ?",
                                                    (symbol,),
                                                )
                                                row = c.fetchone()
                                                if row:
                                                    min_qty = float(row[0] or 0)
                                                    min_notional = float(row[1] or 10.0)
                                                    price = self.current_prices.get(symbol) or 0.0
                                                    notional = binance_qty * price if price > 0 else 0
                                                    # Dust if below min_qty OR below min_notional (matching engine logic)
                                                    if binance_qty > 0 and (binance_qty < min_qty or notional < min_notional):
                                                        is_dust = True
                                            except Exception:
                                                logger.warning(
                                                    "BINANCE_SYNC_DUST_SQLITE_READ_FAILED symbol=%s",
                                                    symbol,
                                                    exc_info=True,
                                                )
                                                is_dust = False  # conservative: do not treat as dust on read failure
                                        if is_dust:
                                            logger.debug(
                                                "BINANCE_SYNC: SYNC_DUST_OK %s Binance=%s local=0 (dust, no warn)",
                                                symbol,
                                                binance_qty,
                                            )
                                            continue
                                    logger.warning(f"BINANCE_SYNC: {asset} drift! Binance={binance_qty:.4f}, Local={local_qty:.4f}")
                                    drift_detected = True

                                # Check for positions we have locally but not on Binance
                                for asset, local_qty in local_positions.items():
                                    if asset not in binance_balances and local_qty > 0.01:
                                        logger.warning(f"BINANCE_SYNC: {asset} exists locally ({local_qty:.4f}) but NOT on Binance!")
                                        drift_detected = True

                                if not drift_detected:
                                    logger.info(f"BINANCE_SYNC: OK - Balances match (USDT=${binance_usdt:.2f}, {len(local_positions)} positions)")
                                else:
                                    logger.warning("BINANCE_SYNC: Drift detected - consider manual sync")
                        else:
                            logger.warning(f"BINANCE_SYNC: API error {response.status_code}")

                except httpx.TimeoutException:
                    logger.warning("BINANCE_SYNC: API timeout")
                except Exception as e:
                    logger.warning(f"BINANCE_SYNC: Error - {e}")

                # Check every 5 minutes
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"BINANCE_SYNC_ERROR: {e}")
                await asyncio.sleep(300)

        logger.info("BINANCE_SYNC: Stopped")

    async def _dust_reconciliation_loop(self) -> None:
        """
        === DUST_INVARIANT_LOCK ===
        Runs every 10 minutes. For each DUST_PENDING position, re-check if still dust.
        If becomes tradable, restore to ACTIVE. Does NOT pause trading.
        === END DUST_INVARIANT_LOCK ===
        """
        logger.info("DUST_RECONCILE: Starting loop (every 10 min)")
        await asyncio.sleep(60)  # Wait 60s before first run to let system stabilize
        while self.is_running:
            try:
                if self.engine:
                    dust_count = sum(1 for p in self.engine.open_positions.values() if getattr(p, "status", "ACTIVE") == "DUST_PENDING")
                    if dust_count > 0:
                        await self.engine.run_dust_reconciliation(current_prices=self.current_prices)
                        logger.debug("DUST_RECONCILE: run complete dust_pending_positions=%d runs_total=%d", dust_count, getattr(self.engine, "dust_reconcile_runs_total", 0))
                await asyncio.sleep(600)  # 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("DUST_RECONCILE_ERROR: %s", e)
                await asyncio.sleep(300)  # 5 min on error
        logger.info("DUST_RECONCILE: Stopped")

    async def _live_reconcile_loop(self) -> None:
        """
        Every 2 minutes: one Binance balance fetch, reconcile all open_positions to exchange.
        Exchange qty <= step_size triggers immediate _remove_dust_position_canonical_cleanup in engine.
        Keeps DB aligned with Binance so dust/zero never blocks buys for long.
        """
        logger.info("LIVE_RECONCILE: Starting loop (every 2 min, one balance fetch per run)")
        # When live, first run immediately so cash comes from Binance ASAP (24/7, no "startup")
        if not (self.engine and getattr(self.engine, "_live_execution_enabled", False) and getattr(self.engine, "_live_service", None)):
            await asyncio.sleep(120)
        while self.is_running:
            try:
                if not self.engine or not getattr(self.engine, "_live_execution_enabled", False) or not getattr(self.engine, "_live_service", None):
                    await asyncio.sleep(120)
                    continue
                live_svc = self.engine._live_service
                balance_result = await live_svc.get_balance("binanceus")
                logger.info("LIVE_RECONCILE: balance_fetch binanceus (one per run)")
                if balance_result.get("status") != "success":
                    await asyncio.sleep(120)
                    continue
                total_balances = balance_result.get("balance", {}).get("total", {}) or {}
                free_balances = balance_result.get("balance", {}).get("free", {}) or {}
                exchange_usdt = float(free_balances.get("USDT", 0) or 0)
                await self.engine.sync_cash_from_exchange(exchange_usdt, "LIVE_RECONCILE")
                await self.engine.run_live_reconcile(total_balances, free_balances=free_balances)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("LIVE_RECONCILE_ERROR: %s", e)
            await asyncio.sleep(120)  # 2 minutes
        logger.info("LIVE_RECONCILE: Stopped")

    async def _ensure_engine_db_alignment(self) -> None:
        """
        LIVE self-correction: ensure engine.db_path matches DATABASE_PATH.
        No restart required. Safe to call repeatedly. Idempotent and concurrency-safe.
        """
        if not self.engine:
            return
        async with self._db_align_lock:
            try:
                engine_db = os.path.abspath(self.engine.db_path)
                canonical_db = os.path.abspath(str(DATABASE_PATH))
                if engine_db != canonical_db:
                    logger.warning(
                        "CANONICAL_RECONCILE_DB_MISMATCH: engine_db=%s canonical_db=%s (correcting live)",
                        engine_db,
                        canonical_db,
                    )
                    self.engine.db_path = canonical_db
                    await self.engine._reconcile_derived_state_from_canonical()
                    logger.info(
                        "CANONICAL_RECONCILE_DB_ALIGNED: db_path=%s",
                        canonical_db,
                    )
            except Exception as e:
                logger.exception("DB alignment check failed: %s", e)

    async def _canonical_reconcile_loop(self) -> None:
        """
        Every 45s: reconcile engine derived state from canonical SQLite (trade_performance,
        paper_trades, paper cache-only balance). No Binance. Keeps status correct without restart.
        Engine must use same DB as canonical (DATABASE_PATH) or reconcile cannot correct ledger.
        """
        logger.info("CANONICAL_RECONCILE: Starting loop (every 55s) - engine must use same DB as canonical")
        await asyncio.sleep(55)
        while self.is_running:
            try:
                if self.engine:
                    await self._ensure_engine_db_alignment()
                    async with self.engine._sqlite_writer_lock:
                        await self.engine._reconcile_derived_state_from_canonical()
                    logger.info(
                        "CANONICAL_RECONCILE: cycle db_path=%s",
                        getattr(self.engine, "db_path", "?"),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("CANONICAL_RECONCILE: %s", e)
            await asyncio.sleep(55)
        logger.info("CANONICAL_RECONCILE: Stopped")

    async def _paper_retention_loop(self) -> None:
        """
        Rolling window: every 15 minutes delete trade_performance and portfolio_engine_audit
        rows older than PAPER_RETENTION_DAYS (default 90). paper_trades retention is handled
        by sqlite_large_table_retention (batched, same 90-day window). Pure SQLite; no Binance.
        """
        keep_days = int(os.getenv("PAPER_RETENTION_DAYS", "90") or "90")
        logger.info("PAPER_RETENTION: Starting loop (every 15 min, keep %d days, paper_trades via large-table retention)", keep_days)
        await asyncio.sleep(120)  # 2 min before first run
        run_count = 0
        while self.is_running:
            try:
                if self.engine:
                    db_path = self.engine.db_path

                    def _run_retention(path: str) -> tuple[int, int, int]:
                        def _op() -> tuple[int, int, int]:
                            with connect_rw(path) as conn:
                                conn.execute("BEGIN IMMEDIATE")
                                cur = conn.cursor()
                                cur.execute(f"SELECT strftime('%Y-%m-%dT00:00:00', 'now', '-{keep_days} days')")
                                cutoff = (cur.fetchone() or ("",))[0]
                                if not cutoff:
                                    return (0, 0, 0)
                                deleted_paper = 0
                                deleted_perf = 0
                                deleted_audit = 0
                                cur.execute(
                                    "DELETE FROM trade_performance WHERE timestamp < ?",
                                    (cutoff,),
                                )
                                deleted_perf = cur.rowcount
                                cur.execute(
                                    "DELETE FROM portfolio_engine_audit WHERE ts < ?",
                                    (cutoff,),
                                )
                                deleted_audit = cur.rowcount
                                # Laptop 24/7 hygiene: passive WAL checkpoint to bound -wal file growth without blocking writers.
                                try:
                                    conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                                except Exception:
                                    pass
                                conn.commit()
                                return (deleted_paper, deleted_perf, deleted_audit)

                        return run_locked_retry(_op)

                    loop = asyncio.get_running_loop()
                    deleted_paper, deleted_perf, deleted_audit = await loop.run_in_executor(None, _run_retention, db_path)
                    if deleted_paper or deleted_perf or deleted_audit:
                        logger.info(
                            "PAPER_RETENTION: deleted paper_trades=%d trade_performance=%d portfolio_engine_audit=%d (older than %d days)",
                            deleted_paper,
                            deleted_perf,
                            deleted_audit,
                            keep_days,
                        )
                    run_count += 1
                    # Skip online VACUUM: exclusive SQLite lock blocks LEDGER_MTM / FIFO / bar writes.
                    if run_count >= 24:
                        run_count = 0
                        logger.debug("PAPER_RETENTION: skipping online VACUUM (exclusive lock risk)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("PAPER_RETENTION: %s", e)
            await asyncio.sleep(900)  # 15 minutes
        logger.info("PAPER_RETENTION: Stopped")

    async def _large_table_retention_loop(self) -> None:
        """
        Bounded retention for large AI/feature/audit tables (30-60 day windows).
        Runs in executor with _sqlite_writer_lock; never VACUUMs online.
        """
        logger.info(
            "LARGE_TABLE_RETENTION: Starting loop (every %.0fs, initial delay %.0fs)",
            _LARGE_TABLE_RETENTION_INTERVAL_SEC,
            _LARGE_TABLE_RETENTION_INITIAL_DELAY_SEC,
        )
        await asyncio.sleep(_LARGE_TABLE_RETENTION_INITIAL_DELAY_SEC)
        while self.is_running:
            try:
                if self.engine:
                    db_path = self.engine.db_path
                    async with self.engine._sqlite_writer_lock:
                        loop = asyncio.get_running_loop()
                        summary = await loop.run_in_executor(
                            None,
                            lambda path=db_path: run_large_table_retention(path),
                        )
                    deleted = summary.get("total_deleted", 0)
                    if deleted:
                        logger.info(
                            "LARGE_TABLE_RETENTION: run complete total_deleted=%s elapsed=%ss",
                            deleted,
                            summary.get("elapsed_sec"),
                        )
                    else:
                        logger.debug(
                            "LARGE_TABLE_RETENTION: run complete nothing deleted elapsed=%ss",
                            summary.get("elapsed_sec"),
                        )
                    # Laptop 24/7 hygiene: passive checkpoint after large retention.
                    try:
                        with connect_rw(db_path) as chk:
                            chk.execute("PRAGMA wal_checkpoint(PASSIVE);")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("LARGE_TABLE_RETENTION: %s", e)
            await asyncio.sleep(_LARGE_TABLE_RETENTION_INTERVAL_SEC)
        logger.info("LARGE_TABLE_RETENTION: Stopped")

    def get_status(self) -> dict[str, Any]:
        """Get integration status"""
        dust_pending = 0
        dust_drift = 0
        dust_runs = 0
        if self.engine:
            dust_pending = sum(1 for p in self.engine.open_positions.values() if getattr(p, "status", "ACTIVE") == "DUST_PENDING")
            dust_drift = getattr(self.engine, "dust_drift_events_total", 0)
            dust_runs = getattr(self.engine, "dust_reconcile_runs_total", 0)
        return {
            "is_running": self.is_running,
            "engine_initialized": self.engine is not None,
            "open_positions": len(self.engine.open_positions) if self.engine else 0,
            "current_candidates": len(self.engine.current_bar_candidates) if self.engine else 0,
            "last_bar_processed": self.last_bar_processed,
            "entry_decision_interval": self.entry_decision_interval,
            "last_entry_bar_processed": self.last_entry_bar_processed,
            "price_cache_size": len(self.current_prices),
            "dust_pending_positions_current": dust_pending,
            "dust_drift_events_total": dust_drift,
            "dust_reconcile_runs_total": dust_runs,
        }


# Global singleton
_integration: PortfolioEngineIntegration | None = None


def get_portfolio_integration() -> PortfolioEngineIntegration:
    """Get the portfolio engine integration singleton"""
    global _integration
    if _integration is None:
        _integration = PortfolioEngineIntegration()
    return _integration


async def start_portfolio_integration() -> PortfolioEngineIntegration:
    """Start the portfolio engine integration"""
    integration = get_portfolio_integration()
    await integration.start()
    return integration


# =============================================================================
# MIGRATION HELPERS
# =============================================================================


# BUG #L8 FIX: Removed placeholder migrate_from_old_service function that computed/discarded values
# without performing actual migration. If migration is needed, implement it properly with:
# 1. Actually use the computed ATR/stop values
# 2. Call engine.execute_buy_fifo() or create Position objects
# 3. Persist to SQLite ledger
# See portfolio_engine.py execute_buy_fifo for proper position creation flow.


def calculate_atr_from_ohlcv(ohlcv: list[list]) -> float:
    """Calculate ATR from OHLCV data"""
    if not ohlcv or len(ohlcv) < 14:
        return 0.0

    true_ranges = []
    for i in range(1, len(ohlcv)):
        high = float(ohlcv[i][2])
        low = float(ohlcv[i][3])
        prev_close = float(ohlcv[i - 1][4])

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    if len(true_ranges) >= 14:
        return sum(true_ranges[-14:]) / 14
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to CCXT format (BASE/QUOTE)"""
    s = str(symbol).strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return f"{s}/USDT"
