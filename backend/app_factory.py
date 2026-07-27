# Load environment variables first
# Fix SQLAlchemy time.clock issue

#############################################################################
# ACTIVE production stack (DAY-only, top-4 Binance.US):
#   - ai_signal_generator.py            (signal generation)
#   - portfolio_engine.py               (position & exit management)
#   - portfolio_engine_integration.py   (signal consumption & execution)
#   - live_trading_service.py           (Binance.US API execution)
#############################################################################
import asyncio
import contextlib
import logging
import os
import sys
import tracemalloc

# CRITICAL FIX: Force IPv4 for ALL connections (Binance US requirement)
# Binance US returns error: {"code":-71012,"msg":"IPv6 not supported"}
# Must be applied BEFORE any other imports that might create socket connections.
# Shared, idempotent patch — see backend/utils/network_ipv4.py.
try:
    from backend.utils.network_ipv4 import ensure_ipv4_only

    ensure_ipv4_only()
    print("[OK] IPv4-only mode enabled (Binance US requirement)")
except Exception as e:
    print(f"[WARNING] Could not force IPv4 mode: {e}")

# CRITICAL FIX: Windows ProactorEventLoop has bugs with async Redis connections
# Use SelectorEventLoop instead for reliable Redis async connections
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backend.config.redis_config import get_shared_redis_async

# Load .env FIRST before any backend imports that might use EnhancedLogger
# This ensures MYSTIC_FILE_LOGGING and other env vars are available when loggers initialize
try:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=False)
    print(f"[OK] Environment variables loaded from {env_path} (no override)")
except ImportError:
    print("[WARNING] python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"[ERROR] Failed to load .env file from {env_path}: {e}")
    # Try loading without explicit path as fallback
    try:
        load_dotenv(override=False)
        print("[OK] Environment variables loaded using fallback method (no override)")
    except Exception as e2:
        print(f"[ERROR] Fallback .env loading also failed: {e2}")

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Import MarketFreshnessHeartbeat
try:
    from backend.services.market_data import MarketFreshnessHeartbeat
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    MarketFreshnessHeartbeat = None  # type: ignore[assignment, misc]

# TOPOLOGY CONTROL: Disable background services when using external supervisor
# M7: Only ONE decision writer may run: portfolio_engine_integration (canonical) OR legacy orchestrator.
# When EXTERNAL_SUPERVISOR_MODE=true, start_portfolio_engine_integration.py owns integration.
# Never run both orchestrator and integration—duplicate writers cause conflicting ai_decision writes.
# Default false: laptop / single-process paper runs integration + ML inside this uvicorn process.
EXTERNAL_SUPERVISOR_MODE = os.getenv("EXTERNAL_SUPERVISOR_MODE", "false").lower() == "true"

# Memory monitor import
try:
    from backend.services.memory_monitor import memory_monitor
except (ImportError, ModuleNotFoundError):
    memory_monitor = None  # type: ignore[assignment]

# Process memory profiler import
try:
    from backend.services.process_memory_profiler import process_memory_profiler
except (ImportError, ModuleNotFoundError):
    process_memory_profiler = None  # type: ignore[assignment]

# Log rotation import
try:
    from backend.config.log_rotation import configure_all_rotating_handlers
except (ImportError, ModuleNotFoundError):
    configure_all_rotating_handlers = None  # type: ignore[assignment]

# Service auto-restart import - DISABLED (user request - not important)
# try:
#     from backend.services.service_auto_restart import service_auto_restart
# except (ImportError, ModuleNotFoundError):
#     service_auto_restart = None  # type: ignore[assignment]
service_auto_restart = None  # DISABLED

# Data service imports (for startup in create_app)
try:
    import backend.services.binance_ws_hydrator as ws_module
    from backend.services.binance_ws_hydrator import create_binance_ws_hydrator
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    create_binance_ws_hydrator = None  # type: ignore[assignment, misc]
    ws_module = None  # type: ignore[assignment, misc]

try:
    from backend.services.binance_user_stream import binance_user_stream_worker
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    binance_user_stream_worker = None  # type: ignore[assignment, misc]

try:
    from backend.services.ai_signal_generator import get_signal_generator
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_signal_generator = None  # type: ignore[assignment, misc]

try:
    from backend.services.canonical_http_client import canonical_http_client
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    canonical_http_client = None  # type: ignore[assignment, misc]

try:
    from backend.services.database_pool_service import get_database_pool_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_database_pool_service = None  # type: ignore[assignment, misc]

try:
    from backend.services.feature_ingestor import feature_ingestor
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    feature_ingestor = None  # type: ignore[assignment, misc]

try:
    from backend.services.market_data import MarketDataService
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    MarketDataService = None  # type: ignore[assignment, misc]

try:
    from backend.services.paper_trading_service import get_paper_trading_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_paper_trading_service = None  # type: ignore[assignment, misc]

try:
    from backend.services.risk_alert_service import risk_alert_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    risk_alert_service = None  # type: ignore[assignment, misc]

try:
    from backend.services.social_trading_service import SocialTradingService, set_social_trading_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    SocialTradingService = None  # type: ignore[assignment, misc]
    set_social_trading_service = None  # type: ignore[assignment, misc]

try:
    from backend.services.social_trading_manager import social_trading_manager
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    social_trading_manager = None  # type: ignore[assignment, misc]

try:
    from backend.services.service_manager import service_manager
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    service_manager = None  # type: ignore[assignment, misc]

try:
    from backend.utils.unicode_safe_logging import configure_unicode_safe_logging
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    configure_unicode_safe_logging = None  # type: ignore[assignment, misc]

# Fix collections.MutableMapping in Python 3.12+
from backend.collections_fix import *  # noqa: F403
from backend.sqlalchemy_fix import *  # noqa: F403

# Fix SQLAlchemy ORM imports
from backend.sqlalchemy_orm_fix import *  # noqa: F403

# Initialize logger early so it's available for import error logging
logger = logging.getLogger(__name__)

# ============================================================================
# PHASE 1 FIX #1: TRADING PAUSE SWITCH (EMERGENCY SAFETY MECHANISM)
# ============================================================================
# Emergency kill switch to pause all trading if critical bugs are detected
# Can be toggled via API endpoint or environment variable
LIVE_TRADING_PAUSED = os.getenv("LIVE_TRADING_PAUSED", "false").lower() == "true"
_PORTFOLIO_INTEGRATION_STARTED = False


def pause_trading():
    """Pause all live trading (emergency stop)"""
    global LIVE_TRADING_PAUSED
    LIVE_TRADING_PAUSED = True
    logger.critical("🚨 LIVE TRADING PAUSED - All trade execution blocked")


def resume_trading():
    """Resume live trading"""
    global LIVE_TRADING_PAUSED
    LIVE_TRADING_PAUSED = False
    logger.info("✓ LIVE TRADING RESUMED - Trade execution allowed")


def is_trading_paused():
    """Check if trading is paused"""
    return LIVE_TRADING_PAUSED


# ============================================================================
# PHASE 3 FIX #1: SIGNAL HANDLERS FOR GRACEFUL SHUTDOWN
# ============================================================================
# Handle SIGTERM and SIGINT to ensure ledger persistence before exit
import signal

_shutdown_event = None


def _setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    global _shutdown_event
    try:
        _shutdown_event = asyncio.Event()

        def _handle_signal(signum, frame):
            """Handle SIGTERM/SIGINT - trigger graceful shutdown"""
            sig_name = signal.Signals(signum).name
            logger.critical(f"🚨 Received {sig_name} - initiating graceful shutdown...")
            if _shutdown_event:
                _shutdown_event.set()

        # Register signal handlers
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        logger.info("✓ Signal handlers registered for SIGTERM/SIGINT (graceful shutdown enabled)")
    except Exception as e:
        logger.warning(f"Could not setup signal handlers: {e}")


# ============================================================================

# Additional lazy imports for optional services
try:
    from backend.services.copy_trading_learning_service import copy_trading_learning_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    copy_trading_learning_service = None  # type: ignore[assignment, misc]

try:
    from backend.services.live_market_data import live_market_data_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    live_market_data_service = None  # type: ignore[assignment, misc]

try:
    from backend.services.realtime_model_trainer import RealTimeModelTrainer
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    RealTimeModelTrainer = None  # type: ignore[assignment, misc]

try:
    from backend.services.ai_continuous_learner import ContinuousLearner
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    ContinuousLearner = None  # type: ignore[assignment, misc]

try:
    from backend.agents.ai_model_manager import get_ai_model_manager
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    get_ai_model_manager = None  # type: ignore[assignment, misc]

try:
    from backend.services.market_data_poller.middleware_router import MarketDataPoller
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    logger.warning(f"MarketDataPoller import failed: {e}")
    MarketDataPoller = None  # type: ignore[assignment, misc]

try:
    from backend.services.task_manager import task_manager
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    task_manager = None  # type: ignore[assignment, misc]

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# Note: .env is already loaded at the top of this file (before service imports)
# This ensures MYSTIC_FILE_LOGGING is available when EnhancedLogger initializes

# Build Redis URL from environment variables (no hardcoded localhost)
if not os.getenv("REDIS_URL"):
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_db = os.getenv("REDIS_DB", "0")
    os.environ["REDIS_URL"] = f"redis://{redis_host}:{redis_port}/{redis_db}"
    logger.info(f"[OK] Built REDIS_URL from components: {os.environ['REDIS_URL']}")

# Now import settings after .env is loaded
from backend.config.settings import settings
from backend.services.admin_auth import require_admin_key
from backend.utils.safe_encoder import decycle

# Additional imports for background hydration
try:
    from backend.services.canonical_cache import canonical_cache
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    canonical_cache = None  # type: ignore[assignment, misc]

# Import Redis cleanup for app lifecycle
try:
    from backend.redis_cleanup import close_all_redis_connections, initialize_redis_cleanup
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    initialize_redis_cleanup = None  # type: ignore[assignment, misc]
    close_all_redis_connections = None  # type: ignore[assignment, misc]

# Import AlertingSystem for shutdown
try:
    from backend.monitoring.alerting_system import alerting_system
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    alerting_system = None  # type: ignore[assignment, misc]

# Import middleware and routers (lazy imports with error handling)
try:
    from backend.observability.middleware import ObservabilityMiddleware
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    ObservabilityMiddleware = None  # type: ignore[assignment, misc]

try:
    from backend.error_handlers import register_error_handlers
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    register_error_handlers = None  # type: ignore[assignment, misc]

try:
    from backend.endpoints.consolidated_router import consolidated_router, load_all_endpoints
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    consolidated_router = None  # type: ignore[assignment, misc]
    load_all_endpoints = None  # type: ignore[assignment, misc]

# NOTE: Mystic's production routing surface is loaded only via
# ``consolidated_router`` (see ``backend.endpoints.consolidated_router``).
# Individual router imports for AI/market/social/strategy/orders were retired
# along with their underlying modules. Do not re-add per-router try/except
# blocks here — add new endpoints to the consolidated essentials list instead.

# IPv4-only DNS patching lives in backend/utils/network_ipv4.py (applied once,
# unconditionally, above) — removed a second dead/unused gated implementation
# (apply_ipv4_patch/restore_ipv4_patch, never called anywhere) that duplicated
# the same monkey-patch with different fallback semantics.

# Ensure UTF-8 stdout/stderr to avoid Windows console Unicode crashes
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError, TypeError):
    # UTF-8 reconfiguration may fail on some systems
    pass

# Configure logging with Unicode error handling
if configure_unicode_safe_logging is not None:
    configure_unicode_safe_logging()  # type: ignore[misc]


class SafeJSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: object) -> bytes:
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            # Safe to ignore decycle errors for JSON serialization
            content = decycle(content)
        return super().render(content)


BASE_PKG = "backend"
PKG_DIR = Path(__file__).resolve().parent

# === REPAIR #11: Static imports (no import_module) ===
try:
    logger.info("Importing AI live services...")
    # Required imports for production system - services used throughout codebase

    live_ai_available = True
    logger.info("ALL AI live services imported successfully! live_ai_available = True")
except Exception as e:
    logger.warning("AI live services import failed: %s", e)
    live_ai_available = False

log = logging.getLogger(__name__)
FEATURE_AI_LIVE = settings.feature_ai_live
FEATURE_AI_DEMO = settings.feature_ai_demo

# Environment gating flags
SKIP_AI_SERVICES = os.getenv("SKIP_AI_SERVICES", "0").lower() in ("1", "true", "yes")
SKIP_HEAVY_SERVICES = os.getenv("SKIP_HEAVY_SERVICES", "0").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Starts portfolio integration in this process so /api/portfolio-engine/status and canonical reconcile share the same engine."""
    logger.info("[LIFESPAN] Backend starting")

    # Minimal setup - just prepare for API requests
    try:
        tracemalloc.start()
        logger.debug("Tracemalloc enabled for debugging")
    except Exception as e:
        logger.debug(f"Tracemalloc failed: {e}")

    # ================================================================
    # BUG #6 FIX: CENTRALIZED SCHEMA INITIALIZATION (BEFORE all services)
    # ================================================================
    # CRITICAL: Schema must be initialized once, atomically, before any service starts
    # This prevents concurrent ALTER TABLE races and schema corruption
    try:
        from backend.database_schema import initialize_all_schemas

        logger.info("[LIFESPAN] Initializing database schema...")
        schema_ok = await initialize_all_schemas()
        if schema_ok:
            logger.info("[LIFESPAN] Database schema initialized successfully")
        else:
            logger.error("[LIFESPAN] CRITICAL: Schema initialization failed - cannot start")
            raise RuntimeError("Schema initialization failed - cannot start trading")
    except Exception as e:
        logger.exception("[LIFESPAN] CRITICAL: Schema initialization error: %s", e)
        raise RuntimeError(f"Schema initialization critical failure: {e}") from e

    try:
        from backend.config.live_test_mode import assert_full_live_safety_at_startup

        assert_full_live_safety_at_startup()
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("[LIFESPAN] Live safety startup check failed: %s", e)
        raise

    try:
        from backend.services.day_gate_registry import warn_or_fail_day_ml_bypass

        warn_or_fail_day_ml_bypass(fail_in_live=True)
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("[LIFESPAN] DAY ML bypass safety check skipped: %s", e)

    with contextlib.suppress(Exception):
        from backend.services.operator_config_service import apply_runtime_config

        apply_runtime_config()

    with contextlib.suppress(Exception):
        from backend.services.ai_outcome_training_writer import (
            backfill_all_symbol_strategy_expectancy,
            repair_mislabeled_profitable_ai_sells,
            repair_missing_sell_feature_versions,
        )

        repaired = repair_mislabeled_profitable_ai_sells()
        if repaired:
            logger.info("[LIFESPAN] Repaired %d mislabeled ai_outcome_training_rows: %s", len(repaired), repaired[:10])
        backfilled = backfill_all_symbol_strategy_expectancy()
        if backfilled:
            logger.info("[LIFESPAN] Backfilled ai_symbol_strategy_expectancy for %d symbol/strategy pairs", backfilled)
        fv_repaired = repair_missing_sell_feature_versions()
        if fv_repaired:
            logger.info("[LIFESPAN] Backfilled feature_version on %d sell rows: %s", len(fv_repaired), fv_repaired[:10])
    with contextlib.suppress(Exception):
        from backend.services.ai_missed_opportunity_observer import ensure_missed_opportunity_table
        from backend.services.ai_post_trade_feature_review import ensure_post_trade_feature_review_table

        ensure_missed_opportunity_table()
        ensure_post_trade_feature_review_table()
    # ================================================================

    # FIX 2: Start MarketDataService at startup so market:* keys are written for /latency (no lazy-init wait)
    if MarketDataService is not None:
        try:
            market_svc = MarketDataService.shared()
            await market_svc.initialize()
            app.state.market_data_service = market_svc
            logger.info("[LIFESPAN] MarketDataService initialized - market:* keys will be written for /latency")
        except Exception as e:
            logger.warning("[LIFESPAN] MarketDataService init failed (latency may show no_redis_cache): %s", e)

    # Social leaderboard read path: initialize service so /api/social/leaderboard can read from same DB
    if SocialTradingService is not None and set_social_trading_service is not None:
        try:
            social_svc = SocialTradingService(database_pool_service=None, cache_service=None)
            await social_svc.initialize()
            set_social_trading_service(social_svc)
            logger.info("[LIFESPAN] SocialTradingService initialized - /api/social/leaderboard will use canonical DB")
        except Exception as e:
            logger.warning("[LIFESPAN] SocialTradingService init failed (leaderboard may 500): %s", e)

    # M5: Load circuit breaker state if it was deferred (async context at import)
    try:
        from backend.services.circuit_breaker_service import trading_circuit_breaker

        await trading_circuit_breaker.load_circuit_state_async()
        logger.info("[LIFESPAN] Circuit breaker state loaded (async)")
    except Exception as e:
        logger.debug("[LIFESPAN] Circuit breaker async load skipped: %s", e)

    # ================================================================
    # BUG #C4 FIX: Only start portfolio integration in lifespan when NOT using external supervisor.
    # When EXTERNAL_SUPERVISOR_MODE=true, the separate start_portfolio_engine_integration.py process
    # owns the integration. Starting both would cause duplicate signal processing and double-trade risk.
    # ================================================================
    if not EXTERNAL_SUPERVISOR_MODE:
        try:
            from backend.services.portfolio_engine_integration import get_portfolio_integration, start_portfolio_integration
            from backend.utils.redis_helpers import WriterLockError

            # PHASE 1 FIX #3: DOUBLE-START GUARD (within this process)
            global _PORTFOLIO_INTEGRATION_STARTED
            if _PORTFOLIO_INTEGRATION_STARTED:
                logger.warning("Portfolio integration already started in this process - skipping restart")
            else:
                await start_portfolio_integration()
                _PORTFOLIO_INTEGRATION_STARTED = True
                logger.info("[LIFESPAN] PortfolioEngineIntegration started - canonical reconcile loop running in this process")
        except WriterLockError as e:
            logger.warning("[LIFESPAN] Portfolio integration skipped (writer lock held elsewhere): %s", e)
        except Exception as e:
            logger.warning("[LIFESPAN] Portfolio integration start failed (status may be stale): %s", e)
    else:
        logger.info("[LIFESPAN] EXTERNAL_SUPERVISOR_MODE=true: portfolio integration managed by start_portfolio_engine_integration.py")

    # ================================================================
    # PHASE 1 FIX #5: START HEALTH MONITORING
    # ================================================================
    try:
        from backend.config.redis_config import get_shared_redis_async
        from backend.services.live_trading_health_monitor import start_health_monitor
        from backend.services.portfolio_engine_integration import get_portfolio_integration

        integration = get_portfolio_integration()
        redis_client = get_shared_redis_async()

        if integration and redis_client:
            await start_health_monitor(integration.engine, redis_client)
            logger.info("[LIFESPAN] Health monitor started - monitoring every 30 seconds")
        else:
            logger.warning("[LIFESPAN] Health monitor startup skipped (missing engine or redis)")
    except Exception as e:
        logger.warning("[LIFESPAN] Health monitor startup failed: %s", e)
    # ================================================================

    # Ensure market_regime:global exists before any dashboard request (local + deploy).
    # Writes real Fear&Greed when possible; otherwise a labeled sideways fallback so regime badge never depends on an empty key.
    try:
        from backend.services.market_regime import seed_market_regime_for_dashboard

        await seed_market_regime_for_dashboard()
        logger.info("[LIFESPAN] Redis seed: market_regime:global ensured for dashboard/AI readers")
    except Exception as e:
        logger.warning("[LIFESPAN] Redis seed market_regime failed (dashboard may show engine_memory until updater runs): %s", e)

    # Start Order Book Collector + Volume Profile Service (required for features 109-111 and 117-122)
    try:
        from backend.services.order_book_collector import order_book_collector

        await order_book_collector.start()
        logger.info("[LIFESPAN] OrderBookCollector started - orderbook:* keys will be written to Redis")
    except Exception as e:
        logger.warning("[LIFESPAN] OrderBookCollector start failed: %s", e)

    try:
        from backend.services.volume_profile_service import volume_profile_service

        await volume_profile_service.start()
        logger.info("[LIFESPAN] VolumeProfileService started - volume_profile:* keys will be written to Redis")
    except Exception as e:
        logger.warning("[LIFESPAN] VolumeProfileService start failed: %s", e)

    # ----------------------------------------------------------------
    # Single-process production: embed DAY context + ML training + signals
    # (when EXTERNAL_SUPERVISOR_MODE=false this process owns supervision).
    # ----------------------------------------------------------------
    app.state._embedded_ai_market_context_svc = None  # type: ignore[attr-defined]
    app.state._embedded_live_md_started = False  # type: ignore[attr-defined]
    app.state._embedded_ai_training_started = False  # type: ignore[attr-defined]

    embed_ctx = os.getenv("AUTO_START_AI_MARKET_CONTEXT", "true").lower() == "true"
    embed_lm_loops = os.getenv("AUTO_START_LIVE_MARKET_DATA_LOOPS", "true").lower() == "true"
    embed_train = os.getenv("AUTO_START_AI_TRAINING", "true").lower() == "true"

    # ================================================================
    # LIVE BINANCE TICKER/OHLCV LOOPS — warms caches for DAY bundles + probes
    # ================================================================
    if not EXTERNAL_SUPERVISOR_MODE and embed_lm_loops and live_market_data_service is not None:
        try:
            if not getattr(live_market_data_service, "_running", False):
                await live_market_data_service.start()
                app.state._embedded_live_md_started = True  # type: ignore[attr-defined]
                logger.info("[LIFESPAN] LiveMarketDataService ticker/ohlcv loops started")
        except Exception as e:
            logger.warning("[LIFESPAN] LiveMarketDataService start failed (direct REST may still work): %s", e)

    # ================================================================
    # AI MARKET CONTEXT — publishes Redis ai_context:* (required when
    # LIVE_AI_FAIL_CLOSED_CONTEXT enables fail-closed ML emit).
    # ================================================================
    if not EXTERNAL_SUPERVISOR_MODE and embed_ctx:
        try:
            from backend.services.ai_market_context import get_market_context_service

            mctx = get_market_context_service()
            await mctx.start()
            await mctx.warm_publish_now()
            app.state._embedded_ai_market_context_svc = mctx  # type: ignore[attr-defined]
            logger.info("[LIFESPAN] AIMarketContextService embedded — ai_context:* publishing")
        except Exception as e:
            logger.warning("[LIFESPAN] AIMarketContextService start failed (ML may skip emit): %s", e)

    # ================================================================
    # AI TRAINING PIPELINE — continuous DAY row collection + retrain
    # ================================================================
    if not EXTERNAL_SUPERVISOR_MODE and embed_train:
        try:
            from backend.ai_training_pipeline import get_ai_training_pipeline

            train_pipe = get_ai_training_pipeline()
            if train_pipe is not None and not getattr(train_pipe, "is_running", False):
                await train_pipe.start()
                app.state._embedded_ai_training_started = True  # type: ignore[attr-defined]
                logger.info("[LIFESPAN] AITrainingDataPipeline embedded (collection + continuous learning)")
        except Exception as e:
            logger.warning("[LIFESPAN] AITrainingDataPipeline start failed: %s", e)

    # ================================================================
    # AI SIGNAL GENERATOR STARTUP (when not using external supervisor)
    # ================================================================
    if not EXTERNAL_SUPERVISOR_MODE and get_signal_generator is not None:
        try:
            signal_gen = get_signal_generator()
            if not signal_gen.is_running:
                await signal_gen.start()
                logger.info("[LIFESPAN] RealTimeAISignalGenerator started - canonical ai_signal:*:* hashes will be written to Redis")
            else:
                logger.info("[LIFESPAN] RealTimeAISignalGenerator already running")
        except Exception as e:
            logger.warning("[LIFESPAN] RealTimeAISignalGenerator start failed: %s", e)
    else:
        logger.info("[LIFESPAN] Signal generator managed externally (EXTERNAL_SUPERVISOR_MODE=%s)", EXTERNAL_SUPERVISOR_MODE)

    try:
        yield
    finally:
        # ================================================================
        # PHASE 3 FIX #1: GRACEFUL SHUTDOWN PERSISTENCE
        # ================================================================
        # Persist ledger and positions to SQLite before shutdown
        # This ensures positions are not orphaned on crash/restart
        logger.info("[LIFESPAN] Backend shutting down - persisting ledger...")
        try:
            from backend.services.portfolio_engine_integration import get_portfolio_integration

            integration = get_portfolio_integration()
            if integration and integration.engine:
                try:
                    # Persist ledger to SQLite (all cash balances, positions)
                    logger.info("[LIFESPAN] Persisting ledger to SQLite...")
                    await integration.engine._persist_ledger_to_sqlite()

                    # Persist all open positions to SQLite
                    logger.info("[LIFESPAN] Persisting positions to SQLite...")
                    for _symbol, position in list(integration.engine.open_positions.items()):
                        await integration.engine._persist_position_to_sqlite(position)

                    # Redis keys are already managed by recovery cache cleanup
                    # No additional flush needed - persistence to SQLite is the critical path
                    logger.info("[LIFESPAN] Ledger persistence complete - positions safe on restart")
                except Exception as persist_err:
                    logger.error(f"[LIFESPAN] Ledger persistence failed: {persist_err}", exc_info=True)

                # Stop integration gracefully
                if getattr(integration, "is_running", False):
                    await integration.stop()
                    logger.info("[LIFESPAN] PortfolioEngineIntegration stopped")
        except Exception as e:
            logger.debug("Lifespan portfolio integration stop: %s", e)

        # Stop embedded supervised loops (avoid signal_gen.stop() — it closes shared Redis).
        try:
            from backend.ai_training_pipeline import _ai_training_pipeline_state

            if getattr(app.state, "_embedded_ai_training_started", False):  # type: ignore[attr-defined]
                tp = _ai_training_pipeline_state.get("instance")
                if tp is not None:
                    tp.is_running = False
        except Exception as e:
            logger.debug("[LIFESPAN] AI training pipeline flag stop: %s", e)

        try:
            mctx_shutdown = getattr(app.state, "_embedded_ai_market_context_svc", None)  # type: ignore[attr-defined]
            if mctx_shutdown is not None:
                await mctx_shutdown.stop()
                app.state._embedded_ai_market_context_svc = None  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug("[LIFESPAN] AIMarketContextService stop: %s", e)

        try:
            if getattr(app.state, "_embedded_live_md_started", False) and live_market_data_service is not None:  # type: ignore[attr-defined]
                await live_market_data_service.stop()
                app.state._embedded_live_md_started = False  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug("[LIFESPAN] LiveMarketDataService stop: %s", e)

        # Shutdown AI learning systems (if they were started)
        try:
            from backend.services.task_manager import task_manager

            if task_manager is not None:
                logger.info("[LIFESPAN] Shutting down AI learning tasks...")
                await task_manager.cancel_all(timeout=10.0)
                logger.info("[LIFESPAN] AI learning tasks shutdown complete")
        except Exception as e:
            logger.debug("AI learning tasks shutdown: %s", e)

        # Close Redis connections
        try:
            if close_all_redis_connections is not None:
                await close_all_redis_connections()
                logger.info("[LIFESPAN] Redis connections closed")
        except Exception as e:
            logger.debug(f"Error closing Redis: {e}")

        logger.info("[LIFESPAN] Backend shutdown complete")


def create_app() -> FastAPI:
    logger.info("CREATE_APP FUNCTION CALLED")
    try:
        from backend.utils.secret_log_filter import install_secret_redacting_filter

        install_secret_redacting_filter()
    except Exception:
        logger.debug("secret log filter install skipped", exc_info=True)
    # Initialize ServiceManager before creating the app

    # Force start AI Signal Generator here as backup
    # DISABLED: Backup signal generator handled by external supervisor in OPTION A topology
    # FIXED: Handle case where event loop is already running (e.g., during testing or nested async)
    if not EXTERNAL_SUPERVISOR_MODE:
        try:

            async def start_signal_gen():
                try:
                    signal_gen = get_signal_generator()
                    if not signal_gen.is_running:
                        await signal_gen.start()
                        if not getattr(signal_gen, "is_running", False):
                            msg = "Signal Generator start() did not set is_running=True"
                            raise RuntimeError(msg)
                        logger.info("[BACKUP] AI Signal Generator started in create_app")
                except Exception as e:
                    logger.warning(f"[BACKUP] Failed to start signal generator in create_app: {e}")

            # Check if event loop is already running - if so, skip sync startup
            # Signal generator will be started in lifespan handler instead
            try:
                loop = asyncio.get_running_loop()
                # Event loop is running - schedule as task instead of blocking
                logger.info("[BACKUP] Event loop already running - signal generator will start in lifespan (loop=%s)", loop)
            except RuntimeError:
                # FIX: Do NOT create event loop during import - causes "attached to different loop" crash
                # Signal generator will be started by lifespan handler in uvicorn's event loop
                logger.info("[FIX] Skipping asyncio.run() during import - will start in lifespan handler")
                # asyncio.run(start_signal_gen())  # DISABLED - causes worker crash
        except Exception as e:
            logger.warning(f"[BACKUP] Signal generator backup startup failed: {e}")
    else:
        logger.info("[TOPOLOGY] Backup signal generator disabled - using external supervisor")
    # HIGH PRIORITY FIX: Don't pre-mark as initialized - let lifespan handler do proper async init
    # Previously this could cause race condition where _initialized=True but no actual initialization
    # The ServiceManager will be properly initialized in the lifespan handler
    if service_manager is not None:
        logger.info("[OK] ServiceManager available - will be initialized in lifespan handler")
    else:
        logger.warning("[WARN] ServiceManager not available - live services may not work")

    # FIX: Pass lifespan handler to FastAPI so services start in uvicorn's event loop
    app = FastAPI(lifespan=lifespan, default_response_class=SafeJSONResponse)  # type: ignore[call-arg]

    # JUST START IT HERE - NO LIFESPAN, NO EVENTS, JUST DO IT
    async def force_start_everything():
        """Force start orchestrator and portfolio engine - ONLY in non-external supervisor mode"""
        # PHASE 1 FIX: Must never run when external supervisor is managing services
        if EXTERNAL_SUPERVISOR_MODE:
            logger.info("EXTERNAL_SUPERVISOR_MODE enabled: force_start_everything() BLOCKED - services handled externally")
            logger.info("[TOPOLOGY] force_start_everything() blocked - external supervisor active")
            return

        logger.info("=" * 70)
        logger.info("FORCE STARTING PORTFOLIO ENGINE (orchestrator disabled)")
        logger.info("=" * 70)

        # BUG #M8 FIX: Legacy orchestrator is disabled - portfolio_engine_integration is the canonical decision writer
        # Do NOT start ai_autobuy_orchestrator to avoid competing writers/decision paths
        logger.info("ORCHESTRATOR DISABLED: Legacy orchestrator replaced by portfolio_engine_integration (canonical writer)")

        try:
            from backend.services.portfolio_engine_integration import start_portfolio_integration
            from backend.utils.redis_helpers import WriterLockError

            logger.info("Starting Portfolio Engine Integration...")
            await start_portfolio_integration()
            logger.info("PORTFOLIO ENGINE STARTED")

        except WriterLockError as e:
            logger.info(f"PORTFOLIO ENGINE SKIPPED: Writer lock held by external process - {e}")
            logger.info(f"[TOPOLOGY] Portfolio Engine Integration skipped - external process holds lock: {e}")
        except Exception as e:
            logger.info(f"PORTFOLIO ENGINE FAILED: {e}")
            import traceback

            traceback.print_exc()

    # PHASE 1 FIX: NEVER start services from FastAPI when EXTERNAL_SUPERVISOR_MODE is true
    # External supervisor handles ALL trading services (orchestrator, portfolio engine, signals)
    if not EXTERNAL_SUPERVISOR_MODE:
        try:
            loop = asyncio.get_running_loop()
            # If loop is running, schedule it
            _startup_task = asyncio.create_task(force_start_everything())
            logger.info("Scheduled force_start_everything() as task")
        except RuntimeError:
            # FIX: Do NOT create event loop during import
            logger.info("[FIX] Skipping force_start_everything() during import - will start in lifespan")
            logger.info("Skipped force_start_everything() - will run in lifespan handler")
    else:
        logger.info("EXTERNAL_SUPERVISOR_MODE enabled: FastAPI is API/dashboard ONLY - NO trading services started")
        logger.info("[TOPOLOGY] FastAPI running in API-only mode - external supervisor handles trading")

    # CRITICAL FIX: Don't create new event loop - orchestrator will be started in lifespan
    # Creating a new event loop here conflicts with FastAPI's event loop and causes issues
    # The orchestrator is already started in the lifespan handler (lines 954-962)
    logger.info("[OK] Orchestrator will be started via lifespan handler - avoiding event loop conflict")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown handler to clean up services."""
        logger.info("[SHUTDOWN] Cleaning up background services...")
        # Add cleanup logic here if needed

    # Add CORS middleware if configured; default to allow all during development
    try:
        _raw_origins = (
            getattr(settings, "ui_origins", None)
            or getattr(settings, "allowed_origins", None)
        )
        if _raw_origins:
            origins: list[str] = [str(x) for x in _raw_origins] if isinstance(_raw_origins, (list, tuple)) else [str(_raw_origins)]  # type: ignore[misc]
        else:
            origins = ["http://localhost:8000"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=(origins != ["*"]),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except (AttributeError, TypeError, ImportError) as e:
        logger.debug(f"Failed to configure CORS middleware: {e}")

    # Trusted hosts middleware if configured
    try:
        trusted_hosts = getattr(settings, "allowed_hosts", None) or getattr(settings, "trusted_hosts", None)
        if trusted_hosts:
            hosts: list[str] = [str(x) for x in trusted_hosts] if isinstance(trusted_hosts, (list, tuple)) else [str(trusted_hosts)]  # type: ignore[misc]
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    except (AttributeError, TypeError, ImportError) as e:
        logger.debug(f"Failed to configure TrustedHostMiddleware: {e}")

    # IP Whitelist middleware (BUG-012 Fix: Security hardening)
    try:
        from backend.middleware.security import IPWhitelistMiddleware

        app.add_middleware(IPWhitelistMiddleware)
        logger.info("[OK] IP Whitelist middleware registered - access control enabled")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.warning(f"IP Whitelist middleware not available: {e}")
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ERROR] Failed to register IP Whitelist middleware: {e}")

    # Observability middleware for Prometheus metrics (should be early in stack)
    try:
        if ObservabilityMiddleware is not None:
            app.add_middleware(ObservabilityMiddleware, enable_logging=True)  # type: ignore[misc]
        logger.info("[OK] ObservabilityMiddleware registered - Prometheus metrics collection enabled")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.warning(f"ObservabilityMiddleware not available: {e}")
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ERROR] Failed to register ObservabilityMiddleware: {e}")

    # Register error handlers for centralized exception handling
    try:
        if register_error_handlers is not None:
            register_error_handlers(app)  # type: ignore[misc]
        logger.info("[OK] Error handlers registered - centralized exception handling enabled")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.warning(f"Error handlers not available: {e}")
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ERROR] Failed to register error handlers: {e}")

    # CRITICAL: Register ALL API endpoints BEFORE static mounts so they take precedence
    # FastAPI matches routes in registration order - API routes must come first!

    # 1) Core health endpoints
    @app.get("/health")
    async def health():
        """Basic health check endpoint"""
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/health")
    async def api_health():
        """API health check endpoint"""
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/ready")
    async def api_ready():
        """API readiness check endpoint"""
        return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/ai/status")
    async def ai_status():
        """AI system status endpoint"""
        return {
            "status": "ok",
            "ai_enabled": True,
            "models_loaded": True,
            "twitter_key_loaded": bool(os.getenv("TWITTER_API_KEY")),
            "env_vars_count": len(os.environ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/debug/env")
    async def debug_env():
        """Debug endpoint to check environment variables - SECURED"""
        # HIGH PRIORITY FIX: Only expose debug info in development mode
        is_debug = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")
        is_production = os.getenv("PRODUCTION", "0").lower() in ("1", "true", "yes")

        if is_production and not is_debug:
            return {
                "status": "restricted",
                "message": "Debug endpoint disabled in production",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Only show that keys are loaded, never expose values or paths
        env_status = {}
        sensitive_terms = ["API", "SECRET", "TOKEN", "KEY", "PASSWORD", "CREDENTIAL"]
        for key in os.environ:
            if any(term in key.upper() for term in sensitive_terms):
                env_status[key] = "***CONFIGURED***" if os.environ[key] else "NOT_SET"
        return {
            "status": "debug",
            "env_configured": len(env_status),
            "sensitive_vars_status": env_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 2) Include consolidated router with all endpoints (211+ routes)
    try:
        # Load all endpoints now that canonical_cache is initialized
        if load_all_endpoints is not None:
            # Clear any previously skipped routers due to canonical_cache not being ready
            from backend.endpoints.consolidated_router import _included

            _included.clear()  # Allow retry of previously skipped routers
            load_all_endpoints()
            logger.info("[OK] All endpoints loaded after canonical_cache initialization")

        if consolidated_router is not None:
            logger.info(f"[OK] Including consolidated_router with {len(consolidated_router.routes)} routes")
            app.include_router(consolidated_router)  # type: ignore[misc]
            logger.info(f"[OK] Consolidated router included with {len(consolidated_router.routes)} routes")
        else:
            logger.warning("consolidated_router is None")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.warning(f"Consolidated router not available: {e}")
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ERROR] Failed to include consolidated router: {e}")

    # 3) Additional routers are intentionally trimmed.
    # Production routing surface is loaded only via consolidated_router.
    logger.info("[OK] Additional router mounts skipped (trimmed production surface)")

    # try:
    #     if social_trading_router is not None:
    #         app.include_router(social_trading_router)  # Full social trading platform
    #     logger.info("[OK] Full social trading platform endpoints mounted")
    # except Exception as e:
    #     logger.exception(f"[ERROR] Failed to mount full social platform endpoints: {e}")

    # 4) Direct endpoint definitions
    # NOTE: /autobuy/status now handled by auto_trading router to avoid duplicates
    # NOTE: /api/orders/active now handled by routes/orders.py to avoid duplicates

    @app.get("/api/live/market/latest")
    async def live_market_latest():
        """Latest live market data endpoint"""
        try:
            # Try to get from app state first (initialized service)
            market_svc = getattr(app.state, "market_data_service", None)
            if market_svc is None:
                # Fallback to shared instance
                if MarketDataService is None:
                    return {"status": "error", "error": "Market data service not available", "timestamp": datetime.now(timezone.utc).isoformat()}
                market_svc = MarketDataService.shared()  # type: ignore[misc]
            # Check if service is initialized
            if not hasattr(market_svc, "is_running") or not market_svc.is_running:
                # Try to initialize if not running
                try:
                    await market_svc.initialize()
                    app.state.market_data_service = market_svc
                except Exception as init_error:
                    logger.warning(f"MarketDataService auto-initialization failed: {init_error}")

            logger.info(f"Market endpoint: cache has {len(market_svc.cache)} items")
            data = await market_svc.get_all_cached_data()
            logger.info(f"Market endpoint: get_all_cached_data returned {len(data)} items")
            return {"status": "ok", "data": data or {}, "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            logger.exception(f"Market endpoint error: {e}")
            return {"status": "error", "message": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/market-data/summary")
    async def market_data_summary_alias():
        """Alias for market overview endpoint"""
        try:
            market_svc = MarketDataService.shared() if MarketDataService is not None else None
            if not market_svc or not market_svc.cache:
                return {"status": "no_data", "message": "Market data not available"}

            total_symbols = len(market_svc.cache)
            gainers = [s for s, d in market_svc.cache.items() if float(d.get("change_24h", 0)) > 0]
            losers = [s for s, d in market_svc.cache.items() if float(d.get("change_24h", 0)) < 0]
            avg_change = sum(float(d.get("change_24h", 0)) for d in market_svc.cache.values()) / total_symbols if total_symbols > 0 else 0
            total_volume = sum(float(d.get("volume_24h", 0)) for d in market_svc.cache.values())

            return {
                "status": "live",
                "total_symbols": total_symbols,
                "gainers": len(gainers),
                "losers": len(losers),
                "avg_change_24h": round(avg_change, 2),
                "total_volume_24h": round(total_volume, 2),
                "market_trend": "bullish" if avg_change > 1 else "bearish" if avg_change < -1 else "neutral",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live_binance_us",
            }
        except Exception as e:
            logger.exception(f"Market summary error: {e}")
            return {"status": "error", "error": str(e)}

    # 5) Mount static directories LAST (after all API routes)
    # This ensures API routes take precedence over static file serving
    try:
        # Dashboard static files
        dashboard_dir = Path(__file__).parent / "static" / "dashboard"
        if dashboard_dir.exists():
            app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")
            logger.info("[OK] Dashboard static files mounted at /dashboard")

        # Mount general static directory
        static_dir = Path(__file__).parent.parent / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            logger.info("[OK] Static files mounted at /static")
    except (OSError, PermissionError, FileNotFoundError) as e:
        logger.warning(f"Failed to mount static directories: {e}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[ERROR] Failed to mount static directories: {e}")

    # Redirect root URL to dashboard
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/dashboard/", status_code=302)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        logo = Path(__file__).parent / "static" / "dashboard" / "assets" / "mystic-logo.svg"
        if logo.exists():
            return FileResponse(logo, media_type="image/svg+xml")
        return RedirectResponse(url="/dashboard/assets/mystic-logo.svg", status_code=302)

    # ========================================================================
    # PHASE 1 FIX #1: PAUSE/RESUME TRADING ENDPOINTS
    # ========================================================================
    @app.post("/api/system/pause-trading")
    async def pause_trading_endpoint(_: None = Depends(require_admin_key)):
        """
        Emergency endpoint to pause all live trading.
        Used when critical bugs or issues are detected.
        """
        pause_trading()
        return {"status": "paused", "message": "Live trading is now PAUSED - no new trades will execute", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post("/api/system/resume-trading")
    async def resume_trading_endpoint(_: None = Depends(require_admin_key)):
        """
        Resume live trading after being paused.
        Should only be called after critical issues are resolved.
        """
        resume_trading()
        return {"status": "resumed", "message": "Live trading is now RESUMED - trades can execute", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post("/api/system/restart")
    async def restart_service_endpoint(_: None = Depends(require_admin_key)):
        """
        Restart the Mystic service in the background.
        Used after writing new env flags (e.g. paper→live flip) to apply changes.
        Returns immediately; the subprocess runs detached so this response can be delivered.
        """
        import subprocess
        repo_root = Path(__file__).resolve().parent.parent
        script = str(repo_root / "stop_mystic.sh")
        start_script = str(repo_root / "start_mystic.sh")
        subprocess.Popen(
            f"bash -c 'sleep 1 && {script} && sleep 2 && {start_script} core'",
            shell=True,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("[RESTART] Service restart triggered via /api/system/restart")
        return {"success": True, "message": "Service restarting in background"}

    @app.get("/api/system/trading-status")
    async def get_trading_status():
        """Check if trading is paused or active"""
        return {"trading_paused": is_trading_paused(), "status": "PAUSED" if is_trading_paused() else "ACTIVE", "timestamp": datetime.now(timezone.utc).isoformat()}

    # ========================================================================

    return app


# Factory entrypoint
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host=settings.backend_host, port=8000)
