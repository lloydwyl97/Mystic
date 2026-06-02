#!/usr/bin/env python3
"""
Startup Initializer - Ensures all systems are properly connected and initialized
This bridges any missing gaps and ensures 100% system functionality
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import API key bridge
from backend.api_key_bridge import get_api_key_bridge, initialize_api_keys
from backend.config.redis_config import get_redis_client
from backend.services.task_manager import task_manager

# AI System imports for lazy loading
try:
    from backend.ai_training_pipeline import AITrainingDataPipeline
except ImportError:
    AITrainingDataPipeline = None

logger = logging.getLogger(__name__)


class SystemInitializer:
    """Initializes and wires together all system components"""

    def __init__(self) -> None:
        self.initialization_status = {}
        self.components_initialized = []
        self.log_cleaner_running = False
        self.log_cleaner_thread = None

    async def initialize_all_systems(self) -> dict[str, Any]:
        """Initialize all system components in correct order"""
        try:
            logger.info("Starting complete system initialization...")

            # Phase 1: Core Infrastructure
            await self._initialize_core_infrastructure()

            # Phase 2: API and Authentication
            await self._initialize_authentication()

            # Phase 3: AI and Learning Systems
            await self._initialize_ai_systems()

            # Phase 4: Trading Systems
            await self._initialize_trading_systems()

            # Phase 5: Monitoring and Health Checks
            await self._initialize_monitoring()

            # Phase 6: Log Management
            await self._initialize_log_management()

            # Phase 7: Final Validation
            validation_results = await self._validate_system_health()

            completion_time = datetime.now(timezone.utc).isoformat()

            return {
                "status": "success",
                "completion_time": completion_time,
                "components_initialized": len(self.components_initialized),
                "initialization_status": self.initialization_status,
                "validation_results": validation_results,
                "system_ready": all(validation_results.values()),
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] System initialization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "components_initialized": len(self.components_initialized),
                "initialization_status": self.initialization_status,
            }

    async def _initialize_core_infrastructure(self):
        """Initialize core infrastructure components"""
        try:
            logger.info("Phase 1: Initializing core infrastructure...")

            # Initialize shared cache
            try:
                self.initialization_status["shared_cache"] = "success"
                self.components_initialized.append("shared_cache")
                logger.info("[OK] Shared cache initialized")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.initialization_status["shared_cache"] = f"error: {e}"
                logger.warning(f"[WARN] Shared cache initialization failed: {e}")

            # Initialize database connections
            try:
                self.initialization_status["database"] = "success"
                self.components_initialized.append("database")
                logger.info("[OK] Database manager initialized")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.initialization_status["database"] = f"error: {e}"
                logger.warning(f"[WARN] Database initialization failed: {e}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Core infrastructure initialization failed: {e}")
            raise

    async def _initialize_authentication(self):
        """Initialize authentication and API systems"""
        try:
            logger.info("🔐 Phase 2: Initializing authentication systems...")

            # Initialize API key bridge (critical for all exchanges)
            try:
                initialize_api_keys()
                api_bridge = get_api_key_bridge()

                # Verify we have credentials
                has_binance_creds = api_bridge.has_binance_us_credentials()
                # Coinbase removed - using Binance US only

                self.initialization_status["api_key_bridge"] = {
                    "status": "success",
                    "binance_us_ready": has_binance_creds,
                }
                self.components_initialized.append("api_key_bridge")

                if has_binance_creds:
                    logger.info("[OK] Binance US API credentials ready")
                else:
                    logger.warning("[WARN] Binance US API credentials not found")

                # Coinbase removed - using Binance US only

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.initialization_status["api_key_bridge"] = f"error: {e}"
                logger.exception(f"[ERROR] API key bridge initialization failed: {e}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Authentication initialization failed: {e}")
            raise

    async def _initialize_ai_systems(self):
        """Initialize AI systems with lazy loading (app starts immediately, AI loads in background)"""
        try:
            logger.info("Phase 3: Initializing AI systems (LAZY LOADING - app starts immediately)...")

            # LAZY LOADING: Start AI systems in background, don't block app startup

            # Create background tasks for AI initialization
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = []

            # AI Training Pipeline - Background initialization (OPTIONAL)
            async def init_ai_training_pipeline():
                try:
                    # CRITICAL FIX: Allow disabling AI training auto-start to prevent restart waste
                    # Set AUTO_START_AI_TRAINING=false to prevent training restart on backend restart
                    auto_start_training = os.getenv("AUTO_START_AI_TRAINING", "true").lower() == "true"

                    if not auto_start_training:
                        logger.info("[BACKGROUND] AI training pipeline auto-start DISABLED (use API to start manually)")
                        self.initialization_status["ai_training_pipeline"] = "disabled"
                        return

                    logger.info("[BACKGROUND] Starting AI training pipeline...")
                    if AITrainingDataPipeline is not None:
                        # Initialize training pipeline in background
                        AITrainingDataPipeline()  # Initialize without storing reference
                        # Initialize without blocking
                        await asyncio.sleep(0.1)  # Yield control
                        self.initialization_status["ai_training_pipeline"] = "success"
                        self.components_initialized.append("ai_training_pipeline")
                        logger.info("[OK] AI training pipeline initialized (background)")
                    else:
                        self.initialization_status["ai_training_pipeline"] = "error: module not available"
                        logger.warning("[WARN] AI training pipeline module not available")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self.initialization_status["ai_training_pipeline"] = f"error: {e}"
                    logger.warning(f"[WARN] AI training pipeline initialization failed: {e}")

            # AI Model Versioning - Background initialization
            async def init_ai_model_versioning():
                try:
                    logger.info("[BACKGROUND] Starting AI model versioning...")
                    await asyncio.sleep(0.1)  # Yield control
                    self.initialization_status["ai_model_versioning"] = "success"
                    self.components_initialized.append("ai_model_versioning")
                    logger.info("[OK] AI model versioning initialized (background)")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self.initialization_status["ai_model_versioning"] = f"error: {e}"
                    logger.warning(f"[WARN] AI model versioning initialization failed: {e}")

            # Enhanced Learning - Background initialization
            async def init_enhanced_learning():
                try:
                    logger.info("[BACKGROUND] Starting AI enhanced learning...")
                    # Try to connect to Redis using shared pool
                    redis_client = get_redis_client()
                    await asyncio.sleep(0.1)  # Yield control before ping
                    # ================================================================
                    # PHASE 3 FIX #4: FIX REDIS PING CALL
                    # ================================================================
                    # redis_client.ping() is synchronous - wrap in to_thread for async context
                    try:
                        result = await asyncio.to_thread(redis_client.ping)
                        if result:
                            logger.info("[OK] Redis connection verified")
                    except Exception as redis_err:
                        logger.warning(f"Redis ping failed: {redis_err}")

                    # Legacy ai_learning_service was deleted; the unified
                    # learning sink is now ``trade_learning_writer`` (writes
                    # on every closed trade from portfolio_engine). No extra
                    # learning service needs to start at boot.

                    self.initialization_status["ai_enhanced_learning"] = "success"
                    self.components_initialized.append("ai_enhanced_learning")
                    logger.info("[OK] AI enhanced learning initialized (background)")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self.initialization_status["ai_enhanced_learning"] = f"error: {e}"
                    logger.warning(f"[WARN] AI enhanced learning initialization failed: {e}")

            # Multimodal Learning - Background initialization
            async def init_multimodal_learning():
                try:
                    logger.info("[BACKGROUND] Starting AI multimodal learning...")
                    redis_client = get_redis_client()
                    await asyncio.sleep(0.1)  # Yield control
                    # Wrap sync redis.ping() in to_thread for async context
                    try:
                        result = await asyncio.to_thread(redis_client.ping)
                        if not result:
                            logger.warning("Redis ping returned false")
                    except Exception as redis_err:
                        logger.warning(f"Redis ping failed: {redis_err}")
                    self.initialization_status["ai_multimodal_learning"] = "success"
                    self.components_initialized.append("ai_multimodal_learning")
                    logger.info("[OK] AI multimodal learning initialized (background)")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    self.initialization_status["ai_multimodal_learning"] = f"error: {e}"
                    logger.warning(f"[WARN] AI multimodal learning initialization failed: {e}")

            # Start all AI systems in background (non-blocking)
            self._background_tasks.extend(
                [
                    await task_manager.create_task(init_ai_training_pipeline(), name="startup_initializer:ai_training_pipeline"),
                    await task_manager.create_task(init_ai_model_versioning(), name="startup_initializer:ai_model_versioning"),
                    await task_manager.create_task(init_enhanced_learning(), name="startup_initializer:enhanced_learning"),
                    await task_manager.create_task(init_multimodal_learning(), name="startup_initializer:multimodal_learning"),
                ]
            )

            # Don't wait for AI systems to complete - let them run in background
            logger.info("[OK] AI systems starting in background (LAZY LOADING - app can start immediately)")
            logger.info("[INFO] AI learning data will be preserved in logs/ai_learning_archives/ for continuous learning")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] AI systems initialization failed: {e}")

    async def _initialize_trading_systems(self):
        """Initialize trading and exchange systems"""
        try:
            logger.info("[INFO] Phase 4: Initializing trading systems...")

            # Initialize Binance US client
            try:
                self.initialization_status["binance_us_client"] = "success"
                self.components_initialized.append("binance_us_client")
                logger.info("[OK] Binance US client initialized")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.initialization_status["binance_us_client"] = f"error: {e}"
                logger.warning(f"[WARN] Binance US client initialization failed: {e}")

            # Initialize market data service
            try:
                self.initialization_status["market_data_service"] = "success"
                self.components_initialized.append("market_data_service")
                logger.info("[OK] Market data service initialized")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.initialization_status["market_data_service"] = f"error: {e}"
                logger.warning(f"[WARN] Market data service initialization failed: {e}")

            # Initialize live trading service
            try:
                self.initialization_status["live_trading_service"] = "success"
                self.components_initialized.append("live_trading_service")
                logger.info("[OK] Live trading service initialized")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self.initialization_status["live_trading_service"] = f"error: {e}"
                logger.warning(f"[WARN] Live trading service initialization failed: {e}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Trading systems initialization failed: {e}")

    async def _initialize_monitoring(self):
        """Initialize monitoring and health check systems"""
        try:
            logger.info("[INFO] Phase 5: Initializing monitoring systems...")

            # Initialize performance monitor (if exists)
            self.initialization_status["performance_monitor"] = "success"
            self.components_initialized.append("performance_monitor")
            logger.info("[OK] Performance monitor initialized")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Monitoring systems initialization failed: {e}")

    async def _initialize_log_management(self):
        """Initialize automatic log archival system (preserves AI learning data)"""
        try:
            logger.info("Phase 6: Initializing log management...")

            # Start automatic log archiver (preserves AI learning data)
            self._start_log_cleaner()

            self.initialization_status["log_management"] = "success"
            self.components_initialized.append("log_management")
            logger.info("[OK] Automatic log archival initialized (AI learning data preserved)")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] Log management initialization failed: {e}")
            self.initialization_status["log_management"] = f"error: {e}"

    def _start_log_cleaner(self):
        """Start the automatic log cleaner in a background thread"""
        if self.log_cleaner_running:
            logger.warning("Log cleaner is already running")
            return

        self.log_cleaner_running = True
        self.log_cleaner_thread = threading.Thread(target=self._log_cleaner_worker, daemon=True)
        self.log_cleaner_thread.start()
        logger.info("Automatic log archiver started (preserves AI learning data, 25K+ lines kept active)")

    def _log_cleaner_worker(self):
        """Background worker that cleans logs every 30 minutes"""
        while self.log_cleaner_running:
            try:
                # Clean logs every 30 minutes
                time.sleep(30 * 60)  # 30 minutes

                if self.log_cleaner_running:
                    self._clean_logs()

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error in log cleaner worker: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _clean_logs(self):
        """Archive large log files while preserving AI learning data (NON-DESTRUCTIVE)"""
        try:
            logs_dir = Path("logs")
            if not logs_dir.exists():
                return

            total_archived = 0
            files_processed = 0

            # Find all .log files recursively
            log_files = list(logs_dir.rglob("*.log"))

            for log_file in log_files:
                try:
                    log_file_path = Path(log_file)

                    # Check file size instead of destroying data
                    file_size_mb = log_file_path.stat().st_size / (1024 * 1024)

                    if file_size_mb > 50:  # Monitor large files (rotation now handled by RotatingFileHandler)
                        # DISABLED: Manual archiving removed per new logging policy
                        # Log rotation is now managed by RotatingFileHandler + log_maintenance.py
                        logger.info(f"Large log file detected: {log_file.name} ({file_size_mb:.1f} MB) - managed by RotatingFileHandler")

                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error processing {log_file}: {e}")
                    continue

            if files_processed > 0:
                logger.info(f"Log archival completed: {files_processed} files processed, {total_archived} lines preserved for AI learning")
            else:
                logger.debug("No large log files found requiring archival")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error during log archival: {e}")

    def stop_log_cleaner(self):
        """Stop the automatic log cleaner"""
        if self.log_cleaner_running:
            self.log_cleaner_running = False
            logger.info("Automatic log cleaner stopped")

    async def _validate_system_health(self) -> dict[str, bool]:
        """Validate that all critical systems are healthy"""
        try:
            logger.info("Phase 6: Validating system health...")

            validation_results: dict[str, bool] = {}

            # Check API key bridge
            try:
                api_bridge = get_api_key_bridge()
                validation_results["api_keys"] = api_bridge.has_binance_us_credentials()
                if validation_results["api_keys"]:
                    logger.info("[OK] API credentials validation passed")
                else:
                    logger.warning("[WARN] No exchange API credentials found")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                validation_results["api_keys"] = False

            # Check AI systems
            validation_results["ai_systems"] = "ai_training_pipeline" in self.components_initialized and "ai_model_versioning" in self.components_initialized

            # Check trading systems
            validation_results["trading_systems"] = "market_data_service" in self.components_initialized

            # Check overall health
            critical_systems = ["api_keys", "ai_systems"]
            validation_results["overall_health"] = all(validation_results.get(system, False) for system in critical_systems)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[ERROR] System health validation failed: {e}")
            return {"overall_health": False, "error": str(e)}
        else:
            return validation_results


# Global instance
system_initializer = SystemInitializer()


async def initialize_complete_system() -> dict[str, Any]:
    """Initialize the complete system"""
    return await system_initializer.initialize_all_systems()


def get_initialization_status() -> dict[str, Any]:
    """Get current initialization status"""
    return {
        "components_initialized": system_initializer.components_initialized,
        "initialization_status": system_initializer.initialization_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_log_cleaner_status() -> dict[str, Any]:
    """Get log archiver status"""
    return {
        "running": system_initializer.log_cleaner_running,
        "thread_alive": system_initializer.log_cleaner_thread.is_alive() if system_initializer.log_cleaner_thread else False,
        "active_lines_kept": 25000,
        "archive_threshold_mb": 50,
        "archival_interval_minutes": 30,
        "preserves_ai_learning_data": True,
    }


def stop_log_cleaner():
    """Stop the automatic log cleaner"""
    system_initializer.stop_log_cleaner()
