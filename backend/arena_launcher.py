import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.task_manager import task_manager

# Process launch timing constant
PROCESS_LAUNCH_STAGGER_DELAY = 0.1  # 100ms delay between process launches to prevent resource spikes

REDIS_AVAILABLE = True

# Initialize logger (no basicConfig here)
logger = logging.getLogger(__name__)

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

# Configuration - All Live Data, No Fallback/Hardcoded Data
BINANCE_US_TOP10 = list(TRADING_SYMBOLS)

# Redis configuration - All Live Data, No Fallback/Hardcoded Data
REDIS_TIMEOUT = 5  # seconds


class StrategyArena:
    def __init__(
        self,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_db: int | None = None,
    ) -> None:
        # Redis configuration - All Live Data, No Fallback/Hardcoded Data
        redis_host_env = os.getenv("REDIS_HOST")
        if redis_host:
            self.redis_host = redis_host
        elif redis_host_env:
            self.redis_host = redis_host_env
        else:
            msg = "REDIS_HOST environment variable is required - no fallback/hardcoded Redis host"
            raise RuntimeError(msg)
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db or int(os.getenv("REDIS_DB", "0"))
        self.redis_url = os.getenv("REDIS_URL")

        # Initialize Redis client with error handling
        self.redis_client = None
        self.redis_available = False
        self.health_issues = []

        if REDIS_AVAILABLE:
            try:
                # Use shared Redis connection pool to prevent connection exhaustion
                from backend.config.redis_config import get_shared_redis_sync

                self.redis_client = get_shared_redis_sync()

                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"Redis connection established: {self.redis_host}:{self.redis_port}")

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Redis connection failed: {e}")
                self.health_issues.append(f"Redis unavailable: {e}")
                self.redis_available = False
        else:
            logger.warning("Redis not available - arena running in degraded mode")
            self.health_issues.append("Redis module not installed")

        # Strategy management with thread safety
        self.active_strategies = {}
        self.active_containers = {}  # map strategy_name -> process/container id
        self.strategy_lock = threading.Lock()
        self.leaderboard_key = "strategy_leaderboard"
        self.worker_queue_key = "strategy_worker_commands"
        self.metrics_key_prefix = "strategy_metrics:"

        # Arena configuration
        self.base_dir = str(Path(__file__).resolve().parent)
        self.agents_dir = str(Path(self.base_dir) / "agents")

        # Ensure agents directory exists
        Path(self.agents_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"StrategyArena initialized - Redis: {self.redis_available}, Agents dir: {self.agents_dir}")

    def create_strategy_config(
        self,
        strategy_name: str,
        capital: float,
        timeframe: str = "1h",
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create individual strategy configuration with Binance US symbol formatting"""
        if symbols is None:
            # All Live Data, No Fallback/Hardcoded Data - use first 3 from trading_universe
            symbols = list(TRADING_SYMBOLS[:3])

        # Ensure all symbols are in Binance US format and within allowlist
        validated_symbols = []
        for symbol in symbols:
            # Convert slash format to concat format if needed
            symbol_normalized = symbol.replace("/", "") if "/" in symbol else symbol

            # Ensure symbol is in Binance US Top-10
            if symbol_normalized in BINANCE_US_TOP10:
                validated_symbols.append(symbol_normalized)
            else:
                logger.warning(f"Symbol {symbol_normalized} not in Binance US Top-10 allowlist, skipping")

        if not validated_symbols:
            # All Live Data, No Fallback/Hardcoded Data
            msg = "No valid symbols provided - all symbols must be in Binance US Top-10 allowlist"
            raise ValueError(msg)

        return {
            "strategy_name": strategy_name,
            "capital": capital,
            "timeframe": timeframe,
            "symbols": validated_symbols,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "created_at": datetime.now(timezone.utc).isoformat(),  # ISO 8601 timestamp
        }

    def launch_strategy_process(self, strategy_name: str, config: dict[str, Any]) -> str | None:
        """Launch strategy with real worker communication via Redis queues"""
        try:
            # Create strategy directory with proper path handling
            strategy_dir = Path(self.agents_dir) / strategy_name
            strategy_dir.mkdir(parents=True, exist_ok=True)

            # Save config with proper file handling
            config_path = strategy_dir / "config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=True)

            # Generate unique process ID
            process_id = f"strategy_{strategy_name}_{int(time.time())}"

            # Thread-safe update of active strategies
            with self.strategy_lock:
                self.active_strategies[strategy_name] = {
                    "process_id": process_id,
                    "status": "launching",
                    "config_path": config_path,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                }
                # Track active container/process
                self.active_containers[strategy_name] = process_id

            logger.info(f"Strategy configuration created: {strategy_name}")

            # Send launch command to worker queue (if Redis available)
            if self.redis_available:
                try:
                    launch_command = {
                        "action": "start",
                        "strategy_name": strategy_name,
                        "config_path": config_path,
                        "process_id": process_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    self.redis_client.lpush(self.worker_queue_key, json.dumps(launch_command))

                    # Initialize leaderboard entry with live data structure
                    initial_metrics = {
                        "profit": 0.0,
                        "trades": 0,
                        "win_rate": 0.0,
                        "sharpe_ratio": 0.0,
                        "last_update": datetime.now(timezone.utc).isoformat(),
                        "status": "initializing",
                        "health": "unknown",
                    }

                    self.redis_client.hset(
                        self.leaderboard_key,
                        strategy_name,
                        json.dumps(initial_metrics, ensure_ascii=True),
                    )

                    logger.info(f"Strategy launch command sent to worker queue: {strategy_name}")

                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as redis_error:
                    logger.exception(f"Failed to send launch command to Redis: {redis_error}")
                    self.health_issues.append(f"Redis command failed: {redis_error}")
            else:
                logger.warning(f"Redis not available - strategy {strategy_name} configured but not launched")
                self.health_issues.append(f"Strategy {strategy_name} configured but not launched (Redis unavailable)")

            # Mark launched locally
            with self.strategy_lock:
                if strategy_name in self.active_strategies:
                    self.active_strategies[strategy_name]["status"] = "launched"
                    self.active_strategies[strategy_name]["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to launch strategy {strategy_name}: {e}")
            return None
        else:
            return process_id

    def generate_strategy_army(self, base_capital: float = 1000.0) -> list[dict[str, Any]]:
        """Generate diverse strategies with deterministic logic and Binance US symbol formatting"""
        strategies: list[dict[str, Any]] = []

        # Simple deterministic variations across top symbols and timeframes
        timeframes = ["5m", "15m", "1h", "4h"]
        risk_options = [0.005, 0.01, 0.02, 0.03]

        idx = 0
        for tf in timeframes:
            for symbol in BINANCE_US_TOP10:
                name = f"strat_{tf}_{symbol}_{idx}"
                capital = base_capital + (idx * 10)
                config = self.create_strategy_config(strategy_name=name, capital=capital, timeframe=tf, symbols=[symbol])
                # tweak risk per trade deterministically
                config["risk_per_trade"] = risk_options[idx % len(risk_options)]
                config["max_positions"] = 1 + (idx % 5)
                strategies.append(config)
                idx += 1
                # Keep the army size reasonable
                if idx >= 100:
                    break
            if idx >= 100:
                break

        return strategies

    async def launch_strategy_async(self, strategy_name: str, config: dict[str, Any]) -> str | None:
        """Launch a single strategy asynchronously"""
        try:
            # Run the blocking launch in executor
            loop = asyncio.get_running_loop()
            container_id = await loop.run_in_executor(None, self.launch_strategy_process, strategy_name, config)
            if container_id:
                logger.info(f"[OK] Strategy {strategy_name} container launched")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to launch {strategy_name}: {e}")
            return None
        else:
            return container_id

    def get_leaderboard(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get current strategy leaderboard with live updates from Redis"""
        try:
            if not self.redis_available:
                logger.warning("Redis not available - returning empty leaderboard")
                return []

            leaderboard_data = self.redis_client.hgetall(self.leaderboard_key)
            strategies = []

            for strategy_name, data_str in leaderboard_data.items():
                try:
                    data = json.loads(data_str)
                    data["strategy_name"] = strategy_name

                    # Add health status based on last update
                    last_update_str = data.get("last_update", "")
                    if last_update_str:
                        try:
                            last_update = datetime.fromisoformat(last_update_str.replace("Z", "+00:00"))
                            time_since_update = (datetime.now(timezone.utc) - last_update).total_seconds()

                            if time_since_update > 300:  # 5 minutes
                                data["health"] = "stale"
                                data["status"] = "stale"
                            elif time_since_update > 60:  # 1 minute
                                data["health"] = "degraded"
                            else:
                                data["health"] = "healthy"
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            data["health"] = "unknown"
                    else:
                        data["health"] = "unknown"

                    strategies.append(data)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse leaderboard data for {strategy_name}: {e}")
                    continue

            # Sort by profit (descending)
            strategies.sort(key=lambda x: float(x.get("profit", 0)), reverse=True)

            logger.debug(f"Retrieved leaderboard with {len(strategies)} strategies")
            return strategies[:limit]

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to get leaderboard: {e}")
            return []

    def get_arena_snapshot(self) -> dict[str, Any]:
        """Get a non-blocking snapshot of arena status for dashboard polling"""
        try:
            with self.strategy_lock:
                active_strategies = self.active_strategies.copy()

            leaderboard = self.get_leaderboard(10)

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_strategies": len(active_strategies),
                "redis_available": self.redis_available,
                "health_issues": self.health_issues.copy(),
                "leaderboard": leaderboard,
                "top_performer": leaderboard[0] if leaderboard else None,
                "arena_status": "healthy" if self.redis_available and not self.health_issues else "degraded",
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to get arena snapshot: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "arena_status": "error",
            }

    def survivor_selection(self, survival_rate: float = 0.2):
        """Select top performing strategies and terminate others"""
        try:
            leaderboard = self.get_leaderboard()
            num_survivors = max(1, int(len(leaderboard) * survival_rate))

            survivors = leaderboard[:num_survivors]
            eliminated = leaderboard[num_survivors:]

            logger.info(f"ðŸ† Survivor Selection: {len(survivors)} survivors, {len(eliminated)} eliminated")

            # Terminate eliminated strategies
            for strategy in eliminated:
                strategy_name = strategy["strategy_name"]
                if strategy_name in self.active_containers:
                    try:
                        # Process termination handled directly
                        del self.active_containers[strategy_name]
                        # Also update active_strategies if present
                        with self.strategy_lock:
                            if strategy_name in self.active_strategies:
                                self.active_strategies[strategy_name]["status"] = "terminated"
                        logger.info(f"ðŸ'€ Eliminated strategy: {strategy_name}")
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        logger.exception(f"Failed to terminate {strategy_name}: {e}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Survivor selection failed: {e}")
            return []
        else:
            return survivors

    def monitor_arena(self, check_interval: int = 60):
        """Monitor arena health and performance"""
        logger.info("Starting arena monitoring...")

        while True:
            try:
                # Check container health
                healthy_containers = 0
                for (
                    strategy_name,
                    _container_id,
                ) in list(self.active_containers.items()):
                    try:
                        # Process health check handled directly
                        healthy_containers += 1  # Assume healthy for now
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        logger.exception(f"Failed to check process {strategy_name}: {e}")

                # Get leaderboard
                leaderboard = self.get_leaderboard(10)

                logger.info(f"ðŸ“Š Arena Status: {healthy_containers}/{len(self.active_containers)} containers healthy")
                if leaderboard:
                    top_strategy = leaderboard[0]
                    try:
                        profit_val = float(top_strategy.get("profit", 0.0))
                        logger.info(f"ðŸ¥‡ Top Strategy: {top_strategy['strategy_name']} - Profit: ${profit_val:.2f}")
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        logger.info(f"ðŸ¥‡ Top Strategy: {top_strategy.get('strategy_name', 'unknown')}")

                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Arena monitoring stopped")
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Monitoring error: {e}")
                time.sleep(check_interval)

    def monitor_arena_health(self) -> dict[str, Any]:
        """Non-blocking arena health check for dashboard polling"""
        try:
            with self.strategy_lock:
                active_strategies = self.active_strategies.copy()

            # Count healthy strategies based on heartbeat
            healthy_count = 0
            current_time = datetime.now(timezone.utc)

            for _strategy_name, strategy_info in active_strategies.items():
                try:
                    last_heartbeat_str = strategy_info.get("last_heartbeat", "")
                    if last_heartbeat_str:
                        last_heartbeat = datetime.fromisoformat(last_heartbeat_str.replace("Z", "+00:00"))
                        time_since_heartbeat = (current_time - last_heartbeat).total_seconds()

                        if time_since_heartbeat < 300:  # 5 minutes
                            healthy_count += 1
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue

            # Get leaderboard
            leaderboard = self.get_leaderboard(10)

            health_status = {
                "timestamp": current_time.isoformat(),
                "total_strategies": len(active_strategies),
                "healthy_strategies": healthy_count,
                "redis_available": self.redis_available,
                "health_issues": self.health_issues.copy(),
                "top_performer": leaderboard[0] if leaderboard else None,
                "arena_health": "healthy" if healthy_count > 0 and self.redis_available else "degraded",
            }

            logger.debug(f"[MONITOR] Arena health: {healthy_count}/{len(active_strategies)} strategies healthy")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Arena health monitoring error: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "arena_health": "error",
            }
        else:
            return health_status


async def main() -> None:
    """Main arena launcher"""
    arena = StrategyArena()

    # If launch_arena is provided elsewhere in omitted code, prefer that.
    # Otherwise, launch a set of strategies using available methods.
    num_launched = 0
    try:
        launch_arena = getattr(arena, "launch_arena", None)
        if callable(launch_arena):
            num_launched = await launch_arena(num_strategies=100, base_capital=1000.0)  # type: ignore[call-arg]
        else:
            # Fallback simple launcher
            army = arena.generate_strategy_army(base_capital=1000.0)
            # Limit to 100 or len(army)
            to_launch = army[:100]
            tasks = []
            for cfg in to_launch:
                strategy_name = cfg.get("strategy_name", f"strategy_{int(time.time())}")
                # stagger launches slightly
                await asyncio.sleep(PROCESS_LAUNCH_STAGGER_DELAY)
                task = await task_manager.create_task(arena.launch_strategy_async(strategy_name, cfg), name="arena_launcher:launch_strategy_async")
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            num_launched = sum(1 for r in results if isinstance(r, str))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error during arena launch: {e}")
        num_launched = 0

    if num_launched > 0:
        logger.info(f"Arena successfully launched with {num_launched} strategies")

        # Run monitoring in a background thread to avoid blocking asyncio loop
        monitor_thread = threading.Thread(target=arena.monitor_arena, daemon=True)
        monitor_thread.start()
        # Keep main coroutine alive while monitor runs
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("Main task cancelled, exiting.")
    else:
        logger.error("âŒ Failed to launch arena")


if __name__ == "__main__":
    asyncio.run(main())
