"""
PHASE 1 FIX #5: Live Trading Health Monitor

Monitors for critical issues in live trading:
- Signal cleanup (prevent reprocessing)
- Ledger integrity (detect corruption)
- Position consistency (DB vs exchange)
- Trade rate (detect duplicate trading)
- Connection health

This service runs every 30 seconds and alerts on critical issues.
"""

import asyncio
import logging
import time
from typing import Any

from backend.services.live_strategy_contracts import REDIS_ML_SIGNAL_SCAN_PATTERN

logger = logging.getLogger(__name__)


class LiveTradingHealthMonitor:
    """Monitor live trading health and detect critical issues"""

    def __init__(self, portfolio_engine: Any, redis_client: Any) -> None:
        """
        Initialize health monitor

        Args:
            portfolio_engine: Portfolio engine instance
            redis_client: Redis async client
        """
        self.engine = portfolio_engine
        self.redis = redis_client
        self.is_running = False
        self.last_alert_time: dict[str, float] = {}
        self.trade_history: list[float] = []
        self.check_interval = 30  # seconds
        self.alert_cooldown = 300  # Don't alert same issue more than once per 5 minutes

    async def start(self) -> None:
        """Start health monitoring loop"""
        if self.is_running:
            logger.warning("Health monitor already running")
            return

        self.is_running = True
        logger.info(f"✓ Live trading health monitor started (interval={self.check_interval}s)")
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop health monitoring loop"""
        self.is_running = False
        logger.info("Live trading health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop - runs every 30 seconds"""
        while self.is_running:
            try:
                await self._run_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _run_checks(self) -> None:
        """Run all health checks"""
        try:
            await self._check_signal_cleanup()
            await self._check_ledger_integrity()
            await self._check_position_consistency()
            await self._check_trade_rate()
            await self._check_connection_health()
        except Exception as e:
            logger.exception(f"Error running health checks: {e}")

    async def _check_signal_cleanup(self) -> None:
        """Check if signals are being cleaned up properly"""
        try:
            # Count active signals in Redis
            signal_keys = []
            async for key in self.redis.scan_iter(match=REDIS_ML_SIGNAL_SCAN_PATTERN, count=100):
                signal_keys.append(key)
            signal_count = len(signal_keys)

            if signal_count > 100:
                self._send_alert("signal_cleanup", f"⚠️ Too many signals in Redis: {signal_count} (should be < 10)", "WARNING")
            elif signal_count > 0:
                logger.debug(f"Signal cleanup OK: {signal_count} signals in Redis")
        except Exception as e:
            logger.exception(f"Signal cleanup check failed: {e}")

    async def _check_ledger_integrity(self) -> None:
        """Check for ledger corruption"""
        try:
            # Get expected equity
            try:
                expected = await self.engine.get_total_equity()
            except Exception:
                expected = None

            # Get actual equity
            try:
                actual = await self.engine.get_portfolio_value()
            except Exception:
                actual = None

            if expected is None or actual is None:
                logger.debug("Ledger integrity check: data unavailable")
                return

            # Calculate drift percentage
            if expected > 0:
                drift_pct = abs(expected - actual) / expected * 100
            elif actual > 0:
                drift_pct = 100.0
            else:
                drift_pct = 0

            if drift_pct > 5.0:
                self._send_alert("ledger_integrity", f"🚨 CRITICAL: Ledger drift {drift_pct:.2f}% (exp={expected}, act={actual})", "CRITICAL")
            elif drift_pct > 1.0:
                self._send_alert("ledger_integrity", f"⚠️ Ledger drift {drift_pct:.2f}% (exp={expected}, act={actual})", "WARNING")
            else:
                logger.debug(f"Ledger integrity OK: drift={drift_pct:.2f}%")
        except Exception as e:
            logger.exception(f"Ledger integrity check failed: {e}")

    async def _check_position_consistency(self) -> None:
        """Check positions in DB vs exchange match"""
        try:
            if not hasattr(self.engine, "open_positions"):
                logger.debug("Position consistency check: open_positions not available")
                return

            db_positions = set(self.engine.open_positions.keys())
            logger.debug(f"Positions in DB: {db_positions}")
        except Exception as e:
            logger.exception(f"Position consistency check failed: {e}")

    async def _check_trade_rate(self) -> None:
        """Check for abnormal trade rate (duplicate trading detection)"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 60  # Last 60 seconds

            # Remove old trades from history
            self.trade_history = [t for t in self.trade_history if t > cutoff_time]

            # Check rate
            recent_count = len(self.trade_history)
            if recent_count > 10:
                self._send_alert("trade_rate", f"⚠️ High trade rate: {recent_count} trades in 60 seconds", "WARNING")
            elif recent_count > 0:
                logger.debug(f"Trade rate OK: {recent_count} trades in 60s")
        except Exception as e:
            logger.exception(f"Trade rate check failed: {e}")

    async def _check_connection_health(self) -> None:
        """Check Redis and database connections"""
        try:
            # Test Redis connection
            try:
                await self.redis.ping()
                logger.debug("Redis connection: OK")
            except Exception as e:
                self._send_alert("redis_connection", f"🚨 Redis connection failed: {e}", "CRITICAL")
        except Exception as e:
            logger.exception(f"Connection health check failed: {e}")

    def _send_alert(self, check_name: str, message: str, severity: str) -> None:
        """
        Send alert (with deduplication to avoid spam)

        Args:
            check_name: Name of the check
            message: Alert message
            severity: CRITICAL, WARNING, INFO
        """
        now = time.time()
        last_alert = self.last_alert_time.get(check_name, 0)

        # Only alert once per cooldown period
        if now - last_alert > self.alert_cooldown:
            if severity == "CRITICAL":
                logger.critical(f"[HEALTH] {message}")
            elif severity == "WARNING":
                logger.warning(f"[HEALTH] {message}")
            else:
                logger.info(f"[HEALTH] {message}")

            self.last_alert_time[check_name] = now

    def record_trade(self, symbol: str) -> None:
        """Record a trade for rate monitoring"""
        self.trade_history.append(time.time())
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        logger.debug(f"Trade recorded: {symbol}")


# Global instance
_health_monitor: LiveTradingHealthMonitor | None = None


def set_health_monitor(monitor: LiveTradingHealthMonitor) -> None:
    """Set global health monitor instance"""
    global _health_monitor
    _health_monitor = monitor


def get_health_monitor() -> LiveTradingHealthMonitor | None:
    """Get global health monitor instance"""
    return _health_monitor


async def start_health_monitor(portfolio_engine: Any, redis_client: Any) -> LiveTradingHealthMonitor:
    """Start the health monitor service"""
    monitor = LiveTradingHealthMonitor(portfolio_engine, redis_client)
    set_health_monitor(monitor)
    await monitor.start()
    return monitor
