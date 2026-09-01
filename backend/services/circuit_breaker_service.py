#!/usr/bin/env python3
"""
Circuit Breaker Service for Fault Tolerance and Resilience
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from backend.database_schema import get_state, set_state

# Import task manager for proper task tracking
try:
    from backend.services.task_manager import task_manager
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    task_manager = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


ACCOUNT_FAILSAFE_EQUITY_FRACTION = 0.90
EQUITY_CIRCUIT_BREAKER_FRACTION = 0.93
# One-tick cash+positions double-count during EXIT_RESIDUAL_PENDING (Ocean
# 2026-09-01: $233 → $296.95) must not become the 7% watermark.
SESSION_HIGH_MAX_ONE_TICK_INCREASE = 0.08
SESSION_HIGH_UNEXPLAINED_GAP = 0.12
SESSION_HIGH_STABLE_BAND = 0.03


def account_failsafe_tripped(current_equity: float, principal: float) -> bool:
    """Authoritative failsafe predicate: equity collapsed vs principal."""
    prin = float(principal or 0.0)
    if prin <= 0.0:
        return False
    return float(current_equity or 0.0) <= prin * ACCOUNT_FAILSAFE_EQUITY_FRACTION


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: tuple = (Exception,)  # Exception types to count as failures
    success_threshold: int = 3  # Successes needed in half-open state


@dataclass
class CircuitStats:
    total_calls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None


class CircuitBreaker:
    """Circuit breaker implementation for service resilience"""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"[RELOAD] Circuit {self.name}: HALF-OPEN - Testing recovery")
                else:
                    msg = f"Circuit {self.name} is OPEN. Next retry in {self._time_until_retry():.1f}s"
                    raise CircuitBreakerOpenError(msg)

            try:
                self.stats.total_calls += 1
                result = await func(*args, **kwargs)

                await self._record_success()
            except self.config.expected_exception:
                await self._record_failure()
                raise
            else:
                return result

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.stats.last_failure_time is None:
            return True
        return (time.time() - self.stats.last_failure_time) >= self.config.recovery_timeout

    def _time_until_retry(self) -> float:
        """Calculate time until next retry attempt"""
        if self.stats.last_failure_time is None:
            return 0.0
        elapsed = time.time() - self.stats.last_failure_time
        return max(0.0, self.config.recovery_timeout - elapsed)

    async def _record_success(self):
        """Record successful call"""
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN and self.stats.consecutive_successes >= self.config.success_threshold:
            self.state = CircuitState.CLOSED
            self.stats.consecutive_successes = 0
            logger.info(f"[OK] Circuit {self.name}: CLOSED - Service recovered")

    async def _record_failure(self):
        """Record failed call"""
        self.stats.total_failures += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"[ERROR] Circuit {self.name}: OPEN - Recovery failed")
        elif self.state == CircuitState.CLOSED and self.stats.consecutive_failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"[ERROR] Circuit {self.name}: OPEN - Too many failures ({self.stats.consecutive_failures})")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "total_failures": self.stats.total_failures,
            "consecutive_failures": self.stats.consecutive_failures,
            "consecutive_successes": self.stats.consecutive_successes,
            "failure_rate": (self.stats.total_failures / self.stats.total_calls) if self.stats.total_calls > 0 else 0,
            "last_failure_time": self.stats.last_failure_time,
            "last_success_time": self.stats.last_success_time,
            "time_until_retry": self._time_until_retry(),
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""


class CircuitBreakerService:
    """Service managing multiple circuit breakers"""

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}
        self._monitoring_task: asyncio.Task | None = None

    def get_or_create_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get existing breaker or create new one"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]

    async def start_monitoring(self):
        """Start background monitoring task"""
        if self._monitoring_task is None:
            if task_manager is not None:
                self._monitoring_task = await task_manager.create_task(self._monitor_breakers(), name="circuit_breaker_service:monitor_breakers")
            else:
                self._monitoring_task = asyncio.create_task(self._monitor_breakers())
            logger.info("Circuit breaker monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None
            logger.info("Circuit breaker monitoring stopped")

    async def _monitor_breakers(self):
        """Monitor circuit breaker health"""
        while True:
            try:
                # Log stats for unhealthy breakers every 30 seconds
                unhealthy_breakers = [name for name, breaker in self.breakers.items() if breaker.state != CircuitState.CLOSED]

                if unhealthy_breakers:
                    stats = {name: self.breakers[name].get_stats() for name in unhealthy_breakers}
                    logger.warning(f"[WARN] Unhealthy circuit breakers: {stats}")

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                # BUG #45 FIX: Clean exit on cancellation
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Circuit breaker monitoring error: {e}")
                await asyncio.sleep(10)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}


# A persisted "active" hard-kill flag older than this is not trusted blindly
# on cold start. Daily loss freeze is inherently a per-trading-day mechanism;
# anything older than a day is almost certainly from a prior session/crash,
# not the current one. Override via CIRCUIT_BREAKER_STALE_STATE_MAX_AGE_SEC.
def _stale_state_max_age_sec() -> float:
    import os

    try:
        return float(os.getenv("CIRCUIT_BREAKER_STALE_STATE_MAX_AGE_SEC", "86400"))
    except (TypeError, ValueError):
        return 86400.0


class TradingCircuitBreaker:
    """
    Hard kill protection for trading system - overrides all other logic.

    Cold-start contract: a persisted "active" hard-kill flag is only trusted
    as *currently* active if its timestamp is recent (see
    _stale_state_max_age_sec). Older persisted "active" state is loaded as
    pending revalidation — never silently either (a) treated as still active
    forever, or (b) blindly cleared. The first live equity/pnl check after
    startup (revalidate_from_live_data / check_all_hard_kills) confirms or
    clears it, and the transition is logged and observable via
    get_cold_start_status().
    """

    def __init__(self):
        self.daily_loss_freeze_active = False
        self.equity_circuit_breaker_active = False
        self.account_failsafe_active = False
        self.session_high_equity = 0.0
        self.last_stable_equity = 0.0
        self.last_daily_reset = None

        # Cold-start observability (see get_cold_start_status()).
        self.needs_revalidation: set[str] = set()
        self.persisted_state_timestamp: str | None = None
        self.persisted_state_age_sec: float | None = None
        self.startup_changed_state: bool = False
        self.last_dependency_check_at: float | None = None

        # Load state from SQLite on initialization
        self._load_circuit_state()

    def _maybe_reset_daily_session_high(self, current_equity: float) -> None:
        """
        Reset session_high_equity to today's equity once per new UTC day.

        Without this, session_high_equity is a permanent all-time watermark
        that only ever increases (see update_session_high). Any equity dip of
        >7% from that peak — from a real drawdown OR a transient bug — then
        latches PAUSE_BUYS until equity fully recovers to within 7% of the
        highest point the account has EVER reached, with no way to clear on
        its own (evidence: a 2026-07-13 bookkeeping bug depressed equity by
        ~$3,750, and the resulting PAUSE_BUYS stayed engaged for 2.5 days
        straight because equity could never climb back above the stale
        pre-bug peak). A daily reset bounds this to "one bad day", not
        "however long it takes to beat an old high-water mark forever".
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_daily_reset != today:
            if self.last_daily_reset is not None and current_equity > 0:
                logger.info(
                    "[CIRCUIT BREAKER] DAILY_SESSION_HIGH_RESET %s -> %s | session_high %.2f -> %.2f",
                    self.last_daily_reset,
                    today,
                    self.session_high_equity,
                    current_equity,
                )
            self.session_high_equity = current_equity
            if current_equity > 0:
                self.last_stable_equity = current_equity
            self.last_daily_reset = today

    def _note_stable_equity(self, current_equity: float) -> None:
        eq = float(current_equity or 0.0)
        if eq <= 0:
            return
        prev = float(self.last_stable_equity or 0.0)
        if prev <= 0 or abs(eq - prev) / prev <= SESSION_HIGH_STABLE_BAND:
            self.last_stable_equity = eq

    def _revert_spiked_session_high(self, current_equity: float) -> bool:
        """Drop a watermark that is not supported by recent stable equity.

        A real 7%+ crash keeps last_stable near the old high. A residual-fill
        double-count leaves last_stable near the true book and a spiked high.
        """
        high = float(self.session_high_equity or 0.0)
        eq = float(current_equity or 0.0)
        if high <= 0 or eq <= 0:
            return False
        if high <= eq * (1.0 + SESSION_HIGH_UNEXPLAINED_GAP):
            return False
        stable = float(self.last_stable_equity or 0.0)
        if stable > 0 and stable >= high * (1.0 - SESSION_HIGH_UNEXPLAINED_GAP):
            return False
        replacement = stable if stable > 0 else eq
        logger.warning(
            "[CIRCUIT BREAKER] SESSION_HIGH_SPIKE_REVERT high=%.2f current=%.2f stable=%.2f -> %.2f",
            high,
            eq,
            stable,
            replacement,
        )
        self.session_high_equity = replacement
        return True

    def update_session_high(self, current_equity: float, *, residual_pending: bool = False) -> None:
        """Update session high equity for circuit breaker calculations"""
        eq = float(current_equity or 0.0)
        self._maybe_reset_daily_session_high(eq)
        if residual_pending:
            logger.warning(
                "[CIRCUIT BREAKER] SESSION_HIGH_SKIP residual_pending equity=%.2f high=%.2f",
                eq,
                self.session_high_equity,
            )
            return
        if eq <= 0:
            return
        self._revert_spiked_session_high(eq)
        if self.session_high_equity > 0 and eq > self.session_high_equity * (1.0 + SESSION_HIGH_MAX_ONE_TICK_INCREASE):
            logger.warning(
                "[CIRCUIT BREAKER] SESSION_HIGH_SPIKE_REJECT current=%.2f high=%.2f",
                eq,
                self.session_high_equity,
            )
            return
        self.session_high_equity = max(self.session_high_equity, eq)
        self._note_stable_equity(eq)

    def check_daily_loss_freeze(self, realized_pnl_today: float, equity: float) -> bool:
        """
        DAILY LOSS FREEZE: Block new entries if daily loss exceeds threshold
        if realized_pnl_today <= -2.5% of equity: block_new_entries = True, mode = SAFE
        """
        threshold = equity * -0.025  # -2.5%
        if realized_pnl_today <= threshold:
            if not self.daily_loss_freeze_active:
                self.daily_loss_freeze_active = True
                logger.critical(f"[HARD KILL] DAILY LOSS FREEZE ACTIVATED | PnL: ${realized_pnl_today:.2f} <= ${threshold:.2f} (-2.5%)")
                return True
        elif self.daily_loss_freeze_active:
            self.daily_loss_freeze_active = False
            logger.info("[HARD KILL] DAILY LOSS FREEZE DEACTIVATED | PnL recovered")
        return self.daily_loss_freeze_active

    def check_equity_circuit_breaker(self, current_equity: float) -> bool:
        """
        EQUITY CIRCUIT BREAKER: Force safe mode and reduce exposure
        if equity <= session_high * 0.93: force_safe_mode(), reduce_exposure_to_30_percent()
        """
        if self.session_high_equity == 0:
            return False

        threshold = self.session_high_equity * EQUITY_CIRCUIT_BREAKER_FRACTION  # 7% drawdown from session high
        if current_equity <= threshold:
            if not self.equity_circuit_breaker_active:
                self.equity_circuit_breaker_active = True
                logger.critical(f"[HARD KILL] EQUITY CIRCUIT BREAKER ACTIVATED | Equity: ${current_equity:.2f} <= ${threshold:.2f} (7% from session high ${self.session_high_equity:.2f})")
                return True
        elif self.equity_circuit_breaker_active:
            self.equity_circuit_breaker_active = False
            logger.info("[HARD KILL] EQUITY CIRCUIT BREAKER DEACTIVATED | Equity recovered")
        return self.equity_circuit_breaker_active

    def check_account_failsafe(self, current_equity: float, principal: float) -> bool:
        """
        ACCOUNT FAILSAFE: pause new entries when equity collapses vs principal.

        if equity <= principal * 0.90: engage failsafe (PAUSE_BUYS via caller).
        Clears automatically when equity recovers above the threshold — previously
        this flag latched forever, so a single bad/transient equity reading
        (e.g. cash-only mid-mark while positions still open) froze buys for days
        after the book had already healed.
        """
        if principal <= 0:
            return bool(self.account_failsafe_active)
        threshold = float(principal) * ACCOUNT_FAILSAFE_EQUITY_FRACTION
        eq = float(current_equity or 0.0)
        if account_failsafe_tripped(eq, principal):
            if not self.account_failsafe_active:
                self.account_failsafe_active = True
                logger.critical(
                    "[HARD KILL] ACCOUNT FAILSAFE ACTIVATED | Equity: $%.2f <= $%.2f (10%% from principal $%.2f)",
                    eq,
                    threshold,
                    float(principal),
                )
            return True
        if self.account_failsafe_active:
            self.account_failsafe_active = False
            logger.info(
                "[HARD KILL] ACCOUNT FAILSAFE DEACTIVATED | Equity recovered to $%.2f > $%.2f (principal $%.2f)",
                eq,
                threshold,
                float(principal),
            )
        return False

    def check_all_hard_kills(self, portfolio_data: dict, market_data: dict | None = None, *, skip_sync_persist: bool = False) -> dict:
        """
        Check all hard kill conditions - returns actions to take
        Note: market_data parameter reserved for future use

        skip_sync_persist: set True when the caller (e.g.
        check_all_hard_kills_async) will persist via the async path right
        after — avoids the harmless-but-noisy "cannot persist synchronously
        in async context" warning on every call from an async trading loop.
        """
        # Prevent unused parameter warning - parameter kept for future API compatibility
        _ = market_data
        current_equity = portfolio_data.get("total_equity", 0)
        principal = portfolio_data.get("principal", 0.0)
        realized_pnl_today = portfolio_data.get("realized_pnl_today", 0)
        residual_pending = bool(portfolio_data.get("residual_pending", False))

        # Update session high
        self.update_session_high(current_equity, residual_pending=residual_pending)

        results = {
            "daily_loss_freeze": self.check_daily_loss_freeze(realized_pnl_today, current_equity),
            "equity_circuit_breaker": self.check_equity_circuit_breaker(current_equity),
            "account_failsafe": self.check_account_failsafe(current_equity, principal),
        }

        # Determine actions
        actions = {
            "force_safe_mode": any(results.values()),
            "block_new_entries": results["daily_loss_freeze"] or results["equity_circuit_breaker"],
            "reduce_exposure_30pct": results["equity_circuit_breaker"],
            "close_all_positions": results["account_failsafe"],
            "pause_trading": results["account_failsafe"],
        }

        # Persist state changes to SQLite (sync path - bails if in async context)
        if not skip_sync_persist:
            self._persist_circuit_state()

        return {"conditions": results, "actions": actions, "any_active": any(results.values())}

    async def check_all_hard_kills_async(self, portfolio_data: dict, market_data: dict | None = None) -> dict:
        """Async variant: check conditions and persist via async (use from async context)."""
        result = self.check_all_hard_kills(portfolio_data, market_data, skip_sync_persist=True)
        await self.persist_circuit_state_async()
        return result

    def _apply_loaded_circuit_data(self, circuit_data: dict[str, Any]) -> None:
        """
        Shared cold-start validation for both sync and async load paths.

        A persisted flag of True is only adopted as currently-active if its
        `updated_at` timestamp is within _stale_state_max_age_sec(). Otherwise
        the flag starts False (does not silently pause a healthy new runtime
        forever) but is recorded in `needs_revalidation` so the next live
        equity/pnl check re-confirms or clears it explicitly and observably —
        it is never simply discarded.
        """
        from datetime import datetime, timezone

        updated_at_raw = circuit_data.get("updated_at")
        self.persisted_state_timestamp = updated_at_raw
        age_sec: float | None = None
        if updated_at_raw:
            try:
                ts = datetime.fromisoformat(str(updated_at_raw).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
            except (ValueError, TypeError):
                age_sec = None
        self.persisted_state_age_sec = age_sec
        stale = age_sec is None or age_sec > _stale_state_max_age_sec()

        persisted = {
            "daily_loss_freeze_active": bool(circuit_data.get("daily_loss_freeze_active", False)),
            "equity_circuit_breaker_active": bool(circuit_data.get("equity_circuit_breaker_active", False)),
            "account_failsafe_active": bool(circuit_data.get("account_failsafe_active", False)),
        }
        self.session_high_equity = float(circuit_data.get("session_high_equity", 0.0) or 0.0)
        self.last_stable_equity = float(circuit_data.get("last_stable_equity", 0.0) or 0.0)
        self.last_daily_reset = circuit_data.get("last_daily_reset") or None

        for flag_name, persisted_value in persisted.items():
            if not persisted_value:
                setattr(self, flag_name, False)
                continue
            if stale:
                # Do not adopt a stale "active" flag as current truth. Flag it
                # for immediate revalidation against live data instead.
                setattr(self, flag_name, False)
                self.needs_revalidation.add(flag_name)
                self.startup_changed_state = True
                logger.warning(
                    "[CIRCUIT BREAKER] COLD_START_STALE_STATE flag=%s persisted_at=%s age_sec=%s max_age_sec=%.0f -> NOT trusted as active; pending live revalidation",
                    flag_name,
                    updated_at_raw,
                    round(age_sec, 1) if age_sec is not None else "unknown",
                    _stale_state_max_age_sec(),
                )
            else:
                setattr(self, flag_name, True)
                logger.info(
                    "[CIRCUIT BREAKER] COLD_START_RECENT_STATE flag=%s persisted_at=%s age_sec=%.1f -> adopted as active",
                    flag_name,
                    updated_at_raw,
                    age_sec,
                )

    def revalidate_from_live_data(self, portfolio_data: dict[str, Any]) -> dict[str, Any]:
        """
        Confirm or clear any flags pending cold-start revalidation using current
        live equity/pnl. Safe to call repeatedly — a no-op once needs_revalidation
        is empty. Records an explicit, observable transition either way.
        """
        import time as _time

        self.last_dependency_check_at = _time.time()
        if not self.needs_revalidation:
            return {"revalidated": [], "confirmed_active": [], "cleared": []}

        current_equity = float(portfolio_data.get("total_equity", 0.0) or 0.0)
        principal = float(portfolio_data.get("principal", 0.0) or 0.0)
        realized_pnl_today = float(portfolio_data.get("realized_pnl_today", 0.0) or 0.0)

        pending = set(self.needs_revalidation)
        confirmed: list[str] = []
        cleared: list[str] = []

        if "daily_loss_freeze_active" in pending:
            threshold = current_equity * -0.025
            if realized_pnl_today <= threshold:
                self.daily_loss_freeze_active = True
                confirmed.append("daily_loss_freeze_active")
            else:
                self.daily_loss_freeze_active = False
                cleared.append("daily_loss_freeze_active")
        if "equity_circuit_breaker_active" in pending:
            self.update_session_high(current_equity, residual_pending=bool(portfolio_data.get("residual_pending", False)))
            if self.session_high_equity > 0 and current_equity <= self.session_high_equity * EQUITY_CIRCUIT_BREAKER_FRACTION:
                self.equity_circuit_breaker_active = True
                confirmed.append("equity_circuit_breaker_active")
            else:
                self.equity_circuit_breaker_active = False
                cleared.append("equity_circuit_breaker_active")
        if "account_failsafe_active" in pending:
            if principal > 0 and current_equity <= principal * 0.90:
                self.account_failsafe_active = True
                confirmed.append("account_failsafe_active")
            else:
                self.account_failsafe_active = False
                cleared.append("account_failsafe_active")

        self.needs_revalidation.clear()
        logger.warning(
            "[CIRCUIT BREAKER] COLD_START_REVALIDATION_COMPLETE confirmed_active=%s cleared=%s equity=%.2f",
            confirmed,
            cleared,
            current_equity,
        )
        return {"revalidated": list(pending), "confirmed_active": confirmed, "cleared": cleared}

    def get_cold_start_status(self) -> dict[str, Any]:
        """Observable cold-start/circuit-breaker state for status endpoints/dashboards."""
        return {
            "daily_loss_freeze_active": self.daily_loss_freeze_active,
            "equity_circuit_breaker_active": self.equity_circuit_breaker_active,
            "account_failsafe_active": self.account_failsafe_active,
            "session_high_equity": self.session_high_equity,
            "last_stable_equity": self.last_stable_equity,
            "last_daily_reset": self.last_daily_reset,
            "persisted_state_timestamp": self.persisted_state_timestamp,
            "persisted_state_age_sec": round(self.persisted_state_age_sec, 1) if self.persisted_state_age_sec is not None else None,
            "stale_state_max_age_sec": _stale_state_max_age_sec(),
            "needs_revalidation": sorted(self.needs_revalidation),
            "startup_changed_state": self.startup_changed_state,
            "last_dependency_check_at": self.last_dependency_check_at,
        }

    def _load_circuit_state(self) -> None:
        """Load circuit breaker state from SQLite (authoritative source)"""
        try:
            import asyncio

            # Handle case where we're already in an async context (e.g., FastAPI startup)
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, defer the loading for later
                logger.info("[CIRCUIT BREAKER] Async context detected, state loading deferred to first use")
                return
            except RuntimeError:
                # No running loop, safe to create one
                pass

            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Load circuit breaker states (with cold-start staleness validation)
            circuit_data = loop.run_until_complete(get_state("risk:circuit_breakers"))
            if circuit_data:
                self._apply_loaded_circuit_data(circuit_data)

            # Load trading paused state (diagnostic only — not authoritative;
            # daily_loss_freeze_active/equity_circuit_breaker_active/account_failsafe_active
            # above are the source of truth after cold-start validation).
            paused_data = loop.run_until_complete(get_state("risk:trading_paused"))
            if paused_data:
                logger.info(f"[CIRCUIT BREAKER] Persisted trading_paused snapshot: {paused_data}")

        except Exception as e:
            logger.warning(f"[CIRCUIT BREAKER] Failed to load state from SQLite: {e}")

    def _persist_circuit_state(self) -> None:
        """Persist circuit breaker state to SQLite (authoritative source)"""
        try:
            import asyncio
            from datetime import datetime, timezone

            # Handle case where we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                logger.warning("[CIRCUIT BREAKER] Cannot persist state synchronously in async context")
                return
            except RuntimeError:
                # No running loop, safe to create one
                pass

            loop = asyncio.get_event_loop()

            # Persist circuit breaker states
            circuit_data = {
                "daily_loss_freeze_active": self.daily_loss_freeze_active,
                "equity_circuit_breaker_active": self.equity_circuit_breaker_active,
                "account_failsafe_active": self.account_failsafe_active,
                "session_high_equity": self.session_high_equity,
                "last_stable_equity": self.last_stable_equity,
                "last_daily_reset": self.last_daily_reset,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            loop.run_until_complete(set_state("risk:circuit_breakers", circuit_data))

            # Always persist current trading_paused truth (true AND false) —
            # previously only written when active, so a resolved condition
            # left a stale "trading_paused: true" row forever.
            any_active = any([self.daily_loss_freeze_active, self.equity_circuit_breaker_active, self.account_failsafe_active])
            paused_data = {
                "trading_paused": any_active,
                "pause_reason": "circuit_breaker_active" if any_active else "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            loop.run_until_complete(set_state("risk:trading_paused", paused_data))

        except Exception as e:
            logger.warning(f"[CIRCUIT BREAKER] Failed to persist state to SQLite: {e}")

    # BUG #M3 FIX: Add async variants for use from async context
    async def load_circuit_state_async(self) -> None:
        """Async version of _load_circuit_state - use from async context."""
        try:
            circuit_data = await get_state("risk:circuit_breakers")
            if circuit_data:
                self._apply_loaded_circuit_data(circuit_data)
            paused_data = await get_state("risk:trading_paused")
            if paused_data:
                logger.info(f"[CIRCUIT BREAKER] Persisted trading_paused snapshot (async): {paused_data}")
        except Exception as e:
            logger.warning(f"[CIRCUIT BREAKER] Failed to load state (async): {e}")

    async def persist_circuit_state_async(self) -> None:
        """Async version of _persist_circuit_state - use from async context.

        Throttled: skip SQLite write when breaker flags/session values are unchanged
        unless CIRCUIT_BREAKER_PERSIST_MIN_INTERVAL_SEC has elapsed (heartbeat).
        """
        try:
            import os
            import time
            from datetime import datetime, timezone

            any_active = any([self.daily_loss_freeze_active, self.equity_circuit_breaker_active, self.account_failsafe_active])
            fingerprint = (
                bool(self.daily_loss_freeze_active),
                bool(self.equity_circuit_breaker_active),
                bool(self.account_failsafe_active),
                float(self.session_high_equity or 0.0),
                float(self.last_stable_equity or 0.0),
                str(self.last_daily_reset or ""),
                bool(any_active),
            )
            now = time.time()
            try:
                min_interval = float(os.getenv("CIRCUIT_BREAKER_PERSIST_MIN_INTERVAL_SEC", "60") or "60")
            except (TypeError, ValueError):
                min_interval = 60.0
            last_fp = getattr(self, "_last_persist_fingerprint", None)
            last_ts = float(getattr(self, "_last_persist_ts", 0.0) or 0.0)
            if last_fp == fingerprint and (now - last_ts) < max(5.0, min_interval):
                return

            circuit_data = {
                "daily_loss_freeze_active": self.daily_loss_freeze_active,
                "equity_circuit_breaker_active": self.equity_circuit_breaker_active,
                "account_failsafe_active": self.account_failsafe_active,
                "session_high_equity": self.session_high_equity,
                "last_stable_equity": self.last_stable_equity,
                "last_daily_reset": self.last_daily_reset,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            await set_state("risk:circuit_breakers", circuit_data)
            paused_data = {
                "trading_paused": any_active,
                "pause_reason": "circuit_breaker_active" if any_active else "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            await set_state("risk:trading_paused", paused_data)
            self._last_persist_fingerprint = fingerprint
            self._last_persist_ts = now
            logger.debug("[CIRCUIT BREAKER] State persisted (async)")
        except Exception as e:
            logger.warning(f"[CIRCUIT BREAKER] Failed to persist state (async): {e}")


# Global circuit breaker service instance
circuit_breaker_service = CircuitBreakerService()

# Global trading circuit breaker instance
trading_circuit_breaker = TradingCircuitBreaker()


# Convenience functions for common use cases
def get_api_breaker(name: str) -> CircuitBreaker:
    """Get circuit breaker configured for personal use"""
    config = CircuitBreakerConfig(
        failure_threshold=10,  # Much higher threshold for personal use
        recovery_timeout=5.0,  # Quick recovery for personal system
        success_threshold=1,  # Single success resets circuit
    )
    return circuit_breaker_service.get_or_create_breaker(f"api_{name}", config)


def get_database_breaker(name: str) -> CircuitBreaker:
    """Get circuit breaker configured for database operations"""
    config = CircuitBreakerConfig(
        failure_threshold=5,  # More tolerant for DB issues
        recovery_timeout=60.0,  # Longer recovery time for DB
        success_threshold=3,
    )
    return circuit_breaker_service.get_or_create_breaker(f"db_{name}", config)


def get_external_service_breaker(name: str) -> CircuitBreaker:
    """Get circuit breaker configured for external services"""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=120.0,  # Longer timeout for external services
        success_threshold=3,
    )
    return circuit_breaker_service.get_or_create_breaker(f"external_{name}", config)
