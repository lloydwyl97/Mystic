"""
Regression: TradingCircuitBreaker cold-start must not blindly trust stale
persisted hard-kill state forever, and must not blindly clear it either.
Stale "active" flags are loaded as pending revalidation; the next live
equity/pnl check confirms or clears them explicitly and observably.

Also: the generic per-dependency CircuitBreaker (HALF_OPEN recovery) is
in-memory only (never persisted), so it is deterministic and cold-start-safe
by construction — covered here for completeness.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from backend.services.circuit_breaker_service import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    TradingCircuitBreaker,
)


def _fresh_breaker() -> TradingCircuitBreaker:
    """A TradingCircuitBreaker with no persisted state (init skips the load-from-SQLite path)."""
    tcb = object.__new__(TradingCircuitBreaker)
    tcb.daily_loss_freeze_active = False
    tcb.equity_circuit_breaker_active = False
    tcb.account_failsafe_active = False
    tcb.session_high_equity = 0.0
    tcb.last_stable_equity = 0.0
    tcb.last_daily_reset = None
    tcb.needs_revalidation = set()
    tcb.persisted_state_timestamp = None
    tcb.persisted_state_age_sec = None
    tcb.startup_changed_state = False
    tcb.last_dependency_check_at = None
    return tcb


def test_recent_valid_active_state_is_trusted():
    tcb = _fresh_breaker()
    recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    tcb._apply_loaded_circuit_data(
        {
            "daily_loss_freeze_active": True,
            "equity_circuit_breaker_active": False,
            "account_failsafe_active": False,
            "session_high_equity": 10_000.0,
            "updated_at": recent_ts,
        }
    )
    assert tcb.daily_loss_freeze_active is True
    assert tcb.needs_revalidation == set()
    assert tcb.startup_changed_state is False


def test_stale_historical_active_state_is_not_blindly_trusted():
    tcb = _fresh_breaker()
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()  # ~March-era stale row
    tcb._apply_loaded_circuit_data(
        {
            "daily_loss_freeze_active": True,
            "equity_circuit_breaker_active": True,
            "account_failsafe_active": True,
            "session_high_equity": 10_000.0,
            "updated_at": stale_ts,
        }
    )
    # Not silently treated as still active...
    assert tcb.daily_loss_freeze_active is False
    assert tcb.equity_circuit_breaker_active is False
    assert tcb.account_failsafe_active is False
    # ...but not silently discarded either — pending explicit revalidation.
    assert tcb.needs_revalidation == {"daily_loss_freeze_active", "equity_circuit_breaker_active", "account_failsafe_active"}
    assert tcb.startup_changed_state is True


def test_stale_paused_state_does_not_survive_indefinitely_once_dependency_is_healthy():
    tcb = _fresh_breaker()
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    tcb._apply_loaded_circuit_data(
        {
            "account_failsafe_active": True,
            "session_high_equity": 10_000.0,
            "updated_at": stale_ts,
        }
    )
    assert "account_failsafe_active" in tcb.needs_revalidation

    # Dependency (equity) is healthy now -> revalidation must clear it, not leave it pending forever.
    result = tcb.revalidate_from_live_data({"total_equity": 25_000.0, "principal": 25_000.0, "realized_pnl_today": 5.0})
    assert "account_failsafe_active" in result["cleared"]
    assert tcb.account_failsafe_active is False
    assert tcb.needs_revalidation == set()


def test_restart_during_active_genuine_outage_is_confirmed_not_cleared():
    """A recent-ish persisted failsafe with equity STILL below the failsafe threshold
    must be confirmed active on revalidation, not waved away."""
    tcb = _fresh_breaker()
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()  # older than daily window, still "stale" by policy
    tcb._apply_loaded_circuit_data(
        {
            "account_failsafe_active": True,
            "session_high_equity": 25_000.0,
            "updated_at": stale_ts,
        }
    )
    assert "account_failsafe_active" in tcb.needs_revalidation

    # Equity genuinely still below 90% of principal -> real, ongoing outage.
    result = tcb.revalidate_from_live_data({"total_equity": 20_000.0, "principal": 25_000.0, "realized_pnl_today": -500.0})
    assert "account_failsafe_active" in result["confirmed_active"]
    assert tcb.account_failsafe_active is True


def test_missing_timestamp_is_treated_as_stale_not_trusted():
    tcb = _fresh_breaker()
    tcb._apply_loaded_circuit_data(
        {
            "daily_loss_freeze_active": True,
            "session_high_equity": 10_000.0,
            # no updated_at at all
        }
    )
    assert tcb.daily_loss_freeze_active is False
    assert "daily_loss_freeze_active" in tcb.needs_revalidation


def test_account_failsafe_clears_when_equity_recovers():
    """Regression: Jul 31 Ocean latch — transient low equity must not freeze buys forever."""
    tcb = _fresh_breaker()
    # Trip on cash-only / depressed equity reading.
    assert tcb.check_account_failsafe(current_equity=3711.39, principal=10_000.0) is True
    assert tcb.account_failsafe_active is True
    # Book healed (cash+positions / total equity back near principal).
    assert tcb.check_account_failsafe(current_equity=9904.00, principal=10_000.0) is False
    assert tcb.account_failsafe_active is False
    # Healthy checks stay clear.
    assert tcb.check_account_failsafe(current_equity=9904.00, principal=10_000.0) is False


def test_account_failsafe_stays_active_while_equity_genuinely_depressed():
    tcb = _fresh_breaker()
    assert tcb.check_account_failsafe(current_equity=8000.0, principal=10_000.0) is True
    assert tcb.check_account_failsafe(current_equity=8500.0, principal=10_000.0) is True
    assert tcb.account_failsafe_active is True
    # Cross back above 90% of principal.
    assert tcb.check_account_failsafe(current_equity=9100.0, principal=10_000.0) is False
    assert tcb.account_failsafe_active is False


def test_check_all_hard_kills_clears_failsafe_action_on_recovery():
    tcb = _fresh_breaker()
    tcb.account_failsafe_active = True
    out = tcb.check_all_hard_kills(
        {"total_equity": 9904.0, "principal": 10_000.0, "realized_pnl_today": 0.0},
        skip_sync_persist=True,
    )
    assert out["conditions"]["account_failsafe"] is False
    assert out["actions"]["close_all_positions"] is False
    assert out["actions"]["pause_trading"] is False
    assert tcb.account_failsafe_active is False


def test_cold_start_status_is_observable():
    tcb = _fresh_breaker()
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    tcb._apply_loaded_circuit_data({"equity_circuit_breaker_active": True, "session_high_equity": 10_000.0, "updated_at": stale_ts})
    status = tcb.get_cold_start_status()
    assert status["needs_revalidation"] == ["equity_circuit_breaker_active"]
    assert status["startup_changed_state"] is True
    assert status["persisted_state_timestamp"] == stale_ts
    assert status["persisted_state_age_sec"] is not None and status["persisted_state_age_sec"] > 86400


def test_revalidate_is_a_safe_noop_when_nothing_pending():
    tcb = _fresh_breaker()
    result = tcb.revalidate_from_live_data({"total_equity": 25_000.0, "principal": 25_000.0, "realized_pnl_today": 0.0})
    assert result == {"revalidated": [], "confirmed_active": [], "cleared": []}


# --- Generic per-dependency CircuitBreaker: HALF_OPEN recovery determinism ---


@pytest.mark.asyncio
async def test_half_open_recovery_is_deterministic_on_success():
    cb = CircuitBreaker("test-dep", CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01, success_threshold=1))

    async def _fail():
        raise RuntimeError("boom")

    async def _ok():
        return "ok"

    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.02)  # past recovery_timeout
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED  # success_threshold=1 closes immediately from HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_failed_recovery_reopens_deterministically():
    cb = CircuitBreaker("test-dep-2", CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.01, success_threshold=1))

    async def _fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.02)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)  # fails again during HALF_OPEN probe
    assert cb.state == CircuitState.OPEN


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_session_high_rejects_residual_double_count_spike():
    """Ocean 2026-09-01: cash already included the XRP sell while qty was still marked."""
    tcb = _fresh_breaker()
    tcb.last_daily_reset = _today_utc()
    tcb.session_high_equity = 236.15
    tcb.last_stable_equity = 236.15
    tcb.update_session_high(296.94630321799997)
    assert tcb.session_high_equity == pytest.approx(236.15)
    assert tcb.check_equity_circuit_breaker(233.15) is False


def test_session_high_skips_update_while_residual_pending():
    tcb = _fresh_breaker()
    tcb.last_daily_reset = _today_utc()
    tcb.session_high_equity = 233.15
    tcb.last_stable_equity = 233.15
    tcb.update_session_high(296.95, residual_pending=True)
    assert tcb.session_high_equity == pytest.approx(233.15)


def test_latched_spike_watermark_reverts_and_does_not_trip():
    tcb = _fresh_breaker()
    tcb.last_daily_reset = _today_utc()
    tcb.session_high_equity = 296.95
    tcb.last_stable_equity = 236.15
    tcb.equity_circuit_breaker_active = True
    out = tcb.check_all_hard_kills(
        {"total_equity": 233.15, "principal": 236.15, "realized_pnl_today": -3.10},
        skip_sync_persist=True,
    )
    assert tcb.session_high_equity == pytest.approx(236.15)
    assert out["conditions"]["equity_circuit_breaker"] is False
    assert out["actions"]["block_new_entries"] is False
    assert tcb.equity_circuit_breaker_active is False


def test_genuine_seven_percent_drawdown_still_trips():
    tcb = _fresh_breaker()
    tcb.last_daily_reset = _today_utc()
    tcb.session_high_equity = 236.15
    tcb.last_stable_equity = 236.15
    crashed = 236.15 * 0.92
    out = tcb.check_all_hard_kills(
        {"total_equity": crashed, "principal": 236.15, "realized_pnl_today": -20.0},
        skip_sync_persist=True,
    )
    assert tcb.session_high_equity == pytest.approx(236.15)
    assert out["conditions"]["equity_circuit_breaker"] is True
    assert out["actions"]["block_new_entries"] is True


@pytest.mark.asyncio
async def test_generic_breaker_is_never_persisted_so_always_starts_closed():
    """No SQLite/Redis persistence for the generic per-dependency breaker —
    confirms it cannot inherit stale cross-restart state at all."""
    cb = CircuitBreaker("fresh-dep")
    assert cb.state == CircuitState.CLOSED
