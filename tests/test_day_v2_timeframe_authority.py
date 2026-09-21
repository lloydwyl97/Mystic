"""Tests for DAY V2 timeframe authority rules."""

import pytest

from backend.services.day_v2.timeframe_authority import (
    DAY_V2_TIMEFRAME_AUTHORITIES,
    CandleCompleteness,
    TimeframeAuthorityViolation,
    TimeframeRole,
    can_invalidate_thesis,
    can_trigger_exit,
    validate_exit_signal,
)


def test_primary_closed_may_trigger_normal_exit():
    ok, reason = can_trigger_exit(TimeframeRole.PRIMARY, CandleCompleteness.CLOSED, is_catastrophic=False)
    assert ok is True, reason


def test_primary_open_may_not_trigger_normal_exit():
    ok, reason = can_trigger_exit(TimeframeRole.PRIMARY, CandleCompleteness.OPEN, is_catastrophic=False)
    assert ok is False, reason
    assert "OPEN" in reason or "forming" in reason.lower()


def test_execution_timeframe_may_not_trigger_normal_exit():
    ok, reason = can_trigger_exit(TimeframeRole.EXECUTION, CandleCompleteness.CLOSED, is_catastrophic=False)
    assert ok is False, reason


def test_execution_timeframe_may_not_invalidate_thesis():
    ok, reason = can_invalidate_thesis(TimeframeRole.EXECUTION, CandleCompleteness.CLOSED)
    assert ok is False, reason


def test_regime_closed_may_confirm_structural_reset():
    authority = DAY_V2_TIMEFRAME_AUTHORITIES[TimeframeRole.REGIME]
    assert authority.may_confirm_structural_reset is True


def test_regime_closed_may_not_trigger_normal_exit():
    ok, reason = can_trigger_exit(TimeframeRole.REGIME, CandleCompleteness.CLOSED, is_catastrophic=False)
    assert ok is False, reason


def test_context_closed_may_invalidate_thesis_but_not_trigger_exit():
    ok_thesis, _ = can_invalidate_thesis(TimeframeRole.CONTEXT, CandleCompleteness.CLOSED)
    assert ok_thesis is True

    ok_exit, _ = can_trigger_exit(TimeframeRole.CONTEXT, CandleCompleteness.CLOSED, is_catastrophic=False)
    assert ok_exit is False


def test_catastrophic_may_always_fire():
    ok, reason = can_trigger_exit(TimeframeRole.CATASTROPHIC, CandleCompleteness.OPEN, is_catastrophic=True)
    assert ok is True, reason


def test_validate_exit_signal_raises_for_open_primary_bar():
    with pytest.raises(TimeframeAuthorityViolation) as exc_info:
        validate_exit_signal(TimeframeRole.PRIMARY, CandleCompleteness.OPEN, is_catastrophic=False)
    err = exc_info.value
    assert err.role == TimeframeRole.PRIMARY
    assert err.completeness == CandleCompleteness.OPEN


def test_validate_exit_signal_allows_closed_primary_bar():
    # Must not raise
    validate_exit_signal(TimeframeRole.PRIMARY, CandleCompleteness.CLOSED, is_catastrophic=False)


def test_partial_1m_bar_cannot_normal_exit():
    ok, _reason = can_trigger_exit(TimeframeRole.EXECUTION, CandleCompleteness.OPEN, is_catastrophic=False)
    assert ok is False


def test_tick_cannot_normal_exit():
    ok, _reason = can_trigger_exit(TimeframeRole.CATASTROPHIC, CandleCompleteness.OPEN, is_catastrophic=False)
    assert ok is False


def test_stale_bar_cannot_trigger_any_exit():
    for role in TimeframeRole:
        ok, reason = can_trigger_exit(role, CandleCompleteness.STALE, is_catastrophic=False)
        assert ok is False, f"{role} stale should not trigger exit; got: {reason}"

        ok_cat, _ = can_trigger_exit(role, CandleCompleteness.STALE, is_catastrophic=True)
        assert ok_cat is False


def test_validate_exit_signal_violation_fields():
    with pytest.raises(TimeframeAuthorityViolation) as exc_info:
        validate_exit_signal(TimeframeRole.REGIME, CandleCompleteness.CLOSED, is_catastrophic=False)
    err = exc_info.value
    assert err.role == TimeframeRole.REGIME
    assert err.is_catastrophic is False


def test_primary_bar_seconds_is_900():
    authority = DAY_V2_TIMEFRAME_AUTHORITIES[TimeframeRole.PRIMARY]
    assert authority.bar_seconds == 900


def test_regime_bar_seconds_is_14400():
    authority = DAY_V2_TIMEFRAME_AUTHORITIES[TimeframeRole.REGIME]
    assert authority.bar_seconds == 14400


def test_catastrophic_bar_seconds_is_zero():
    authority = DAY_V2_TIMEFRAME_AUTHORITIES[TimeframeRole.CATASTROPHIC]
    assert authority.bar_seconds == 0


def test_all_roles_covered_in_authority_table():
    for role in TimeframeRole:
        assert role in DAY_V2_TIMEFRAME_AUTHORITIES, f"{role!r} missing from DAY_V2_TIMEFRAME_AUTHORITIES"
