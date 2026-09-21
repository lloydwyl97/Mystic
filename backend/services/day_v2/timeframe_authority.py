"""DAY V2 Timeframe Authority.

Each timeframe in the DAY V2 system has explicit authority over specific
decisions. This module defines what each timeframe is and is not allowed
to do.

KEY INVARIANT: An open (still-forming) 15m bar, a partial 1m bar, or a
tick alone must NOT trigger a normal DAY thesis exit. Only CLOSED bars of
the appropriate role may trigger normal exits or invalidate thesis. The
CATASTROPHIC role is the sole exception (event-driven, not bar-driven).

This prevents the current production problem of exits being triggered on
intra-bar noise rather than confirmed structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TimeframeRole(str, Enum):
    PRIMARY = "PRIMARY"  # 15m: opportunity formation, normal thesis evaluation, exit decisions
    CONTEXT = "CONTEXT"  # 1H: directional state, context scoring
    REGIME = "REGIME"  # 4H: regime classification, structural boundaries
    EXECUTION = "EXECUTION"  # tick/1m: execution quality, spread/impact safety only
    CATASTROPHIC = "CATASTROPHIC"  # tick/1m: extreme protection only


class CandleCompleteness(str, Enum):
    CLOSED = "CLOSED"  # Bar is complete — authoritative for thesis decisions
    OPEN = "OPEN"  # Bar is still forming — not authoritative for normal exits
    STALE = "STALE"  # Data is older than expected — safety concern only


@dataclass(frozen=True)
class TimeframeAuthority:
    """Defines what decisions a timeframe is permitted to drive."""

    role: TimeframeRole
    bar_seconds: int
    may_invalidate_thesis: bool  # only CLOSED bars of PRIMARY/CONTEXT/REGIME
    may_trigger_normal_exit: bool  # only CLOSED PRIMARY bars
    may_trigger_catastrophic_exit: bool  # CATASTROPHIC always; PRIMARY CLOSED always
    may_form_opportunity: bool  # PRIMARY and CONTEXT closed only
    may_confirm_structural_reset: bool  # REGIME closed only


# Canonical authority table.
DAY_V2_TIMEFRAME_AUTHORITIES: dict[TimeframeRole, TimeframeAuthority] = {
    TimeframeRole.PRIMARY: TimeframeAuthority(
        role=TimeframeRole.PRIMARY,
        bar_seconds=900,  # 15m
        may_invalidate_thesis=True,
        may_trigger_normal_exit=True,
        may_trigger_catastrophic_exit=True,
        may_form_opportunity=True,
        may_confirm_structural_reset=False,
    ),
    TimeframeRole.CONTEXT: TimeframeAuthority(
        role=TimeframeRole.CONTEXT,
        bar_seconds=3600,  # 1H
        may_invalidate_thesis=True,
        may_trigger_normal_exit=False,
        may_trigger_catastrophic_exit=False,
        may_form_opportunity=True,
        may_confirm_structural_reset=False,
    ),
    TimeframeRole.REGIME: TimeframeAuthority(
        role=TimeframeRole.REGIME,
        bar_seconds=14400,  # 4H
        may_invalidate_thesis=True,
        may_trigger_normal_exit=False,
        may_trigger_catastrophic_exit=False,
        may_form_opportunity=False,
        may_confirm_structural_reset=True,
    ),
    TimeframeRole.EXECUTION: TimeframeAuthority(
        role=TimeframeRole.EXECUTION,
        bar_seconds=60,  # 1m
        may_invalidate_thesis=False,
        may_trigger_normal_exit=False,
        may_trigger_catastrophic_exit=False,
        may_form_opportunity=False,
        may_confirm_structural_reset=False,
    ),
    TimeframeRole.CATASTROPHIC: TimeframeAuthority(
        role=TimeframeRole.CATASTROPHIC,
        bar_seconds=0,  # tick/event-driven
        may_invalidate_thesis=False,
        may_trigger_normal_exit=False,
        may_trigger_catastrophic_exit=True,
        may_form_opportunity=False,
        may_confirm_structural_reset=False,
    ),
}


class TimeframeAuthorityViolation(Exception):  # noqa: N818
    """Raised when a timeframe attempts an action it is not authorized to take."""

    def __init__(
        self,
        role: TimeframeRole,
        completeness: CandleCompleteness,
        is_catastrophic: bool,
        reason: str,
    ) -> None:
        self.role = role
        self.completeness = completeness
        self.is_catastrophic = is_catastrophic
        self.reason = reason
        super().__init__(f"TimeframeAuthorityViolation: role={role!r} completeness={completeness!r} is_catastrophic={is_catastrophic} — {reason}")


def can_trigger_exit(
    role: TimeframeRole,
    completeness: CandleCompleteness,
    is_catastrophic: bool,
) -> tuple[bool, str]:
    """Return (allowed, reason).

    For normal exits: only PRIMARY CLOSED bars.
    For catastrophic exits: CATASTROPHIC role always, PRIMARY CLOSED always.
    An OPEN bar of any role cannot trigger a normal exit.
    A STALE bar cannot trigger any exit (safety unknown).
    """
    authority = DAY_V2_TIMEFRAME_AUTHORITIES[role]

    if completeness == CandleCompleteness.STALE:
        return (
            False,
            f"{role.value} bar is STALE — cannot trigger any exit until data refreshes",
        )

    if is_catastrophic:
        if authority.may_trigger_catastrophic_exit:
            return True, f"{role.value} may trigger catastrophic exit"
        return (
            False,
            f"{role.value} is not authorized to trigger catastrophic exits",
        )

    # Normal exit
    if completeness == CandleCompleteness.OPEN:
        return (
            False,
            f"{role.value} bar is still OPEN — a forming bar cannot trigger a normal exit",
        )

    if authority.may_trigger_normal_exit:
        return True, f"{role.value} CLOSED bar may trigger normal exit"

    return (
        False,
        f"{role.value} is not authorized to trigger normal exits (only PRIMARY CLOSED bars may)",
    )


def can_invalidate_thesis(
    role: TimeframeRole,
    completeness: CandleCompleteness,
) -> tuple[bool, str]:
    """Return (allowed, reason).

    Only CLOSED bars of PRIMARY, CONTEXT, or REGIME may invalidate the thesis.
    """
    authority = DAY_V2_TIMEFRAME_AUTHORITIES[role]

    if completeness == CandleCompleteness.STALE:
        return (
            False,
            f"{role.value} bar is STALE — thesis invalidation deferred until fresh data",
        )

    if completeness == CandleCompleteness.OPEN:
        return (
            False,
            f"{role.value} bar is still OPEN — forming bar cannot invalidate thesis",
        )

    if authority.may_invalidate_thesis:
        return True, f"{role.value} CLOSED bar may invalidate thesis"

    return (
        False,
        f"{role.value} is not authorized to invalidate thesis",
    )


def validate_exit_signal(
    timeframe_role: TimeframeRole,
    bar_completeness: CandleCompleteness,
    is_catastrophic: bool,
) -> None:
    """Validate that the given timeframe + completeness is allowed to trigger an exit.

    Raises TimeframeAuthorityViolation if not allowed.
    """
    allowed, reason = can_trigger_exit(timeframe_role, bar_completeness, is_catastrophic)
    if not allowed:
        raise TimeframeAuthorityViolation(
            role=timeframe_role,
            completeness=bar_completeness,
            is_catastrophic=is_catastrophic,
            reason=reason,
        )
