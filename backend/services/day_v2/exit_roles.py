"""DAY V2 Exit Roles — shadow-only.

Five distinct exit roles, each with a specific thesis and strict input/output
contract. None of these roles may route to order execution. They return
ExitEvaluation objects for shadow logging only.

All roles call assert_no_live_authority() before any evaluation to ensure
they cannot be wired into live execution accidentally.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from backend.services.day_v2.config import (
    DAY_V2_CATASTROPHIC_ATR_MULTIPLIER,
    DAY_V2_MAX_HOLD_MINUTES,
    DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
    DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT,
)
from backend.services.day_v2.engine_identity import EngineId, assert_no_live_authority

if TYPE_CHECKING:
    pass


class ExitRole(str, Enum):
    DAY_V2_CATASTROPHIC_PROTECTION = "DAY_V2_CATASTROPHIC_PROTECTION"
    DAY_V2_STRUCTURAL_INVALIDATION = "DAY_V2_STRUCTURAL_INVALIDATION"
    DAY_V2_TIME_EXPIRATION = "DAY_V2_TIME_EXPIRATION"
    DAY_V2_WINNER_PROTECTION = "DAY_V2_WINNER_PROTECTION"
    DAY_V2_OBJECTIVE_COMPLETE = "DAY_V2_OBJECTIVE_COMPLETE"


@dataclass(frozen=True)
class ExitEvaluation:
    """Result of a shadow exit role evaluation.

    is_shadow_only is always True. This dataclass cannot be used to route
    to order execution.
    """

    role: ExitRole
    should_exit: bool
    confidence: float  # 0.0 - 1.0
    reason: str
    detail: str = ""
    net_pnl_pct: float = 0.0
    hold_minutes: float = 0.0
    is_shadow_only: bool = field(default=True)

    def __post_init__(self) -> None:
        if not self.is_shadow_only:
            raise ValueError("ExitEvaluation.is_shadow_only must be True — ExitEvaluation objects must never route to live execution")


# ---------------------------------------------------------------------------
# Role 1: Catastrophic Protection
# ---------------------------------------------------------------------------


def evaluate_catastrophic_protection(
    *,
    entry_price: float,
    current_price: float,
    atr_pct: float,
    hold_minutes: float,
) -> ExitEvaluation:
    """Evaluate extreme adverse moves independent of bar closure.

    This is the ONLY DAY V2 role that is event-driven rather than bar-driven.
    It is independent of all other roles. It does NOT check hold time, bar
    status, or thesis. It must not fire on favorable moves.

    Fires when: (entry - current) / entry >= ATR_MULTIPLIER * atr_pct
    """
    role = ExitRole.DAY_V2_CATASTROPHIC_PROTECTION

    if current_price <= 0 or entry_price <= 0:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="invalid_price",
            detail=f"entry={entry_price} current={current_price}",
            hold_minutes=hold_minutes,
        )

    adverse_move = (entry_price - current_price) / entry_price
    threshold = DAY_V2_CATASTROPHIC_ATR_MULTIPLIER * atr_pct

    if adverse_move <= 0:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="no_adverse_move",
            detail=f"current={current_price:.6f} > entry={entry_price:.6f}",
            net_pnl_pct=-adverse_move,
            hold_minutes=hold_minutes,
        )

    if adverse_move >= threshold:
        return ExitEvaluation(
            role=role,
            should_exit=True,
            confidence=1.0,
            reason="catastrophic_adverse_move",
            detail=(f"adverse_move={adverse_move:.4%} >= {DAY_V2_CATASTROPHIC_ATR_MULTIPLIER}x ATR threshold={threshold:.4%} (atr_pct={atr_pct:.4%})"),
            net_pnl_pct=-adverse_move,
            hold_minutes=hold_minutes,
        )

    return ExitEvaluation(
        role=role,
        should_exit=False,
        confidence=0.0,
        reason="within_catastrophic_threshold",
        detail=(f"adverse_move={adverse_move:.4%} < threshold={threshold:.4%}"),
        net_pnl_pct=-adverse_move,
        hold_minutes=hold_minutes,
    )


# ---------------------------------------------------------------------------
# Role 2: Structural Invalidation
# ---------------------------------------------------------------------------


def evaluate_structural_invalidation(
    *,
    entry_price: float,
    current_price: float,
    structural_anchor_price: float,
    regime: str,
    n_closed_15m_bars: int,
) -> ExitEvaluation:
    """Evaluate whether the original opportunity context is still valid.

    This role answers: 'Is the original opportunity still a valid thesis?'
    — NOT 'Is the position losing money?'

    Requires at least DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED closed 15m
    bars. A partial bar alone does NOT fire this.

    Fires when price closes below structural anchor on a closed 15m bar.
    The reason string explains WHY the structure is invalidated.
    """
    role = ExitRole.DAY_V2_STRUCTURAL_INVALIDATION
    required_bars = DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED

    if n_closed_15m_bars < required_bars:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="insufficient_closed_bars",
            detail=(f"only {n_closed_15m_bars} closed 15m bars, need >= {required_bars} before structural invalidation can fire"),
            net_pnl_pct=(current_price - entry_price) / entry_price,
        )

    if current_price < structural_anchor_price:
        relative_breach = (structural_anchor_price - current_price) / structural_anchor_price
        return ExitEvaluation(
            role=role,
            should_exit=True,
            confidence=min(1.0, relative_breach / 0.005),  # full confidence at 0.5% breach
            reason="structural_anchor_violated",
            detail=(
                f"price {current_price:.6f} closed below structural anchor "
                f"{structural_anchor_price:.6f} (breach={relative_breach:.4%}) "
                f"after {n_closed_15m_bars} closed 15m bars in {regime!r} regime — "
                f"the original opportunity setup is no longer structurally valid"
            ),
            net_pnl_pct=(current_price - entry_price) / entry_price,
        )

    return ExitEvaluation(
        role=role,
        should_exit=False,
        confidence=0.0,
        reason="structure_intact",
        detail=(f"price {current_price:.6f} above anchor {structural_anchor_price:.6f} in {regime!r} regime"),
        net_pnl_pct=(current_price - entry_price) / entry_price,
    )


# ---------------------------------------------------------------------------
# Role 3: Time Expiration
# ---------------------------------------------------------------------------


def evaluate_time_expiration(
    *,
    hold_minutes: float,
    max_hold_minutes: int,
    net_pnl_pct: float,
) -> ExitEvaluation:
    """Evaluate whether the holding horizon has passed.

    Time expiration is a safety backstop for positions that never resolved.
    It does not replace structural invalidation or winner protection.

    Does NOT fire if net_pnl_pct > 0 — a winning position is not expired;
    let winner protection handle it. max_hold_minutes must come from
    DAY_V2_MAX_HOLD_MINUTES, never from legacy coin profiles.
    """
    role = ExitRole.DAY_V2_TIME_EXPIRATION

    if hold_minutes < max_hold_minutes:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="within_hold_window",
            detail=f"hold={hold_minutes:.1f}m < max={max_hold_minutes}m",
            hold_minutes=hold_minutes,
            net_pnl_pct=net_pnl_pct,
        )

    if net_pnl_pct > 0:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="time_expired_but_position_profitable",
            detail=(f"hold={hold_minutes:.1f}m >= max={max_hold_minutes}m but net_pnl_pct={net_pnl_pct:.4%} > 0 — winner protection should handle"),
            hold_minutes=hold_minutes,
            net_pnl_pct=net_pnl_pct,
        )

    return ExitEvaluation(
        role=role,
        should_exit=True,
        confidence=0.8,
        reason="hold_horizon_expired",
        detail=(f"hold={hold_minutes:.1f}m >= max={max_hold_minutes}m and net_pnl_pct={net_pnl_pct:.4%} <= 0 — thesis predictive window has closed"),
        hold_minutes=hold_minutes,
        net_pnl_pct=net_pnl_pct,
    )


# ---------------------------------------------------------------------------
# Role 4: Winner Protection
# ---------------------------------------------------------------------------


def evaluate_winner_protection(
    *,
    entry_price: float,
    highest_price: float,
    current_price: float,
    atr_pct: float,
    hold_minutes: float,
    net_pnl_pct: float,
) -> ExitEvaluation:
    """Lock in meaningful favorable development.

    Winner protection activates ONLY after meaningful favorable development
    (MFE >= DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT, default 0.8%).

    Trail distance = max(0.5%, 1.5 * atr_pct).
    This floor is 4-10x wider than the current legacy trail (0.20-0.25%).

    Must not fire because a position moved 0.20-0.25% in our favor.
    The 1.5x ATR trail floor is intentionally much wider than the current
    legacy trailing stop to avoid premature exits on normal noise.
    """
    role = ExitRole.DAY_V2_WINNER_PROTECTION

    if entry_price <= 0 or highest_price <= 0:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="invalid_price",
            hold_minutes=hold_minutes,
            net_pnl_pct=net_pnl_pct,
        )

    mfe_pct = (highest_price - entry_price) / entry_price
    min_mfe = DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT

    if mfe_pct < min_mfe:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="mfe_gate_not_reached",
            detail=(f"mfe_pct={mfe_pct:.4%} < min_mfe_gate={min_mfe:.4%} — winner protection has not activated yet"),
            hold_minutes=hold_minutes,
            net_pnl_pct=net_pnl_pct,
        )

    # Trail distance: max(0.5%, 1.5 * ATR)
    trail_distance = max(0.005, 1.5 * atr_pct)
    trail_trigger = highest_price * (1.0 - trail_distance)

    if current_price <= trail_trigger:
        pullback_from_high = (highest_price - current_price) / highest_price
        return ExitEvaluation(
            role=role,
            should_exit=True,
            confidence=min(1.0, pullback_from_high / trail_distance),
            reason="winner_protection_trail_triggered",
            detail=(
                f"current={current_price:.6f} <= trail_trigger={trail_trigger:.6f} "
                f"(highest={highest_price:.6f}, trail_distance={trail_distance:.4%}, "
                f"1.5xATR={1.5 * atr_pct:.4%}, mfe_pct={mfe_pct:.4%})"
            ),
            hold_minutes=hold_minutes,
            net_pnl_pct=net_pnl_pct,
        )

    return ExitEvaluation(
        role=role,
        should_exit=False,
        confidence=0.0,
        reason="within_winner_protection_trail",
        detail=(f"current={current_price:.6f} > trail_trigger={trail_trigger:.6f} (mfe_pct={mfe_pct:.4%}, trail_distance={trail_distance:.4%})"),
        hold_minutes=hold_minutes,
        net_pnl_pct=net_pnl_pct,
    )


# ---------------------------------------------------------------------------
# Role 5: Objective Complete
# ---------------------------------------------------------------------------


def evaluate_objective_complete(
    *,
    entry_price: float,
    current_price: float,
    thesis_target_price: float,
    n_closed_15m_bars: int,
    net_pnl_pct: float,
) -> ExitEvaluation:
    """Evaluate whether the structural target has been reached.

    Objective complete means the target price level was reached, NOT that
    we've cleared a cost floor. It fires on target hit confirmed on a CLOSED
    15m bar — not on a 0.4% net floor.

    thesis_target_price must be set at entry based on structural analysis,
    not a fixed %-based formula.

    A position with net_pnl_pct >= 0.4% that hasn't hit structural target
    should NOT be exited by this role — let winner protection handle it.
    """
    role = ExitRole.DAY_V2_OBJECTIVE_COMPLETE

    if thesis_target_price <= entry_price:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="invalid_thesis_target",
            detail=(f"thesis_target_price={thesis_target_price:.6f} <= entry_price={entry_price:.6f} — target must be above entry"),
            net_pnl_pct=net_pnl_pct,
        )

    if n_closed_15m_bars < 1:
        return ExitEvaluation(
            role=role,
            should_exit=False,
            confidence=0.0,
            reason="no_closed_bar_yet",
            detail="objective complete requires at least one closed 15m bar",
            net_pnl_pct=net_pnl_pct,
        )

    if current_price >= thesis_target_price:
        return ExitEvaluation(
            role=role,
            should_exit=True,
            confidence=1.0,
            reason="structural_target_reached",
            detail=(f"current={current_price:.6f} >= target={thesis_target_price:.6f} on closed bar (entry={entry_price:.6f})"),
            net_pnl_pct=net_pnl_pct,
        )

    return ExitEvaluation(
        role=role,
        should_exit=False,
        confidence=0.0,
        reason="structural_target_not_reached",
        detail=(f"current={current_price:.6f} < target={thesis_target_price:.6f}"),
        net_pnl_pct=net_pnl_pct,
    )


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------


def evaluate_all_roles(
    *,
    entry_price: float,
    highest_price: float,
    current_price: float,
    structural_anchor_price: float,
    thesis_target_price: float,
    atr_pct: float,
    hold_minutes: float,
    max_hold_minutes: int = DAY_V2_MAX_HOLD_MINUTES,
    net_pnl_pct: float,
    regime: str,
    n_closed_15m_bars: int,
    engine_id: EngineId,
) -> list[ExitEvaluation]:
    """Evaluate all DAY V2 exit roles and return all results.

    MUST call assert_no_live_authority before any evaluation. This is the
    hard guard that ensures shadow-only code cannot be wired to execution.

    Returns list of ExitEvaluation for all five roles (both True and False).
    If any individual role raises, that role returns should_exit=False with
    reason="evaluation_error" — the remaining roles still run.
    """
    assert_no_live_authority(engine_id)

    results: list[ExitEvaluation] = []

    role_fns = [
        (
            ExitRole.DAY_V2_CATASTROPHIC_PROTECTION,
            lambda: evaluate_catastrophic_protection(
                entry_price=entry_price,
                current_price=current_price,
                atr_pct=atr_pct,
                hold_minutes=hold_minutes,
            ),
        ),
        (
            ExitRole.DAY_V2_STRUCTURAL_INVALIDATION,
            lambda: evaluate_structural_invalidation(
                entry_price=entry_price,
                current_price=current_price,
                structural_anchor_price=structural_anchor_price,
                regime=regime,
                n_closed_15m_bars=n_closed_15m_bars,
            ),
        ),
        (
            ExitRole.DAY_V2_TIME_EXPIRATION,
            lambda: evaluate_time_expiration(
                hold_minutes=hold_minutes,
                max_hold_minutes=max_hold_minutes,
                net_pnl_pct=net_pnl_pct,
            ),
        ),
        (
            ExitRole.DAY_V2_WINNER_PROTECTION,
            lambda: evaluate_winner_protection(
                entry_price=entry_price,
                highest_price=highest_price,
                current_price=current_price,
                atr_pct=atr_pct,
                hold_minutes=hold_minutes,
                net_pnl_pct=net_pnl_pct,
            ),
        ),
        (
            ExitRole.DAY_V2_OBJECTIVE_COMPLETE,
            lambda: evaluate_objective_complete(
                entry_price=entry_price,
                current_price=current_price,
                thesis_target_price=thesis_target_price,
                n_closed_15m_bars=n_closed_15m_bars,
                net_pnl_pct=net_pnl_pct,
            ),
        ),
    ]

    for role, fn in role_fns:
        try:
            results.append(fn())
        except Exception:
            tb = traceback.format_exc()
            results.append(
                ExitEvaluation(
                    role=role,
                    should_exit=False,
                    confidence=0.0,
                    reason="evaluation_error",
                    detail=tb[:500],
                    net_pnl_pct=net_pnl_pct,
                    hold_minutes=hold_minutes,
                )
            )

    return results
