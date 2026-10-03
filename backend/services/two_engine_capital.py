"""Two-engine capital allocator for Mystic (DAY_V2 + SCALP_V2).

Both engines share one physical Binance.US USDT balance. This module provides the
single canonical allocator that splits *current* strategy equity into per-engine
deployable budgets:

- DAY target   = DAY_CAPITAL_SHARE   x current strategy equity
- SCALP target = SCALP_CAPITAL_SHARE x current strategy equity

Committed capital per engine = sum(engine-owned position cost) + active engine
entry reservations. No cross-engine borrowing in this implementation: each engine
may only deploy within its own remaining budget.

Existing (grandfathered) positions that exceed the new target are never force-sold;
new entries simply obey the allocator until capital flows back to free USDT.

Every BUY must still satisfy the physical-cash gate: actual free USDT minus active
reservations (both engines) minus fee reserve must cover the executable order cost.
When virtual budget exists but physical cash does not, the gate reports
PHYSICAL_CASH_TEMPORARILY_UNAVAILABLE.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field

DAY_V2_ENGINE_ID = "DAY_V2"
SCALP_V2_ENGINE_ID = "SCALP_V2"

KNOWN_ENGINE_IDS = (DAY_V2_ENGINE_ID, SCALP_V2_ENGINE_ID)

PHYSICAL_CASH_UNAVAILABLE = "PHYSICAL_CASH_TEMPORARILY_UNAVAILABLE"
ENGINE_BUDGET_EXCEEDED = "ENGINE_BUDGET_EXCEEDED"
CAPITAL_CONFIG_INVALID = "CAPITAL_CONFIG_INVALID"

DEFAULT_DAY_SHARE = 0.50
DEFAULT_SCALP_SHARE = 0.50


def get_capital_shares() -> tuple[float, float]:
    """Return (day_share, scalp_share) from config. Raises ValueError if invalid."""
    try:
        day_share = float(os.getenv("DAY_CAPITAL_SHARE", str(DEFAULT_DAY_SHARE)))
    except Exception:
        day_share = DEFAULT_DAY_SHARE
    try:
        scalp_share = float(os.getenv("SCALP_CAPITAL_SHARE", str(DEFAULT_SCALP_SHARE)))
    except Exception:
        scalp_share = DEFAULT_SCALP_SHARE
    if day_share < 0 or scalp_share < 0 or (day_share + scalp_share) > 1.0:
        raise ValueError(f"Invalid capital shares: DAY={day_share} SCALP={scalp_share} (sum must be <= 1.0)")
    return day_share, scalp_share


@dataclass
class EngineBudget:
    engine_id: str
    target_capital: float = 0.0
    committed_positions: float = 0.0
    committed_reservations: float = 0.0
    remaining_budget: float = 0.0

    @property
    def committed_total(self) -> float:
        return self.committed_positions + self.committed_reservations


def _default_day_budget() -> EngineBudget:
    return EngineBudget(engine_id=DAY_V2_ENGINE_ID)


def _default_scalp_budget() -> EngineBudget:
    return EngineBudget(engine_id=SCALP_V2_ENGINE_ID)


@dataclass
class CapitalSnapshot:
    equity: float = 0.0
    free_cash: float = 0.0
    day: EngineBudget = field(default_factory=_default_day_budget)
    scalp: EngineBudget = field(default_factory=_default_scalp_budget)
    reservations_total: float = 0.0
    protected_equity: float = 0.0

    @property
    def total_account_equity(self) -> float:
        return self.equity

    @property
    def strategy_owned_equity(self) -> float:
        return max(0.0, self.equity - self.protected_equity)

    @property
    def day_target(self) -> float:
        return self.day.target_capital

    @property
    def scalp_target(self) -> float:
        return self.scalp.target_capital

    def as_dict(self) -> dict[str, float]:
        return {
            "total_account_equity": self.total_account_equity,
            "strategy_owned_equity": self.strategy_owned_equity,
            "protected_equity": self.protected_equity,
            "day_target": self.day_target,
            "scalp_target": self.scalp_target,
            "free_cash": self.free_cash,
            "reservations_total": self.reservations_total,
            "day_remaining": self.day.remaining_budget,
            "scalp_remaining": self.scalp.remaining_budget,
        }

    def for_engine(self, engine_id: str) -> EngineBudget:
        if str(engine_id or "") == SCALP_V2_ENGINE_ID:
            return self.scalp
        return self.day


def _position_committed_cost(pos: object) -> float:
    """Cost basis committed by one open position (original cost preferred)."""
    try:
        orig = float(getattr(pos, "original_position_cost", 0) or 0)
        if orig > 0:
            return orig
        qty = float(getattr(pos, "quantity", 0) or 0)
        entry = float(getattr(pos, "entry_price", 0) or 0)
        return max(0.0, qty * entry)
    except Exception:
        return 0.0


def _active_reservations_by_engine(db_path: str, exclude_reservation_id: str = "") -> dict[str, float]:
    """Sum ACTIVE reservation notionals per engine sleeve. Never raises.

    Queries day_entry_reservations directly: the SCALP_V2 sleeve counts toward
    SCALP; every other sleeve (DAY_V2 / ACTIVE / legacy / empty) is DAY-side
    heritage. Expired rows are excluded even if the staleness sweep has not
    relabelled them yet.
    """
    totals: dict[str, float] = {DAY_V2_ENGINE_ID: 0.0, SCALP_V2_ENGINE_ID: 0.0}
    try:
        if not db_path or str(db_path) == ":memory:":
            return totals
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            rows = conn.execute(
                "SELECT sleeve, notional_usd, expires_at FROM day_entry_reservations WHERE status='ACTIVE' AND reservation_id != ?",
                (str(exclude_reservation_id or ""),),
            ).fetchall()
        finally:
            conn.close()
        now = time.time()
        for sleeve, notional, expires_at in rows:
            try:
                exp = float(expires_at or 0)
            except (TypeError, ValueError):
                exp = 0.0
            if exp and exp <= now:
                continue
            amount = max(0.0, float(notional or 0))
            if str(sleeve or "") == SCALP_V2_ENGINE_ID:
                totals[SCALP_V2_ENGINE_ID] += amount
            else:
                totals[DAY_V2_ENGINE_ID] += amount
    except Exception:
        pass
    return totals


def symbol_marks(position_marks: dict | None) -> dict[str, float]:
    """Engine per-lot marks (keys may be ENGINE::SYM) collapsed to bare symbols."""
    out: dict[str, float] = {}
    for key, mark in (position_marks or {}).items():
        try:
            out[str(key).split("::")[-1]] = float(mark)
        except (TypeError, ValueError):
            continue
    return out


def _protected_market_value(db_path: str, prices: dict | None) -> float:
    if not db_path or str(db_path) == ":memory:":
        return 0.0
    try:
        from backend.services.protected_external_inventory import protected_equity

        return max(0.0, float(protected_equity(db_path, prices)[0]))
    except Exception:
        return 0.0


def compute_snapshot(
    db_path: str,
    equity: float,
    free_cash: float,
    open_positions: dict | None = None,
    exclude_reservation_id: str = "",
    prices: dict | None = None,
) -> CapitalSnapshot:
    """Build the canonical two-engine capital snapshot from CURRENT account state.

    Engine targets split strategy-owned equity: total account equity minus
    protected/unmatched external inventory, which neither engine owns.
    Physical free cash remains the final constraint in check_engine_budget.
    """
    day_share, scalp_share = get_capital_shares()
    equity_f = max(0.0, float(equity or 0))
    snap = CapitalSnapshot(equity=equity_f, free_cash=max(0.0, float(free_cash or 0)))
    snap.protected_equity = min(equity_f, _protected_market_value(db_path, prices))
    owned = snap.strategy_owned_equity
    snap.day.target_capital = owned * day_share
    snap.scalp.target_capital = owned * scalp_share

    for _key, pos in (open_positions or {}).items():
        try:
            from backend.services.protected_external_inventory import consumes_strategy_slot

            if not consumes_strategy_slot(pos):
                continue
        except Exception:
            pass
        engine = str(getattr(pos, "engine_id", "") or "")
        cost = _position_committed_cost(pos)
        if engine == SCALP_V2_ENGINE_ID:
            snap.scalp.committed_positions += cost
        elif engine == DAY_V2_ENGINE_ID:
            snap.day.committed_positions += cost
        # Unknown/legacy engine lots consume physical cash but are not attributed
        # to either engine budget (they predate engine-scoped accounting).

    res = _active_reservations_by_engine(db_path, exclude_reservation_id)
    snap.day.committed_reservations = res[DAY_V2_ENGINE_ID]
    snap.scalp.committed_reservations = res[SCALP_V2_ENGINE_ID]
    snap.reservations_total = res[DAY_V2_ENGINE_ID] + res[SCALP_V2_ENGINE_ID]

    snap.day.remaining_budget = max(0.0, snap.day.target_capital - snap.day.committed_total)
    snap.scalp.remaining_budget = max(0.0, snap.scalp.target_capital - snap.scalp.committed_total)
    return snap


def scalp_order_notional(
    snap: CapitalSnapshot,
    *,
    slots: int,
    size_mult: float = 1.0,
    emergency_max_notional: float = 0.0,
) -> float:
    """SCALP order notional: sleeve / slots x adaptive size, inside the remaining sleeve.

    ``emergency_max_notional`` is an absolute ceiling only when explicitly
    configured (> 0). The engine-budget and physical-cash gates still run at
    order time.
    """
    base = snap.scalp.target_capital / max(1, int(slots))
    notional = max(0.0, base * float(size_mult or 1.0))
    notional = min(notional, max(0.0, snap.scalp.remaining_budget))
    if float(emergency_max_notional or 0.0) > 0:
        notional = min(notional, float(emergency_max_notional))
    return notional


def check_engine_budget(
    db_path: str,
    engine_id: str,
    order_cost: float,
    equity: float,
    free_cash: float,
    open_positions: dict | None = None,
    fee_reserve: float = 0.0,
    exclude_reservation_id: str = "",
    prices: dict | None = None,
) -> tuple[bool, str, CapitalSnapshot | None]:
    """Gate a prospective BUY of `order_cost` for `engine_id`.

    Returns (ok, reason, snapshot). Reasons:
      "" when ok; ENGINE_BUDGET_EXCEEDED when the engine's own budget is
      insufficient; PHYSICAL_CASH_TEMPORARILY_UNAVAILABLE when virtual budget
      exists but real free USDT (net of both engines' reservations and fee
      reserve) cannot cover the order; CAPITAL_CONFIG_INVALID on bad config.
    `exclude_reservation_id` is the caller's own reservation for this order,
    which must not be counted against it a second time.
    """
    try:
        snap = compute_snapshot(db_path, equity, free_cash, open_positions, exclude_reservation_id, prices)
    except ValueError:
        return False, CAPITAL_CONFIG_INVALID, None
    budget = snap.for_engine(engine_id)
    cost = max(0.0, float(order_cost or 0))
    if cost > budget.remaining_budget:
        return False, ENGINE_BUDGET_EXCEEDED, snap
    physical = snap.free_cash - snap.reservations_total - max(0.0, float(fee_reserve or 0))
    if cost > max(0.0, physical):
        return False, PHYSICAL_CASH_UNAVAILABLE, snap
    return True, "", snap
