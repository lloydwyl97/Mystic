"""Fill-proven base quantity a strategy lot may hold or sell.

A lot's quantity is what its own venue fills left behind:

    BUY executed qty - base-asset BUY fee - SELL executed qty

The exchange asset balance is shared by DAY_V2, SCALP_V2, protected external
inventory and dust. It is only a physical ceiling and never a source of lot
quantity. Reconciliation may shrink a lot to the exchange balance; it may never
grow one past its fills.
"""

from __future__ import annotations

import logging
import sqlite3
from decimal import Decimal

logger = logging.getLogger(__name__)

STRATEGY_ENGINES = frozenset({"DAY_V2", "SCALP_V2"})
GENERIC_MANUAL_REASONS = frozenset({"MANUAL", "MANUAL_EXIT"})
UNMATCHED_OVERSELL_REASON = "protected_inventory_oversell:dust_reconcile_quantity_inflation"


def _base_asset(symbol: str) -> str:
    raw = str(symbol or "").upper().replace("-", "/")
    if "/" in raw:
        return raw.split("/", maxsplit=1)[0]
    return raw[:-4] if raw.endswith("USDT") else raw


def fill_owned_quantity(conn: sqlite3.Connection, trade_id: str, symbol: str) -> Decimal | None:
    """Net base quantity this lot's own fills hold. None when no BUY fill is on record."""
    tid = str(trade_id or "").strip()
    if not tid:
        return None
    try:
        rows = conn.execute(
            """
            SELECT UPPER(side), executed_qty, fee_amount, UPPER(COALESCE(fee_asset, ''))
            FROM live_exchange_fills
            WHERE mystic_trade_id = ?
            """,
            (tid,),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    base = _base_asset(symbol)
    bought = Decimal("0")
    sold = Decimal("0")
    has_buy = False
    for side, qty, fee, fee_asset in rows:
        amount = Decimal(str(qty or 0))
        if side == "BUY":
            has_buy = True
            bought += amount
            if fee_asset == base:
                bought -= Decimal(str(fee or 0))
        elif side == "SELL":
            sold += amount
    if not has_buy:
        return None
    owned = bought - sold
    return owned if owned > 0 else Decimal("0")


def fill_owned_quantity_at(db_path: str, trade_id: str, symbol: str) -> Decimal | None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    except sqlite3.OperationalError:
        return None
    try:
        return fill_owned_quantity(conn, trade_id, symbol)
    finally:
        conn.close()


def capped_lot_quantity(*, proposed: float, booked: float, owned: Decimal | float | None) -> float:
    """The quantity a lot may be set to. Never above its fills, never grown from an unproven source."""
    ceiling = float(owned) if owned is not None else float(booked or 0.0)
    return max(0.0, min(float(proposed or 0.0), ceiling))


def is_generic_manual_strategy_exit(engine_id: str, recorded_reason: str) -> bool:
    """A DAY_V2 / SCALP_V2 close that carries no strategy reason."""
    return str(engine_id or "") in STRATEGY_ENGINES and str(recorded_reason or "").strip().upper() in GENERIC_MANUAL_REASONS
