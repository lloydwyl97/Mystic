"""Live contributed-capital and trailing-buy P&L basis.

Restart/bootstrap may adopt exchange cash and positions. It must not reset
contributed principal to current equity. Balance reconciliation is not
trading profit. Trailing-buy scorecard stays anchored at the 9039923 cash
repair until that executor produces real fills.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from decimal import Decimal
from typing import Any

from backend.services.day_entry_spendable import money

logger = logging.getLogger(__name__)

TRAILING_BUY_ANCHOR_EQUITY = Decimal("228.06746265")
TRAILING_BUY_ANCHOR_SHA = "9039923fb45e5bb5382e582c1fa205bbb8361f3d"
SCORECARD_KEY = "day_trailing_buy_scorecard"
RECON_KEY = "live_cash_reconciliation"


def apply_external_capital_flow(principal: object, amount: object) -> Decimal:
    """Deposits/withdrawals change contribution basis, not trading P&L."""
    return money(principal) + money(amount)


def preserve_principal(*, stored_principal: object, equity: object) -> Decimal:
    """Keep an existing basis. Initialize only when none exists."""
    prior = money(stored_principal)
    if prior > 0:
        return prior
    return money(equity)


def cash_reconciliation_delta(*, previous_cash: object, exchange_cash: object) -> Decimal:
    return money(exchange_cash) - money(previous_cash)


def apply_bootstrap_cash(
    *,
    stored_principal: object,
    previous_cash: object,
    exchange_cash: object,
    positions_value: object = 0,
) -> dict[str, Decimal]:
    """Adopt exchange cash; do not set principal = equity."""
    cash = money(exchange_cash)
    positions = money(positions_value)
    equity = cash + positions
    principal = preserve_principal(stored_principal=stored_principal, equity=equity)
    delta = cash_reconciliation_delta(previous_cash=previous_cash, exchange_cash=cash)
    return {
        "principal": principal,
        "cash": cash,
        "positions_value": positions,
        "equity": equity,
        "reconciliation_adjustment": delta,
    }


def trailing_buy_scorecard(
    *,
    live_fills_since_anchor: object = 0,
    current_equity: object | None = None,
    live_realized_pnl: object | None = None,
    live_unrealized_pnl: object = 0,
) -> dict[str, Any]:
    """Fill-based live P&L since the 9039923 cash-repair equity.

    Realized uses live fill PnL. Marked total is current equity minus the
    anchor. Paper history is never included.
    """
    realized = money(live_realized_pnl) if live_realized_pnl is not None else money(live_fills_since_anchor)
    unrealized = money(live_unrealized_pnl)
    if current_equity is not None:
        total = money(current_equity) - TRAILING_BUY_ANCHOR_EQUITY
    else:
        total = realized + unrealized
    return {
        "anchor_equity": str(TRAILING_BUY_ANCHOR_EQUITY),
        "anchor_sha": TRAILING_BUY_ANCHOR_SHA,
        "realized_pnl": float(realized),
        "unrealized_pnl": float(unrealized),
        "total_pnl": float(total),
        "current_equity": str(money(current_equity)) if current_equity is not None else None,
    }


def persist_operational_json(db_path: str, key: str, payload: dict[str, Any]) -> None:
    if not db_path:
        return
    raw = json.dumps(payload, separators=(",", ":"))
    with sqlite3.connect(db_path, timeout=5) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operational_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_ts TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO operational_state(key, value_json, updated_ts)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts
            """,
            (key, raw),
        )
        conn.commit()


def load_operational_json(db_path: str, key: str) -> dict[str, Any]:
    if not db_path:
        return {}
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            row = conn.execute("SELECT value_json FROM operational_state WHERE key=?", (key,)).fetchone()
        if not row or not row[0]:
            return {}
        data = json.loads(row[0])
        return data if isinstance(data, dict) else {}
    except (sqlite3.Error, TypeError, ValueError):
        return {}
