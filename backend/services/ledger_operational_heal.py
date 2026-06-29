"""Operational ledger heals — not strategy performance, not synthetic PnL."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HEAL_STATE_KEY = "ledger_operational_heal_history"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ledger_row(conn: sqlite3.Connection) -> dict[str, float]:
    row = conn.execute(
        """
        SELECT principal, cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity
        FROM portfolio_engine_ledger WHERE id = 1
        """
    ).fetchone()
    if not row:
        return {}
    keys = (
        "principal",
        "cash_balance",
        "positions_value",
        "realized_pnl",
        "unrealized_pnl",
        "total_equity",
    )
    return {k: float(v or 0.0) for k, v in zip(keys, row, strict=False)}


def apply_sell_cash_credit_heal(
    db_path: str | Path,
    *,
    amount_usd: float,
    reference_sell_id: int,
    heal_key: str = "LEDGER_HEAL_SELL_CASH_CREDIT_3059",
    reason: str = "Recover missing sell proceeds from paper sell #3059 (cash not credited on NET_PROFIT exit)",
) -> dict[str, Any]:
    """
    One-time operational cash/equity heal — restores missing sell proceeds only.

    Does not mutate paper_trades, does not create fake PnL rows, excluded from strategy metrics.
    """
    db_path = Path(db_path)
    amount = float(amount_usd)
    if amount <= 0.0:
        raise ValueError("heal amount must be positive")

    ts = _utc_now()
    conn = sqlite3.connect(str(db_path))
    try:
        pre = _ledger_row(conn)
        if not pre:
            raise RuntimeError("portfolio_engine_ledger row missing")

        open_n = int(conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] or 0)
        if open_n != 0:
            raise RuntimeError(f"refusing heal with {open_n} open positions")

        post_cash = float(pre["cash_balance"]) + amount
        post_equity = float(pre["total_equity"]) + amount
        post = {
            **pre,
            "cash_balance": post_cash,
            "total_equity": post_equity,
        }

        conn.execute(
            """
            UPDATE portfolio_engine_ledger SET
                cash_balance = ?,
                total_equity = ?,
                last_updated = ?
            WHERE id = 1
            """,
            (post_cash, post_equity, ts),
        )

        heal_record = {
            "heal_key": heal_key,
            "heal_amount_usd": round(amount, 4),
            "reference_paper_sell_id": int(reference_sell_id),
            "reason": reason,
            "healed_at": ts,
            "pre_ledger": {k: round(v, 4) for k, v in pre.items()},
            "post_ledger": {k: round(v, 4) for k, v in post.items()},
            "not_strategy_performance": True,
            "not_forward_pnl": True,
            "not_fake_pnl": True,
            "not_new_trade": True,
            "excluded_from_strategy_metrics": True,
        }

        hist: list[Any] = []
        row = conn.execute(
            "SELECT value_json FROM operational_state WHERE key = ?",
            (HEAL_STATE_KEY,),
        ).fetchone()
        if row and row[0]:
            try:
                hist = json.loads(row[0])
            except json.JSONDecodeError:
                hist = []
        if not isinstance(hist, list):
            hist = []
        hist.append(heal_record)
        conn.execute(
            """
            INSERT INTO operational_state(key, value_json, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_ts = excluded.updated_ts
            """,
            (HEAL_STATE_KEY, json.dumps(hist), ts),
        )

        pre_ledger_json = json.dumps({k: round(v, 4) for k, v in pre.items()})
        post_ledger_json = json.dumps({k: round(v, 4) for k, v in post.items()})
        conn.execute(
            """
            INSERT INTO portfolio_engine_audit (
                ts, action, symbol, qty, price, fees, slippage,
                decision_id, trade_id, ranked_candidates_json,
                pre_ledger_json, post_ledger_json,
                pre_positions_digest, post_positions_digest,
                invariant_ok, invariant_diff, entry_reason, exit_reason, sleeve
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                "LEDGER_HEAL",
                "SYSTEM",
                0.0,
                0.0,
                0.0,
                0.0,
                None,
                heal_key,
                json.dumps({"reference_paper_sell_id": int(reference_sell_id)}),
                pre_ledger_json,
                post_ledger_json,
                "{}",
                "{}",
                1,
                amount,
                heal_key,
                reason,
                "SYSTEM",
            ),
        )
        conn.commit()
        return {"success": True, "heal": heal_record}
    finally:
        conn.close()


__all__ = ["HEAL_STATE_KEY", "apply_sell_cash_credit_heal"]
