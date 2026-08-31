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


ORPHAN_BUY_HEAL_STATE_KEY = "ledger_orphan_buy_cash_restore"


def apply_orphan_buy_cash_restore(
    db_path: str | Path,
    *,
    buy_ids: list[int],
    heal_key: str,
    reason: str,
) -> dict[str, Any]:
    """Credit cash for identified BUY lots whose inventory was destroyed without a SELL.

    Does not change realized_pnl. Marks each BUY diagnostics_json ORPHAN_CASH_RESTORED
    so detectors cannot restore zombie positions later.
    """
    from backend.services.atomic_execution_book import find_unclosed_buy_cash_debits

    db_path = Path(db_path)
    wanted = {int(i) for i in buy_ids}
    if not wanted:
        raise ValueError("buy_ids required")
    orphans = [o for o in find_unclosed_buy_cash_debits(db_path) if int(o.get("id") or 0) in wanted]
    found_ids = {int(o.get("id") or 0) for o in orphans}
    missing = wanted - found_ids
    if missing:
        raise RuntimeError(f"buy_ids are not unclosed cash-debit orphans: {sorted(missing)}")

    ts = _utc_now()
    conn = sqlite3.connect(str(db_path))
    try:
        pre = _ledger_row(conn)
        if not pre:
            raise RuntimeError("portfolio_engine_ledger row missing")
        open_n = int(conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] or 0)
        if open_n != 0:
            raise RuntimeError(f"refusing orphan cash restore with {open_n} open positions")

        restored_rows: list[dict[str, Any]] = []
        amount = 0.0
        for o in orphans:
            qty = float(o.get("remaining_position") or o.get("quantity") or 0.0)
            price = float(o.get("price") or 0.0)
            notional = qty * price
            if notional <= 0:
                raise RuntimeError(f"invalid orphan notional for buy id={o.get('id')}")
            amount += notional
            restored_rows.append(
                {
                    "id": int(o.get("id") or 0),
                    "trade_id": str(o.get("trade_id") or ""),
                    "symbol": str(o.get("symbol") or ""),
                    "quantity": qty,
                    "price": price,
                    "notional": round(notional, 8),
                    "created_at": o.get("created_at"),
                }
            )

        post_cash = float(pre["cash_balance"]) + amount
        post_equity = float(pre["total_equity"]) + amount
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
        for o in orphans:
            row = conn.execute(
                "SELECT diagnostics_json FROM paper_trades WHERE id = ?",
                (int(o["id"]),),
            ).fetchone()
            raw = row[0] if row else None
            try:
                diag = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                diag = {"_raw": raw}
            if not isinstance(diag, dict):
                diag = {"_raw": diag}
            diag["ORPHAN_CASH_RESTORED"] = True
            diag["orphan_cash_restore_heal_key"] = heal_key
            diag["orphan_cash_restore_at"] = ts
            conn.execute(
                "UPDATE paper_trades SET diagnostics_json = ?, remaining_position = 0 WHERE id = ?",
                (json.dumps(diag, separators=(",", ":")), int(o["id"])),
            )

        heal_record = {
            "heal_key": heal_key,
            "heal_amount_usd": round(amount, 8),
            "reference_paper_buy_ids": [int(o["id"]) for o in orphans],
            "restored_buys": restored_rows,
            "reason": reason,
            "healed_at": ts,
            "pre_ledger": {k: round(v, 8) for k, v in pre.items()},
            "post_ledger": {
                **{k: round(v, 8) for k, v in pre.items()},
                "cash_balance": round(post_cash, 8),
                "total_equity": round(post_equity, 8),
            },
            "not_strategy_performance": True,
            "not_forward_pnl": True,
            "not_fake_pnl": True,
            "not_new_trade": True,
            "excluded_from_strategy_metrics": True,
        }
        conn.execute(
            """
            INSERT INTO operational_state(key, value_json, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_ts = excluded.updated_ts
            """,
            (ORPHAN_BUY_HEAL_STATE_KEY, json.dumps(heal_record), ts),
        )
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
                json.dumps({"reference_paper_buy_ids": [int(o["id"]) for o in orphans]}),
                json.dumps(heal_record["pre_ledger"]),
                json.dumps(heal_record["post_ledger"]),
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


__all__ = [
    "HEAL_STATE_KEY",
    "ORPHAN_BUY_HEAL_STATE_KEY",
    "apply_orphan_buy_cash_restore",
    "apply_sell_cash_credit_heal",
]
