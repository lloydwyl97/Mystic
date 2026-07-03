"""Forward paper accounting helpers for ALLWEATHER_BREAKOUT_PULLBACK validation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORWARD_EPOCH_KEY = "allweather_forward_paper_epoch"
REBASE_HISTORY_KEY = "paper_ledger_rebase_history"
FORWARD_PRINCIPAL_USD = 25_000.0

SYNTHETIC_WHERE = "COALESCE(is_synthetic, 0) = 1 OR COALESCE(paper_run_id, '') LIKE 'PAPER_LIFECYCLE_SMOKE%'"
NON_SYNTHETIC_WHERE = f"NOT ({SYNTHETIC_WHERE})"

ADMIN_EXIT_SQL = (
    "COALESCE(exit_type, '') NOT IN ("
    "'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR', 'RESEARCH_RESET_EXIT'"
    ")"
)


def forward_sell_filter_sql(*, epoch_start: str | None = None) -> tuple[str, tuple[Any, ...]]:
    """SQL WHERE clause for dashboard/strategy closed trades (non-synthetic, non-admin)."""
    parts = [
        "UPPER(side) = 'SELL'",
        "pnl IS NOT NULL",
        ADMIN_EXIT_SQL,
        NON_SYNTHETIC_WHERE,
    ]
    params: list[Any] = []
    if epoch_start:
        parts.append("timestamp >= ?")
        params.append(epoch_start)
    return " AND ".join(parts), tuple(params)


def reconcile_forward_ledger(
    db_path: str | Path,
    *,
    positions_value: float,
    unrealized_pnl: float,
    force: bool = False,
) -> dict[str, Any]:
    """
    Heal portfolio_engine_ledger cash/equity from forward-epoch strategy PnL.

    INVARIANT: total_equity = forward_principal + forward_realized + unrealized
               cash_balance = total_equity - positions_value
    """
    epoch = get_forward_epoch(db_path)
    if not epoch or not epoch.get("epoch_started_at"):
        return {"skipped": True, "reason": "no_forward_epoch"}

    bd = compute_pnl_breakdown(db_path, epoch=epoch)
    principal = float(bd.get("forward_principal_usd") or FORWARD_PRINCIPAL_USD)
    fwd_realized = float(bd.get("realized_pnl_forward_usd") or 0.0)
    pos_val = float(positions_value or 0.0)
    unreal = float(unrealized_pnl or 0.0)
    total_equity = principal + fwd_realized + unreal
    cash = total_equity - pos_val

    if not force:
        conn_check = sqlite3.connect(str(db_path))
        try:
            row = conn_check.execute(
                "SELECT cash_balance, total_equity, realized_pnl, positions_value, unrealized_pnl FROM portfolio_engine_ledger WHERE id=1"
            ).fetchone()
            if row:
                if (
                    abs(float(row[0] or 0) - cash) < 0.05
                    and abs(float(row[1] or 0) - total_equity) < 0.05
                    and abs(float(row[2] or 0) - fwd_realized) < 0.05
                    and abs(float(row[3] or 0) - pos_val) < 0.05
                    and abs(float(row[4] or 0) - unreal) < 0.05
                ):
                    return {
                        "skipped": True,
                        "reason": "already_aligned",
                        "forward_epoch_started_at": bd.get("forward_epoch_started_at"),
                        "total_equity_usd": round(total_equity, 4),
                    }
        finally:
            conn_check.close()

    ts = _utc_now()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            UPDATE portfolio_engine_ledger SET
                principal = ?,
                cash_balance = ?,
                positions_value = ?,
                realized_pnl = ?,
                unrealized_pnl = ?,
                total_equity = ?,
                last_updated = ?
            WHERE id = 1
            """,
            (principal, cash, pos_val, fwd_realized, unreal, total_equity, ts),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "success": True,
        "forward_epoch_started_at": bd.get("forward_epoch_started_at"),
        "principal_usd": principal,
        "realized_pnl_forward_usd": fwd_realized,
        "cash_usd": round(cash, 4),
        "total_equity_usd": round(total_equity, 4),
        "positions_value_usd": pos_val,
        "unrealized_pnl_usd": unreal,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_forward_epoch(db_path: str | Path) -> dict[str, Any] | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT value_json FROM operational_state WHERE key=?",
            (FORWARD_EPOCH_KEY,),
        ).fetchone()
        if not row or not row[0]:
            return None
        payload = json.loads(row[0])
        return payload if isinstance(payload, dict) else None
    finally:
        conn.close()


def compute_pnl_breakdown(db_path: str | Path, *, epoch: dict[str, Any] | None = None) -> dict[str, Any]:
    epoch = epoch if epoch is not None else get_forward_epoch(db_path)
    epoch_start = str((epoch or {}).get("epoch_started_at") or "")
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT COALESCE(SUM(pnl), 0.0)
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND pnl IS NOT NULL
              AND {NON_SYNTHETIC_WHERE}
              AND COALESCE(exit_type, '') NOT IN (
                'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR', 'RESEARCH_RESET_EXIT'
              )
            """
        )
        all_time_non_synthetic = float((cur.fetchone() or (0.0,))[0] or 0.0)

        cur.execute(
            f"""
            SELECT COALESCE(SUM(pnl), 0.0)
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND pnl IS NOT NULL
              AND ({SYNTHETIC_WHERE})
            """
        )
        synthetic_smoke_pnl = float((cur.fetchone() or (0.0,))[0] or 0.0)

        forward_non_synthetic = 0.0
        if epoch_start:
            cur.execute(
                f"""
                SELECT COALESCE(SUM(pnl), 0.0)
                FROM paper_trades
                WHERE UPPER(side) = 'SELL' AND pnl IS NOT NULL
                  AND timestamp >= ?
                  AND {NON_SYNTHETIC_WHERE}
                  AND COALESCE(exit_type, '') NOT IN (
                    'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR', 'RESEARCH_RESET_EXIT'
                  )
                """,
                (epoch_start,),
            )
            forward_non_synthetic = float((cur.fetchone() or (0.0,))[0] or 0.0)

        pre_rebase = float((epoch or {}).get("prior_ledger_realized_pnl") or 0.0)
        if epoch_start and pre_rebase == 0.0:
            cur.execute(
                f"""
                SELECT COALESCE(SUM(pnl), 0.0)
                FROM paper_trades
                WHERE UPPER(side) = 'SELL' AND pnl IS NOT NULL
                  AND timestamp < ?
                  AND {NON_SYNTHETIC_WHERE}
                """,
                (epoch_start,),
            )
            pre_rebase = float((cur.fetchone() or (0.0,))[0] or 0.0)

        row = cur.execute("SELECT principal, cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        ledger = {}
        if row:
            ledger = {
                "principal_usd": float(row[0] or 0.0),
                "cash_usd": float(row[1] or 0.0),
                "positions_value_usd": float(row[2] or 0.0),
                "realized_pnl_ledger_stored": float(row[3] or 0.0),
                "unrealized_pnl_usd": float(row[4] or 0.0),
                "total_equity_usd": float(row[5] or 0.0),
            }

        forward_equity = FORWARD_PRINCIPAL_USD + forward_non_synthetic + float(ledger.get("unrealized_pnl_usd") or 0.0)
        return {
            "forward_epoch_started_at": epoch_start or None,
            "forward_principal_usd": FORWARD_PRINCIPAL_USD,
            "realized_pnl_forward_usd": round(forward_non_synthetic, 4),
            "synthetic_smoke_pnl_usd": round(synthetic_smoke_pnl, 4),
            "pre_rebase_history_pnl_usd": round(pre_rebase, 4),
            "all_time_non_synthetic_realized_pnl_usd": round(all_time_non_synthetic, 4),
            "forward_equity_usd": round(forward_equity, 4),
            "forward_cash_usd": round(FORWARD_PRINCIPAL_USD + forward_non_synthetic, 4),
            "synthetic_smoke_excluded_from_forward": True,
            "not_strategy_performance_tags": {
                "is_synthetic": True,
                "not_strategy_performance": True,
                "not_forward_pnl": True,
                "not_live_trade": True,
                "not_real_money": True,
            },
            "ledger": ledger,
        }
    finally:
        conn.close()


def reset_forward_paper_baseline(
    db_path: str | Path,
    *,
    principal_usd: float = FORWARD_PRINCIPAL_USD,
    reason: str = "Forward paper validation baseline — exclude synthetic smoke from equity",
) -> dict[str, Any]:
    db_path = Path(db_path)
    ts = _utc_now()
    conn = sqlite3.connect(str(db_path))
    try:
        prior = conn.execute("SELECT principal, cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        prior_ledger = dict(
            zip(
                ["principal", "cash_balance", "positions_value", "realized_pnl", "unrealized_pnl", "total_equity"],
                prior or (0,) * 6,
                strict=False,
            )
        )
        open_before = int(conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] or 0)

        conn.execute("DELETE FROM portfolio_engine_positions")
        conn.execute(
            """
            UPDATE portfolio_engine_ledger SET
                principal = ?,
                cash_balance = ?,
                positions_value = 0.0,
                realized_pnl = 0.0,
                unrealized_pnl = 0.0,
                total_equity = ?,
                account_status = 'HEALTHY',
                trading_paused = 0,
                pause_reason = NULL,
                last_updated = ?
            WHERE id = 1
            """,
            (principal_usd, principal_usd, principal_usd, ts),
        )

        epoch_record = {
            "event": "ALLWEATHER_FORWARD_PAPER_EPOCH",
            "epoch_started_at": ts,
            "forward_principal_usd": principal_usd,
            "prior_ledger": prior_ledger,
            "prior_ledger_realized_pnl": float(prior_ledger.get("realized_pnl") or 0.0),
            "prior_open_positions": open_before,
            "synthetic_smoke_excluded_from_forward": True,
            "not_strategy_performance": True,
            "not_forward_pnl": True,
            "not_live_trade": True,
            "not_real_money": True,
            "reason": reason,
        }
        conn.execute(
            """
            INSERT INTO operational_state(key, value_json, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts
            """,
            (FORWARD_EPOCH_KEY, json.dumps(epoch_record), ts),
        )

        hist: list[Any] = []
        row = conn.execute(
            "SELECT value_json FROM operational_state WHERE key=?",
            (REBASE_HISTORY_KEY,),
        ).fetchone()
        if row and row[0]:
            try:
                hist = json.loads(row[0])
            except json.JSONDecodeError:
                hist = []
        if not isinstance(hist, list):
            hist = []
        hist.append({**epoch_record, "event": "PAPER_LEDGER_FORWARD_BASELINE"})
        conn.execute(
            """
            INSERT INTO operational_state(key, value_json, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts
            """,
            (REBASE_HISTORY_KEY, json.dumps(hist), ts),
        )
        conn.commit()
        return {
            "success": True,
            "epoch": epoch_record,
            "ledger_after": {
                "principal_usd": principal_usd,
                "cash_usd": principal_usd,
                "equity_usd": principal_usd,
                "realized_pnl_forward_usd": 0.0,
                "unrealized_pnl_usd": 0.0,
                "open_positions": 0,
            },
        }
    finally:
        conn.close()


__all__ = [
    "ADMIN_EXIT_SQL",
    "FORWARD_EPOCH_KEY",
    "FORWARD_PRINCIPAL_USD",
    "NON_SYNTHETIC_WHERE",
    "compute_pnl_breakdown",
    "forward_sell_filter_sql",
    "get_forward_epoch",
    "reconcile_forward_ledger",
    "reset_forward_paper_baseline",
]
