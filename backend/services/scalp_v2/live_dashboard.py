"""Read-only live SCALP_V2 views for the /api/scalp dashboard endpoints.

Live SCALP_V2 runs inside the portfolio engine integration and books into
paper_trades / portfolio_engine_positions under engine_id='SCALP_V2'. The
retired paper runner's scalp_paper_* tables and Redis snapshot are no longer
written, so nothing here reads them.
"""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from backend.services.binance_scalp.scalp_attribution_report import _fee_burden_bucket, _hold_bucket, _rollup
from backend.services.scalp_v2.loss_breaker import _CLOSED_SELLS

ENGINE_ID = "SCALP_V2"


def _connect_ro(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _epoch(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _realized(conn: sqlite3.Connection, extra_sql: str = "", params: tuple = ()) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM(pnl), 0), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) {_CLOSED_SELLS} {extra_sql}",
        params,
    ).fetchone()
    return {"sells": int(row[0] or 0), "realized_pnl_usd": round(float(row[1] or 0.0), 6), "wins": int(row[2] or 0), "losses": int(row[3] or 0)}


def _setup_by_opportunity(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        return {str(r[0]): str(r[1] or "") for r in conn.execute("SELECT opportunity_id, setup_family FROM scalp_v2_opportunities")}
    except sqlite3.OperationalError:
        return {}


def open_lots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Slot-consuming SCALP_V2 lots (dust and protected inventory excluded)."""
    from backend.services.protected_external_inventory import consumes_strategy_slot

    rows = conn.execute(
        "SELECT symbol, engine_id, status, quantity, entry_price, entry_time, scalp_opportunity_id FROM portfolio_engine_positions WHERE engine_id = ? AND quantity > 0",
        (ENGINE_ID,),
    ).fetchall()
    return [dict(r) for r in rows if consumes_strategy_slot(SimpleNamespace(**dict(r)))]


def latest_cycle(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        cycle_ts = conn.execute("SELECT MAX(cycle_ts) FROM scalp_v2_decisions").fetchone()[0]
    except sqlite3.OperationalError:
        cycle_ts = None
    if cycle_ts is None:
        return {"cycle_ts": None, "decisions": []}
    rows = conn.execute("SELECT symbol, result, reason FROM scalp_v2_decisions WHERE cycle_ts = ? ORDER BY id", (cycle_ts,)).fetchall()
    return {"cycle_ts": float(cycle_ts), "decisions": [dict(r) for r in rows]}


def _cycle_verdict(decisions: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    results = [str(d.get("result") or "") for d in decisions]
    if "FILLED" in results:
        overall = "FILLED"
    elif "FAILED" in results:
        overall = "FAILED"
    else:
        overall = "NO_ENTRY" if results else None
    reasons = Counter(str(d.get("reason") or "") for d in decisions if str(d.get("result") or "").startswith("REJECTED:"))
    return overall, (reasons.most_common(1)[0][0] if reasons else None)


def live_status(db_path: str, redis_client: Any) -> dict[str, Any]:
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.structural_mode import live_entry_enabled
    from backend.services.task_health_monitor import CRITICAL_TASK_THRESHOLDS_SEC, SCALP_V2_LIVE_LOOP_TASK, heartbeat_age, read_heartbeat_sync
    from backend.services.two_engine_claim import SCALP_MAX_OPEN_POSITIONS

    beat_hash = read_heartbeat_sync(SCALP_V2_LIVE_LOOP_TASK, redis_client) or {}
    age = heartbeat_age(beat_hash)
    threshold = CRITICAL_TASK_THRESHOLDS_SEC[SCALP_V2_LIVE_LOOP_TASK]
    active = age is not None and age <= threshold
    halt_reason = str(beat_hash.get("halt_reason") or "")
    try:
        mode = get_scalp_config().resolved_structural_mode()
    except Exception as exc:
        mode = f"REFUSED:{type(exc).__name__}"
    live_armed = live_entry_enabled(mode)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with closing(_connect_ro(db_path)) as conn:
        pnl_today = _realized(conn, "AND date(timestamp) = ?", (today,))
        pnl_all = _realized(conn)
        lots = open_lots(conn)
        cycle = latest_cycle(conn)

    if not active:
        op_mode, blocked = "runner_dead", ("SCALP_V2_LOOP_HEARTBEAT_STALE" if age is not None else "SCALP_V2_LOOP_NEVER_BEAT")
    elif not live_armed:
        op_mode, blocked = "entry_disarmed", f"STRUCTURAL_MODE_{mode}"
    elif halt_reason:
        op_mode, blocked = "breaker_halt", halt_reason
    elif len(lots) >= SCALP_MAX_OPEN_POSITIONS:
        op_mode, blocked = "max_open_positions_reached", "SCALP_MAX_OPEN_POSITIONS"
    else:
        op_mode, blocked = "entry_scan_active", None
    overall, top_blocker = _cycle_verdict(cycle["decisions"])
    out: dict[str, Any] = {
        "runner_active": active,
        "engine": "scalp",
        "engine_id": ENGINE_ID,
        "source": "scalp_v2_live",
        "loop_heartbeat_age_sec": round(age, 1) if age is not None else None,
        "loop_heartbeat_threshold_sec": threshold,
        "structural_mode": mode,
        "scalp_live": live_armed,
        "entry_armed": bool(live_armed and not halt_reason),
        "open_scalp_positions": len(lots),
        "max_open_positions": SCALP_MAX_OPEN_POSITIONS,
        "pnl_summary": {"engine": "scalp", "today": pnl_today, "all_time": pnl_all, "open_positions": len(lots)},
        "operational_summary": {"operational_mode": op_mode, "entry_blocked_reason": blocked},
        "overall_decision": overall,
        "top_blocker": top_blocker,
        "latest_cycle": {
            "cycle_ts": cycle["cycle_ts"],
            "age_sec": round(time.time() - cycle["cycle_ts"], 1) if cycle["cycle_ts"] is not None else None,
            "decisions": cycle["decisions"],
        },
        "structural_breaker": {
            "status": "OPEN" if halt_reason else "CLOSED",
            "reason": halt_reason or None,
            "recovery_until": str(beat_hash.get("halt_until") or "") or None,
        },
    }
    if not active:
        out["note"] = "SCALP_V2 entry loop heartbeat is stale or missing (runs inside start_portfolio_engine_integration.py)."
    return out


def live_positions(db_path: str) -> dict[str, Any]:
    now = time.time()
    with closing(_connect_ro(db_path)) as conn:
        lots = open_lots(conn)
        setups = _setup_by_opportunity(conn)
    positions = []
    for lot in lots:
        entry_epoch = _epoch(lot.get("entry_time"))
        positions.append(
            {
                "symbol": lot["symbol"],
                "setup": setups.get(str(lot.get("scalp_opportunity_id") or "")) or None,
                "quantity": float(lot["quantity"] or 0.0),
                "entry_price": float(lot["entry_price"] or 0.0),
                "entry_time_epoch": entry_epoch,
                "hold_seconds": round(max(0.0, now - entry_epoch), 1) if entry_epoch else None,
                "status": lot["status"],
                "opportunity_id": lot.get("scalp_opportunity_id"),
            }
        )
    return {"engine": "scalp", "engine_id": ENGINE_ID, "open_count": len(positions), "positions": positions, "ledger": None}


def live_trades(db_path: str, *, limit: int, days: int | None = None) -> dict[str, Any]:
    sql = (
        "SELECT trade_id, symbol, side, quantity, price, fees_paid, pnl, pnl_pct, exit_reason, created_at, timestamp, counts_toward_realized "
        "FROM paper_trades WHERE engine_id = ? AND COALESCE(is_synthetic, 0) = 0"
    )
    params: list[Any] = [ENGINE_ID]
    if days is not None:
        sql += " AND julianday(timestamp) >= julianday('now', ?)"
        params.append(f"-{int(days)} days")
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with closing(_connect_ro(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    trades = [
        {
            "trade_id": r["trade_id"],
            "symbol": r["symbol"],
            "side": r["side"],
            "quantity": r["quantity"],
            "price": r["price"],
            "notional": round(float(r["quantity"] or 0.0) * float(r["price"] or 0.0), 6),
            "fee_usd": r["fees_paid"],
            "pnl_usd": r["pnl"],
            "pnl_pct": r["pnl_pct"],
            "exit_reason": r["exit_reason"],
            "created_at": r["created_at"],
            "timestamp": r["timestamp"],
            "counts_toward_realized": r["counts_toward_realized"],
        }
        for r in rows
    ]
    return {"engine": "scalp", "engine_id": ENGINE_ID, "count": len(trades), "trades": trades}


def live_scoreboard(db_path: str, *, days: int) -> dict[str, Any]:
    with closing(_connect_ro(db_path)) as conn:
        rows = conn.execute(
            "SELECT date(timestamp) AS day, COUNT(*) AS trades, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            f"SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses, ROUND(SUM(pnl), 6) AS net_pnl {_CLOSED_SELLS} "
            "GROUP BY date(timestamp) ORDER BY day DESC LIMIT ?",
            (int(days),),
        ).fetchall()
    return {"engine": "scalp", "engine_id": ENGINE_ID, "days": days, "rows": [dict(r) for r in rows]}


def live_closed_sell_count(db_path: str) -> int:
    with closing(_connect_ro(db_path)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) {_CLOSED_SELLS}").fetchone()[0])


def live_attribution(db_path: str, *, days: int | None = None) -> dict[str, Any]:
    window_sql, params = ("AND julianday(timestamp) >= julianday('now', ?)", (f"-{int(days)} days",)) if days is not None else ("", ())
    with closing(_connect_ro(db_path)) as conn:
        sells = conn.execute(
            f"SELECT trade_id, symbol, pnl, exit_reason, regime, hold_time_seconds, fees_paid, scalp_opportunity_id {_CLOSED_SELLS} {window_sql} ORDER BY id DESC",
            params,
        ).fetchall()
        buys: dict[str, list[float]] = {}
        for r in conn.execute("SELECT scalp_opportunity_id, fees_paid, quantity, price FROM paper_trades WHERE engine_id = ? AND upper(side) = 'BUY'", (ENGINE_ID,)):
            acc = buys.setdefault(str(r[0] or ""), [0.0, 0.0])
            acc[0] += float(r[1] or 0.0)
            acc[1] += float(r[2] or 0.0) * float(r[3] or 0.0)
        setups = _setup_by_opportunity(conn)
    rows = []
    for s in sells:
        opp = str(s["scalp_opportunity_id"] or "")
        buy_fee, buy_notional = buys.get(opp, (0.0, 0.0)) if opp else (0.0, 0.0)
        cost_burden = (buy_fee + float(s["fees_paid"] or 0.0)) / buy_notional if buy_notional > 0 else None
        rows.append(
            {
                "trade_id": s["trade_id"],
                "symbol": s["symbol"],
                "pnl_usd": float(s["pnl"] or 0.0),
                "exit_reason": s["exit_reason"] or "unknown",
                "setup": setups.get(opp) or "unknown",
                "regime": s["regime"] or "unknown",
                "hold_bucket": _hold_bucket(s["hold_time_seconds"]),
                "fee_burden_bucket": _fee_burden_bucket(cost_burden),
            }
        )
    return {
        "engine": "scalp",
        "engine_id": ENGINE_ID,
        "days": days,
        "closed_sells": len(rows),
        "total_net_pnl_usd": round(sum(r["pnl_usd"] for r in rows), 4),
        "by_symbol": _rollup(rows, "symbol"),
        "by_setup": _rollup(rows, "setup"),
        "by_regime": _rollup(rows, "regime"),
        "by_exit_reason": _rollup(rows, "exit_reason"),
        "by_hold_bucket": _rollup(rows, "hold_bucket"),
        "by_fee_burden": _rollup(rows, "fee_burden_bucket"),
    }


__all__ = [
    "live_attribution",
    "live_closed_sell_count",
    "live_positions",
    "live_scoreboard",
    "live_status",
    "live_trades",
]
