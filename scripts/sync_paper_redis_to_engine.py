#!/usr/bin/env python3
"""Align Paper/Redis positions and balances with portfolio_engine SQLite truth."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("/home/mystic/mystic/mystic_trading.db")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def main() -> int:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        ledger = conn.execute("SELECT cash_balance, realized_pnl FROM portfolio_engine_ledger WHERE id = 1").fetchone()
        engine_rows = conn.execute(
            """
            SELECT symbol, quantity, entry_price, trade_id, entry_time,
                   COALESCE(repair_add_count, 0), COALESCE(last_repair_add_ts, 0),
                   COALESCE(entry_strategy_id, 'day'), COALESCE(sleeve, 'ACTIVE')
            FROM portfolio_engine_positions
            WHERE quantity > 0
            ORDER BY symbol
            """
        ).fetchall()
    finally:
        conn.close()

    if not ledger:
        print("ERROR: portfolio_engine_ledger row missing", file=sys.stderr)
        return 1

    from backend.config.redis_config import SharedRedisState

    redis = SharedRedisState.get_async_client()
    if redis is None:
        print("ERROR: Redis unavailable", file=sys.stderr)
        return 1

    now_iso = datetime.now(timezone.utc).isoformat()
    engine_by_symbol = {str(r["symbol"]): dict(r) for r in engine_rows}
    expected_symbols = set(engine_by_symbol.keys())

    live_marks: dict[str, float] = {}
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:8000/api/portfolio-engine/status", timeout=10) as resp:
            import json

            body = json.loads(resp.read())
        for pos in body.get("data", {}).get("open_positions") or []:
            sym = str(pos.get("symbol") or "")
            if sym and pos.get("current_price"):
                live_marks[sym] = float(pos["current_price"])
    except Exception:
        pass

    active_raw = await redis.smembers("paper:positions:active")
    active_before = {(s.decode() if isinstance(s, (bytes, bytearray)) else str(s)) for s in (active_raw or [])}
    print("ACTIVE_BEFORE", sorted(active_before))

    removed: list[str] = []
    for sym in sorted(active_before):
        if sym not in expected_symbols:
            await redis.delete(f"paper:position:{sym}")
            await redis.srem("paper:positions:active", sym)
            removed.append(sym)
            print("REMOVED_STALE", sym)

    synced: list[str] = []
    for sym, row in sorted(engine_by_symbol.items()):
        qty = float(row["quantity"])
        entry = float(row["entry_price"])
        mark = float(live_marks.get(sym) or entry)
        unreal = (mark - entry) * qty
        position_key = f"paper:position:{sym}"
        existing = await redis.hgetall(position_key)
        decoded = {(k.decode() if isinstance(k, (bytes, bytearray)) else str(k)): (v.decode() if isinstance(v, (bytes, bytearray)) else str(v)) for k, v in (existing or {}).items()}
        redis_qty = float(decoded.get("quantity") or 0)
        redis_avg = float(decoded.get("average_price") or 0)
        redis_mark = float(decoded.get("current_price") or 0)
        redis_repair = int(float(decoded.get("repair_add_count") or -1))
        needs_write = (
            sym not in active_before
            or abs(redis_qty - qty) > 1e-9
            or abs(redis_avg - entry) > 1e-6
            or abs(redis_mark - mark) > 1e-6
            or redis_repair != int(row.get("repair_add_count") or 0)
            or not decoded
        )
        if needs_write:
            created_at = decoded.get("created_at") or now_iso
            try:
                created_at = datetime.fromtimestamp(float(row["entry_time"]), tz=timezone.utc).isoformat()
            except (TypeError, ValueError, OSError):
                pass
            payload = {
                "symbol": sym,
                "quantity": str(qty),
                "average_price": str(entry),
                "current_price": str(mark),
                "unrealized_pnl": str(unreal),
                "realized_pnl": "0.0",
                "entry_commission": "0.0",
                "exit_commission": "0.0",
                "created_at": created_at,
                "last_updated": now_iso,
                "sleeve": str(row.get("sleeve") or "ACTIVE"),
                "repair_add_count": str(int(row.get("repair_add_count") or 0)),
                "last_repair_add_ts": str(float(row.get("last_repair_add_ts") or 0)),
                "entry_strategy_id": str(row.get("entry_strategy_id") or "day"),
            }
            for field, value in payload.items():
                await redis.hset(position_key, key=field, value=value)
            await redis.expire(position_key, 86400)
            await redis.sadd("paper:positions:active", sym)
            synced.append(sym)
            print("SYNCED", sym, "qty", qty, "entry", entry, "trade_id", row["trade_id"])

    cash = str(float(ledger["cash_balance"]))
    realized = str(float(ledger["realized_pnl"]))
    await redis.set("paper_trading:cash_balance", cash)
    await redis.set("paper:cash_balance", cash)
    await redis.set("paper_trading:realized_pnl_total", realized)

    active_after_raw = await redis.smembers("paper:positions:active")
    active_after = sorted((s.decode() if isinstance(s, (bytes, bytearray)) else str(s)) for s in (active_after_raw or []))
    print("ACTIVE_AFTER", active_after)
    print("CASH", float(ledger["cash_balance"]))
    print("REALIZED", float(ledger["realized_pnl"]))
    print("REMOVED", removed)
    print("SYNCED", synced)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
