#!/usr/bin/env python3
"""Paper-only stale pre-correction position clear — honest marks, admin labels, flat positions."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

DB = PROJECT_ROOT / "mystic_trading.db"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BACKUP = PROJECT_ROOT / f"mystic_trading.db.backup_before_stale_corrected_path_position_clear_{TS}"
REPORT: dict = {"timestamp_utc": TS, "backup_path": str(BACKUP)}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
POST_SELL_COOLDOWN_WALL_SEC = int(os.getenv("POST_SELL_COOLDOWN_WALL_SEC", "2400"))
CLOSE_REASON = "STALE_PRE_CORRECTION_POSITION_CLEAR"
AUDIT_ACTION = "STALE_PRE_CORRECTION_POSITION_CLEAR"
LESSON = (
    "position opened before corrected AI/model/training/feature path; "
    "admin-cleared so Mystic can start a fresh corrected-path DAY cycle"
)


def _mystic_uvicorn_running() -> bool:
    import subprocess

    try:
        r = subprocess.run(
            ["pgrep", "-f", r"uvicorn backend\.(main|app_factory)"],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0
    except Exception:
        return False


def preflight_checks() -> dict:
    """Refuse unsafe or redundant admin clears."""
    mode = os.getenv("EXECUTION_MODE", "").lower()
    live = os.getenv("LIVE_TRADES_ALLOWED", "").lower()
    if mode != "paper":
        raise SystemExit(f"ABORT: execution_mode={mode!r} — admin clear is paper-only")
    if live in ("true", "1", "yes", "on"):
        raise SystemExit("ABORT: live_trades_allowed=true — stop before admin clear")

    live_orders_permitted = None
    try:
        from backend.config.live_test_mode import get_live_execution_snapshot

        snap = get_live_execution_snapshot()
        live_orders_permitted = bool(getattr(snap, "live_orders_permitted", False))
        if live_orders_permitted:
            raise SystemExit("ABORT: live_orders_permitted=true — stop before admin clear")
    except SystemExit:
        raise
    except Exception as ex:
        REPORT["live_orders_check_warning"] = str(ex)

    if _mystic_uvicorn_running():
        raise SystemExit("ABORT: Mystic uvicorn is running — run ./stop_mystic.sh first")

    conn = sqlite3.connect(str(DB))
    try:
        cur = conn.cursor()
        open_syms = [r[0] for r in cur.execute("SELECT symbol FROM portfolio_engine_positions ORDER BY symbol")]
        if not open_syms:
            stale_cnt = cur.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE exit_type=?",
                (CLOSE_REASON,),
            ).fetchone()[0]
            if int(stale_cnt or 0) >= len(SYMBOLS):
                raise SystemExit(
                    f"ABORT: already flat with prior {CLOSE_REASON} rows — idempotent guard (nothing to clear)"
                )
            raise SystemExit("ABORT: no open positions to clear")
        if set(open_syms) != set(SYMBOLS):
            raise SystemExit(f"ABORT: open positions {open_syms} != expected {SYMBOLS}")
        ledger = cur.execute(
            "SELECT cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1"
        ).fetchone()
        REPORT["ledger_before"] = {
            "cash_balance": float(ledger[0]),
            "positions_value": float(ledger[1]),
            "realized_pnl": float(ledger[2]),
            "unrealized_pnl": float(ledger[3]),
            "total_equity": float(ledger[4]),
        }
    finally:
        conn.close()

    return {
        "execution_mode": mode,
        "live_trades_allowed": live,
        "live_orders_permitted": live_orders_permitted,
        "open_symbols": open_syms,
    }


def fetch_reference_marks(conn: sqlite3.Connection | None = None) -> dict[str, float]:
    marks: dict[str, float] = {}
    mark_sources: dict[str, str] = {}

    for sym in SYMBOLS:
        pair = sym.replace("/", "")
        url = f"https://api.binance.us/api/v3/ticker/price?symbol={pair}"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
            price = float(data.get("price", 0))
            if price > 0:
                marks[sym] = price
                mark_sources[sym] = "binance_us_ticker"
        except Exception as ex:
            REPORT.setdefault("mark_fetch_warnings", []).append(f"{sym}:{ex}")

    if conn is None:
        conn = sqlite3.connect(str(DB))
        close_conn = True
    else:
        close_conn = False
    try:
        conn.row_factory = sqlite3.Row
        for sym in SYMBOLS:
            if sym in marks and marks[sym] > 0:
                continue
            row = conn.execute(
                """
                SELECT entry_price, COALESCE(highest_price, 0) AS highest_price
                FROM portfolio_engine_positions
                WHERE symbol=?
                """,
                (sym,),
            ).fetchone()
            if row:
                highest = float(row["highest_price"] or 0)
                entry = float(row["entry_price"] or 0)
                mark = highest if highest > 0 else entry
                if mark > 0:
                    marks[sym] = mark
                    mark_sources[sym] = "sqlite_highest_price" if highest > 0 else "sqlite_entry_price"
    finally:
        if close_conn:
            conn.close()

    REPORT["mark_sources"] = mark_sources
    return marks


def clear_redis_paper_state(cash_balance: float, realized_pnl: float) -> None:
    try:
        import redis

        r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True)
        for sym in SYMBOLS:
            r.delete(f"paper:position:{sym}")
            r.srem("paper:positions:active", sym)
        r.set("paper:cash_balance", str(cash_balance))
        r.set("paper:realized_pnl_total", str(realized_pnl))
        REPORT["redis_paper_cleared"] = True
        REPORT["redis_cash_balance"] = cash_balance
        REPORT["redis_realized_pnl"] = realized_pnl
    except Exception as ex:
        REPORT["redis_paper_cleared"] = False
        REPORT["redis_error"] = str(ex)


def run_clear() -> None:
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, TAKER_FEE

    REPORT["preflight"] = preflight_checks()
    existing_backups = sorted(
        PROJECT_ROOT.glob("mystic_trading.db.backup_before_stale_corrected_path_position_clear_*")
    )
    if existing_backups:
        REPORT["backup_path"] = str(existing_backups[-1])
        REPORT["backup_reused_existing"] = True
    else:
        shutil.copy2(DB, BACKUP)
        REPORT["backup_path"] = str(BACKUP)

    conn = sqlite3.connect(str(DB), timeout=120.0)
    conn.execute("PRAGMA journal_mode=WAL")
    marks = fetch_reference_marks(conn)
    now_iso = datetime.now(timezone.utc).isoformat()
    now_epoch = time.time()
    cooldown_until = now_epoch + POST_SELL_COOLDOWN_WALL_SEC

    conn.execute("BEGIN IMMEDIATE")

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    positions = {
        r["symbol"]: dict(r)
        for r in cur.execute("SELECT symbol, quantity, entry_price, trade_id, entry_fee FROM portfolio_engine_positions ORDER BY symbol")
    }
    REPORT["positions_before"] = list(positions.keys())

    ledger_row = cur.execute(
        "SELECT cash_balance, realized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1"
    ).fetchone()
    cash_balance = float(ledger_row[0] or 0.0)
    realized_total = float(ledger_row[1] or 0.0)
    equity_before = float(ledger_row[2] or 0.0)

    admin_rows: dict[str, list] = {"paper_trades": [], "close_ledger": [], "audit": [], "learning": []}
    total_admin_realized = 0.0
    per_symbol_pnl: dict[str, float] = {}

    for sym in SYMBOLS:
        if sym not in positions:
            continue
        pos = positions[sym]
        qty = float(pos["quantity"])
        entry = float(pos["entry_price"])
        entry_fee = float(pos.get("entry_fee") or 0.0)
        mark = float(marks.get(sym) or 0.0) or entry

        fee = qty * mark * TAKER_FEE
        proceeds = qty * mark - fee
        entry_cost = qty * entry + entry_fee
        realized_pnl = proceeds - entry_cost
        pnl_pct = (realized_pnl / entry_cost) if entry_cost > 0 else 0.0
        gross_pct = (mark - entry) / entry if entry > 0 else 0.0
        net_pct = gross_pct - ESTIMATED_ROUNDTRIP_COST

        total_admin_realized += realized_pnl
        per_symbol_pnl[sym] = float(realized_pnl)
        cash_balance += proceeds
        realized_total += realized_pnl

        paper_run_row = cur.execute(
            "SELECT paper_run_id FROM paper_trades WHERE symbol=? AND UPPER(side)='BUY' ORDER BY id DESC LIMIT 1",
            (sym,),
        ).fetchone()
        paper_run_id = (paper_run_row[0] if paper_run_row else "default") or "default"

        sell_trade_id = f"stale_clear_{sym.replace('/', '')}_{int(now_epoch * 1000)}"
        explain = {
            "close_reason": CLOSE_REASON,
            "closed_by": "USER_ADMIN_RESET",
            "admin_clear": True,
            "stale_pre_correction_clear": True,
            "not_ai_profit_sell": True,
            "not_ai_trade": True,
            "not_strategy_sell": True,
            "not_live_trade": True,
            "good_trade": False,
            "bad_trade": True,
            "lesson": LESSON,
            "exit_is_reference_mark": True,
            "reference_mark": mark,
            "original_trade_id": pos.get("trade_id"),
        }
        diagnostics = {
            "source": "USER_ADMIN_RESET",
            "reason": CLOSE_REASON,
            "is_synthetic": 1,
            "admin_clear": True,
            "stale_pre_correction_clear": True,
            "not_ai_trade": True,
            "not_ai_profit_sell": True,
            "not_strategy_sell": True,
            "not_live_trade": True,
            "exit_is_reference_mark": True,
        }

        cur.execute(
            """
            INSERT INTO paper_trades (
                trade_id, paper_run_id, mode, symbol, side, quantity, price,
                entry_price, pnl, pnl_pct, remaining_position, hold_time_seconds,
                fees_paid, slippage_cost, exit_type, timestamp, status,
                explainability_json, diagnostics_json, sleeve, exit_reason,
                source, is_synthetic, mark_price, strategy_id
            ) VALUES (?, ?, 'paper', ?, 'SELL', ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, 'executed', ?, ?, 'ACTIVE', ?, ?, 1, ?, 'day')
            """,
            (
                sell_trade_id,
                paper_run_id,
                sym,
                qty,
                mark,
                entry,
                realized_pnl,
                pnl_pct,
                fee,
                max(0.0, (entry - mark) * qty),
                CLOSE_REASON,
                now_iso,
                json.dumps(explain),
                json.dumps(diagnostics),
                CLOSE_REASON,
                "USER_ADMIN_RESET",
                mark,
            ),
        )
        pt_id = cur.lastrowid
        admin_rows["paper_trades"].append({"id": pt_id, "trade_id": sell_trade_id, "symbol": sym, "pnl": realized_pnl})

        cur.execute(
            "UPDATE paper_trades SET remaining_position=0 WHERE symbol=? AND UPPER(side)='BUY'",
            (sym,),
        )

        cur.execute(
            """
            INSERT INTO position_close_ledger (
                symbol, closed_at, closed_at_epoch, close_reason, manual_sell,
                realized_profit, realized_profit_unknown, cooldown_until,
                quantity, entry_price, exit_price, sell_trade_id, detail
            ) VALUES (?, ?, ?, ?, 0, ?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                now_iso,
                now_epoch,
                CLOSE_REASON,
                float(realized_pnl),
                float(cooldown_until),
                qty,
                entry,
                mark,
                sell_trade_id,
                "closed_by=USER_ADMIN_RESET;admin_clear=true;stale_pre_correction_clear=true;not_ai_profit_sell=true;not_strategy_sell=true",
            ),
        )
        admin_rows["close_ledger"].append({"id": cur.lastrowid, "symbol": sym, "cooldown_until": cooldown_until})

        pre_ledger = dict(REPORT.get("ledger_before") or {})
        post_ledger = {
            "cash_balance": cash_balance,
            "positions_value": 0.0,
            "realized_pnl": realized_total,
            "unrealized_pnl": 0.0,
            "total_equity": cash_balance,
        }
        cur.execute(
            """
            INSERT INTO portfolio_engine_audit (
                ts, action, symbol, qty, price, fees, slippage, trade_id,
                pre_ledger_json, post_ledger_json, entry_reason, exit_reason, sleeve,
                invariant_ok, invariant_diff
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', 1, ?)
            """,
            (
                now_iso,
                AUDIT_ACTION,
                sym,
                qty,
                mark,
                fee,
                max(0.0, (entry - mark) * qty),
                sell_trade_id,
                json.dumps(pre_ledger),
                json.dumps(post_ledger),
                "USER_ADMIN_RESET",
                CLOSE_REASON,
                float(realized_pnl),
            ),
        )
        admin_rows["audit"].append({"id": cur.lastrowid, "symbol": sym})

        cur.execute(
            """
            INSERT INTO trade_learning_outcomes (
                written_at_utc, mode, symbol, entry_timestamp, exit_timestamp,
                entry_price, exit_price, quantity, fees_paid, slippage_cost,
                net_profit_usd, net_profit_pct, hold_seconds, decision_reason,
                manual_sell_flag, close_reason, dust_remaining_qty,
                dust_remaining_notional_usdt, realized_profit_unknown, extra_json
            ) VALUES (
                datetime('now'), 'paper', ?, NULL, ?, ?, ?, ?, ?, 0,
                ?, ?, NULL, ?, 0, ?, 0, 0, 0, ?
            )
            """,
            (
                sym.replace("/", ""),
                now_epoch,
                entry,
                mark,
                qty,
                fee,
                float(realized_pnl),
                float(net_pct),
                f"{CLOSE_REASON}:USER_ADMIN_RESET",
                CLOSE_REASON,
                json.dumps(
                    {
                        "source": "USER_ADMIN_RESET",
                        "admin_clear": True,
                        "stale_pre_correction_clear": True,
                        "not_ai_profit_sell": True,
                        "not_ai_trade": True,
                        "not_strategy_sell": True,
                        "not_live_trade": True,
                        "good_trade": False,
                        "bad_trade": True,
                        "lesson": LESSON,
                        "exit_is_reference_mark": True,
                        "closed_by": "USER_ADMIN_RESET",
                    }
                ),
            ),
        )
        admin_rows["learning"].append({"id": cur.lastrowid, "symbol": sym})

        REPORT.setdefault("cooldowns", {})[sym] = {
            "cooldown_until_epoch": cooldown_until,
            "cooldown_until_iso": datetime.fromtimestamp(cooldown_until, tz=timezone.utc).isoformat(),
            "wall_sec": POST_SELL_COOLDOWN_WALL_SEC,
        }

    cur.execute("DELETE FROM portfolio_engine_positions")
    cur.execute(
        """
        UPDATE portfolio_engine_ledger SET
            cash_balance=?,
            positions_value=0,
            realized_pnl=?,
            unrealized_pnl=0,
            total_equity=?,
            account_status='HEALTHY',
            trading_paused=0,
            pause_reason='',
            last_updated=?
        WHERE id=1
        """,
        (cash_balance, realized_total, cash_balance, now_iso),
    )
    conn.commit()
    conn.close()

    REPORT["admin_rows"] = admin_rows
    REPORT["per_symbol_realized_pnl"] = per_symbol_pnl
    REPORT["total_admin_realized_pnl"] = total_admin_realized
    REPORT["ledger_after"] = {
        "cash_balance": cash_balance,
        "positions_value": 0.0,
        "realized_pnl": realized_total,
        "unrealized_pnl": 0.0,
        "total_equity": cash_balance,
        "equity_delta_from_before": cash_balance - equity_before,
    }
    REPORT["labels"] = {
        "close_reason": CLOSE_REASON,
        "closed_by": "USER_ADMIN_RESET",
        "source": "USER_ADMIN_RESET",
        "is_synthetic": 1,
        "admin_clear": True,
        "stale_pre_correction_clear": True,
        "not_ai_profit_sell": True,
        "not_strategy_sell": True,
        "not_live_trade": True,
        "good_trade": False,
        "bad_trade": True,
    }
    clear_redis_paper_state(cash_balance, realized_total)


if __name__ == "__main__":
    run_clear()
    print(json.dumps(REPORT, indent=2, default=str))
