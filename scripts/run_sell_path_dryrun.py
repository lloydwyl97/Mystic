#!/usr/bin/env python3
"""
Isolated sell-path / protected-execution dry-run against a copied DB.
Does not touch production mystic_trading.db or running services.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before backend imports that read config
_env = PROJECT_ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

PROD_DB = PROJECT_ROOT / "mystic_trading.db"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
TEST_DB = PROJECT_ROOT / f"mystic_trading.db.test_sell_path_{TS}"
REPORT: dict[str, Any] = {"timestamp_utc": TS, "test_db": str(TEST_DB)}


def _q(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return list(conn.execute(sql, params))
    finally:
        conn.close()


def _q1(db: Path, sql: str, params: tuple = ()) -> Any:
    rows = _q(db, sql, params)
    return rows[0][0] if rows else None


def baseline(db: Path) -> dict[str, Any]:
    positions = _q(
        db,
        "SELECT symbol, quantity, entry_price, repair_add_count FROM portfolio_engine_positions ORDER BY symbol",
    )
    return {
        "positions": positions,
        "positions_count": len(positions),
        "paper_trades": int(_q1(db, "SELECT COUNT(*) FROM paper_trades") or 0),
        "paper_buys": int(_q1(db, "SELECT COUNT(*) FROM paper_trades WHERE UPPER(side)='BUY'") or 0),
        "paper_sells": int(_q1(db, "SELECT COUNT(*) FROM paper_trades WHERE UPPER(side)='SELL'") or 0),
        "audit_rows": int(_q1(db, "SELECT COUNT(*) FROM portfolio_engine_audit") or 0),
        "audit_sells": int(_q1(db, "SELECT COUNT(*) FROM portfolio_engine_audit WHERE UPPER(action)='SELL'") or 0),
        "close_ledger": int(_q1(db, "SELECT COUNT(*) FROM position_close_ledger") or 0),
        "learning_rows": int(_q1(db, "SELECT COUNT(*) FROM trade_learning_outcomes") or 0),
    }


def copy_db(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()


def fetch_api(path: str) -> tuple[int, Any]:
    url = f"http://localhost:8000{path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as ex:
        return 0, {"error": str(ex)}


async def run_test() -> None:
    from backend.config.protected_execution import (
        DEPTH_INSUFFICIENT,
        PRICE_IMPACT_TOO_HIGH,
        USE_PROTECTED_LIMIT_EXECUTION,
    )
    from backend.config.trading_economics import (
        ESTIMATED_ROUNDTRIP_COST,
        MIN_NET_PROFIT_TO_SELL,
        TAKER_FEE,
    )
    from backend.services.day_active_market_bundle import (
        async_fetch_day_active_ohlcv_bundle,
        validate_day_active_bundle,
    )
    from backend.services.execution_mode_service import is_live_execution_allowed_sync
    from backend.services.portfolio_engine import PortfolioEngine
    from backend.services.protected_limit_execution import run_protected_preflight
    from backend.utils.symbols import normalize_symbol

    REPORT["test_method"] = "copied_db_isolated_engine"
    REPORT["production_db"] = str(PROD_DB)

    # PHASE 1 — baseline
    prod_before = baseline(PROD_DB)
    REPORT["production_baseline"] = prod_before

    exec_status, exec_body = fetch_api("/api/portfolio-engine/execution-protection")
    REPORT["execution_protection_api"] = {"http": exec_status, "data": exec_body.get("data") if isinstance(exec_body, dict) else exec_body}

    from backend.services.execution_mode_service import get_execution_status

    exec_mode = await get_execution_status()
    REPORT["execution_mode"] = exec_mode
    REPORT["economics"] = {
        "MIN_NET_PROFIT_TO_SELL": MIN_NET_PROFIT_TO_SELL,
        "ESTIMATED_ROUNDTRIP_COST": ESTIMATED_ROUNDTRIP_COST,
        "TAKER_FEE": TAKER_FEE,
        "USE_PROTECTED_LIMIT_EXECUTION": USE_PROTECTED_LIMIT_EXECUTION,
    }

    # ai_signal keys (read-only Redis)
    ai_keys: list[str] = []
    try:
        from backend.config.redis_config import get_shared_redis_async

        redis = get_shared_redis_async()
        async for key in redis.scan_iter("ai:signal:*", count=50):
            ai_keys.append(key.decode() if isinstance(key, bytes) else str(key))
        ai_keys.sort()
    except Exception as ex:
        ai_keys = [f"redis_scan_error:{ex}"]
    REPORT["ai_signal_keys_sample"] = ai_keys[:20]

    test_before = dict(prod_before)
    REPORT["test_db_baseline"] = test_before

    # PHASE 2 — copy DB
    copy_db(PROD_DB, TEST_DB)
    REPORT["test_db_created"] = True

    # PHASE 3 — isolated engine
    engine = PortfolioEngine(db_path=str(TEST_DB), test_mode=True)
    engine._live_execution_enabled = False
    engine._live_service = None
    await engine.initialize_from_db()
    await engine._recompute_positions_values()

    symbol = "XRP/USDT"
    ns = normalize_symbol(symbol)
    position = engine.open_positions.get(ns)
    if not position:
        REPORT["error"] = f"no position for {ns}"
        return

    entry = float(position.entry_price)
    qty = float(position.quantity)
    extra = 0.001
    test_mark = entry * (1.0 + ESTIMATED_ROUNDTRIP_COST + MIN_NET_PROFIT_TO_SELL + extra)
    net_pct_at_mark = (test_mark - entry) / entry - ESTIMATED_ROUNDTRIP_COST

    REPORT["test_symbol"] = ns
    REPORT["test_entry"] = entry
    REPORT["test_quantity"] = qty
    REPORT["test_mark"] = test_mark
    REPORT["threshold_math"] = {
        "formula": "entry * (1 + ESTIMATED_ROUNDTRIP_COST + MIN_NET_PROFIT_TO_SELL + 0.001)",
        "ESTIMATED_ROUNDTRIP_COST": ESTIMATED_ROUNDTRIP_COST,
        "MIN_NET_PROFIT_TO_SELL": MIN_NET_PROFIT_TO_SELL,
        "extra_buffer": extra,
        "net_pct_at_mark": net_pct_at_mark,
        "passes_exit_floor": net_pct_at_mark + 1e-12 >= MIN_NET_PROFIT_TO_SELL,
    }

    # Inject profitable fresh mark into engine price cache (test harness only)
    engine._price_cache.set(ns, test_mark)

    # Real DAY bundle for exit context gate
    hold_bds: dict[str, dict] = {}
    hold_ms: dict[str, list[str]] = {}
    bundle_missing: list[str] = []
    try:
        from backend.services.live_market_data import live_market_data_service

        if live_market_data_service:
            bd = await async_fetch_day_active_ohlcv_bundle(live_market_data_service, ns)
            ok, miss = validate_day_active_bundle(bd)
            hold_bds[ns] = bd if isinstance(bd, dict) else {}
            hold_ms[ns] = [] if ok else list(miss)
            bundle_missing = list(miss) if not ok else []
        else:
            hold_bds[ns] = {}
            hold_ms[ns] = ["live_market_data_service_unavailable"]
            bundle_missing = hold_ms[ns]
    except Exception as ex:
        hold_bds[ns] = {}
        hold_ms[ns] = [f"bundle_fetch_error:{ex}"]
        bundle_missing = hold_ms[ns]

    REPORT["day_bundle_ok"] = len(bundle_missing) == 0
    REPORT["day_bundle_missing"] = bundle_missing

    current_bar = int(time.time() / 60) * 60
    current_prices = {ns: test_mark}

    # Direct _check_exit_conditions probe
    exit_check_passed = False
    exit_check_detail = ""
    if len(bundle_missing) == 0:
        chk = await engine._check_exit_conditions(
            position,
            test_mark,
            current_bar,
            day_hold_bundle=hold_bds.get(ns),
            day_hold_missing=hold_ms.get(ns),
        )
        exit_check_passed = chk is not None
        exit_check_detail = "passed_and_called_execute" if chk else "returned_none"
        if chk:
            REPORT["exit_check_direct_result"] = {
                "symbol": chk.get("symbol"),
                "realized_pnl": chk.get("realized_pnl"),
                "pnl_pct": chk.get("pnl_pct"),
                "exit_type": chk.get("exit_type"),
            }
    else:
        exit_check_detail = f"skipped_day_bundle_missing:{bundle_missing}"

    REPORT["check_exit_conditions_passed"] = exit_check_passed
    REPORT["check_exit_conditions_detail"] = exit_check_detail

    # If direct check already sold, skip monitor; else try monitor on fresh engine state
    sells_after_direct = int(_q1(TEST_DB, "SELECT COUNT(*) FROM paper_trades WHERE UPPER(side)='SELL'") or 0)
    if sells_after_direct == 0 and len(bundle_missing) == 0:
        # Reload engine state for monitor path (direct check may have mutated position)
        engine2 = PortfolioEngine(db_path=str(TEST_DB), test_mode=True)
        engine2._live_execution_enabled = False
        engine2._live_service = None
        await engine2.initialize_from_db()
        engine2._price_cache.set(ns, test_mark)
        exits = await engine2.monitor_all_positions(
            current_prices,
            current_bar,
            symbols={ns},
            hold_day_bundles=hold_bds,
            hold_day_missing=hold_ms,
        )
        REPORT["monitor_exits"] = exits
        REPORT["monitor_path_used"] = True
    else:
        REPORT["monitor_path_used"] = sells_after_direct > 0

    from backend.services.protected_limit_execution import get_last_execution_protection_state

    pf_state = get_last_execution_protection_state(taker_fee=TAKER_FEE)
    REPORT["protected_preflight_after_sell"] = pf_state
    REPORT["protected_preflight_ran"] = pf_state.get("last_preflight_passed") is not None
    REPORT["protected_preflight_result"] = {
        "passed": pf_state.get("last_preflight_passed"),
        "reject_reason": pf_state.get("last_preflight_reject_reason"),
        "execution_mode": pf_state.get("last_execution_mode"),
        "expected_avg_fill": pf_state.get("last_expected_avg_fill"),
        "protected_limit_price": pf_state.get("last_protected_limit_price"),
        "spread_pct": pf_state.get("spread_pct"),
        "price_impact_pct": pf_state.get("last_price_impact_pct"),
    }

    test_after_sell = baseline(TEST_DB)
    test_sells = int(_q1(TEST_DB, "SELECT COUNT(*) FROM paper_trades WHERE UPPER(side)='SELL'") or 0)
    REPORT["test_sell_executed"] = test_sells > test_before["paper_sells"]

    # PHASE 4 — verify test DB outputs
    sell_row = _q(
        TEST_DB,
        "SELECT trade_id, symbol, side, quantity, price, pnl, pnl_pct, exit_type, mode, diagnostics_json FROM paper_trades WHERE UPPER(side)='SELL' ORDER BY rowid DESC LIMIT 1",
    )
    audit_sell = _q(
        TEST_DB,
        "SELECT id, action, symbol, qty, price, trade_id FROM portfolio_engine_audit WHERE UPPER(action)='SELL' ORDER BY id DESC LIMIT 1",
    )
    close_row = _q(
        TEST_DB,
        "SELECT symbol, close_reason, realized_profit, cooldown_until FROM position_close_ledger ORDER BY id DESC LIMIT 1",
    )
    learn_row = _q(
        TEST_DB,
        "SELECT symbol, close_reason, payload_json FROM trade_learning_outcomes ORDER BY rowid DESC LIMIT 1",
    )

    learn_payload = {}
    if learn_row:
        try:
            learn_payload = json.loads(learn_row[0][2] or "{}")
        except json.JSONDecodeError:
            learn_payload = {}

    pos_after = _q(TEST_DB, "SELECT symbol, quantity FROM portfolio_engine_positions WHERE symbol=?", (ns,))

    REPORT["test_db_verification"] = {
        "position_after": pos_after,
        "position_removed_or_reduced": not pos_after or float(pos_after[0][1] or 0) < qty - 1e-9,
        "sell_rows_total": test_sells,
        "new_sells": test_sells - test_before["paper_sells"],
        "paper_sell_row": sell_row[0] if sell_row else None,
        "audit_sell_row": audit_sell[0] if audit_sell else None,
        "close_ledger_row": close_row[0] if close_row else None,
        "learning_row_symbol": learn_row[0][0] if learn_row else None,
        "learning_close_reason": learn_row[0][1] if learn_row else None,
        "good_trade": learn_payload.get("good_trade"),
        "bad_trade": learn_payload.get("bad_trade"),
        "net_profit_usd": learn_payload.get("net_profit_usd") or learn_payload.get("realized_profit"),
        "net_profit_pct": learn_payload.get("net_profit_pct"),
        "diagnostics_execution_mode": (json.loads(sell_row[0][9]).get("execution_mode") if sell_row and sell_row[0][9] else None),
    }

    # Duplicate check
    dup_sells = _q1(
        TEST_DB,
        "SELECT COUNT(*) FROM paper_trades WHERE UPPER(side)='SELL' AND timestamp > datetime('now', '-1 hour')",
    )
    dup_audit = _q1(
        TEST_DB,
        "SELECT COUNT(*) FROM portfolio_engine_audit WHERE UPPER(action)='SELL' AND ts > datetime('now', '-1 hour')",
    )
    REPORT["duplicate_check"] = {
        "recent_sell_trades": int(dup_sells or 0),
        "recent_audit_sells": int(dup_audit or 0),
        "no_duplicates": int(dup_sells or 0) <= 1 and int(dup_audit or 0) <= 1,
    }

    # PHASE 5 — oversized preflight reject (no order)
    counts_before_reject = baseline(TEST_DB)
    huge_qty = qty * 5000.0
    reject_pf = await run_protected_preflight(
        symbol=ns,
        side="SELL",
        quantity=huge_qty,
        reference_price=test_mark,
        live_capable=False,
    )
    counts_after_reject = baseline(TEST_DB)
    REPORT["oversized_preflight_test"] = {
        "quantity": huge_qty,
        "passed": reject_pf.passed,
        "reject_reason": reject_pf.reject_reason,
        "expected_codes": [PRICE_IMPACT_TOO_HIGH, DEPTH_INSUFFICIENT],
        "reason_ok": reject_pf.reject_reason in (PRICE_IMPACT_TOO_HIGH, DEPTH_INSUFFICIENT),
        "no_db_mutation": counts_before_reject == counts_after_reject,
        "counts_before": counts_before_reject,
        "counts_after": counts_after_reject,
    }

    # PHASE 6 — production unchanged
    prod_after = baseline(PROD_DB)
    REPORT["production_after"] = prod_after
    REPORT["production_unchanged"] = prod_before == prod_after

    _, status_body = fetch_api("/api/portfolio-engine/status")
    api_positions = []
    if isinstance(status_body, dict):
        api_positions = status_body.get("data", {}).get("positions") or []
    REPORT["production_api_positions_count"] = len(api_positions)

    REPORT["live_trades_allowed"] = is_live_execution_allowed_sync()
    REPORT["test_db_path"] = str(TEST_DB)
    REPORT["test_db_removable"] = True


def main() -> None:
    asyncio.run(run_test())
    print(json.dumps(REPORT, indent=2, default=str))


if __name__ == "__main__":
    main()
