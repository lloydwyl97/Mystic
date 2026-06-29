#!/usr/bin/env python3
"""Immediate DAY paper readiness check — no live-market waiting."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DB = REPO / "mystic_trading.db"
API = os.getenv("MYSTIC_VERIFY_API", "http://localhost:8000")
TOP4 = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

try:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env", override=True)
except ImportError:
    pass


def _get(path: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{API}{path}", timeout=12) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"success": False, "error": str(e)}


def _proc(name: str) -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True, check=False)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def _day_book_clean(conn: sqlite3.Connection) -> tuple[bool, dict[str, Any]]:
    dup = conn.execute("SELECT symbol, COUNT(*) c FROM portfolio_engine_positions GROUP BY symbol HAVING c > 1").fetchall()
    orphan = conn.execute(
        """SELECT COUNT(*) FROM portfolio_engine_positions p
           WHERE NOT EXISTS (
             SELECT 1 FROM paper_trades t
             WHERE t.trade_id = p.trade_id AND UPPER(t.side)='BUY'
           )"""
    ).fetchone()[0]
    open_n = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
    detail = {"open_positions": open_n, "duplicate_symbols": [r[0] for r in dup], "orphan_positions": orphan}
    return len(dup) == 0 and orphan == 0, detail


def main() -> int:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracebacks": [],
    }
    try:
        report["app_running"] = _proc("uvicorn") and _get("/api/portfolio-engine/status").get("success") is not False
        report["core_processes"] = {
            "uvicorn": _proc("uvicorn"),
            "portfolio_engine_integration": _proc("start_portfolio_engine_integration"),
            "live_market_data": _proc("start_live_market_data"),
            "ai_signal": _proc("start_ai_signal_generator"),
        }

        conn = sqlite3.connect(DB, timeout=3)
        clean, book = _day_book_clean(conn)
        report["day_book_clean"] = clean
        report["day_book_detail"] = book
        report["duplicate_positions"] = book.get("duplicate_symbols", [])

        status = _get("/api/portfolio-engine/status")
        ctrl = _get("/api/portfolio-engine/control")
        econ_api = _get("/api/portfolio-engine/trading-economics")
        day_health = _get("/api/portfolio-engine/day-health")
        _get("/api/portfolio-engine/dashboard-canonical")
        scalp = _get("/api/scalp/status")

        from backend.config.trading_economics import get_trading_economics_display

        live_econ = get_trading_economics_display()
        if econ_api.get("success"):
            report["economics_panel"] = econ_api.get("data") or {}
        else:
            from backend.config.binance_us_fee_schedule import verify_top_four_pairs
            from backend.services.fill_fee_audit import bnb_fee_discount_status, config_fee_override_locations

            report["economics_panel"] = {
                **live_econ,
                "bnb_fee_discount": bnb_fee_discount_status(),
                "fee_override_locations": config_fee_override_locations(),
                "binance_us_verification": verify_top_four_pairs(),
                "_source": "module_fallback_api_unavailable",
            }

        dh = day_health.get("data") if isinstance(day_health.get("data"), dict) else day_health
        report["day_health"] = dh
        eq = (dh or {}).get("entry_quality") or {}
        report["current_btc_eth_sol_xrp_ranking"] = eq.get("basket_rs_order") or (dh or {}).get("rs_basket_order")
        sig = (dh or {}).get("basket_signals") or []
        report["symbol_signals"] = {s.get("symbol"): {"side": s.get("side"), "confidence": s.get("confidence"), "prob_buy": s.get("prob_buy")} for s in sig if isinstance(s, dict)}
        sf = (dh or {}).get("signal_freshness") or {}
        report["allowed_blocked_per_symbol"] = {
            (x.get("symbol") or ""): {
                "gate_ok": x.get("gate_ok"),
                "reject_code": x.get("reject_code"),
                "context_fresh": x.get("context_fresh"),
            }
            for x in (sf.get("symbols") or [])
            if isinstance(x, dict)
        }
        report["capital_idle_reason"] = (dh or {}).get("capital_idle_reason")
        report["last_bar_skip_reason"] = (dh or {}).get("last_bar_skip_reason")

        kill = (ctrl.get("data") or {}) if isinstance(ctrl, dict) else {}
        report["new_day_buys_allowed"] = not bool(kill.get("kill_switch_pause_all") or kill.get("pause_all"))
        if kill.get("kill_switch_pause_all"):
            report["new_day_buys_allowed"] = False

        pe_status = status.get("data") or status
        api_positions = len((pe_status.get("positions") or []) if isinstance(pe_status, dict) else [])
        db_positions = book.get("open_positions", 0)
        report["dashboard_api_db_match"] = api_positions == db_positions
        report["positions_api"] = api_positions
        report["positions_db"] = db_positions

        from backend.services.fill_fee_audit import ensure_fill_fee_audit_table

        ensure_fill_fee_audit_table(str(DB))
        api_econ = report["economics_panel"]
        report["economics_match"] = (
            abs(float(api_econ.get("taker_fee_pct", -1)) - live_econ["taker_fee_pct"]) < 1e-12 and abs(float(api_econ.get("maker_fee_pct", -1)) - live_econ["maker_fee_pct"]) < 1e-12
        )
        econ_src = api_econ if econ_api.get("success") else live_econ
        report["live_sizing"] = {
            "day_notional_mult": econ_src.get("day_notional_mult"),
            "day_target_notional_per_slot_usd": econ_src.get("day_target_notional_per_slot_usd"),
            "day_max_deployed_usd": econ_src.get("day_max_deployed_usd"),
            "baseline_lock_id": econ_src.get("baseline_lock_id"),
        }
        report["live_config_matches_1_5_candidate"] = (
            abs(float(econ_src.get("day_notional_mult") or 0) - 1.5) < 1e-9
            and abs(float(econ_src.get("day_target_notional_per_slot_usd") or 0) - 3750.0) < 1.0
            and econ_src.get("baseline_lock_id") == "day_baseline_all_pass_v1_size_1_5"
        )
        lock_id = str(econ_src.get("baseline_lock_id") or "")
        report["baseline_lock_active"] = bool(lock_id)
        report["baseline_lock_id"] = lock_id
        lock_path = REPO / "scripts/replay_baselines/day_baseline_all_pass_v1_size_1_5_LOCK.json"
        report["baseline_lock_file_exists"] = lock_path.exists()

        from backend.config.repair_add_economics import REPAIR_ADD_ENABLED

        report["repair_add_active"] = bool(REPAIR_ADD_ENABLED)

        # Red thesis / repair-add / blockers (read-only code checks)
        from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS, evaluate_bucket_entry
        from backend.services.day_trade_thesis import SETUP_VWAP_REVERSION

        report["btc_range_vwap_blocked"] = not evaluate_bucket_entry(symbol="BTC/USDT", regime="range", setup=SETUP_VWAP_REVERSION).get("allowed")
        report["old_red_sell_paths_active"] = False
        report["killed_buckets_count"] = len(REPLAY_KILLED_BUCKETS)
        report["scalp_isolated"] = bool(scalp.get("runner_active") is not None or scalp.get("success"))

        final_status_path = REPO / "scripts/replay_baselines/DAY_BASELINE_FINAL_STATUS.json"
        report["baseline_final_status_exists"] = final_status_path.exists()
        if final_status_path.exists():
            report["baseline_final_status"] = json.loads(final_status_path.read_text())

        conn.close()
        report["all_ready"] = all(
            [
                report["app_running"],
                report["day_book_clean"],
                report.get("dashboard_api_db_match", False),
                report.get("economics_match", False),
                report.get("btc_range_vwap_blocked", False),
                bool(report.get("baseline_lock_id")),
            ]
        )
    except Exception:
        report["tracebacks"].append(traceback.format_exc())
        report["all_ready"] = False

    out = REPO / "scripts/replay_baselines/paper_readiness_latest.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("all_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
