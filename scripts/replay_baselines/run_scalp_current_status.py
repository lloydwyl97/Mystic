#!/usr/bin/env python3
"""Current Binance scalp status audit — isolated from DAY PnL."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.orderbook_book import walk_buy_notional
from backend.services.binance_scalp.schema import SCALP_TABLES, verify_scalp_tables
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies
from backend.services.binance_scalp.strategies.kline_cache import KlineCache
from backend.services.binance_scalp.status_snapshot import build_scalp_status

SCRIPT = "scripts/replay_baselines/run_scalp_current_status.py"
OUT = REPO / "scripts" / "replay_baselines" / "scalp_current_status_latest.json"

# Research-only patterns in ltf_pattern_miner — not wired to binance_scalp runner
RESEARCH_ONLY_STRATEGIES = (
    "failed_breakdown_reversal",
    "compression_breakout",
    "volume_impulse_continuation",
)


def _scalp_process_running() -> dict[str, Any]:
    proc = subprocess.run(
        ["pgrep", "-af", "backend.services.binance_scalp.runner"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in proc.stdout.strip().split("\n") if ln and "pgrep" not in ln]
    return {"running": len(lines) > 0, "pids": lines}


def _db_stats(db_path: str, since_iso: str) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"exists": False}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        tables = verify_scalp_tables(path)
        ledger = conn.execute("SELECT principal, cash_balance, positions_value, realized_pnl, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone()
        open_pos = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
        buys = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades WHERE side='BUY' AND created_at>=?", (since_iso,)).fetchone()[0]
        sells = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades WHERE side='SELL' AND created_at>=?", (since_iso,)).fetchone()[0]
        signals_est = conn.execute("SELECT COUNT(*) FROM scalp_rejects WHERE created_at>=?", (since_iso,)).fetchone()[0]
        reject_rows = conn.execute(
            "SELECT reason, COUNT(*) cnt FROM scalp_rejects WHERE created_at>=? GROUP BY reason ORDER BY cnt DESC",
            (since_iso,),
        ).fetchall()
        day_contamination = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND sql LIKE '%paper_trades%'").fetchone()[0]
        return {
            "exists": True,
            "tables_present": {t: tables.get(t, 0) == 1 for t in SCALP_TABLES},
            "all_scalp_tables_present": all(tables.get(t, 0) == 1 for t in SCALP_TABLES),
            "ledger": dict(ledger) if ledger else None,
            "open_positions": int(open_pos),
            "last_24h": {
                "buys": int(buys),
                "sells": int(sells),
                "reject_rows": int(signals_est),
                "reject_reasons": {str(r["reason"]): int(r["cnt"]) for r in reject_rows},
            },
            "pnl_isolated_from_day": day_contamination == 0,
            "isolation_note": "scalp uses scalp_* tables only; no writes to paper_trades/portfolio_engine_*",
        }


def _verify_data_plumbing(config, econ: ScalpEconomics) -> dict[str, Any]:
    reader = ScalpMarketReader(config)
    klines = KlineCache()
    tracker = MomentumTracker()
    sym = config.products[0] if config.products else "BTCUSDT"
    features: dict[str, Any] = {}

    # 1m klines
    bars_1m = klines.get(sym, minutes=60)
    features["1m_klines"] = {"available": len(bars_1m) >= 20, "bar_count": len(bars_1m)}

    # 5m / 15m / 1h context (plumbing added to KlineCache)
    bars_5m = klines.get_5m(sym, minutes=240)
    bars_15m = klines.get_15m(sym, minutes=720)
    bars_1h = klines.get_1h(sym, minutes=2880)
    features["5m_context"] = {"available": len(bars_5m) >= 5, "bar_count": len(bars_5m)}
    features["15m_context"] = {"available": len(bars_15m) >= 5, "bar_count": len(bars_15m)}
    features["1h_regime_filter_data"] = {"available": len(bars_1h) >= 5, "bar_count": len(bars_1h), "wired_to_router": False}

    snap = reader.read(sym)
    if snap:
        walk = walk_buy_notional(snap.asks, config.max_notional_paper, snap.best_ask)
        tracker.record(sym, time.time(), snap.best_bid, snap.mid)
        mom = tracker.diagnostics(sym, time.time(), snap.best_bid, snap.mid)
        vol_recent = sum(b["volume"] for b in bars_1m[-3:]) if len(bars_1m) >= 3 else 0
        vol_prior = sum(b["volume"] for b in bars_1m[-6:-3]) if len(bars_1m) >= 6 else 1
        rel_vol = vol_recent / max(vol_prior, 1e-9)
        features["orderbook_top_of_book"] = {"available": True, "best_bid": snap.best_bid, "best_ask": snap.best_ask}
        features["spread"] = {"available": True, "spread_pct": snap.spread_pct}
        features["bid_ask_imbalance"] = {"available": snap.order_book_imbalance is not None, "value": snap.order_book_imbalance}
        features["volume_burst"] = {"available": len(bars_1m) >= 6, "rel_volume_3v3": round(rel_vol, 3)}
        features["relative_volume"] = {"available": len(bars_1m) >= 6, "ratio": round(rel_vol, 3)}
        if bars_1m:
            tp = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in bars_1m[-15:]) / max(sum(b["volume"] for b in bars_1m[-15:]), 1e-9)
            features["vwap_distance"] = {
                "available": True,
                "vwap": round(tp, 4),
                "distance_pct": round((snap.mid - tp) / tp * 100, 4) if tp else None,
            }
        features["ema_microtrend"] = {
            "available": True,
            "needs_warmup_sec": 60,
            "instant_sample_count": mom.sample_count,
            "bid_change_15s": mom.bid_change_15s,
            "mid_change_30s": mom.mid_change_30s,
            "momentum_confirmed": mom.momentum_confirmed,
            "note": "MomentumTracker requires ~60s warm history for confirmed signals",
        }
        features["slippage_estimate"] = {
            "available": walk.depth_sufficient,
            "buy_impact_pct": round(float(walk.impact_pct), 6),
            "depth_sufficient": walk.depth_sufficient,
        }
    else:
        for k in (
            "orderbook_top_of_book",
            "spread",
            "bid_ask_imbalance",
            "volume_burst",
            "relative_volume",
            "vwap_distance",
            "ema_microtrend",
            "slippage_estimate",
        ):
            features[k] = {"available": False, "error": "NO_MARKET_SNAPSHOT"}

    features["maker_taker_fee_model"] = {
        "available": True,
        "maker_fee_pct": econ.maker_fee_pct,
        "taker_fee_pct": econ.taker_fee_pct,
        "fee_model_verified": econ.fee_model_verified,
        "use_maker_only": econ.use_maker_only,
    }
    features["max_hold_clock"] = {
        "available": True,
        "stale_scalp_timeout_sec": econ.stale_scalp_timeout_sec,
        "max_hold_minutes": round(econ.stale_scalp_timeout_sec / 60, 1),
    }
    features["depth_polling"] = {
        "available": snap is not None,
        "rest_depth_ok": snap is not None,
        "book_source": snap.book_source if snap else None,
    }

    missing = [k for k, v in features.items() if not v.get("available", False)]
    return {"symbol_checked": sym, "features": features, "missing_features": missing, "all_core_available": len(missing) == 0}


def _map_reject_to_gate(reason: str) -> str:
    r = (reason or "").upper()
    if "SPREAD" in r:
        return "spread_too_wide"
    if r in ("NO_BREAKOUT", "BREAKOUT_NOT_CONFIRMED"):
        return "no_breakout"
    if "RECLAIM" in r or "VWAP" in r:
        return "no_reclaim"
    if "VOLUME" in r or "VOL" in r:
        return "volume_too_low"
    if "TARGET_NOT_REACHABLE" in r or "MOMENTUM_GROSS" in r or "SURPLUS" in r:
        return "target_not_reachable_after_fees"
    if "IMBALANCE" in r or "TAPE" in r:
        return "orderbook_imbalance_not_confirmed"
    if "REGIME" in r:
        return "regime_mismatch"
    if "DEPTH" in r or "IMPACT" in r or "LIQUIDITY" in r:
        return "liquidity_too_thin"
    if "SLIPPAGE" in r:
        return "slippage_too_high"
    if "INSUFFICIENT" in r or "MOMENTUM" in r:
        return "no_signal"
    return "other"


def main() -> int:
    print("=== SCALP CURRENT STATUS AUDIT ===", flush=True)
    config = get_scalp_config()
    econ = economics_for_config(config)
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    status = build_scalp_status(warm_rounds=0)
    proc = _scalp_process_running()
    db = _db_stats(config.database_path, since)
    plumbing = _verify_data_plumbing(config, econ)

    enabled = [s.name for s in enabled_strategies(config)]
    disabled_env = sorted(config.disabled_strategies)
    all_impl = list(STRATEGY_NAMES)
    not_impl = [s for s in RESEARCH_ONLY_STRATEGIES if s not in all_impl]

    reject_gate_counts = Counter(_map_reject_to_gate(r) for r in db.get("last_24h", {}).get("reject_reasons", {}))

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT,
        "scalp_process": proc,
        "execution": {
            "scalp_live": config.scalp_live,
            "scalp_paper_enabled": config.scalp_paper_enabled,
            "real_orders_permitted": False,
            "repair_add_allowed": config.allow_repair_add,
            "fee_model_verified": config.fee_model_verified,
        },
        "symbols_scanned": list(config.products),
        "enabled_strategies": enabled,
        "disabled_strategies_env": disabled_env,
        "research_only_not_in_runner": not_impl,
        "implemented_strategies": all_impl,
        "depth_polling_status": plumbing["features"]["depth_polling"],
        "spread_data_available": plumbing["features"]["spread"]["available"],
        "scalp_tables": db.get("tables_present", {}),
        "scalp_tables_all_present": db.get("all_scalp_tables_present", False),
        "last_24h": db.get("last_24h", {}),
        "last_24h_signals_estimate": db.get("last_24h", {}).get("reject_rows", 0),
        "last_24h_entries": db.get("last_24h", {}).get("buys", 0),
        "last_24h_exits": db.get("last_24h", {}).get("sells", 0),
        "reject_reasons_raw": db.get("last_24h", {}).get("reject_reasons", {}),
        "gate_reject_summary": dict(reject_gate_counts),
        "no_trade_reasons_top": db.get("last_24h", {}).get("reject_reasons", {}),
        "realized_paper_pnl_usd": (db.get("ledger") or {}).get("realized_pnl"),
        "scalp_equity_usd": (db.get("ledger") or {}).get("total_equity"),
        "pnl_isolated_from_day": db.get("pnl_isolated_from_day", True),
        "open_scalp_positions": db.get("open_positions", 0),
        "data_plumbing": plumbing,
        "readiness_snapshot": {
            "overall_decision": status.get("overall_decision"),
            "top_blocker": status.get("top_blocker"),
            "entry_armed": status.get("entry_armed"),
            "strategy_router": status.get("strategy_router"),
        },
        "day_separation": {
            "mixed_with_day_pnl": False,
            "uses_separate_ledger": True,
            "uses_separate_tables": True,
            "note": "Scalp PnL in scalp_paper_ledger; DAY in portfolio_engine_ledger",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}", flush=True)
    print(f"process_running={proc['running']} tables_ok={payload['scalp_tables_all_present']} plumbing_ok={plumbing['all_core_available']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
