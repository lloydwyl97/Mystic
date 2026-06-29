#!/usr/bin/env python3
"""Deep diagnostics for Phase 4 paper calibration trades."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.calibration_profiles import apply_profile
from backend.services.binance_scalp.economics import ScalpEconomics

TRADE_IDS = (
    "scalp_paper_XRPUSDT_1780927511905",
    "scalp_paper_XRPUSDT_1780927617915",
    "scalp_paper_SOLUSDT_1780928639117",
)
STALE_OPTS = (180, 210, 240, 300)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000"
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "30", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows = json.loads(proc.stdout)
    return [
        {
            "ts_ms": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in rows
    ]


def parse_ts(ts: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts[:19], fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return 0


def simulate_stale_outcomes(
    bars: list[dict],
    entry_ms: int,
    entry_px: float,
    entry_spread: float,
    econ: ScalpEconomics,
) -> dict[str, dict]:
    rt = econ.roundtrip_cost_pct(entry_spread, 0.0, 0.0)
    target_px = entry_px * (1.0 + econ.net_profit_target_pct + rt)
    post = [b for b in bars if b["ts_ms"] >= entry_ms]
    out: dict[str, dict] = {}
    for stale in STALE_OPTS:
        exit_px = entry_px
        exit_reason = "OPEN"
        hold_sec = stale
        hit_target = False
        for b in post:
            age = (b["ts_ms"] - entry_ms) / 1000.0
            if b["high"] >= target_px:
                exit_px = target_px
                exit_reason = "NET_PROFIT_TARGET"
                hold_sec = age
                hit_target = True
                break
            if age >= stale:
                exit_px = b["close"]
                exit_reason = "STALE_SCALP_TIMEOUT"
                hold_sec = age
                break
        gross = (exit_px - entry_px) / entry_px
        net = gross - rt
        out[f"{stale}s"] = {
            "exit_reason": exit_reason,
            "hold_sec": round(hold_sec, 1),
            "exit_price": exit_px,
            "net_pct": round(net * 100, 4),
            "pnl_usd_25": round(25.0 * net, 4),
            "hit_target": hit_target,
            "win": net >= econ.net_profit_target_pct,
        }
    return out


def main() -> int:
    econ = apply_profile(ScalpEconomics.from_env(), "moderate")
    db = REPO / "mystic_trading.db"
    report: list[dict] = []

    with sqlite3.connect(db) as conn:
        for tid in TRADE_IDS:
            buy = conn.execute(
                "SELECT symbol, price, created_at, diagnostics_json FROM scalp_paper_trades WHERE trade_id=?",
                (tid,),
            ).fetchone()
            sell = conn.execute(
                "SELECT price, pnl_usd, exit_reason, created_at, entry_price FROM scalp_paper_trades WHERE trade_id=?",
                (tid + "_SELL",),
            ).fetchone()
            pos = conn.execute(
                "SELECT diagnostics_json FROM scalp_paper_positions WHERE trade_id=?",
                (tid,),
            ).fetchone()
            if not buy or not sell:
                continue

            sym, entry_px, buy_ts, buy_diag_raw = buy
            exit_px, pnl, exit_reason, sell_ts, _ = sell
            diag = json.loads(buy_diag_raw) if buy_diag_raw else {}
            pos_diag = json.loads(pos[0]) if pos and pos[0] else {}
            pf = diag.get("preflight") or {}
            reach = pf.get("reachability") or {}
            rank_row = (pos_diag.get("symbol_ranking") or {}).get("ranking", [{}])
            rank = rank_row[0] if rank_row else {}

            entry_ms = parse_ts(buy_ts)
            exit_ms = parse_ts(sell_ts)
            bars = fetch_klines(sym, entry_ms - 120_000, exit_ms + 360_000)

            window = [b for b in bars if entry_ms <= b["ts_ms"] <= exit_ms + 60_000]
            max_hi = max((b["high"] for b in window), default=entry_px)
            min_lo = min((b["low"] for b in window), default=entry_px)
            max_fav = (max_hi - entry_px) / entry_px
            max_adv = (entry_px - min_lo) / entry_px
            spread = float(pf.get("spread_pct") or 0.0)
            rt = econ.roundtrip_cost_pct(spread, 0.0, 0.0)
            target_px = entry_px * (1.0 + econ.net_profit_target_pct + rt)
            hit_target_during = max_hi >= target_px

            stale_why = None
            if exit_reason == "STALE_SCALP_TIMEOUT":
                stale_why = f"held {(exit_ms - entry_ms) / 1000:.0f}s without net target; max_fav={max_fav * 100:.3f}% max_adv={max_adv * 100:.3f}% target_px={target_px:.6f} max_hi={max_hi:.6f}"

            report.append(
                {
                    "trade_id": tid,
                    "symbol": sym,
                    "entry_time": buy_ts,
                    "exit_time": sell_ts,
                    "hold_sec": round((exit_ms - entry_ms) / 1000.0, 1),
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "exit_reason": exit_reason,
                    "pnl_usd": pnl,
                    "entry_spread_pct": round(spread * 100, 4),
                    "entry_impact_pct": round(float(pf.get("buy_impact_pct") or 0) * 100, 4),
                    "projected_gross_pct": round(float(reach.get("projected_gross_move_pct") or 0) * 100, 4),
                    "required_gross_pct": round(float(reach.get("required_gross_move_pct") or 0) * 100, 4),
                    "surplus_pct": round(float(reach.get("projected_surplus_pct") or 0) * 100, 4),
                    "ranking_score": rank.get("score"),
                    "momentum": {
                        "confirmed": reach.get("momentum_confirmed"),
                        "mid_change_15s": reach.get("mid_change_15s"),
                        "mid_change_30s": reach.get("mid_change_30s"),
                        "mid_change_60s": reach.get("mid_change_60s"),
                        "breakout_strength_pct": reach.get("breakout_strength_pct"),
                    },
                    "breakout": {
                        "confirmed": reach.get("breakout_confirmed"),
                        "recent_range_pct": round(float(reach.get("recent_range_pct") or 0) * 100, 4),
                    },
                    "max_favorable_move_pct": round(max_fav * 100, 4),
                    "max_adverse_move_pct": round(max_adv * 100, 4),
                    "hit_target_during_hold": hit_target_during,
                    "stale_exit_reason": stale_why,
                    "stale_timeout_what_if": simulate_stale_outcomes(bars, entry_ms, float(entry_px), spread, econ),
                    "xrp_cap_would_block": {
                        "0.10%": spread > 0.001,
                        "0.09%": spread > 0.0009,
                        "0.08%": spread > 0.0008,
                        "0.07%": spread > 0.0007,
                    }
                    if sym == "XRPUSDT"
                    else None,
                }
            )

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
