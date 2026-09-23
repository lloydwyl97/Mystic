#!/usr/bin/env python3
"""DAY V2 Structural Entry Replay.

Compares OLD policy (generic 14bps/4bps/15min dip-rebound) against
DAY_STRUCTURAL_PULLBACK_V1 (structural zone + 5m confirmation required)
using historical feature_ohlcv bars from the production database.

Usage:
    python3 scripts/day_v2_structural_entry_replay.py [--db PATH]

Outputs:
    - Data window stats (symbol, bar count, date range)
    - OLD policy results: calibration + validation
    - NEW policy results: calibration + validation
    - Deployment condition pass/fail
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.day_v2.config import DAY_STRUCTURAL_PULLBACK_V1, DAY_V2_UNIVERSE
from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit
from backend.services.day_v2.live_signal import evaluate_entry_signal
from backend.services.day_v2.structural_entry import evaluate_structural_zone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLD_POLICY_MIN_DIP_BPS = 14.0
OLD_POLICY_REBOUND_BPS = 4.0
OLD_POLICY_TIMEOUT_SEC = 900.0  # 15 minutes

STRUCTURAL_POLICY_LIFETIME_SEC = 3600.0  # 60 minutes
ROUNDTRIP_COST_FRACTION = 0.0006  # 3bps maker x 2 = 6bps total

MIN_BARS_FOR_VALID_REPLAY = 2000  # per symbol; fewer = inconclusive

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_ohlcv(db_path: str, symbol: str, interval: str) -> list[dict]:
    """Load bars oldest-first for a symbol/interval from feature_ohlcv.

    Tries both BTCUSDT and BTC-USDT symbol formats.
    """
    variants = [symbol, symbol.replace("USDT", "-USDT"), symbol.replace("USDT", "/USDT")]
    with sqlite3.connect(db_path, timeout=30) as conn:
        for sym in variants:
            rows = conn.execute(
                "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE symbol=? AND interval=? ORDER BY ts ASC",
                (sym, interval),
            ).fetchall()
            if rows:
                return [{"ts": r[0], "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])} for r in rows]
    return []


def _load_1m_bars_window(db_path: str, symbol: str, since_ts: float, until_ts: float) -> list[dict]:
    """Load 1m bars in a time window for 5m bar synthesis."""
    variants = [symbol, symbol.replace("USDT", "-USDT"), symbol.replace("USDT", "/USDT")]
    with sqlite3.connect(db_path, timeout=30) as conn:
        for sym in variants:
            rows = conn.execute(
                "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE symbol=? AND interval='1m' AND ts>=? AND ts<? ORDER BY ts ASC",
                (sym, int(since_ts), int(until_ts)),
            ).fetchall()
            if rows:
                return [{"ts": r[0], "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])} for r in rows]
    return []


# ---------------------------------------------------------------------------
# Synthetic 5m bar helpers (identical logic to five_min_confirm.py)
# ---------------------------------------------------------------------------


def _five_min_boundary(ts: float) -> int:
    return int(ts) - (int(ts) % 300)


def _build_synthetic_5m_from_1m(bars_1m: list[dict], boundary_ts: int) -> dict | None:
    expected = {boundary_ts + i * 60 for i in range(5)}
    present = {int(b["ts"]): b for b in bars_1m if int(b["ts"]) in expected}
    if len(present) != 5:
        return None
    ordered = [present[boundary_ts + i * 60] for i in range(5)]
    if any(float(b["open"]) <= 0 or float(b["close"]) <= 0 for b in ordered):
        return None
    return {
        "ts": boundary_ts,
        "open": float(ordered[0]["open"]),
        "high": max(float(b["high"]) for b in ordered),
        "low": min(float(b["low"]) for b in ordered),
        "close": float(ordered[-1]["close"]),
        "volume": sum(float(b["volume"]) for b in ordered),
    }


def _has_5m_confirmation(db_path: str, symbol: str, since_ts: float, until_ts: float, reclaim_level: float) -> bool:
    """Check if a red-then-reclaim pattern appears in [since_ts, until_ts)."""
    bars_1m = _load_1m_bars_window(db_path, symbol, since_ts, until_ts)
    # Build synthetic 5m bars
    if not bars_1m:
        return False
    start = _five_min_boundary(since_ts)
    end = _five_min_boundary(until_ts - 300)
    synthetic: list[dict] = []
    t = start
    while t <= end:
        bar = _build_synthetic_5m_from_1m(bars_1m, t)
        if bar:
            synthetic.append(bar)
        t += 300
    # Find red-then-reclaim sequence
    red_bar = None
    for bar in synthetic:
        if red_bar is None:
            if bar["close"] < bar["open"]:
                red_bar = bar
        else:
            if bar["close"] > red_bar["high"] or (reclaim_level > 0 and bar["close"] > reclaim_level):
                return True
            if bar["close"] < bar["open"]:
                red_bar = bar
    return False


# ---------------------------------------------------------------------------
# Policy simulations
# ---------------------------------------------------------------------------


def _simulate_old_policy(
    bars_15m: list[dict],
    bars_1h: list[dict],
    bars_4h: list[dict],
    symbol: str,
) -> list[dict]:
    """Simulate the OLD 14bps/4bps/15min policy. Returns list of trade dicts."""
    trades: list[dict] = []
    seen_opps: set[str] = set()

    for idx in range(32, len(bars_15m) - 1):
        signal = evaluate_entry_signal(symbol, bars_15m[: idx + 1], bars_1h, bars_4h)
        if signal is None:
            continue
        if signal.opportunity_id in seen_opps:
            continue  # same opportunity — no deduplication in old policy (shows recycling)

        arm_ts = float(bars_15m[idx]["ts"])
        arm_price = float(bars_15m[idx]["close"])

        # Simulate dip + rebound within 15-minute window
        dip_target = arm_price * (1.0 - OLD_POLICY_MIN_DIP_BPS / 10000.0)
        expire_ts = arm_ts + OLD_POLICY_TIMEOUT_SEC
        lowest = arm_price
        fill_price = None

        for future_idx in range(idx + 1, len(bars_15m)):
            bar = bars_15m[future_idx]
            bar_ts = float(bar["ts"])
            if bar_ts > expire_ts:
                break
            low = float(bar["low"])
            close = float(bar["close"])

            if low <= dip_target:
                lowest = min(lowest, low)
                rebound_target = lowest * (1.0 + OLD_POLICY_REBOUND_BPS / 10000.0)
                if close >= rebound_target:
                    fill_price = rebound_target
                    break

        if fill_price is None:
            continue

        seen_opps.add(signal.opportunity_id)

        # Simulate exit using the DAY V2 exit evaluator
        entry_ts = arm_ts
        entry_price = fill_price
        highest_price = entry_price
        exit_price = None
        exit_reason = "UNKNOWN"
        exit_ts = None

        for future_idx in range(idx + 1, len(bars_15m)):
            bar = bars_15m[future_idx]
            hold_min = (float(bar["ts"]) - entry_ts) / 60.0
            current_price = float(bar["close"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            highest_price = max(highest_price, bar_high)
            decision = evaluate_day_v2_exit(
                engine_id="DAY_V2",
                entry_price=entry_price,
                current_price=current_price,
                bar_low=bar_low,
                highest_price=highest_price,
                atr_at_entry=float(signal.atr),
                structural_anchor=float(signal.structural_anchor),
                target_price=float(signal.target_price),
                entry_time=entry_ts,
                estimated_roundtrip_cost=ROUNDTRIP_COST_FRACTION,
            )
            if decision:
                exit_price = float(decision.get("exit_price_estimate") or current_price)
                exit_reason = str(decision.get("reason") or "EXIT")
                exit_ts = float(bar["ts"])
                break

        if exit_price is None:
            exit_price = float(bars_15m[-1]["close"])
            exit_reason = "END_OF_DATA"
            exit_ts = float(bars_15m[-1]["ts"])

        pnl_pct = (exit_price - entry_price) / entry_price
        fee_pct = ROUNDTRIP_COST_FRACTION
        net_pct = pnl_pct - fee_pct
        hold_min = ((exit_ts or entry_ts) - entry_ts) / 60.0

        trades.append(
            {
                "symbol": symbol,
                "setup": signal.setup,
                "opportunity_id": signal.opportunity_id,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_min": hold_min,
                "pnl_pct": pnl_pct,
                "fee_pct": fee_pct,
                "net_pct": net_pct,
                "notional": 50.0,  # approximate
            }
        )

    return trades


def _simulate_new_policy(
    bars_15m: list[dict],
    bars_1h: list[dict],
    bars_4h: list[dict],
    symbol: str,
    db_path: str,
    max_fills_24h_sym: int = 2,
    max_fills_24h_total: int = 8,
) -> list[dict]:
    """Simulate DAY_STRUCTURAL_PULLBACK_V1 policy."""
    trades: list[dict] = []
    consumed_opps: set[str] = set()
    # Per-symbol and total fill counts in rolling 24h windows
    fill_times_sym: list[float] = []
    fill_times_total: list[float] = []

    for idx in range(32, len(bars_15m) - 1):
        signal = evaluate_entry_signal(symbol, bars_15m[: idx + 1], bars_1h, bars_4h)
        if signal is None:
            continue

        opp_id = signal.opportunity_id
        if opp_id in consumed_opps:
            continue  # permanently consumed

        arm_ts = float(bars_15m[idx]["ts"])

        # Frequency guard (rolling 24h in simulation time)
        cutoff = arm_ts - 86400.0
        active_sym = [t for t in fill_times_sym if t >= cutoff]
        active_total = [t for t in fill_times_total if t >= cutoff]
        if len(active_sym) >= max_fills_24h_sym:
            continue  # DAY_ENTRY_FREQUENCY_LIMIT per symbol
        if len(active_total) >= max_fills_24h_total:
            continue  # DAY_ENTRY_FREQUENCY_LIMIT total

        # Structural zone
        zone = evaluate_structural_zone(signal)
        if not zone.valid:
            continue  # MISSING_STRUCTURAL_ENTRY_LEVEL

        expire_ts = arm_ts + STRUCTURAL_POLICY_LIFETIME_SEC

        # Wait for pullback into structural zone + 5m confirmation
        fill_price = None
        fill_ts = None

        for future_idx in range(idx + 1, len(bars_15m)):
            bar = bars_15m[future_idx]
            bar_ts = float(bar["ts"])
            if bar_ts > expire_ts:
                break
            low = float(bar["low"])
            close = float(bar["close"])

            # Check if price entered the structural zone
            if low <= zone.zone_high and close >= zone.zone_low:
                # Check for 5m confirmation in the window [arm_ts, bar_ts + 900s)
                confirmed = _has_5m_confirmation(
                    db_path,
                    symbol,
                    since_ts=arm_ts,
                    until_ts=bar_ts + 900.0,
                    reclaim_level=float(zone.reclaim_level),
                )
                if confirmed:
                    fill_price = close
                    fill_ts = bar_ts
                    break

        if fill_price is None:
            continue

        consumed_opps.add(opp_id)
        fill_times_sym.append(fill_ts)
        fill_times_total.append(fill_ts)

        # Simulate exit
        entry_ts = fill_ts or arm_ts
        entry_price = fill_price
        highest_price = entry_price
        exit_price = None
        exit_reason = "UNKNOWN"
        exit_ts = None

        for future_idx in range(idx + 1, len(bars_15m)):
            bar = bars_15m[future_idx]
            current_price = float(bar["close"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            highest_price = max(highest_price, bar_high)
            decision = evaluate_day_v2_exit(
                engine_id="DAY_V2",
                entry_price=entry_price,
                current_price=current_price,
                bar_low=bar_low,
                highest_price=highest_price,
                atr_at_entry=float(signal.atr),
                structural_anchor=float(signal.structural_anchor),
                target_price=float(signal.target_price),
                entry_time=entry_ts,
                estimated_roundtrip_cost=ROUNDTRIP_COST_FRACTION,
            )
            if decision:
                exit_price = float(decision.get("exit_price_estimate") or current_price)
                exit_reason = str(decision.get("reason") or "EXIT")
                exit_ts = float(bar["ts"])
                break

        if exit_price is None:
            exit_price = float(bars_15m[-1]["close"])
            exit_reason = "END_OF_DATA"
            exit_ts = float(bars_15m[-1]["ts"])

        pnl_pct = (exit_price - entry_price) / entry_price
        fee_pct = ROUNDTRIP_COST_FRACTION
        net_pct = pnl_pct - fee_pct
        hold_min = ((exit_ts or entry_ts) - entry_ts) / 60.0

        trades.append(
            {
                "symbol": symbol,
                "setup": signal.setup,
                "opportunity_id": opp_id,
                "entry_ts": entry_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_min": hold_min,
                "pnl_pct": pnl_pct,
                "fee_pct": fee_pct,
                "net_pct": net_pct,
                "notional": 50.0,
            }
        )

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(trades: list[dict], window_days: float) -> dict:
    if not trades:
        return {
            "count": 0,
            "per_day": 0.0,
            "win_rate": 0.0,
            "gross_pnl_pct": 0.0,
            "total_fee_pct": 0.0,
            "net_pnl_pct": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_min": 0.0,
            "median_hold_min": 0.0,
            "duplicate_opp_fills": 0,
        }
    n = len(trades)
    wins = [t["net_pct"] for t in trades if t["net_pct"] > 0]
    losses = [t["net_pct"] for t in trades if t["net_pct"] <= 0]
    net_pcts = [t["net_pct"] for t in trades]
    gross = sum(t["pnl_pct"] for t in trades)
    fees = sum(t["fee_pct"] for t in trades)
    net = sum(net_pcts)

    # Max drawdown (equity curve)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_ts"]):
        equity += t["net_pct"] * 100.0
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    holds = sorted([t["hold_min"] for t in trades])
    median_hold = holds[len(holds) // 2] if holds else 0.0

    # Duplicate opportunity fills
    opp_counts: dict[str, int] = {}
    for t in trades:
        opp_counts[t["opportunity_id"]] = opp_counts.get(t["opportunity_id"], 0) + 1
    duplicates = sum(1 for c in opp_counts.values() if c > 1)

    pf = (sum(wins) / (-sum(losses))) if losses and sum(losses) != 0 else float("inf")

    return {
        "count": n,
        "per_day": n / max(window_days, 1),
        "win_rate": len(wins) / n if n else 0.0,
        "gross_pnl_pct": gross * 100,
        "total_fee_pct": fees * 100,
        "net_pnl_pct": net * 100,
        "avg_winner": (sum(wins) / len(wins) * 100) if wins else 0.0,
        "avg_loser": (sum(losses) / len(losses) * 100) if losses else 0.0,
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_dd, 2),
        "avg_hold_min": sum(holds) / len(holds) if holds else 0.0,
        "median_hold_min": median_hold,
        "duplicate_opp_fills": duplicates,
    }


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n  {'─' * 50}")
    print(f"  {label}")
    print(f"  {'─' * 50}")
    print(f"  Entries:           {m['count']} ({m['per_day']:.1f}/day)")
    print(f"  Win rate:          {m['win_rate'] * 100:.1f}%")
    print(f"  Gross PnL:         {m['gross_pnl_pct']:+.2f}%")
    print(f"  Total fees:        {m['total_fee_pct']:.2f}%")
    print(f"  Net PnL:           {m['net_pnl_pct']:+.2f}%")
    print(f"  Avg winner:        {m['avg_winner']:+.3f}%")
    print(f"  Avg loser:         {m['avg_loser']:+.3f}%")
    print(f"  Profit factor:     {m['profit_factor']}")
    print(f"  Max drawdown:      {m['max_drawdown']:.2f}%")
    print(f"  Avg hold:          {m['avg_hold_min']:.0f} min")
    print(f"  Median hold:       {m['median_hold_min']:.0f} min")
    print(f"  Duplicate opp fills: {m['duplicate_opp_fills']} (must be 0 for NEW policy)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="mystic_trading.db")
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        # Try repo root
        repo_db = Path(__file__).parent.parent / "mystic_trading.db"
        if repo_db.exists():
            db_path = str(repo_db)
        else:
            print(f"ERROR: DB not found at {args.db}. Pass --db PATH.")
            sys.exit(1)

    print("=" * 60)
    print("DAY V2 STRUCTURAL ENTRY REPLAY")
    print(f"DB: {db_path}")
    print(f"Policy: {DAY_STRUCTURAL_PULLBACK_V1}")
    print("=" * 60)

    all_old_trades: list[dict] = []
    all_new_trades: list[dict] = []
    symbol_stats: list[dict] = []

    for symbol in DAY_V2_UNIVERSE:
        print(f"\n[{symbol}] Loading bars...")
        bars_15m = _load_ohlcv(db_path, symbol, "15m")
        bars_1h = _load_ohlcv(db_path, symbol, "1h")
        bars_4h = _load_ohlcv(db_path, symbol, "4h")

        n15 = len(bars_15m)
        if n15 < MIN_BARS_FOR_VALID_REPLAY:
            print(f"  WARNING: Only {n15} 15m bars — insufficient for meaningful comparison (need {MIN_BARS_FOR_VALID_REPLAY})")
            continue

        ts_start = bars_15m[0]["ts"]
        ts_end = bars_15m[-1]["ts"]
        window_days = (ts_end - ts_start) / 86400.0
        split_idx = int(n15 * 0.8)

        print(f"  15m bars: {n15} | Window: {window_days:.1f} days")
        print(f"  Calibration: bars [0..{split_idx}] | Validation: [{split_idx}..{n15}]")

        # Split bars for calibration (first 80%) and validation (last 20%)
        cal_15m = bars_15m[:split_idx]
        val_15m = bars_15m[split_idx:]
        cal_days = (cal_15m[-1]["ts"] - cal_15m[0]["ts"]) / 86400.0 if len(cal_15m) > 1 else 1
        val_days = (val_15m[-1]["ts"] - val_15m[0]["ts"]) / 86400.0 if len(val_15m) > 1 else 1

        print("\n  Simulating OLD policy (calibration)...")
        old_cal = _simulate_old_policy(cal_15m, bars_1h, bars_4h, symbol)
        print("  Simulating OLD policy (validation)...")
        old_val = _simulate_old_policy(val_15m, bars_1h, bars_4h, symbol)
        print("  Simulating NEW policy (calibration)...")
        new_cal = _simulate_new_policy(cal_15m, bars_1h, bars_4h, symbol, db_path)
        print("  Simulating NEW policy (validation)...")
        new_val = _simulate_new_policy(val_15m, bars_1h, bars_4h, symbol, db_path)

        print(f"\n[{symbol}] — {window_days:.1f} days | {n15} 15m bars")
        _print_metrics("OLD policy — CALIBRATION", _metrics(old_cal, cal_days))
        _print_metrics("OLD policy — VALIDATION", _metrics(old_val, val_days))
        _print_metrics("NEW policy — CALIBRATION", _metrics(new_cal, cal_days))
        _print_metrics("NEW policy — VALIDATION", _metrics(new_val, val_days))

        symbol_stats.append(
            {
                "symbol": symbol,
                "bars": n15,
                "days": window_days,
                "old_cal": _metrics(old_cal, cal_days),
                "old_val": _metrics(old_val, val_days),
                "new_cal": _metrics(new_cal, cal_days),
                "new_val": _metrics(new_val, val_days),
            }
        )
        all_old_trades += old_cal + old_val
        all_new_trades += new_cal + new_val

    # ---------------------------------------------------------------------------
    # Aggregate and deployment conditions
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    if not symbol_stats:
        print("ERROR: No symbols had sufficient bar data for replay.")
        sys.exit(1)

    total_days = max(s["days"] for s in symbol_stats)
    old_m = _metrics(all_old_trades, total_days)
    new_m = _metrics(all_new_trades, total_days)
    _print_metrics("OLD POLICY — ALL SYMBOLS COMBINED", old_m)
    _print_metrics("NEW POLICY — ALL SYMBOLS COMBINED", new_m)

    print("\n" + "=" * 60)
    print("DEPLOYMENT CONDITIONS")
    print("=" * 60)

    conditions: list[tuple[str, bool, str]] = []

    def _cond(name: str, passed: bool, detail: str) -> None:
        conditions.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    _cond(
        "Entry frequency substantially reduced",
        new_m["per_day"] < old_m["per_day"],
        f"OLD {old_m['per_day']:.1f}/day vs NEW {new_m['per_day']:.1f}/day",
    )
    _cond(
        "Zero duplicate fingerprint fills (NEW)",
        new_m["duplicate_opp_fills"] == 0,
        f"{new_m['duplicate_opp_fills']} duplicates",
    )
    _cond(
        "Lower total fees (NEW vs OLD)",
        new_m["total_fee_pct"] <= old_m["total_fee_pct"],
        f"OLD {old_m['total_fee_pct']:.2f}% vs NEW {new_m['total_fee_pct']:.2f}%",
    )
    _cond(
        "NEW drawdown not worse than OLD",
        new_m["max_drawdown"] <= old_m["max_drawdown"] * 1.2,  # 20% tolerance
        f"OLD {old_m['max_drawdown']:.2f}% vs NEW {new_m['max_drawdown']:.2f}%",
    )
    _cond(
        "Sufficient bar coverage",
        len(symbol_stats) >= 3,
        f"{len(symbol_stats)}/{len(DAY_V2_UNIVERSE)} symbols had sufficient data",
    )

    all_pass = all(p for _, p, _ in conditions)
    print(f"\n{'DEPLOYMENT: AUTHORIZED' if all_pass else 'DEPLOYMENT: BLOCKED'}")
    if not all_pass:
        failed = [n for n, p, _ in conditions if not p]
        print(f"Failing conditions: {failed}")
        sys.exit(1)

    print("\nAll deployment conditions passed.")
    print("Note: Replay validates structure. Live results depend on market conditions.")


if __name__ == "__main__":
    main()
