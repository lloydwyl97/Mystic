"""SCALP V2 True Event-Driven Replay.

Uses the 169+ FILLED trailing buy intents as entry signals (REAL entries that
actually happened on the live/paper system). For each entry:
1. Load actual buy price from paper_trades (fill price, not arm midpoint)
2. Load subsequent 15m candles from feature_ohlcv (up to 300 minutes)
3. Apply LEGACY and SCALP V2 exit rules to determine simulated exit
4. Calculate P&L after 0.06% round-trip cost (ESTIMATED_ROUNDTRIP_COST)

Walk-forward split:
  Calibration:    first 50% of entries (by chronological order)
  Validation:     next 25%
  Untouched final: last 25% (NEVER used for optimization)

Qualification criteria (UNTOUCHED FINAL PERIOD only):
  - Net expectancy per trade > 0 after 0.06% round-trip cost
  - Profit factor > 1.0

WARNING: with ~42 untouched trades, the standard error on win rate is
approximately 7-8pp. A positive result is NOT statistically certain.

Runnable from Ocean:
    ssh mystic-prod "cd /home/mystic/mystic && \
        sudo -u mystic venv/bin/python3 scripts/research/scalp_v2_true_replay.py"
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.getenv(
    "MYSTIC_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "mystic_trading.db"),
)
DB_PATH = os.path.abspath(DB_PATH)

ROUND_TRIP_COST_PCT = 0.0006  # 0.06% = 6 bps

# Candle simulation window: maximum 300 minutes (20 x 15m bars)
MAX_HOLD_BARS = 20  # 20 x 15m = 300 minutes

# Coin profile: 0.25% trail (existing production profile)
TRAIL_PCT_DEFAULT = 0.0025  # 0.25%
TRAIL_PCT_WIDE = 0.0050  # 0.50% — SCALP_V2_WIDER_TRAIL variant

# LEGACY: net-profit floor (must clear round-trip cost)
LEGACY_MIN_NET_PROFIT_PCT = 0.004  # 0.4%

# LEGACY: giveback exit parameters (from production DAY_ env vars)
LEGACY_GIVEBACK_MIN_MFE_PCT = 0.0015  # DAY_GIVEBACK_MIN_MFE_PCT=0.0015
LEGACY_GIVEBACK_TRIGGER_PCT = -0.0015  # DAY_GIVEBACK_TRIGGER=-0.15% net

# LEGACY: stall exit parameters
LEGACY_STALL_MIN_HOLD_MIN = 120  # DAY_STALL_MIN_HOLD_MIN=120
LEGACY_STALL_MIN_ADVERSE_PCT = 0.003  # DAY_STALL_MIN_ADVERSE_PCT=0.003


# ---------------------------------------------------------------------------
# Exit reason constants
# ---------------------------------------------------------------------------

EXIT_STOP_LOSS = "STOP_LOSS_EXIT"
EXIT_NET_PROFIT = "NET_PROFIT_EXIT"
EXIT_TRAILING = "TRAILING_STOP_EXIT"
EXIT_GIVEBACK = "GIVEBACK_EXIT"
EXIT_STALL = "STALL_EXIT"
EXIT_TIME = "TIME_STOP"
EXIT_OPEN = "STILL_OPEN"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """A single entry from a FILLED trailing buy intent."""

    intent_id: str
    symbol: str
    arm_ts: float  # Unix epoch when intent was armed
    buy_price: float  # Actual fill price from paper_trades
    stop_price: float  # Stop price at entry
    tp1_price: float  # Take-profit-1 price at entry
    buy_ts_str: str  # ISO timestamp of buy
    buy_ts_epoch: float  # Epoch of buy fill


@dataclass
class SimResult:
    """Result of simulating one entry with one parameter set."""

    intent_id: str
    symbol: str
    buy_price: float
    exit_price: float
    exit_reason: str
    gross_pnl_pct: float  # Before round-trip cost
    net_pnl_pct: float  # After 0.06% round-trip cost
    hold_bars: int
    hold_minutes: float
    mfe_pct: float  # Max favorable excursion (gross)
    mae_pct: float  # Max adverse excursion (gross, negative)


@dataclass
class PeriodStats:
    """Aggregate statistics for a period/parameter set combination."""

    name: str
    period: str
    n: int = 0
    wins: int = 0
    total_net_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    avg_winner_net: float = 0.0
    avg_loser_net: float = 0.0
    win_results: list[float] = field(default_factory=list)
    loss_results: list[float] = field(default_factory=list)
    exit_reasons: dict[str, int] = field(default_factory=dict)

    def record(self, r: SimResult) -> None:
        self.n += 1
        self.total_net_pnl += r.net_pnl_pct
        self.total_gross_pnl += r.gross_pnl_pct
        self.exit_reasons[r.exit_reason] = self.exit_reasons.get(r.exit_reason, 0) + 1
        if r.net_pnl_pct > 0:
            self.wins += 1
            self.win_results.append(r.net_pnl_pct)
        else:
            self.loss_results.append(r.net_pnl_pct)

    def finalize(self) -> None:
        if self.win_results:
            self.avg_winner_net = sum(self.win_results) / len(self.win_results)
        if self.loss_results:
            self.avg_loser_net = sum(self.loss_results) / len(self.loss_results)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n > 0 else 0.0

    @property
    def expectancy_per_trade(self) -> float:
        return self.total_net_pnl / self.n if self.n > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(self.win_results)
        gross_losses = abs(sum(self.loss_results))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def payoff_ratio(self) -> float:
        if not self.loss_results or not self.win_results:
            return 0.0
        return abs(self.avg_winner_net) / abs(self.avg_loser_net)


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------


def load_entries(db_path: str) -> list[Entry]:
    """Load all FILLED trailing buy intents with matching BUY paper_trades."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT
            ti.intent_id,
            ti.symbol,
            ti.arm_ts,
            pt.price       AS buy_price,
            pt.stop_price  AS stop_price,
            pt.take_profit_price AS tp1_price,
            pt.timestamp   AS buy_ts_str
        FROM day_trailing_buy_intents ti
        JOIN paper_trades pt
            ON pt.trade_id = ti.trade_id AND pt.side = 'BUY'
        WHERE ti.status = 'FILLED'
        ORDER BY ti.arm_ts
        """
    ).fetchall()
    conn.close()

    entries = []
    for r in rows:
        intent_id, symbol, arm_ts, buy_price, stop_price, tp1_price, buy_ts_str = r
        if not buy_price or buy_price <= 0:
            continue
        # Parse epoch from ISO string
        try:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(buy_ts_str.replace("Z", "+00:00"))
            buy_ts_epoch = dt.timestamp()
        except Exception:
            buy_ts_epoch = float(arm_ts or 0)

        # Normalize symbol: BTC/USDT -> BTC-USDT for feature_ohlcv
        entries.append(
            Entry(
                intent_id=str(intent_id),
                symbol=str(symbol),
                arm_ts=float(arm_ts or 0),
                buy_price=float(buy_price),
                stop_price=float(stop_price or 0),
                tp1_price=float(tp1_price or 0),
                buy_ts_str=str(buy_ts_str),
                buy_ts_epoch=float(buy_ts_epoch),
            )
        )
    return entries


def _ohlcv_symbol(symbol: str) -> str:
    """Convert paper_trades symbol (BTC/USDT) to feature_ohlcv symbol (BTC-USDT)."""
    return symbol.replace("/", "-")


def load_candles_after(db_path: str, symbol: str, after_epoch: float, limit: int = MAX_HOLD_BARS + 2) -> list[dict[str, Any]]:
    """Load 15m candles for symbol after the given epoch timestamp.

    feature_ohlcv.ts is stored as a datetime string (e.g. '2026-09-21 03:45:00.000000').
    We must compare it as a string, not as a float, to avoid SQLite type-affinity
    issues (TEXT > REAL always in SQLite, so a float parameter would match every row).
    """
    from datetime import datetime
    from datetime import timezone as _tz

    ohlcv_sym = _ohlcv_symbol(symbol)
    # Convert epoch -> UTC datetime string for proper string comparison in SQLite
    after_dt_str = datetime.fromtimestamp(after_epoch, tz=_tz.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT open, high, low, close, ts
        FROM feature_ohlcv
        WHERE symbol = ? AND interval = '15m' AND ts > ?
        ORDER BY ts ASC
        LIMIT ?
        """,
        (ohlcv_sym, after_dt_str, limit),
    ).fetchall()
    conn.close()

    candles = []
    for r in rows:
        try:
            from datetime import datetime, timezone

            ts_str = str(r["ts"])
            # feature_ohlcv ts is stored as a datetime string (no timezone = UTC)
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts_epoch = dt.timestamp()
            except Exception:
                ts_epoch = 0.0
            candles.append(
                {
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "ts_str": ts_str,
                    "ts_epoch": ts_epoch,
                }
            )
        except Exception:
            continue
    return candles


# ---------------------------------------------------------------------------
# Simulation logic
# ---------------------------------------------------------------------------


def simulate_entry(
    entry: Entry,
    candles: list[dict[str, Any]],
    *,
    trail_pct: float,
    giveback_enabled: bool,
    stall_enabled: bool,
    min_net_profit_pct: float = LEGACY_MIN_NET_PROFIT_PCT,
    cost_pct: float = ROUND_TRIP_COST_PCT,
) -> SimResult:
    """Simulate exits for one entry using the given parameter set.

    Returns SimResult with gross_pnl_pct and net_pnl_pct after cost_pct deduction.
    """
    buy_price = entry.buy_price
    stop_price = entry.stop_price if entry.stop_price > 0 else buy_price * (1 - 0.01)
    tp1_price = entry.tp1_price if entry.tp1_price > 0 else buy_price * (1 + 0.014)

    trail_high = buy_price  # Trail starts at entry (not activated until cost-aware MFE)
    trail_activated = False
    trail_activation_mfe = 0.004  # 0.4% MFE to activate trail (matches existing logic)
    trail_stop = 0.0

    highest_price = buy_price
    lowest_price = buy_price
    bar_idx = 0

    for bar_idx, c in enumerate(candles[:MAX_HOLD_BARS]):
        bar_high = float(c["high"])
        bar_low = float(c["low"])
        bar_close = float(c["close"])
        hold_min = (bar_idx + 1) * 15.0

        # Update high water and low
        highest_price = max(highest_price, bar_high)
        lowest_price = min(lowest_price, bar_low)

        gross_pnl_pct = (bar_close - buy_price) / buy_price
        mfe_pct = (highest_price - buy_price) / buy_price
        mae_pct = (lowest_price - buy_price) / buy_price

        # --- STOP LOSS (highest priority) ---
        if stop_price > 0 and bar_low <= stop_price:
            exit_price = stop_price
            gross = (exit_price - buy_price) / buy_price
            return SimResult(
                intent_id=entry.intent_id,
                symbol=entry.symbol,
                buy_price=buy_price,
                exit_price=exit_price,
                exit_reason=EXIT_STOP_LOSS,
                gross_pnl_pct=gross,
                net_pnl_pct=gross - cost_pct,
                hold_bars=bar_idx + 1,
                hold_minutes=hold_min,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
            )

        # --- NET PROFIT EXIT ---
        net_gross = (bar_close - buy_price) / buy_price
        if bar_close >= tp1_price and net_gross >= min_net_profit_pct:
            exit_price = bar_close
            gross = net_gross
            return SimResult(
                intent_id=entry.intent_id,
                symbol=entry.symbol,
                buy_price=buy_price,
                exit_price=exit_price,
                exit_reason=EXIT_NET_PROFIT,
                gross_pnl_pct=gross,
                net_pnl_pct=gross - cost_pct,
                hold_bars=bar_idx + 1,
                hold_minutes=hold_min,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
            )

        # --- TRAILING STOP ---
        # Activate trail once MFE clears the net-profit floor (cost-aware)
        mfe_gross = (highest_price - buy_price) / buy_price
        if mfe_gross >= trail_activation_mfe and not trail_activated:
            trail_activated = True
            trail_high = highest_price
            trail_stop = trail_high * (1 - trail_pct)

        if trail_activated:
            if bar_high > trail_high:
                trail_high = bar_high
                trail_stop = trail_high * (1 - trail_pct)
            if bar_close < trail_stop:
                exit_price = bar_close
                gross = (exit_price - buy_price) / buy_price
                return SimResult(
                    intent_id=entry.intent_id,
                    symbol=entry.symbol,
                    buy_price=buy_price,
                    exit_price=exit_price,
                    exit_reason=EXIT_TRAILING,
                    gross_pnl_pct=gross,
                    net_pnl_pct=gross - cost_pct,
                    hold_bars=bar_idx + 1,
                    hold_minutes=hold_min,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                )

        # --- GIVEBACK EXIT (LEGACY only when enabled) ---
        if giveback_enabled:
            if mfe_gross >= LEGACY_GIVEBACK_MIN_MFE_PCT and gross_pnl_pct <= LEGACY_GIVEBACK_TRIGGER_PCT:
                exit_price = bar_close
                gross = gross_pnl_pct
                return SimResult(
                    intent_id=entry.intent_id,
                    symbol=entry.symbol,
                    buy_price=buy_price,
                    exit_price=exit_price,
                    exit_reason=EXIT_GIVEBACK,
                    gross_pnl_pct=gross,
                    net_pnl_pct=gross - cost_pct,
                    hold_bars=bar_idx + 1,
                    hold_minutes=hold_min,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                )

        # --- STALL EXIT (LEGACY only when enabled) ---
        if stall_enabled and hold_min >= LEGACY_STALL_MIN_HOLD_MIN:
            if gross_pnl_pct < 0 and abs(gross_pnl_pct) >= LEGACY_STALL_MIN_ADVERSE_PCT:
                exit_price = bar_close
                gross = gross_pnl_pct
                return SimResult(
                    intent_id=entry.intent_id,
                    symbol=entry.symbol,
                    buy_price=buy_price,
                    exit_price=exit_price,
                    exit_reason=EXIT_STALL,
                    gross_pnl_pct=gross,
                    net_pnl_pct=gross - cost_pct,
                    hold_bars=bar_idx + 1,
                    hold_minutes=hold_min,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                )

    # --- TIME STOP / STILL OPEN ---
    if candles:
        exit_price = float(candles[min(bar_idx, len(candles) - 1)]["close"])
    else:
        exit_price = buy_price

    hold_min = (min(bar_idx + 1, len(candles))) * 15.0 if candles else 0.0
    gross = (exit_price - buy_price) / buy_price
    mfe_pct = (highest_price - buy_price) / buy_price
    mae_pct = (lowest_price - buy_price) / buy_price
    reason = EXIT_TIME if len(candles) >= MAX_HOLD_BARS else EXIT_OPEN

    return SimResult(
        intent_id=entry.intent_id,
        symbol=entry.symbol,
        buy_price=buy_price,
        exit_price=exit_price,
        exit_reason=reason,
        gross_pnl_pct=gross,
        net_pnl_pct=gross - cost_pct,
        hold_bars=len(candles),
        hold_minutes=hold_min,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
    )


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

PARAM_SETS = [
    {
        "name": "LEGACY",
        "trail_pct": TRAIL_PCT_DEFAULT,
        "giveback_enabled": True,
        "stall_enabled": True,
    },
    {
        "name": "SCALP_V2_BASE",
        "trail_pct": TRAIL_PCT_DEFAULT,
        "giveback_enabled": False,
        "stall_enabled": False,
    },
    {
        "name": "SCALP_V2_WIDER_TRAIL",
        "trail_pct": TRAIL_PCT_WIDE,
        "giveback_enabled": False,
        "stall_enabled": False,
    },
]


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------


def run_replay(db_path: str) -> None:
    print(f"\n{'=' * 70}")
    print("SCALP V2 TRUE EVENT-DRIVEN REPLAY")
    print(f"DB: {db_path}")
    print(f"Round-trip cost: {ROUND_TRIP_COST_PCT * 100:.3f}%")
    print(f"{'=' * 70}\n")

    # Load entries
    entries = load_entries(db_path)
    print(f"Loaded {len(entries)} FILLED trailing buy intents with BUY trades")

    if not entries:
        print("ERROR: No entries found. Check DB path and day_trailing_buy_intents table.")
        return

    # Walk-forward split
    n = len(entries)
    cal_end = n // 2
    val_end = cal_end + n // 4
    # final period is everything after val_end

    cal_entries = entries[:cal_end]
    val_entries = entries[cal_end:val_end]
    final_entries = entries[val_end:]

    print(f"Split: calibration={len(cal_entries)}, validation={len(val_entries)}, untouched_final={len(final_entries)}")
    print(f"Calibration period:  {entries[0].buy_ts_str[:10]} to {entries[cal_end - 1].buy_ts_str[:10] if cal_entries else 'N/A'}")
    print(f"Validation period:   {entries[cal_end].buy_ts_str[:10] if val_entries else 'N/A'} to {entries[val_end - 1].buy_ts_str[:10] if val_entries else 'N/A'}")
    print(f"Untouched final:     {entries[val_end].buy_ts_str[:10] if final_entries else 'N/A'} to {entries[-1].buy_ts_str[:10]}")
    print()

    periods = [
        ("CALIBRATION", cal_entries),
        ("VALIDATION", val_entries),
        ("UNTOUCHED_FINAL", final_entries),
    ]

    all_results: dict[str, dict[str, PeriodStats]] = {}

    for period_name, period_entries in periods:
        if not period_entries:
            continue
        all_results[period_name] = {}

        for params in PARAM_SETS:
            ps_name = params["name"]
            stats = PeriodStats(name=ps_name, period=period_name)

            for entry in period_entries:
                candles = load_candles_after(db_path, entry.symbol, entry.buy_ts_epoch)
                result = simulate_entry(
                    entry,
                    candles,
                    trail_pct=params["trail_pct"],
                    giveback_enabled=params["giveback_enabled"],
                    stall_enabled=params["stall_enabled"],
                )
                stats.record(result)

            stats.finalize()
            all_results[period_name][ps_name] = stats

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------

    for period_name, period_entries in periods:
        if not period_entries or period_name not in all_results:
            continue

        print(f"\n{'=' * 70}")
        print(f"  PERIOD: {period_name}  (n={len(period_entries)})")
        print(f"{'=' * 70}")
        print(f"{'Metric':<30} {'LEGACY':>18} {'SCALP_V2_BASE':>18} {'SCALP_V2_WIDER':>18}")
        print("-" * 84)

        stats_by_name = all_results[period_name]
        leg = stats_by_name.get("LEGACY")
        base = stats_by_name.get("SCALP_V2_BASE")
        wide = stats_by_name.get("SCALP_V2_WIDER_TRAIL")

        def fmt(v: float | None, pct: bool = False, decimals: int = 4) -> str:
            if v is None:
                return "N/A".rjust(18)
            if pct:
                return f"{v * 100:.3f}%".rjust(18)
            return f"{v:.{decimals}f}".rjust(18)

        def fmti(v: int | None) -> str:
            if v is None:
                return "N/A".rjust(18)
            return f"{v:,}".rjust(18)

        rows = [
            ("Trade count", [fmti(s.n) for s in [leg, base, wide]]),
            ("Win rate", [fmt(s.win_rate, pct=True) for s in [leg, base, wide]]),
            ("Avg winner (net)", [fmt(s.avg_winner_net, pct=True) for s in [leg, base, wide]]),
            ("Avg loser (net)", [fmt(s.avg_loser_net, pct=True) for s in [leg, base, wide]]),
            ("Payoff ratio", [fmt(s.payoff_ratio, decimals=3) for s in [leg, base, wide]]),
            ("Gross expect/trade", [fmt(s.total_gross_pnl / s.n if s.n else 0, pct=True) for s in [leg, base, wide]]),
            ("Net expect/trade", [fmt(s.expectancy_per_trade, pct=True) for s in [leg, base, wide]]),
            ("Profit factor", [fmt(s.profit_factor, decimals=3) for s in [leg, base, wide]]),
            ("Total net P&L", [fmt(s.total_net_pnl, pct=True) for s in [leg, base, wide]]),
        ]

        for label, vals in rows:
            print(f"  {label:<28} {vals[0]} {vals[1]} {vals[2]}")

        # Exit reason breakdown
        print()
        for ps_name, s in stats_by_name.items():
            if s and s.n > 0:
                print(f"  {ps_name} exit reasons:")
                for reason, cnt in sorted(s.exit_reasons.items(), key=lambda x: -x[1]):
                    print(f"    {reason:<35} {cnt:>3} ({cnt / s.n * 100:.1f}%)")
        print()

    # ---------------------------------------------------------------------------
    # Qualification verdict (UNTOUCHED FINAL only)
    # ---------------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("  QUALIFICATION VERDICT (UNTOUCHED FINAL PERIOD)")
    print(f"{'=' * 70}")

    final_n = len(final_entries)
    import math

    se_win_rate = math.sqrt(0.5 * 0.5 / final_n) if final_n > 0 else 0.0

    print(f"\n  Sample size: {final_n} trades")
    print(f"  Standard error on win rate: approx {se_win_rate * 100:.1f}pp")
    print()
    print("  NOTE: Results are based on a small sample. A true positive result is")
    print("  NOT statistically certain even if the point estimate is positive.")
    print()

    if "UNTOUCHED_FINAL" not in all_results:
        print("  ERROR: No untouched final period results.")
        return

    final_stats = all_results["UNTOUCHED_FINAL"]
    for ps_name, s in final_stats.items():
        if not s or s.n == 0:
            continue
        qualifies = s.expectancy_per_trade > 0 and s.profit_factor > 1.0
        verdict = "✓ QUALIFIES" if qualifies else "✗ FAILS"
        print(f"  {ps_name}: {verdict}")
        print(f"    Net expectancy/trade = {s.expectancy_per_trade * 100:.4f}%  (threshold: > 0)")
        print(f"    Profit factor = {s.profit_factor:.4f}  (threshold: > 1.0)")
        if qualifies:
            print("    → This engine meets the qualification bar for deployment evaluation.")
        else:
            failure = []
            if s.expectancy_per_trade <= 0:
                failure.append(f"net expectancy {s.expectancy_per_trade * 100:.4f}% ≤ 0")
            if s.profit_factor <= 1.0:
                failure.append(f"profit factor {s.profit_factor:.4f} ≤ 1.0")
            print(f"    → Failure reason(s): {'; '.join(failure)}")
        print()

    # Final answers to the three key questions
    print(f"{'=' * 70}")
    print("  SPECIFIC QUESTIONS")
    print(f"{'=' * 70}")

    base_final = final_stats.get("SCALP_V2_BASE")
    wide_final = final_stats.get("SCALP_V2_WIDER_TRAIL")
    leg_final = final_stats.get("LEGACY")

    if base_final and leg_final:
        q1 = base_final.expectancy_per_trade > 0 and base_final.profit_factor > 1.0
        print("\n  Q1: Does disabling giveback make the system net positive after 0.06% cost")
        print("      on the UNTOUCHED FINAL period?")
        print(f"      SCALP_V2_BASE net expect/trade: {base_final.expectancy_per_trade * 100:.4f}%")
        print(f"      Answer: {'YES' if q1 else 'NO'}")

    if base_final and wide_final:
        print("\n  Q2: Does disabling stall additionally improve the untouched final period?")
        print("      (Both SCALP_V2 variants already disable stall)")
        print("      Already disabled in both SCALP_V2 variants.")

    if wide_final and base_final:
        q3_better = wide_final.expectancy_per_trade > base_final.expectancy_per_trade
        print("\n  Q3: Does widening the trail help or hurt (on untouched final)?")
        print(f"      SCALP_V2_BASE net expect: {base_final.expectancy_per_trade * 100:.4f}%")
        print(f"      SCALP_V2_WIDER net expect: {wide_final.expectancy_per_trade * 100:.4f}%")
        print(f"      Answer: {'HELPS' if q3_better else 'HURTS'}")

    print(f"\n{'=' * 70}")
    print("  DEPLOYMENT DECISION")
    print(f"{'=' * 70}")

    if leg_final and leg_final.expectancy_per_trade > 0 and leg_final.profit_factor > 1.0:
        print("\n  LEGACY qualifies → existing system already qualifies; no new deployment needed.")
    else:
        # Check if any SCALP_V2 qualifies
        qualifying = [ps_name for ps_name, s in final_stats.items() if s and s.n > 0 and s.expectancy_per_trade > 0 and s.profit_factor > 1.0 and ps_name != "LEGACY"]
        if qualifying:
            print(f"\n  Qualifying engine(s): {', '.join(qualifying)}")
            print("  → Proceed to deployment per PART 9 protocol.")
        else:
            print("\n  No engine qualifies on the untouched final period.")
            print("  → Do NOT deploy. Existing system continues running unchanged.")
            print()
            for ps_name, s in final_stats.items():
                if s and s.n > 0:
                    print(f"     {ps_name}: net_expect={s.expectancy_per_trade * 100:.4f}% PF={s.profit_factor:.3f} — does not meet both criteria")

    print()
    print("Replay complete.")


if __name__ == "__main__":
    run_replay(DB_PATH)
