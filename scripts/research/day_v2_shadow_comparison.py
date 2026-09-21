"""DAY V2 Shadow Comparison — structural simulation using existing paper trades.

Run from repo root:
    python3 scripts/research/day_v2_shadow_comparison.py

This script is READ-ONLY. No writes to database or files.

IMPORTANT DISCLAIMERS:
  - This comparison is NOT a backtest of DAY V2 — it is a structural simulation
    using the same trades with different filters.
  - Sample size limitations are stated explicitly.
  - The DAY V2 column represents a minimum-intervention structural change,
    not a calibrated strategy.
  - Profitability claims are made ONLY if the data clearly supports them.
"""

import datetime
import math
import os
import sqlite3
import sys
from collections import defaultdict
from statistics import mean, median

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mystic_trading.db")


def connect_readonly(path: str) -> sqlite3.Connection:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"DB not found at path: {abs_path}")
        sys.exit(0)
    uri = f"file:{abs_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{100.0 * num / den:.1f}%"


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    print()
    print(f"  --- {title} ---")


def _parse_ts(ts_raw) -> float | None:
    if ts_raw is None:
        return None
    try:
        return datetime.datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def get_pnl(t: dict) -> float:
    if t.get("pnl_usd_net") is not None:
        return float(t["pnl_usd_net"])
    if t.get("pnl") is not None:
        return float(t["pnl"])
    return 0.0


def get_fees(t: dict) -> float:
    if t.get("entry_fee_usd") is not None and t.get("exit_fee_usd") is not None:
        return float(t["entry_fee_usd"]) + float(t["exit_fee_usd"])
    if t.get("fees_paid") is not None:
        return float(t["fees_paid"])
    return 0.0


def compute_cohort_stats(trades: list[dict], label: str, day_range: float) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "label": label,
            "n": 0,
            "rtpd": 0.0,
            "avg_hold_min": float("nan"),
            "win_rate": 0.0,
            "avg_winner": float("nan"),
            "avg_loser": float("nan"),
            "payoff": float("nan"),
            "gross_exp": float("nan"),
            "total_fees": 0.0,
            "net_exp": float("nan"),
            "profit_factor": float("nan"),
            "total_pnl": 0.0,
        }

    pnls = [get_pnl(t) for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    fees = sum(get_fees(t) for t in trades)

    gross_wins = sum(winners) if winners else 0.0
    gross_losses = abs(sum(losers)) if losers else 0.0

    hold_secs = [float(t["hold_time_seconds"]) for t in trades if t.get("hold_time_seconds") is not None and float(t["hold_time_seconds"]) > 0]

    return {
        "label": label,
        "n": n,
        "rtpd": n / max(day_range, 1),
        "avg_hold_min": mean(hold_secs) / 60 if hold_secs else float("nan"),
        "win_rate": len(winners) / n if n else 0.0,
        "avg_winner": mean(winners) if winners else float("nan"),
        "avg_loser": mean(losers) if losers else float("nan"),
        "payoff": abs(mean(winners) / mean(losers)) if (winners and losers) else float("nan"),
        "gross_exp": mean(pnls) if pnls else float("nan"),
        "total_fees": fees,
        "net_exp": (sum(pnls) - fees) / n if n else float("nan"),
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else float("inf"),
        "total_pnl": sum(pnls),
    }


def print_cohort_row(stats: dict) -> None:
    label = stats["label"]
    n = stats["n"]
    rtpd = stats["rtpd"]
    avg_hold = stats["avg_hold_min"]
    wr = stats["win_rate"]
    aw = stats["avg_winner"]
    al = stats["avg_loser"]
    pr = stats["payoff"]
    ge = stats["gross_exp"]
    fe = stats["total_fees"]
    ne = stats["net_exp"]
    pf = stats["profit_factor"]
    tot = stats["total_pnl"]

    def fmt(v) -> str:
        if isinstance(v, float) and math.isnan(v):
            return "    N/A"
        if v == float("inf"):
            return "    inf"
        return f"${v:>6.2f}"

    print(f"\n  [{label}]")
    print(f"    Trades:          {n}")
    print(f"    RT/day:          {rtpd:.2f}")
    if not (isinstance(avg_hold, float) and math.isnan(avg_hold)):
        print(f"    Avg hold (min):  {avg_hold:.1f}")
    else:
        print("    Avg hold:       N/A")
    print(f"    Win rate:        {wr * 100:.1f}%")
    print(f"    Avg winner:      {fmt(aw)}")
    print(f"    Avg loser:       {fmt(al)}")
    if not (isinstance(pr, float) and (math.isnan(pr) or math.isinf(pr))):
        print(f"    Payoff ratio:    {pr:.2f}")
    else:
        print("    Payoff ratio:    N/A")
    print(f"    Gross exp/trade: {fmt(ge)}")
    print(f"    Total fees est:  ${fe:.2f}")
    print(f"    Net exp/trade:   {fmt(ne)}")
    if not (isinstance(pf, float) and (math.isnan(pf) or math.isinf(pf))):
        print(f"    Profit factor:   {pf:.2f}")
    else:
        print("    Profit factor:   N/A")
    print(f"    Total gross PnL: ${tot:.2f}")


def main() -> None:
    con = connect_readonly(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        SELECT
            symbol, pnl, pnl_pct, pnl_usd_net, pnl_pct_net,
            hold_time_seconds, fees_paid, entry_fee_usd, exit_fee_usd,
            exit_reason, entry_price, atr_at_entry, entry_timestamp,
            timestamp, trade_id, strategy_id, regime
        FROM paper_trades
        WHERE side = 'SELL'
        ORDER BY entry_timestamp ASC
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    trades = [dict(zip(cols, r, strict=False)) for r in rows]

    cur.execute("""
        SELECT MIN(entry_timestamp), MAX(timestamp)
        FROM paper_trades WHERE side = 'SELL'
    """)
    date_range = cur.fetchone()
    con.close()

    if not trades:
        print("No SELL trades found.")
        return

    # Compute day range
    ts_min = _parse_ts(date_range[0]) if date_range[0] else None
    ts_max = _parse_ts(date_range[1]) if date_range[1] else None
    day_range = (ts_max - ts_min) / 86400 if (ts_min and ts_max) else 1.0

    print()
    print("=" * 70)
    print("  DAY V2 SHADOW COMPARISON REPORT")
    print("=" * 70)
    print("\n  DISCLAIMER:")
    print("  This comparison is NOT a backtest of DAY V2.")
    print("  It is a structural simulation using the same trades with different filters.")
    print(f"\n  Sample: {len(trades)} trades over {day_range:.1f} days")
    if date_range[0] and date_range[1]:
        print(f"  Date range: {date_range[0]} to {date_range[1]}")
    print("\n  DAY V2 column = minimum-intervention structural change, not a calibrated strategy.")
    print("  Profitability claims stated only if data clearly supports them.")

    # -----------------------------------------------------------------------
    # COHORT 1: LEGACY_DAY_LIVE — all trades as-is
    # -----------------------------------------------------------------------
    legacy_cohort = trades
    legacy_stats = compute_cohort_stats(legacy_cohort, "LEGACY_DAY_LIVE", day_range)

    # -----------------------------------------------------------------------
    # COHORT 2: SCALP_V2_CANDIDATE — re-label by hold time / exit characteristics
    # -----------------------------------------------------------------------
    # Show which trades look "scalp-like": hold < 30m or short GIVEBACK/STALL/TRAILING
    short_exits = {"GIVEBACK_EXIT", "STALL_EXIT", "TRAILING_STOP_EXIT"}
    scalp_like = [t for t in trades if ((t.get("hold_time_seconds") is not None and float(t["hold_time_seconds"]) < 1800) or (t.get("exit_reason") or "") in short_exits)]
    scalp_stats = compute_cohort_stats(scalp_like, "SCALP_V2_CANDIDATE (scalp-like trades)", day_range)

    # -----------------------------------------------------------------------
    # COHORT 3: DAY_V2_SHADOW — structural simulation
    # One trade per symbol per 2h window (first-entry-per-cluster)
    # Exclude trailing stops where hold < 2h (would have been held longer in DAY V2)
    # -----------------------------------------------------------------------
    CLUSTER_WINDOW_SECS = 2 * 3600
    MIN_HOLD_FOR_TRAILING_SECS = 2 * 3600

    symbols = sorted({t["symbol"] for t in trades})

    # Group into clusters per symbol
    day_v2_trades: list[dict] = []
    trailing_excluded = 0

    for sym in symbols:
        sym_trades = sorted(
            [t for t in trades if t["symbol"] == sym],
            key=lambda t: _parse_ts(t.get("entry_timestamp")) or 0,
        )
        if not sym_trades:
            continue

        clusters: list[list[dict]] = []
        current_cluster = [sym_trades[0]]
        cluster_start_ts = _parse_ts(sym_trades[0].get("entry_timestamp")) or 0

        for t in sym_trades[1:]:
            ts = _parse_ts(t.get("entry_timestamp"))
            if ts is None:
                continue
            if ts - cluster_start_ts <= CLUSTER_WINDOW_SECS:
                current_cluster.append(t)
            else:
                clusters.append(current_cluster)
                current_cluster = [t]
                cluster_start_ts = ts
        clusters.append(current_cluster)

        for cluster in clusters:
            # Take only the first trade per cluster
            first = cluster[0]
            # Exclude trailing stops where hold < 2h (DAY V2 would not have exited these)
            er = first.get("exit_reason") or ""
            hold = float(first.get("hold_time_seconds") or 0)
            if er == "TRAILING_STOP_EXIT" and hold < MIN_HOLD_FOR_TRAILING_SECS:
                trailing_excluded += 1
                continue
            day_v2_trades.append(first)

    day_v2_stats = compute_cohort_stats(day_v2_trades, "DAY_V2_SHADOW (first-per-cluster, trail-filter)", day_range)

    # -----------------------------------------------------------------------
    # Print comparison
    # -----------------------------------------------------------------------
    section("COHORT COMPARISON — OVERALL")
    print_cohort_row(legacy_stats)
    print_cohort_row(scalp_stats)
    print_cohort_row(day_v2_stats)

    print(f"\n  Note: DAY V2 cohort excluded {trailing_excluded} short trailing-stop exits")
    print("  (trades with TRAILING_STOP_EXIT and hold < 2h that DAY V2 lifecycle would not have exited)")

    # -----------------------------------------------------------------------
    # Short-hold characterization for SCALP_V2_CANDIDATE
    # -----------------------------------------------------------------------
    section("SCALP-LIKE CHARACTERIZATION (< 30m hold or short exit)")
    short_hold = [t for t in trades if t.get("hold_time_seconds") is not None and float(t["hold_time_seconds"]) < 1800]
    print(f"  Trades < 30m hold: {len(short_hold)} / {len(trades)} ({100.0 * len(short_hold) / len(trades):.1f}%)")
    short_exit_trades = [t for t in trades if (t.get("exit_reason") or "") in short_exits]
    print(f"  GIVEBACK/STALL/TRAILING exits: {len(short_exit_trades)} / {len(trades)} ({100.0 * len(short_exit_trades) / len(trades):.1f}%)")
    for er in sorted(short_exits):
        grp = [t for t in trades if (t.get("exit_reason") or "") == er]
        if grp:
            avg_hold = mean(float(t["hold_time_seconds"]) for t in grp if t.get("hold_time_seconds") is not None) / 60
            pnls = [get_pnl(t) for t in grp]
            print(f"    {er}: n={len(grp)}, avg_hold={avg_hold:.1f}m, avg_pnl=${mean(pnls):.2f}, win_rate={pct(sum(1 for p in pnls if p > 0), len(grp))}")

    # -----------------------------------------------------------------------
    # Per-symbol breakdown
    # -----------------------------------------------------------------------
    section("PER-SYMBOL BREAKDOWN")
    header = f"  {'Symbol':<10} {'Col':<40} {'N':>5} {'Win%':>6} {'AvgPnL':>8} {'TotPnL':>9}"
    print(header)
    print(f"  {'-' * 10} {'-' * 40} {'-' * 5} {'-' * 6} {'-' * 8} {'-' * 9}")
    for sym in symbols:
        for cohort_label, cohort in [
            ("LEGACY", [t for t in legacy_cohort if t["symbol"] == sym]),
            ("SCALP_V2", [t for t in scalp_like if t["symbol"] == sym]),
            ("DAY_V2", [t for t in day_v2_trades if t["symbol"] == sym]),
        ]:
            n = len(cohort)
            if n == 0:
                print(f"  {sym:<10} {cohort_label:<40} {0:>5}")
                continue
            pnls = [get_pnl(t) for t in cohort]
            wins = sum(1 for p in pnls if p > 0)
            print(f"  {sym:<10} {cohort_label:<40} {n:>5} {pct(wins, n):>6} ${mean(pnls):>6.2f} ${sum(pnls):>7.2f}")
        print()

    # -----------------------------------------------------------------------
    # Uncertainty statement
    # -----------------------------------------------------------------------
    section("UNCERTAINTY AND LIMITATIONS")
    net_available = sum(1 for t in trades if t.get("pnl_usd_net") is not None)
    print(f"  - True net PnL available for {net_available}/{len(trades)} trades. Remaining use gross pnl column (pre-fee).")
    print(f"  - DAY V2 cohort size: {len(day_v2_trades)} trades. At ~{len(day_v2_trades) / max(day_range, 1):.1f} RT/day over {day_range:.0f} days,")
    wr_day_v2 = day_v2_stats.get("win_rate", 0)
    n_day_v2 = day_v2_stats.get("n", 0)
    if n_day_v2 >= 10:
        # Very rough SE estimate: se_winrate = sqrt(p*(1-p)/n)
        import math

        se = math.sqrt(wr_day_v2 * (1 - wr_day_v2) / n_day_v2)
        print(f"    win rate SE ≈ {se * 100:.1f}pp. A true 3pp edge needs ~{int(9 * wr_day_v2 * (1 - wr_day_v2) / 0.0009)} trades.")
    print("  - The DAY V2 filter does NOT optimize parameters — it only applies structural rules from the spec.")
    print("  - No DAY V2 cohort result claims profitability from this simulation alone.")

    print()
    print("=" * 70)
    print("  END OF SHADOW COMPARISON REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()
