"""DAY V2 Calibration — read-only analysis of paper trade history.

Run from repo root:
    python3 scripts/research/day_v2_calibration.py

This script is READ-ONLY. It never writes to any database or file.
All metrics are computed from paper_trades in mystic_trading.db.

For trades where pnl_usd_net is null, pnl column is used and marked (gross/pre-fee).
For unavailable metrics, output clearly states "UNAVAILABLE: reason".
"""

import os
import sqlite3
import sys
from collections import defaultdict
from statistics import mean, median, stdev

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mystic_trading.db")


def connect_readonly(path: str) -> sqlite3.Connection:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"DB not found at path: {abs_path}")
        sys.exit(0)
    uri = f"file:{abs_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{100.0 * numerator / denominator:.1f}%"


def fmt_float(v, suffix="") -> str:
    if v is None:
        return "None"
    return f"{v:.4f}{suffix}"


def percentiles(data: list[float], ps: list[int]) -> dict[int, float]:
    if not data:
        return {p: float("nan") for p in ps}
    s = sorted(data)
    n = len(s)
    result = {}
    for p in ps:
        idx = (p / 100) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        result[p] = s[lo] * (1 - frac) + s[hi] * frac
    return result


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    print()
    print(f"  --- {title} ---")


def main() -> None:
    con = connect_readonly(DB_PATH)
    cur = con.cursor()

    # Fetch all SELL trades
    cur.execute("""
        SELECT
            symbol,
            pnl,
            pnl_pct,
            pnl_usd_net,
            pnl_pct_net,
            hold_time_seconds,
            fees_paid,
            entry_fee_usd,
            exit_fee_usd,
            exit_reason,
            entry_price,
            stop_price,
            take_profit_price,
            atr_at_entry,
            regime,
            strategy_id,
            entry_timestamp,
            timestamp,
            trade_id,
            sleeve
        FROM paper_trades
        WHERE side = 'SELL'
        ORDER BY entry_timestamp ASC
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    trades = [dict(zip(cols, r, strict=False)) for r in rows]

    if not trades:
        print("No SELL trades found in paper_trades.")
        con.close()
        return

    # Resolve pnl: prefer pnl_usd_net where available
    def get_pnl(t: dict) -> tuple[float, bool]:
        """Return (pnl_value, is_net)."""
        if t.get("pnl_usd_net") is not None:
            return float(t["pnl_usd_net"]), True
        if t.get("pnl") is not None:
            return float(t["pnl"]), False
        return 0.0, False

    def get_fees(t: dict) -> float:
        if t.get("entry_fee_usd") is not None and t.get("exit_fee_usd") is not None:
            return float(t["entry_fee_usd"]) + float(t["exit_fee_usd"])
        if t.get("fees_paid") is not None:
            return float(t["fees_paid"])
        return 0.0

    total = len(trades)
    net_count = sum(1 for t in trades if t.get("pnl_usd_net") is not None)
    gross_only = total - net_count

    print()
    print("=" * 70)
    print("  DAY V2 CALIBRATION REPORT")
    print("=" * 70)
    print(f"  Total SELL trades: {total}")
    print(f"  With pnl_usd_net (true net): {net_count}")
    print(f"  Gross-only (pnl column, pre-fee approximation): {gross_only}")

    # -----------------------------------------------------------------------
    section("1. PER-SYMBOL METRICS")
    # -----------------------------------------------------------------------
    symbols = sorted({t["symbol"] for t in trades})
    for sym in symbols:
        st = [t for t in trades if t["symbol"] == sym]
        n = len(st)
        hold_secs = [float(t["hold_time_seconds"]) for t in st if t.get("hold_time_seconds") is not None]
        pnl_vals = [get_pnl(t) for t in st]
        winners = [p for p, _ in pnl_vals if p > 0]
        losers = [p for p, _ in pnl_vals if p <= 0]
        gross_pnl = sum(p for p, _ in pnl_vals)
        fee_sum = sum(get_fees(t) for t in st)
        net_pnl = sum(p for p, is_net in pnl_vals if is_net)
        gross_approx = sum(p for p, is_net in pnl_vals if not is_net)

        subsection(sym)
        print(f"    Trades: {n}")
        if hold_secs:
            print(f"    Hold time: avg={mean(hold_secs) / 60:.1f}m  median={median(hold_secs) / 60:.1f}m")
        else:
            print("    Hold time: UNAVAILABLE: all null")
        print(f"    Win rate: {pct(len(winners), n)}")
        if winners:
            print(f"    Avg winner PnL: ${mean(winners):.2f}")
        else:
            print("    Avg winner PnL: N/A (no winners)")
        if losers:
            print(f"    Avg loser PnL:  ${mean(losers):.2f}")
        else:
            print("    Avg loser PnL:  N/A (no losers)")
        if winners and losers:
            print(f"    Payoff ratio:   {abs(mean(winners) / mean(losers)):.2f}")
        else:
            print("    Payoff ratio:   N/A")
        print(f"    Gross PnL:      ${gross_pnl:.2f}")
        print(f"    Fee estimate:   ${fee_sum:.2f}")
        if net_count > 0:
            print(f"    Net PnL (true): ${net_pnl:.2f}  ({net_count} net trades)")
        if gross_only > 0:
            print(f"    Gross approx PnL: ${gross_approx:.2f} ({gross_only} trades, pre-fee)")

    # -----------------------------------------------------------------------
    section("2. EXIT REASON BREAKDOWN")
    # -----------------------------------------------------------------------
    exit_groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        er = t.get("exit_reason") or "UNKNOWN"
        exit_groups[er].append(t)

    print(f"  {'EXIT_REASON':<40} {'N':>5} {'WIN%':>6} {'AVG_PNL':>9} {'AVG_HOLD_M':>11}")
    print(f"  {'-' * 40} {'-' * 5} {'-' * 6} {'-' * 9} {'-' * 11}")
    for er in sorted(exit_groups, key=lambda x: -len(exit_groups[x])):
        grp = exit_groups[er]
        n = len(grp)
        pnl_vals = [get_pnl(t)[0] for t in grp]
        wins = sum(1 for p in pnl_vals if p > 0)
        avg_pnl = mean(pnl_vals) if pnl_vals else 0.0
        hold_secs = [float(t["hold_time_seconds"]) for t in grp if t.get("hold_time_seconds") is not None]
        avg_hold = mean(hold_secs) / 60 if hold_secs else float("nan")
        print(f"  {er:<40} {n:>5} {pct(wins, n):>6} ${avg_pnl:>7.2f} {avg_hold:>10.1f}m")

    # -----------------------------------------------------------------------
    section("3. RECOVERY ANALYSIS")
    # -----------------------------------------------------------------------
    # For each STALL_EXIT / GIVEBACK_EXIT trade, look for a subsequent BUY
    # in the same symbol within 4 hours
    cur.execute("""
        SELECT symbol, entry_timestamp, timestamp, pnl, pnl_usd_net, side, trade_id
        FROM paper_trades
        ORDER BY entry_timestamp ASC
    """)
    all_trades_rows = cur.fetchall()
    all_trades_cols = [d[0] for d in cur.description]
    all_trades = [dict(zip(all_trades_cols, r, strict=False)) for r in all_trades_rows]

    # Build buy list by symbol
    buys_by_sym: dict[str, list[dict]] = defaultdict(list)
    for t in all_trades:
        if t["side"] == "BUY":
            buys_by_sym[t["symbol"]].append(t)

    RECOVERY_WINDOW_SECS = 4 * 3600

    def recovery_analysis(exit_reason_filter: str) -> None:
        exits = [t for t in trades if (t.get("exit_reason") or "") == exit_reason_filter]
        if not exits:
            print(f"  {exit_reason_filter}: no trades found")
            return

        followed_by_profitable = 0
        followed_by_any = 0
        no_timestamp = 0

        for exit_trade in exits:
            exit_ts_raw = exit_trade.get("timestamp")
            if not exit_ts_raw:
                no_timestamp += 1
                continue
            try:
                import datetime

                exit_ts = datetime.datetime.fromisoformat(str(exit_ts_raw).replace("Z", "+00:00")).timestamp()
            except Exception:
                no_timestamp += 1
                continue

            sym = exit_trade["symbol"]
            subsequent_buys = [
                b
                for b in buys_by_sym.get(sym, [])
                if b.get("entry_timestamp") and _parse_ts(b["entry_timestamp"]) is not None and 0 < _parse_ts(b["entry_timestamp"]) - exit_ts <= RECOVERY_WINDOW_SECS
            ]
            if subsequent_buys:
                followed_by_any += 1
                # Check if subsequent sell was profitable
                for buy in subsequent_buys:
                    # find matching sell
                    matching_sells = [t for t in all_trades if t["side"] == "SELL" and t["symbol"] == sym and t.get("entry_timestamp") == buy.get("entry_timestamp")]
                    if matching_sells:
                        pnl_v = matching_sells[0].get("pnl_usd_net") or matching_sells[0].get("pnl") or 0
                        if float(pnl_v) > 0:
                            followed_by_profitable += 1
                            break

        print(
            f"  {exit_reason_filter}:"
            f"\n    Total: {len(exits)}"
            f"\n    Followed by any re-entry within 4h: {followed_by_any} / {len(exits)}"
            f"\n    Followed by profitable re-entry within 4h: {followed_by_profitable} / {len(exits)}"
        )
        if no_timestamp > 0:
            print(f"    (skipped {no_timestamp} trades with no timestamp)")

    def _parse_ts(ts_raw) -> float | None:
        if ts_raw is None:
            return None
        import datetime

        try:
            return datetime.datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    subsection("Stall Exit Recovery")
    recovery_analysis("STALL_EXIT")
    subsection("Giveback Exit Recovery")
    recovery_analysis("GIVEBACK_EXIT")
    subsection("Trailing Stop Recovery")
    print("  TRAILING_STOP_EXIT recovery: UNAVAILABLE — cannot answer without post-exit candle data")

    # -----------------------------------------------------------------------
    section("4. REPEATED ENTRY / OPPORTUNITY CLUSTERING")
    # -----------------------------------------------------------------------
    # Group BUY entries in the same symbol within 2-hour windows
    CLUSTER_WINDOW_SECS = 2 * 3600

    cluster_counts: list[int] = []
    total_clustered_first_only_pnl: list[float] = []
    total_all_pnl: list[float] = []

    for sym in symbols:
        sym_buys = sorted(
            [t for t in all_trades if t["side"] == "BUY" and t["symbol"] == sym],
            key=lambda t: _parse_ts(t.get("entry_timestamp")) or 0,
        )
        sym_sells = {t.get("entry_timestamp"): t for t in trades if t["symbol"] == sym}

        clusters: list[list[dict]] = []
        if not sym_buys:
            continue

        current_cluster = [sym_buys[0]]
        cluster_start_ts = _parse_ts(sym_buys[0].get("entry_timestamp")) or 0

        for buy in sym_buys[1:]:
            ts = _parse_ts(buy.get("entry_timestamp"))
            if ts is None:
                continue
            if ts - cluster_start_ts <= CLUSTER_WINDOW_SECS:
                current_cluster.append(buy)
            else:
                clusters.append(current_cluster)
                current_cluster = [buy]
                cluster_start_ts = ts
        clusters.append(current_cluster)

        for cluster in clusters:
            cluster_counts.append(len(cluster))
            # First trade only simulation
            first_buy = cluster[0]
            sell_match = sym_sells.get(first_buy.get("entry_timestamp"))
            if sell_match:
                pnl_v = get_pnl(sell_match)[0]
                total_clustered_first_only_pnl.append(pnl_v)
            # All trades pnl
            for buy in cluster:
                sell_match = sym_sells.get(buy.get("entry_timestamp"))
                if sell_match:
                    total_all_pnl.append(get_pnl(sell_match)[0])

    multi_entry_clusters = [c for c in cluster_counts if c > 1]
    print(f"  Total 2h clusters: {len(cluster_counts)}")
    print(f"  Single-entry clusters: {len(cluster_counts) - len(multi_entry_clusters)}")
    print(f"  Multi-entry clusters (>1 entry in 2h): {len(multi_entry_clusters)}")
    if cluster_counts:
        print(f"  Avg entries per cluster: {mean(cluster_counts):.2f}")
        print(f"  Max entries per cluster: {max(cluster_counts)}")
    if total_clustered_first_only_pnl:
        print(
            f"\n  Simulation: first-trade-only-per-cluster"
            f"\n    Total trades (first-only): {len(total_clustered_first_only_pnl)}"
            f"\n    Net PnL (first-only):      ${sum(total_clustered_first_only_pnl):.2f}"
            f"\n    Win rate (first-only):     "
            f"{pct(sum(1 for p in total_clustered_first_only_pnl if p > 0), len(total_clustered_first_only_pnl))}"
        )
    if total_all_pnl:
        print(f"\n  All-trades net PnL (for comparison): ${sum(total_all_pnl):.2f}")

    # -----------------------------------------------------------------------
    section("5. CALIBRATION METRICS")
    # -----------------------------------------------------------------------

    subsection("ATR at Entry")
    atr_vals = [float(t["atr_at_entry"]) for t in trades if t.get("atr_at_entry") is not None and float(t["atr_at_entry"]) > 0]
    if atr_vals:
        for sym in symbols:
            sym_atr = [float(t["atr_at_entry"]) for t in trades if t["symbol"] == sym and t.get("atr_at_entry") is not None and float(t["atr_at_entry"]) > 0]
            if sym_atr:
                print(f"    {sym}: avg ATR%={mean(sym_atr) * 100:.4f}%  n={len(sym_atr)}")
    else:
        print("    UNAVAILABLE: atr_at_entry null for all records")

    subsection("Hold Time Distribution (all symbols)")
    hold_all = [float(t["hold_time_seconds"]) for t in trades if t.get("hold_time_seconds") is not None and float(t["hold_time_seconds"]) > 0]
    if hold_all:
        ps = percentiles(hold_all, [25, 50, 75, 95])
        print(f"    p25={ps[25] / 60:.1f}m  p50={ps[50] / 60:.1f}m  p75={ps[75] / 60:.1f}m  p95={ps[95] / 60:.1f}m")
    else:
        print("    UNAVAILABLE: hold_time_seconds null or 0 for all records")

    subsection("Regime Distribution")
    regime_counts: dict[str, int] = defaultdict(int)
    for t in trades:
        regime_counts[t.get("regime") or "UNKNOWN"] += 1
    if all(k == "UNKNOWN" for k in regime_counts):
        print("    UNAVAILABLE: regime column null for all records")
    else:
        for r, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {r}: {cnt}")

    subsection("Strategy ID Distribution")
    strategy_counts: dict[str, int] = defaultdict(int)
    for t in trades:
        strategy_counts[t.get("strategy_id") or "UNKNOWN"] += 1
    if all(k == "UNKNOWN" for k in strategy_counts):
        print("    UNAVAILABLE: strategy_id null for all records")
    else:
        for s, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"    {s}: {cnt}")

    # -----------------------------------------------------------------------
    section("6. ANSWERING THE 10 CALIBRATION QUESTIONS")
    # -----------------------------------------------------------------------

    # Q1: Adverse movement of eventual winners
    subsection("Q1: Adverse movement of eventual winners")
    winners_data = [t for t in trades if get_pnl(t)[0] > 0]
    if winners_data:
        # Use stop_price as proxy for max adverse if available
        stop_proxies = []
        for t in winners_data:
            if t.get("stop_price") and t.get("entry_price"):
                ep = float(t["entry_price"])
                sp = float(t["stop_price"])
                if sp < ep:
                    stop_proxies.append((ep - sp) / ep)
        if stop_proxies:
            print(f"    Proxy (stop_price gap from entry for eventual winners):\n    avg={mean(stop_proxies) * 100:.3f}%  median={median(stop_proxies) * 100:.3f}%  n={len(stop_proxies)}")
        else:
            print("    UNAVAILABLE: stop_price not usable as adverse proxy (null or above entry for all winners)")
    else:
        print("    UNAVAILABLE: no winners in dataset")

    # Q2: Favorable movement before successful continuation
    subsection("Q2: Favorable movement before continuation (MFE proxy via TP)")
    tp_proxies = []
    for t in trades:
        if t.get("take_profit_price") and t.get("entry_price") and float(t["take_profit_price"]) > 0 and float(t["entry_price"]) > 0:
            tp_proxies.append((float(t["take_profit_price"]) - float(t["entry_price"])) / float(t["entry_price"]))
    if tp_proxies:
        print(f"    TP target vs entry (proxy for expected MFE):\n    avg={mean(tp_proxies) * 100:.3f}%  median={median(tp_proxies) * 100:.3f}%  n={len(tp_proxies)}")
    else:
        print("    UNAVAILABLE: take_profit_price null for all records")

    # Q3: Edge decay by hold time bucket
    subsection("Q3: PnL distribution by hold_time bucket")
    buckets = [
        ("0-30m", 0, 30),
        ("30-60m", 30, 60),
        ("60-120m", 60, 120),
        ("120-300m", 120, 300),
        ("300m+", 300, 99999),
    ]
    print(f"    {'Bucket':<12} {'N':>5} {'Win%':>6} {'AvgPnL':>9} {'TotalPnL':>11}")
    for label, lo, hi in buckets:
        bucket_trades = [t for t in trades if t.get("hold_time_seconds") is not None and lo * 60 <= float(t["hold_time_seconds"]) < hi * 60]
        if not bucket_trades:
            print(f"    {label:<12} {'0':>5}")
            continue
        bpnls = [get_pnl(t)[0] for t in bucket_trades]
        bwins = sum(1 for p in bpnls if p > 0)
        print(f"    {label:<12} {len(bucket_trades):>5} {pct(bwins, len(bucket_trades)):>6} ${mean(bpnls):>7.2f} ${sum(bpnls):>9.2f}")

    # Q4: Stall exit — position that later recovered
    subsection("Q4: STALL_EXIT recovery (answered in Section 3 above)")
    print("    See Recovery Analysis section above.")

    # Q5: Giveback exit — position that later recovered
    subsection("Q5: GIVEBACK_EXIT recovery (answered in Section 3 above)")
    print("    See Recovery Analysis section above.")

    # Q6: Trailing stop before larger move
    subsection("Q6: TRAILING_STOP_EXIT — did the move continue?")
    print("    UNAVAILABLE: cannot answer without post-exit candle data. No OHLCV bars stored relative to exit timestamp in paper_trades.")

    # Q7: Same-opportunity clusters
    subsection("Q7: Same-opportunity clusters (answered in Section 4 above)")
    print("    See Repeated Entry section above.")

    # Q8: One-trade-per-cluster simulation (answered in Section 4)
    subsection("Q8: Turnover/results with one trade per cluster")
    print("    See Repeated Entry section simulation above.")

    # Q9: TP target realism
    subsection("Q9: Are 1.2-1.5% TP targets realistic?")
    if tp_proxies:
        tp_vals = [
            (float(t["take_profit_price"]) - float(t["entry_price"])) / float(t["entry_price"])
            for t in trades
            if t.get("take_profit_price") and t.get("entry_price") and float(t["take_profit_price"]) > 0
        ]
        within_15pct = sum(1 for v in tp_vals if v <= 0.015)
        above_15pct = sum(1 for v in tp_vals if v > 0.015)
        print(f"    TP targets <= 1.5%: {within_15pct} / {len(tp_vals)}\n    TP targets >  1.5%: {above_15pct} / {len(tp_vals)}")
        pnl_winner_pcts = []
        for t in winners_data:
            if t.get("pnl_pct_net") is not None:
                pnl_winner_pcts.append(float(t["pnl_pct_net"]))
            elif t.get("pnl_pct") is not None:
                pnl_winner_pcts.append(float(t["pnl_pct"]))
        if pnl_winner_pcts:
            above_12 = sum(1 for p in pnl_winner_pcts if p >= 0.012)
            above_15 = sum(1 for p in pnl_winner_pcts if p >= 0.015)
            print(f"    Winners actually achieving >= 1.2% pnl_pct: {above_12} / {len(pnl_winner_pcts)}")
            print(f"    Winners actually achieving >= 1.5% pnl_pct: {above_15} / {len(pnl_winner_pcts)}")
    else:
        print("    UNAVAILABLE: take_profit_price null — cannot assess target realism")

    # Q10: Hold time distribution — fraction > 3h, > 5h
    subsection("Q10: Is 5-6 hour max hold supported?")
    if hold_all:
        over_3h = sum(1 for h in hold_all if h > 3 * 3600)
        over_5h = sum(1 for h in hold_all if h > 5 * 3600)
        print(f"    Trades held > 3h: {over_3h} / {len(hold_all)} ({pct(over_3h, len(hold_all))})")
        print(f"    Trades held > 5h: {over_5h} / {len(hold_all)} ({pct(over_5h, len(hold_all))})")
        ps = percentiles(hold_all, [75, 90, 95])
        print(f"    p75={ps[75] / 3600:.2f}h  p90={ps[90] / 3600:.2f}h  p95={ps[95] / 3600:.2f}h")
    else:
        print("    UNAVAILABLE: hold_time_seconds null or 0")

    print()
    print("=" * 70)
    print("  END OF CALIBRATION REPORT")
    print("=" * 70)
    con.close()


if __name__ == "__main__":
    main()
