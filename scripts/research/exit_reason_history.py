"""Read-only: which DAY exit reasons have actually fired, and when?

The path-aware guard collapses the exit ladder to five reachable reasons. If that
holds in production then NET_PROFIT_EXIT / TP1_PARTIAL_EXIT / STOP_LOSS_EXIT /
TIME_STOP_EXIT should stop appearing once the guard took effect.
"""

from __future__ import annotations

import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"

PATH_AWARE_REACHABLE = {
    "EXTREME_PROTECTION_EXIT",
    "DAY_RISK_FLOOR_EXIT",
    "GIVEBACK_EXIT",
    "STALL_EXIT_DEAD_NO_MFE",
    "TRAILING_STOP_EXIT",
}
LEGACY_ONLY = {
    "NET_PROFIT_EXIT",
    "TP1_PARTIAL_EXIT",
    "STOP_LOSS_EXIT",
    "TIME_STOP_EXIT",
    "THESIS_INVALIDATION_EXIT",
    "FAILED_RECLAIM_EXIT",
    "ADAPTIVE_LOSS_EXIT",
    "PROGRESS_DECAY_EXIT",
    "DAY_4H_STRUCTURE_BREAK_EXIT",
    "PATH_EXECUTABLE_PROFIT",
}


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== all-time exit_reason counts on SELL rows ===")
    print(f"  {'reason':<34} {'n':>5} {'first':<20} {'last':<20} {'sum_pnl':>12}")
    rows = list(
        c.execute("SELECT COALESCE(exit_reason,'(null)') r, COUNT(*) n, MIN(created_at) f, MAX(created_at) l, SUM(pnl) p FROM paper_trades WHERE UPPER(side)='SELL' GROUP BY r ORDER BY n DESC")
    )
    for r in rows:
        pnl = f"{r['p']:.2f}" if r["p"] is not None else "-"
        print(f"  {r['r'][:34]:<34} {r['n']:>5} {str(r['f'])[:19]:<20} {str(r['l'])[:19]:<20} {pnl:>12}")

    print("\n=== classification ===")
    seen = {r["r"] for r in rows}
    for name, group in (("path-aware reachable", PATH_AWARE_REACHABLE), ("legacy-only", LEGACY_ONLY)):
        print(f"  {name}:")
        for reason in sorted(group):
            hit = next((r for r in rows if r["r"] == reason), None)
            if hit:
                print(f"    {reason:<32} n={hit['n']:<5} last_seen={str(hit['l'])[:19]}")
            else:
                print(f"    {reason:<32} NEVER FIRED")
    extra = seen - PATH_AWARE_REACHABLE - LEGACY_ONLY
    if extra:
        print(f"  other reasons present: {sorted(extra)}")

    print("\n=== SELL rows in the last 7 days, by reason ===")
    for r in c.execute(
        "SELECT COALESCE(exit_reason,'(null)') r, COUNT(*) n, SUM(pnl) p FROM paper_trades WHERE UPPER(side)='SELL' AND created_at >= datetime('now','-7 days') GROUP BY r ORDER BY n DESC"
    ):
        pnl = f"{r['p']:.2f}" if r["p"] is not None else "-"
        print(f"  {r['r'][:36]:<36} n={r['n']:<5} sum_pnl={pnl}")

    print("\n=== profitable exits: did any green position exit via a profit reason? ===")
    for r in c.execute("SELECT COALESCE(exit_reason,'(null)') r, COUNT(*) n, ROUND(AVG(pnl_pct),6) avg_pct FROM paper_trades WHERE UPPER(side)='SELL' AND pnl > 0 GROUP BY r ORDER BY n DESC"):
        print(f"  {r['r'][:36]:<36} winners={r['n']:<5} avg_pnl_pct={r['avg_pct']}")

    print("\n=== exit_type distribution (TAKE_PROFIT_1 vs MANUAL) ===")
    for r in c.execute("SELECT COALESCE(exit_type,'(null)') t, COUNT(*) n FROM paper_trades WHERE UPPER(side)='SELL' GROUP BY t ORDER BY n DESC"):
        print(f"  {r['t']!s:<30} {r['n']}")


if __name__ == "__main__":
    main()
