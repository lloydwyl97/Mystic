"""
SCALP V2 100-Trade Checkpoint Monitor
======================================
Read-only. Never modifies trading decisions, positions, intents, or parameters.

Runs as a background process. At the 100th qualifying SCALP V2 closed trade
it generates the full report and writes it to:
  logs/scalp_v2_100_trade_report.json
  logs/scalp_v2_100_trade_report.txt

Qualifying trade: engine_id='SCALP_V2', side='SELL', status='executed',
  pnl_usd_net IS NOT NULL, is_synthetic IS NOT 1.

Also captures giveback counterfactual data for every GIVEBACK_EXIT
as it appears, storing to table scalp_v2_giveback_cf (created if absent).

Usage:
    python3 scripts/scalp_v2_checkpoint_monitor.py [--db PATH] [--poll 60]
    # or add to cron / run via start_mystic.sh (read-only, safe to run always)
"""

import argparse
import json
import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] scalp_monitor: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CF_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS scalp_v2_giveback_cf (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT UNIQUE,
    symbol          TEXT,
    side            TEXT,
    entry_price     REAL,
    exit_price      REAL,
    quantity        REAL,
    pnl_usd_net     REAL,
    pnl_pct_net     REAL,
    hold_seconds    REAL,
    entry_ts        TEXT,
    exit_ts         TEXT,
    mfe_price       REAL,
    mfe_bps         REAL,
    mae_price       REAL,
    mae_bps         REAL,
    high_water      REAL,
    stop_price      REAL,
    take_profit_price REAL,
    trailing_stop_price REAL,
    exit_reason     TEXT,
    -- post-exit 1m path (JSON array of {ts, o, h, l, c})
    post_exit_1m_bars TEXT,
    -- counterfactual: what exit would have fired first if giveback were disabled
    cf_first_exit   TEXT,
    cf_exit_ts      TEXT,
    cf_exit_price   REAL,
    cf_pnl_usd_net  REAL,
    cf_saved_or_cost REAL,   -- positive = giveback saved money, negative = cost money
    cf_computed     INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
)
"""

_LAST_SEEN_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS scalp_v2_monitor_state (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CF_TABLE_DDL)
    conn.execute(_LAST_SEEN_TABLE_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_state(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM scalp_v2_monitor_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def _set_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scalp_v2_monitor_state (key, value) VALUES (?,?)",
        (key, str(value)),
    )
    conn.commit()


def _load_qualifying_trades(conn: sqlite3.Connection) -> list[dict]:
    """Load all qualifying SCALP V2 closed trades (read-only)."""
    rows = conn.execute(
        """
        SELECT
            id, trade_id, symbol, side, quantity, price, entry_price,
            pnl_usd_net, pnl_pct_net, hold_time_seconds, exit_reason,
            entry_timestamp, timestamp,
            stop_price, take_profit_price, atr_at_entry,
            fees_paid, entry_fee_usd, exit_fee_usd,
            scalp_opportunity_id, engine_id, order_id,
            slippage_pct_used, spread_pct_used
        FROM paper_trades
        WHERE engine_id = 'SCALP_V2'
          AND side       = 'SELL'
          AND status     = 'executed'
          AND pnl_usd_net IS NOT NULL
          AND (is_synthetic IS NULL OR is_synthetic != 1)
        ORDER BY id ASC
        """
    ).fetchall()
    cols = [
        "id",
        "trade_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "entry_price",
        "pnl_usd_net",
        "pnl_pct_net",
        "hold_time_seconds",
        "exit_reason",
        "entry_timestamp",
        "timestamp",
        "stop_price",
        "take_profit_price",
        "atr_at_entry",
        "fees_paid",
        "entry_fee_usd",
        "exit_fee_usd",
        "scalp_opportunity_id",
        "engine_id",
        "order_id",
        "slippage_pct_used",
        "spread_pct_used",
    ]
    return [dict(zip(cols, r, strict=False)) for r in rows]


def _load_post_exit_1m_bars(conn: sqlite3.Connection, symbol: str, exit_ts: str, limit: int = 30) -> list[dict]:
    """Load up to `limit` 1m bars starting from exit_ts."""
    # Normalise symbol: feature_ohlcv may use slash or hyphen format
    for sym in (symbol, symbol.replace("/", "-"), symbol.replace("/", "")):
        rows = conn.execute(
            """
            SELECT ts, open, high, low, close
            FROM feature_ohlcv
            WHERE symbol=? AND interval='1m' AND ts >= ?
            ORDER BY ts ASC LIMIT ?
            """,
            (sym, exit_ts, limit),
        ).fetchall()
        if rows:
            return [{"ts": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4]} for r in rows]
    return []


def _compute_cf(trade: dict, post_bars: list[dict]) -> dict:
    """
    Compute giveback counterfactual: if giveback had NOT fired,
    which alternative exit would have triggered first and at what price?

    Alternatives checked (in priority order):
      1. STOP_LOSS (hard stop_price)
      2. TRAILING_STOP (trailing_stop_price from highest_price x trail_pct)
      3. NET_PROFIT (take_profit_price)
      4. TIME_EXPIRY (SCALP_HOLD_MAX_MINUTES = 20 min = 1200s)

    Returns dict with cf_first_exit, cf_exit_ts, cf_exit_price,
    cf_pnl_usd_net, cf_saved_or_cost.
    """
    ep = float(trade["entry_price"] or 0)
    qty = float(trade["quantity"] or 0)
    stop = float(trade.get("stop_price") or 0)
    tp = float(trade.get("take_profit_price") or 0)
    actual_exit = float(trade["price"] or 0)
    actual_pnl = float(trade["pnl_usd_net"] or 0)

    # Approximate fees
    fee_est = abs(float(trade.get("fees_paid") or 0) or (actual_exit * qty * 0.0002))

    # Build a "trailing stop" estimate: SCALP_MICRO_TP_GIVEBACK_FRAC=0.35 giveback of MFE
    # We don't have intra-trade 1m bars before exit, so we use the highest close
    # in the post_exit bars as a proxy for what the market did after.
    # For the counterfactual, we walk forward from the giveback exit point.

    cf_exit_type = "TIME_EXPIRY"
    cf_exit_ts = None
    cf_exit_price = ep  # worst case: back to entry
    for bar in post_bars:
        bar_c = float(bar["c"])
        bar_l = float(bar["l"])
        bar_ts = bar["ts"]

        # STOP LOSS fires when low goes below stop
        if stop > 0 and bar_l <= stop:
            cf_exit_type = "STOP_LOSS"
            cf_exit_ts = bar_ts
            cf_exit_price = stop
            break

        # NET PROFIT fires when close >= take_profit
        if tp > 0 and bar_c >= tp:
            cf_exit_type = "NET_PROFIT"
            cf_exit_ts = bar_ts
            cf_exit_price = tp
            break

        # TIME EXPIRY — use bar count as proxy (30 bars = 30 min, use 20 min = 20 bars)
        idx = post_bars.index(bar)
        if idx >= 20:
            cf_exit_type = "TIME_EXPIRY"
            cf_exit_ts = bar_ts
            cf_exit_price = bar_c
            break
    else:
        # Exhausted post_exit bars without a trigger — use last bar price
        if post_bars:
            cf_exit_type = "END_OF_DATA"
            cf_exit_ts = post_bars[-1]["ts"]
            cf_exit_price = float(post_bars[-1]["c"])

    cf_pnl = (cf_exit_price - ep) * qty - fee_est
    cf_saved = actual_pnl - cf_pnl  # positive = giveback was better (saved money vs alternative)

    return {
        "cf_first_exit": cf_exit_type,
        "cf_exit_ts": str(cf_exit_ts or ""),
        "cf_exit_price": round(cf_exit_price, 8),
        "cf_pnl_usd_net": round(cf_pnl, 6),
        "cf_saved_or_cost": round(cf_saved, 6),
    }


def _capture_givebacks(conn: sqlite3.Connection, trades: list[dict]) -> None:
    """For each GIVEBACK_EXIT not yet captured, fetch post-exit bars and compute CF."""
    givebacks = [t for t in trades if (t.get("exit_reason") or "").upper() == "GIVEBACK_EXIT"]
    if not givebacks:
        return

    existing = {r[0] for r in conn.execute("SELECT trade_id FROM scalp_v2_giveback_cf").fetchall()}

    new_count = 0
    for t in givebacks:
        tid = t.get("trade_id") or str(t["id"])
        if tid in existing:
            continue

        symbol = t["symbol"]
        exit_ts = t.get("timestamp") or ""
        post_bars = _load_post_exit_1m_bars(conn, symbol, exit_ts, limit=30)
        cf = _compute_cf(t, post_bars)

        conn.execute(
            """
            INSERT OR IGNORE INTO scalp_v2_giveback_cf (
                trade_id, symbol, side, entry_price, exit_price, quantity,
                pnl_usd_net, pnl_pct_net, hold_seconds,
                entry_ts, exit_ts,
                stop_price, take_profit_price,
                exit_reason, post_exit_1m_bars,
                cf_first_exit, cf_exit_ts, cf_exit_price,
                cf_pnl_usd_net, cf_saved_or_cost, cf_computed
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
            """,
            (
                tid,
                symbol,
                "SELL",
                t.get("entry_price"),
                t.get("price"),
                t.get("quantity"),
                t.get("pnl_usd_net"),
                t.get("pnl_pct_net"),
                t.get("hold_time_seconds"),
                t.get("entry_timestamp"),
                exit_ts,
                t.get("stop_price"),
                t.get("take_profit_price"),
                t.get("exit_reason"),
                json.dumps(post_bars),
                cf["cf_first_exit"],
                cf["cf_exit_ts"],
                cf["cf_exit_price"],
                cf["cf_pnl_usd_net"],
                cf["cf_saved_or_cost"],
            ),
        )
        new_count += 1

    if new_count:
        conn.commit()
        log.info("GIVEBACK_CF: captured %d new giveback counterfactuals", new_count)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(trades: list[dict], conn: sqlite3.Connection, db_path: str) -> dict:
    """Build the full 100-trade report dict."""
    report: dict[str, Any] = {}

    # --- Account snapshot ---
    led = conn.execute("SELECT principal, cash_balance, realized_pnl, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
    report["account"] = {
        "principal": round(float(led[0]), 4) if led else None,
        "cash_balance": round(float(led[1]), 4) if led else None,
        "realized_pnl_ledger": round(float(led[2]), 4) if led else None,
        "total_equity": round(float(led[3]), 4) if led else None,
        "scalp_v2_net_pnl": round(sum(float(t["pnl_usd_net"]) for t in trades), 4),
        "scalp_v2_gross_fees": round(sum(float(t.get("fees_paid") or 0) for t in trades), 4),
    }

    # --- Overall SCALP V2 ---
    wins = [t for t in trades if float(t["pnl_usd_net"]) > 0]
    losses = [t for t in trades if float(t["pnl_usd_net"]) <= 0]
    win_pnl = sum(float(t["pnl_usd_net"]) for t in wins)
    loss_pnl = sum(float(t["pnl_usd_net"]) for t in losses)
    total_pnl = win_pnl + loss_pnl
    wr = len(wins) / len(trades) * 100 if trades else 0
    avg_win = win_pnl / len(wins) if wins else 0
    avg_loss = loss_pnl / len(losses) if losses else 0
    pf = abs(win_pnl / loss_pnl) if loss_pnl else float("inf")
    breakeven_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100 if (avg_win + abs(avg_loss)) > 0 else 50
    holds = [float(t.get("hold_time_seconds") or 0) for t in trades if t.get("hold_time_seconds")]
    dates = sorted({(t.get("timestamp") or "")[:10] for t in trades if t.get("timestamp")})

    # Max consecutive losses
    max_consec = cur_consec = 0
    for t in trades:
        if float(t["pnl_usd_net"]) <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    report["overall"] = {
        "qualifying_trades": len(trades),
        "date_range": f"{dates[0] if dates else '?'} → {dates[-1] if dates else '?'}",
        "days": len(dates),
        "trades_per_day": round(len(trades) / len(dates), 1) if dates else None,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(wr, 2),
        "avg_winner_usd": round(avg_win, 4),
        "avg_loser_usd": round(avg_loss, 4),
        "payoff_ratio": round(abs(avg_win / avg_loss), 3) if avg_loss else None,
        "breakeven_win_rate_pct": round(breakeven_wr, 1),
        "gross_pnl": round(total_pnl, 4),
        "total_fees": round(sum(float(t.get("fees_paid") or 0) for t in trades), 4),
        "net_pnl": round(total_pnl, 4),
        "expectancy_per_trade": round(total_pnl / len(trades), 4),
        "profit_factor": round(pf, 3),
        "avg_hold_sec": round(sum(holds) / len(holds), 1) if holds else None,
        "median_hold_sec": round(sorted(holds)[len(holds) // 2], 1) if holds else None,
        "max_consecutive_losses": max_consec,
    }

    # --- Per symbol ---
    report["per_symbol"] = {}
    by_sym: dict[str, list] = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    for sym, ts in sorted(by_sym.items()):
        sw = [t for t in ts if float(t["pnl_usd_net"]) > 0]
        sl = [t for t in ts if float(t["pnl_usd_net"]) <= 0]
        sp = sum(float(t["pnl_usd_net"]) for t in ts)
        spf = abs(sum(float(t["pnl_usd_net"]) for t in sw)) / abs(sum(float(t["pnl_usd_net"]) for t in sl)) if sl else float("inf")
        sh = [float(t.get("hold_time_seconds") or 0) for t in ts if t.get("hold_time_seconds")]
        report["per_symbol"][sym] = {
            "trades": len(ts),
            "wins": len(sw),
            "losses": len(sl),
            "win_rate_pct": round(len(sw) / len(ts) * 100, 1),
            "net_pnl": round(sp, 4),
            "avg_winner": round(sum(float(t["pnl_usd_net"]) for t in sw) / len(sw), 4) if sw else 0,
            "avg_loser": round(sum(float(t["pnl_usd_net"]) for t in sl) / len(sl), 4) if sl else 0,
            "payoff_ratio": round(abs(sum(float(t["pnl_usd_net"]) for t in sw) / len(sw)) / abs(sum(float(t["pnl_usd_net"]) for t in sl) / len(sl)), 3) if sw and sl else None,
            "profit_factor": round(spf, 3),
            "fees": round(sum(float(t.get("fees_paid") or 0) for t in ts), 4),
            "avg_hold_sec": round(sum(sh) / len(sh), 1) if sh else None,
        }

    # --- Per exit reason ---
    report["exit_attribution"] = {}
    by_exit: dict[str, list] = defaultdict(list)
    for t in trades:
        by_exit[t.get("exit_reason") or "UNKNOWN"].append(t)
    for ex, ts in sorted(by_exit.items(), key=lambda x: -len(x[1])):
        ew = [t for t in ts if float(t["pnl_usd_net"]) > 0]
        ep_list = [float(t["pnl_usd_net"]) for t in ts]
        eh = [float(t.get("hold_time_seconds") or 0) for t in ts if t.get("hold_time_seconds")]
        report["exit_attribution"][ex] = {
            "count": len(ts),
            "wins": len(ew),
            "win_rate_pct": round(len(ew) / len(ts) * 100, 1),
            "gross_pnl": round(sum(ep_list), 4),
            "net_pnl": round(sum(ep_list), 4),
            "fees": round(sum(float(t.get("fees_paid") or 0) for t in ts), 4),
            "avg_pnl": round(sum(ep_list) / len(ts), 4),
            "avg_hold_sec": round(sum(eh) / len(eh), 1) if eh else None,
            "symbols": sorted({t["symbol"] for t in ts}),
        }

    # --- Giveback counterfactual ---
    cf_rows = conn.execute("SELECT * FROM scalp_v2_giveback_cf").fetchall()
    cf_cols = [d[0] for d in conn.execute("PRAGMA table_info(scalp_v2_giveback_cf)").fetchall()]
    cf_list = [dict(zip(cf_cols, r, strict=False)) for r in cf_rows]

    if cf_list:
        gb_actual_pnl = sum(float(r["pnl_usd_net"] or 0) for r in cf_list)
        gb_cf_pnl = sum(float(r["cf_pnl_usd_net"] or 0) for r in cf_list)
        gb_diff = gb_cf_pnl - gb_actual_pnl  # positive = giveback cost money vs alternative

        cf_exits: dict[str, int] = defaultdict(int)
        for r in cf_list:
            cf_exits[str(r.get("cf_first_exit") or "UNKNOWN")] += 1

        report["giveback_counterfactual"] = {
            "giveback_count": len(cf_list),
            "actual_total_pnl": round(gb_actual_pnl, 4),
            "counterfactual_total_pnl_no_giveback": round(gb_cf_pnl, 4),
            "delta_cf_minus_actual": round(gb_diff, 4),
            "interpretation": (f"counterfactual BETTER than giveback by ${gb_diff:.4f}" if gb_diff > 0 else f"giveback BETTER than counterfactual by ${-gb_diff:.4f}"),
            "cf_alternative_exit_distribution": dict(cf_exits),
            "giveback_saved_count": sum(1 for r in cf_list if float(r.get("cf_saved_or_cost") or 0) > 0),
            "giveback_cost_count": sum(1 for r in cf_list if float(r.get("cf_saved_or_cost") or 0) <= 0),
            "rows": [
                {
                    "trade_id": r["trade_id"],
                    "symbol": r["symbol"],
                    "actual_pnl": r["pnl_usd_net"],
                    "cf_pnl": r["cf_pnl_usd_net"],
                    "cf_exit": r["cf_first_exit"],
                    "cf_saved_or_cost": r["cf_saved_or_cost"],
                    "exit_ts": r["exit_ts"],
                }
                for r in cf_list
            ],
        }
    else:
        report["giveback_counterfactual"] = {"note": "No giveback counterfactual data captured yet."}

    # --- Opportunity recycling ---
    opp_trades: dict[str, list] = defaultdict(list)
    for t in trades:
        oid = t.get("scalp_opportunity_id") or "no_opp"
        opp_trades[oid].append(t)
    multi_entry_opps = {k: v for k, v in opp_trades.items() if len(v) > 1}
    report["opportunity_recycling"] = {
        "unique_opportunities": len(opp_trades),
        "single_trade_opps": sum(1 for v in opp_trades.values() if len(v) == 1),
        "multi_trade_opps": len(multi_entry_opps),
        "max_trades_per_opp": max(len(v) for v in opp_trades.values()),
    }

    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["qualifying_trade_count"] = len(trades)
    return report


def _write_report(report: dict, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "scalp_v2_100_trade_report.json"
    txt_path = log_dir / "scalp_v2_100_trade_report.txt"

    json_path.write_text(json.dumps(report, indent=2))

    # Human-readable text
    lines = [
        "=" * 72,
        "SCALP V2 100-TRADE CHECKPOINT REPORT",
        f"Generated: {report['generated_at']}",
        "=" * 72,
        "",
        "ACCOUNT",
        "-" * 40,
    ]
    for k, v in report["account"].items():
        lines.append(f"  {k:<35} {v}")

    lines += ["", "OVERALL SCALP V2", "-" * 40]
    for k, v in report["overall"].items():
        lines.append(f"  {k:<35} {v}")

    lines += ["", "PER SYMBOL", "-" * 40]
    for sym, d in report["per_symbol"].items():
        lines.append(f"  {sym}")
        for k, v in d.items():
            lines.append(f"    {k:<33} {v}")

    lines += ["", "EXIT ATTRIBUTION", "-" * 40]
    for ex, d in report["exit_attribution"].items():
        lines.append(f"  {ex}")
        for k, v in d.items():
            lines.append(f"    {k:<33} {v}")

    lines += ["", "GIVEBACK COUNTERFACTUAL", "-" * 40]
    gcf = report.get("giveback_counterfactual", {})
    for k, v in gcf.items():
        if k != "rows":
            lines.append(f"  {k:<35} {v}")
    if "rows" in gcf:
        lines.append("  rows:")
        for row in gcf["rows"]:
            lines.append(f"    {row}")

    lines += ["", "OPPORTUNITY RECYCLING", "-" * 40]
    for k, v in report.get("opportunity_recycling", {}).items():
        lines.append(f"  {k:<35} {v}")

    lines += ["", "=" * 72, "END OF REPORT"]
    txt_path.write_text("\n".join(lines))

    log.warning(
        "CHECKPOINT_REPORT_WRITTEN json=%s txt=%s",
        str(json_path),
        str(txt_path),
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SCALP V2 100-trade checkpoint monitor")
    parser.add_argument("--db", default="/home/mystic/mystic/mystic_trading.db")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--log-dir", default="/home/mystic/mystic/logs")
    args = parser.parse_args()

    db_path = args.db
    log_dir = Path(args.log_dir)

    log.info(
        "MONITOR_START db=%s poll=%ds target=100 qualifying SCALP_V2 trades",
        db_path,
        args.poll,
    )

    report_written = False

    while True:
        try:
            conn = sqlite3.connect(db_path, timeout=15)
            _ensure_schema(conn)

            trades = _load_qualifying_trades(conn)
            count = len(trades)

            # Always capture giveback CFs incrementally
            _capture_givebacks(conn, trades)

            already_reported = _get_state(conn, "report_written_at")

            log.info(
                "MONITOR_POLL qualifying_trades=%d target=100 givebacks=%d report_done=%s",
                count,
                len([t for t in trades if (t.get("exit_reason") or "").upper() == "GIVEBACK_EXIT"]),
                bool(already_reported),
            )

            if count >= 100 and not already_reported and not report_written:
                log.warning("CHECKPOINT_100_REACHED generating report ...")
                report = _generate_report(trades[:100], conn, db_path)
                _write_report(report, log_dir)
                _set_state(conn, "report_written_at", report["generated_at"])
                report_written = True
                log.warning("CHECKPOINT_REPORT_COMPLETE trades=%d", len(trades[:100]))

            conn.close()

        except Exception:
            log.exception("MONITOR_ERROR poll failed")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
