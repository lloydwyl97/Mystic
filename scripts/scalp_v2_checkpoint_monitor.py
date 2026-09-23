"""
SCALP V2 100-Trade Checkpoint Monitor
======================================
Read-only. Never modifies trading decisions, positions, intents, or parameters.

Runs as a singleton background process (PID-file locked — only one instance
allowed). At the 100th qualifying SCALP V2 closed trade it generates the full
report and writes it to:
  logs/scalp_v2_100_trade_report.json
  logs/scalp_v2_100_trade_report.txt

Qualifying trade: engine_id='SCALP_V2', side='SELL', status='executed',
  pnl_usd_net IS NOT NULL, is_synthetic IS NOT 1.

Also captures giveback counterfactual data for every GIVEBACK_EXIT
as it appears, storing to table scalp_v2_giveback_cf (created if absent).

Counterfactual algorithm v2 (2026-09-22):
  Measures remaining hold time from the ORIGINAL entry timestamp, not 20
  additional minutes after the giveback exit. Max-hold ceiling = 300 minutes
  (effective_max_hold_min from coin profile — NOT the retired paper-runner
  SCALP_HOLD_MAX_MINUTES=20). Missing 1m bar data (feature_ohlcv has 0 1m
  rows on Ocean) or missing stop/target prices → COUNTERFACTUAL_UNAVAILABLE.
  Entry price is never substituted as a default exit price.

Production exit ladder (path_aware=True, all engines):
  1.  DAY_RISK_FLOOR / EXTREME_PROTECTION  (catastrophic backstop ≥2%)
  2.  STOP_LOSS          (coin profile sl=1.0%, from entry — hard loss floor)
  3.  GIVEBACK_EXIT      (high-water pullback, fires BEFORE trailing stop)
  4.  TRAILING_STOP_EXIT (cost-aware ratchet once activated)
  5.  NET_PROFIT_EXIT    (resolved thesis/adaptive target + min_net floor)
  6.  TAKE_PROFIT_1      (coin profile tp=1.4% outer objective)
  7.  STALL_EXIT_DEAD    (dead inventory, no meaningful excursion)
  8.  TIME_STOP_EXIT     (max hold 300 min ceiling, unconditional)

TRAILING_STOP appears exactly once as executable evaluation (step 4).

Usage:
    python3 scripts/scalp_v2_checkpoint_monitor.py [--db PATH] [--poll 60]
    # Singleton: a second invocation exits immediately if one is already running.
    # PID file: /run/mystic/scalp_v2_monitor.pid (or --pid-file override)
"""

import argparse
import fcntl
import json
import logging
import os
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
# Constants
# ---------------------------------------------------------------------------

# Production max hold ceiling (from coin profile max_hold_min).
# This is NOT SCALP_HOLD_MAX_MINUTES=20 (retired paper runner env var).
_PROD_MAX_HOLD_SEC: int = 300 * 60  # 18 000 seconds

_DEFAULT_PID_FILE = "/run/mystic/scalp_v2_monitor.pid"

# ---------------------------------------------------------------------------
# PID-file singleton lock
# ---------------------------------------------------------------------------


def _acquire_pid_lock(pid_file: str) -> "int | None":
    """Acquire an OS-enforced exclusive lock on *pid_file*.

    Returns the open file descriptor on success (keep it open for the life of
    the process — the kernel releases the lock automatically on process exit or
    crash).  Returns None if another instance already holds the lock.
    """
    try:
        Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # /run/mystic may already exist or be created by udev

    try:
        fd = open(pid_file, "w")
    except OSError as exc:
        log.error("PID_LOCK_OPEN_FAILED path=%s err=%s", pid_file, exc)
        return None

    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None  # Another instance is running

    fd.write(str(os.getpid()))
    fd.flush()
    log.info("PID_LOCK_ACQUIRED pid=%d file=%s", os.getpid(), pid_file)
    return fd  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CF_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS scalp_v2_giveback_cf (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id              TEXT UNIQUE,
    symbol                TEXT,
    side                  TEXT,
    entry_price           REAL,
    exit_price            REAL,
    quantity              REAL,
    pnl_usd_net           REAL,
    pnl_pct_net           REAL,
    hold_seconds          REAL,
    entry_ts              TEXT,
    exit_ts               TEXT,
    mfe_price             REAL,
    mfe_bps               REAL,
    mae_price             REAL,
    mae_bps               REAL,
    high_water            REAL,
    stop_price            REAL,
    take_profit_price     REAL,
    trailing_stop_price   REAL,
    exit_reason           TEXT,
    -- post-exit 1m path (JSON array of {ts, o, h, l, c})
    post_exit_1m_bars     TEXT,
    -- counterfactual result fields (v2 algorithm — see module docstring)
    cf_status             TEXT DEFAULT 'PENDING',
    cf_missing_data_reason TEXT,
    cf_first_exit         TEXT,
    cf_exit_ts            TEXT,
    cf_exit_price         REAL,
    cf_pnl_usd_net        REAL,
    cf_saved_or_cost      REAL,   -- positive = giveback saved money, negative = cost money
    cf_computed           INTEGER DEFAULT 0,
    cf_max_hold_sec       REAL,
    cf_remaining_hold_sec REAL,
    -- audit: 1 when this row's CF was computed by the old v1 algorithm
    -- (which used entry price as default and 20-min max hold from paper runner)
    cf_audit_superseded   INTEGER DEFAULT 0,
    created_at            TEXT DEFAULT (datetime('now'))
)
"""

_LAST_SEEN_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS scalp_v2_monitor_state (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""

_NEW_COLUMNS: list[tuple[str, str]] = [
    ("cf_status", "TEXT DEFAULT 'PENDING'"),
    ("cf_missing_data_reason", "TEXT"),
    ("cf_max_hold_sec", "REAL"),
    ("cf_remaining_hold_sec", "REAL"),
    ("cf_audit_superseded", "INTEGER DEFAULT 0"),
]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CF_TABLE_DDL)
    conn.execute(_LAST_SEEN_TABLE_DDL)
    conn.commit()


def _migrate_cf_schema(conn: sqlite3.Connection) -> None:
    """Add v2 columns if absent and mark v1-computed rows for recompute."""
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(scalp_v2_giveback_cf)").fetchall()}
    added = []
    for col_name, col_def in _NEW_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE scalp_v2_giveback_cf ADD COLUMN {col_name} {col_def}")
            added.append(col_name)
    if added:
        log.info("CF_SCHEMA_MIGRATED added_columns=%s", added)

    # Mark all previously computed rows as superseded by the v1 algorithm.
    # They used: (a) SCALP_HOLD_MAX_MINUTES=20 as the ceiling (wrong — 300 min),
    # (b) time measured from giveback exit, not original entry (wrong), and
    # (c) entry price as cf_exit_price default when no bars (invalid).
    # Set cf_status='SUPERSEDED_BY_CF_V1' so the recompute pass can find them.
    n = conn.execute(
        """UPDATE scalp_v2_giveback_cf
           SET cf_audit_superseded = 1,
               cf_status = 'SUPERSEDED_BY_CF_V1'
           WHERE cf_computed = 1
             AND (cf_status IS NULL
                  OR cf_status NOT IN ('COMPUTED','UNAVAILABLE','INTRABAR_AMBIGUOUS'))"""
    ).rowcount
    if n:
        log.info("CF_SCHEMA_MIGRATED marked_superseded=%d rows", n)
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


def _load_post_exit_1m_bars(conn: sqlite3.Connection, symbol: str, exit_ts: str, limit: int) -> list[dict]:
    """Load up to *limit* 1m bars starting from *exit_ts*.

    Returns an empty list when feature_ohlcv has no 1m rows for the symbol
    (currently always empty on Ocean — verified 2026-09-22).
    """
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


# ---------------------------------------------------------------------------
# Counterfactual algorithm v2
# ---------------------------------------------------------------------------


def _compute_cf(trade: dict, post_bars: list[dict]) -> dict:
    """Compute giveback counterfactual using algorithm v2.

    "What exit would have fired first if GIVEBACK_EXIT had not fired?"

    Remaining hold window is measured from the ORIGINAL entry timestamp:
        remaining_sec = max(0, _PROD_MAX_HOLD_SEC - hold_seconds_at_giveback)

    where hold_seconds_at_giveback is the hold_time_seconds column from the
    SELL row (the actual hold before the giveback exit fired).

    Production exit ladder checked (in order, GIVEBACK suppressed):
        1. STOP_LOSS    — bar low ≤ stop_price
        2. NET_PROFIT   — bar high ≥ take_profit_price
        3. TIME_STOP_EXIT — end of remaining hold window

    If the same bar triggers both STOP_LOSS and NET_PROFIT the result is
    marked INTRABAR_AMBIGUOUS; the conservative assumption (stop first) is
    used for the PnL estimate.

    Returns cf_status in:
        COMPUTED            — all data available, result is reliable
        INTRABAR_AMBIGUOUS  — computed but one bar hit both stop and target
        UNAVAILABLE         — required data missing; no PnL estimate produced
    """
    ep = float(trade["entry_price"] or 0)
    qty = float(trade["quantity"] or 0)
    stop = float(trade.get("stop_price") or 0)
    tp = float(trade.get("take_profit_price") or 0)
    actual_pnl = float(trade["pnl_usd_net"] or 0)
    actual_exit_price = float(trade.get("price") or 0)
    hold_seconds = float(trade.get("hold_time_seconds") or 0)

    remaining_sec = max(0.0, float(_PROD_MAX_HOLD_SEC) - hold_seconds)
    remaining_bars = int(remaining_sec / 60)  # 1m bars that remain in the hold window

    missing_reasons: list[str] = []
    if not post_bars:
        missing_reasons.append("no_1m_bars_in_feature_ohlcv")
    if stop <= 0 and tp <= 0:
        missing_reasons.append("stop_price_and_take_profit_null_in_paper_trades")

    if missing_reasons:
        return {
            "cf_status": "UNAVAILABLE",
            "cf_missing_data_reason": "; ".join(missing_reasons),
            "cf_first_exit": None,
            "cf_exit_ts": None,
            "cf_exit_price": None,
            "cf_pnl_usd_net": None,
            "cf_saved_or_cost": None,
            "cf_max_hold_sec": float(_PROD_MAX_HOLD_SEC),
            "cf_remaining_hold_sec": remaining_sec,
        }

    # Approximate round-trip fee using recorded fee when available
    fee_est = abs(float(trade.get("fees_paid") or 0)) or (actual_exit_price * qty * 0.0002)

    bars_in_window = post_bars[:remaining_bars] if remaining_bars > 0 else []

    cf_exit_type: str | None = None
    cf_exit_ts: str | None = None
    cf_exit_price: float | None = None
    cf_ambiguous = False

    for bar in bars_in_window:
        bar_h = float(bar["h"])
        bar_l = float(bar["l"])
        bar_ts = str(bar["ts"])

        stop_hit = stop > 0 and bar_l <= stop
        tp_hit = tp > 0 and bar_h >= tp

        if stop_hit and tp_hit:
            # Both levels touched in the same 1m bar — intrabar ambiguity.
            # Conservative: assume the adverse direction (stop) fires first.
            cf_ambiguous = True
            cf_exit_type = "STOP_LOSS"
            cf_exit_ts = bar_ts
            cf_exit_price = stop
            break
        elif stop_hit:
            cf_exit_type = "STOP_LOSS"
            cf_exit_ts = bar_ts
            cf_exit_price = stop
            break
        elif tp_hit:
            cf_exit_type = "NET_PROFIT"
            cf_exit_ts = bar_ts
            cf_exit_price = tp
            break

    # If no trigger found within the remaining window, time stop fires at last bar
    if cf_exit_type is None:
        if bars_in_window:
            cf_exit_type = "TIME_STOP_EXIT"
            cf_exit_ts = str(bars_in_window[-1]["ts"])
            cf_exit_price = float(bars_in_window[-1]["c"])
        else:
            # remaining_sec > 0 but no bars available in that window
            return {
                "cf_status": "UNAVAILABLE",
                "cf_missing_data_reason": "no_bars_within_remaining_hold_window",
                "cf_first_exit": "TIME_STOP_EXIT",
                "cf_exit_ts": None,
                "cf_exit_price": None,
                "cf_pnl_usd_net": None,
                "cf_saved_or_cost": None,
                "cf_max_hold_sec": float(_PROD_MAX_HOLD_SEC),
                "cf_remaining_hold_sec": remaining_sec,
            }

    if cf_exit_price is None:
        # Defensive — should not reach here
        return {
            "cf_status": "UNAVAILABLE",
            "cf_missing_data_reason": "cf_exit_price_unresolvable",
            "cf_first_exit": cf_exit_type,
            "cf_exit_ts": cf_exit_ts,
            "cf_exit_price": None,
            "cf_pnl_usd_net": None,
            "cf_saved_or_cost": None,
            "cf_max_hold_sec": float(_PROD_MAX_HOLD_SEC),
            "cf_remaining_hold_sec": remaining_sec,
        }

    cf_pnl = (cf_exit_price - ep) * qty - fee_est
    cf_saved = actual_pnl - cf_pnl  # positive means giveback saved money vs. alternative

    return {
        "cf_status": "INTRABAR_AMBIGUOUS" if cf_ambiguous else "COMPUTED",
        "cf_missing_data_reason": "intrabar_stop_and_target_same_bar" if cf_ambiguous else None,
        "cf_first_exit": cf_exit_type,
        "cf_exit_ts": cf_exit_ts,
        "cf_exit_price": round(cf_exit_price, 8),
        "cf_pnl_usd_net": round(cf_pnl, 6),
        "cf_saved_or_cost": round(cf_saved, 6),
        "cf_max_hold_sec": float(_PROD_MAX_HOLD_SEC),
        "cf_remaining_hold_sec": remaining_sec,
    }


# ---------------------------------------------------------------------------
# Giveback capture
# ---------------------------------------------------------------------------


def _capture_givebacks(conn: sqlite3.Connection, trades: list[dict]) -> None:
    """Capture / recompute giveback counterfactuals for all GIVEBACK_EXIT trades.

    Three cases:
      1. trade_id not yet in table → INSERT fresh row.
      2. trade_id present AND cf_status in ('SUPERSEDED_BY_CF_V1', 'PENDING') →
         UPDATE with corrected v2 values.
      3. trade_id present AND cf_status in ('COMPUTED','UNAVAILABLE',
         'INTRABAR_AMBIGUOUS') → skip (already correct).
    """
    givebacks = [t for t in trades if (t.get("exit_reason") or "").upper() == "GIVEBACK_EXIT"]
    if not givebacks:
        return

    existing: dict[str, str] = {r[0]: (r[1] or "PENDING") for r in conn.execute("SELECT trade_id, cf_status FROM scalp_v2_giveback_cf").fetchall()}

    terminal_statuses = frozenset({"COMPUTED", "UNAVAILABLE", "INTRABAR_AMBIGUOUS"})
    new_count = updated_count = 0

    for t in givebacks:
        tid = t.get("trade_id") or str(t["id"])
        existing_status = existing.get(tid)

        if existing_status in terminal_statuses:
            continue  # Already computed with v2 algorithm

        symbol = t["symbol"]
        exit_ts = t.get("timestamp") or ""
        hold_seconds = float(t.get("hold_time_seconds") or 0)
        remaining_bars = max(0, int((float(_PROD_MAX_HOLD_SEC) - hold_seconds) / 60))
        post_bars = _load_post_exit_1m_bars(conn, symbol, exit_ts, limit=remaining_bars + 5)
        cf = _compute_cf(t, post_bars)

        if existing_status is None:
            # New row — INSERT
            conn.execute(
                """
                INSERT OR IGNORE INTO scalp_v2_giveback_cf (
                    trade_id, symbol, side, entry_price, exit_price, quantity,
                    pnl_usd_net, pnl_pct_net, hold_seconds,
                    entry_ts, exit_ts,
                    stop_price, take_profit_price,
                    exit_reason, post_exit_1m_bars,
                    cf_status, cf_missing_data_reason,
                    cf_first_exit, cf_exit_ts, cf_exit_price,
                    cf_pnl_usd_net, cf_saved_or_cost,
                    cf_computed, cf_max_hold_sec, cf_remaining_hold_sec,
                    cf_audit_superseded
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    cf["cf_status"],
                    cf.get("cf_missing_data_reason"),
                    cf.get("cf_first_exit"),
                    cf.get("cf_exit_ts"),
                    cf.get("cf_exit_price"),
                    cf.get("cf_pnl_usd_net"),
                    cf.get("cf_saved_or_cost"),
                    1,
                    cf.get("cf_max_hold_sec"),
                    cf.get("cf_remaining_hold_sec"),
                    0,
                ),
            )
            new_count += 1
        else:
            # Superseded/pending row — UPDATE in-place
            conn.execute(
                """
                UPDATE scalp_v2_giveback_cf
                SET cf_status              = ?,
                    cf_missing_data_reason = ?,
                    cf_first_exit          = ?,
                    cf_exit_ts             = ?,
                    cf_exit_price          = ?,
                    cf_pnl_usd_net         = ?,
                    cf_saved_or_cost       = ?,
                    cf_computed            = 1,
                    cf_max_hold_sec        = ?,
                    cf_remaining_hold_sec  = ?,
                    cf_audit_superseded    = 0,
                    post_exit_1m_bars      = ?
                WHERE trade_id = ?
                """,
                (
                    cf["cf_status"],
                    cf.get("cf_missing_data_reason"),
                    cf.get("cf_first_exit"),
                    cf.get("cf_exit_ts"),
                    cf.get("cf_exit_price"),
                    cf.get("cf_pnl_usd_net"),
                    cf.get("cf_saved_or_cost"),
                    cf.get("cf_max_hold_sec"),
                    cf.get("cf_remaining_hold_sec"),
                    json.dumps(post_bars),
                    tid,
                ),
            )
            updated_count += 1

    if new_count or updated_count:
        conn.commit()
        log.info(
            "GIVEBACK_CF: new=%d recomputed=%d (v2 algorithm)",
            new_count,
            updated_count,
        )


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
        sl_ = [t for t in ts if float(t["pnl_usd_net"]) <= 0]
        sp = sum(float(t["pnl_usd_net"]) for t in ts)
        spf = abs(sum(float(t["pnl_usd_net"]) for t in sw)) / abs(sum(float(t["pnl_usd_net"]) for t in sl_)) if sl_ else float("inf")
        sh = [float(t.get("hold_time_seconds") or 0) for t in ts if t.get("hold_time_seconds")]
        report["per_symbol"][sym] = {
            "trades": len(ts),
            "wins": len(sw),
            "losses": len(sl_),
            "win_rate_pct": round(len(sw) / len(ts) * 100, 1),
            "net_pnl": round(sp, 4),
            "avg_winner": round(sum(float(t["pnl_usd_net"]) for t in sw) / len(sw), 4) if sw else 0,
            "avg_loser": round(sum(float(t["pnl_usd_net"]) for t in sl_) / len(sl_), 4) if sl_ else 0,
            "payoff_ratio": round(
                abs(sum(float(t["pnl_usd_net"]) for t in sw) / len(sw)) / abs(sum(float(t["pnl_usd_net"]) for t in sl_) / len(sl_)),
                3,
            )
            if sw and sl_
            else None,
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
            "net_pnl": round(sum(ep_list), 4),
            "fees": round(sum(float(t.get("fees_paid") or 0) for t in ts), 4),
            "avg_pnl": round(sum(ep_list) / len(ts), 4),
            "avg_hold_sec": round(sum(eh) / len(eh), 1) if eh else None,
            "symbols": sorted({t["symbol"] for t in ts}),
        }

    # --- Giveback counterfactual ---
    cf_rows = conn.execute("SELECT * FROM scalp_v2_giveback_cf WHERE cf_audit_superseded = 0").fetchall()
    cf_cols = [d[0] for d in conn.execute("PRAGMA table_info(scalp_v2_giveback_cf)").fetchall()]
    cf_list = [dict(zip(cf_cols, r, strict=False)) for r in cf_rows]

    computed_rows = [r for r in cf_list if (r.get("cf_status") or "") in ("COMPUTED", "INTRABAR_AMBIGUOUS")]
    unavailable_rows = [r for r in cf_list if (r.get("cf_status") or "") == "UNAVAILABLE"]

    cf_summary: dict[str, Any] = {
        "total_giveback_exits": len(cf_list),
        "cf_computed": len(computed_rows),
        "cf_unavailable": len(unavailable_rows),
        "cf_algorithm_version": "v2_2026-09-22",
        "cf_max_hold_sec": _PROD_MAX_HOLD_SEC,
        "unavailability_reasons": sorted({r.get("cf_missing_data_reason") or "unknown" for r in unavailable_rows}),
    }
    if computed_rows:
        gb_actual_pnl = sum(float(r["pnl_usd_net"] or 0) for r in computed_rows)
        gb_cf_pnl = sum(float(r["cf_pnl_usd_net"] or 0) for r in computed_rows)
        gb_diff = gb_cf_pnl - gb_actual_pnl
        cf_exits: dict[str, int] = defaultdict(int)
        for r in computed_rows:
            cf_exits[str(r.get("cf_first_exit") or "UNKNOWN")] += 1
        cf_summary.update(
            {
                "computed_actual_total_pnl": round(gb_actual_pnl, 4),
                "computed_cf_total_pnl_no_giveback": round(gb_cf_pnl, 4),
                "delta_cf_minus_actual": round(gb_diff, 4),
                "interpretation": (f"counterfactual BETTER than giveback by ${gb_diff:.4f}" if gb_diff > 0 else f"giveback BETTER than counterfactual by ${-gb_diff:.4f}"),
                "cf_alternative_exit_distribution": dict(cf_exits),
                "giveback_saved_count": sum(1 for r in computed_rows if float(r.get("cf_saved_or_cost") or 0) > 0),
                "giveback_cost_count": sum(1 for r in computed_rows if float(r.get("cf_saved_or_cost") or 0) <= 0),
            }
        )
    cf_summary["rows"] = [
        {
            "trade_id": r["trade_id"],
            "symbol": r["symbol"],
            "cf_status": r.get("cf_status"),
            "actual_pnl": r["pnl_usd_net"],
            "cf_pnl": r.get("cf_pnl_usd_net"),
            "cf_exit": r.get("cf_first_exit"),
            "cf_remaining_hold_sec": r.get("cf_remaining_hold_sec"),
            "cf_missing_data_reason": r.get("cf_missing_data_reason"),
            "exit_ts": r.get("exit_ts"),
        }
        for r in cf_list
    ]
    report["giveback_counterfactual"] = cf_summary

    # --- Opportunity recycling ---
    opp_trades: dict[str, list] = defaultdict(list)
    for t in trades:
        oid = t.get("scalp_opportunity_id") or "no_opp"
        opp_trades[oid].append(t)
    report["opportunity_recycling"] = {
        "unique_opportunities": len(opp_trades),
        "single_trade_opps": sum(1 for v in opp_trades.values() if len(v) == 1),
        "multi_trade_opps": sum(1 for v in opp_trades.values() if len(v) > 1),
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
    parser.add_argument(
        "--pid-file",
        default=_DEFAULT_PID_FILE,
        help="PID lock file path (default: /run/mystic/scalp_v2_monitor.pid)",
    )
    args = parser.parse_args()

    # Singleton enforcement: refuse to run if another instance already holds the lock.
    _pid_fd = _acquire_pid_lock(args.pid_file)
    if _pid_fd is None:
        log.warning(
            "MONITOR_ALREADY_RUNNING pid_file=%s — exiting; only one instance allowed",
            args.pid_file,
        )
        return

    db_path = args.db
    log_dir = Path(args.log_dir)

    log.info(
        "MONITOR_START pid=%d db=%s poll=%ds pid_file=%s target=100 SCALP_V2 trades",
        os.getpid(),
        db_path,
        args.poll,
        args.pid_file,
    )

    report_written = False

    while True:
        try:
            conn = sqlite3.connect(db_path, timeout=15)
            _ensure_schema(conn)
            _migrate_cf_schema(conn)

            trades = _load_qualifying_trades(conn)
            count = len(trades)

            # Always capture / recompute giveback CFs incrementally
            _capture_givebacks(conn, trades)

            givebacks_seen = len([t for t in trades if (t.get("exit_reason") or "").upper() == "GIVEBACK_EXIT"])
            already_reported = _get_state(conn, "report_written_at")

            log.info(
                "MONITOR_POLL qualifying_trades=%d target=100 givebacks=%d report_done=%s",
                count,
                givebacks_seen,
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
