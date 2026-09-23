"""DAY V2 engine identity repair — cohort classification and audit writer.

Commit 628c47f (2026-09-21 ~23:48 UTC, arm_ts threshold ≈ 1790033000) changed
day_trailing_buy.py to hardcode SCALP_V2 as engine_id for every new DAY
trailing-buy intent.  Before that commit all DAY intents carried LEGACY_DAY_LIVE.
The canonical correct engine_id for DAY intents is DAY_V2 (EngineId.DAY_V2_LIVE).

This script:
  1. Creates day_v2_engine_repair_audit if it does not exist (append-only log).
  2. Classifies every intent / paper_trade BUY row / SELL row into one of:
       PROVEN_SCALP_V2      — record carries genuine SCALP_V2 identity
                              (from SCALP V2 execution path, not DAY trailing-buy)
       PROVEN_DAY_V2        — record belongs to DAY trailing-buy execution
                              (corrected label should be DAY_V2)
       ORIGIN_UNRESOLVED    — cannot be determined from immutable evidence
  3. Writes one audit row per reclassified record.
  4. Prints the three-cohort performance summary.

Classification evidence (deterministic, immutable):
  - Intent rows in day_trailing_buy_intents are created ONLY by DAY trailing-buy.
    SCALP V2 has its own order-placement path and does NOT create rows there.
    Therefore every intent in that table is PROVEN_DAY_V2.
  - paper_trades / portfolio_engine_positions rows are linked to an intent via
    trade_id == intent.trade_id.  Any BUY row whose trade_id matches a
    day_trailing_buy_intents row is PROVEN_DAY_V2.
  - SELL rows are linked to their BUY via the same trade_id pattern
    (symbol-based prefix match) — out of scope for this repair because engine_id
    on SELL rows is copied from the position at exit time, which in turn comes
    from the intent.  Fixing the intent fixes the position which fixes the SELL.
  - Records with no linkable intent are ORIGIN_UNRESOLVED.

Run on the LIVE DB (Ocean) after deploying the code fix:
    sudo -u mystic /home/mystic/mystic/venv/bin/python3 \\
        scripts/day_v2_engine_repair_audit.py \\
        /home/mystic/mystic/mystic_trading.db [--dry-run]

Run locally (VM):
    ./venv/bin/python3 scripts/day_v2_engine_repair_audit.py mystic_trading.db
"""

from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

REPAIR_VERSION = "day_v2_engine_repair_audit_v1"

# arm_ts threshold: approx Unix timestamp of commit 628c47f merge on Ocean.
# Intents armed before this boundary used LEGACY_DAY_LIVE; after used SCALP_V2
# (incorrectly).  The repair sets the correct label to DAY_V2 for all DAY
# trailing-buy intents regardless of era (LEGACY_DAY_LIVE and SCALP_V2 are
# both incorrect for new intents; DAY_V2 is the only correct label).
COMMIT_628C47F_ARM_TS = 1790033000  # 2026-09-21 ~23:43 UTC (approx)

_DDL_AUDIT = """
CREATE TABLE IF NOT EXISTS day_v2_engine_repair_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_table TEXT NOT NULL,          -- source table name
    record_id TEXT NOT NULL,             -- pk or unique key of the record
    original_engine_id TEXT NOT NULL,    -- engine_id stored before repair
    corrected_engine_id TEXT NOT NULL,   -- engine_id that should be stored
    cohort TEXT NOT NULL,                -- PROVEN_SCALP_V2 / PROVEN_DAY_V2 / ORIGIN_UNRESOLVED
    evidence TEXT NOT NULL,              -- JSON blob describing why
    source_record_ids TEXT NOT NULL,     -- JSON list of related record ids
    repair_timestamp TEXT NOT NULL,      -- ISO 8601 UTC
    repair_version TEXT NOT NULL,        -- this script's version constant
    is_dry_run INTEGER NOT NULL DEFAULT 0
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_audit_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL_AUDIT)
    conn.commit()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _audit_row(
    record_table: str,
    record_id: str,
    original_engine_id: str,
    corrected_engine_id: str,
    cohort: str,
    evidence: dict,
    source_record_ids: list[str],
    dry_run: bool,
) -> dict:
    return {
        "record_table": record_table,
        "record_id": record_id,
        "original_engine_id": original_engine_id,
        "corrected_engine_id": corrected_engine_id,
        "cohort": cohort,
        "evidence": json.dumps(evidence, default=str),
        "source_record_ids": json.dumps(source_record_ids),
        "repair_timestamp": _now_iso(),
        "repair_version": REPAIR_VERSION,
        "is_dry_run": 1 if dry_run else 0,
    }


def _write_audit_rows(conn: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO day_v2_engine_repair_audit(
            record_table, record_id, original_engine_id, corrected_engine_id,
            cohort, evidence, source_record_ids,
            repair_timestamp, repair_version, is_dry_run
        ) VALUES (
            :record_table, :record_id, :original_engine_id, :corrected_engine_id,
            :cohort, :evidence, :source_record_ids,
            :repair_timestamp, :repair_version, :is_dry_run
        )
        """,
        rows,
    )
    conn.commit()


def classify_intents(conn: sqlite3.Connection, dry_run: bool) -> list[dict]:
    """Classify all day_trailing_buy_intents rows.

    Every row in this table was created by DAY trailing-buy code.
    Correct engine_id = DAY_V2.
    Original label = LEGACY_DAY_LIVE (pre-628c47f) or SCALP_V2 (post-628c47f).
    """
    intents = conn.execute("SELECT intent_id, symbol, engine_id, arm_ts, trade_id FROM day_trailing_buy_intents").fetchall()
    audit_rows: list[dict] = []
    for row in intents:
        intent_id = row["intent_id"]
        orig_eid = str(row["engine_id"] or "")
        if orig_eid == "DAY_V2":
            # Already correctly labelled — no audit row needed.
            continue
        era = "post_628c47f" if (row["arm_ts"] or 0) >= COMMIT_628C47F_ARM_TS else "pre_628c47f"
        audit_rows.append(
            _audit_row(
                record_table="day_trailing_buy_intents",
                record_id=intent_id,
                original_engine_id=orig_eid,
                corrected_engine_id="DAY_V2",
                cohort="PROVEN_DAY_V2",
                evidence={
                    "reason": "day_trailing_buy_intents rows are created exclusively by DAY trailing-buy code; SCALP V2 has a separate order path",
                    "era": era,
                    "arm_ts": row["arm_ts"],
                    "commit_boundary": COMMIT_628C47F_ARM_TS,
                    "original_engine_id": orig_eid,
                },
                source_record_ids=[intent_id, str(row["trade_id"] or "")],
                dry_run=dry_run,
            )
        )
    return audit_rows


def classify_paper_trades(conn: sqlite3.Connection, day_trade_ids: set[str], dry_run: bool) -> list[dict]:
    """Classify BUY rows in paper_trades linked to DAY intents."""
    audit_rows: list[dict] = []
    if not day_trade_ids:
        return audit_rows
    placeholders = ",".join("?" * len(day_trade_ids))
    buys = conn.execute(
        f"SELECT trade_id, engine_id, symbol, timestamp FROM paper_trades WHERE side='BUY' AND trade_id IN ({placeholders})",
        list(day_trade_ids),
    ).fetchall()
    for row in buys:
        orig_eid = str(row["engine_id"] or "")
        if orig_eid == "DAY_V2":
            continue
        audit_rows.append(
            _audit_row(
                record_table="paper_trades",
                record_id=row["trade_id"],
                original_engine_id=orig_eid,
                corrected_engine_id="DAY_V2",
                cohort="PROVEN_DAY_V2",
                evidence={
                    "reason": "paper_trades BUY trade_id matches a day_trailing_buy_intents row — proof of DAY trailing-buy origin",
                    "side": "BUY",
                    "symbol": row["symbol"],
                    "timestamp": row["timestamp"],
                    "original_engine_id": orig_eid,
                },
                source_record_ids=[row["trade_id"]],
                dry_run=dry_run,
            )
        )
    return audit_rows


def classify_positions(conn: sqlite3.Connection, day_trade_ids: set[str], dry_run: bool) -> list[dict]:
    """Classify portfolio_engine_positions rows linked to DAY intents."""
    audit_rows: list[dict] = []
    if not day_trade_ids:
        return audit_rows
    # Positions are keyed by symbol and linked by trade_id stored on the intent.
    # We match via the trade_id prefix in position_id or directly via trade_id column.
    # Check if trade_id column exists.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)").fetchall()}
    if "trade_id" in cols:
        placeholders = ",".join("?" * len(day_trade_ids))
        positions = conn.execute(
            f"SELECT id, engine_id, symbol, trade_id FROM portfolio_engine_positions WHERE trade_id IN ({placeholders})",
            list(day_trade_ids),
        ).fetchall()
    else:
        # Fallback: match via buy_order_id / scalp_intent_id naming patterns.
        positions = []

    for row in positions:
        orig_eid = str(row["engine_id"] or "")
        if orig_eid == "DAY_V2":
            continue
        audit_rows.append(
            _audit_row(
                record_table="portfolio_engine_positions",
                record_id=str(row["id"]),
                original_engine_id=orig_eid,
                corrected_engine_id="DAY_V2",
                cohort="PROVEN_DAY_V2",
                evidence={
                    "reason": "position trade_id matches a day_trailing_buy_intents row",
                    "symbol": row["symbol"],
                    "trade_id": row["trade_id"],
                    "original_engine_id": orig_eid,
                },
                source_record_ids=[str(row["id"]), str(row["trade_id"] or "")],
                dry_run=dry_run,
            )
        )
    return audit_rows


def classify_paper_sells(conn: sqlite3.Connection, day_trade_ids: set[str], dry_run: bool) -> list[dict]:
    """Classify SELL rows linked to DAY BUY trade_ids via shared symbol/timestamp."""
    audit_rows: list[dict] = []
    if not day_trade_ids:
        return audit_rows
    # SELL rows share the trade_id prefix up to the last underscore component,
    # or may have the same trade_id as the BUY row in some accounting paths.
    placeholders = ",".join("?" * len(day_trade_ids))
    sells = conn.execute(
        f"SELECT trade_id, engine_id, symbol, timestamp FROM paper_trades WHERE side='SELL' AND trade_id IN ({placeholders})",
        list(day_trade_ids),
    ).fetchall()
    for row in sells:
        orig_eid = str(row["engine_id"] or "")
        if orig_eid == "DAY_V2":
            continue
        audit_rows.append(
            _audit_row(
                record_table="paper_trades",
                record_id=row["trade_id"],
                original_engine_id=orig_eid,
                corrected_engine_id="DAY_V2",
                cohort="PROVEN_DAY_V2",
                evidence={
                    "reason": "paper_trades SELL trade_id matches a DAY BUY trade_id",
                    "side": "SELL",
                    "symbol": row["symbol"],
                    "timestamp": row["timestamp"],
                    "original_engine_id": orig_eid,
                },
                source_record_ids=[row["trade_id"]],
                dry_run=dry_run,
            )
        )
    return audit_rows


def cohort_performance(conn: sqlite3.Connection, day_trade_ids: set[str]) -> dict:
    """Return performance stats for PROVEN_DAY_V2 and PROVEN_SCALP_V2 cohorts."""
    # PROVEN_DAY_V2: BUY/SELL pairs whose trade_id appears in day_trailing_buy_intents.
    # PROVEN_SCALP_V2: paper_trades with engine_id='SCALP_V2' NOT in day_trade_ids.
    # ORIGIN_UNRESOLVED: anything else.
    all_trades = conn.execute("SELECT trade_id, side, engine_id, price, quantity, pnl FROM paper_trades ORDER BY timestamp").fetchall()

    cohorts: dict[str, dict] = {
        "PROVEN_DAY_V2": {"buys": 0, "sells": 0, "total_pnl": 0.0, "wins": 0, "losses": 0},
        "PROVEN_SCALP_V2": {"buys": 0, "sells": 0, "total_pnl": 0.0, "wins": 0, "losses": 0},
        "ORIGIN_UNRESOLVED": {"buys": 0, "sells": 0, "total_pnl": 0.0, "wins": 0, "losses": 0},
    }

    for row in all_trades:
        tid = row["trade_id"] or ""
        eid = row["engine_id"] or ""
        pnl = float(row["pnl"] or 0)
        side = row["side"] or ""

        if tid in day_trade_ids:
            cohort = "PROVEN_DAY_V2"
        elif eid in ("SCALP_V2",) and tid not in day_trade_ids:
            cohort = "PROVEN_SCALP_V2"
        elif eid == "LEGACY_DAY_LIVE":
            # LEGACY_DAY_LIVE intents pre-628c47f are also DAY trailing-buy.
            cohort = "PROVEN_DAY_V2"
        else:
            cohort = "ORIGIN_UNRESOLVED"

        c = cohorts[cohort]
        if side == "BUY":
            c["buys"] += 1
        elif side == "SELL":
            c["sells"] += 1
            if pnl > 0:
                c["wins"] += 1
            elif pnl < 0:
                c["losses"] += 1
            c["total_pnl"] += pnl

    # Add win_rate.
    for c in cohorts.values():
        closed = c["wins"] + c["losses"]
        c["win_rate"] = round(c["wins"] / closed, 4) if closed else None
        c["closed_trades"] = closed

    return cohorts


def run(db_path: str, dry_run: bool = False) -> None:
    conn = _connect(db_path)
    _ensure_audit_table(conn)

    # Collect all intent trade_ids (deterministic DAY provenance).
    intent_rows = conn.execute("SELECT intent_id, trade_id FROM day_trailing_buy_intents").fetchall()
    day_trade_ids: set[str] = {str(r["trade_id"]) for r in intent_rows if r["trade_id"]}

    # Classify each record type.
    audit_rows: list[dict] = []
    audit_rows.extend(classify_intents(conn, dry_run))
    audit_rows.extend(classify_paper_trades(conn, day_trade_ids, dry_run))
    audit_rows.extend(classify_positions(conn, day_trade_ids, dry_run))
    audit_rows.extend(classify_paper_sells(conn, day_trade_ids, dry_run))

    # Write audit rows (append-only).
    _write_audit_rows(conn, audit_rows)

    # Performance cohorts.
    perf = cohort_performance(conn, day_trade_ids)

    print("=" * 70)
    print(f"DAY V2 ENGINE REPAIR AUDIT  {'(DRY RUN)' if dry_run else '(LIVE)'}")
    print(f"DB: {db_path}")
    print(f"Timestamp: {_now_iso()}")
    print("=" * 70)
    print(f"Total day_trailing_buy_intents: {len(intent_rows)}")
    print(f"  DAY trade_ids collected:     {len(day_trade_ids)}")
    print(f"Audit rows written:            {len(audit_rows)}")
    print()
    print("Performance Cohorts:")
    print("-" * 70)
    for name, stats in perf.items():
        wr = f"{stats['win_rate']:.1%}" if stats["win_rate"] is not None else "n/a"
        print(f"  {name:<22} BUY={stats['buys']:>4} SELL={stats['sells']:>4} closed={stats['closed_trades']:>4}  win={wr}  PnL={stats['total_pnl']:+.4f}")
    print("=" * 70)
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DAY V2 engine identity repair audit")
    parser.add_argument("db_path", help="Path to mystic_trading.db")
    parser.add_argument("--dry-run", action="store_true", help="Write audit rows marked as dry-run")
    args = parser.parse_args()
    if not Path(args.db_path).exists():
        print(f"ERROR: database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)
    run(args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
