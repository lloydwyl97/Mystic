#!/usr/bin/env python3
"""Rebuild frozen 1d/1w-derived dims of stored DAY vectors (dry run unless --apply).

Fetches completed Binance.US 1d/1w klines deep enough for the DAY bundle depth
at the frozen boundary, then rewrites only ai_inference_log.features_json/ctx_json
and ai_outcome_training_rows.features_json/context_json. --apply requires an
existing database backup and stopped services. Raw trade/accounting tables are
fingerprinted before and after and must be identical.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.canonical_candle_intervals import CANONICAL_SYMBOLS, interval_ms
from backend.services.day_htf_feature_rebuild import DEPTH_1D, DEPTH_1W, DEPTH_4H, _iso_to_ms, rebuild_contaminated_rows

RAW_TABLES = ("paper_trades", "portfolio_engine_ledger", "portfolio_engine_positions", "position_close_ledger", "trade_learning_outcomes", "live_order_ledger")
TRAINING_RAW_COLUMNS_EXCLUDED = ("features_json", "context_json")


def _fingerprint(conn: sqlite3.Connection) -> dict[str, str]:
    out: dict[str, str] = {}
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in RAW_TABLES:
        if table not in names:
            continue
        h = hashlib.sha256()
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid"):
            h.update(repr(row).encode())
        out[table] = h.hexdigest()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)") if r[1] not in TRAINING_RAW_COLUMNS_EXCLUDED]
    h = hashlib.sha256()
    for row in conn.execute(f"SELECT {','.join(cols)} FROM ai_outcome_training_rows ORDER BY id"):
        h.update(repr(row).encode())
    out["ai_outcome_training_rows(non-feature columns)"] = h.hexdigest()
    return out


async def _completed_klines(symbol: str, interval: str, start_ms: int) -> list[list]:
    from backend.services.canonical_candle_pipeline import CanonicalCandlePipeline

    pipe = CanonicalCandlePipeline()
    width = interval_ms(interval)
    now_ms = int(time.time() * 1000)
    rows: list[list] = []
    cursor = start_ms
    while cursor < now_ms:
        page = await pipe.fetch_binance(symbol, interval, start_ms=cursor, end_ms=min(now_ms, cursor + width * 999), limit=1000)
        if not page:
            break
        for k in page:
            if int(k[0]) + width <= now_ms:
                rows.append([int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
        nxt = int(page[-1][0]) + width
        if nxt <= cursor:
            break
        cursor = nxt
    dedup = {r[0]: r for r in rows}
    return [dedup[k] for k in sorted(dedup)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/mystic/mystic/mystic_trading.db")
    ap.add_argument("--boundary-utc", default="2026-09-18T00:35:00+00:00")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default="", help="existing backup of --db; required with --apply")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    boundary_ms = _iso_to_ms(args.boundary_utc)
    if boundary_ms is None:
        print("bad --boundary-utc", file=sys.stderr)
        return 2
    if args.apply and (not args.backup or not Path(args.backup).is_file() or Path(args.backup).stat().st_size == 0):
        print("--apply requires an existing non-empty --backup", file=sys.stderr)
        return 2

    start_1d = boundary_ms - (DEPTH_1D + 3) * interval_ms("1d")
    start_1w = boundary_ms - (DEPTH_1W + 3) * interval_ms("1w")
    # Rows built before 4h lost authority sit in the first day after the boundary.
    start_4h = boundary_ms - (DEPTH_4H + 3) * interval_ms("4h")

    async def _load() -> tuple[dict[str, list[list]], dict[str, list[list]], dict[str, list[list]]]:
        d1: dict[str, list[list]] = {}
        w1: dict[str, list[list]] = {}
        h4: dict[str, list[list]] = {}
        for sym in CANONICAL_SYMBOLS:
            d1[sym] = await _completed_klines(sym, "1d", start_1d)
            w1[sym] = await _completed_klines(sym, "1w", start_1w)
            h4[sym] = await _completed_klines(sym, "4h", start_4h)
        return d1, w1, h4

    bars_1d, bars_1w, bars_4h = asyncio.run(_load())
    uri = args.db if args.apply else f"file:{args.db}?mode=ro"
    conn = sqlite3.connect(uri, uri=not args.apply, timeout=60)
    try:
        before = _fingerprint(conn)
        report = rebuild_contaminated_rows(conn, bars_1d=bars_1d, bars_1w=bars_1w, bars_4h=bars_4h, boundary_ms=boundary_ms, apply=args.apply)
        after = _fingerprint(conn)
    finally:
        conn.close()
    report["bars"] = {s: {"1d": len(bars_1d[s]), "1w": len(bars_1w[s])} for s in bars_1d}
    report["raw_tables_unchanged"] = before == after
    report["raw_fingerprints"] = sorted(before)
    report["mode"] = "apply" if args.apply else "dry_run"
    text = json.dumps(report, indent=2, default=str)
    if args.report:
        Path(args.report).write_text(text)
    summary = {k: v for k, v in report.items() if k != "training"}
    summary["training"] = {k: v for k, v in report.get("training", {}).items() if k != "ids"}
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report["raw_tables_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
