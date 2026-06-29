#!/usr/bin/env python3
"""Dump ai_strategy_score_weights buckets (regime::SETUP aware)."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables


def dump_buckets(db_path: str = DATABASE_PATH, symbol: str | None = None) -> dict:
    ensure_ai_canonical_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if symbol:
            sym = symbol.upper().replace("/", "")
            rows = conn.execute(
                """
                SELECT symbol, regime, component_name, weight, previous_weight, sample_count, updated_at
                FROM ai_strategy_score_weights
                WHERE UPPER(REPLACE(symbol,'/','')) = UPPER(?)
                ORDER BY regime, component_name
                """,
                (sym,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT symbol, regime, component_name, weight, previous_weight, sample_count, updated_at
                FROM ai_strategy_score_weights
                ORDER BY symbol, regime, component_name
                """
            ).fetchall()

    buckets: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        sym = str(row["symbol"])
        regime = str(row["regime"])
        buckets.setdefault(sym, {}).setdefault(regime, []).append(
            {
                "component": row["component_name"],
                "weight": row["weight"],
                "previous_weight": row["previous_weight"],
                "sample_count": row["sample_count"],
                "updated_at": row["updated_at"],
            }
        )

    return {
        "db_path": db_path,
        "symbol_filter": symbol,
        "bucket_count": sum(len(v) for b in buckets.values() for v in b.values()),
        "symbols": sorted(buckets.keys()),
        "buckets": buckets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump DAY adaptive learning buckets")
    parser.add_argument("--symbol", help="Filter to one symbol (e.g. SOLUSDT)")
    parser.add_argument("--db", default=DATABASE_PATH)
    args = parser.parse_args()
    payload = dump_buckets(args.db, args.symbol)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
