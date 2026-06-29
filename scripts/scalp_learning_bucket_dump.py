#!/usr/bin/env python3
"""Dump scalp_strategy_score_weights buckets (micro_regime::SETUP)."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.scalp_strategy_score_weight_writer import ensure_scalp_strategy_score_weights_table

TABLE = "scalp_strategy_score_weights"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol")
    args = parser.parse_args()
    db = get_scalp_config().database_path
    ensure_scalp_strategy_score_weights_table(db)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        if args.symbol:
            sym = args.symbol.upper().replace("/", "")
            rows = conn.execute(
                f"SELECT symbol, regime, component_name, weight, sample_count, updated_at FROM {TABLE} WHERE symbol=? ORDER BY regime, component_name",
                (sym,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT symbol, regime, component_name, weight, sample_count, updated_at FROM {TABLE} ORDER BY symbol, regime, component_name"
            ).fetchall()
    buckets: dict[str, dict[str, list]] = {}
    for row in rows:
        sym = row["symbol"]
        reg = row["regime"]
        buckets.setdefault(sym, {}).setdefault(reg, []).append(
            {
                "component": row["component_name"],
                "weight": row["weight"],
                "sample_count": row["sample_count"],
                "updated_at": row["updated_at"],
            }
        )
    payload = {"bucket_count": len(rows), "buckets": buckets, "strategy_id": "scalp", "isolated_from_day": True}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
