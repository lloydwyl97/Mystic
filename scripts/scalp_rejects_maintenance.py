#!/usr/bin/env python3
"""Prune scalp_rejects and optionally VACUUM (offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.sqlite_large_table_retention import (
    run_large_table_retention,
    run_offline_vacuum_and_integrity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scalp rejects retention + optional offline VACUUM")
    parser.add_argument("--vacuum", action="store_true", help="VACUUM after delete (stop Mystic first)")
    parser.add_argument("--unlimited", action="store_true", help="Delete until caught up")
    args = parser.parse_args()

    cfg = get_scalp_config()
    db = Path(cfg.database_path)
    before = db.stat().st_size if db.is_file() else 0

    import sqlite3

    with sqlite3.connect(db) as conn:
        before_rows = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]

    out = run_large_table_retention(
        db,
        unlimited=args.unlimited,
        batch_size=5000,
        max_batches_per_table=5000 if args.unlimited else 200,
        max_run_seconds=600.0 if args.unlimited else 60.0,
    )

    with sqlite3.connect(db) as conn:
        after_rows = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]

    after = db.stat().st_size if db.is_file() else 0
    report = {
        "database": str(db),
        "rows_before": before_rows,
        "rows_after": after_rows,
        "rows_deleted": max(0, before_rows - after_rows),
        "size_before_mb": round(before / 1024 / 1024, 2),
        "size_after_mb": round(after / 1024 / 1024, 2),
        "retention": out,
    }

    if args.vacuum:
        vac = run_offline_vacuum_and_integrity(db)
        report["vacuum"] = vac
        report["size_after_vacuum_mb"] = round(db.stat().st_size / 1024 / 1024, 2)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
