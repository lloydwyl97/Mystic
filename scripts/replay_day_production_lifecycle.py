#!/usr/bin/env python3
"""Run the production-faithful DAY lifecycle replay on a Mystic SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.day_production_lifecycle_replay import run_all_arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "mystic_trading.db"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    try:
        report = run_all_arms(conn)
    finally:
        conn.close()
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
