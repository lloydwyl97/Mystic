"""Read-only DAY decision-intelligence report. Does not trade, push, or restart."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.database_schema import DATABASE_PATH
from backend.services.day_decision_intelligence import run_decision_intelligence


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DATABASE_PATH
    conn = sqlite3.connect(path)
    try:
        report = run_decision_intelligence(conn, mode="paper", run_challengers=False)
    finally:
        conn.close()
    print(json.dumps(report, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
