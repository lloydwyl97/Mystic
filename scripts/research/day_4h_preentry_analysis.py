#!/usr/bin/env python3
"""Offline 4H pre-entry reconstruction. Research only. No live gates."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_4h_entry_features import classify_feature_availability
from backend.services.day_4h_entry_research import analyze_ocean_66


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/day_4h_preentry_analysis.json")
    conn = sqlite3.connect(db)
    report = analyze_ocean_66(conn, allow_network=True)
    report["feature_availability"] = classify_feature_availability()
    rows = report.pop("rows")
    report["row_count"] = len(rows)
    out.write_text(json.dumps(report, default=str, indent=2))
    print(json.dumps({k: report[k] for k in ("n", "briefing53", "paired", "reconstructed_4h", "class_counts", "incremental")}, default=str, indent=2))
    print(f"wrote {out}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
