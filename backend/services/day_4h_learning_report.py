"""CLI: offline 4H entry learning report. No trading side effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.database_schema import DATABASE_PATH
from backend.services.day_4h_entry_scorecard import build_scorecard
from backend.services.day_experiment_registry import registry, seed_historical
from backend.services.day_forward_lock import register_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DAY 4H entry learning report (offline)")
    parser.add_argument("--window", default="24h", choices=("24h", "7d", "30d"))
    parser.add_argument("--db", default=str(DATABASE_PATH))
    args = parser.parse_args(argv)
    db = Path(args.db)
    seed_historical(db)
    register_lock(db)
    report = build_scorecard(db, window=args.window)
    report["experiment_registry"] = {"arm_count": registry(db)["arm_count"], "promoted_count": 0}
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
