#!/usr/bin/env python3
"""Print canonical DAY and SCALP clean-sample stats. Do not invent cutoffs.

Reports two books:
  clean_runtime      = valid post-cutoff strategy trades
  model_controlled   = only trades where the accepted predictor was the BUY authority
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.clean_acceptance import day_clean_rows, scalp_clean_rows


def _book(block: dict, key: str) -> dict:
    src = block.get(key) or block.get("clean") or {}
    return {
        "n": src.get("n", 0),
        "wins": src.get("wins", 0),
        "wr": src.get("wr"),
        "net": src.get("net"),
        "expectancy": src.get("expectancy"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--day-db", default="/home/mystic/mystic/mystic_trading.db")
    p.add_argument("--scalp-db", default="/home/mystic/mystic/mystic_scalp.db")
    args = p.parse_args()
    day = day_clean_rows(args.day_db)
    scalp = scalp_clean_rows(args.scalp_db)
    out = {
        "cutoff_utc": day.get("cutoff_utc"),
        "DAY_CLEAN_RUNTIME": _book(day, "clean_runtime"),
        "DAY_MODEL_CONTROLLED": _book(day, "model_controlled"),
        "SCALP_CLEAN_RUNTIME": _book(scalp, "clean_runtime"),
        "SCALP_MODEL_CONTROLLED": _book(scalp, "model_controlled"),
        "day": day,
        "scalp": scalp,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
