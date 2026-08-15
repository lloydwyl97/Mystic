#!/usr/bin/env python3
"""Print canonical DAY and SCALP clean-sample stats. Do not invent cutoffs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.clean_acceptance import day_clean_rows, scalp_clean_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--day-db", default="/home/mystic/mystic/mystic_trading.db")
    p.add_argument("--scalp-db", default="/home/mystic/mystic/mystic_scalp.db")
    args = p.parse_args()
    out = {
        "day": day_clean_rows(args.day_db),
        "scalp": scalp_clean_rows(args.scalp_db),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
