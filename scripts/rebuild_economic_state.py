#!/usr/bin/env python3
"""Rebuild the current economic version's adaptive state from current-version evidence.

Run with the portfolio engine stopped. Dry run unless --apply.
  scripts/rebuild_economic_state.py --db mystic_trading.db [--engine all|scalp] [--apply]
--engine scalp rebuilds SCALP's version only and leaves DAY untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.economic_state_rebuild import rebuild, rebuild_scalp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--engine", choices=("all", "scalp"), default="all")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--now", type=float, default=None)
    args = parser.parse_args()
    run = rebuild_scalp if args.engine == "scalp" else rebuild
    result = run(args.db, now=args.now, apply=args.apply)
    print(json.dumps(result, indent=2, default=str))
    return 1 if result.get("refused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
