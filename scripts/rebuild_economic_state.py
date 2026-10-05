#!/usr/bin/env python3
"""Rebuild the current economic version's adaptive state from current-version evidence.

Run with the portfolio engine stopped. Dry run unless --apply.
  scripts/rebuild_economic_state.py --db mystic_trading.db [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.economic_state_rebuild import rebuild


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--now", type=float, default=None)
    args = parser.parse_args()
    result = rebuild(args.db, now=args.now, apply=args.apply)
    print(json.dumps(result, indent=2, default=str))
    return 1 if result.get("refused") else 0


if __name__ == "__main__":
    raise SystemExit(main())
