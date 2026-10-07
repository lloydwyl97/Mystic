#!/usr/bin/env python3
"""Regenerate DAY / SCALP derived economic state under the current exit contract.

Replays the recorded candidate rows, realized closes and 1m bars through the live
learners (``economic_state_rebuild.regenerate``), then rebuilds ``hold_remaining``
for engines whose continuation authority it is
(``continuation_backfill.rebuild_hold_remaining``). Back up the adaptive and
continuation tables and stop the portfolio engine first. Dry run unless --apply.
  scripts/regenerate_policy_state.py --db mystic_trading.db [--engine all|day|scalp] [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.continuation_backfill import rebuild_hold_remaining
from backend.services.economic_state_rebuild import regenerate

ENGINES = {"day": ("DAY_V2",), "scalp": ("SCALP_V2",), "all": ("DAY_V2", "SCALP_V2")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--engine", choices=tuple(ENGINES), default="all")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--now", type=float, default=None)
    args = parser.parse_args()
    now = float(args.now if args.now is not None else time.time())
    out = {}
    for engine in ENGINES[args.engine]:
        out[engine] = {
            "economic": regenerate(args.db, engine, now=now, apply=args.apply),
            "hold_remaining": rebuild_hold_remaining(args.db, engine, now=now, apply=args.apply),
        }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
