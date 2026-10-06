#!/usr/bin/env python3
"""Seed the learned policy gap (realized net minus the fill's own market label)
from current-version closes entered at or after the engine's economic anchor.

Dry run by default; --apply writes. Refuses once any policy_gap state exists for
the engine, and a candidate row already learned is never learned twice.

  scripts/seed_policy_gap.py --db mystic_trading.db --engine DAY_V2 [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.adaptive_learning import seed_policy_gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--engine", required=True, choices=("DAY_V2", "SCALP_V2"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    out = seed_policy_gap(args.db, args.engine, apply=args.apply)
    print(json.dumps(out, default=str))
    return 1 if out["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
