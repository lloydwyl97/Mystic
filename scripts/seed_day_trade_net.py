#!/usr/bin/env python3
"""Seed DAY_V2 realized-net learning (``trade_net``) from current-version closes
entered at or after the economic anchor.

Dry run by default; --apply writes. Refuses once any DAY trade_net state exists,
so a close already learned live is never counted twice. Nothing else is modified.

  scripts/seed_day_trade_net.py --db mystic_trading.db --entered-since 2026-10-05T02:02:29 [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.adaptive_learning import seed_day_trade_net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--entered-since", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    out = seed_day_trade_net(args.db, args.entered_since, apply=args.apply)
    print(json.dumps(out, default=str))
    return 1 if out["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
