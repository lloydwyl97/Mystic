#!/usr/bin/env python3
"""Read-only Binance.US scalp readiness snapshot — no writes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.status_snapshot import build_scalp_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Scalp readiness status (read-only)")
    parser.add_argument(
        "--warm-rounds",
        type=int,
        default=0,
        help="Momentum warm rounds before snapshot (5s each; 12+ for 60s trend; 0=instant)",
    )
    parser.add_argument("--pretty", action="store_true", help="Indented JSON output")
    args = parser.parse_args()

    status = build_scalp_status(warm_rounds=max(0, args.warm_rounds))
    if args.pretty:
        print(json.dumps(status, indent=2, default=str))
    else:
        print(json.dumps(status, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
