#!/usr/bin/env python3
"""Deterministic causal replay: committed baseline learner vs the current tree.

Read-only against --db. Usage:
  scripts/economic_replay.py --db mystic_trading.db --baseline-sha 02d4fd6 [--out replay.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.economic_replay import run_replay


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--now", type=float, default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = _clean(run_replay(args.db, baseline_sha=args.baseline_sha, repo=ROOT, now=args.now, folds=args.folds))
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    summary = {name: {"day": p["day"]["all"], "day_by_setup": p["day"]["by_setup"], "scalp": {k: v for k, v in p["scalp"].items() if k != "folds"}} for name, p in result["policies"].items()}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
