#!/usr/bin/env python3
"""Rebuild continuation rows from current-version history and optionally install them.

Report mode is the default. ``--apply`` replaces only hold_remaining rows for the
current economic version. It does not rewrite trades, fills, or entry metrics.
"""

from __future__ import annotations

import argparse
import json
import tempfile

from backend.services.continuation_backfill import build_observations, install_continuation, walk_forward, write_observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = walk_forward(args.db)
    installed = None
    if args.apply:
        observations, inventory = build_observations(args.db)
        if not observations:
            raise SystemExit("no continuation observations; refusing to replace state")
        with tempfile.TemporaryDirectory() as tmp:
            state = f"{tmp}/continuation.db"
            write_observations(state, observations)
            installed = install_continuation(args.db, state, inventory)
    print(json.dumps({"report": report, "installed": installed}, default=str))


if __name__ == "__main__":
    main()
