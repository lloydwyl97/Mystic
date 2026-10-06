#!/usr/bin/env python3
"""Compare the installed continuation with the hold-advantage surface.

Report mode is the default. ``--install`` copies the advantage surface only for
engines whose out-of-sample net beats the installed continuation without a
material drawdown increase. It does not rebuild the pairwise remaining target
and it does not rewrite trades, fills, or entry metrics.
"""

from __future__ import annotations

import argparse
import json
import tempfile

from backend.services.continuation_backfill import (
    DAY_ENGINE,
    SCALP_ENGINE,
    build_advantage_labels,
    continuation_accepts,
    continuation_predicts,
    remember_labels,
    train_surface,
    walk_advantage,
    walk_forward,
    write_surface,
)
from backend.services.continuation_surface import install_surface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--as-of", type=float, default=None)
    parser.add_argument("--install", nargs="*", choices=(DAY_ENGINE, SCALP_ENGINE))
    parser.add_argument("--how", choices=("best", "blended"))
    parser.add_argument("--apply", action="store_true", help="removed; the pairwise remaining target is not installed")
    args = parser.parse_args()
    if args.apply:
        raise SystemExit("refusing to install the pairwise remaining target")
    legacy = walk_forward(args.db, as_of=args.as_of)
    repaired = walk_advantage(args.db, as_of=args.as_of)
    acceptance: dict[str, dict[str, bool]] = {}
    for engine in (DAY_ENGINE, SCALP_ENGINE):
        legacy_oos = (legacy.get("engines") or {}).get(engine, {}).get("oos_learned") or {}
        block = (repaired.get("engines") or {}).get(engine) or {}
        predicts = continuation_predicts(block.get("prediction"))
        acceptance[engine] = {how: bool(predicts and continuation_accepts((block.get(how) or {}).get("oos") or {}, legacy_oos)) for how in ("best", "blended")}
    installed = None
    if args.install is not None:
        if not args.how:
            raise SystemExit("--how is required with --install")
        chosen = [engine for engine in args.install if acceptance[engine][args.how]]
        refused = [engine for engine in args.install if engine not in chosen]
        if refused or not chosen:
            raise SystemExit(f"refusing advantage install; accepted={chosen} refused={refused}")
        labels, inventory = build_advantage_labels(args.db, as_of=args.as_of)
        with tempfile.TemporaryDirectory() as tmp:
            state = f"{tmp}/surface.db"
            write_surface(state, train_surface(labels))
            installed = install_surface(args.db, state, inventory, how=args.how, engines=tuple(chosen))
        remember_labels(args.db, [item for item in labels if item.engine in chosen])
    print(json.dumps({"legacy": legacy, "advantage": repaired, "acceptance": acceptance, "installed": installed}, default=str))


if __name__ == "__main__":
    main()
