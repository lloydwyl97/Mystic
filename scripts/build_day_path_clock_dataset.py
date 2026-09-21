#!/usr/bin/env python3
"""Build / validate the clock-consistent DAY path research dataset.

Offline only. Does not train. Does not change live ranking.
Refuses to attach outcomes for sealed-lock groups.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_4h_entry_features import COINS
from backend.services.day_forward_lock import FORWARD_LOCK_START
from backend.services.day_model_readiness import evaluate_readiness, format_snapshot
from backend.services.day_path_clock_compare import compare_legacy_vs_clock, refuse_legacy_coefficients_on_clock_features
from backend.services.day_path_clock_dataset import (
    assert_group_integrity,
    build_group_record,
    dataset_counts,
    load_asof_1m_bars,
    lock_window_status,
)
from backend.services.day_path_clock_v2 import SCHEMA_VERSION, clock_challenger_export_schema, feature_contract, future_acceptance_bar


def _load_prelock_groups(db_path: str, cutoff: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT decision_group_id, created_at
            FROM day_decision_group_records
            WHERE created_at < ?
            ORDER BY created_at
            """,
            (cutoff,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [(str(a), str(b)) for a, b in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="DAY path clock-v2 research dataset (offline)")
    parser.add_argument("--db", default="mystic_trading.db")
    parser.add_argument("--cutoff", default=FORWARD_LOCK_START)
    parser.add_argument("--out", default="")
    parser.add_argument("--limit-groups", type=int, default=0)
    args = parser.parse_args()
    db = str(Path(args.db))
    lock = lock_window_status(db)
    print("LOCK", json.dumps(lock, default=str))
    if lock.get("inspected"):
        print("REFUSING: lock inspected=true; do not build outcome-bearing research exports")
        return 2
    if not lock.get("historical_66_excluded"):
        print("REFUSING: historical_66_excluded is not true")
        return 2

    groups = []
    samples = []
    for gid, created in _load_prelock_groups(db, args.cutoff):
        bars = {sym: load_asof_1m_bars(db, sym, created) for sym in COINS}
        record = build_group_record(
            decision_group_id=gid,
            decision_ts=created,
            bars_by_symbol=bars,
            lock_cutoff=args.cutoff,
            attach_labels=True,
        )
        assert_group_integrity(record)
        groups.append(record)
        for cand in record["candidates"]:
            if cand["symbol"] == "HOLD":
                continue
            samples.append(
                {
                    "as_of": created,
                    "symbol": cand["symbol"],
                    "bars": bars.get(cand["symbol"]) or [],
                    "btc_bars": bars.get("BTCUSDT") or [],
                    "legacy_btc_ret_5": 0.0,
                }
            )
        if args.limit_groups and len(groups) >= args.limit_groups:
            break

    comparison = compare_legacy_vs_clock(samples)
    readiness = evaluate_readiness(db, cutoff=args.cutoff)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_contract": feature_contract(),
        "challenger": clock_challenger_export_schema(),
        "acceptance": future_acceptance_bar(),
        "lock": lock,
        "counts": dataset_counts(groups),
        "legacy_vs_clock": comparison,
        "legacy_coefficients_on_clock_inputs": refuse_legacy_coefficients_on_clock_features(),
        "readiness_ready": bool(readiness.get("ready")),
        "readiness_reasons": readiness.get("reasons_not_ready"),
        "train": False,
        "groups": groups,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print("WROTE", args.out)
    print("COUNTS", json.dumps(payload["counts"], default=str))
    print("COMPARE", json.dumps(comparison, default=str))
    print(format_snapshot(readiness))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
