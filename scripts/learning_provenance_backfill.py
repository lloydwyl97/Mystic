#!/usr/bin/env python3
"""Backfill engine / strategy / setup / dust provenance on learning rows.

Touches only provenance:
  trade_learning_outcomes.extra_json   -> engine_id, trade_id, strategy, setup, is_dust, label_strategy
  ai_outcome_training_rows.strategy_id -> 'scalp' (SCALP_V2 fill) or 'dust' (DUST outcome)
Never price, quantity, fees, P&L or timestamps.

A learning row is linked to its SELL fill only when exactly one SELL on the same
symbol lies within --window-sec of the row's close time. Ambiguous or unmatched
rows are left unchanged (dust rows still get the dust flag).

Default is a dry run. --apply writes a JSON backup of every original value first.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.learning_provenance import DUST_LABEL, LEGACY_ENGINE, SCALP_ENGINE, UNKNOWN_SETUP, _lookup_setup, strategy_for_engine


def _sym(s: str) -> str:
    return str(s or "").upper().replace("/", "").replace("-", "")


def _epoch(ts: str) -> float | None:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def _load_sells(conn: sqlite3.Connection) -> dict[str, list[tuple[float, dict]]]:
    out: dict[str, list[tuple[float, dict]]] = {}
    for row in conn.execute("SELECT id, symbol, timestamp, engine_id, trade_id, scalp_opportunity_id FROM paper_trades WHERE upper(side)='SELL'"):
        ts = _epoch(row[2])
        if ts is None:
            continue
        out.setdefault(_sym(row[1]), []).append((ts, {"id": row[0], "engine_id": str(row[3] or "").upper() or LEGACY_ENGINE, "trade_id": str(row[4] or ""), "opp": str(row[5] or "")}))
    for rows in out.values():
        rows.sort(key=lambda r: r[0])
    return out


def _match(sells: dict[str, list[tuple[float, dict]]], symbol: str, ts: float | None, window: float) -> dict | None:
    if ts is None:
        return None
    rows = sells.get(_sym(symbol)) or []
    keys = [r[0] for r in rows]
    lo, hi = bisect.bisect_left(keys, ts - window), bisect.bisect_right(keys, ts + window)
    hits = rows[lo:hi]
    return hits[0][1] if len(hits) == 1 else None


def plan(db_path: str, window: float) -> dict:
    conn = sqlite3.connect(db_path, timeout=30)
    sells = _load_sells(conn)
    setup_cache: dict[tuple[str, str, str], str] = {}

    def setup_for(engine: str, trade_id: str, opp: str) -> str:
        key = (engine, trade_id, opp)
        if key not in setup_cache:
            setup_cache[key] = _lookup_setup(db_path, engine, trade_id=trade_id, opportunity_id=opp) or UNKNOWN_SETUP
        return setup_cache[key]

    tlo_updates: list[dict] = []
    counts = {"tlo_scanned": 0, "tlo_already_tagged": 0, "tlo_linked": 0, "tlo_dust_only": 0, "tlo_unmatched": 0, "tlo_scalp": 0, "tlo_day": 0}
    for rid, symbol, exit_ts, close_reason, extra_raw in conn.execute("SELECT id, symbol, exit_timestamp, close_reason, extra_json FROM trade_learning_outcomes"):
        counts["tlo_scanned"] += 1
        try:
            extra = json.loads(extra_raw) if extra_raw else {}
        except (TypeError, ValueError):
            continue
        if not isinstance(extra, dict) or "engine_id" in extra:
            counts["tlo_already_tagged"] += 1
            continue
        dust = str(close_reason or "").upper() == "DUST_WRITEOFF"
        hit = _match(sells, symbol, float(exit_ts) if exit_ts else None, window)
        new = dict(extra)
        if hit:
            strategy = strategy_for_engine(hit["engine_id"])
            new.update(
                {
                    "engine_id": hit["engine_id"],
                    "trade_id": hit["trade_id"],
                    "strategy": strategy,
                    "setup": setup_for(hit["engine_id"], hit["trade_id"], hit["opp"]),
                    "is_dust": dust,
                    "label_strategy": DUST_LABEL if dust else strategy,
                    "provenance_backfill": "sell_fill_match",
                }
            )
            counts["tlo_linked"] += 1
            counts["tlo_scalp" if hit["engine_id"] == SCALP_ENGINE else "tlo_day"] += 1
        elif dust:
            new.update({"is_dust": True, "label_strategy": DUST_LABEL, "provenance_backfill": "dust_close_reason"})
            counts["tlo_dust_only"] += 1
        else:
            counts["tlo_unmatched"] += 1
            continue
        tlo_updates.append({"id": rid, "old_extra_json": extra_raw, "new_extra_json": json.dumps(new)})

    aotr_updates: list[dict] = []
    counts.update({"aotr_scanned": 0, "aotr_to_dust": 0, "aotr_to_scalp": 0, "aotr_unmatched_or_day": 0})
    for rid, symbol, closed_at, outcome_class, strategy_id in conn.execute("SELECT id, symbol, closed_at_utc, outcome_class, strategy_id FROM ai_outcome_training_rows"):
        counts["aotr_scanned"] += 1
        if str(strategy_id or "") != "day":
            continue
        if str(outcome_class or "").upper() == "DUST":
            aotr_updates.append({"id": rid, "old_strategy_id": strategy_id, "new_strategy_id": DUST_LABEL})
            counts["aotr_to_dust"] += 1
            continue
        hit = _match(sells, symbol, _epoch(closed_at), window)
        if hit and hit["engine_id"] == SCALP_ENGINE:
            aotr_updates.append({"id": rid, "old_strategy_id": strategy_id, "new_strategy_id": "scalp"})
            counts["aotr_to_scalp"] += 1
        else:
            counts["aotr_unmatched_or_day"] += 1
    conn.close()
    return {"counts": counts, "tlo": tlo_updates, "aotr": aotr_updates}


def apply(db_path: str, result: dict, backup_path: Path) -> None:
    backup_path.write_text(json.dumps({"db": db_path, "created_at": time.time(), "trade_learning_outcomes": result["tlo"], "ai_outcome_training_rows": result["aotr"]}))
    conn = sqlite3.connect(db_path, timeout=60)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany("UPDATE trade_learning_outcomes SET extra_json=? WHERE id=?", [(u["new_extra_json"], u["id"]) for u in result["tlo"]])
        conn.executemany("UPDATE ai_outcome_training_rows SET strategy_id=? WHERE id=? AND strategy_id=?", [(u["new_strategy_id"], u["id"], u["old_strategy_id"]) for u in result["aotr"]])
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--window-sec", type=float, default=20.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default="")
    args = ap.parse_args()
    result = plan(args.db, args.window_sec)
    print(json.dumps(result["counts"], indent=2))
    if not args.apply:
        print("dry-run: no rows changed")
        return 0
    backup = Path(args.backup or f"learning_provenance_backup_{int(time.time())}.json")
    apply(args.db, result, backup)
    print(f"applied; backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
