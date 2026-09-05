"""Offline clock-v2 collection, coverage, and readiness monitoring.

Does not rank, size, exit, or train. Feature snapshots never store outcomes.
Writing snapshots never sets lock inspected=true.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS
from backend.services.day_experiment_registry import record_planned_clock_v2
from backend.services.day_forward_lock import FORWARD_LOCK_START
from backend.services.day_forward_lock import TABLE as TABLE_LOCK
from backend.services.day_model_readiness import (
    CHRONOLOGICAL_BLOCK_HOURS,
    evaluate_readiness,
    format_snapshot,
)
from backend.services.day_path_clock_dataset import in_sealed_lock, load_asof_1m_bars, lock_window_status
from backend.services.day_path_clock_features import build_clock_features, normalize_bars, parse_as_of
from backend.services.day_path_clock_v2 import (
    REQUIRED_CLOCK_V2_FIELDS,
    SCHEMA_VERSION,
    planned_challenger_specification,
)
from backend.services.day_path_input_validity import MAX_GAP_SEC, MAX_LAST_BAR_AGE_SEC

TABLE_FEATURES = "day_path_clock_feature_snapshots"
TABLE_HISTORY = "day_path_clock_readiness_history"
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "production_exit_net_bps",
        "mfe_bps",
        "mae_bps",
        "markout_15m_net_bps",
        "markout_30m_net_bps",
        "markout_1h_net_bps",
        "markout_2h_net_bps",
        "markout_3h_net_bps",
        "markout_4h_net_bps",
        "clock_net_bps",
        "clock_gross_bps",
        "regret_vs_hold_bps",
    }
)

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_FEATURES} (
    decision_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decision_timestamp TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    eligible INTEGER,
    feature_json TEXT NOT NULL,
    quote_json TEXT,
    cost_json TEXT,
    quality_json TEXT,
    lock_window INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE TABLE IF NOT EXISTS {TABLE_HISTORY} (
    recorded_at TEXT PRIMARY KEY,
    ocean_sha TEXT,
    usable_groups INTEGER,
    complete_feature_groups INTEGER,
    mature_labels INTEGER,
    required_labels INTEGER,
    chronological_blocks INTEGER,
    required_blocks INTEGER,
    feature_coverage REAL,
    label_coverage REAL,
    data_readiness TEXT,
    failure_reasons TEXT
);
"""


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def ensure_research_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def snapshot_columns() -> tuple[str, ...]:
    return (
        "decision_group_id",
        "symbol",
        "created_at",
        "decision_timestamp",
        "feature_schema_version",
        "eligible",
        "feature_json",
        "quote_json",
        "cost_json",
        "quality_json",
        "lock_window",
    )


def bar_quality(bars: Any, *, as_of: Any) -> dict[str, Any]:
    when = parse_as_of(as_of)
    parsed = normalize_bars(bars)
    reasons: list[str] = []
    if when is None:
        return {"valid": False, "reasons": ["unparseable_as_of"]}
    future = [b for b in parsed if b.ts > when]
    if future:
        reasons.append("future_data")
    clipped = [b for b in parsed if b.ts <= when]
    stamps = [b.ts for b in clipped]
    if any(stamps[i] < stamps[i - 1] for i in range(1, len(stamps))):
        reasons.append("not_monotonic")
    if any(stamps[i] == stamps[i - 1] for i in range(1, len(stamps))):
        reasons.append("duplicate_timestamps")
    gaps = [(stamps[i] - stamps[i - 1]).total_seconds() for i in range(1, len(stamps))]
    if gaps and max(gaps) > MAX_GAP_SEC:
        reasons.append("gap_exceeded")
    if clipped and (when - clipped[-1].ts).total_seconds() > MAX_LAST_BAR_AGE_SEC:
        reasons.append("stale_last_bar")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "row_count": len(clipped),
        "max_gap_seconds": max(gaps) if gaps else None,
        "latest_bar_age_seconds": (when - clipped[-1].ts).total_seconds() if clipped else None,
    }


def missing_required_fields(feats: dict[str, Any], *, symbol: str) -> list[str]:
    if str(symbol).upper() == HOLD_SYMBOL:
        return [] if feats.get("symbol") == HOLD_SYMBOL else ["symbol"]
    missing = []
    for name in REQUIRED_CLOCK_V2_FIELDS:
        if feats.get(name) is None:
            missing.append(name)
    return missing


def _candidate_context(contract: dict[str, Any], cand_row: dict[str, Any], symbol: str) -> dict[str, Any]:
    fourh = {}
    group_tel = contract.get("4h_entry_telemetry") or {}
    if isinstance(group_tel, dict):
        fourh = dict(group_tel.get(symbol) or {})
    feats = cand_row.get("features") if isinstance(cand_row.get("features"), dict) else _loads(cand_row.get("feature_json"))
    if not fourh and isinstance(feats, dict):
        fourh = dict(feats.get("4h_entry_telemetry") or {})
    return {
        "p_buy": cand_row.get("p_buy"),
        "legacy_path_ev": cand_row.get("path_ev"),
        "final_rank_score": cand_row.get("final_rank_score"),
        "structure": fourh,
        "eligible": cand_row.get("eligible", 1) not in (0, False),
        "spread_bps": fourh.get("spread_bps") or contract.get("spread_bps"),
    }


def build_group_features(db_path: str | Path, *, group_id: str, created_at: str, contract: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    locked = in_sealed_lock(created_at)
    bars = {sym: load_asof_1m_bars(db_path, sym, created_at) for sym in COINS}
    btc_bars = bars.get("BTCUSDT") or []
    rows = []
    missing_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    complete = True
    for symbol in (*COINS, HOLD_SYMBOL):
        ctx = _candidate_context(contract, candidates.get(symbol) or {}, symbol)
        quality = bar_quality(bars.get(symbol) or [], as_of=created_at) if symbol != HOLD_SYMBOL else {"valid": True, "reasons": []}
        for reason in quality.get("reasons") or []:
            quality_counts[reason] += 1
        feats = build_clock_features(
            bars.get(symbol) or [],
            as_of=created_at,
            symbol=symbol,
            btc_bars=btc_bars,
            p_buy=ctx.get("p_buy"),
            legacy_path_ev=0.0 if symbol == HOLD_SYMBOL else ctx.get("legacy_path_ev"),
            final_rank_score=0.0 if symbol == HOLD_SYMBOL else ctx.get("final_rank_score"),
            structure=ctx.get("structure"),
            quote_spread_bps=ctx.get("spread_bps"),
        )
        missing = missing_required_fields(feats, symbol=symbol)
        for name in missing:
            missing_counts[name] += 1
        if missing:
            complete = False
        snapshot = {
            "decision_group_id": group_id,
            "symbol": symbol,
            "created_at": created_at,
            "decision_timestamp": created_at,
            "feature_schema_version": SCHEMA_VERSION,
            "eligible": bool(ctx.get("eligible", True)),
            "features": feats,
            "quotes": {"spread_bps": feats.get("spread_bps")},
            "costs": {
                "estimated_all_in_cost_bps": feats.get("estimated_all_in_cost_bps"),
                "commission_rt_bps": feats.get("commission_rt_bps"),
                "expected_slippage_bps": feats.get("expected_slippage_bps"),
            },
            "quality": quality,
            "missing_fields": missing,
            "lock_window": locked,
        }
        if FORBIDDEN_OUTCOME_KEYS.intersection(snapshot) or FORBIDDEN_OUTCOME_KEYS.intersection(feats):
            raise RuntimeError("feature snapshot leaked an outcome key")
        rows.append(snapshot)
    status = "complete" if complete else ("unusable" if len(rows) < 5 else "partial")
    return {
        "decision_group_id": group_id,
        "decision_timestamp": created_at,
        "lock_window": locked,
        "status": status,
        "candidates": rows,
        "missing_field_counts": dict(missing_counts),
        "quality_reason_counts": dict(quality_counts),
    }


def load_forward_groups(db_path: str | Path, *, start: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        groups = [
            dict(r)
            for r in conn.execute(
                f"SELECT decision_group_id, created_at, selected_action, selected_symbol, contract_json FROM {TABLE_GROUPS} WHERE created_at>=? ORDER BY created_at",
                (start,),
            )
        ]
        cands: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in conn.execute(f"SELECT decision_group_id, symbol, eligible, p_buy, path_ev, final_rank_score, feature_json FROM {TABLE_CANDIDATES}"):
            item = dict(row)
            item["features"] = _loads(item.pop("feature_json"))
            cands[str(row["decision_group_id"])][str(row["symbol"])] = item
    finally:
        conn.close()
    for group in groups:
        group["contract"] = _loads(group.pop("contract_json"))
        group["candidates"] = cands.get(str(group["decision_group_id"]), {})
    return groups


def audit_clock_coverage(db_path: str | Path, *, start: str) -> dict[str, Any]:
    groups = load_forward_groups(db_path, start=start)
    built = []
    for group in groups:
        built.append(
            build_group_features(
                db_path,
                group_id=str(group["decision_group_id"]),
                created_at=str(group["created_at"]),
                contract=group.get("contract") or {},
                candidates=group.get("candidates") or {},
            )
        )
    field_present: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    missing_reasons: Counter[str] = Counter()
    quality_reasons: Counter[str] = Counter()
    complete = partial = unusable = 0
    for item in built:
        status = item["status"]
        if status == "complete":
            complete += 1
        elif status == "partial":
            partial += 1
        else:
            unusable += 1
        missing_reasons.update(item.get("missing_field_counts") or {})
        quality_reasons.update(item.get("quality_reason_counts") or {})
        for cand in item["candidates"]:
            if cand["symbol"] == HOLD_SYMBOL:
                continue
            feats = cand["features"]
            for name in REQUIRED_CLOCK_V2_FIELDS:
                field_total[name] += 1
                if feats.get(name) is not None:
                    field_present[name] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "effective_feature_start": start,
        "groups_total": len(built),
        "groups_complete": complete,
        "groups_partial": partial,
        "groups_unusable": unusable,
        "independent_decision_groups": len(built),
        "candidate_coin_rows": len(built) * 4,
        "hold_rows": len(built),
        "field_coverage": {
            name: {
                "present": field_present[name],
                "total": field_total[name],
                "rate": (field_present[name] / field_total[name] if field_total[name] else None),
            }
            for name in REQUIRED_CLOCK_V2_FIELDS
        },
        "missing_field_reasons": dict(missing_reasons),
        "invalid_group_quality_reasons": dict(quality_reasons),
        "zero_imputed": False,
        "snapshots": built,
    }


def persist_feature_snapshots(research_db: str | Path, coverage: dict[str, Any]) -> int:
    ensure_research_schema(research_db)
    conn = sqlite3.connect(str(research_db), timeout=30)
    written = 0
    try:
        for group in coverage.get("snapshots") or []:
            for cand in group.get("candidates") or []:
                feat = dict(cand.get("features") or {})
                if FORBIDDEN_OUTCOME_KEYS.intersection(feat):
                    raise RuntimeError("refusing to persist outcome fields")
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {TABLE_FEATURES}(
                        decision_group_id, symbol, created_at, decision_timestamp,
                        feature_schema_version, eligible, feature_json, quote_json,
                        cost_json, quality_json, lock_window
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        cand["decision_group_id"],
                        cand["symbol"],
                        cand["created_at"],
                        cand["decision_timestamp"],
                        SCHEMA_VERSION,
                        1 if cand.get("eligible") else 0,
                        json.dumps(feat, default=str),
                        json.dumps(cand.get("quotes") or {}, default=str),
                        json.dumps(cand.get("costs") or {}, default=str),
                        json.dumps({"quality": cand.get("quality"), "missing_fields": cand.get("missing_fields")}, default=str),
                        1 if cand.get("lock_window") else 0,
                    ),
                )
                written += 1
        conn.commit()
    finally:
        conn.close()
    return written


def lock_inspected(db_path: str | Path) -> bool:
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        row = conn.execute(f"SELECT inspected FROM {TABLE_LOCK} ORDER BY created_at").fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return any(bool(r[0]) for r in row)


def append_readiness_history(research_db: str | Path, row: dict[str, Any]) -> None:
    ensure_research_schema(research_db)
    conn = sqlite3.connect(str(research_db), timeout=30)
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {TABLE_HISTORY}(
                recorded_at, ocean_sha, usable_groups, complete_feature_groups,
                mature_labels, required_labels, chronological_blocks, required_blocks,
                feature_coverage, label_coverage, data_readiness, failure_reasons
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.get("recorded_at") or datetime.now(timezone.utc).isoformat(),
                row.get("ocean_sha"),
                row.get("usable_groups"),
                row.get("complete_feature_groups"),
                row.get("mature_labels"),
                row.get("required_labels"),
                row.get("chronological_blocks"),
                row.get("required_blocks"),
                json.dumps(row.get("feature_coverage"), default=str) if isinstance(row.get("feature_coverage"), (dict, list)) else row.get("feature_coverage"),
                json.dumps(row.get("label_coverage"), default=str) if isinstance(row.get("label_coverage"), (dict, list)) else row.get("label_coverage"),
                row.get("data_readiness"),
                json.dumps(row.get("failure_reasons") or [], default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def chronological_blocks(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}
    for group in groups:
        created = parse_as_of(group.get("created_at"))
        if created is None:
            continue
        key = int(created.timestamp() // (CHRONOLOGICAL_BLOCK_HOURS * 3600))
        bucket = buckets.setdefault(
            key,
            {
                "block_id": key,
                "start": datetime.fromtimestamp(key * CHRONOLOGICAL_BLOCK_HOURS * 3600, tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp((key + 1) * CHRONOLOGICAL_BLOCK_HOURS * 3600, tz=timezone.utc).isoformat(),
                "groups": 0,
                "selected_actions": 0,
                "hold_actions": 0,
            },
        )
        bucket["groups"] += 1
        action = str(group.get("selected_action") or "")
        if action.upper() == "HOLD":
            bucket["hold_actions"] += 1
        elif action.upper().startswith("BUY"):
            bucket["selected_actions"] += 1
    return [buckets[k] for k in sorted(buckets)]


def run_pipeline(
    db_path: str | Path,
    *,
    research_db: str | Path,
    ocean_sha: str = "",
    cutoff: str = FORWARD_LOCK_START,
) -> dict[str, Any]:
    if lock_inspected(db_path):
        raise RuntimeError("refusing pipeline: lock inspected=true")
    inspected_before = lock_inspected(db_path)
    report = evaluate_readiness(db_path, cutoff=cutoff)
    span = (report.get("checks") or {}).get("G_forward_span") or {}
    start = str(span.get("effective_window_start") or cutoff)
    coverage = audit_clock_coverage(db_path, start=start)
    persist_feature_snapshots(research_db, coverage)
    record_planned_clock_v2(research_db)
    if lock_inspected(db_path) != inspected_before:
        raise RuntimeError("pipeline must not flip lock inspected")
    history = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "ocean_sha": ocean_sha,
        "usable_groups": span.get("decision_groups"),
        "complete_feature_groups": coverage["groups_complete"],
        "mature_labels": span.get("mature_authoritative_trade_labels"),
        "required_labels": span.get("required_mature_trade_labels"),
        "chronological_blocks": span.get("chronological_blocks"),
        "required_blocks": span.get("required_chronological_blocks"),
        "feature_coverage": ((report.get("checks") or {}).get("D_feature_coverage") or {}).get("coverage"),
        "label_coverage": ((report.get("checks") or {}).get("C_label_maturity") or {}).get("group_label_coverage"),
        "data_readiness": "PASS" if report.get("ready") else "FAIL",
        "failure_reasons": report.get("reasons_not_ready") or [],
    }
    append_readiness_history(research_db, history)
    report["clock_v2_progress"] = {
        "groups_total": coverage["groups_total"],
        "groups_complete": coverage["groups_complete"],
        "groups_partial": coverage["groups_partial"],
        "groups_unusable": coverage["groups_unusable"],
        "independent_decision_groups": coverage["independent_decision_groups"],
        "candidate_coin_rows": coverage["candidate_coin_rows"],
    }
    return {
        "readiness": report,
        "coverage": {k: v for k, v in coverage.items() if k != "snapshots"},
        "history": history,
        "lock": lock_window_status(db_path),
        "blocks": chronological_blocks(load_forward_groups(db_path, start=start)),
        "planned": planned_challenger_specification(),
        "lock_inspected_after": lock_inspected(db_path),
        "snapshot": format_snapshot(report),
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clock-v2 research pipeline (offline)")
    parser.add_argument("--db", default=os.getenv("MYSTIC_DB_PATH", "mystic_trading.db"))
    parser.add_argument("--research-db", default="day_path_clock_research.db")
    parser.add_argument("--ocean-sha", default="")
    parser.add_argument("--cutoff", default=FORWARD_LOCK_START)
    args = parser.parse_args()
    out = run_pipeline(args.db, research_db=args.research_db, ocean_sha=args.ocean_sha, cutoff=args.cutoff)
    print(out["snapshot"])
    print("\nCLOCK-V2 COVERAGE")
    print(json.dumps(out["coverage"], indent=2, default=str))
    print("\nCHRONOLOGICAL BLOCKS")
    print(json.dumps(out["blocks"], indent=2, default=str))


if __name__ == "__main__":
    _cli()
