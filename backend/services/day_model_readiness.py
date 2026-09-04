"""Offline data-readiness gate for DAY ranking-model research.

Answers one question: is the stored learning data trustworthy enough that a challenger
trained on it would mean something? It reads the decision ledger, the outcome labels and
the accounting tables, and it never reads or writes ranking, sizing, exits, the order path
or the book. Nothing here is imported by a trading code path.

The gate deliberately reports *why* it is not ready rather than a date to wait until. Sample
requirements are derived from the size of the predeclared challenger feature set, not chosen
by hand, so they move with the model rather than with the calendar.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_decision_label_contract import TABLE_LABELS
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS
from backend.services.day_experiment_registry import TABLE as TABLE_REGISTRY
from backend.services.day_forward_lock import FORWARD_LOCK_START, challenger_export_schema
from backend.services.day_forward_lock import TABLE as TABLE_LOCK

# Engineering thresholds. Conservative and fixed in advance; never tuned to make a run pass.
MIN_MATURE_LABEL_COVERAGE = 0.95
MIN_FEATURE_COVERAGE = 0.95
MIN_EVENTS_PER_FEATURE = 10
MIN_CHRONOLOGICAL_BLOCKS = 5
CHRONOLOGICAL_BLOCK_HOURS = 24
LABEL_RECONCILE_TOLERANCE_BPS = 0.5
MATURITY_HORIZON_SEC = 4 * 3600
HISTORICAL_66_WINDOW = ("2026-08-25T00:00:00+00:00", "2026-09-02T00:00:00+00:00")
BUY_TRADE_ID_RE = re.compile(r"buy_trade_id=([^;]+)")


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"check": name, "pass": bool(passed), **detail}


def decision_role(group: dict[str, Any]) -> str:
    """Mirror of the scorecard's role split: ranking a coin is not the same as trading it."""
    if not str(group.get("selected_action") or "").upper().startswith("BUY"):
        return "HOLD"
    if group.get("execute_authorized") == 1 and str(group.get("fill_trade_id") or "").strip():
        return "traded"
    if group.get("execute_authorized") == 0:
        return "blocked_after_ranking"
    return "ranking_only"


def load_state(db_path: str | Path) -> dict[str, Any]:
    """Read every table the gate needs in one pass. Read-only; opens the DB in ro mode."""
    uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    out: dict[str, Any] = {"groups": {}, "labels": defaultdict(dict), "candidates": defaultdict(dict)}
    try:
        for row in conn.execute(
            "SELECT decision_group_id, created_at, selected_action, selected_symbol, contract_json, "
            f"execute_authorized, fill_trade_id, lifecycle_state, feature_artifact_ref FROM {TABLE_GROUPS}"
        ):
            item = dict(row)
            item["contract"] = _loads(item.pop("contract_json"))
            out["groups"][str(row["decision_group_id"])] = item
        for row in conn.execute(f"SELECT * FROM {TABLE_LABELS}"):
            item = dict(row)
            item["label"] = _loads(item.pop("label_json"))
            out["labels"][str(row["decision_group_id"])][str(row["symbol"])] = item
        for row in conn.execute(
            f"SELECT decision_group_id, symbol, p_buy, path_ev, final_rank_score, feature_json FROM {TABLE_CANDIDATES}"
        ):
            item = dict(row)
            item["features"] = _loads(item.pop("feature_json"))
            out["candidates"][str(row["decision_group_id"])][str(row["symbol"])] = item
        out["lock"] = [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE_LOCK}")]
        out["registry"] = [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE_REGISTRY}")]
        out["closes"] = [dict(r) for r in conn.execute("SELECT * FROM position_close_ledger")]
        out["fifo"] = [
            dict(r)
            for r in conn.execute(
                "SELECT symbol, trade_id, quantity, price, remaining_position FROM paper_trades "
                "WHERE side='BUY' AND COALESCE(remaining_position,0) > 0"
            )
        ]
        out["book"] = [dict(r) for r in conn.execute("SELECT symbol, quantity, entry_price, status FROM portfolio_engine_positions")]
        ledger = conn.execute("SELECT total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        out["equity"] = float(ledger["total_equity"]) if ledger else None
    finally:
        conn.close()
    return out


# --------------------------------------------------------------------------------------
# A. production label integrity
# --------------------------------------------------------------------------------------
def check_production_label_integrity(state: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    now = now if now is not None else time.time()
    matured_traded = 0
    joined = 0
    missing: list[str] = []
    fabricated: list[str] = []
    for gid, group in state["groups"].items():
        created = _parse_iso(group.get("created_at"))
        role = decision_role(group)
        lab = state["labels"].get(gid, {}).get(str(group.get("selected_symbol") or ""))
        if lab and str(lab.get("provenance")) == "authoritative" and role not in ("traded", "HOLD"):
            fabricated.append(gid)
        if role != "traded" or created is None:
            continue
        if created.timestamp() + MATURITY_HORIZON_SEC > now:
            continue
        matured_traded += 1
        if lab and str(lab.get("provenance")) == "authoritative":
            joined += 1
        else:
            missing.append(gid)
    rate = (joined / matured_traded) if matured_traded else None
    return _check(
        "A_production_label_integrity",
        bool(matured_traded and not fabricated and rate is not None and rate >= MIN_MATURE_LABEL_COVERAGE),
        {
            "matured_traded_groups": matured_traded,
            "authoritative_joins": joined,
            "join_rate": rate,
            "required_join_rate": MIN_MATURE_LABEL_COVERAGE,
            "unjoined_groups": missing[:10],
            "authoritative_without_fill": fabricated[:10],
        },
    )


# --------------------------------------------------------------------------------------
# B. counterfactual integrity
# --------------------------------------------------------------------------------------
def check_counterfactual_integrity(state: dict[str, Any]) -> dict[str, Any]:
    """A ranking loser must never carry the residue of a fill it never had."""
    violations: dict[str, int] = defaultdict(int)
    examples: list[str] = []
    counterfactuals = 0
    for gid, group in state["groups"].items():
        role = decision_role(group)
        selected = str(group.get("selected_symbol") or "")
        for symbol, lab in state["labels"].get(gid, {}).items():
            if symbol == HOLD_SYMBOL:
                continue
            is_real_fill = role == "traded" and symbol == selected
            payload = lab.get("label") or {}
            if is_real_fill:
                if payload.get("counterfactual") is True and str(lab.get("provenance")) == "authoritative":
                    violations["authoritative_fill_flagged_counterfactual"] += 1
                continue
            counterfactuals += 1
            if lab.get("exit_reason"):
                violations["counterfactual_has_exit_reason"] += 1
                examples.append(f"{gid}:{symbol}")
            if lab.get("production_exit_net_bps") is not None:
                violations["counterfactual_has_production_net"] += 1
            if _num(lab.get("commission_bps")):
                violations["counterfactual_has_commission"] += 1
            if payload.get("counterfactual") is not True:
                violations["not_flagged_counterfactual"] += 1
            if str(lab.get("provenance")) not in ("reconstructed", "estimated"):
                violations["counterfactual_bad_provenance"] += 1
    return _check(
        "B_counterfactual_integrity",
        not violations,
        {
            "counterfactual_labels": counterfactuals,
            "violations": dict(violations),
            "examples": examples[:10],
        },
    )


# --------------------------------------------------------------------------------------
# C. label maturity
# --------------------------------------------------------------------------------------
def check_label_maturity(state: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    now = now if now is not None else time.time()
    horizons = {"15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
    mature = dict.fromkeys(horizons, 0)
    present = dict.fromkeys(horizons, 0)
    matured_groups = 0
    labeled_groups = 0
    for gid, group in state["groups"].items():
        created = _parse_iso(group.get("created_at"))
        if created is None:
            continue
        age = now - created.timestamp()
        if age >= MATURITY_HORIZON_SEC:
            matured_groups += 1
            if gid in state["labels"]:
                labeled_groups += 1
        for name, secs in horizons.items():
            if age < secs:
                continue
            for symbol in COINS:
                lab = state["labels"].get(gid, {}).get(symbol)
                if not lab:
                    continue
                mature[name] += 1
                if (lab.get("label") or {}).get("markouts", {}).get(name) is not None:
                    present[name] += 1
    coverage = (labeled_groups / matured_groups) if matured_groups else None
    return _check(
        "C_label_maturity",
        bool(coverage is not None and coverage >= MIN_MATURE_LABEL_COVERAGE),
        {
            "matured_groups": matured_groups,
            "labeled_groups": labeled_groups,
            "group_label_coverage": coverage,
            "required_coverage": MIN_MATURE_LABEL_COVERAGE,
            "markout_coverage": {k: (present[k] / mature[k] if mature[k] else None) for k in horizons},
        },
    )


# --------------------------------------------------------------------------------------
# D. feature coverage
# --------------------------------------------------------------------------------------
def _candidate_vector(group: dict[str, Any], candidates: dict[str, Any], symbol: str) -> dict[str, Any]:
    """The challenger's inputs for one coin, gathered from wherever production stores them."""
    contract = {str(c.get("symbol")): c for c in (group.get("contract") or {}).get("candidates") or []}.get(symbol) or {}
    row = candidates.get(symbol) or {}
    telemetry = (row.get("features") or {}).get("4h_entry_telemetry") or contract.get("4h_entry_telemetry") or {}
    return {
        "p_buy": row.get("p_buy"),
        "path_ev": row.get("path_ev"),
        "final_rank_score": row.get("final_rank_score"),
        "4h_telemetry": telemetry.get("distance_to_4h_break_bps"),
        "spread_bps": contract.get("spread_bps"),
        "slippage_model": contract.get("expected_slippage") if contract.get("expected_slippage") is not None else contract.get("predicted_impact"),
        "quote_timestamp": contract.get("quote_timestamp"),
        "eligible": contract.get("eligible"),
    }


def feature_availability_start(state: dict[str, Any]) -> str | None:
    """First decision after which every later decision carries the whole challenger vector.

    The 4H entry telemetry was switched on mid-stream, so the lock's calendar cutoff can be
    earlier than the point where the feature set actually exists. Training that reached back
    past this boundary would be training on rows the model could never have scored.
    """
    ordered = sorted(state["groups"].values(), key=lambda g: str(g.get("created_at") or ""))
    complete: list[bool] = []
    for group in ordered:
        gid = str(group["decision_group_id"])
        rows = state["candidates"].get(gid, {})
        vectors = [_candidate_vector(group, rows, s) for s in COINS]
        usable = [v for v in vectors if v.get("eligible") is not False]
        complete.append(bool(usable) and all(v["4h_telemetry"] is not None and v["spread_bps"] is not None for v in usable))
    start = None
    for index in range(len(ordered) - 1, -1, -1):
        if not complete[index]:
            break
        start = str(ordered[index].get("created_at"))
    return start


def check_feature_coverage(state: dict[str, Any], *, cutoff: str = FORWARD_LOCK_START) -> dict[str, Any]:
    """Coverage of the predeclared challenger inputs, measured where a challenger would read them.

    Only eligible candidates in the forward window count. An excluded coin has no `p_buy` by
    design, and pre-cutoff groups predate the telemetry, so scoring either as missing data
    would understate coverage rather than reveal a gap. Quote-derived costs live on the
    group contract's candidate list, not on the candidate row's feature blob.
    """
    available_from = feature_availability_start(state)
    effective = max([t for t in (cutoff, available_from) if t], default=cutoff)
    start = _parse_iso(effective)
    fields = ("p_buy", "path_ev", "final_rank_score", "4h_telemetry", "spread_bps", "slippage_model", "quote_timestamp")
    counts: dict[str, int] = defaultdict(int)
    total = 0
    ineligible = 0
    for gid, group in state["groups"].items():
        created = _parse_iso(group.get("created_at"))
        if created is None or (start and created < start):
            continue
        rows = state["candidates"].get(gid, {})
        for symbol in COINS:
            if symbol not in rows:
                continue
            vector = _candidate_vector(group, rows, symbol)
            if vector.get("eligible") is False:
                ineligible += 1
                continue
            total += 1
            for field in fields:
                if vector.get(field) is not None:
                    counts[field] += 1
    coverage = {k: (counts[k] / total if total else None) for k in fields}
    blocking = {k: v for k, v in coverage.items() if v is None or v < MIN_FEATURE_COVERAGE}
    return _check(
        "D_feature_coverage",
        bool(total and not blocking),
        {
            "lock_cutoff": cutoff,
            "feature_available_from": available_from,
            "effective_window_start": effective,
            "eligible_candidate_rows": total,
            "excluded_candidate_rows": ineligible,
            "coverage": coverage,
            "required_coverage": MIN_FEATURE_COVERAGE,
            "below_threshold": blocking,
        },
    )


# --------------------------------------------------------------------------------------
# E. time authority
# --------------------------------------------------------------------------------------
def check_time_authority(state: dict[str, Any]) -> dict[str, Any]:
    """Two properties the data itself must show: UTC-stable parsing and horizon isolation."""
    from backend.services.day_4h_label_runner import parse_epoch_utc

    sample = "2026-09-02 03:00:11.000000"
    original = os.environ.get("TZ")
    readings: list[float | None] = []
    try:
        for zone in ("UTC", "America/Chicago", "Asia/Tokyo"):
            os.environ["TZ"] = zone
            time.tzset()
            readings.append(parse_epoch_utc(sample))
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()
    tz_stable = len({r for r in readings if r is not None}) == 1 and None not in readings

    # Horizon isolation, judged against the clock rather than the recorded cutoff: a markout at
    # horizon h may exist only once the group is at least h old. The recorded
    # `market_data_cutoff` is separately reported because labels written before it was widened
    # describe the lifecycle window instead of the furthest bar consumed.
    horizon_secs = {"15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
    now_epoch = time.time()
    premature = 0
    checked = 0
    stale_cutoff = 0
    for gid, group in state["groups"].items():
        created = _parse_iso(group.get("created_at"))
        if created is None:
            continue
        for lab in state["labels"].get(gid, {}).values():
            payload = lab.get("label") or {}
            markouts = payload.get("markouts") or {}
            cutoff_dt = _parse_iso(payload.get("market_data_cutoff"))
            furthest = 0.0
            for name, secs in horizon_secs.items():
                if markouts.get(name) is None:
                    continue
                checked += 1
                furthest = max(furthest, float(secs))
                if created.timestamp() + secs > now_epoch + 1.0:
                    premature += 1
            if cutoff_dt is not None and furthest and created.timestamp() + furthest > cutoff_dt.timestamp() + 1.0:
                stale_cutoff += 1
    return _check(
        "E_time_authority",
        bool(tz_stable and premature == 0),
        {
            "timezone_stable": tz_stable,
            "timezone_readings": readings,
            "markout_horizons_checked": checked,
            "premature_markouts": premature,
            "labels_with_pre_widening_cutoff_metadata": stale_cutoff,
        },
    )


# --------------------------------------------------------------------------------------
# F. accounting
# --------------------------------------------------------------------------------------
def is_residual_writeoff(reason: Any, realized_profit: Any, entry_price: Any, quantity: Any) -> bool:
    if str(reason or "").upper() == "DUST_WRITEOFF":
        return True
    profit, price, qty = _num(realized_profit), _num(entry_price), _num(quantity)
    if profit is None or price is None or qty is None:
        return False
    notional = price * qty
    return notional > 0 and profit <= -0.99 * notional


def fifo_residual_report(state: dict[str, Any]) -> dict[str, Any]:
    """Quantify FIFO lots the engine book does not carry, per symbol and against equity."""
    book: dict[str, float] = defaultdict(float)
    for row in state["book"]:
        book[str(row["symbol"])] += float(row.get("quantity") or 0.0)
    fifo: dict[str, float] = defaultdict(float)
    price: dict[str, float] = {}
    lots: dict[str, int] = defaultdict(int)
    for row in state["fifo"]:
        symbol = str(row["symbol"])
        fifo[symbol] += float(row.get("remaining_position") or 0.0)
        lots[symbol] += 1
        price.setdefault(symbol, float(row.get("price") or 0.0))
    equity = state.get("equity")
    per_symbol = {}
    total_usd = 0.0
    for symbol in sorted(set(fifo) | set(book)):
        delta = fifo.get(symbol, 0.0) - book.get(symbol, 0.0)
        usd = delta * price.get(symbol, 0.0)
        total_usd += max(usd, 0.0)
        per_symbol[symbol] = {
            "fifo_remaining_qty": fifo.get(symbol, 0.0),
            "book_qty": book.get(symbol, 0.0),
            "unreconciled_qty": delta,
            "unreconciled_usd": usd,
            "lots": lots.get(symbol, 0),
        }
    return {
        "per_symbol": per_symbol,
        "total_unreconciled_usd": total_usd,
        "equity": equity,
        "bps_of_equity": (total_usd / equity * 10000.0) if equity else None,
    }


def check_accounting(state: dict[str, Any]) -> dict[str, Any]:
    """Labels must reproduce the production close ledger exactly.

    The gate scores reconciliation, not the size of the FIFO residue. A residue of stranded
    coin fragments is reported for operators but does not by itself corrupt a label, because
    labels are built from the close ledger's economic rows and never from FIFO remainders.
    """
    by_buy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state["closes"]:
        match = BUY_TRADE_ID_RE.search(str(row.get("detail") or ""))
        if match:
            by_buy[match.group(1).strip()].append(row)
    worst = 0.0
    compared = 0
    mismatches: list[str] = []
    for gid, group in state["groups"].items():
        if decision_role(group) != "traded":
            continue
        selected = str(group.get("selected_symbol") or "")
        lab = state["labels"].get(gid, {}).get(selected)
        if not lab or str(lab.get("provenance")) != "authoritative":
            continue
        econ = [
            r
            for r in by_buy.get(str(group.get("fill_trade_id") or "").strip(), [])
            if not is_residual_writeoff(r.get("close_reason"), r.get("realized_profit"), r.get("entry_price"), r.get("quantity"))
        ]
        if not econ:
            continue
        pnl = sum(float(r.get("realized_profit") or 0.0) for r in econ)
        qty = sum(float(r.get("quantity") or 0.0) for r in econ)
        entry = float(econ[0].get("entry_price") or 0.0)
        if not qty or not entry:
            continue
        recomputed = pnl / (qty * entry) * 10000.0
        stored = _num(lab.get("production_exit_net_bps"))
        if stored is None:
            continue
        compared += 1
        drift = abs(recomputed - stored)
        worst = max(worst, drift)
        if drift > LABEL_RECONCILE_TOLERANCE_BPS:
            mismatches.append(f"{gid}:{drift:.4f}bps")
    residual = fifo_residual_report(state)
    return _check(
        "F_accounting",
        bool(compared and not mismatches),
        {
            "labels_reconciled": compared,
            "max_abs_drift_bps": worst,
            "tolerance_bps": LABEL_RECONCILE_TOLERANCE_BPS,
            "mismatches": mismatches[:10],
            "fifo_residual": residual,
        },
    )


# --------------------------------------------------------------------------------------
# G. forward chronological span
# --------------------------------------------------------------------------------------
def check_forward_span(state: dict[str, Any], *, cutoff: str = FORWARD_LOCK_START, now: float | None = None) -> dict[str, Any]:
    """Usable forward evidence, counted from the later of the lock cutoff and feature availability."""
    now = now if now is not None else time.time()
    available_from = feature_availability_start(state)
    effective = max([t for t in (cutoff, available_from) if t], default=cutoff)
    start = _parse_iso(effective)
    features = len(challenger_export_schema().get("inputs") or [])
    required_trades = features * MIN_EVENTS_PER_FEATURE
    groups = 0
    trades = 0
    mature_labels = 0
    blocks: set[int] = set()
    first: datetime | None = None
    last: datetime | None = None
    per_symbol: dict[str, int] = defaultdict(int)
    holds = 0
    for gid, group in state["groups"].items():
        created = _parse_iso(group.get("created_at"))
        if created is None or (start and created < start):
            continue
        groups += 1
        first = created if first is None or created < first else first
        last = created if last is None or created > last else last
        blocks.add(int(created.timestamp() // (CHRONOLOGICAL_BLOCK_HOURS * 3600)))
        role = decision_role(group)
        if role == "HOLD":
            holds += 1
        if role != "traded":
            continue
        trades += 1
        per_symbol[str(group.get("selected_symbol"))] += 1
        lab = state["labels"].get(gid, {}).get(str(group.get("selected_symbol") or ""))
        if lab and str(lab.get("provenance")) == "authoritative":
            mature_labels += 1
    days = ((last - first).total_seconds() / 86400.0) if (first and last) else 0.0
    ok = mature_labels >= required_trades and len(blocks) >= MIN_CHRONOLOGICAL_BLOCKS
    return _check(
        "G_forward_span",
        ok,
        {
            "lock_cutoff": cutoff,
            "feature_available_from": available_from,
            "effective_window_start": effective,
            "calendar_days": round(days, 2),
            "decision_groups": groups,
            "selected_trades": trades,
            "HOLD_groups": holds,
            "mature_authoritative_trade_labels": mature_labels,
            "challenger_feature_count": features,
            "events_per_feature_required": MIN_EVENTS_PER_FEATURE,
            "required_mature_trade_labels": required_trades,
            "chronological_blocks": len(blocks),
            "required_chronological_blocks": MIN_CHRONOLOGICAL_BLOCKS,
            "per_symbol_trades": dict(per_symbol),
        },
    )


# --------------------------------------------------------------------------------------
# H / I. lock protection and experiment registry
# --------------------------------------------------------------------------------------
def check_locked_test_protection(state: dict[str, Any]) -> dict[str, Any]:
    locks = state.get("lock") or []
    if not locks:
        return _check("H_locked_test_protection", False, {"locks": 0, "reason": "no forward lock registered"})
    latest = locks[-1]
    meta = _loads(latest.get("meta_json"))
    inspected = bool(latest.get("inspected"))
    excluded = bool(meta.get("historical_66_excluded"))
    return _check(
        "H_locked_test_protection",
        bool(not inspected and excluded),
        {
            "locks": len(locks),
            "experiment_id": latest.get("experiment_id"),
            "dataset_cutoff": latest.get("dataset_cutoff"),
            "inspected": inspected,
            "historical_66_excluded": excluded,
            "historical_66_window": meta.get("historical_66_window") or list(HISTORICAL_66_WINDOW),
        },
    )


def check_experiment_registry(state: dict[str, Any], *, minimum_arms: int = 12) -> dict[str, Any]:
    rows = state.get("registry") or []
    promoted = sum(1 for r in rows if r.get("promoted"))
    return _check(
        "I_experiment_registry",
        len(rows) >= minimum_arms,
        {
            "recorded_arms": len(rows),
            "minimum_expected_arms": minimum_arms,
            "promoted": promoted,
        },
    )


# --------------------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------------------
def evaluate_readiness(db_path: str | Path, *, cutoff: str = FORWARD_LOCK_START, now: float | None = None) -> dict[str, Any]:
    """Run every check and return ready/not-ready with the reasons attached."""
    try:
        state = load_state(db_path)
    except sqlite3.Error as exc:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ready": False,
            "reasons_not_ready": [f"database unreadable: {exc}"],
            "checks": {},
        }
    checks = [
        check_production_label_integrity(state, now=now),
        check_counterfactual_integrity(state),
        check_label_maturity(state, now=now),
        check_feature_coverage(state, cutoff=cutoff),
        check_time_authority(state),
        check_accounting(state),
        check_forward_span(state, cutoff=cutoff, now=now),
        check_locked_test_protection(state),
        check_experiment_registry(state),
    ]
    by_name = {c["check"]: c for c in checks}
    failed = [c["check"] for c in checks if not c["pass"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": not failed,
        "reasons_not_ready": failed,
        "checks": by_name,
        "sample_support": sample_support(state, cutoff=cutoff, now=now),
    }


def sample_support(state: dict[str, Any], *, cutoff: str = FORWARD_LOCK_START, now: float | None = None) -> dict[str, Any]:
    """Evidence available to a first challenger. The decision group is the primary unit.

    Candidate rows are five per group and share one market snapshot, so they are reported
    separately and must not be mistaken for independent observations.
    """
    now = now if now is not None else time.time()
    available_from = feature_availability_start(state)
    effective = max([t for t in (cutoff, available_from) if t], default=cutoff)
    start = _parse_iso(effective)
    groups = [g for g in state["groups"].values() if (_parse_iso(g.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)) >= (start or datetime.min.replace(tzinfo=timezone.utc))]
    matured = [g for g in groups if (_parse_iso(g.get("created_at")) or datetime.now(timezone.utc)).timestamp() + MATURITY_HORIZON_SEC <= now]
    traded = [g for g in matured if decision_role(g) == "traded"]
    labels = state["labels"]
    mature_candidates = sum(len(labels.get(str(g["decision_group_id"]), {})) for g in matured)
    features = len(challenger_export_schema().get("inputs") or [])
    per_symbol: dict[str, int] = defaultdict(int)
    for g in traded:
        per_symbol[str(g.get("selected_symbol"))] += 1
    blocks = {int((_parse_iso(g.get("created_at")) or datetime.now(timezone.utc)).timestamp() // (CHRONOLOGICAL_BLOCK_HOURS * 3600)) for g in matured}
    return {
        "effective_window_start": effective,
        "mature_decision_groups": len(matured),
        "mature_selected_trades": len(traded),
        "mature_candidate_labels": mature_candidates,
        "challenger_feature_count": features,
        "events_per_candidate_feature": round(len(traded) / features, 3) if features else None,
        "chronological_folds_possible": max(0, len(blocks) - 1),
        "per_symbol_trade_support": dict(per_symbol),
        "HOLD_support": sum(1 for g in matured if decision_role(g) == "HOLD"),
        "primary_unit": "decision_group",
        "note": "candidate rows are 5x groups and share one snapshot; they are not independent",
    }


ACCEPTANCE_STANDARD = (
    "net-positive after genuine Binance.US costs",
    "profit factor above 1",
    "beats the current production champion",
    "better in a majority of chronological validation folds",
    "positive on the new untouched chronological lock",
    "not materially worse in maximum drawdown",
    "robust under conservative spread and slippage",
    "free of leakage",
    "beats a HOLD-aware baseline, where HOLD scores exactly 0",
)


def acceptance_standard() -> dict[str, Any]:
    """The bar a future challenger must clear. Documentation only; nothing here trains or scores.

    Recorded next to the gate so the standard cannot drift between the run that produces a
    candidate and the run that judges it. An oracle cannot qualify, because every criterion is
    evaluated on data the model never saw.
    """
    return {
        "criteria": list(ACCEPTANCE_STANDARD),
        "champion": "production entry behaviour at f942fea (unchanged)",
        "hold_value_bps": 0.0,
        "actions": [*COINS, HOLD_SYMBOL],
        "target": "expected executable net bps after costs",
        "disqualifiers": [
            "hindsight or MFE-credit labels as the training target",
            "thresholds tuned on the locked test period",
            "permanent HOLD or trade-opinion permission behaviour",
            "any artifact that merely loses less than the champion",
        ],
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DAY model data-readiness gate (offline, read-only)")
    parser.add_argument("--db", default=os.getenv("MYSTIC_DB_PATH", "mystic_trading.db"))
    parser.add_argument("--cutoff", default=FORWARD_LOCK_START)
    args = parser.parse_args()
    print(json.dumps(evaluate_readiness(args.db, cutoff=args.cutoff), indent=2, default=str))


if __name__ == "__main__":
    _cli()


__all__ = [
    "ACCEPTANCE_STANDARD",
    "CHRONOLOGICAL_BLOCK_HOURS",
    "LABEL_RECONCILE_TOLERANCE_BPS",
    "MIN_CHRONOLOGICAL_BLOCKS",
    "MIN_EVENTS_PER_FEATURE",
    "MIN_FEATURE_COVERAGE",
    "MIN_MATURE_LABEL_COVERAGE",
    "acceptance_standard",
    "check_accounting",
    "check_counterfactual_integrity",
    "check_experiment_registry",
    "check_feature_coverage",
    "check_forward_span",
    "check_label_maturity",
    "check_locked_test_protection",
    "check_production_label_integrity",
    "check_time_authority",
    "decision_role",
    "evaluate_readiness",
    "feature_availability_start",
    "fifo_residual_report",
    "is_residual_writeoff",
    "load_state",
    "sample_support",
]
