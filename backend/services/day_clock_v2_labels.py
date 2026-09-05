"""CLOCK-V2 fixed-horizon target labels. Research only.

Writes the 3h executable-net target for every production-available action in a
v5 DEVELOPMENT decision group, plus HOLD = 0. Separate table from
``day_decision_outcome_labels`` so the generic 4H-entry lock and its labels are
neither read nor modified.

The generic labeler emits 15m / 30m / 1h / 2h / 4h markouts and has no 3h
horizon at all, so the frozen v5 target had no label source. This module
supplies it using the existing clock labeler, whose horizon table already
includes 3h.

Never inspects the 4H lock. Never trains. Never touches trading.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.execution_cost_model import (
    expected_exchange_commission_rt_pct,
    expected_slippage_rt_pct,
    honest_all_in_rt_pct,
)
from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_clock_v2_partition import DEVELOPMENT, partition_for
from backend.services.day_path_clock_dataset import load_asof_1m_bars
from backend.services.day_path_clock_features import clip_asof, normalize_bars, parse_as_of
from backend.services.day_path_clock_labels import build_clock_labels
from backend.services.day_path_clock_v2 import (
    EXECUTABLE_PRICE_METHOD,
    PRIMARY_TARGET,
    PRIMARY_TARGET_HORIZON_NAME,
    PRIMARY_TARGET_HORIZON_SEC,
)
from backend.services.day_path_clock_v2_capture import TABLE_ARTIFACT

logger = logging.getLogger(__name__)

TABLE_V5_LABELS = "day_clock_v2_outcome_labels"
LABEL_CONTRACT_VERSION = "day_clock_v2_target_3h_v1"
TARGET_NAME = "executable_net_bps_3h"
DEFAULT_BATCH = 200

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_V5_LABELS} (
    decision_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decision_timestamp TEXT NOT NULL,
    label_contract_version TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_horizon_sec INTEGER NOT NULL,
    executable_net_bps_3h REAL,
    executable_gross_bps_3h REAL,
    commission_bps REAL,
    spread_bps REAL,
    slippage_bps REAL,
    all_in_cost_bps REAL,
    executable_price_method TEXT,
    horizon_provenance TEXT,
    market_data_cutoff TEXT,
    label_valid INTEGER NOT NULL DEFAULT 0,
    label_invalid_reason TEXT,
    clock_v2_partition TEXT NOT NULL,
    label_json TEXT,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_clock_v2_labels_created ON {TABLE_V5_LABELS}(created_at);
"""

INVALID_IMMATURE = "HORIZON_NOT_MATURE"
INVALID_NO_BARS = "NO_SOURCE_BARS_AT_HORIZON"
INVALID_NO_ENTRY = "NO_DECISION_ENTRY_PRICE"
INVALID_NOT_AVAILABLE = "ACTION_NOT_PRODUCTION_AVAILABLE"


def labels_enabled() -> bool:
    return str(os.getenv("DAY_CLOCK_V2_LABELS", "true")).strip().lower() in {"1", "true", "yes", "on"}


def ensure_v5_label_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hold_label(*, decision_group_id: str, decision_ts: str) -> dict[str, Any]:
    """HOLD is exactly 0 bps at every horizon, by contract."""
    return {
        "decision_group_id": decision_group_id,
        "symbol": HOLD_SYMBOL,
        "decision_timestamp": decision_ts,
        "target_name": TARGET_NAME,
        "target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "executable_net_bps_3h": 0.0,
        "executable_gross_bps_3h": 0.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "all_in_cost_bps": 0.0,
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "horizon_provenance": "hold_reference",
        "market_data_cutoff": None,
        "label_valid": True,
        "label_invalid_reason": None,
    }


def build_v5_label(
    *,
    db_path: str | Path,
    decision_group_id: str,
    symbol: str,
    decision_ts: Any,
    action_available: Any,
    entry_px: Any = None,
    spread_bps: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One action's 3h executable-net label. Identical methodology for all actions."""
    when = parse_as_of(decision_ts)
    decision_iso = when.isoformat() if when else str(decision_ts)
    if symbol == HOLD_SYMBOL:
        return hold_label(decision_group_id=decision_group_id, decision_ts=decision_iso)
    base = {
        "decision_group_id": decision_group_id,
        "symbol": symbol,
        "decision_timestamp": decision_iso,
        "target_name": TARGET_NAME,
        "target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "executable_net_bps_3h": None,
        "executable_gross_bps_3h": None,
        "commission_bps": expected_exchange_commission_rt_pct() * 1e4,
        "spread_bps": float(spread_bps) if spread_bps is not None else None,
        "slippage_bps": expected_slippage_rt_pct() * 1e4,
        "all_in_cost_bps": honest_all_in_rt_pct(symbol) * 1e4,
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "horizon_provenance": None,
        "market_data_cutoff": None,
        "label_valid": False,
        "label_invalid_reason": None,
    }
    if action_available is not True:
        base["label_invalid_reason"] = INVALID_NOT_AVAILABLE
        return base
    stamp = now or _now()
    if when is None:
        base["label_invalid_reason"] = INVALID_IMMATURE
        return base
    horizon_end = when.timestamp() + PRIMARY_TARGET_HORIZON_SEC
    if stamp.timestamp() + 1e-9 < horizon_end:
        base["label_invalid_reason"] = INVALID_IMMATURE
        return base
    horizon_at = datetime.fromtimestamp(horizon_end, tz=timezone.utc)
    raw = load_asof_1m_bars(db_path, symbol, horizon_at, limit=PRIMARY_TARGET_HORIZON_SEC // 60 + 240)
    bars = clip_asof(normalize_bars(raw), horizon_at)
    if not bars:
        base["label_invalid_reason"] = INVALID_NO_BARS
        return base
    if entry_px in (None, ""):
        base["label_invalid_reason"] = INVALID_NO_ENTRY
        return base
    labels = build_clock_labels(
        bars,
        decision_ts=decision_iso,
        symbol=symbol,
        cost_bps=float(base["all_in_cost_bps"] or 0.0),
        commission_bps=float(base["commission_bps"] or 0.0),
        spread_bps=float(base["spread_bps"] or 0.0),
        slippage_bps=float(base["slippage_bps"] or 0.0),
        entry_px=float(entry_px),
        now=stamp,
    )
    net = (labels.get("clock_net_bps") or {}).get(PRIMARY_TARGET_HORIZON_NAME)
    gross = (labels.get("clock_gross_bps") or {}).get(PRIMARY_TARGET_HORIZON_NAME)
    provenance = (labels.get("horizon_provenance") or {}).get(PRIMARY_TARGET_HORIZON_NAME)
    base["executable_net_bps_3h"] = net
    base["executable_gross_bps_3h"] = gross
    base["horizon_provenance"] = provenance
    base["market_data_cutoff"] = (labels.get("market_data_cutoff") or {}).get(PRIMARY_TARGET_HORIZON_NAME)
    base["label_valid"] = net is not None
    if net is None:
        base["label_invalid_reason"] = INVALID_NO_BARS
    return base


def _load_pending_groups(db_path: str | Path, *, limit: int) -> list[dict[str, Any]]:
    """DEVELOPMENT-partition groups with artifacts but no complete label set."""
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if TABLE_ARTIFACT not in tables:
            return []
        labeled: set[str] = set()
        if TABLE_V5_LABELS in tables:
            labeled = {
                str(r[0])
                for r in conn.execute(
                    f"SELECT decision_group_id FROM {TABLE_V5_LABELS} WHERE label_valid=1 GROUP BY decision_group_id HAVING COUNT(*) >= 1"
                )
            }
        rows = [
            dict(r)
            for r in conn.execute(
                f"""
                SELECT decision_group_id, symbol, decision_timestamp, action_available,
                       clock_v2_partition, quote_json, feature_json
                FROM {TABLE_ARTIFACT}
                ORDER BY created_at
                """
            )
        ]
    finally:
        conn.close()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["decision_group_id"]), []).append(row)
    out: list[dict[str, Any]] = []
    for gid, arts in grouped.items():
        if gid in labeled:
            continue
        stamped = next((a.get("clock_v2_partition") for a in arts if a.get("clock_v2_partition")), None)
        when = next((a.get("decision_timestamp") for a in arts if a.get("decision_timestamp")), None)
        part = str(stamped) if stamped else partition_for(when)
        if part != DEVELOPMENT:
            continue
        out.append({"decision_group_id": gid, "artifacts": arts, "partition": part})
        if len(out) >= limit:
            break
    return out


def _entry_and_spread(art: dict[str, Any]) -> tuple[Any, Any]:
    try:
        quote = json.loads(art.get("quote_json") or "{}")
    except (TypeError, ValueError):
        quote = {}
    try:
        feats = json.loads(art.get("feature_json") or "{}")
    except (TypeError, ValueError):
        feats = {}
    entry = quote.get("entry_px") or quote.get("mid") or quote.get("best_ask")
    spread = quote.get("spread_bps")
    if spread is None:
        spread = feats.get("spread_bps")
    return entry, spread


def persist_v5_labels(db_path: str | Path, labels: list[dict[str, Any]]) -> int:
    ensure_v5_label_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    written = 0
    try:
        for lab in labels:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_V5_LABELS}(
                    decision_group_id, symbol, created_at, decision_timestamp,
                    label_contract_version, target_name, target_horizon_sec,
                    executable_net_bps_3h, executable_gross_bps_3h,
                    commission_bps, spread_bps, slippage_bps, all_in_cost_bps,
                    executable_price_method, horizon_provenance, market_data_cutoff,
                    label_valid, label_invalid_reason, clock_v2_partition, label_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    lab["decision_group_id"],
                    lab["symbol"],
                    _now().isoformat(),
                    lab["decision_timestamp"],
                    LABEL_CONTRACT_VERSION,
                    lab["target_name"],
                    int(lab["target_horizon_sec"]),
                    lab.get("executable_net_bps_3h"),
                    lab.get("executable_gross_bps_3h"),
                    lab.get("commission_bps"),
                    lab.get("spread_bps"),
                    lab.get("slippage_bps"),
                    lab.get("all_in_cost_bps"),
                    lab.get("executable_price_method"),
                    lab.get("horizon_provenance"),
                    lab.get("market_data_cutoff"),
                    1 if lab.get("label_valid") else 0,
                    lab.get("label_invalid_reason"),
                    DEVELOPMENT,
                    json.dumps(lab, default=str),
                ),
            )
            written += 1
        conn.commit()
    finally:
        conn.close()
    return written


def run_v5_label_batch(db_path: str | Path, *, limit: int = DEFAULT_BATCH, now: datetime | None = None) -> dict[str, Any]:
    """Label matured v5 DEVELOPMENT groups. Fail-open, never raises upward."""
    summary = {
        "contract": LABEL_CONTRACT_VERSION,
        "groups_scanned": 0,
        "labels_written": 0,
        "valid": 0,
        "immature": 0,
        "errors": 0,
        "partition": DEVELOPMENT,
    }
    if not labels_enabled() or not db_path:
        return summary
    try:
        # Create the target table up front so the v5 label authority exists and is
        # inspectable from the moment the contract goes live, not only once the
        # first DEVELOPMENT group matures three hours later.
        ensure_v5_label_schema(db_path)
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_LABEL schema init failed: %s", exc)
        summary["errors"] += 1
        return summary
    try:
        pending = _load_pending_groups(db_path, limit=limit)
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_LABEL scan failed: %s", exc)
        summary["errors"] += 1
        return summary
    stamp = now or _now()
    for group in pending:
        summary["groups_scanned"] += 1
        by_sym = {str(a["symbol"]): a for a in group["artifacts"]}
        built: list[dict[str, Any]] = []
        try:
            for sym in (*COINS, HOLD_SYMBOL):
                art = by_sym.get(sym)
                if art is None:
                    continue
                entry, spread = _entry_and_spread(art)
                avail = art.get("action_available")
                lab = build_v5_label(
                    db_path=db_path,
                    decision_group_id=group["decision_group_id"],
                    symbol=sym,
                    decision_ts=art.get("decision_timestamp"),
                    action_available=True if sym == HOLD_SYMBOL else (None if avail is None else bool(avail)),
                    entry_px=entry,
                    spread_bps=spread,
                    now=stamp,
                )
                built.append(lab)
            if any(lab.get("label_invalid_reason") == INVALID_IMMATURE for lab in built):
                summary["immature"] += 1
                continue
            summary["labels_written"] += persist_v5_labels(db_path, built)
            summary["valid"] += sum(1 for lab in built if lab.get("label_valid"))
        except Exception as exc:
            logger.warning("DAY_CLOCK_V2_LABEL group %s failed: %s", group["decision_group_id"], exc)
            summary["errors"] += 1
    return summary


def load_v5_label_presence(db_path: str | Path) -> dict[tuple[str, str], bool]:
    """Validity flags only. Never returns target values."""
    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_V5_LABELS,)).fetchone() is None:
            return {}
        rows = conn.execute(f"SELECT decision_group_id, symbol, label_valid FROM {TABLE_V5_LABELS}").fetchall()
    finally:
        conn.close()
    return {(str(gid), str(sym)): bool(valid) for gid, sym, valid in rows}


def v5_label_contract() -> dict[str, Any]:
    return {
        "contract_version": LABEL_CONTRACT_VERSION,
        "target": PRIMARY_TARGET,
        "target_name": TARGET_NAME,
        "target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "target_horizon_name": PRIMARY_TARGET_HORIZON_NAME,
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "commission_methodology": "expected_exchange_commission_rt",
        "spread_methodology": "decision_time_quote_spread_bps",
        "slippage_methodology": "expected_slippage_rt",
        "identical_methodology_for_every_action": True,
        "hold_value_bps": 0.0,
        "partition": DEVELOPMENT,
        "production_exit_may_replace_target": False,
        "table": TABLE_V5_LABELS,
        "separate_from_generic_4h_labels": True,
        "generic_4h_lock_read": False,
        "why_new_table": (
            "day_decision_outcome_labels has no 3h horizon (15m/30m/1h/2h/4h only) and belongs to "
            "the sealed 4H-entry experiment. A separate table supplies the frozen v5 3h target "
            "without reading or altering that lock."
        ),
    }


__all__ = [
    "INVALID_IMMATURE",
    "INVALID_NOT_AVAILABLE",
    "LABEL_CONTRACT_VERSION",
    "TABLE_V5_LABELS",
    "TARGET_NAME",
    "build_v5_label",
    "ensure_v5_label_schema",
    "hold_label",
    "load_v5_label_presence",
    "persist_v5_labels",
    "run_v5_label_batch",
    "v5_label_contract",
]
