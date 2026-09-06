"""CLOCK-V2 fixed-horizon target labels. Research only.

Writes the 3h executable-net target for every production-available action in a
v5 DEVELOPMENT decision group, plus HOLD = 0. Separate table from
``day_decision_outcome_labels`` so the generic 4H-entry lock and its labels are
neither read nor modified.

Label market-data authority is Redis canonical 1m, then Binance.US REST closed
1m klines. feature_ohlcv persist timestamps are never candle identity.

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
from backend.services.day_clock_v2_label_source import (
    INTERVAL,
    INVALID_MISMATCH,
    INVALID_PIT,
    LABEL_SOURCE_VERSION,
    RETRYABLE_REASONS,
    STATUS_COMPLETE,
    STATUS_PENDING_LABEL_SOURCE,
    STATUS_PENDING_NOT_MATURE,
    STATUS_TERMINAL_INVALID,
    TERMINAL_REASONS,
    pit_ok,
    resolve_v5_horizon_candle,
)
from backend.services.day_clock_v2_partition import DEVELOPMENT, partition_for
from backend.services.day_path_clock_features import parse_as_of
from backend.services.day_path_clock_v2 import (
    EXECUTABLE_PRICE_METHOD,
    PRIMARY_TARGET,
    PRIMARY_TARGET_HORIZON_NAME,
    PRIMARY_TARGET_HORIZON_SEC,
)
from backend.services.day_path_clock_v2_capture import TABLE_ARTIFACT

logger = logging.getLogger(__name__)

TABLE_V5_LABELS = "day_clock_v2_outcome_labels"
TABLE_V5_LABELS_HISTORY = "day_clock_v2_outcome_labels_history"
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
    label_source_version TEXT,
    label_source TEXT,
    label_status TEXT,
    target_bar_open_ts TEXT,
    target_bar_close_ts TEXT,
    source_verified INTEGER,
    source_fetch_timestamp TEXT,
    exchange_symbol TEXT,
    label_interval TEXT,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_clock_v2_labels_created ON {TABLE_V5_LABELS}(created_at);
CREATE TABLE IF NOT EXISTS {TABLE_V5_LABELS_HISTORY} (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    archived_at TEXT NOT NULL,
    decision_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT,
    decision_timestamp TEXT,
    label_contract_version TEXT,
    label_source_version TEXT,
    label_source TEXT,
    label_status TEXT,
    label_valid INTEGER,
    label_invalid_reason TEXT,
    label_json TEXT
);
"""

_LABEL_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("label_source_version", "TEXT"),
    ("label_source", "TEXT"),
    ("label_status", "TEXT"),
    ("target_bar_open_ts", "TEXT"),
    ("target_bar_close_ts", "TEXT"),
    ("source_verified", "INTEGER"),
    ("source_fetch_timestamp", "TEXT"),
    ("exchange_symbol", "TEXT"),
    ("label_interval", "TEXT"),
)

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
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_V5_LABELS})")}
        for name, decl in _LABEL_MIGRATIONS:
            if name not in cols:
                conn.execute(f"ALTER TABLE {TABLE_V5_LABELS} ADD COLUMN {name} {decl}")
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
        "label_source_version": LABEL_SOURCE_VERSION,
        "label_source": "hold_reference",
        "label_status": STATUS_COMPLETE,
        "target_bar_open_ts": None,
        "target_bar_close_ts": None,
        "source_verified": True,
        "source_fetch_timestamp": None,
        "exchange_symbol": HOLD_SYMBOL,
        "label_interval": INTERVAL,
    }


def _executable_net(*, entry_px: float, exit_px: float, cost_bps: float) -> tuple[float, float]:
    """Frozen formula: (exit/entry - 1) * 1e4 minus named all-in cost."""
    gross = (exit_px - entry_px) / entry_px * 1e4
    return gross, gross - float(cost_bps)


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
    redis_client: Any = None,
    rest_fetch: Any = None,
    source_resolver: Any = None,
) -> dict[str, Any]:
    """One action's 3h executable-net label. Identical methodology for all actions."""
    del db_path  # market-data authority is Redis/REST, not feature_ohlcv
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
        "label_source_version": LABEL_SOURCE_VERSION,
        "label_source": None,
        "label_status": STATUS_PENDING_LABEL_SOURCE,
        "target_bar_open_ts": None,
        "target_bar_close_ts": None,
        "source_verified": False,
        "source_fetch_timestamp": None,
        "exchange_symbol": symbol,
        "label_interval": INTERVAL,
    }
    if action_available is not True:
        base["label_invalid_reason"] = INVALID_NOT_AVAILABLE
        base["label_status"] = STATUS_TERMINAL_INVALID
        return base
    stamp = now or _now()
    if when is None:
        base["label_invalid_reason"] = INVALID_IMMATURE
        base["label_status"] = STATUS_PENDING_NOT_MATURE
        return base
    horizon_end = when.timestamp() + PRIMARY_TARGET_HORIZON_SEC
    if stamp.timestamp() + 1e-9 < horizon_end:
        base["label_invalid_reason"] = INVALID_IMMATURE
        base["label_status"] = STATUS_PENDING_NOT_MATURE
        return base
    horizon_at = datetime.fromtimestamp(horizon_end, tz=timezone.utc)
    base["market_data_cutoff"] = horizon_at.isoformat()
    resolver = source_resolver or resolve_v5_horizon_candle
    resolved = resolver(
        symbol,
        horizon_at,
        now=stamp,
        redis_client=redis_client,
        rest_fetch=rest_fetch,
    )
    base["source_fetch_timestamp"] = resolved.get("source_fetch_timestamp")
    base["label_source"] = resolved.get("label_source")
    base["source_verified"] = bool(resolved.get("source_verified"))
    base["target_bar_open_ts"] = resolved.get("target_bar_open_ts")
    base["target_bar_close_ts"] = resolved.get("target_bar_close_ts")
    base["exchange_symbol"] = resolved.get("exchange_symbol") or symbol
    if resolved.get("status") == STATUS_PENDING_NOT_MATURE:
        base["label_invalid_reason"] = INVALID_IMMATURE
        base["label_status"] = STATUS_PENDING_NOT_MATURE
        return base
    if not resolved.get("ok"):
        reason = str(resolved.get("reason") or INVALID_NO_BARS)
        base["label_invalid_reason"] = reason
        if reason in TERMINAL_REASONS:
            base["label_status"] = STATUS_TERMINAL_INVALID
        else:
            base["label_status"] = STATUS_PENDING_LABEL_SOURCE
        return base
    candle = resolved.get("candle")
    if candle is None or not pit_ok(decision_ts=when, horizon_at=horizon_at, candle=candle):
        base["label_invalid_reason"] = INVALID_PIT
        base["label_status"] = STATUS_TERMINAL_INVALID
        return base
    if entry_px in (None, ""):
        base["label_invalid_reason"] = INVALID_NO_ENTRY
        base["label_status"] = STATUS_TERMINAL_INVALID
        return base
    entry = float(entry_px)
    if entry <= 0:
        base["label_invalid_reason"] = INVALID_NO_ENTRY
        base["label_status"] = STATUS_TERMINAL_INVALID
        return base
    gross, net = _executable_net(entry_px=entry, exit_px=float(candle.close), cost_bps=float(base["all_in_cost_bps"] or 0.0))
    base["executable_gross_bps_3h"] = gross
    base["executable_net_bps_3h"] = net
    base["horizon_provenance"] = resolved.get("label_source")
    base["label_valid"] = True
    base["label_status"] = STATUS_COMPLETE
    base["label_invalid_reason"] = None
    return base


def action_is_available(art: dict[str, Any]) -> bool | None:
    raw = art.get("action_available")
    if raw is None:
        return None
    return bool(raw)


def required_actions(artifacts: list[dict[str, Any]]) -> list[str]:
    """Production-available coins plus HOLD. HOLD alone never completes a group."""
    needed = [HOLD_SYMBOL]
    for art in artifacts:
        sym = str(art.get("symbol") or "")
        if sym == HOLD_SYMBOL:
            continue
        if action_is_available(art) is True:
            needed.append(sym)
    return needed


def _active_valid_map(conn: sqlite3.Connection) -> dict[tuple[str, str], bool]:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_V5_LABELS})")}
    if "label_source_version" in cols:
        rows = conn.execute(
            f"""
            SELECT decision_group_id, symbol, label_valid, label_source_version, label_status
            FROM {TABLE_V5_LABELS}
            """
        ).fetchall()
        out: dict[tuple[str, str], bool] = {}
        for gid, sym, valid, version, status in rows:
            out[(str(gid), str(sym))] = bool(valid) and str(version or "") == LABEL_SOURCE_VERSION
            if status == STATUS_TERMINAL_INVALID and not valid:
                out[(str(gid), str(sym))] = False
        return out
    return {(str(gid), str(sym)): False for gid, sym, _ in conn.execute(f"SELECT decision_group_id, symbol, label_valid FROM {TABLE_V5_LABELS}")}


def group_label_status(
    artifacts: list[dict[str, Any]],
    labels: dict[tuple[str, str], dict[str, Any]],
    *,
    now: datetime | None = None,
) -> str:
    gid = str((artifacts[0] or {}).get("decision_group_id") or "")
    when = parse_as_of(next((a.get("decision_timestamp") for a in artifacts if a.get("decision_timestamp")), None))
    stamp = now or _now()
    if when is not None and stamp.timestamp() + 1e-9 < when.timestamp() + PRIMARY_TARGET_HORIZON_SEC:
        return STATUS_PENDING_NOT_MATURE
    required = required_actions(artifacts)
    statuses = []
    for sym in required:
        lab = labels.get((gid, sym)) or {}
        if lab.get("label_valid") and lab.get("label_source_version") == LABEL_SOURCE_VERSION:
            statuses.append(STATUS_COMPLETE)
            continue
        reason = lab.get("label_invalid_reason")
        if reason in TERMINAL_REASONS or reason == INVALID_NO_ENTRY:
            statuses.append(STATUS_TERMINAL_INVALID)
        elif reason == INVALID_IMMATURE:
            statuses.append(STATUS_PENDING_NOT_MATURE)
        else:
            statuses.append(STATUS_PENDING_LABEL_SOURCE)
    if statuses and all(s == STATUS_COMPLETE for s in statuses):
        return STATUS_COMPLETE
    if any(s == STATUS_PENDING_NOT_MATURE for s in statuses):
        return STATUS_PENDING_NOT_MATURE
    if any(s == STATUS_PENDING_LABEL_SOURCE for s in statuses) or not statuses:
        return STATUS_PENDING_LABEL_SOURCE
    return STATUS_TERMINAL_INVALID


def _load_pending_groups(db_path: str | Path, *, limit: int) -> list[dict[str, Any]]:
    """DEVELOPMENT groups missing a complete active-version label set."""
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if TABLE_ARTIFACT not in tables:
            return []
        valid = _active_valid_map(conn) if TABLE_V5_LABELS in tables else {}
        terminal: set[tuple[str, str]] = set()
        if TABLE_V5_LABELS in tables:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_V5_LABELS})")}
            if "label_status" in cols:
                terminal = {
                    (str(r[0]), str(r[1]))
                    for r in conn.execute(
                        f"""
                        SELECT decision_group_id, symbol FROM {TABLE_V5_LABELS}
                        WHERE label_source_version=? AND label_status=?
                        """,
                        (LABEL_SOURCE_VERSION, STATUS_TERMINAL_INVALID),
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
        stamped = next((a.get("clock_v2_partition") for a in arts if a.get("clock_v2_partition")), None)
        when = next((a.get("decision_timestamp") for a in arts if a.get("decision_timestamp")), None)
        part = str(stamped) if stamped else partition_for(when)
        if part != DEVELOPMENT:
            continue
        needed = required_actions(arts)
        complete = all(valid.get((gid, sym)) for sym in needed)
        if complete:
            continue
        # Terminal on a required coin stays visible so a later cycle can report it,
        # but is not retried unless the reason is retryable (no valid v2 row).
        retry_needed = False
        for sym in needed:
            if valid.get((gid, sym)):
                continue
            if (gid, sym) in terminal:
                continue
            retry_needed = True
            break
        if not retry_needed and any((gid, sym) in terminal for sym in needed):
            continue
        if not retry_needed:
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


def _archive_existing(conn: sqlite3.Connection, group_id: str, symbol: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_V5_LABELS})")}
    if not cols:
        return
    cur = conn.execute(
        f"SELECT * FROM {TABLE_V5_LABELS} WHERE decision_group_id=? AND symbol=?",
        (group_id, symbol),
    )
    row = cur.fetchone()
    if row is None:
        return
    names = [d[0] for d in cur.description]
    payload = dict(zip(names, row, strict=True))
    version = str(payload.get("label_source_version") or "")
    if version == LABEL_SOURCE_VERSION:
        return
    conn.execute(
        f"""
        INSERT INTO {TABLE_V5_LABELS_HISTORY}(
            archived_at, decision_group_id, symbol, created_at, decision_timestamp,
            label_contract_version, label_source_version, label_source, label_status,
            label_valid, label_invalid_reason, label_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            _now().isoformat(),
            payload.get("decision_group_id"),
            payload.get("symbol"),
            payload.get("created_at"),
            payload.get("decision_timestamp"),
            payload.get("label_contract_version"),
            payload.get("label_source_version"),
            payload.get("label_source"),
            payload.get("label_status"),
            payload.get("label_valid"),
            payload.get("label_invalid_reason"),
            payload.get("label_json"),
        ),
    )


def persist_v5_labels(db_path: str | Path, labels: list[dict[str, Any]]) -> int:
    ensure_v5_label_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    written = 0
    try:
        for lab in labels:
            _archive_existing(conn, str(lab["decision_group_id"]), str(lab["symbol"]))
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_V5_LABELS}(
                    decision_group_id, symbol, created_at, decision_timestamp,
                    label_contract_version, target_name, target_horizon_sec,
                    executable_net_bps_3h, executable_gross_bps_3h,
                    commission_bps, spread_bps, slippage_bps, all_in_cost_bps,
                    executable_price_method, horizon_provenance, market_data_cutoff,
                    label_valid, label_invalid_reason, clock_v2_partition, label_json,
                    label_source_version, label_source, label_status,
                    target_bar_open_ts, target_bar_close_ts, source_verified,
                    source_fetch_timestamp, exchange_symbol, label_interval
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    lab.get("label_source_version") or LABEL_SOURCE_VERSION,
                    lab.get("label_source"),
                    lab.get("label_status"),
                    lab.get("target_bar_open_ts"),
                    lab.get("target_bar_close_ts"),
                    1 if lab.get("source_verified") else 0,
                    lab.get("source_fetch_timestamp"),
                    lab.get("exchange_symbol"),
                    lab.get("label_interval") or INTERVAL,
                ),
            )
            written += 1
        conn.commit()
    finally:
        conn.close()
    return written


def run_v5_label_batch(
    db_path: str | Path,
    *,
    limit: int = DEFAULT_BATCH,
    now: datetime | None = None,
    redis_client: Any = None,
    rest_fetch: Any = None,
    source_resolver: Any = None,
) -> dict[str, Any]:
    """Label matured v5 DEVELOPMENT groups. Fail-open, never raises upward."""
    summary = {
        "contract": LABEL_CONTRACT_VERSION,
        "label_source_version": LABEL_SOURCE_VERSION,
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
                    redis_client=redis_client,
                    rest_fetch=rest_fetch,
                    source_resolver=source_resolver,
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
    """Validity flags only for the active label-source version. Never returns targets."""
    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_V5_LABELS,)).fetchone() is None:
            return {}
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_V5_LABELS})")}
        if "label_source_version" in cols:
            rows = conn.execute(
                f"""
                SELECT decision_group_id, symbol, label_valid, label_source_version
                FROM {TABLE_V5_LABELS}
                """
            ).fetchall()
            return {
                (str(gid), str(sym)): bool(valid) and str(version or "") == LABEL_SOURCE_VERSION
                for gid, sym, valid, version in rows
            }
        return {}
    finally:
        conn.close()


def v5_label_contract() -> dict[str, Any]:
    return {
        "contract_version": LABEL_CONTRACT_VERSION,
        "label_source_version": LABEL_SOURCE_VERSION,
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
        "history_table": TABLE_V5_LABELS_HISTORY,
        "feature_ohlcv_is_label_authority": False,
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
    "INVALID_MISMATCH",
    "INVALID_NOT_AVAILABLE",
    "LABEL_CONTRACT_VERSION",
    "RETRYABLE_REASONS",
    "TABLE_V5_LABELS",
    "TABLE_V5_LABELS_HISTORY",
    "TARGET_NAME",
    "build_v5_label",
    "ensure_v5_label_schema",
    "group_label_status",
    "hold_label",
    "load_v5_label_presence",
    "persist_v5_labels",
    "required_actions",
    "run_v5_label_batch",
    "v5_label_contract",
]
