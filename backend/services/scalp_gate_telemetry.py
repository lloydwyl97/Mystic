"""SCALP gate counters, shadow rejects, and attribution helpers."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.scalp_gate_registry import DECISION_POLICY_VERSION, get_gate, map_reason_to_gate

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scalp_gate_counters (
    date TEXT NOT NULL,
    gate_id TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    setup TEXT NOT NULL DEFAULT '',
    regime TEXT NOT NULL DEFAULT '',
    evaluated INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    hard_blocked INTEGER NOT NULL DEFAULT 0,
    sized_down INTEGER NOT NULL DEFAULT 0,
    rank_penalized INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (date, gate_id, symbol, setup, regime)
);

CREATE TABLE IF NOT EXISTS scalp_shadow_rejects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    decision_id TEXT,
    symbol TEXT NOT NULL,
    setup TEXT,
    gate_id TEXT NOT NULL,
    bar_timestamp INTEGER,
    entry_price REAL,
    stop_price REAL,
    target_price REAL,
    fee_model_version TEXT,
    policy_version TEXT,
    detail TEXT,
    diag_json TEXT,
    bar_closed INTEGER NOT NULL DEFAULT 1,
    hyp_exit_price REAL,
    hyp_mfe_pct REAL,
    hyp_mae_pct REAL,
    hyp_net_pnl REAL,
    hyp_resolved_at TEXT,
    hyp_exit_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_scalp_shadow_rejects_gate ON scalp_shadow_rejects(gate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_scalp_shadow_unresolved ON scalp_shadow_rejects(hyp_resolved_at);
"""


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_scalp_gate_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def record_gate_event(
    db_path: str | Path,
    *,
    gate_id: str = "",
    reason: str = "",
    symbol: str = "",
    outcome: str = "evaluated",
    setup: str = "",
    regime: str = "",
    decision_id: str = "",
    detail: str = "",
) -> None:
    try:
        ensure_scalp_gate_schema(db_path)
        gid = str(gate_id or "").strip() or map_reason_to_gate(reason)
        if not gid:
            return
        day = _utc_today()
        cols = {
            "evaluated": 1,
            "passed": 1 if outcome == "passed" else 0,
            "hard_blocked": 1 if outcome == "hard_blocked" else 0,
            "sized_down": 1 if outcome == "sized_down" else 0,
            "rank_penalized": 1 if outcome == "rank_penalized" else 0,
        }
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute(
                """
                INSERT INTO scalp_gate_counters(
                    date, gate_id, symbol, setup, regime,
                    evaluated, passed, hard_blocked, sized_down, rank_penalized, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, gate_id, symbol, setup, regime) DO UPDATE SET
                    evaluated = evaluated + excluded.evaluated,
                    passed = passed + excluded.passed,
                    hard_blocked = hard_blocked + excluded.hard_blocked,
                    sized_down = sized_down + excluded.sized_down,
                    rank_penalized = rank_penalized + excluded.rank_penalized,
                    updated_at = excluded.updated_at
                """,
                (
                    day,
                    gid,
                    str(symbol or ""),
                    str(setup or ""),
                    str(regime or ""),
                    cols["evaluated"],
                    cols["passed"],
                    cols["hard_blocked"],
                    cols["sized_down"],
                    cols["rank_penalized"],
                    _utc_now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        if detail or decision_id or reason:
            logger.debug(
                "SCALP_GATE_EVENT gate=%s outcome=%s symbol=%s reason=%s detail=%s",
                gid,
                outcome,
                symbol,
                reason,
                detail,
            )
    except Exception:
        logger.debug("scalp record_gate_event failed gate=%s", gate_id or reason, exc_info=True)


def record_shadow_reject(
    db_path: str | Path,
    *,
    symbol: str,
    gate_id: str = "",
    reason: str = "",
    setup: str = "",
    entry_price: float = 0.0,
    stop_price: float = 0.0,
    target_price: float = 0.0,
    bar_timestamp: int | None = None,
    detail: str = "",
    diag: dict[str, Any] | None = None,
    decision_id: str = "",
) -> None:
    try:
        ensure_scalp_gate_schema(db_path)
        sym = str(symbol or "").strip().upper()
        if not sym:
            return
        gid = str(gate_id or "").strip() or map_reason_to_gate(reason)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute(
                """
                INSERT INTO scalp_shadow_rejects(
                    created_at, decision_id, symbol, setup, gate_id, bar_timestamp,
                    entry_price, stop_price, target_price, fee_model_version, policy_version,
                    detail, diag_json, bar_closed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    _utc_now(),
                    decision_id or None,
                    sym,
                    setup or None,
                    gid,
                    int(bar_timestamp or 0) or None,
                    float(entry_price) if entry_price else None,
                    float(stop_price) if stop_price else None,
                    float(target_price) if target_price else None,
                    "binance_us_taker_v1",
                    DECISION_POLICY_VERSION,
                    detail or reason or "",
                    json.dumps(diag or {}, default=str)[:8000],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.debug("scalp record_shadow_reject failed gate=%s", gate_id or reason, exc_info=True)


def counters_today(db_path: str | Path, *, date: str | None = None) -> list[dict[str, Any]]:
    ensure_scalp_gate_schema(db_path)
    day = date or _utc_today()
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT gate_id,
                   SUM(evaluated) AS evaluated,
                   SUM(passed) AS passed,
                   SUM(hard_blocked) AS hard_blocked,
                   SUM(sized_down) AS sized_down,
                   SUM(rank_penalized) AS rank_penalized
            FROM scalp_gate_counters
            WHERE date = ?
            GROUP BY gate_id
            ORDER BY hard_blocked DESC, evaluated DESC
            """,
            (day,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            spec = get_gate(str(d["gate_id"]))
            d["layer"] = spec.layer if spec else None
            d["behavior"] = spec.behavior if spec else None
            d["purpose"] = spec.purpose if spec else None
            out.append(d)
        return out
    finally:
        conn.close()


def shadow_rejects_summary(db_path: str | Path, *, limit: int = 50) -> dict[str, Any]:
    ensure_scalp_gate_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        by_gate = conn.execute(
            """
            SELECT gate_id, COUNT(*) AS n,
                   SUM(CASE WHEN hyp_resolved_at IS NULL THEN 1 ELSE 0 END) AS unresolved,
                   SUM(CASE WHEN hyp_net_pnl IS NOT NULL THEN hyp_net_pnl ELSE 0 END) AS hyp_net_sum
            FROM scalp_shadow_rejects
            GROUP BY gate_id
            ORDER BY n DESC
            """
        ).fetchall()
        recent = conn.execute(
            """
            SELECT id, created_at, symbol, setup, gate_id, entry_price, hyp_net_pnl, hyp_exit_reason, hyp_resolved_at
            FROM scalp_shadow_rejects
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return {"by_gate": [dict(r) for r in by_gate], "recent": [dict(r) for r in recent]}
    finally:
        conn.close()


def attribution_report(db_path: str | Path, *, date: str | None = None) -> dict[str, Any]:
    ensure_scalp_gate_schema(db_path)
    day = date or _utc_today()
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        executed = conn.execute(
            """
            SELECT COUNT(*) AS trades,
                   COALESCE(SUM(pnl_usd), 0) AS net_pnl,
                   COALESCE(SUM(fee_usd), 0) AS fees,
                   COALESCE(SUM(slippage_usd), 0) AS slip,
                   SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins
            FROM scalp_paper_trades
            WHERE UPPER(side)='SELL' AND date(created_at)=?
            """,
            (day,),
        ).fetchone()
        by_exit = conn.execute(
            """
            SELECT COALESCE(exit_reason, 'UNKNOWN') AS exit_reason, COUNT(*) AS n, COALESCE(SUM(pnl_usd),0) AS net_pnl
            FROM scalp_paper_trades
            WHERE UPPER(side)='SELL' AND date(created_at)=?
            GROUP BY 1
            ORDER BY n DESC
            """,
            (day,),
        ).fetchall()
        genuine = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), 0) AS wins,
                   COALESCE(SUM(pnl_usd), 0) AS net_pnl
            FROM scalp_paper_trades
            WHERE UPPER(side)='SELL' AND date(created_at)=?
              AND COALESCE(json_extract(diagnostics_json, '$.soft_rank_entry'), 0) = 0
              AND COALESCE(json_extract(diagnostics_json, '$.passed'), 1) = 1
            """,
            (day,),
        ).fetchone()
        shadow = conn.execute(
            """
            SELECT gate_id, COUNT(*) AS rejects,
                   SUM(CASE WHEN hyp_net_pnl IS NOT NULL AND hyp_net_pnl > 0 THEN 1 ELSE 0 END) AS hyp_wins,
                   SUM(CASE WHEN hyp_net_pnl IS NOT NULL AND hyp_net_pnl <= 0 THEN 1 ELSE 0 END) AS hyp_losses,
                   COALESCE(SUM(hyp_net_pnl), 0) AS hyp_net_sum,
                   SUM(CASE WHEN hyp_resolved_at IS NULL THEN 1 ELSE 0 END) AS unresolved
            FROM scalp_shadow_rejects
            WHERE date(created_at)=?
            GROUP BY gate_id
            ORDER BY rejects DESC
            """,
            (day,),
        ).fetchall()
        no_signal = conn.execute(
            "SELECT COALESCE(SUM(hard_blocked),0) FROM scalp_gate_counters WHERE date=? AND gate_id='STRATEGY_NO_SIGNAL'",
            (day,),
        ).fetchone()[0]
        soft_blocked = conn.execute(
            "SELECT COALESCE(SUM(hard_blocked),0) FROM scalp_gate_counters WHERE date=? AND gate_id='SOFT_RANK_BLOCKED'",
            (day,),
        ).fetchone()[0]
        return {
            "date": day,
            "policy_version": DECISION_POLICY_VERSION,
            "executed": dict(executed) if executed else {},
            "genuine_pass_closes": dict(genuine) if genuine else {},
            "by_exit_reason": [dict(r) for r in by_exit],
            "gate_opportunity": [dict(r) for r in shadow],
            "no_signal_count": int(no_signal or 0),
            "soft_rank_blocked_count": int(soft_blocked or 0),
            "gate_counters": counters_today(db_path, date=day),
        }
    finally:
        conn.close()


def _hyp_net_usd(entry: float, exit_px: float, *, notional: float = 100.0, fee_rt: float = 0.002) -> float:
    if entry <= 0:
        return 0.0
    gross = (exit_px - entry) / entry * notional
    return float(gross - notional * fee_rt * 2.0)


def _walk_scalp_brackets(
    bars: list[dict[str, Any]],
    *,
    entry: float,
    stop: float,
    target: float,
    max_bars: int = 30,
) -> dict[str, Any] | None:
    if not bars or entry <= 0:
        return None
    mfe = 0.0
    mae = 0.0
    for i, b in enumerate(bars[:max_bars]):
        high = float(b.get("high") or 0.0)
        low = float(b.get("low") or 0.0)
        close = float(b.get("close") or 0.0)
        if high > 0:
            mfe = max(mfe, (high - entry) / entry)
        if low > 0:
            mae = min(mae, (low - entry) / entry)
        hit_stop = stop > 0 and low > 0 and low <= stop
        hit_tgt = target > entry and high >= target
        if hit_stop:
            return {
                "exit_price": stop,
                "exit_reason": "SHADOW_STOP",
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, stop),
            }
        if hit_tgt:
            return {
                "exit_price": target,
                "exit_reason": "SHADOW_TARGET",
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, target),
            }
        if i == max_bars - 1 and close > 0:
            return {
                "exit_price": close,
                "exit_reason": "SHADOW_TIME",
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, close),
            }
    return None


def resolve_shadow_rejects_closed_bar(db_path: str | Path, *, max_rows: int = 100) -> dict[str, Any]:
    ensure_scalp_gate_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    resolved = 0
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, entry_price, stop_price, target_price, symbol
            FROM scalp_shadow_rejects
            WHERE hyp_resolved_at IS NULL AND entry_price IS NOT NULL AND entry_price > 0
            ORDER BY id ASC LIMIT ?
            """,
            (int(max_rows),),
        ).fetchall()
        now = _utc_now()
        for r in rows:
            try:
                created = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
            except Exception:
                age_h = 0.0
            if age_h < 24.0:
                continue
            entry = float(r["entry_price"])
            stop = float(r["stop_price"] or 0.0)
            if stop > 0 and stop < entry:
                hyp_pnl = _hyp_net_usd(entry, stop)
                reason = "SHADOW_EXPIRE_ASSUME_STOP"
                exit_px = stop
            else:
                hyp_pnl = 0.0
                reason = "SHADOW_EXPIRE_FLAT"
                exit_px = entry
            conn.execute(
                """
                UPDATE scalp_shadow_rejects
                SET hyp_resolved_at=?, hyp_exit_reason=?, hyp_net_pnl=?, hyp_exit_price=?
                WHERE id=?
                """,
                (now, reason, float(hyp_pnl), float(exit_px), int(r["id"])),
            )
            resolved += 1
        conn.commit()
        return {"resolved": resolved, "scanned": len(rows)}
    finally:
        conn.close()


async def resolve_shadow_rejects_async(db_path: str | Path, *, max_rows: int = 40) -> dict[str, Any]:
    ensure_scalp_gate_schema(db_path)
    try:
        from backend.services.binance_scalp.strategies.kline_cache import closed_bars_only, fetch_1m_bars
    except Exception as exc:
        logger.debug("scalp shadow async deps unavailable: %s", exc)
        return resolve_shadow_rejects_closed_bar(db_path, max_rows=max_rows)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    resolved = 0
    scanned = 0
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, entry_price, stop_price, target_price, symbol, bar_timestamp
            FROM scalp_shadow_rejects
            WHERE hyp_resolved_at IS NULL AND entry_price IS NOT NULL AND entry_price > 0
              AND stop_price IS NOT NULL AND stop_price > 0
            ORDER BY id ASC LIMIT ?
            """,
            (int(max_rows),),
        ).fetchall()
        scanned = len(rows)
        now = _utc_now()
        for r in rows:
            symbol = str(r["symbol"] or "")
            entry = float(r["entry_price"])
            stop = float(r["stop_price"] or 0.0)
            target = float(r["target_price"] or 0.0)
            bar_ts = int(r["bar_timestamp"] or 0)
            try:
                bars = closed_bars_only(fetch_1m_bars(symbol, minutes=90), interval_sec=60)
            except Exception:
                continue
            if bar_ts > 0:
                ts_sec = bar_ts // 1000 if bar_ts > 10_000_000_000 else bar_ts
                future = [b for b in bars if int(b.get("ts") or 0) > ts_sec]
            else:
                try:
                    created = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
                    ts_sec = int(created.timestamp())
                except Exception:
                    ts_sec = 0
                future = [b for b in bars if int(b.get("ts") or 0) > ts_sec] if ts_sec else bars[-30:]
            if len(future) < 2:
                continue
            walked = _walk_scalp_brackets(future, entry=entry, stop=stop, target=target)
            if not walked:
                continue
            conn.execute(
                """
                UPDATE scalp_shadow_rejects
                SET hyp_resolved_at=?, hyp_exit_reason=?, hyp_net_pnl=?, hyp_exit_price=?,
                    hyp_mfe_pct=?, hyp_mae_pct=?, bar_closed=1
                WHERE id=?
                """,
                (
                    now,
                    walked["exit_reason"],
                    float(walked["net_pnl"]),
                    float(walked["exit_price"]),
                    float(walked["mfe_pct"]),
                    float(walked["mae_pct"]),
                    int(r["id"]),
                ),
            )
            resolved += 1
        conn.commit()
    finally:
        conn.close()
    expired = resolve_shadow_rejects_closed_bar(db_path, max_rows=max_rows)
    return {"resolved": resolved, "scanned": scanned, "expired": expired, "ts": time.time()}


__all__ = [
    "attribution_report",
    "counters_today",
    "ensure_scalp_gate_schema",
    "record_gate_event",
    "record_shadow_reject",
    "resolve_shadow_rejects_async",
    "resolve_shadow_rejects_closed_bar",
    "shadow_rejects_summary",
]
