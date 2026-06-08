"""
Observation-only recorder when DAY buys are blocked by capacity or cooldown.

Does not execute trades or change gates.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

logger = logging.getLogger(__name__)

TABLE = "ai_missed_opportunity_observations"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at_utc TEXT NOT NULL,
    block_reason TEXT NOT NULL,
    attempted_symbol TEXT,
    active_positions INTEGER,
    max_positions INTEGER,
    signals_json TEXT NOT NULL,
    evaluation_json TEXT,
    evaluated_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS ix_missed_opp_obs_at ON {TABLE}(observed_at_utc);
"""


def ensure_missed_opportunity_table(db_path: str = DATABASE_PATH) -> None:
    ensure_ai_canonical_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        for stmt in _SCHEMA.strip().split(";\n"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()


def _redis_field(mapping: dict, key: str) -> str | None:
    if not mapping:
        return None
    v = mapping.get(key)
    if v is None:
        v = mapping.get(key.encode())  # type: ignore[index]
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


def _snapshot_top4_signals() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if r is None:
            return rows
        for sym in TRADING_SYMBOLS:
            key = f"ai_signal:day:{sym}"
            raw = r.hgetall(key) or {}
            if not raw:
                continue
            side = _redis_field(raw, "side") or _redis_field(raw, "prediction") or _redis_field(raw, "action")
            conf_raw = _redis_field(raw, "confidence") or _redis_field(raw, "winner_probability")
            ts = _redis_field(raw, "timestamp") or _redis_field(raw, "ctx_ts_utc") or _redis_field(raw, "signal_content_timestamp")
            base = sym.replace("USDT", "")
            px = 0.0
            raw_px = r.hget(f"price:{base}", "v")
            if raw_px:
                with contextlib.suppress(TypeError, ValueError):
                    px = float(raw_px.decode() if isinstance(raw_px, bytes) else raw_px)
            rows.append(
                {
                    "symbol": sym,
                    "side": side,
                    "confidence": float(conf_raw or 0),
                    "prob_buy": float(_redis_field(raw, "prob_buy") or conf_raw or 0),
                    "feature_version": int(float(_redis_field(raw, "feature_version") or 0)),
                    "signal_price": float(_redis_field(raw, "price") or _redis_field(raw, "v") or px or 0),
                    "timestamp": ts,
                }
            )
        rows.sort(key=lambda x: float(x.get("confidence") or 0), reverse=True)
    except Exception as exc:
        logger.debug("missed_opp signal snapshot failed: %s", exc)
    return rows


def record_missed_opportunity_observation(
    *,
    block_reason: str,
    attempted_symbol: str | None = None,
    active_positions: int | None = None,
    max_positions: int | None = None,
    db_path: str = DATABASE_PATH,
) -> None:
    """Best-effort insert when capacity/cooldown blocks a buy. Never raises."""
    reason = (block_reason or "").strip()
    watch_prefixes = (
        "MAX_POSITIONS",
        "POST_SELL_COOLDOWN",
        "GLOBAL_SELL_COOLDOWN",
        "POSITION_ALREADY_OPEN",
        "DISCIPLINE_GATE",
        "SLEEVE_LIMIT",
        "CASH_INVARIANT",
        "HARD_FUSE",
        "EXCHANGE_CONSTRAINT",
        "REPAIR_ADD_",
        "already_open",
        "QUALITY_",
        "ENTRY_CONTEXT",
        "SIGNAL_CONTENT",
        "DAY_CAPITAL_IDLE",
        "DAY_OPPORTUNITY_COST",
        "ENTRY_QUALITY",
        "SETUP_CREDIT",
        "PROTECTED_PREFLIGHT",
        "SPREAD_TOO_WIDE",
    )
    if not any(reason.startswith(w) or w in reason for w in watch_prefixes):
        return
    try:
        ensure_missed_opportunity_table(db_path)
        signals = _snapshot_top4_signals()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                f"""
                INSERT INTO {TABLE} (
                    observed_at_utc, block_reason, attempted_symbol,
                    active_positions, max_positions, signals_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    reason[:240],
                    attempted_symbol,
                    active_positions,
                    max_positions,
                    json.dumps(signals, separators=(",", ":")),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("record_missed_opportunity_observation failed: %s", exc)


def _evaluate_row(row: dict[str, Any], db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Hypothetical post-block price move — observation only."""
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST_PCT, MIN_NET_PROFIT_TO_SELL

    signals = json.loads(row.get("signals_json") or "[]")
    if not isinstance(signals, list):
        signals = []
    prices_now: dict[str, float] = {}
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if r:
            for sym in TRADING_SYMBOLS:
                base = sym.replace("USDT", "")
                px = r.hget(f"price:{base}", "v")
                if px:
                    prices_now[sym] = float(px)
    except Exception:
        pass

    attempted = str(row.get("attempted_symbol") or "")
    preflight_would_pass = None
    preflight_note = "not_evaluated"
    try:
        with sqlite3.connect(db_path) as conn:
            if attempted:
                rej = conn.execute(
                    """
                    SELECT COUNT(*) FROM portfolio_engine_rejects
                    WHERE symbol=? AND side='BUY' AND filter_name='PROTECTED_PREFLIGHT'
                      AND ts >= datetime('now', '-2 hours')
                    """,
                    (attempted,),
                ).fetchone()
                cnt = int(rej[0] or 0) if rej else 0
                preflight_would_pass = cnt == 0
                preflight_note = "no_recent_protected_preflight_reject" if preflight_would_pass else "recent_protected_preflight_reject"
    except Exception:
        preflight_note = "preflight_lookup_failed"

    evaluated: list[dict[str, Any]] = []
    best_sym = None
    best_move = float("-inf")
    best_gross = None
    for s in signals:
        sym = str(s.get("symbol") or "")
        side = str(s.get("side") or "").upper()
        conf = float(s.get("confidence") or 0)
        entry = float(s.get("signal_price") or 0)
        now = float(prices_now.get(sym) or 0)
        gross_pct = (now - entry) / entry if entry > 0 and now > 0 else 0.0
        net_pct = gross_pct - float(ESTIMATED_ROUNDTRIP_COST_PCT)
        would_hit_floor = net_pct >= float(MIN_NET_PROFIT_TO_SELL)
        item = {
            "symbol": sym,
            "signal": side,
            "confidence": round(conf, 4),
            "entry_price": entry,
            "later_price": now,
            "later_max_move_gross_pct": round(gross_pct, 6),
            "gross_pct": round(gross_pct, 6),
            "net_pct": round(net_pct, 6),
            "would_hit_profit_floor": would_hit_floor,
            "protected_preflight_would_pass": preflight_would_pass if sym == attempted or not attempted else None,
        }
        evaluated.append(item)
        if net_pct > best_move:
            best_move = net_pct
            best_sym = sym
            best_gross = gross_pct

    return {
        "observation_id": row.get("id"),
        "observed_at_utc": row.get("observed_at_utc"),
        "block_reason": row.get("block_reason"),
        "attempted_symbol": attempted,
        "active_positions": row.get("active_positions"),
        "max_positions": row.get("max_positions"),
        "signals": evaluated,
        "best_symbol_after_fact": best_sym,
        "best_net_pct_after_fact": round(best_move, 6) if best_move > float("-inf") else None,
        "best_gross_move_pct": round(best_gross, 6) if best_gross is not None else None,
        "protected_preflight_would_pass": preflight_would_pass,
        "protected_preflight_note": preflight_note,
        "note": "observation_only_no_execution",
    }


_BACKFILL_FILTER_NAMES = (
    "POSITION_ALREADY_OPEN",
    "MAX_POSITIONS_REACHED",
    "POST_SELL_COOLDOWN_LEDGER",
    "POST_SELL_COOLDOWN",
    "GLOBAL_SELL_COOLDOWN",
    "PROTECTED_PREFLIGHT",
    "DISCIPLINE_GATE",
    "SLEEVE_LIMIT",
    "CASH_INVARIANT",
    "HARD_FUSE",
    "EXCHANGE_CONSTRAINT",
)


def backfill_missed_opportunity_observations(
    *,
    limit: int = 30,
    db_path: str = DATABASE_PATH,
) -> dict[str, Any]:
    """Backfill observation rows from portfolio_engine_rejects (read-only source)."""
    ensure_missed_opportunity_table(db_path)
    limit = max(1, min(100, int(limit)))
    inserted = 0
    skipped = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rejects = conn.execute(
            """
            SELECT ts, symbol, reason, filter_name
            FROM portfolio_engine_rejects
            WHERE UPPER(side) = 'BUY'
              AND (
                filter_name IN ({})
                OR filter_name LIKE 'REPAIR_ADD_%'
                OR filter_name LIKE 'DISCIPLINE_%'
              )
            ORDER BY ts DESC
            LIMIT ?
            """.format(",".join("?" * len(_BACKFILL_FILTER_NAMES))),
            (*_BACKFILL_FILTER_NAMES, limit * 5),
        ).fetchall()
        for row in rejects:
            if inserted >= limit:
                break
            sym = str(row["symbol"] or "")
            filt = str(row["filter_name"] or "")
            ts = str(row["ts"] or "")
            reason = str(row["reason"] or "")
            block = f"{filt}:{reason}" if filt and filt not in reason else (filt or reason)
            dup = conn.execute(
                f"""
                SELECT 1 FROM {TABLE}
                WHERE attempted_symbol = ? AND block_reason LIKE ?
                  AND observed_at_utc >= datetime(?, '-1 hour')
                LIMIT 1
                """,
                (sym, f"{filt}%", ts),
            ).fetchone()
            if dup:
                skipped += 1
                continue
            conn.execute(
                f"""
                INSERT INTO {TABLE} (
                    observed_at_utc, block_reason, attempted_symbol,
                    active_positions, max_positions, signals_json
                ) VALUES (?, ?, ?, NULL, NULL, ?)
                """,
                (
                    ts,
                    block[:240],
                    sym,
                    json.dumps([], separators=(",", ":")),
                ),
            )
            inserted += 1
        conn.commit()
    return {
        "requested": limit,
        "inserted": inserted,
        "skipped_duplicate": skipped,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_missed_opportunity_report(limit: int = 50, db_path: str = DATABASE_PATH) -> dict[str, Any]:
    ensure_missed_opportunity_table(db_path)
    limit = max(1, min(200, int(limit)))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    items = [_evaluate_row(dict(r), db_path=db_path) for r in rows]
    return {
        "observation_only": True,
        "count": len(items),
        "observations": items,
    }


__all__ = [
    "backfill_missed_opportunity_observations",
    "ensure_missed_opportunity_table",
    "get_missed_opportunity_report",
    "record_missed_opportunity_observation",
]
