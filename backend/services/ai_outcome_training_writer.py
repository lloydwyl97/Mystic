"""
Write ai_outcome_training_rows on trade close with correct good/bad labels.

Observation/learning only — does not change execution gates.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

logger = logging.getLogger(__name__)


def _normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return s


def classify_outcome_label(
    *,
    close_reason: str,
    net_profit_usd: float | None,
    net_profit_pct: float | None,
    manual_sell: bool = False,
) -> tuple[str, int, str]:
    """
    Returns (good_bad_memory_class, outcome_label, outcome_class).
    """
    reason = (close_reason or "").upper()
    if reason in ("ADMIN_POSITION_CLEAR", "ADMIN_CLEAR", "STALE_PRE_CORRECTION_POSITION_CLEAR"):
        return "BAD", 0, "ADMIN"
    if manual_sell or reason == "HUMAN_MANUAL_SELL":
        if net_profit_usd is not None and float(net_profit_usd) > 0:
            return "GOOD", 1, "WIN"
        return "BAD", 0, "LOSS"
    if reason == "DUST_WRITEOFF":
        return "BAD", 0, "DUST"
    profitable = net_profit_usd is not None and float(net_profit_usd) > 0
    pct_ok = net_profit_pct is not None and float(net_profit_pct) >= float(MIN_NET_PROFIT_TO_SELL)
    if reason == "AI_NET_PROFIT_SELL" or "NET_PROFIT" in reason or reason.startswith("TP"):
        if profitable and pct_ok:
            return "GOOD", 1, "WIN"
        if profitable:
            return "GOOD", 1, "WIN"
        return "BAD", 0, "LOSS"
    if net_profit_usd is not None and float(net_profit_usd) > 0:
        return "GOOD", 1, "WIN"
    return "BAD", 0, "LOSS"


def record_outcome_training_row(
    *,
    symbol: str,
    opened_at_utc: str,
    closed_at_utc: str,
    hold_seconds: float | None,
    entry_price: float | None,
    exit_price: float | None,
    net_profit_usd: float | None,
    net_profit_pct: float | None,
    gross_pnl_pct: float | None,
    close_reason: str,
    strategy_id: str = "day",
    features_json: str | None = None,
    context_json: str | None = None,
    explainability: dict[str, Any] | None = None,
    manual_sell: bool = False,
    db_path: str = DATABASE_PATH,
) -> int | None:
    """Insert or replace outcome row. Returns row id or None on failure."""
    try:
        ensure_ai_canonical_tables(db_path)
        sym = _normalize_symbol(symbol)
        gb, ol, oc = classify_outcome_label(
            close_reason=close_reason,
            net_profit_usd=net_profit_usd,
            net_profit_pct=net_profit_pct,
            manual_sell=manual_sell,
        )
        ex = explainability or {}
        ingested = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO ai_outcome_training_rows (
                    symbol, opened_at_utc, closed_at_utc, hold_seconds,
                    entry_price, exit_price, realized_pct, outcome_label, outcome_class,
                    features_json, context_json, strategy_id,
                    good_bad_memory_class, gross_pnl_pct, net_pnl_pct,
                    trade_was_worth_taking, ingested_at_utc,
                    score_components_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, opened_at_utc, closed_at_utc) DO UPDATE SET
                    good_bad_memory_class=excluded.good_bad_memory_class,
                    outcome_label=excluded.outcome_label,
                    outcome_class=excluded.outcome_class,
                    net_pnl_pct=excluded.net_pnl_pct,
                    gross_pnl_pct=excluded.gross_pnl_pct,
                    trade_was_worth_taking=excluded.trade_was_worth_taking,
                    ingested_at_utc=excluded.ingested_at_utc
                """,
                (
                    sym,
                    opened_at_utc,
                    closed_at_utc,
                    hold_seconds,
                    entry_price,
                    exit_price,
                    net_profit_pct,
                    ol,
                    oc,
                    features_json,
                    context_json,
                    strategy_id,
                    gb,
                    gross_pnl_pct,
                    net_profit_pct,
                    1 if ol == 1 else 0,
                    ingested,
                    json.dumps(
                        {
                            "feature_version": ex.get("feature_version"),
                            "feature_dim": ex.get("feature_dim"),
                            "artifact_path": ex.get("artifact_path"),
                            "artifact_sha256": ex.get("artifact_sha256"),
                            "model_trained_at": ex.get("model_trained_at"),
                            "model_accuracy": ex.get("model_accuracy"),
                            "close_reason": close_reason,
                        },
                        separators=(",", ":"),
                    ),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0) or None
    except Exception as exc:
        logger.warning("record_outcome_training_row failed symbol=%s: %s", symbol, exc)
        return None


def repair_mislabeled_profitable_ai_sells(db_path: str = DATABASE_PATH) -> list[int]:
    """
    Correct profitable AI_NET_PROFIT_SELL rows linked to paper_trades.
    Returns list of ai_outcome_training_rows.id values updated.
    """
    changed: list[int] = []
    try:
        ensure_ai_canonical_tables(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            sells = conn.execute(
                """
                SELECT trade_id, symbol, entry_timestamp, timestamp, pnl, pnl_pct, exit_type, explainability_json
                FROM paper_trades
                WHERE side='SELL' AND pnl IS NOT NULL AND pnl > 0
                  AND COALESCE(exit_type,'') NOT IN ('ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR')
                ORDER BY timestamp DESC LIMIT 100
                """
            ).fetchall()
            for s in sells:
                ex = {}
                try:
                    ex = json.loads(s["explainability_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass
                close_reason = str(ex.get("exit_trigger") or ex.get("close_reason") or s["exit_type"] or "")
                if "NET_PROFIT" not in close_reason.upper() and str(s["exit_type"] or "") not in (
                    "TAKE_PROFIT_FULL",
                    "TAKE_PROFIT_1",
                    "TP1",
                    "TP",
                ):
                    continue
                sym = _normalize_symbol(str(s["symbol"]))
                entry_ts = s["entry_timestamp"]
                closed_ts = s["timestamp"]
                if not entry_ts or not closed_ts:
                    continue
                if isinstance(entry_ts, (int, float)) or (isinstance(entry_ts, str) and entry_ts.replace(".", "", 1).isdigit()):
                    try:
                        entry_iso = datetime.fromtimestamp(float(entry_ts), tz=timezone.utc).isoformat()
                    except (TypeError, ValueError, OSError):
                        entry_iso = str(entry_ts)
                else:
                    entry_iso = str(entry_ts)
                if isinstance(closed_ts, (int, float)) or (isinstance(closed_ts, str) and closed_ts.replace(".", "", 1).isdigit()):
                    try:
                        closed_iso = datetime.fromtimestamp(float(closed_ts), tz=timezone.utc).isoformat()
                    except (TypeError, ValueError, OSError):
                        closed_iso = str(closed_ts)
                else:
                    closed_iso = str(closed_ts)
                pnl = float(s["pnl"] or 0)
                pnl_pct = float(s["pnl_pct"] or 0)
                gb, ol, oc = classify_outcome_label(
                    close_reason=str(s["exit_type"] or "AI_NET_PROFIT_SELL"),
                    net_profit_usd=pnl,
                    net_profit_pct=pnl_pct,
                )
                if gb != "GOOD":
                    continue
                row = conn.execute(
                    """
                    SELECT id, good_bad_memory_class FROM ai_outcome_training_rows
                    WHERE symbol=? AND closed_at_utc LIKE ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (sym, closed_iso[:10] + "%"),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        """
                        SELECT id, good_bad_memory_class FROM ai_outcome_training_rows
                        WHERE symbol=? ORDER BY id DESC LIMIT 1
                        """,
                        (sym,),
                    ).fetchone()
                if row is None:
                    rid = record_outcome_training_row(
                        symbol=sym,
                        opened_at_utc=entry_iso,
                        closed_at_utc=closed_iso,
                        hold_seconds=max(0.0, float(closed_ts) - float(entry_ts)) if entry_ts and closed_ts else None,
                        entry_price=float(ex.get("entry_price") or 0) or None,
                        exit_price=None,
                        net_profit_usd=pnl,
                        net_profit_pct=pnl_pct,
                        gross_pnl_pct=pnl_pct,
                        close_reason="AI_NET_PROFIT_SELL",
                        explainability=ex,
                        db_path=db_path,
                    )
                    if rid:
                        changed.append(rid)
                    continue
                if str(row["good_bad_memory_class"] or "").upper() == "GOOD":
                    continue
                conn.execute(
                    """
                    UPDATE ai_outcome_training_rows
                    SET good_bad_memory_class=?, outcome_label=?, outcome_class=?, trade_was_worth_taking=1,
                        net_pnl_pct=?, ingested_at_utc=?
                    WHERE id=?
                    """,
                    (gb, ol, oc, pnl_pct, datetime.now(timezone.utc).isoformat(), row["id"]),
                )
                changed.append(int(row["id"]))
            conn.commit()
    except Exception as exc:
        logger.warning("repair_mislabeled_profitable_ai_sells failed: %s", exc)
    return changed


def repair_missing_sell_feature_versions(
    db_path: str = DATABASE_PATH,
    *,
    default_feature_version: int = 5,
    default_feature_dim: int = 145,
    limit: int = 200,
) -> list[str]:
    """
    Backfill feature_version/feature_dim on historical SELL explainability_json rows.
    Observation-only repair for diagnostics; does not alter trading behavior.
    """
    changed: list[str] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT trade_id, explainability_json
                FROM paper_trades
                WHERE side='SELL' AND explainability_json IS NOT NULL AND explainability_json != ''
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for row in rows:
                trade_id = str(row["trade_id"] or "")
                try:
                    explain = json.loads(row["explainability_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(explain, dict):
                    continue
                if explain.get("feature_version") and explain.get("feature_dim"):
                    continue
                explain["feature_version"] = int(explain.get("feature_version") or default_feature_version)
                explain["feature_dim"] = int(explain.get("feature_dim") or default_feature_dim)
                conn.execute(
                    "UPDATE paper_trades SET explainability_json=? WHERE trade_id=?",
                    (json.dumps(explain, separators=(",", ":")), trade_id),
                )
                changed.append(trade_id)
            conn.commit()
    except Exception as exc:
        logger.warning("repair_missing_sell_feature_versions failed: %s", exc)
    return changed


__all__ = [
    "classify_outcome_label",
    "record_outcome_training_row",
    "repair_mislabeled_profitable_ai_sells",
    "repair_missing_sell_feature_versions",
]
