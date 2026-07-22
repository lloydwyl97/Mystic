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

    ``outcome_class`` is exit-conditioned (STALL_LOSS / GIVEBACK_LOSS / …) so
    trainers can up-weight structural losers vs NET_PROFIT wins.
    """
    reason = (close_reason or "").upper()
    if reason in ("ADMIN_POSITION_CLEAR", "ADMIN_CLEAR", "STALE_PRE_CORRECTION_POSITION_CLEAR"):
        return "BAD", 0, "ADMIN"
    if manual_sell or reason == "HUMAN_MANUAL_SELL":
        if net_profit_usd is not None and float(net_profit_usd) > 0:
            return "GOOD", 1, "MANUAL_WIN"
        return "BAD", 0, "MANUAL_LOSS"
    if reason == "DUST_WRITEOFF":
        return "BAD", 0, "DUST"
    profitable = net_profit_usd is not None and float(net_profit_usd) > 0
    pct_ok = net_profit_pct is not None and float(net_profit_pct) >= float(MIN_NET_PROFIT_TO_SELL)
    if reason == "AI_NET_PROFIT_SELL" or "NET_PROFIT" in reason or reason.startswith("TP"):
        if profitable:
            return "GOOD", 1, "NET_PROFIT_WIN"
        return "BAD", 0, "NET_PROFIT_FAIL"
    if "STALL" in reason:
        return ("GOOD", 1, "STALL_WIN") if profitable else ("BAD", 0, "STALL_LOSS")
    if "GIVEBACK" in reason:
        return ("GOOD", 1, "GIVEBACK_WIN") if profitable else ("BAD", 0, "GIVEBACK_LOSS")
    if "TIME_STOP" in reason:
        return ("GOOD", 1, "TIME_STOP_WIN") if profitable else ("BAD", 0, "TIME_STOP_LOSS")
    if profitable:
        return "GOOD", 1, "WIN"
    return "BAD", 0, "LOSS"


def _propagate_symbol_strategy_expectancy(conn: sqlite3.Connection, sym: str, strategy_id: str) -> None:
    """
    Recompute and upsert ai_symbol_strategy_expectancy for (symbol, strategy_id)
    from ai_outcome_training_rows so learning actually feeds the next decision.

    Schema-agnostic: the live table predates ai_canonical_storage's newer columns,
    so only columns that actually exist are written.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_symbol_strategy_expectancy)").fetchall()}
        if not cols:
            return
        agg = conn.execute(
            """
            SELECT
                COALESCE(AVG(CASE WHEN net_pnl_pct IS NOT NULL THEN net_pnl_pct
                                  WHEN realized_pct IS NOT NULL THEN realized_pct END), 0.0) AS expectancy,
                SUM(CASE WHEN UPPER(COALESCE(good_bad_memory_class,''))='GOOD' THEN 1 ELSE 0 END) AS good_count,
                SUM(CASE WHEN UPPER(COALESCE(good_bad_memory_class,''))='BAD'  THEN 1 ELSE 0 END) AS bad_count,
                COUNT(*) AS total_trades
            FROM ai_outcome_training_rows
            WHERE (UPPER(symbol)=UPPER(?) OR UPPER(symbol)=UPPER(?))
              AND LOWER(COALESCE(strategy_id,'day'))=LOWER(?)
            """,
            (sym, sym.replace("/", ""), strategy_id),
        ).fetchone()
        if not agg:
            return
        expectancy, good_count, bad_count, total_trades = (
            float(agg[0] or 0.0),
            int(agg[1] or 0),
            int(agg[2] or 0),
            int(agg[3] or 0),
        )
        now = datetime.now(timezone.utc).isoformat()
        values: dict[str, Any] = {
            "symbol": sym,
            "strategy_id": strategy_id,
            "expectancy": expectancy,
            "good_count": good_count,
            "bad_count": bad_count,
        }
        if "total_trades" in cols:
            values["total_trades"] = total_trades
        if "last_net_outcome" in cols:
            values["last_net_outcome"] = expectancy
        if "last_outcome_at_utc" in cols:
            values["last_outcome_at_utc"] = now
        if "updated_at_utc" in cols:
            values["updated_at_utc"] = now
        if "updated_at" in cols:
            values["updated_at"] = now
        ins_cols = [c for c in values if c in cols]
        placeholders = ", ".join("?" for _ in ins_cols)
        update_cols = [c for c in ins_cols if c not in ("symbol", "strategy_id")]
        set_clause = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        conn.execute(
            f"INSERT INTO ai_symbol_strategy_expectancy ({', '.join(ins_cols)}) VALUES ({placeholders}) ON CONFLICT(symbol, strategy_id) DO UPDATE SET {set_clause}",
            tuple(values[c] for c in ins_cols),
        )
    except Exception as exc:
        logger.warning("propagate_symbol_strategy_expectancy failed symbol=%s: %s", sym, exc)


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
                    features_json=COALESCE(excluded.features_json, ai_outcome_training_rows.features_json),
                    context_json=COALESCE(excluded.context_json, ai_outcome_training_rows.context_json),
                    score_components_json=COALESCE(excluded.score_components_json, ai_outcome_training_rows.score_components_json),
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
                            "ai_confidence": ex.get("ai_confidence"),
                            "confidence": ex.get("confidence"),
                            "entry_buy_margin": ex.get("entry_buy_margin"),
                            "buy_margin": ex.get("buy_margin"),
                            "trend_score": ex.get("trend_score"),
                            "chop_score": ex.get("chop_score"),
                            "coin_expectancy": ex.get("coin_expectancy"),
                            "signal_ctx_rs_btc": ex.get("signal_ctx_rs_btc"),
                            "entry_spread_pct": ex.get("entry_spread_pct"),
                            "selected_net_expected_value": ex.get("selected_net_expected_value"),
                            "day_route_regime": ex.get("day_route_regime"),
                            "setup_type": ex.get("setup_type") or ex.get("entry_thesis"),
                            "final_selection_score": ex.get("final_selection_score"),
                            "feature_health_pass": ex.get("feature_health_pass"),
                            "feature_health_pct": ex.get("feature_health_pct"),
                            "feature_health_bad_count": ex.get("feature_health_bad_count"),
                            "setup_score": ex.get("setup_score"),
                            "execution_quality_score": ex.get("execution_quality_score"),
                            "feature_health_score": ex.get("feature_health_score"),
                            "block_scores_json": ex.get("block_scores_json"),
                            "intelligence_rank_delta": ex.get("intelligence_rank_delta"),
                        },
                        separators=(",", ":"),
                    ),
                ),
            )
            _propagate_symbol_strategy_expectancy(conn, sym, strategy_id or "day")
            conn.commit()
            row_id = int(cur.lastrowid or 0) or None
        try:
            from backend.services.ai_strategy_score_weight_writer import propagate_adaptive_score_weights_for_close

            ex_propagate = dict(ex)
            ex_propagate["good_bad_memory_class"] = gb
            ex_propagate["net_pnl_pct"] = net_profit_pct
            propagate_adaptive_score_weights_for_close(
                symbol=sym,
                strategy_id=strategy_id or "day",
                explainability=ex_propagate,
                db_path=db_path,
            )
        except Exception as exc:
            logger.warning(
                "propagate_adaptive_score_weights_for_close deferred failed symbol=%s: %s",
                sym,
                exc,
            )
        return row_id
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
                strat_row = conn.execute(
                    "SELECT COALESCE(strategy_id,'day') FROM ai_outcome_training_rows WHERE id=?",
                    (row["id"],),
                ).fetchone()
                sid = str(strat_row[0] if strat_row else "day")
                _propagate_symbol_strategy_expectancy(conn, sym, sid)
            conn.commit()
    except Exception as exc:
        logger.warning("repair_mislabeled_profitable_ai_sells failed: %s", exc)
    return changed


def backfill_all_symbol_strategy_expectancy(db_path: str = DATABASE_PATH) -> int:
    """Recompute ai_symbol_strategy_expectancy for every symbol/strategy in outcome rows."""
    updated = 0
    try:
        ensure_ai_canonical_tables(db_path)
        with sqlite3.connect(db_path) as conn:
            pairs = conn.execute(
                """
                SELECT DISTINCT symbol, COALESCE(strategy_id, 'day')
                FROM ai_outcome_training_rows
                WHERE symbol IS NOT NULL AND TRIM(symbol) != ''
                """
            ).fetchall()
            for sym, sid in pairs:
                _propagate_symbol_strategy_expectancy(conn, str(sym), str(sid or "day"))
                updated += 1
            conn.commit()
    except Exception as exc:
        logger.warning("backfill_all_symbol_strategy_expectancy failed: %s", exc)
    return updated


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


def backfill_outcome_features_from_inference(db_path: str = DATABASE_PATH, *, window_sec: int = 900) -> int:
    """
    Fill NULL features_json on ai_outcome_training_rows from nearest ai_inference_log row.
    Matches by normalized symbol within ±window_sec of opened_at_utc.
    Returns number of rows updated.
    """
    updated = 0
    try:
        ensure_ai_canonical_tables(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, symbol, opened_at_utc, strategy_id
                FROM ai_outcome_training_rows
                WHERE features_json IS NULL OR features_json = '' OR features_json = 'null'
                """
            ).fetchall()
            for row in rows:
                sym = str(row["symbol"] or "")
                bus = sym.replace("/", "").upper()
                opened = str(row["opened_at_utc"] or "")
                if not opened:
                    continue
                # Prefer inference near open; allow either BTCUSDT or BTC/USDT forms.
                inf = conn.execute(
                    """
                    SELECT features_json, feature_version, feature_dim, decision_id, strategy_id
                    FROM ai_inference_log
                    WHERE features_json IS NOT NULL AND length(features_json) > 2
                      AND (
                        UPPER(REPLACE(symbol, '/', '')) = ?
                        OR UPPER(symbol) = ?
                        OR UPPER(symbol) = ?
                      )
                      AND ABS(
                        (julianday(REPLACE(SUBSTR(ts_utc, 1, 19), 'T', ' ')) -
                         julianday(REPLACE(SUBSTR(?, 1, 19), 'T', ' '))) * 86400.0
                      ) <= ?
                    ORDER BY ABS(
                      (julianday(REPLACE(SUBSTR(ts_utc, 1, 19), 'T', ' ')) -
                       julianday(REPLACE(SUBSTR(?, 1, 19), 'T', ' '))) * 86400.0
                    ) ASC
                    LIMIT 1
                    """,
                    (bus, bus, sym.upper(), opened, float(window_sec), opened),
                ).fetchone()
                if not inf or not inf["features_json"]:
                    continue
                ctx = json.dumps(
                    {
                        "_feature_version": int(inf["feature_version"] or 5),
                        "feature_version": int(inf["feature_version"] or 5),
                        "feature_dim": int(inf["feature_dim"] or 0) or None,
                        "decision_id": inf["decision_id"],
                        "_live_ai_strategy": str(inf["strategy_id"] or row["strategy_id"] or "day"),
                        "backfilled_from_inference": True,
                    },
                    separators=(",", ":"),
                )
                conn.execute(
                    "UPDATE ai_outcome_training_rows SET features_json=?, context_json=COALESCE(context_json, ?) WHERE id=?",
                    (inf["features_json"], ctx, int(row["id"])),
                )
                updated += 1
            conn.commit()
    except Exception as exc:
        logger.warning("backfill_outcome_features_from_inference failed: %s", exc)
    return updated


__all__ = [
    "classify_outcome_label",
    "record_outcome_training_row",
    "repair_mislabeled_profitable_ai_sells",
    "repair_missing_sell_feature_versions",
    "backfill_outcome_features_from_inference",
]
