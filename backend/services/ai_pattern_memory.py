"""
AI trade pattern memory — writes closed-trade feature vectors into
``ai_good_trade_patterns`` / ``ai_bad_trade_patterns`` and computes a
similarity-to-history score consumed by the sizing engine's
``memory_factor`` (``portfolio_engine.py:_compute_dynamic_sizing_multiplier``).

Prior to this module, ``decision_data['good_pattern_similarity']`` and
``decision_data['bad_pattern_similarity']`` were read by the sizing engine
but never written anywhere in the codebase — the tables existed (schema-only,
0 rows) but nothing populated them, so ``memory_factor`` was silently always
neutral (1.0) for every trade. This module closes that gap.

Both the write and read paths are best-effort: any failure returns a neutral
result and never raises into the live trading engine. Schema drift between
hosts (columns added by hand outside of ``ai_canonical_storage.py`` on some
deployments) is handled via runtime column introspection, mirroring the
``_existing_columns`` pattern already used in ``ai_canonical_storage.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv("MYSTIC_DB_PATH", "mystic_trading.db")

# Feature keys used for the pattern vector — all already computed at entry
# time on TradeExplainability, so no extra feature engineering is required.
_VECTOR_KEYS = ("chop_score", "coin_edge_score", "trend_score", "confidence")

PATTERN_MEMORY_ENABLED = os.getenv("AI_PATTERN_MEMORY_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
_SIMILARITY_LOOKBACK = int(os.getenv("AI_PATTERN_MEMORY_LOOKBACK", "20"))
_MIN_SYMBOL_SAMPLES = int(os.getenv("AI_PATTERN_MEMORY_MIN_SYMBOL_SAMPLES", "3"))


def build_pattern_vector(
    *,
    chop_score: float | None,
    coin_edge_score: float | None,
    trend_score: float | None,
    confidence: float | None,
) -> dict[str, float]:
    return {
        "chop_score": float(chop_score or 0.0),
        "coin_edge_score": float(coin_edge_score or 0.0),
        "trend_score": float(trend_score or 0.0),
        "confidence": float(confidence or 0.0),
    }


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _insert_dynamic(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> None:
    """Insert only into columns that actually exist on this host's table.

    ``created_at`` is always supplied explicitly rather than relying on a
    DB-side DEFAULT — some deployments have hand-migrated copies of these
    tables where the DEFAULT clause was dropped, leaving a bare NOT NULL
    column that would otherwise reject the insert.
    """
    cols = _existing_columns(conn, table)
    if not cols:
        return
    use = {k: v for k, v in values.items() if k in cols}
    if "created_at" in cols and "created_at" not in use:
        use["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not use:
        return
    col_names = list(use.keys())
    placeholders = ", ".join(["?"] * len(col_names))
    sql = f"INSERT INTO {table} ({', '.join(col_names)}) VALUES ({placeholders})"
    conn.execute(sql, [use[c] for c in col_names])


def record_trade_pattern(
    *,
    db_path: str = DATABASE_PATH,
    symbol: str,
    strategy_id: str,
    vector: dict[str, float],
    net_outcome_pct: float,
    net_pnl: float,
    hold_seconds: float,
    reason: str,
    trade_id: str = "",
    entry_time_iso: str = "",
    exit_time_iso: str = "",
) -> bool:
    """Write one closed trade's feature vector to the good/bad pattern table.

    Returns True on success, False on any failure (never raises).
    """
    if not PATTERN_MEMORY_ENABLED:
        return False
    try:
        table = "ai_good_trade_patterns" if net_pnl > 0 else "ai_bad_trade_patterns"
        now_iso = exit_time_iso or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        values = {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "closed_at_utc": now_iso,
            "pattern_vector_json": json.dumps(vector, separators=(",", ":")),
            "net_outcome_pct": float(net_outcome_pct),
            "net_pnl": float(net_pnl),
            "hold_seconds": float(hold_seconds or 0.0),
            "reason": str(reason or ""),
            "similarity_key": f"{symbol}:{strategy_id}",
            "trade_id": str(trade_id or ""),
            "entry_time": entry_time_iso,
            "exit_time": now_iso,
        }
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            _insert_dynamic(conn, table, values)
            conn.commit()
        return True
    except Exception:
        logger.debug("PATTERN_MEMORY_WRITE_FAILED symbol=%s", symbol, exc_info=True)
        return False


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in _VECTOR_KEYS)
    norm_a = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in _VECTOR_KEYS))
    norm_b = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in _VECTOR_KEYS))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _fetch_recent_vectors(conn: sqlite3.Connection, table: str, symbol: str, strategy_id: str, limit: int) -> list[dict[str, float]]:
    cols = _existing_columns(conn, table)
    if "pattern_vector_json" not in cols:
        return []
    rows: list[str] = []
    if "symbol" in cols and "strategy_id" in cols:
        rows = [
            r[0]
            for r in conn.execute(
                f"SELECT pattern_vector_json FROM {table} WHERE symbol=? AND strategy_id=? AND pattern_vector_json IS NOT NULL ORDER BY id DESC LIMIT ?",
                (symbol, strategy_id, limit),
            ).fetchall()
        ]
        if len(rows) < _MIN_SYMBOL_SAMPLES:
            # Cold start for this symbol — broaden to strategy-wide history.
            rows = [
                r[0]
                for r in conn.execute(
                    f"SELECT pattern_vector_json FROM {table} WHERE strategy_id=? AND pattern_vector_json IS NOT NULL ORDER BY id DESC LIMIT ?",
                    (strategy_id, limit),
                ).fetchall()
            ]
    out = []
    for raw in rows:
        try:
            out.append(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return out


def _upsert_memory_score(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    strategy_id: str,
    good_similarity: float,
    bad_similarity: float,
) -> None:
    cols = _existing_columns(conn, "ai_trade_memory_scores")
    if not cols:
        return
    existing = conn.execute(
        "SELECT id FROM ai_trade_memory_scores WHERE symbol=? AND strategy_id=? ORDER BY id DESC LIMIT 1",
        (symbol, strategy_id),
    ).fetchone()
    values: dict[str, Any] = {"good_similarity": good_similarity, "bad_similarity": bad_similarity}
    if "memory_score" in cols:
        values["memory_score"] = good_similarity - bad_similarity
    if "memory_bonus" in cols:
        values["memory_bonus"] = max(0.0, good_similarity - bad_similarity)
    if "memory_penalty" in cols:
        values["memory_penalty"] = max(0.0, bad_similarity - good_similarity)
    if "updated_at" in cols:
        values["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if existing:
        use = {k: v for k, v in values.items() if k in cols}
        set_clause = ", ".join(f"{k}=?" for k in use)
        conn.execute(
            f"UPDATE ai_trade_memory_scores SET {set_clause} WHERE id=?",
            [*use.values(), existing[0]],
        )
    else:
        values["symbol"] = symbol
        values["strategy_id"] = strategy_id
        use = {k: v for k, v in values.items() if k in cols}
        if "created_at" in cols and "created_at" not in use:
            use["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        col_names = list(use.keys())
        placeholders = ", ".join(["?"] * len(col_names))
        conn.execute(
            f"INSERT INTO ai_trade_memory_scores ({', '.join(col_names)}) VALUES ({placeholders})",
            [use[c] for c in col_names],
        )


def backfill_pattern_memory_from_outcomes(
    *,
    db_path: str = DATABASE_PATH,
    limit: int = 250,
) -> dict[str, int]:
    """Seed good/bad pattern tables from recent ``ai_outcome_training_rows``.

    Pattern memory was empty on hosts that never closed trades after the writer
    shipped; without history, sizing ``memory_factor`` stays neutral. Idempotent
    via ``trade_id`` / closed_at uniqueness best-effort (duplicates are harmless).
    """
    stats = {"scanned": 0, "written_good": 0, "written_bad": 0, "skipped": 0}
    if not PATTERN_MEMORY_ENABLED:
        return stats
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, strategy_id, closed_at_utc, opened_at_utc,
                       hold_seconds, net_pnl_pct, score_components_json
                FROM ai_outcome_training_rows
                WHERE score_components_json IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
    except Exception:
        logger.debug("PATTERN_MEMORY_BACKFILL_READ_FAILED", exc_info=True)
        return stats

    for row in rows:
        stats["scanned"] += 1
        try:
            (
                row_id,
                symbol,
                strategy_id,
                closed_at,
                opened_at,
                hold_seconds,
                net_pnl_pct,
                score_raw,
            ) = row
            comps = json.loads(score_raw) if isinstance(score_raw, str) else {}
            if not isinstance(comps, dict):
                stats["skipped"] += 1
                continue
            trend = comps.get("trend_score")
            chop = comps.get("chop_score")
            edge = comps.get("coin_edge_score")
            if edge is None:
                edge = comps.get("coin_expectancy")
            conf = comps.get("confidence")
            if conf is None:
                conf = comps.get("ai_confidence")
            if trend is None and chop is None and edge is None and conf is None:
                stats["skipped"] += 1
                continue
            vector = build_pattern_vector(
                chop_score=float(chop or 0.0),
                coin_edge_score=float(edge or 0.0),
                trend_score=float(trend or 0.0),
                confidence=float(conf or 0.0),
            )
            net_pct = float(net_pnl_pct or 0.0)
            # Signed pct is enough for good/bad split (>0).
            net_pnl = net_pct
            sym = str(symbol or "").replace("/", "").replace("-", "").upper()
            if sym.endswith("USDT") is False and sym:
                sym = f"{sym}USDT" if not sym.endswith("USD") else sym
            ok = record_trade_pattern(
                db_path=db_path,
                symbol=sym,
                strategy_id=str(strategy_id or "day").strip().lower() or "day",
                vector=vector,
                net_outcome_pct=net_pct,
                net_pnl=net_pnl,
                hold_seconds=float(hold_seconds or 0.0),
                reason="backfill_outcome_row",
                trade_id=f"outcome:{row_id}",
                entry_time_iso=str(opened_at or ""),
                exit_time_iso=str(closed_at or ""),
            )
            if ok:
                if net_pnl > 0:
                    stats["written_good"] += 1
                else:
                    stats["written_bad"] += 1
            else:
                stats["skipped"] += 1
        except Exception:
            stats["skipped"] += 1
            continue
    logger.info(
        "PATTERN_MEMORY_BACKFILL scanned=%d good=%d bad=%d skipped=%d",
        stats["scanned"],
        stats["written_good"],
        stats["written_bad"],
        stats["skipped"],
    )
    return stats


def compute_pattern_memory_similarity(
    *,
    db_path: str = DATABASE_PATH,
    symbol: str,
    strategy_id: str,
    vector: dict[str, float],
) -> tuple[float, float]:
    """Return (good_pattern_similarity, bad_pattern_similarity) in [0, 1] for the given
    candidate feature vector, based on cosine similarity to recent closed-trade history.

    Best-effort: returns (0.0, 0.0) on any failure or when there is no history yet.
    """
    if not PATTERN_MEMORY_ENABLED:
        return 0.0, 0.0
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            good_vecs = _fetch_recent_vectors(conn, "ai_good_trade_patterns", symbol, strategy_id, _SIMILARITY_LOOKBACK)
            bad_vecs = _fetch_recent_vectors(conn, "ai_bad_trade_patterns", symbol, strategy_id, _SIMILARITY_LOOKBACK)
            good_sim = 0.0
            bad_sim = 0.0
            if good_vecs:
                # Similarity is naturally in [-1, 1]; clamp to [0, 1] since the sizing
                # engine treats these as one-sided "how much does this look like history" scores.
                good_sim = max(0.0, sum(_cosine_similarity(vector, v) for v in good_vecs) / len(good_vecs))
            if bad_vecs:
                bad_sim = max(0.0, sum(_cosine_similarity(vector, v) for v in bad_vecs) / len(bad_vecs))
            with contextlib.suppress(Exception):
                _upsert_memory_score(conn, symbol=symbol, strategy_id=strategy_id, good_similarity=good_sim, bad_similarity=bad_sim)
                conn.commit()
            return good_sim, bad_sim
    except Exception:
        logger.debug("PATTERN_MEMORY_READ_FAILED symbol=%s", symbol, exc_info=True)
        return 0.0, 0.0
