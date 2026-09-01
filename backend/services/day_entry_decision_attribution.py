"""Join DAY BUY rows to decision/inference evidence. Observability only.

Does not change order execution. Ambiguous joins are marked, never guessed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


def _load_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _features_hash(raw: Any) -> str:
    if not raw:
        return ""
    if isinstance(raw, dict):
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    else:
        payload = str(raw)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def backfill_day_buy_attribution(db_path: str, *, trade_date: str) -> list[dict[str, Any]]:
    """Attach decision IDs to today's BUY rows when the join is unique."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: list[dict[str, Any]] = []
    try:
        buys = conn.execute(
            """
            SELECT trade_id, symbol, timestamp, decision_id, explainability_json
            FROM paper_trades
            WHERE UPPER(side)='BUY'
              AND date(timestamp)=?
              AND COALESCE(strategy_id, 'day') LIKE '%day%'
            """,
            (trade_date,),
        ).fetchall()
        has_decisions = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='day_decision_records'").fetchone()
        for buy in buys:
            explain = _load_json(buy["explainability_json"])
            decision_id = str(buy["decision_id"] or explain.get("decision_id") or "")
            symbol = str(buy["symbol"] or "")
            ts = str(buy["timestamp"] or "")
            matches: list[sqlite3.Row] = []
            if has_decisions:
                if decision_id:
                    matches = conn.execute(
                        """
                        SELECT decision_id, symbol, created_at, detail_json, ml_score
                        FROM day_decision_records
                        WHERE decision_id=? AND symbol=?
                        """,
                        (decision_id, symbol),
                    ).fetchall()
                if not matches:
                    matches = conn.execute(
                        """
                        SELECT decision_id, symbol, created_at, detail_json, ml_score
                        FROM day_decision_records
                        WHERE symbol=? AND date(created_at)=date(?)
                        """,
                        (symbol, ts),
                    ).fetchall()
            status = "unmatched"
            chosen = None
            if len(matches) == 1:
                status = "unique"
                chosen = matches[0]
            elif len(matches) > 1:
                status = "ambiguous"
            infer_id = ""
            feat_hash = ""
            if chosen is not None:
                try:
                    detail = _load_json(chosen["detail_json"])
                except (IndexError, KeyError):
                    detail = {}
                infer_id = str(detail.get("inference_log_id") or detail.get("ai_inference_log_id") or "")
                feat_hash = str(detail.get("features_json_hash") or "")
                if not feat_hash and infer_id:
                    row = conn.execute(
                        "SELECT id, features_json FROM ai_inference_log WHERE id=? LIMIT 1",
                        (infer_id,),
                    ).fetchone()
                    if row:
                        feat_hash = _features_hash(row["features_json"] if "features_json" in row else row[1])
                explain["decision_id"] = str(chosen["decision_id"])
                explain["ml_score"] = chosen["ml_score"] if "ml_score" in chosen else explain.get("ml_score")
                explain["ai_inference_log_id"] = infer_id
                explain["features_json_hash"] = feat_hash
                explain["attribution_join_status"] = status
                conn.execute(
                    """
                    UPDATE paper_trades
                    SET decision_id=?, explainability_json=?
                    WHERE trade_id=? AND UPPER(side)='BUY'
                    """,
                    (str(chosen["decision_id"]), json.dumps(explain), buy["trade_id"]),
                )
            else:
                explain["attribution_join_status"] = status
                conn.execute(
                    "UPDATE paper_trades SET explainability_json=? WHERE trade_id=? AND UPPER(side)='BUY'",
                    (json.dumps(explain), buy["trade_id"]),
                )
            out.append(
                {
                    "trade_id": buy["trade_id"],
                    "symbol": symbol,
                    "status": status,
                    "decision_id": str(chosen["decision_id"]) if chosen is not None else decision_id,
                    "features_json_hash": feat_hash,
                }
            )
        conn.commit()
    finally:
        conn.close()
    return out
