"""
Read-only trade drill-down — single coherent packet per trade_id.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.database_schema import DATABASE_PATH

_ADMIN_EXIT_TYPES = frozenset({"ADMIN_POSITION_CLEAR", "STALE_PRE_CORRECTION_POSITION_CLEAR"})


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row}


def _parse_json_field(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw


def build_trade_drilldown(trade_id: str, db_path: str = DATABASE_PATH) -> dict[str, Any]:
    tid = str(trade_id or "").strip()
    if not tid:
        return {"found": False, "error": "trade_id required"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM paper_trades WHERE trade_id = ? LIMIT 1", (tid,))
        paper = _row_to_dict(cur.fetchone())
        if not paper:
            cur.execute("SELECT * FROM paper_trades WHERE id = ? LIMIT 1", (tid,))
            paper = _row_to_dict(cur.fetchone())

        if not paper:
            return {"found": False, "trade_id": tid, "error": "trade not found"}

        resolved_tid = str(paper.get("trade_id") or tid)
        side = str(paper.get("side") or "").upper()
        symbol = str(paper.get("symbol") or "")

        cur.execute("SELECT * FROM portfolio_engine_audit WHERE trade_id = ? ORDER BY id", (resolved_tid,))
        audit_rows = [_row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT * FROM position_close_ledger WHERE sell_trade_id = ? ORDER BY id DESC LIMIT 1",
            (resolved_tid,),
        )
        close_ledger = _row_to_dict(cur.fetchone())
        if close_ledger is None and side == "BUY":
            cur.execute(
                "SELECT * FROM position_close_ledger WHERE detail LIKE ? ORDER BY id DESC LIMIT 1",
                (f"%buy_trade_id={resolved_tid}%",),
            )
            close_ledger = _row_to_dict(cur.fetchone())

        linked_sell_paper: dict[str, Any] | None = None
        linked_sell_audits: list[dict[str, Any] | None] = []
        if side == "BUY" and close_ledger:
            detail_raw = str(close_ledger.get("detail") or "")
            canonical_sell_tid = None
            for part in detail_raw.split(";"):
                if part.startswith("canonical_sell_trade_id="):
                    canonical_sell_tid = part.split("=", 1)[1].strip()
                    break
            if canonical_sell_tid:
                cur.execute("SELECT * FROM paper_trades WHERE trade_id = ? LIMIT 1", (canonical_sell_tid,))
                linked_sell_paper = _row_to_dict(cur.fetchone())
                cur.execute(
                    "SELECT * FROM portfolio_engine_audit WHERE trade_id = ? ORDER BY id",
                    (canonical_sell_tid,),
                )
                linked_sell_audits = [_row_to_dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT * FROM trade_performance WHERE trade_id = ? ORDER BY id DESC LIMIT 1",
            (paper.get("id"),),
        )
        performance = _row_to_dict(cur.fetchone())
        if performance is None and linked_sell_paper:
            sell_diag = _parse_json_field(linked_sell_paper.get("diagnostics_json"))
            if isinstance(sell_diag, dict):
                ex_oid = sell_diag.get("exchange_sell_order_id")
                if ex_oid is not None:
                    cur.execute(
                        "SELECT * FROM trade_performance WHERE trade_id = ? ORDER BY id DESC LIMIT 1",
                        (ex_oid,),
                    )
                    performance = _row_to_dict(cur.fetchone())

        cur.execute(
            "SELECT * FROM ai_post_trade_feature_reviews WHERE trade_id = ? ORDER BY id DESC LIMIT 1",
            (resolved_tid,),
        )
        post_review = _row_to_dict(cur.fetchone())
        if post_review is None:
            cur.execute(
                "SELECT * FROM ai_post_trade_feature_reviews WHERE trade_id LIKE ? ORDER BY id DESC LIMIT 1",
                (f"%{resolved_tid}%",),
            )
            post_review = _row_to_dict(cur.fetchone())

        learning = None
        if side == "BUY":
            cur.execute(
                "SELECT * FROM trade_learning_outcomes WHERE extra_json LIKE ? ORDER BY id DESC LIMIT 1",
                (f'%"buy_trade_id":"{resolved_tid}"%',),
            )
            learning = _row_to_dict(cur.fetchone())
        if side == "SELL" and learning is None:
            cur.execute(
                "SELECT * FROM trade_learning_outcomes WHERE symbol LIKE ? ORDER BY id DESC LIMIT 5",
                (f"%{symbol.replace('/USDT', '').replace('/USDT', '')}%",),
            )
            for r in cur.fetchall():
                d = _row_to_dict(r)
                if not d:
                    continue
                extra = _parse_json_field(d.get("extra_json"))
                if isinstance(extra, dict) and str(extra.get("sell_trade_id") or extra.get("trade_id") or "") == resolved_tid:
                    learning = d
                    break
            if learning is None and close_ledger:
                cur.execute(
                    "SELECT * FROM trade_learning_outcomes WHERE symbol LIKE ? ORDER BY id DESC LIMIT 1",
                    (f"%{symbol.split('/', maxsplit=1)[0]}%",),
                )
                learning = _row_to_dict(cur.fetchone())

        explain = _parse_json_field(paper.get("explainability_json"))
        diagnostics = _parse_json_field(paper.get("diagnostics_json"))
        if isinstance(explain, dict):
            feature_version = explain.get("feature_version")
            feature_dim = explain.get("feature_dim")
            signal_ts = explain.get("timestamp") or explain.get("entry_timestamp")
            context_ts = explain.get("context_timestamp") or explain.get("ctx_timestamp")
            entry_strategy_id = explain.get("entry_strategy_id") or explain.get("live_ai_strategy")
            strategy_id = paper.get("strategy_id") or explain.get("strategy_id")
            buy_margin = explain.get("entry_buy_margin") or explain.get("buy_margin")
        else:
            feature_version = feature_dim = signal_ts = context_ts = entry_strategy_id = strategy_id = buy_margin = None

        exit_type = str(paper.get("exit_type") or "")
        mode = str(paper.get("mode") or "paper")
        is_admin = exit_type in _ADMIN_EXIT_TYPES or bool(paper.get("is_synthetic"))
        is_ai_trade = not is_admin and exit_type not in _ADMIN_EXIT_TYPES

        preflight = {}
        if isinstance(diagnostics, dict):
            preflight = {
                "orderbook_best_bid": diagnostics.get("orderbook_best_bid"),
                "orderbook_best_ask": diagnostics.get("orderbook_best_ask"),
                "spread_pct": diagnostics.get("spread_pct"),
                "expected_avg_fill": diagnostics.get("expected_avg_fill"),
                "protected_limit_price": diagnostics.get("protected_limit_price"),
                "price_impact_pct": diagnostics.get("price_impact_pct"),
                "execution_mode": diagnostics.get("execution_mode"),
                "book_age_sec": diagnostics.get("book_age_sec"),
                "reject_reason": diagnostics.get("reject_reason"),
                "passed": not diagnostics.get("reject_reason"),
            }

        dup_checks: dict[str, Any] = {}
        cur.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE trade_id = ?",
            (resolved_tid,),
        )
        dup_checks["paper_trades_with_trade_id"] = int(cur.fetchone()[0])
        cur.execute(
            "SELECT COUNT(*) FROM portfolio_engine_audit WHERE trade_id = ? AND action = ?",
            (resolved_tid, side if side in ("BUY", "SELL") else "BUY"),
        )
        dup_checks["audit_rows_same_action"] = int(cur.fetchone()[0])
        dup_checks["duplicate_detected"] = dup_checks["paper_trades_with_trade_id"] > 1

        model_id = None
        if isinstance(explain, dict):
            model_id = explain.get("model_id") or explain.get("model_artifact_id")

        all_audit_rows = audit_rows + [r for r in linked_sell_audits if r]
        return {
            "found": True,
            "trade_id": resolved_tid,
            "paper_trade": paper,
            "linked_sell_paper_trade": linked_sell_paper,
            "audit_rows": all_audit_rows,
            "audit_buy_count": sum(1 for r in all_audit_rows if r and r.get("action") == "BUY"),
            "audit_sell_count": sum(1 for r in all_audit_rows if r and r.get("action") == "SELL"),
            "position_close_ledger": close_ledger,
            "trade_learning_outcome": learning,
            "trade_performance": performance,
            "post_trade_feature_review": post_review,
            "protected_preflight": preflight,
            "executable_fill_gate": {
                "side": side,
                "preflight_passed": preflight.get("passed"),
                "reject_reason": preflight.get("reject_reason") or "",
            },
            "feature_version": feature_version,
            "feature_dim": feature_dim,
            "model_artifact_id": model_id,
            "signal_timestamp": signal_ts,
            "context_timestamp": context_ts,
            "entry_strategy_id": entry_strategy_id,
            "strategy_id": strategy_id,
            "buy_margin": buy_margin,
            "close_reason": paper.get("exit_type") or (close_ledger or {}).get("close_reason"),
            "source": mode,
            "is_admin_or_synthetic": is_admin,
            "is_ai_trade": is_ai_trade,
            "not_ai_trade": not is_ai_trade,
            "duplicate_check": dup_checks,
            "explainability": explain if isinstance(explain, dict) else None,
        }
    finally:
        conn.close()
