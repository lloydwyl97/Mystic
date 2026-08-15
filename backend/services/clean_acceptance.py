"""Canonical clean-sample queries for DAY and SCALP acceptance.

Never compare mixed timestamp formats as raw strings.
Never delete history. The stored cutoff moves only when the operator
explicitly marks a new clean-sample start.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.services.binance_scalp.historical_forensic import _parse_ts
from backend.services.entry_decision_authority import (
    extract_provenance,
    is_model_controlled,
    summarize_book,
)
from backend.services.validation_cutoff import (
    is_strategy_acceptance_eligible,
    read_validation_cutoff,
)

DAY_CUTOFF_UTC = "2026-08-15T02:03:02.291353+00:00"
SCALP_CUTOFF_UTC = "2026-08-15T02:03:02.291353+00:00"


def parse_ts(raw: Any) -> datetime | None:
    return _parse_ts(raw)


def cutoff_dt(db_path: str, *, fallback: str) -> datetime:
    row = read_validation_cutoff(db_path)
    text = (row or {}).get("cutoff_utc") or fallback
    parsed = parse_ts(text)
    if parsed is None:
        parsed = parse_ts(fallback)
    assert parsed is not None
    return parsed


def _eligible(exit_reason: Any, trade_id: Any, explainability: Any) -> bool:
    extra = {}
    if isinstance(explainability, dict):
        extra = explainability
    elif explainability:
        try:
            extra = json.loads(explainability)
        except Exception:
            extra = {}
    return is_strategy_acceptance_eligible(
        exit_reason=str(exit_reason or ""),
        trade_id=str(trade_id or ""),
        extra=extra,
    )


def day_clean_rows(db_path: str) -> dict[str, Any]:
    """Clean DAY = SELL whose BUY/entry is at/after stored cutoff, not reconciliation."""
    cut = cutoff_dt(db_path, fallback=DAY_CUTOFF_UTC)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    created_col = "created_at" if "created_at" in cols else "timestamp"
    sells = list(
        conn.execute(
            f"""
            SELECT id, trade_id, symbol, pnl, {created_col} AS created_at, timestamp, entry_timestamp,
                   exit_reason, explainability_json
            FROM paper_trades WHERE UPPER(side)='SELL' ORDER BY id
            """
        )
    )
    buys = list(
        conn.execute(
            """
            SELECT trade_id, entry_timestamp, explainability_json
            FROM paper_trades WHERE UPPER(side)='BUY'
            """
        )
    )
    buy_by_entry: dict[str, dict[str, Any]] = {}
    buy_by_tid: dict[str, dict[str, Any]] = {}
    for b in buys:
        try:
            payload = json.loads(b["explainability_json"] or "{}")
        except Exception:
            payload = {}
        if b["entry_timestamp"]:
            buy_by_entry[str(b["entry_timestamp"])] = payload
        if b["trade_id"]:
            buy_by_tid[str(b["trade_id"])] = payload
    open_pos = []
    try:
        open_pos = list(conn.execute("SELECT symbol, quantity, entry_price, trade_id, status FROM portfolio_engine_positions WHERE status='ACTIVE'"))
    except sqlite3.OperationalError:
        open_pos = []
    conn.close()

    post_exit = []
    clean = []
    historical_string_trap = []
    for r in sells:
        entry = parse_ts(r["entry_timestamp"])
        exit_t = parse_ts(r["timestamp"]) or parse_ts(r["created_at"])
        try:
            sell_ex = json.loads(r["explainability_json"] or "{}")
        except Exception:
            sell_ex = {}
        buy_ex = buy_by_tid.get(str(r["trade_id"] or "")) or buy_by_entry.get(str(r["entry_timestamp"] or "")) or {}
        merged = dict(buy_ex)
        if isinstance(sell_ex, dict):
            merged.update({k: v for k, v in sell_ex.items() if v not in (None, "")})
        eligible = _eligible(r["exit_reason"], r["id"], r["explainability_json"])
        rec = {
            "id": r["id"],
            "trade_id": r["trade_id"],
            "symbol": r["symbol"],
            "pnl": float(r["pnl"] or 0),
            "entry_timestamp": r["entry_timestamp"],
            "exit_timestamp": r["timestamp"],
            "created_at": r["created_at"],
            "exit_reason": r["exit_reason"],
            "buy_after_cutoff": bool(entry and entry >= cut),
            "sell_after_cutoff": bool(exit_t and exit_t >= cut),
            "strategy_acceptance_eligible": eligible,
            "clean": bool(entry and entry >= cut and eligible),
            "model_controlled": is_model_controlled(merged, engine="day"),
            "entry_policy_version": merged.get("entry_policy_version"),
            "model_version": merged.get("model_version") or merged.get("forward_net_model_version"),
            "selected_action": merged.get("selected_action"),
            "selection_reason": merged.get("selection_reason"),
            "provenance": extract_provenance(merged),
        }
        if rec["sell_after_cutoff"]:
            post_exit.append(rec)
        if rec["clean"]:
            clean.append(rec)
        # Reproduce the broken string compare that created the n=11 report.
        ts = str(r["timestamp"] or "")
        if ts >= "2026-08-14 23:11" and eligible:
            historical_string_trap.append(rec)

    def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_book(rows)

    model_controlled = [x for x in clean if x.get("model_controlled")]
    return {
        "engine": "day",
        "cutoff_utc": cut.isoformat(),
        "rule": "SELL with parsed entry_timestamp >= cutoff AND not reconciliation/manual-inventory",
        "do_not_use": "raw string compare of ISO timestamp vs 'YYYY-MM-DD HH:MM' (T > space includes all same-day rows)",
        "clean": _summ(clean),
        "clean_runtime": _summ(clean),
        "model_controlled": _summ(model_controlled),
        "post_cutoff_exits_including_recon": _summ(post_exit),
        "broken_string_compare_population": _summ(historical_string_trap),
        "open_positions": [dict(r) for r in open_pos],
    }


def _scalp_entry_dt(trade_id: str, created_at: str, diagnostics: dict[str, Any]) -> datetime | None:
    for key in ("entry_time", "entry_timestamp", "entry_time_epoch"):
        parsed = parse_ts(diagnostics.get(key))
        if parsed:
            return parsed
    tid = str(trade_id or "")
    parts = tid.replace("_SELL", "").replace("_BUY", "").split("_")
    for part in reversed(parts):
        if part.isdigit() and len(part) >= 12:
            try:
                return datetime.fromtimestamp(int(part) / 1000.0, tz=timezone.utc)
            except (OSError, ValueError):
                continue
    return parse_ts(created_at)


def scalp_clean_rows(db_path: str) -> dict[str, Any]:
    """Clean SCALP = SELL whose entry epoch/time is at/after stored cutoff."""
    cut = cutoff_dt(db_path, fallback=SCALP_CUTOFF_UTC)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sells = list(
        conn.execute(
            """
            SELECT id, trade_id, symbol, pnl_usd, created_at, exit_reason, diagnostics_json
            FROM scalp_paper_trades WHERE UPPER(side)='SELL' ORDER BY id
            """
        )
    )
    buys = list(
        conn.execute(
            """
            SELECT trade_id, diagnostics_json
            FROM scalp_paper_trades WHERE UPPER(side)='BUY'
            """
        )
    )
    buy_by_tid = {}
    for b in buys:
        try:
            buy_by_tid[str(b["trade_id"] or "")] = json.loads(b["diagnostics_json"] or "{}")
        except Exception:
            buy_by_tid[str(b["trade_id"] or "")] = {}
    opens = list(conn.execute("SELECT symbol, status, entry_time, trade_id FROM scalp_paper_positions WHERE status='OPEN'"))
    conn.close()
    clean = []
    for r in sells:
        try:
            diag = json.loads(r["diagnostics_json"] or "{}")
        except Exception:
            diag = {}
        buy_tid = str(r["trade_id"] or "").replace("_SELL", "")
        buy_diag = buy_by_tid.get(buy_tid) or {}
        merged = dict(buy_diag)
        if isinstance(diag, dict):
            merged.update({k: v for k, v in diag.items() if v not in (None, "")})
        entry = _scalp_entry_dt(str(r["trade_id"] or ""), str(r["created_at"] or ""), merged)
        exit_t = parse_ts(r["created_at"])
        eligible = _eligible(r["exit_reason"], r["id"], diag)
        rec = {
            "id": r["id"],
            "trade_id": r["trade_id"],
            "symbol": r["symbol"],
            "pnl": float(r["pnl_usd"] or 0),
            "entry_timestamp": entry.isoformat() if entry else None,
            "exit_timestamp": r["created_at"],
            "exit_reason": r["exit_reason"],
            "buy_after_cutoff": bool(entry and entry >= cut),
            "sell_after_cutoff": bool(exit_t and exit_t >= cut),
            "strategy_acceptance_eligible": eligible,
            "soft_rank_entry": merged.get("soft_rank_entry", diag.get("soft_rank_entry")),
            "clean": bool(entry and entry >= cut and eligible),
            "model_controlled": is_model_controlled(merged, engine="scalp"),
            "entry_policy_version": merged.get("entry_policy_version") or merged.get("decision_policy_version"),
            "model_version": merged.get("model_version") or merged.get("forward_net_model_version"),
            "selected_action": merged.get("selected_action"),
            "selection_reason": merged.get("selection_reason"),
            "source_opportunity_id": merged.get("source_opportunity_id"),
            "provenance": extract_provenance(merged),
        }
        if rec["clean"]:
            clean.append(rec)

    model_controlled = [x for x in clean if x.get("model_controlled")]
    return {
        "engine": "scalp",
        "cutoff_utc": cut.isoformat(),
        "rule": "SELL with parsed entry time from trade_id epoch or diagnostics >= cutoff AND not reconciliation",
        "clean": summarize_book(clean),
        "clean_runtime": summarize_book(clean),
        "model_controlled": summarize_book(model_controlled),
        "open_positions": [dict(r) for r in opens],
    }
