"""Deterministic post-227e036 learning quantity backfill and venue-conversion audit cleanup.

Repairs ONLY factual quantity columns on affected post-227e036 learning rows when
immutable fill/order provenance proves exact sold quantity, and reclassifies
venue-converted dust rows that were spuriously written as DUST_WRITEOFF / bad_trade.

Never modifies executed paper_trades, realized P&L, net_pct, fees, entry, or exit prices.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.services.engine_strategy_dust import (
    EVENT_CONVERSION,
    ensure_schema,
    match_dust_conversion,
    record_external_balance_event,
)

POST_227E036_SHA = "227e03641e6ff8dbc3b15df6bda96bb14c3d0c1b"


def _extract_live_close_order_id(event_key: str | None) -> str:
    raw = str(event_key or "").strip()
    if raw.startswith("live-close:"):
        return raw.split("live-close:", 1)[1].strip()
    return ""


def _resolve_proven_sold_quantity(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    event_key: str | None,
    extra: dict[str, Any],
    has_fills_table: bool,
    has_paper_trades: bool,
) -> tuple[float | None, str]:
    """Return (proven_sold_qty, proof_reason_or_unprovable_reason)."""
    oid = _extract_live_close_order_id(event_key)
    if not oid:
        return None, "missing_live_close_order_id"

    sym_norm = str(symbol or "").strip().upper()
    sym_compact = sym_norm.replace("/", "")

    fill_qty: float | None = None
    if has_fills_table:
        frow = conn.execute(
            """
            SELECT executed_qty
            FROM live_exchange_fills
            WHERE UPPER(side) = 'SELL'
              AND exchange_order_id = ?
              AND (UPPER(symbol) = ? OR REPLACE(UPPER(symbol), '/', '') = ?)
            ORDER BY id DESC LIMIT 1
            """,
            (oid, sym_norm, sym_compact),
        ).fetchone()
        if frow is not None and frow[0] is not None and float(frow[0]) > 0:
            fill_qty = float(frow[0])

    pt_qty: float | None = None
    if has_paper_trades:
        prow = conn.execute(
            """
            SELECT quantity
            FROM paper_trades
            WHERE UPPER(side) = 'SELL'
              AND order_id = ?
              AND (UPPER(symbol) = ? OR REPLACE(UPPER(symbol), '/', '') = ?)
            ORDER BY id DESC LIMIT 1
            """,
            (oid, sym_norm, sym_compact),
        ).fetchone()
        if prow is not None and prow[0] is not None and float(prow[0]) > 0:
            pt_qty = float(prow[0])

    prov_sell_qty: float | None = None
    if extra.get("sell_qty") is not None:
        try:
            cand = float(extra["sell_qty"])
            if cand > 0:
                prov_sell_qty = cand
        except (TypeError, ValueError):
            prov_sell_qty = None

    # Require at least one immutable execution table (live_exchange_fills or paper_trades).
    if fill_qty is None and pt_qty is None:
        return None, f"no_immutable_fill_or_paper_sell_for_order:{oid}"

    authoritative = fill_qty if fill_qty is not None else pt_qty
    assert authoritative is not None

    if fill_qty is not None and pt_qty is not None and abs(fill_qty - pt_qty) > 1e-12:
        return None, f"fill_vs_paper_qty_mismatch:{fill_qty}!={pt_qty}"
    if prov_sell_qty is not None and abs(authoritative - prov_sell_qty) > 1e-12:
        return None, f"provenance_sell_qty_mismatch:{authoritative}!={prov_sell_qty}"

    source = "live_exchange_fills+paper_trades" if (fill_qty is not None and pt_qty is not None) else ("live_exchange_fills" if fill_qty is not None else "paper_trades")
    return authoritative, source


def backfill_post_227e036_learning_quantities(
    db_path: str,
    *,
    target_sha: str = POST_227E036_SHA,
) -> dict[str, Any]:
    """Repair ONLY factual quantity columns on affected post-227e036 learning rows."""
    report: dict[str, Any] = {
        "target_sha": target_sha,
        "scanned": 0,
        "tlo_updated": [],
        "tlo_already_correct": [],
        "slo_updated": [],
        "unprovable_rows": [],
    }
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trade_learning_outcomes" not in tables:
            return report

        has_fills = "live_exchange_fills" in tables
        has_pt = "paper_trades" in tables
        has_slo = "scalp_learning_outcomes" in tables

        tlo_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(trade_learning_outcomes)").fetchall()}
        if "code_sha" not in tlo_cols:
            return report

        rows = conn.execute(
            """
            SELECT id, symbol, engine_id, close_reason, quantity, event_key, extra_json
            FROM trade_learning_outcomes
            WHERE code_sha = ?
              AND COALESCE(close_reason, '') NOT IN ('DUST_WRITEOFF', 'EXTERNAL_BALANCE_CONVERSION')
            ORDER BY id ASC
            """,
            (target_sha,),
        ).fetchall()

        conn.execute("BEGIN IMMEDIATE")
        for row in rows:
            report["scanned"] += 1
            row_id = int(row["id"])
            symbol = str(row["symbol"] or "")
            old_qty = float(row["quantity"]) if row["quantity"] is not None else None
            extra: dict[str, Any] = {}
            if row["extra_json"]:
                try:
                    extra = json.loads(row["extra_json"])
                except (TypeError, ValueError):
                    extra = {}

            proven_qty, reason = _resolve_proven_sold_quantity(
                conn,
                symbol=symbol,
                event_key=row["event_key"],
                extra=extra,
                has_fills_table=has_fills,
                has_paper_trades=has_pt,
            )
            if proven_qty is None:
                report["unprovable_rows"].append(
                    {
                        "id": row_id,
                        "symbol": symbol,
                        "event_key": row["event_key"],
                        "stored_quantity": old_qty,
                        "reason": reason,
                    }
                )
                continue

            if old_qty is None or abs(old_qty - proven_qty) > 1e-12:
                conn.execute(
                    "UPDATE trade_learning_outcomes SET quantity = ? WHERE id = ?",
                    (proven_qty, row_id),
                )
                report["tlo_updated"].append(
                    {
                        "id": row_id,
                        "symbol": symbol,
                        "engine_id": row["engine_id"],
                        "old_quantity": old_qty,
                        "new_quantity": proven_qty,
                        "proof_source": reason,
                    }
                )
            else:
                report["tlo_already_correct"].append(row_id)

            if has_slo:
                slo_row = conn.execute(
                    "SELECT id, quantity FROM scalp_learning_outcomes WHERE source_id = ?",
                    (row_id,),
                ).fetchone()
                if slo_row is not None:
                    old_slo_qty = float(slo_row["quantity"]) if slo_row["quantity"] is not None else None
                    if old_slo_qty is None or abs(old_slo_qty - proven_qty) > 1e-12:
                        conn.execute(
                            "UPDATE scalp_learning_outcomes SET quantity = ? WHERE id = ?",
                            (proven_qty, int(slo_row["id"])),
                        )
                        report["slo_updated"].append(
                            {
                                "id": int(slo_row["id"]),
                                "source_id": row_id,
                                "symbol": symbol,
                                "old_quantity": old_slo_qty,
                                "new_quantity": proven_qty,
                            }
                        )
        conn.commit()
    return report


def verify_and_clean_venue_converted_dust_learning_row(
    db_path: str,
    *,
    tlo_id: int,
    venue_conversions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify adaptive learning cleanliness for a spurious DUST_WRITEOFF row and reclassify if venue-converted."""
    out: dict[str, Any] = {
        "tlo_id": tlo_id,
        "found": False,
        "adaptive_impact": {
            "adaptive_close_learned": 0,
            "scalp_learning_outcomes": 0,
            "ai_outcome_training_rows": 0,
            "day_outcome_attribution": 0,
            "ai_bad_trade_patterns": 0,
        },
        "reclassified": False,
        "venue_ref": "",
    }
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trade_learning_outcomes" not in tables:
            return out
        row = conn.execute("SELECT * FROM trade_learning_outcomes WHERE id = ?", (tlo_id,)).fetchone()
        if row is None:
            return out
        out["found"] = True

        extra: dict[str, Any] = {}
        if row["extra_json"]:
            try:
                extra = json.loads(row["extra_json"])
            except (TypeError, ValueError):
                extra = {}

        trade_id = str(extra.get("trade_id") or "")
        symbol = str(row["symbol"] or "")
        asset = symbol.split("/", 1)[0].upper() if "/" in symbol else symbol.replace("USDT", "").upper()
        entry_ts = float(row["entry_timestamp"] or 0.0)
        qty = float(row["quantity"] or row["dust_remaining_qty"] or 0.0)

        if "adaptive_close_learned" in tables and trade_id:
            out["adaptive_impact"]["adaptive_close_learned"] = int(conn.execute("SELECT COUNT(*) FROM adaptive_close_learned WHERE close_key = ?", (trade_id,)).fetchone()[0])
        if "scalp_learning_outcomes" in tables:
            out["adaptive_impact"]["scalp_learning_outcomes"] = int(conn.execute("SELECT COUNT(*) FROM scalp_learning_outcomes WHERE source_id = ?", (tlo_id,)).fetchone()[0])
        if "day_outcome_attribution" in tables and trade_id:
            out["adaptive_impact"]["day_outcome_attribution"] = int(conn.execute("SELECT COUNT(*) FROM day_outcome_attribution WHERE trade_id = ?", (trade_id,)).fetchone()[0])
        if "ai_bad_trade_patterns" in tables and trade_id:
            out["adaptive_impact"]["ai_bad_trade_patterns"] = int(conn.execute("SELECT COUNT(*) FROM ai_bad_trade_patterns WHERE trade_id = ?", (trade_id,)).fetchone()[0])

        match = match_dust_conversion(venue_conversions, asset=asset, after_epoch=entry_ts)
        if match is None:
            return out

        conn.execute("BEGIN IMMEDIATE")
        ensure_schema(conn)
        record_external_balance_event(
            conn,
            symbol=symbol,
            quantity=qty,
            event_class=EVENT_CONVERSION,
            source="audit_reclassify_spurious_dust_writeoff",
            engine_id=str(row["engine_id"] or "LEGACY_DAY_LIVE"),
            source_trade_id=trade_id or f"tlo:{tlo_id}",
            venue_ref=str(match.get("tran_id") or ""),
            venue_time_utc=str(match.get("operate_time_utc") or ""),
            evidence=match,
        )
        extra["bad_trade"] = False
        extra["good_trade"] = False
        extra["lesson"] = "venue_small_balance_conversion_not_strategy_loss"
        extra["event_class"] = EVENT_CONVERSION
        extra["venue_ref"] = str(match.get("tran_id") or "")
        extra["venue_time_utc"] = str(match.get("operate_time_utc") or "")
        if row["close_reason"] and row["close_reason"] != EVENT_CONVERSION:
            extra.setdefault("reclassified_from_close_reason", str(row["close_reason"]))
        conn.execute(
            """
            UPDATE trade_learning_outcomes
            SET close_reason = ?, decision_reason = ?, extra_json = ?
            WHERE id = ?
            """,
            (
                EVENT_CONVERSION,
                f"external_event:{EVENT_CONVERSION}",
                json.dumps(extra, separators=(",", ":"), default=str),
                tlo_id,
            ),
        )
        if "position_close_ledger" in tables and row["event_key"]:
            conn.execute(
                """
                UPDATE position_close_ledger
                SET close_reason = ?, cooldown_until = 0.0
                WHERE event_key = ? AND close_reason = 'DUST_WRITEOFF'
                """,
                (EVENT_CONVERSION, str(row["event_key"])),
            )
        conn.commit()
        out["reclassified"] = True
        out["venue_ref"] = str(match.get("tran_id") or "")
    return out
