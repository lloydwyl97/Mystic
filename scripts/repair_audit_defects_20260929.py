#!/usr/bin/env python3
"""Historical repair for the 2026-09-29 audit defects.

Dry run by default; pass --apply with the service stopped. Every changed row is
copied to ownership_repair_backup first. Only exact provenance is repaired:

1. Real DAY/SCALP closes mislabeled dust -> metadata only, then each canonical
   downstream writer runs once (ledger: audit_learning_repair_applied).
2. Engine dust overwritten by same-engine re-entry -> engine_strategy_dust;
   the XRP share leaves protected inventory.
3. Phantom strategy dust removed by the venue dust conversion -> retired as
   EXTERNAL_BALANCE_CONVERSION (dust-log tranId); no P&L, learning, cooldown.
4. Multi-chunk sells -> chunk fees from venue myTrades; derived economics fixed.
5. Close-ledger MANUAL_EXIT on strategy closes -> the recorded strategy reason.
6. DAY/SCALP sell rows missing their setup -> BUY row / engine setup record.
7. SCALP rows in DAY attribution -> removed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.audit_defect_repair import (
    fee_correction,
    load_json,
    real_close_evidence,
    repaired_learning_extra,
    verify_chunk_trades,
)
from backend.services.engine_strategy_dust import (
    DUST_TABLE,
    EVENT_CONVERSION,
    match_dust_conversion,
    record_external_balance_event,
    retire_held_dust,
    zero_lot_remaining,
)
from backend.services.engine_strategy_dust import (
    ensure_schema as ensure_dust_schema,
)
from backend.services.ownership_repair import _backup, ensure_backup_table

REASON = "audit_20260929"
APPLIED = "audit_learning_repair_applied"
CHUNKS = "live_sell_chunk_fills"
NO_LEARNING = {"MANUAL_UNMATCHED", "HUMAN_MANUAL_SELL", "DUST_WRITEOFF"}

# Same-engine re-entry overwrote these DUST_PENDING rows. Quantity and trade id
# come from LOT_QTY_OWNERSHIP_CAPPED (fill_owned) log lines and match the
# lot's own BUY minus SELL fills net of base-asset commission.
OVERWRITTEN_DUST = (
    ("SCALP_V2", "SOL/USDT", "scalp_v2_SOLUSDT_1790604179476", 0.0009166),
    ("DAY_V2", "BTC/USDT", "mystic_BTC/USDT_1790647275527", 9.84e-06),
    ("SCALP_V2", "XRP/USDT", "scalp_v2_XRPUSDT_1790612481481", 0.09474),
)


def _iso(epoch: float | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    ensure_backup_table(conn)
    ensure_dust_schema(conn)
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {APPLIED} (
            tlo_id INTEGER NOT NULL, writer TEXT NOT NULL, applied_at TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{{}}', PRIMARY KEY (tlo_id, writer))"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {CHUNKS} (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sell_fill_row_id INTEGER NOT NULL,
            exchange_order_id TEXT NOT NULL, venue_trade_id TEXT NOT NULL,
            qty REAL NOT NULL, price REAL NOT NULL, cost REAL NOT NULL,
            fee REAL NOT NULL, fee_asset TEXT, venue_ts TEXT, recorded_at TEXT NOT NULL,
            UNIQUE (venue_trade_id))"""
    )


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _applied(conn: sqlite3.Connection, tlo_id: int, writer: str) -> bool:
    if not _has_table(conn, APPLIED):
        return False
    return conn.execute(f"SELECT 1 FROM {APPLIED} WHERE tlo_id=? AND writer=?", (tlo_id, writer)).fetchone() is not None


def _mark(conn: sqlite3.Connection, tlo_id: int, writer: str, detail: dict[str, Any]) -> None:
    conn.execute(
        f"INSERT OR IGNORE INTO {APPLIED} (tlo_id, writer, applied_at, detail_json) VALUES (?,?,?,?)",
        (tlo_id, writer, datetime.now(timezone.utc).isoformat(), json.dumps(detail, default=str)),
    )


async def _venue() -> Any:
    from backend.services.live_trading_service import LiveTradingService

    svc = LiveTradingService()
    await svc._ensure_initialized()
    return svc.binance


# ---------------------------------------------------------------- learning ---


def plan_learning(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    out = []
    for r in conn.execute(
        """SELECT * FROM trade_learning_outcomes
           WHERE json_valid(extra_json) AND COALESCE(json_extract(extra_json,'$.is_dust'),0)=1
           ORDER BY exit_timestamp"""
    ).fetchall():
        reason = str(r["close_reason"] or "").upper()
        if reason in NO_LEARNING or reason.startswith("FALSE_"):
            continue
        extra = load_json(r["extra_json"])
        tid = str(extra.get("trade_id") or "")
        engine = str(extra.get("engine_id") or "")
        if engine not in ("DAY_V2", "SCALP_V2") or not tid:
            continue
        fills = [dict(f) for f in conn.execute("SELECT * FROM live_exchange_fills WHERE mystic_trade_id=? AND UPPER(side)='SELL'", (tid,)).fetchall()]
        ev = real_close_evidence(float(r["exit_timestamp"] or 0.0), fills)
        out.append(
            {
                "tlo_id": int(r["id"]),
                "symbol": r["symbol"],
                "engine_id": engine,
                "trade_id": tid,
                "close_reason": r["close_reason"],
                "net_profit_usd": r["net_profit_usd"],
                "net_profit_pct": r["net_profit_pct"],
                "exit_timestamp": r["exit_timestamp"],
                "entry_timestamp": r["entry_timestamp"],
                "entry_price": r["entry_price"],
                "exit_price": r["exit_price"],
                "hold_seconds": r["hold_seconds"],
                "residual_qty": float(r["dust_remaining_qty"] or 0.0),
                "extra": extra,
                "evidence": ev,
                "action": "REPAIR_REAL_CLOSE" if ev else "KEEP_TRUE_DUST",
            }
        )
    return out


def _buy_explain(conn: sqlite3.Connection, trade_id: str) -> tuple[dict[str, Any], str]:
    row = conn.execute(
        "SELECT explainability_json, decision_id FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' ORDER BY id DESC LIMIT 1",
        (trade_id,),
    ).fetchone()
    if not row:
        return {}, ""
    return load_json(row[0]), str(row[1] or "")


def run_writers(db_path: str, conn: sqlite3.Connection, item: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    from backend.services.ai_outcome_training_writer import record_outcome_training_row
    from backend.services.ai_pattern_memory import build_pattern_vector, record_trade_pattern
    from backend.services.ai_post_trade_feature_review import _lookup_entry_features
    from backend.services.day_trade_thesis import resolve_setup_identity

    tlo_id = item["tlo_id"]
    extra = item["extra"]
    engine = item["engine_id"]
    strategy = "scalp" if engine == "SCALP_V2" else "day"
    ex, decision_id = _buy_explain(conn, item["trade_id"])
    if strategy == "scalp" and extra.get("setup"):
        ex["setup_type"] = extra["setup"]
    ex["raw_exit_reason"] = extra.get("raw_exit_reason") or item["close_reason"]
    ex["canonical_exit_reason"] = item["close_reason"]
    ident = resolve_setup_identity(ex)
    if ident.get("setup_type_canonical"):
        ex.update({k: ident[k] for k in ("setup_type_canonical", "setup_type_raw", "day_route_regime", "adaptive_regime")})
        ex["setup_type"] = ident["setup_type_canonical"]
        ex["entry_thesis"] = ident["entry_thesis"] or ident["setup_type_canonical"]
    opened = _iso(item["entry_timestamp"])
    closed = _iso(item["exit_timestamp"])
    sym_bus = str(item["symbol"]).replace("/", "").upper()
    feats = _lookup_entry_features(db_path, decision_id=decision_id, symbol=sym_bus, opened_at_utc=opened) or _lookup_entry_features(
        db_path, decision_id=decision_id, symbol=str(item["symbol"]), opened_at_utc=opened
    )
    pnl = float(item["net_profit_usd"] or 0.0)
    pct = float(item["net_profit_pct"] or 0.0)
    report: dict[str, Any] = {"tlo_id": tlo_id, "trade_id": item["trade_id"], "symbol": item["symbol"], "engine": engine, "setup": ex.get("setup_type"), "pnl": pnl}

    exit_epoch = int(float(item["exit_timestamp"] or 0.0))
    train_exists = (
        conn.execute(
            "SELECT COUNT(*) FROM ai_outcome_training_rows WHERE symbol IN (?,?) AND ABS(strftime('%s', closed_at_utc) - ?) < 120",
            (item["symbol"], sym_bus, exit_epoch),
        ).fetchone()[0]
        > 0
    )
    attr_exists = conn.execute("SELECT COUNT(*) FROM day_outcome_attribution WHERE trade_id=?", (item["trade_id"],)).fetchone()[0] > 0
    pattern_exists = any(
        conn.execute(f"SELECT COUNT(*) FROM {t} WHERE trade_id=?", (item["trade_id"],)).fetchone()[0] > 0 for t in ("ai_good_trade_patterns", "ai_bad_trade_patterns") if _has_table(conn, t)
    )
    # A close that already has a training row ran the full DAY learning call at
    # the time; the bandit and market memory keep no per-trade key, so they are
    # only written for closes the dust gate skipped entirely.
    already = {
        "training_row": train_exists,
        "pattern_memory": pattern_exists,
        "day_attribution": attr_exists,
        "day_bandit": train_exists,
        "market_memory": train_exists,
    }

    def once(writer: str, fn) -> None:
        if _applied(conn, tlo_id, writer):
            report[writer] = "already_applied"
            return
        if already.get(writer):
            report[writer] = "skipped_existing"
            return
        if not apply:
            report[writer] = "would_apply"
            return
        detail = fn() or {}
        _mark(conn, tlo_id, writer, detail if isinstance(detail, dict) else {"result": detail})
        conn.commit()
        report[writer] = detail if isinstance(detail, dict) and detail else "applied"

    ctx = json.dumps({"_live_ai_strategy": strategy, "engine_id": engine, "trade_id": item["trade_id"], "setup": extra.get("setup"), "is_dust": False, "decision_id": decision_id})
    once(
        "training_row",
        lambda: {
            "row_id": record_outcome_training_row(
                symbol=item["symbol"],
                opened_at_utc=opened or closed,
                closed_at_utc=closed,
                hold_seconds=item["hold_seconds"],
                entry_price=item["entry_price"],
                exit_price=item["exit_price"],
                net_profit_usd=pnl,
                net_profit_pct=pct,
                gross_pnl_pct=pct,
                close_reason=item["close_reason"],
                strategy_id=strategy,
                features_json=json.dumps(feats) if feats else None,
                context_json=ctx if feats else None,
                explainability=ex,
                db_path=db_path,
            )
        },
    )
    once(
        "pattern_memory",
        lambda: {
            "ok": record_trade_pattern(
                db_path=db_path,
                symbol=item["symbol"],
                strategy_id=strategy,
                vector=build_pattern_vector(chop_score=ex.get("chop_score"), coin_edge_score=ex.get("coin_edge_score"), trend_score=ex.get("trend_score"), confidence=ex.get("ai_confidence")),
                net_outcome_pct=pct,
                net_pnl=pnl,
                hold_seconds=float(item["hold_seconds"] or 0.0),
                reason=item["close_reason"],
                trade_id=item["trade_id"],
                entry_time_iso=opened or "",
                exit_time_iso=closed or "",
            )
        },
    )
    if strategy == "day":
        from backend.services.day_market_memory import update_market_memory_on_close_sync
        from backend.services.day_outcome_attribution import record_outcome_attribution
        from backend.services.day_outcome_bandit import arm_key, record_bandit_outcome

        once(
            "day_attribution",
            lambda: {
                "result": record_outcome_attribution(
                    trade_id=item["trade_id"],
                    symbol=item["symbol"],
                    explainability=ex,
                    net_profit_usd=pnl,
                    net_profit_pct=pct,
                    close_reason=item["close_reason"],
                    hold_seconds=int(item["hold_seconds"] or 0),
                    entry_features=feats,
                    db_path=db_path,
                )
            },
        )
        setup = str(ex.get("setup_type_canonical") or ex.get("setup_type") or ex.get("entry_thesis") or "")
        regime = str(ex.get("day_route_regime") or ex.get("regime") or "range")

        def _bandit() -> dict[str, Any]:
            key = arm_key(item["symbol"], setup, regime)
            cols = "alpha, beta, wins, losses, total_pnl, n_obs"
            before = conn.execute(f"SELECT {cols} FROM day_outcome_bandit_arms WHERE arm_key=?", (key,)).fetchone()
            res = record_bandit_outcome(symbol=item["symbol"], setup=setup, regime=regime, pnl_usd=pnl, exit_reason=item["close_reason"], db_path=db_path, trade_id=item["trade_id"])
            after = conn.execute(f"SELECT {cols} FROM day_outcome_bandit_arms WHERE arm_key=?", (key,)).fetchone()
            return {"arm": key, "regime": regime, "before": list(before) if before else None, "after": list(after) if after else None, "applied": bool((res or {}).get("applied", True))}

        once("day_bandit", _bandit)
        once(
            "market_memory",
            lambda: (
                update_market_memory_on_close_sync(
                    item["symbol"],
                    setup=str(ex.get("setup_type") or ex.get("entry_thesis") or ""),
                    net_pnl_pct=pct,
                    close_reason=item["close_reason"],
                    outcome_class=str(ex.get("outcome_reason") or ""),
                )
                or {"ok": True}
            ),
        )
    role = conn.execute("SELECT COUNT(*) FROM market_role_trade_outcomes WHERE buy_trade_id=?", (item["trade_id"],)).fetchone()
    report["market_role_rows"] = int(role[0]) if role else 0
    return report


# -------------------------------------------------------------- inventory ---


def plan_inventory(conn: sqlite3.Connection, dust_log: list[dict[str, Any]]) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    plan: dict[str, Any] = {"overwritten": [], "phantom": [], "protected": []}
    for engine, sym, tid, qty in OVERWRITTEN_DUST:
        exists = conn.execute(f"SELECT status FROM {DUST_TABLE} WHERE source_trade_id=?", (tid,)).fetchone() if _has_table(conn, DUST_TABLE) else None
        buy = conn.execute("SELECT price, timestamp FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' ORDER BY id LIMIT 1", (tid,)).fetchone()
        entry_epoch = datetime.fromisoformat(str(buy["timestamp"])).timestamp() if buy else 0.0
        conv = match_dust_conversion(dust_log, asset=sym.split("/")[0], after_epoch=entry_epoch)
        plan["overwritten"].append(
            {
                "engine_id": engine,
                "symbol": sym,
                "source_trade_id": tid,
                "qty": qty,
                "entry_price": float(buy["price"]) if buy else 0.0,
                "entry_epoch": entry_epoch,
                "conversion": conv,
                "exists": dict(exists) if exists else None,
            }
        )
    for p in conn.execute("SELECT engine_id, symbol, trade_id, quantity, entry_time, status FROM portfolio_engine_positions WHERE status='DUST_PENDING'").fetchall():
        conv = match_dust_conversion(dust_log, asset=str(p["symbol"]).split("/")[0], after_epoch=float(p["entry_time"] or 0.0))
        plan["phantom"].append({**dict(p), "conversion": conv})
    xrp = next((o for o in plan["overwritten"] if o["symbol"] == "XRP/USDT" and not o["exists"]), None)
    prot = conn.execute("SELECT symbol, quantity, cost_price, source_trade_id FROM protected_external_inventory WHERE symbol='XRP/USDT'").fetchone()
    if xrp and prot:
        plan["protected"].append({"symbol": "XRP/USDT", "before": float(prot["quantity"]), "after": round(float(prot["quantity"]) - float(xrp["qty"]), 12), "source_trade_id": prot["source_trade_id"]})
    return plan


def apply_inventory(conn: sqlite3.Connection, plan: dict[str, Any], exchange_free: dict[str, float]) -> list[str]:
    done: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for o in plan["overwritten"]:
        if o["exists"]:
            continue
        asset = o["symbol"].split("/")[0]
        held_on_venue = float(exchange_free.get(asset, 0.0) or 0.0) > 0.0
        if not held_on_venue and not o["conversion"]:
            done.append(f"SKIP overwritten {o['source_trade_id']}: venue 0 and no conversion record")
            continue
        conn.execute(
            f"""INSERT OR IGNORE INTO {DUST_TABLE}
                (engine_id, symbol, source_trade_id, quantity, quantity_exact, entry_price, provenance_json, status, created_at)
                VALUES (?,?,?,?,?,?,?,'HELD',?)""",
            (o["engine_id"], o["symbol"], o["source_trade_id"], o["qty"], format(o["qty"], ".12g"), o["entry_price"], json.dumps({"entry_time": o["entry_epoch"], "repair": REASON}), now),
        )
        if not held_on_venue:
            conv = o["conversion"]
            record_external_balance_event(
                conn,
                symbol=o["symbol"],
                quantity=o["qty"],
                event_class=EVENT_CONVERSION,
                source=f"{REASON}:overwritten_dust",
                engine_id=o["engine_id"],
                source_trade_id=o["source_trade_id"],
                venue_ref=conv["tran_id"],
                venue_time_utc=conv["operate_time_utc"],
                evidence=conv,
            )
            retire_held_dust(conn, o["source_trade_id"], event_class=EVENT_CONVERSION, venue_ref=conv["tran_id"])
            done.append(f"overwritten {o['source_trade_id']} {o['qty']} -> HELD -> RETIRED {EVENT_CONVERSION} tranId={conv['tran_id']}")
        else:
            done.append(f"overwritten {o['source_trade_id']} {o['qty']} -> HELD (venue holds {asset})")
    for pr in plan["protected"]:
        _backup(conn, "protected_external_inventory", "symbol", pr["symbol"], "engine_dust_out_of_protected", REASON)
        conn.execute("UPDATE protected_external_inventory SET quantity=?, updated_at=? WHERE symbol=?", (pr["after"], time.time(), pr["symbol"]))
        done.append(f"protected {pr['symbol']} {pr['before']} -> {pr['after']}")
    for p in plan["phantom"]:
        asset = str(p["symbol"]).split("/")[0]
        if float(exchange_free.get(asset, 0.0) or 0.0) > 0.0:
            continue
        conv = p["conversion"]
        if not conv:
            done.append(f"SKIP phantom {p['trade_id']}: no conversion record (engine retires after absence window)")
            continue
        _backup(conn, "portfolio_engine_positions", "trade_id", p["trade_id"], "retire_phantom_dust", REASON)
        record_external_balance_event(
            conn,
            symbol=p["symbol"],
            quantity=float(p["quantity"]),
            event_class=EVENT_CONVERSION,
            source=f"{REASON}:phantom_dust",
            engine_id=p["engine_id"],
            source_trade_id=p["trade_id"],
            venue_ref=conv["tran_id"],
            venue_time_utc=conv["operate_time_utc"],
            evidence=conv,
        )
        conn.execute("DELETE FROM portfolio_engine_positions WHERE trade_id=? AND status='DUST_PENDING'", (p["trade_id"],))
        done.append(f"phantom {p['engine_id']} {p['symbol']} {p['trade_id']} {p['quantity']} -> RETIRED {EVENT_CONVERSION} tranId={conv['tran_id']}")
    for (tid,) in conn.execute("SELECT DISTINCT source_trade_id FROM external_balance_events WHERE source_trade_id != ''").fetchall():
        for (buy_id,) in conn.execute("SELECT id FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' AND COALESCE(remaining_position,0)>0", (tid,)).fetchall():
            _backup(conn, "paper_trades", "id", buy_id, "retired_lot_remaining_zero", REASON)
        if zero_lot_remaining(conn, tid):
            done.append(f"buy remaining_position -> 0 for retired {tid}")
    return done


# ------------------------------------------------------------ multi-chunk ---


def plan_chunks(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM live_exchange_fills WHERE UPPER(side)='SELL'
           AND ABS(cost_quote - executed_qty*avg_fill_price) > 0.01 ORDER BY id"""
    ).fetchall()
    out = []
    for r in rows:
        paper = conn.execute("SELECT * FROM paper_trades WHERE order_id=? AND UPPER(side)='SELL'", (r["exchange_order_id"],)).fetchone()
        out.append({"fill": dict(r), "paper": dict(paper) if paper else None})
    return out


async def fetch_chunk_truth(ex: Any, item: dict[str, Any]) -> dict[str, Any] | None:
    f = item["fill"]
    paper = item["paper"]
    if not paper:
        return None
    t = datetime.fromisoformat(str(paper["timestamp"])).timestamp()
    since = int((t - 120) * 1000)
    trades = await asyncio.to_thread(ex.fetch_my_trades, str(f["symbol"]), since, 100)
    window = [x for x in trades if since <= int(x.get("timestamp") or 0) <= int((t + 20) * 1000)]
    oid = str(f["exchange_order_id"])
    sells = [x for x in window if str(x.get("side")).lower() == "sell"]
    anchor = [x for x in sells if str(x.get("order")) == oid]
    if not anchor:
        return None
    # Chunks of one close are consecutive IOC orders; a sell far from the anchor belongs to another close.
    t_anchor = max(int(x["timestamp"]) for x in anchor)
    cluster = [x for x in sells if t_anchor - 60_000 <= int(x["timestamp"]) <= t_anchor + 1_000]
    v = verify_chunk_trades(f, cluster)
    if v:
        v["trades"] = cluster
    return v


def apply_chunk(conn: sqlite3.Connection, item: dict[str, Any], v: dict[str, Any]) -> dict[str, Any] | None:
    f, paper = item["fill"], item["paper"]
    if conn.execute(f"SELECT 1 FROM {CHUNKS} WHERE sell_fill_row_id=? LIMIT 1", (f["id"],)).fetchone():
        return None
    missing = fee_correction(float(f["fee_amount"] or 0.0), v)
    if missing is None:
        return None
    for t in v["trades"]:
        fee = t.get("fee") or {}
        conn.execute(
            f"INSERT OR IGNORE INTO {CHUNKS} (sell_fill_row_id, exchange_order_id, venue_trade_id, qty, price, cost, fee, fee_asset, venue_ts, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                f["id"],
                str(t.get("order")),
                str(t.get("id")),
                float(t["amount"]),
                float(t["price"]),
                float(t.get("cost") or 0.0),
                float(fee.get("cost") or 0.0),
                str(fee.get("currency") or ""),
                _iso(int(t["timestamp"]) / 1000.0),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    before = {k: paper.get(k) for k in ("fees_paid", "exit_fee_usd", "pnl", "pnl_usd_net", "pnl_pct", "pnl_pct_net")}
    if missing <= 0:
        return {"paper_id": paper["id"], "missing_fee": 0.0, "before": before, "after": before}
    _backup(conn, "paper_trades", "id", paper["id"], "multi_chunk_fee_repair", REASON)
    basis = float(paper.get("quantity") or 0.0) * float(paper.get("entry_price") or 0.0)
    new = {
        "fees_paid": float(paper.get("fees_paid") or 0.0) + missing,
        "exit_fee_usd": float(paper.get("exit_fee_usd") or 0.0) + missing if paper.get("exit_fee_usd") is not None else None,
        "pnl": float(paper.get("pnl") or 0.0) - missing,
        "pnl_usd_net": float(paper.get("pnl_usd_net")) - missing if paper.get("pnl_usd_net") is not None else None,
    }
    if basis > 0:
        new["pnl_pct"] = float(paper["pnl_pct"]) - missing / basis if paper.get("pnl_pct") is not None else None
        new["pnl_pct_net"] = float(paper["pnl_pct_net"]) - missing / basis if paper.get("pnl_pct_net") is not None else None
    sets = {k: val for k, val in new.items() if val is not None}
    conn.execute(f"UPDATE paper_trades SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), paper["id"]))
    t_close = datetime.fromisoformat(str(paper["timestamp"])).timestamp()
    tlo_rows = conn.execute(
        """SELECT id, net_profit_usd, fees_paid FROM trade_learning_outcomes
           WHERE symbol=? AND ABS(exit_timestamp-?)<120
             AND (json_extract(extra_json,'$.trade_id')=? OR ABS(COALESCE(net_profit_usd,0)-?)<1e-9)""",
        (paper["symbol"], t_close, str(f.get("mystic_trade_id") or ""), float(paper.get("pnl") or 0.0)),
    ).fetchall()
    tlo = tlo_rows[0] if len(tlo_rows) == 1 else None
    if tlo:
        _backup(conn, "trade_learning_outcomes", "id", tlo[0], "multi_chunk_fee_repair", REASON)
        conn.execute("UPDATE trade_learning_outcomes SET net_profit_usd=?, fees_paid=? WHERE id=?", (float(tlo[1]) - missing, float(tlo[2] or 0.0) + missing, tlo[0]))
    pcl = conn.execute("SELECT id, realized_profit FROM position_close_ledger WHERE sell_trade_id=? AND realized_profit IS NOT NULL", (paper["trade_id"],)).fetchone()
    if pcl:
        _backup(conn, "position_close_ledger", "id", pcl[0], "multi_chunk_fee_repair", REASON)
        conn.execute("UPDATE position_close_ledger SET realized_profit=? WHERE id=?", (float(pcl[1]) - missing, pcl[0]))
    tp = conn.execute("SELECT id, pnl FROM trade_performance WHERE trade_id=?", (paper["trade_id"],)).fetchone()
    if tp:
        _backup(conn, "trade_performance", "id", tp[0], "multi_chunk_fee_repair", REASON)
        conn.execute("UPDATE trade_performance SET pnl=? WHERE id=?", (float(tp[1] or 0.0) - missing, tp[0]))
    return {
        "paper_id": paper["id"],
        "fill_row": f["id"],
        "orders": v["order_ids"],
        "missing_fee": missing,
        "before": before,
        "after": {**before, **sets},
        "tlo": tlo[0] if tlo else None,
        "close_ledger": pcl[0] if pcl else None,
    }


# ------------------------------------------------------------ labels/setup ---


def plan_label_repairs(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    from backend.services.engine_lot_ownership import GENERIC_MANUAL_REASONS

    conn.row_factory = sqlite3.Row
    ledger = []
    for r in conn.execute(
        "SELECT id, symbol, close_reason, sell_trade_id, closed_at FROM position_close_ledger WHERE UPPER(close_reason) IN ({})".format(",".join("?" * len(GENERIC_MANUAL_REASONS))),
        tuple(GENERIC_MANUAL_REASONS),
    ).fetchall():
        p = conn.execute("SELECT exit_reason, engine_id FROM paper_trades WHERE trade_id=? AND UPPER(side)='SELL'", (r["sell_trade_id"],)).fetchone()
        if p and str(p["engine_id"] or "") in ("DAY_V2", "SCALP_V2") and str(p["exit_reason"] or "").upper() not in GENERIC_MANUAL_REASONS and p["exit_reason"]:
            ledger.append({"id": r["id"], "sell_trade_id": r["sell_trade_id"], "before": r["close_reason"], "after": p["exit_reason"]})
    setup = []
    for r in conn.execute(
        """SELECT id, trade_id, engine_id, symbol, timestamp, explainability_json, scalp_opportunity_id FROM paper_trades
           WHERE UPPER(side)='SELL' AND engine_id IN ('DAY_V2','SCALP_V2')"""
    ).fetchall():
        ex = load_json(r["explainability_json"])
        if any(str(ex.get(k) or "").strip() for k in ("setup_type_canonical", "setup_type", "entry_thesis")):
            continue
        fill = conn.execute("SELECT mystic_trade_id FROM live_exchange_fills WHERE exchange_order_id=(SELECT order_id FROM paper_trades WHERE id=?) LIMIT 1", (r["id"],)).fetchone()
        buy_tid = str(fill[0]) if fill and fill[0] else ""
        src, val = "", ""
        if r["engine_id"] == "DAY_V2" and buy_tid:
            bex, _ = _buy_explain(conn, buy_tid)
            val = str(bex.get("setup_type_canonical") or bex.get("setup_type") or "")
            src = "buy_row_explainability"
        elif r["engine_id"] == "SCALP_V2" and r["scalp_opportunity_id"]:
            o = conn.execute("SELECT setup_family FROM scalp_v2_opportunities WHERE opportunity_id=? ORDER BY id DESC LIMIT 1", (r["scalp_opportunity_id"],)).fetchone()
            val, src = (str(o[0]) if o and o[0] else ""), "scalp_v2_opportunities"
        setup.append({"id": r["id"], "engine_id": r["engine_id"], "symbol": r["symbol"], "timestamp": r["timestamp"], "buy_trade_id": buy_tid, "setup": val.upper(), "source": src})
    attribution = [dict(r) for r in conn.execute("SELECT id, trade_id, symbol, closed_at_utc FROM day_outcome_attribution WHERE trade_id LIKE 'scalp%'").fetchall()]
    return {"close_ledger": ledger, "setup": setup, "scalp_in_day_attribution": attribution}


def apply_label_repairs(conn: sqlite3.Connection, plan: dict[str, list[dict[str, Any]]]) -> list[str]:
    done = []
    for r in plan["close_ledger"]:
        _backup(conn, "position_close_ledger", "id", r["id"], "strategy_exit_label", REASON)
        conn.execute("UPDATE position_close_ledger SET close_reason=? WHERE id=?", (r["after"], r["id"]))
        done.append(f"close_ledger {r['id']} {r['before']} -> {r['after']}")
    for r in plan["setup"]:
        if not r["setup"]:
            continue
        row = _backup(conn, "paper_trades", "id", r["id"], "setup_label_backfill", REASON)
        ex = load_json((row or {}).get("explainability_json"))
        ex.update({"setup_type_canonical": r["setup"], "setup_type": r["setup"], "setup_source": r["source"]})
        if r["engine_id"] == "DAY_V2":
            ex.setdefault("entry_thesis", r["setup"])
        conn.execute("UPDATE paper_trades SET explainability_json=? WHERE id=?", (json.dumps(ex, default=str), r["id"]))
        done.append(f"setup paper {r['id']} -> {r['setup']} ({r['source']})")
    for r in plan["scalp_in_day_attribution"]:
        _backup(conn, "day_outcome_attribution", "id", r["id"], "remove_scalp_from_day_attribution", REASON)
        conn.execute("DELETE FROM day_outcome_attribution WHERE id=?", (r["id"],))
        done.append(f"day_attribution {r['id']} ({r['trade_id']}) removed")
    return done


# -------------------------------------------------------------------- main ---


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/mystic/mystic/mystic_trading.db")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="/tmp/audit_repair_20260929.json")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row
    if args.apply:
        _ensure_tables(conn)
        conn.commit()
    ex = await _venue()
    dust_log = list((await asyncio.to_thread(ex.request, "asset/query/dust-logs", "sapi", "GET", {})).get("userDustConvertHistory") or [])
    bal = await asyncio.to_thread(ex.fetch_balance)
    free_total = {k: float(v or 0.0) for k, v in (bal.get("total") or {}).items()}
    report: dict[str, Any] = {"apply": args.apply, "exchange_total": {k: v for k, v in free_total.items() if v}}

    learning = plan_learning(conn)
    report["learning_plan"] = [{k: v for k, v in i.items() if k != "extra"} for i in learning]
    inv = plan_inventory(conn, dust_log)
    report["inventory_plan"] = inv
    labels = plan_label_repairs(conn)
    report["label_plan"] = labels
    chunks = plan_chunks(conn)
    chunk_truth = []
    for c in chunks:
        try:
            v = await fetch_chunk_truth(ex, c)
        except Exception as exc:
            v = None
            c["error"] = str(exc)[:200]
        chunk_truth.append((c, v))
    report["chunk_plan"] = [
        {
            "fill_row": c["fill"]["id"],
            "paper": (c["paper"] or {}).get("id"),
            "verified": bool(v),
            "orders": (v or {}).get("order_ids"),
            "fees": (v or {}).get("fees"),
            "recorded_fee": c["fill"]["fee_amount"],
            "missing_fee": fee_correction(float(c["fill"]["fee_amount"] or 0), v) if v else None,
        }
        for c, v in chunk_truth
    ]

    if args.apply:
        for item in learning:
            if item["action"] != "REPAIR_REAL_CLOSE":
                continue
            row = _backup(conn, "trade_learning_outcomes", "id", item["tlo_id"], "dust_mislabel_metadata", REASON)
            new_extra = repaired_learning_extra(load_json((row or {}).get("extra_json")), item["evidence"], residual_qty=item["residual_qty"])
            conn.execute("UPDATE trade_learning_outcomes SET extra_json=? WHERE id=?", (json.dumps(new_extra, default=str), item["tlo_id"]))
            item["extra"] = new_extra
        conn.commit()
        report["inventory_applied"] = apply_inventory(conn, inv, free_total)
        report["labels_applied"] = apply_label_repairs(conn, labels)
        report["chunks_applied"] = [r for c, v in chunk_truth if v for r in [apply_chunk(conn, c, v)] if r]
        delta = sum(float(r["missing_fee"]) for r in report["chunks_applied"])
        if delta:
            _backup(conn, "portfolio_engine_ledger", "id", 1, "multi_chunk_fee_repair", REASON)
            conn.execute("UPDATE portfolio_engine_ledger SET realized_pnl = realized_pnl - ? WHERE id=1", (delta,))
        conn.commit()
    writers = [run_writers(args.db, conn, i, apply=args.apply) for i in learning if i["action"] == "REPAIR_REAL_CLOSE"]
    report["writers"] = writers
    if args.apply:
        from backend.services.ai_learning_ingestion import ingest_scalp_outcomes

        report["scalp_ingest"] = ingest_scalp_outcomes(args.db)
    Path(args.report).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:60000])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
