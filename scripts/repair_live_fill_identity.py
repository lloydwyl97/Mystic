"""Backfill venue trade ids and document balance gaps. Does not place orders.

Run only while the portfolio engine is stopped. Idempotent.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv("/home/mystic/mystic/.env")

from backend.services.scalp_v2.accounting_repair import (
    apply_trade_id_backfill,
    exclude_duplicate_realized,
    record_residual,
    record_supplements,
)
from backend.utils.binance_credentials import get_binance_us_api_key, get_binance_us_secret_key

DB = "/home/mystic/mystic/mystic_trading.db"


def main() -> int:
    import ccxt

    key = get_binance_us_api_key()
    secret = get_binance_us_secret_key()
    if not key or not secret:
        print("missing binance credentials")
        return 1
    ex = ccxt.binanceus({"apiKey": key, "secret": secret, "enableRateLimit": True})
    conn = sqlite3.connect(DB, timeout=60)
    conn.row_factory = sqlite3.Row
    known = {str(r[0]) for r in conn.execute("SELECT exchange_order_id FROM live_exchange_fills")}
    empty = list(
        conn.execute(
            """
            SELECT exchange_order_id, symbol, side, event_ts_exchange
            FROM live_exchange_fills
            WHERE COALESCE(venue_trade_ids_json,'[]') IN ('[]','','null')
            """
        )
    )
    print("empty_trade_id_rows", len(empty), "known_orders", len(known))
    by_order: dict[str, dict] = {}
    venue_orders: dict[str, dict] = {}
    symbols = sorted({str(r["symbol"]) for r in empty} | {"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"})
    since = int(datetime(2026, 9, 15, tzinfo=timezone.utc).timestamp() * 1000)
    for sym in symbols:
        cursor = since
        while True:
            batch = ex.fetch_my_trades(sym, since=cursor, limit=500)
            if not batch:
                break
            for t in batch:
                info = t.get("info") or {}
                oid = str(t.get("order") or info.get("orderId") or "")
                tid = str(t.get("id") or info.get("id") or "")
                fee = t.get("fee") or {}
                slot = venue_orders.setdefault(
                    oid,
                    {
                        "symbol": t.get("symbol"),
                        "side": str(t.get("side") or "").upper(),
                        "trade_ids": [],
                        "qty": 0.0,
                        "quote": 0.0,
                        "fee_amount": 0.0,
                        "fee_asset": fee.get("currency") or "",
                        "taker_or_maker": t.get("takerOrMaker") or "",
                    },
                )
                if tid and tid not in slot["trade_ids"]:
                    slot["trade_ids"].append(tid)
                slot["qty"] += float(t.get("amount") or 0)
                slot["quote"] += float(t.get("cost") or 0)
                slot["fee_amount"] += float(fee.get("cost") or 0)
            last = batch[-1]["timestamp"]
            if last <= cursor or len(batch) < 500:
                break
            cursor = last + 1
        print("fetched", sym, "orders", sum(1 for o in venue_orders.values() if o["symbol"] == sym))

    for oid, slot in venue_orders.items():
        if not oid or oid not in known:
            continue
        price = (slot["quote"] / slot["qty"]) if slot["qty"] else 0
        by_order[oid] = {
            "trade_ids": slot["trade_ids"],
            "order_ids": [oid],
            "taker_or_maker": slot["taker_or_maker"],
            "price": price,
        }
    updated = apply_trade_id_backfill(conn, by_order)
    missing = []
    for oid, slot in venue_orders.items():
        if not oid or oid in known:
            continue
        price = (slot["quote"] / slot["qty"]) if slot["qty"] else 0
        missing.append(
            {
                "exchange_order_id": oid,
                "symbol": slot["symbol"],
                "side": slot["side"],
                "trade_ids": slot["trade_ids"],
                "qty": slot["qty"],
                "price": price,
                "fee_amount": slot["fee_amount"],
                "fee_asset": slot["fee_asset"],
                "taker_or_maker": slot["taker_or_maker"],
                "parent_order_id": "",
                "note": "venue order absent from live_exchange_fills; quantity stays on the aggregated row",
            }
        )
    added = record_supplements(conn, missing)
    flagged = exclude_duplicate_realized(conn)
    bal = ex.fetch_balance()
    totals = bal.get("total") or {}
    names = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT"}
    print("RESIDUALS")
    for asset, sym in names.items():
        exch = float(totals.get(asset) or 0)
        row = conn.execute(
            "SELECT COALESCE(SUM(quantity),0) FROM portfolio_engine_positions WHERE symbol=?",
            (sym,),
        ).fetchone()
        pos = float(row[0] or 0)
        record_residual(conn, sym, exch, pos, "exchange total minus summed position lots including dust")
        print(sym, "exchange", exch, "positions", pos, "gap", round(exch - pos, 8))
    conn.commit()
    still_empty = conn.execute("SELECT COUNT(*) FROM live_exchange_fills WHERE COALESCE(venue_trade_ids_json,'[]') IN ('[]','','null')").fetchone()[0]
    print("backfilled", updated, "supplements", added, "realized_flags", flagged, "still_empty", still_empty)
    conn.close()
    return 0 if still_empty == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
