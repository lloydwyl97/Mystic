"""Probe the live fill identity chain end to end.

Reports, for recent live BUYs, whether each link of
decision -> intent -> reservation -> client_order -> exchange_order -> venue
fill -> position is actually persisted, and where the venue response keeps the
per-fill trade ids. Read-only.
"""

from __future__ import annotations

import json
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "/home/mystic/mystic/mystic_trading.db"


def cols(conn, table):
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=20)
    conn.row_factory = sqlite3.Row

    print("=" * 78)
    print("LIVE_EXCHANGE_FILLS")
    print("=" * 78)
    lef = cols(conn, "live_exchange_fills")
    if not lef:
        print("  table absent")
    else:
        n = conn.execute("SELECT COUNT(*) FROM live_exchange_fills").fetchone()[0]
        print(f"  rows: {n}")
        print(
            "  with fill_ids: {}   with venue_trade_ids: {}   with intent: {}   with decision: {}".format(
                *conn.execute(
                    """SELECT
                        SUM(CASE WHEN COALESCE(fill_ids_json,'[]') NOT IN ('[]','','null') THEN 1 ELSE 0 END),
                        SUM(CASE WHEN COALESCE(venue_trade_ids_json,'[]') NOT IN ('[]','','null') THEN 1 ELSE 0 END),
                        SUM(CASE WHEN LENGTH(COALESCE(intent_id,''))>0 THEN 1 ELSE 0 END),
                        SUM(CASE WHEN LENGTH(COALESCE(decision_id,''))>0 THEN 1 ELSE 0 END)
                       FROM live_exchange_fills"""
                ).fetchone()
            )
        )
        print("\n  --- newest 6 rows ---")
        for r in conn.execute(
            """SELECT id, symbol, side, exchange_order_id, client_order_id, fill_ids_json,
                      venue_trade_ids_json, fill_count, executed_qty, avg_fill_price,
                      mystic_trade_id, intent_id, decision_id, order_status,
                      missing_fields_json, event_ts_exchange
               FROM live_exchange_fills ORDER BY id DESC LIMIT 6"""
        ):
            d = dict(r)
            print(f"  #{d['id']} {d['symbol']} {d['side']} order={d['exchange_order_id']} coid={d['client_order_id']} qty={d['executed_qty']} px={d['avg_fill_price']}")
            print(f"      fill_ids={d['fill_ids_json']} venue_trade_ids={d['venue_trade_ids_json']} count={d['fill_count']} status={d['order_status']}")
            print(f"      trade={d['mystic_trade_id']} intent={d['intent_id']} decision={d['decision_id']}")
            print(f"      missing={d['missing_fields_json']} ts={d['event_ts_exchange']}")

        # Where does the venue actually put the per-fill trade ids?
        print("\n  --- venue response shape (newest row with raw_json) ---")
        row = conn.execute("SELECT id, raw_json FROM live_exchange_fills WHERE LENGTH(COALESCE(raw_json,''))>2 ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            print("      no raw_json stored")
        else:
            try:
                raw = json.loads(row["raw_json"])
            except Exception as exc:
                print(f"      unparseable: {exc}")
                raw = {}
            print(f"      row #{row['id']} top-level keys: {sorted(raw.keys())}")
            print(f"      raw['trades'] = {raw.get('trades')!r}")
            info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
            print(f"      info keys: {sorted(info.keys())}")
            fills = info.get("fills")
            print(f"      info['fills'] present={fills is not None} type={type(fills).__name__}")
            if isinstance(fills, list) and fills:
                print(f"      info['fills'][0] = {json.dumps(fills[0], default=str)}")
                print(f"      info['fills'] count = {len(fills)}")
                keys = sorted({k for f in fills if isinstance(f, dict) for k in f})
                print(f"      per-fill keys = {keys}")

    print()
    print("=" * 78)
    print("TRAILING BUY INTENTS (terminal)")
    print("=" * 78)
    ti = cols(conn, "day_trailing_buy_intents")
    if not ti:
        print("  table absent")
    else:
        print(f"  columns: {ti}")
        for r in conn.execute(
            """SELECT * FROM day_trailing_buy_intents
               WHERE UPPER(COALESCE(status,'')) IN ('FILLED','EXPIRED','CANCELLED','CANCELED')
               ORDER BY rowid DESC LIMIT 6"""
        ):
            d = dict(r)
            keep = {
                k: d.get(k)
                for k in (
                    "intent_id",
                    "symbol",
                    "status",
                    "decision_id",
                    "reservation_id",
                    "client_order_id",
                    "exchange_order_id",
                    "order_id",
                    "trade_id",
                    "fill_id",
                    "filled_price",
                    "filled_qty",
                )
                if k in d
            }
            print(f"  {json.dumps(keep, default=str)}")

    print()
    print("=" * 78)
    print("RESERVATIONS")
    print("=" * 78)
    rv = cols(conn, "day_entry_reservations")
    if not rv:
        print("  table absent")
    else:
        print(f"  columns: {rv}")
        for r in conn.execute("SELECT status, COUNT(*) c FROM day_entry_reservations GROUP BY status"):
            print(f"  status={r['status']:<12} count={r['c']}")
        print("\n  --- newest 8 ---")
        for r in conn.execute("SELECT * FROM day_entry_reservations ORDER BY rowid DESC LIMIT 8"):
            d = dict(r)
            keep = {k: d.get(k) for k in ("reservation_id", "symbol", "status", "amount_usd", "intent_id", "decision_id", "created_at", "released_at") if k in d}
            print(f"  {json.dumps(keep, default=str)}")

    print()
    print("=" * 78)
    print("POSITIONS: traceability columns")
    print("=" * 78)
    pc = cols(conn, "portfolio_engine_positions")
    trace = [c for c in pc if any(t in c for t in ("decision", "intent", "order", "fill", "reservation"))]
    print(f"  traceability columns present: {trace}")
    if trace:
        sel = ", ".join(["symbol", "quantity", "entry_price", *trace])
        for r in conn.execute(f"SELECT {sel} FROM portfolio_engine_positions ORDER BY rowid DESC LIMIT 8"):
            print(f"  {json.dumps(dict(r), default=str)}")
        for c in trace:
            n = conn.execute(f"SELECT COUNT(*) FROM portfolio_engine_positions WHERE LENGTH(COALESCE({c},''))>0").fetchone()[0]
            tot = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
            print(f"  {c}: {n}/{tot} populated")

    print()
    print("=" * 78)
    print("LIVE ROWS IN paper_trades vs live_exchange_fills")
    print("=" * 78)
    live = "LOWER(COALESCE(mode,''))='live' AND COALESCE(is_synthetic,0)=0"
    tot = conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE {live}").fetchone()[0]
    with_oid = conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE {live} AND LENGTH(COALESCE(order_id,''))>0").fetchone()[0]
    print(f"  live paper_trades rows: {tot}   with order_id: {with_oid}")
    try:
        orphan = conn.execute(
            f"""SELECT COUNT(*) FROM live_exchange_fills lef
                WHERE NOT EXISTS (SELECT 1 FROM paper_trades pt
                                  WHERE {live} AND pt.order_id = lef.exchange_order_id)"""
        ).fetchone()[0]
        print(f"  live_exchange_fills rows with no matching paper_trades.order_id: {orphan}")
        for r in conn.execute(
            f"""SELECT lef.id, lef.symbol, lef.side, lef.exchange_order_id, lef.event_ts_exchange
                FROM live_exchange_fills lef
                WHERE NOT EXISTS (SELECT 1 FROM paper_trades pt
                                  WHERE {live} AND pt.order_id = lef.exchange_order_id)
                ORDER BY lef.id DESC LIMIT 5"""
        ):
            print(f"      unmatched: {dict(r)}")
    except sqlite3.Error as exc:
        print(f"  join failed: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
