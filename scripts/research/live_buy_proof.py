"""Read-only: trace one real post-deploy BUY from ranked decision to exchange fill.

Section 3 asks for live proof on the next natural qualifying event rather than a
forced order. This stitches together the decision record, the trailing-buy intent,
the hold episodes, the trade row and the resulting position for the newest BUY.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def iso(ts: object) -> str:
    try:
        f = float(ts)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(ts)
    if f > 2e12:
        f /= 1000.0
    return dt.datetime.fromtimestamp(f, tz=dt.timezone.utc).strftime("%H:%M:%S")


def cols(c: sqlite3.Connection, t: str) -> list[str]:
    return [r[1] for r in c.execute(f"PRAGMA table_info({t})")]


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== newest BUY rows in paper_trades ===")
    pc = cols(c, "paper_trades")
    want = [
        x
        for x in (
            "trade_id",
            "symbol",
            "side",
            "quantity",
            "price",
            "commission",
            "order_id",
            "status",
            "decision_id",
            "mode",
            "created_at",
            "entry_fee_usd",
            "slippage_cost",
            "liquidity_side",
            "exit_reason",
        )
        if x in pc
    ]
    sel = ", ".join(want)
    rows = list(c.execute(f"SELECT {sel} FROM paper_trades WHERE UPPER(side)='BUY' ORDER BY created_at DESC LIMIT 4"))
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in dict(r).items() if v not in (None, "")))

    if not rows:
        print("  none")
        return

    newest = rows[0]
    did = str(newest["decision_id"] or "") if "decision_id" in dict(newest) else ""
    tid = str(newest["trade_id"] or "") if "trade_id" in dict(newest) else ""
    sym = str(newest["symbol"] or "")
    print(f"\n### tracing symbol={sym} decision_id={did} trade_id={tid}")

    print("\n=== 1. decision record for that decision_id ===")
    got = list(c.execute("SELECT * FROM day_decision_records WHERE decision_id=?", (did,)))
    if not got:
        print("  NO DECISION ROW (ledger gap)")
    for r in got:
        d = dict(r)
        print(f"  created={d.get('created_at')} symbol={d.get('symbol')} final={d.get('final_decision')} block={d.get('first_hard_block')!r} mode={d.get('mode')}")
        print(f"  setup={d.get('setup')} ml_score={d.get('ml_score')}")
        print(f"  detail={str(d.get('detail_json'))[:220]}")

    print("\n=== 2. trailing-buy intent lifecycle (hold episodes) ===")
    try:
        eps = list(
            c.execute(
                "SELECT * FROM day_decision_hold_episodes WHERE decision_id=? ORDER BY first_seen_ts",
                (did,),
            )
        )
        if not eps:
            print("  no episodes for this decision_id")
        for e in eps:
            d = dict(e)
            print(
                f"  {iso(d.get('first_seen_ts'))}->{iso(d.get('last_seen_ts'))} "
                f"{d.get('category')!s:<22} reason={d.get('exact_reason')!s:<20} "
                f"n={d.get('observation_count')} intent={str(d.get('intent_id'))[:20]}"
            )
            print(f"      observed={str(d.get('last_observed_json'))[:110]}")
            print(f"      required={str(d.get('required_json'))[:110]}")
    except sqlite3.OperationalError as e:
        print(f"  {e}")

    print("\n=== 3. audit row (fees, slippage, invariant) ===")
    try:
        acols = cols(c, "portfolio_engine_audit")
        key = "decision_id" if "decision_id" in acols else "trade_id"
        val = did if key == "decision_id" else tid
        for r in c.execute(
            f"SELECT * FROM portfolio_engine_audit WHERE {key}=?",
            (val,),
        ):
            d = dict(r)
            print(f"  ts={iso(d.get('ts'))} action={d.get('action')} qty={d.get('qty')} price={d.get('price')} fees={d.get('fees')} slip={d.get('slippage')} invariant_ok={d.get('invariant_ok')}")
            print(f"      entry_reason={d.get('entry_reason')} sleeve={d.get('sleeve')}")
    except sqlite3.OperationalError as e:
        print(f"  {e}")

    print("\n=== 4. resulting position ===")
    try:
        for r in c.execute(
            "SELECT symbol, quantity, entry_price, entry_time, trade_id, status, "
            "entry_decision_id, entry_fee, stop_price, take_profit_1_price, "
            "trailing_stop_price, highest_price, lowest_price "
            "FROM portfolio_engine_positions WHERE symbol=?",
            (sym,),
        ):
            d = dict(r)
            print(f"  entry={iso(d.get('entry_time'))} " + "  ".join(f"{k}={v}" for k, v in d.items() if k != "entry_time"))
    except sqlite3.OperationalError as e:
        print(f"  {e}")

    print("\n=== 5. exchange identifiers present? ===")
    for k in ("order_id", "trade_id"):
        if k in dict(newest):
            v = newest[k]
            print(f"  {k:<12} {v!r}  durable={bool(str(v or '').strip())}")

    print("\n=== 6. active trailing-buy intents right now ===")
    for t in ("day_trailing_buy_intents", "trailing_buy_intents"):
        try:
            icols = cols(c, t)
            if not icols:
                continue
            print(f"  table {t}: {', '.join(icols)}")
            for r in c.execute(f"SELECT * FROM {t} ORDER BY rowid DESC LIMIT 6"):
                d = dict(r)
                keep = {
                    k: d[k]
                    for k in (
                        "symbol",
                        "status",
                        "intent_id",
                        "decision_id",
                        "arm_ask",
                        "trail_low",
                        "expires_at",
                    )
                    if k in d
                }
                print("    " + "  ".join(f"{k}={v}" for k, v in keep.items()))
        except sqlite3.OperationalError:
            continue


if __name__ == "__main__":
    main()
