"""Venue-proven dust retirement, ownership identity, deposit residual sequencing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.services.balance_residuals import classify_fiat_residual
from backend.services.balance_sync_ownership import load_asset_ownership, quantity_drift
from backend.services.engine_strategy_dust import (
    EVENT_CONVERSION,
    ensure_schema,
    held_lots,
    match_dust_conversion,
    preserve_overwritten_dust,
    record_external_balance_event,
    retire_held_dust,
)
from backend.services.external_capital_flows import parse_fiat_history, sync_flows
from backend.services.live_account_basis import TRAILING_BUY_ANCHOR_EQUITY

REPO = Path(__file__).resolve().parents[1]


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "own.db")
    conn.execute(
        "CREATE TABLE portfolio_engine_positions (engine_id TEXT, symbol TEXT, quantity REAL, status TEXT, trade_id TEXT, entry_price REAL, entry_time REAL, "
        "entry_order_id TEXT, entry_fill_ids_json TEXT, scalp_opportunity_id TEXT, dust_detected_at REAL)"
    )
    conn.execute("CREATE TABLE paper_trades (trade_id TEXT, side TEXT, remaining_position REAL)")
    conn.execute("CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY, principal REAL, cash_balance REAL)")
    conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, 0, 0)")
    ensure_schema(conn)
    return conn


def _dust_pending(conn, engine: str, sym: str, qty: float, tid: str) -> None:
    conn.execute("DELETE FROM portfolio_engine_positions WHERE engine_id=? AND symbol=?", (engine, sym))
    conn.execute(
        "INSERT INTO portfolio_engine_positions (engine_id, symbol, quantity, status, trade_id, entry_price, entry_time) VALUES (?,?,?,?,?,?,?)",
        (engine, sym, qty, "DUST_PENDING", tid, 100.0, 1791000000.0),
    )
    conn.execute("INSERT INTO paper_trades VALUES (?, 'BUY', ?)", (tid, qty))


DUST_LOG = [
    {
        "operateTime": 1791144128000,
        "userAssetDribbletDetails": [
            {"tranId": 2469793917, "fromAsset": "BTC", "amount": "0.00002919", "toAsset": "USDT", "serviceChargeAmount": "0.12467649"},
            {"tranId": 2469793919, "fromAsset": "ETH", "amount": "0.00017614", "toAsset": "USDT"},
        ],
    }
]


def test_two_btc_lots_are_distinct_and_physical_equals_aggregate(tmp_path):
    conn = _db(tmp_path)
    for tid in ("mystic_BTC/USDT_1791022569726", "mystic_BTC/USDT_1791063071020"):
        _dust_pending(conn, "DAY_V2", "BTC/USDT", 0.00000962, tid)
        preserve_overwritten_dust(conn, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id=f"{tid}_next")
    _dust_pending(conn, "DAY_V2", "BTC/USDT", 0.00000995, "mystic_BTC/USDT_1791119779423")
    owned = load_asset_ownership(conn)["BTC"]
    assert abs(owned.total - 0.00002919) < 1e-15
    assert not quantity_drift(0.00002919, owned.total, price=120000.0)
    report = owned.report()
    assert "HELD_DUST[1791022569726]" in report and "HELD_DUST[1791063071020]" in report


def test_same_lot_cannot_be_held_twice(tmp_path):
    conn = _db(tmp_path)
    _dust_pending(conn, "DAY_V2", "BTC/USDT", 0.00000962, "lot_1")
    assert preserve_overwritten_dust(conn, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id="lot_2")
    preserve_overwritten_dust(conn, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id="lot_3")
    assert [h["source_trade_id"] for h in held_lots(conn, "BTC/USDT")] == ["lot_1"]


def test_venue_proven_conversion_retires_dust_idempotently(tmp_path):
    conn = _db(tmp_path)
    for tid in ("a_1", "b_2"):
        _dust_pending(conn, "DAY_V2", "BTC/USDT", 0.00000962, tid)
        preserve_overwritten_dust(conn, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id=f"{tid}x")
    match = match_dust_conversion(DUST_LOG, asset="BTC", after_epoch=1791000000.0)
    assert match and match["tran_id"] == "2469793917" and match["amount"] == 0.00002919
    for _ in range(2):
        for h in held_lots(conn, "BTC/USDT"):
            record_external_balance_event(
                conn, symbol="BTC/USDT", quantity=h["quantity"], event_class=EVENT_CONVERSION, source="test", source_trade_id=h["source_trade_id"], venue_ref=match["tran_id"], evidence=match
            )
            assert retire_held_dust(conn, h["source_trade_id"], event_class=EVENT_CONVERSION, venue_ref=match["tran_id"])
    assert held_lots(conn, "BTC/USDT") == []
    assert conn.execute("SELECT COUNT(*) FROM external_balance_events").fetchone()[0] == 2
    assert not retire_held_dust(conn, "a_1", event_class=EVENT_CONVERSION)
    assert conn.execute("SELECT SUM(remaining_position) FROM paper_trades WHERE trade_id IN ('a_1','b_2')").fetchone()[0] == 0


def test_no_venue_record_means_no_conversion_match():
    assert match_dust_conversion(DUST_LOG, asset="SOL", after_epoch=0) is None
    assert match_dust_conversion(DUST_LOG, asset="BTC", after_epoch=1791144129.0) is None


def test_day_scalp_and_dust_same_coin_reconcile_and_exchange_never_creates_ownership(tmp_path):
    conn = _db(tmp_path)
    conn.execute("INSERT INTO portfolio_engine_positions (engine_id, symbol, quantity, status, trade_id) VALUES ('DAY_V2','XRP/USDT',145.17096,'ACTIVE','d')")
    conn.execute("INSERT INTO portfolio_engine_positions (engine_id, symbol, quantity, status, trade_id) VALUES ('SCALP_V2','XRP/USDT',0.0933,'DUST_PENDING','s')")
    conn.execute(
        "INSERT INTO engine_strategy_dust (engine_id, symbol, source_trade_id, quantity, status, created_at) VALUES ('DAY_V2','XRP/USDT','mystic_XRP/USDT_1791003670268',0.06976,'HELD','now')"
    )
    owned = load_asset_ownership(conn)
    assert abs(owned["XRP"].total - 145.33402) < 1e-9
    assert not quantity_drift(145.33402, owned["XRP"].total, price=2.5)
    assert quantity_drift(146.33402, owned["XRP"].total, price=2.5)
    assert "BTC" not in owned


def _flow_payload(order_id: str, amount: str, create_ms: int) -> dict:
    return {"assetLogRecordList": [{"orderId": order_id, "orderStatus": "Successful", "fiatCurrency": "USD", "amount": amount, "transactionFee": "13.97", "createTime": create_ms}]}


def test_deposit_applied_once(tmp_path):
    conn = _db(tmp_path)
    conn.close()
    db = str(tmp_path / "own.db")
    flows = parse_fiat_history(_flow_payload("c907857b", "336.03", 1791144082789), direction="DEPOSIT")
    new1, base1 = sync_flows(db, flows)
    new2, base2 = sync_flows(db, flows)
    assert len(new1) == 1 and new2 == []
    assert base1 == base2 == float(TRAILING_BUY_ANCHOR_EQUITY) + 336.03


def test_new_deposit_remainder_reclassified_once_but_genuine_change_warns(tmp_path):
    conn = _db(tmp_path)
    conn.close()
    db = str(tmp_path / "own.db")
    with sqlite3.connect(db) as c:
        assert classify_fiat_residual(c, "USD", 0.098) == "MATCHED"
        c.execute("UPDATE documented_balance_residuals SET updated_at='2026-10-03 15:59:03'")
        assert classify_fiat_residual(c, "USD", 0.0080) == "CHANGED"
    sync_flows(db, parse_fiat_history(_flow_payload("c907857b", "336.03", 1791144082789), direction="DEPOSIT"))
    with sqlite3.connect(db) as c:
        assert classify_fiat_residual(c, "USD", 0.0080) == "MATCHED"
        assert classify_fiat_residual(c, "USD", 0.0080) == "MATCHED"
        assert classify_fiat_residual(c, "USD", 0.0500) == "CHANGED"
        assert classify_fiat_residual(c, "USD", 250.0) == "CHANGED"


def test_flow_sync_runs_before_balance_comparison():
    src = (REPO / "backend/services/portfolio_engine_integration.py").read_text()
    loop = src[src.index("# Query Binance account balance") - 600 : src.index("async def _sync_external_capital_flows")]
    assert loop.index("_sync_external_capital_flows(api_key, api_secret)") < loop.index("# Query Binance account balance")
    assert loop.count("_sync_external_capital_flows(") == 1
