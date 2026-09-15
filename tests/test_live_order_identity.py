"""Every confirmed live fill must persist its exchange identity.

Before this, a confirmed Binance.US fill left no durable identifier: the order
id, client order id, per-fill trade ids, fee asset and venue timestamps were
read into memory, logged, then dropped. ``paper_trades.order_id`` was NULL on
all 502 recorded live rows and a live SELL stored no identifier at all, so
reconciliation had to pair rows to fills by quantity and time.

These tests pin the identifier set and the chain that links the trailing-buy
intent, the FIFO trade rows, the position, recovery state and reconciliation.
"""

from __future__ import annotations

import json
import sqlite3

from backend.services.live_order_identity import (
    REQUIRED_FIELDS,
    TABLE,
    coverage,
    ensure_schema,
    extract_identity,
    fills_for_intent,
    fills_for_order,
    fills_for_trade,
    iso_or_blank,
    record_fill,
)

# Shape of a real Binance.US market-buy response as CCXT returns it, including
# the per-fill trade list and the base-asset commission Binance actually
# charges on a spot buy.
BINANCE_BUY = {
    "id": "26417215",
    "clientOrderId": "mystic_day_btc_1788600000",
    "symbol": "BTC/USDT",
    "side": "buy",
    "type": "market",
    "status": "closed",
    "amount": 0.00021,
    "price": 95000.0,
    "average": 95012.34,
    "filled": 0.00021,
    "cost": 19.9525914,
    "timestamp": 1788600000123,
    "fee": {"cost": 2.1e-07, "currency": "BTC"},
    "trades": [
        {"id": "5512341", "order": "26417215", "amount": 0.00012, "price": 95010.0, "fee": {"cost": 1.2e-07, "currency": "BTC"}},
        {"id": "5512342", "order": "26417215", "amount": 0.00009, "price": 95015.5, "fee": {"cost": 9e-08, "currency": "BTC"}},
    ],
    "info": {"status": "FILLED", "executedQty": "0.00021000", "cummulativeQuoteQty": "19.95259140"},
}

BINANCE_SELL = {
    "id": "26417999",
    "clientOrderId": "mystic_day_btc_exit_1788603600",
    "symbol": "BTC/USDT",
    "side": "sell",
    "status": "closed",
    "filled": 0.00021,
    "average": 95480.0,
    "cost": 20.0508,
    "timestamp": 1788603600456,
    "fee": {"cost": 0.02005, "currency": "USDT"},
    "trades": [{"id": "5512500", "order": "26417999", "amount": 0.00021, "price": 95480.0}],
    "info": {"status": "FILLED", "executedQty": "0.00021000", "cummulativeQuoteQty": "20.05080000"},
}


def _db(tmp_path) -> str:
    path = str(tmp_path / "identity.db")
    ensure_schema(path)
    return path


def _rows(db: str) -> list[dict]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE} ORDER BY id")]


def test_every_required_identifier_is_extracted_from_a_real_buy_response():
    ident = extract_identity(
        BINANCE_BUY,
        symbol="BTC/USDT",
        side="BUY",
        mystic_trade_id="mystic_buy_BTCUSDT_1788600000",
        intent_id="tbi_btc_1",
        client_order_id="mystic_day_btc_1788600000",
    )
    assert ident.exchange_order_id == "26417215"
    assert ident.client_order_id == "mystic_day_btc_1788600000"
    assert ident.fill_ids == ["5512341", "5512342"]
    assert ident.executed_qty == 0.00021
    assert ident.avg_fill_price == 95012.34
    assert ident.fee_amount == 2.1e-07
    assert ident.fee_asset == "BTC"
    assert ident.symbol == "BTC/USDT"
    assert ident.side == "BUY"
    assert ident.event_ts_exchange.startswith("2026-")
    assert ident.missing() == []


def test_sell_response_persists_the_same_identifier_set():
    ident = extract_identity(BINANCE_SELL, symbol="BTC/USDT", side="SELL", mystic_trade_id="mystic_sell_BTCUSDT_1788603600")
    assert ident.exchange_order_id == "26417999"
    assert ident.client_order_id == "mystic_day_btc_exit_1788603600"
    assert ident.fill_ids == ["5512500"]
    assert ident.executed_qty == 0.00021
    assert ident.avg_fill_price == 95480.0
    assert ident.fee_amount == 0.02005
    assert ident.fee_asset == "USDT"
    assert ident.missing() == []


def test_required_field_set_is_the_full_identifier_contract():
    """Guard against silently shrinking what counts as a complete identity."""
    assert set(REQUIRED_FIELDS) == {
        "exchange_order_id",
        "symbol",
        "side",
        "executed_qty",
        "avg_fill_price",
        "event_ts_exchange",
    }


def test_fee_asset_comes_from_the_engine_commission_breakdown_when_given():
    # extract_live_commission emits {"amount", "asset", "usd"} per item.
    ident = extract_identity(
        {"id": "1", "filled": 1.0, "average": 10.0, "timestamp": 1788600000000},
        symbol="SOL/USDT",
        side="BUY",
        fee_items=[{"amount": 0.001, "asset": "SOL", "usd": 0.01}],
        fee_amount=0.01,
    )
    assert ident.fee_asset == "SOL"
    assert ident.missing() == []


def test_multi_asset_fee_keeps_every_asset():
    ident = extract_identity(
        {"id": "1", "filled": 1.0, "average": 10.0, "timestamp": 1788600000000},
        symbol="SOL/USDT",
        side="BUY",
        fee_items=[{"amount": 0.001, "asset": "SOL", "usd": 0.01}, {"amount": 0.002, "asset": "BNB", "usd": 0.0}],
        fee_amount=0.01,
    )
    assert ident.fee_asset == "SOL,BNB"


def test_fee_asset_is_required_only_once_a_fee_was_charged():
    zero_fee = extract_identity({"id": "1", "filled": 1.0, "average": 10.0, "timestamp": 1788600000000}, symbol="SOL/USDT", side="BUY")
    assert zero_fee.fee_amount == 0.0
    assert "fee_asset" not in zero_fee.missing()

    charged = extract_identity(
        {"id": "1", "filled": 1.0, "average": 10.0, "timestamp": 1788600000000, "commission": 0.5},
        symbol="SOL/USDT",
        side="BUY",
    )
    assert charged.fee_amount == 0.5
    assert "fee_asset" in charged.missing()


def test_executed_quantity_and_price_fall_back_to_engine_values_only_when_venue_omits_them():
    venue_silent = extract_identity(
        {"id": "9", "timestamp": 1788600000000},
        symbol="XRP/USDT",
        side="BUY",
        fallback_qty=12.0,
        fallback_price=1.42,
    )
    assert venue_silent.executed_qty == 12.0
    assert venue_silent.avg_fill_price == 1.42
    # The venue's own figures win, because they are what settled.
    venue_spoke = extract_identity(
        {"id": "9", "filled": 11.5, "average": 1.4188, "timestamp": 1788600000000},
        symbol="XRP/USDT",
        side="BUY",
        fallback_qty=12.0,
        fallback_price=1.42,
    )
    assert venue_spoke.executed_qty == 11.5
    assert venue_spoke.avg_fill_price == 1.4188


def test_partial_fill_records_what_actually_executed_not_what_was_requested():
    partial = dict(BINANCE_BUY)
    partial["status"] = "open"
    partial["amount"] = 0.00050
    partial["filled"] = 0.00021
    ident = extract_identity(partial, symbol="BTC/USDT", side="BUY", fallback_qty=0.00050)
    assert ident.executed_qty == 0.00021
    assert ident.order_status == "open"


def test_recorded_fill_is_readable_by_order_trade_and_intent(tmp_path):
    db = _db(tmp_path)
    ident = extract_identity(
        BINANCE_BUY,
        symbol="BTC/USDT",
        side="BUY",
        mystic_trade_id="mystic_buy_BTCUSDT_1788600000",
        intent_id="tbi_btc_1",
        decision_id="day_BTCUSDT_1",
    )
    assert record_fill(db, ident) is True

    by_order = fills_for_order(db, "26417215")
    by_trade = fills_for_trade(db, "mystic_buy_BTCUSDT_1788600000")
    by_intent = fills_for_intent(db, "tbi_btc_1")
    assert len(by_order) == len(by_trade) == len(by_intent) == 1
    row = by_order[0]
    assert row["client_order_id"] == "mystic_day_btc_1788600000"
    assert json.loads(row["fill_ids_json"]) == ["5512341", "5512342"]
    assert row["fill_count"] == 2
    assert row["fee_asset"] == "BTC"
    assert row["decision_id"] == "day_BTCUSDT_1"
    assert row["event_ts_recorded"]
    assert json.loads(row["missing_fields_json"]) == []


def test_identity_links_intent_to_trade_row_to_reconciliation(tmp_path):
    """The chain the repair exists to create.

    intent_id -> exchange_order_id -> mystic_trade_id is navigable in both
    directions from one table, so a FIFO row can be traced to a venue order and
    a venue order back to the intent that armed it.
    """
    db = _db(tmp_path)
    record_fill(
        db,
        extract_identity(
            BINANCE_BUY,
            symbol="BTC/USDT",
            side="BUY",
            mystic_trade_id="mystic_buy_BTCUSDT_1788600000",
            intent_id="tbi_btc_1",
        ),
    )
    record_fill(
        db,
        extract_identity(
            BINANCE_SELL,
            symbol="BTC/USDT",
            side="SELL",
            mystic_trade_id="mystic_sell_BTCUSDT_1788603600",
            position_entry_ts=iso_or_blank(1788600000.0),
        ),
    )

    entry = fills_for_intent(db, "tbi_btc_1")[0]
    assert entry["side"] == "BUY"
    assert entry["mystic_trade_id"] == "mystic_buy_BTCUSDT_1788600000"

    exit_rows = [r for r in _rows(db) if r["side"] == "SELL"]
    assert len(exit_rows) == 1
    # The exit carries the position it closed, so entry and exit join without
    # relying on quantity/time inference.
    assert exit_rows[0]["position_entry_ts"] == iso_or_blank(1788600000.0)
    assert exit_rows[0]["exchange_order_id"] == "26417999"


def test_same_order_is_never_double_booked(tmp_path):
    db = _db(tmp_path)
    ident = extract_identity(BINANCE_BUY, symbol="BTC/USDT", side="BUY", mystic_trade_id="t1")
    assert record_fill(db, ident) is True
    assert record_fill(db, ident) is True  # retry of the same commit
    assert len(fills_for_order(db, "26417215")) == 1


def test_buy_and_sell_of_the_same_order_id_are_separate_rows(tmp_path):
    db = _db(tmp_path)
    record_fill(db, extract_identity({**BINANCE_BUY, "id": "SAME"}, symbol="BTC/USDT", side="BUY", mystic_trade_id="t1"))
    record_fill(db, extract_identity({**BINANCE_SELL, "id": "SAME"}, symbol="BTC/USDT", side="SELL", mystic_trade_id="t2"))
    assert len(fills_for_order(db, "SAME")) == 2


def test_a_fill_with_no_order_id_is_refused_and_logged(tmp_path, caplog):
    db = _db(tmp_path)
    ident = extract_identity({"filled": 1.0, "average": 2.0}, symbol="ETH/USDT", side="BUY")
    assert ident.exchange_order_id == ""
    assert record_fill(db, ident) is False
    assert _rows(db) == []
    assert "LIVE_FILL_IDENTITY_MISSING_ORDER_ID" in caplog.text


def test_incomplete_identity_is_still_stored_but_flagged(tmp_path, caplog):
    """A settled trade is never discarded over bookkeeping; it is flagged."""
    db = _db(tmp_path)
    ident = extract_identity({"id": "77", "filled": 1.0, "average": 3.0}, symbol="ETH/USDT", side="BUY")
    assert record_fill(db, ident) is True
    row = fills_for_order(db, "77")[0]
    assert json.loads(row["missing_fields_json"]) == ["event_ts_exchange"]
    assert "LIVE_FILL_IDENTITY_INCOMPLETE" in caplog.text


def test_timestamps_are_iso_and_survive_second_or_millisecond_input():
    ms = extract_identity({"id": "1", "filled": 1.0, "average": 1.0, "timestamp": 1788600000123}, symbol="A/USDT", side="BUY")
    sec = extract_identity({"id": "1", "filled": 1.0, "average": 1.0, "timestamp": 1788600000}, symbol="A/USDT", side="BUY")
    assert ms.event_ts_exchange.startswith("2026-")
    assert sec.event_ts_exchange.startswith("2026-")
    assert iso_or_blank(0) == ""
    assert iso_or_blank(None) == ""
    assert iso_or_blank("not-a-number") == ""


def test_transact_time_is_used_when_top_level_timestamp_is_absent():
    ident = extract_identity(
        {"id": "1", "filled": 1.0, "average": 1.0, "info": {"transactTime": 1788600000123}},
        symbol="A/USDT",
        side="BUY",
    )
    assert ident.event_ts_exchange.startswith("2026-")
    assert ident.missing() == []


def test_coverage_reports_identifier_completeness(tmp_path):
    db = str(tmp_path / "cov.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (trade_id TEXT, mode TEXT, side TEXT, order_id TEXT, is_synthetic INTEGER DEFAULT 0)")
        conn.executemany(
            "INSERT INTO paper_trades (trade_id, mode, side, order_id) VALUES (?,?,?,?)",
            [("old1", "live", "BUY", None), ("old2", "live", "SELL", None), ("new1", "live", "BUY", "26417215")],
        )
        conn.commit()
    ensure_schema(db)
    record_fill(db, extract_identity(BINANCE_BUY, symbol="BTC/USDT", side="BUY", mystic_trade_id="new1"))

    cov = coverage(db)
    assert cov["live_rows"] == 3
    # The two historical rows predate identifier persistence; the new one does not.
    assert cov["live_rows_with_order_id"] == 1
    assert cov["identity_rows"] == 1
    assert cov["identity_rows_incomplete"] == 0
    assert cov["identity_order_ids"] == 1


def test_readers_are_safe_before_the_table_exists(tmp_path):
    missing = str(tmp_path / "absent.db")
    assert fills_for_order(missing, "1") == []
    assert fills_for_trade(missing, "1") == []
    assert fills_for_intent(missing, "1") == []
    assert coverage(missing)["identity_rows"] == 0
