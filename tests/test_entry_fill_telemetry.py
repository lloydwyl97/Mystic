import sqlite3

from backend.services.entry_fill_telemetry import (
    build_entry_reference_telemetry,
    persist_entry_reference_row,
)


def test_entry_telemetry_computes_mid_and_slip():
    tel = build_entry_reference_telemetry(
        best_bid=100.0,
        best_ask=100.2,
        submitted_order_price=100.2,
        fill_price=100.2,
        decision_ts=1.0,
        fill_ts=1.4,
        is_maker=False,
    )
    assert tel["decision_midpoint"] == 100.1
    assert tel["mark_price"] == 100.1
    assert abs(tel["entry_slippage_from_midpoint_pct"] - (0.1 / 100.1)) < 1e-12
    assert tel["is_maker"] is False
    assert abs(tel["fill_latency_sec"] - 0.4) < 1e-9


def test_persist_sets_mark_price_without_touching_fill():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE paper_trades (trade_id TEXT, side TEXT, price REAL, mark_price REAL, slippage_pct_implied REAL, spread_pct_used REAL, context_snapshot_json TEXT, diagnostics_json TEXT)"
    )
    conn.execute("INSERT INTO paper_trades(trade_id, side, price) VALUES ('t1', 'BUY', 100.2)")
    tel = build_entry_reference_telemetry(
        best_bid=100.0,
        best_ask=100.2,
        submitted_order_price=100.2,
        fill_price=100.2,
        is_maker=False,
    )
    persist_entry_reference_row(conn, trade_id="t1", telemetry=tel, context_snapshot_json="{}", diagnostics_json="{}")
    row = conn.execute("SELECT price, mark_price FROM paper_trades WHERE trade_id='t1'").fetchone()
    assert row[0] == 100.2
    assert abs(row[1] - 100.1) < 1e-12
