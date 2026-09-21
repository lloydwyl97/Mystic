"""Reconcile mirrors must not create a second economic close."""

from __future__ import annotations

import sqlite3

from backend.services.live_close_integrity import CLASS_RECONCILE_MIRROR, claim_economic_close, classify_paper_sell_row
from backend.services.live_recovered_close_writer import RecoveredCloseFill, persist_recovered_close


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            trade_id TEXT, paper_run_id TEXT, mode TEXT, symbol TEXT, side TEXT,
            quantity REAL, price REAL, entry_price REAL, pnl REAL, pnl_pct REAL,
            remaining_position REAL, hold_time_seconds INTEGER, fees_paid REAL,
            slippage_cost REAL, exit_type TEXT, timestamp TEXT, status TEXT,
            explainability_json TEXT, diagnostics_json TEXT, sleeve TEXT,
            exit_reason TEXT, entry_timestamp TEXT, decision_id TEXT,
            strategy_id TEXT, confidence REAL, order_id TEXT
        );
        CREATE TABLE portfolio_engine_audit (
            id INTEGER PRIMARY KEY, ts TEXT, action TEXT, symbol TEXT, qty REAL,
            price REAL, fees REAL, slippage REAL, decision_id TEXT, trade_id TEXT,
            ranked_candidates_json TEXT, pre_ledger_json TEXT, post_ledger_json TEXT,
            pre_positions_digest TEXT, post_positions_digest TEXT,
            invariant_ok INTEGER, invariant_diff REAL, entry_reason TEXT,
            exit_reason TEXT, sleeve TEXT
        );
        CREATE TABLE position_close_ledger (
            id INTEGER PRIMARY KEY, detail TEXT
        );
        CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, extra_json TEXT, close_reason TEXT);
        CREATE TABLE trade_performance (id INTEGER PRIMARY KEY, trade_id INTEGER, side TEXT);
        """
    )
    conn.execute(
        """INSERT INTO paper_trades (trade_id, symbol, side, quantity, price, timestamp, mode, order_id, status, exit_type)
           VALUES ('buy1','SOL/USDT','BUY',0.1,114.0,'2026-09-18T20:00:00+00:00','live','911073800','executed','')"""
    )
    conn.commit()


def _fill(**kw):
    base = {
        "buy_trade_id": "buy1",
        "symbol": "SOL/USDT",
        "quantity": 0.1,
        "entry_price": 114.0,
        "exit_price": 113.8,
        "exchange_sell_order_id": "911073809",
        "closed_at_iso": "2026-09-18T20:04:15+00:00",
        "closed_at_epoch": 1758225855.0,
        "source": "test",
        "fill_recovered": True,
        "realized_profit_usd": -0.02,
        "fee_usd": 0.01,
        "venue_trade_ids": "2630100123",
    }
    base.update(kw)
    return RecoveredCloseFill(**base)


def test_reconcile_skips_when_venue_sell_already_exists(tmp_path):
    db = str(tmp_path / "rec.db")
    conn = sqlite3.connect(db)
    _schema(conn)
    conn.execute(
        """INSERT INTO paper_trades
           (trade_id,symbol,side,quantity,price,timestamp,mode,order_id,status,exit_type,pnl)
           VALUES ('mystic_sell_SOL_1','SOL/USDT','SELL',0.1,113.8,'2026-09-18T20:04:10+00:00','live','911073809','executed','TRAILING_STOP_EXIT',-0.02)"""
    )
    conn.commit()
    conn.close()
    out = persist_recovered_close(_fill(), db_path=db, write_trade_performance=False)
    assert out["existing"].get("venue_order_already_attributed") or out["existing"].get("paper_trades_sell")
    assert "paper_trades_sell" not in out.get("created", {})
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
    conn.close()
    assert n == 1


def test_reconcile_row_is_not_a_strategy_class():
    assert (
        classify_paper_sell_row(
            {
                "exit_type": "EXCHANGE_RECONCILE_CLOSE",
                "exit_reason": "EXCHANGE_RECONCILE_CLOSE",
                "mode": "live",
                "order_id": "911073809",
            }
        )
        == CLASS_RECONCILE_MIRROR
    )


def test_reconcile_claim_blocks_second_economic_close(tmp_path):
    db = str(tmp_path / "claim.db")
    conn = sqlite3.connect(db)
    _schema(conn)
    conn.close()
    persist_recovered_close(_fill(), db_path=db, write_trade_performance=False)
    assert claim_economic_close(db, "911073809", "SELL") is False


def test_reconcile_retry_cannot_create_duplicate_economic_sell(tmp_path):
    db = str(tmp_path / "retry.db")
    conn = sqlite3.connect(db)
    _schema(conn)
    conn.close()
    first = persist_recovered_close(_fill(), db_path=db, write_trade_performance=True)
    second = persist_recovered_close(_fill(), db_path=db, write_trade_performance=True)
    assert first.get("economic_sell_written") is False
    assert second.get("economic_sell_written") is False
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
    conn.close()
    assert n == 0


def test_reconcile_rejects_trade_id_masquerading_as_order_id(tmp_path):
    db = str(tmp_path / "fake.db")
    conn = sqlite3.connect(db)
    _schema(conn)
    conn.close()
    out = persist_recovered_close(_fill(exchange_sell_order_id="2630592", venue_trade_ids="2630592"), db_path=db)
    assert out.get("economic_sell_written") is False
    assert "missing_real_venue_sell_identity" in out.get("errors", [])
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
    conn.close()
    assert n == 0
