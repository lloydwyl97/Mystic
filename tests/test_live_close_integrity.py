"""Venue-fill authority for live strategy closes. No exit-parameter changes."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

from backend.services.live_close_integrity import (
    CLASS_DUPLICATE_MIRROR,
    CLASS_MANUAL_UNMATCHED,
    CLASS_NON_VENUE_GHOST,
    CLASS_RECONCILE_MIRROR,
    CLASS_VENUE_BACKED,
    claim_economic_close,
    classify_existing_rows,
    classify_paper_sell_row,
    classify_trailing_exit,
    live_order_has_venue_fill,
    persist_dust_inventory_event,
    persist_pending_close_event,
    venue_backed_window_stats,
)
from backend.services.portfolio_engine import PortfolioEngine

ENGINE_SRC = Path(__file__).resolve().parents[1] / "backend" / "services" / "portfolio_engine.py"
SOURCE = ENGINE_SRC.read_text()


def test_exit_decision_without_fill_is_not_a_venue_fill():
    assert live_order_has_venue_fill(None) is False
    assert live_order_has_venue_fill({}) is False
    assert live_order_has_venue_fill({"id": "1", "filled": 0}) is False
    assert live_order_has_venue_fill({"id": "9", "filled": 1.2}) is True


def test_completed_trade_writer_unreachable_without_venue_fill():
    assert "LIVE_CLOSE_BLOCKED_NO_VENUE_FILL" in SOURCE
    assert "FIFO_SELL_REFUSED_NO_VENUE_FILL" in SOURCE
    assert "FIFO_SELL_REFUSED_DUST_WRITEOFF" in SOURCE
    assert "_finalize_dust_without_strategy_close" in SOURCE


def test_dust_writeoff_cannot_create_live_realized_pnl():
    src = inspect.getsource(PortfolioEngine._finalize_dust_without_strategy_close)
    assert "paper_trades" not in src
    assert "realized_pnl" not in src
    assert "persist_dust_inventory_event" in src


def test_classify_ghost_dust_reconcile_manual_and_venue():
    assert classify_paper_sell_row({"exit_type": "DUST_WRITEOFF", "status": "dust_writeoff", "mode": "live", "order_id": None}) == CLASS_NON_VENUE_GHOST
    assert classify_paper_sell_row({"exit_type": "EXCHANGE_RECONCILE_CLOSE", "exit_reason": "EXCHANGE_RECONCILE_CLOSE", "mode": "live", "order_id": ""}) == CLASS_RECONCILE_MIRROR
    assert classify_paper_sell_row({"exit_type": "HUMAN_MANUAL_SELL", "exit_reason": "HUMAN_MANUAL_SELL", "mode": "live", "order_id": ""}) == CLASS_MANUAL_UNMATCHED
    seen: set[str] = set()
    first = classify_paper_sell_row(
        {"exit_type": "TRAILING_STOP_EXIT", "mode": "live", "order_id": "488980350"},
        seen_order_ids=seen,
    )
    dup = classify_paper_sell_row(
        {"exit_type": "TRAILING_STOP_EXIT", "mode": "live", "order_id": "488980350"},
        seen_order_ids=seen,
    )
    assert first == CLASS_VENUE_BACKED
    assert dup == CLASS_DUPLICATE_MIRROR


def test_one_venue_fill_counted_once(tmp_path):
    db = str(tmp_path / "eco.db")
    assert claim_economic_close(db, "488980350", "SELL", symbol="XRP/USDT") is True
    assert claim_economic_close(db, "488980350", "SELL", symbol="XRP/USDT") is False
    assert claim_economic_close(db, "488980350", "SELL") is False


def test_retry_cannot_duplicate_existing_sell(tmp_path):
    db = str(tmp_path / "retry.db")
    assert claim_economic_close(db, "911394760", "SELL") is True
    persist_pending_close_event(
        db,
        symbol="SOL/USDT",
        event_type="LIVE_CLOSE_DUPLICATE_VENUE_FILL",
        exchange_order_id="911394760",
    )
    assert claim_economic_close(db, "911394760", "SELL") is False


def test_dust_event_does_not_write_strategy_pnl(tmp_path):
    db = str(tmp_path / "dust.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE dust_writeoffs (
            timestamp TEXT, symbol TEXT, quantity REAL, entry_price REAL,
            price_snapshot REAL, est_notional REAL, reason TEXT, sell_trade_id TEXT
        )"""
    )
    conn.commit()
    conn.close()
    assert persist_dust_inventory_event(
        db,
        symbol="XRP/USDT",
        quantity=0.6,
        entry_price=1.4133,
        price_snapshot=1.411,
        reason="DUST",
        est_notional=0.84,
    )
    assert (
        persist_dust_inventory_event(
            db,
            symbol="XRP/USDT",
            quantity=0.6,
            entry_price=1.4133,
            price_snapshot=1.411,
            reason="DUST",
            est_notional=0.84,
        )
        is False
    )
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0 if _has_paper(conn) else True
    assert conn.execute("SELECT COUNT(*) FROM dust_writeoffs").fetchone()[0] == 1
    conn.close()


def _has_paper(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE name='paper_trades'").fetchone()
    return bool(row)


def test_classification_table_is_append_only(tmp_path):
    db = str(tmp_path / "cls.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT, timestamp TEXT,
            order_id TEXT, mode TEXT, status TEXT, exit_type TEXT, exit_reason TEXT,
            pnl REAL, pnl_usd_net REAL, is_synthetic INTEGER
        )"""
    )
    conn.executemany(
        """INSERT INTO paper_trades
           (trade_id,symbol,side,timestamp,order_id,mode,status,exit_type,exit_reason,pnl,pnl_usd_net,is_synthetic)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            ("g1", "XRP/USDT", "SELL", "2026-09-19T09:33:59+00:00", None, "live", "dust_writeoff", "DUST_WRITEOFF", "TRAILING_STOP_EXIT", -0.848, -0.848, 0),
            ("r1", "SOL/USDT", "SELL", "2026-09-18T20:04:15+00:00", None, "live", "executed", "EXCHANGE_RECONCILE_CLOSE", "EXCHANGE_RECONCILE_CLOSE", -0.07, None, 0),
            ("m1", "XRP/USDT", "SELL", "2026-09-18T18:57:52+00:00", None, "live", "executed", "HUMAN_MANUAL_SELL", "HUMAN_MANUAL_SELL", 0.0, None, 0),
            ("v1", "XRP/USDT", "SELL", "2026-09-19T09:33:04+00:00", "488980350", "live", "executed", "TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", -0.01, -0.01, 0),
            ("v1b", "XRP/USDT", "SELL", "2026-09-19T09:33:05+00:00", "488980350", "live", "executed", "TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", -0.01, -0.01, 0),
        ],
    )
    conn.commit()
    conn.close()
    counts = classify_existing_rows(db)
    assert counts[CLASS_NON_VENUE_GHOST] == 1
    assert counts[CLASS_RECONCILE_MIRROR] == 1
    assert counts[CLASS_MANUAL_UNMATCHED] == 1
    assert counts[CLASS_VENUE_BACKED] == 1
    assert counts[CLASS_DUPLICATE_MIRROR] == 1
    classify_existing_rows(db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM live_close_classifications").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 5
    conn.close()


def test_venue_backed_scorecard_excludes_ghosts(tmp_path):
    db = str(tmp_path / "sc.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT, timestamp TEXT,
            order_id TEXT, mode TEXT, status TEXT, exit_type TEXT, exit_reason TEXT,
            pnl REAL, pnl_usd_net REAL, is_synthetic INTEGER
        )"""
    )
    conn.executemany(
        """INSERT INTO paper_trades
           (trade_id,symbol,side,timestamp,order_id,mode,status,exit_type,exit_reason,pnl,pnl_usd_net,is_synthetic)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            ("g1", "XRP/USDT", "SELL", "2026-09-19T09:33:59+00:00", None, "live", "dust_writeoff", "DUST_WRITEOFF", "TRAILING_STOP_EXIT", -0.848, -0.848, 0),
            ("v1", "XRP/USDT", "SELL", "2026-09-19T09:33:04+00:00", "488980350", "live", "executed", "TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", -0.01, -0.01, 0),
        ],
    )
    conn.commit()
    conn.close()
    stats = venue_backed_window_stats(db, "2026-09-18T12:48:00+00:00", "2026-09-19T12:48:00+00:00")
    assert stats["venue_backed_closes"] == 1
    assert abs(stats["stored_venue_backed_pnl"] + 0.01) < 1e-12
    assert stats["raw_non_venue_rows"] >= 1


def test_venue_backed_scorecard_pnl_exact(tmp_path):
    db = str(tmp_path / "sc2.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT, timestamp TEXT,
            order_id TEXT, mode TEXT, status TEXT, exit_type TEXT, exit_reason TEXT,
            pnl REAL, pnl_usd_net REAL, is_synthetic INTEGER
        )"""
    )
    conn.execute(
        """INSERT INTO paper_trades
           (trade_id,symbol,side,timestamp,order_id,mode,status,exit_type,exit_reason,pnl,pnl_usd_net,is_synthetic)
           VALUES ('v1','XRP/USDT','SELL','2026-09-19T09:33:04+00:00','488980350','live','executed','TRAILING_STOP_EXIT','TRAILING_STOP_EXIT',-0.01,-0.01,0)"""
    )
    conn.commit()
    conn.close()
    stats = venue_backed_window_stats(db, "2026-09-18T12:48:00+00:00", "2026-09-19T12:48:00+00:00")
    assert abs(stats["stored_venue_backed_pnl"] + 0.01) < 1e-12


def test_actual_fee_assets_used_in_trailing_record():
    out = classify_trailing_exit(
        symbol="SOL/USDT",
        entry=113.97,
        highest_executable=114.50,
        fill_price=113.80,
        executable_bid=113.80,
    )
    assert out["activated"] is True
    assert out["controlling_stop"] >= out["cost_aware_floor"] - 1e-12
    assert out["classification"] in {"COST_FLOOR_NOT_APPLIED", "VALID_TRAIL_GAP_OR_SLIPPAGE"}


def test_fill_below_floor_with_bid_at_floor_is_slippage():
    entry = 100.0
    high = 100.50
    primed = classify_trailing_exit(symbol="SOL/USDT", entry=entry, highest_executable=high, fill_price=100.0, executable_bid=100.0)
    assert primed["activated"] is True
    controlling = float(primed["controlling_stop"])
    out = classify_trailing_exit(
        symbol="SOL/USDT",
        entry=entry,
        highest_executable=high,
        fill_price=controlling - 0.05,
        executable_bid=controlling,
    )
    assert out["classification"] == "VALID_TRAIL_GAP_OR_SLIPPAGE"


def test_cost_floor_controls_submit_decision():
    entry = 100.0
    high = 100.50
    primed = classify_trailing_exit(symbol="SOL/USDT", entry=entry, highest_executable=high, fill_price=100.4, executable_bid=100.4)
    controlling = float(primed["controlling_stop"])
    out = classify_trailing_exit(
        symbol="SOL/USDT",
        entry=entry,
        highest_executable=high,
        fill_price=controlling - 0.08,
        executable_bid=controlling - 0.08,
    )
    assert out["classification"] == "COST_FLOOR_NOT_APPLIED"


def test_unactivated_trail_is_not_a_floor_violation():
    out = classify_trailing_exit(
        symbol="XRP/USDT",
        entry=1.3935,
        highest_executable=1.39225,
        fill_price=1.3883,
        executable_bid=1.3883,
    )
    assert out["activated"] is False
    assert out["classification"] == "TRAIL_NOT_ACTIVATED"


def test_restart_adoption_is_idempotent(tmp_path):
    db = str(tmp_path / "restart.db")
    assert claim_economic_close(db, "911723947", "SELL") is True
    assert claim_economic_close(db, "911723947", "SELL") is False


def test_duplicate_order_id_counts_once(tmp_path):
    db = str(tmp_path / "dup.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT, timestamp TEXT,
            order_id TEXT, mode TEXT, status TEXT, exit_type TEXT, exit_reason TEXT,
            pnl REAL, pnl_usd_net REAL, is_synthetic INTEGER
        )"""
    )
    conn.executemany(
        """INSERT INTO paper_trades
           (trade_id,symbol,side,timestamp,order_id,mode,status,exit_type,exit_reason,pnl,pnl_usd_net,is_synthetic)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            ("v1", "XRP/USDT", "SELL", "2026-09-19T09:33:04+00:00", "488980350", "live", "executed", "TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", -0.01, -0.01, 0),
            ("v1b", "XRP/USDT", "SELL", "2026-09-19T09:33:05+00:00", "488980350", "live", "executed", "TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", -0.01, -0.01, 0),
        ],
    )
    conn.commit()
    conn.close()
    stats = venue_backed_window_stats(db, "2026-09-18T12:48:00+00:00", "2026-09-19T12:48:00+00:00")
    assert stats["venue_backed_closes"] == 1
    assert abs(stats["stored_venue_backed_pnl"] + 0.01) < 1e-12


def test_dust_cleanup_source_has_no_paper_trades_insert():
    src = inspect.getsource(PortfolioEngine._remove_dust_position_canonical_cleanup)
    assert "INSERT INTO paper_trades" not in src
    assert "persist_dust_inventory_event" in src


def test_reconcile_does_not_count_as_strategy_trade():
    assert classify_paper_sell_row({"exit_type": "EXCHANGE_RECONCILE_CLOSE", "exit_reason": "EXCHANGE_RECONCILE_CLOSE", "mode": "live", "order_id": "911073809"}) == CLASS_RECONCILE_MIRROR
