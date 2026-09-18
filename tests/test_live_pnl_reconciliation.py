"""Live P&L must be presented separately from paper and from the legacy total.

The stored ``portfolio_engine_ledger.realized_pnl`` mixes simulated paper
profit with live results. These tests pin that the three figures stay separate,
that the primary displayed result is the exchange-reconciled live number, and
that no historical row is rewritten to achieve it.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from backend.services.live_pnl_reconciliation import (
    DAY_SYMBOLS,
    _epoch_ms,
    _qty_close,
    _reconcile_symbol,
    classify_unmatched_venue_fills,
    presentation_fields,
    read_recorded_live,
)

REPO = Path(__file__).resolve().parents[1]


def _seed(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL,
            pnl REAL, timestamp TEXT, order_id TEXT, exit_type TEXT,
            is_synthetic INTEGER DEFAULT 0
        );
        CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY, realized_pnl REAL);
        INSERT INTO portfolio_engine_ledger (id, realized_pnl) VALUES (1, 964.60);
        """
    )
    rows = [
        # live: one round trip, -2.00
        ("live", "BTC/USDT", "BUY", 0.001, 60000.0, None, "2026-09-10 10:00:00", None, None, 0),
        ("live", "BTC/USDT", "SELL", 0.001, 58000.0, -2.00, "2026-09-10 11:00:00", None, "NET_PROFIT", 0),
        # live dust write-off: excluded from the live result
        ("live", "XRP/USDT", "SELL", 0.5, 0.5, -0.25, "2026-09-10 12:00:00", None, "DUST_WRITEOFF", 0),
        # paper: +500, must never be counted as live
        ("paper", "ETH/USDT", "SELL", 0.1, 3000.0, 500.00, "2026-08-01 10:00:00", None, "NET_PROFIT", 0),
        # synthetic rows are excluded everywhere
        ("live", "SOL/USDT", "SELL", 1.0, 100.0, 999.0, "2026-09-10 13:00:00", None, "NET_PROFIT", 1),
    ]
    conn.executemany(
        """INSERT INTO paper_trades
           (mode, symbol, side, quantity, price, pnl, timestamp, order_id, exit_type, is_synthetic)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()


def test_recorded_live_separates_live_paper_and_legacy(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _seed(db)
    out = read_recorded_live(str(db))

    assert out["live_recorded_usd"] == pytest.approx(-2.00)
    assert out["paper_realized_usd"] == pytest.approx(500.00)
    assert out["live_dust_writeoff_usd"] == pytest.approx(-0.25)
    assert out["legacy_mixed_total_usd"] == pytest.approx(964.60)
    # Paper is not folded into the live figure.
    assert out["live_recorded_usd"] != out["legacy_mixed_total_usd"]


def test_synthetic_rows_are_excluded(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _seed(db)
    out = read_recorded_live(str(db))
    assert out["live_recorded_usd"] == pytest.approx(-2.00), "synthetic +999 row must not count"


def test_legacy_value_is_read_not_rewritten(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _seed(db)
    read_recorded_live(str(db))
    conn = sqlite3.connect(db)
    stored = conn.execute("SELECT realized_pnl FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    conn.close()
    assert stored == pytest.approx(964.60), "stored legacy value must survive untouched"
    assert rows == 5, "no row may be deleted"


def test_reconcile_counts_unmatched_quantity() -> None:
    recorded = {
        ("BTC/USDT", "BUY"): [
            {"qty": 0.001, "price": 60000.0, "ts": 1_000_000, "order_id": "", "exit_type": ""},
            {"qty": 0.002, "price": 60000.0, "ts": 2_000_000, "order_id": "", "exit_type": ""},
        ],
        ("BTC/USDT", "SELL"): [
            {"qty": 0.001, "price": 58000.0, "ts": 3_000_000, "order_id": "", "exit_type": "NET_PROFIT"},
        ],
    }
    venue = {
        "fills": [
            {"id": "1", "order": "o1", "side": "BUY", "qty": 0.001, "cost": 60.0, "ts": 1_000_050, "fee_cost": 0.06, "fee_ccy": "USDT"},
            {"id": "2", "order": "o2", "side": "SELL", "qty": 0.001, "cost": 58.0, "ts": 3_000_050, "fee_cost": 0.058, "fee_ccy": "USDT"},
            {"id": "3", "order": "o3", "side": "SELL", "qty": 0.009, "cost": 520.0, "ts": 4_000_000, "fee_cost": 0.52, "fee_ccy": "USDT"},
        ]
    }
    rec = _reconcile_symbol("BTC/USDT", recorded, venue)

    assert rec.matched_recorded_rows == 2, "the 0.001 buy and the 0.001 sell are fully covered"
    assert rec.unmatched_recorded_rows == 1, "the 0.002 buy has no venue quantity behind it"
    assert rec.unmatched_venue_fills == 1, "the 0.009 venue sell is never claimed by a row"
    assert rec.venue_gross_usd == pytest.approx(578.0 - 60.0)
    assert rec.venue_fee_quote_usd == pytest.approx(0.638)


def test_one_venue_fill_can_cover_several_fifo_rows() -> None:
    """The engine writes one row per FIFO lot; the venue reports one fill."""
    recorded = {
        ("SOL/USDT", "SELL"): [
            {"qty": 1.0, "price": 100.0, "ts": 10, "order_id": "", "exit_type": "NET_PROFIT"},
            {"qty": 2.0, "price": 100.0, "ts": 11, "order_id": "", "exit_type": "NET_PROFIT"},
            {"qty": 3.0, "price": 100.0, "ts": 12, "order_id": "", "exit_type": "NET_PROFIT"},
        ]
    }
    venue = {"fills": [{"id": "1", "order": "o1", "side": "SELL", "qty": 6.0, "cost": 600.0, "ts": 13, "fee_cost": 0.6, "fee_ccy": "USDT"}]}
    rec = _reconcile_symbol("SOL/USDT", recorded, venue)

    assert rec.matched_recorded_rows == 3, "all three lots are covered by the single fill"
    assert rec.unmatched_recorded_rows == 0
    assert rec.unmatched_venue_fills == 0, "the fill is fully consumed"
    assert rec.matched_fills == 1


def test_one_row_can_span_several_partial_fills() -> None:
    """A venue order split into partials must not read as unreconciled."""
    recorded = {("ETH/USDT", "BUY"): [{"qty": 0.06, "price": 3000.0, "ts": 100, "order_id": "", "exit_type": ""}]}
    venue = {
        "fills": [
            {"id": "1", "order": "o1", "side": "BUY", "qty": 0.02, "cost": 60.0, "ts": 101, "fee_cost": 0.06, "fee_ccy": "USDT"},
            {"id": "2", "order": "o1", "side": "BUY", "qty": 0.02, "cost": 60.0, "ts": 102, "fee_cost": 0.06, "fee_ccy": "USDT"},
            {"id": "3", "order": "o1", "side": "BUY", "qty": 0.02, "cost": 60.0, "ts": 103, "fee_cost": 0.06, "fee_ccy": "USDT"},
        ]
    }
    rec = _reconcile_symbol("ETH/USDT", recorded, venue)

    assert rec.matched_recorded_rows == 1
    assert rec.unmatched_recorded_rows == 0
    assert rec.matched_fills == 3, "every partial contributed"
    assert rec.unmatched_venue_fills == 0


def test_dust_writeoff_rows_are_not_expected_to_have_fills() -> None:
    recorded = {
        ("XRP/USDT", "SELL"): [
            {"qty": 0.5, "price": 0.5, "ts": 1_000, "order_id": "", "exit_type": "DUST_WRITEOFF"},
        ]
    }
    rec = _reconcile_symbol("XRP/USDT", recorded, {"fills": []})
    assert rec.unmatched_recorded_rows == 0, "write-offs have no exchange order by design"


def test_base_asset_fees_tracked_separately_from_quote() -> None:
    recorded: dict = {}
    venue = {
        "fills": [
            {"id": "1", "order": "o", "side": "BUY", "qty": 1.0, "cost": 100.0, "ts": 1, "fee_cost": 0.001, "fee_ccy": "SOL"},
            {"id": "2", "order": "o", "side": "SELL", "qty": 1.0, "cost": 101.0, "ts": 2, "fee_cost": 0.10, "fee_ccy": "USDT"},
        ]
    }
    rec = _reconcile_symbol("SOL/USDT", recorded, venue)
    assert rec.venue_fee_quote_usd == pytest.approx(0.10)
    assert rec.venue_fee_base == {"SOL": pytest.approx(0.001)}


def test_presentation_primary_is_reconciled_live_not_paper_or_legacy() -> None:
    recon = {
        "ok": True,
        "error": "",
        "live_reconciled_usd": -27.88,
        "live_recorded_usd": -9.47,
        "paper_realized_usd": 973.98,
        "legacy_mixed_total_usd": 964.60,
        "live_venue_fee_quote_usd": 2.11,
        "matched_fills": 120,
        "matched_recorded_rows": 118,
        "unmatched_recorded_rows": 3,
        "unmatched_venue_fills": 7,
        "window_start": "2026-09-01T00:00:00+00:00",
        "window_end": "2026-09-14T00:00:00+00:00",
        "qty_coverage_pct": 88.5,
        "notes": [],
    }
    p = presentation_fields(recon, is_live=True)

    assert p["primary_result_usd"] == pytest.approx(-27.88), "primary must be the reconciled live result"
    assert p["primary_result_is_exchange_reconciled"] is True
    assert "LIVE" in p["primary_result_label"]
    assert p["primary_result_usd"] != p["paper_realized_usd"]
    assert p["primary_result_usd"] != p["legacy_mixed_total_usd"]
    assert "historical" in p["legacy_mixed_total_label"].lower()
    assert "not live profit" in p["legacy_mixed_total_label"].lower()
    assert p["paper_is_not_live_performance"] is True
    assert p["matched_fills"] == 120
    assert p["unmatched_recorded_rows"] == 3
    assert p["unmatched_venue_fills"] == 7
    assert p["exchange_fees_quote_usd"] == pytest.approx(2.11)
    assert p["reconciliation_window_start"] and p["reconciliation_window_end"]


def test_presentation_falls_back_to_recorded_and_says_so_when_venue_unavailable() -> None:
    recon = {
        "ok": False,
        "error": "venue fetch failed: timeout",
        "live_recorded_usd": -9.47,
        "paper_realized_usd": 973.98,
        "legacy_mixed_total_usd": 964.60,
    }
    p = presentation_fields(recon, is_live=True)
    assert p["primary_result_is_exchange_reconciled"] is False
    assert p["primary_result_usd"] == pytest.approx(-9.47), "never substitutes paper when the venue is down"
    assert p["live_reconciled_usd"] is None
    assert p["reconciliation_error"]


def test_paper_is_never_the_primary_result_in_paper_mode() -> None:
    recon = {"ok": True, "error": "", "live_reconciled_usd": 0.0, "live_recorded_usd": 0.0, "paper_realized_usd": 973.98, "legacy_mixed_total_usd": 964.60}
    p = presentation_fields(recon, is_live=False)
    assert p["primary_result_usd"] == pytest.approx(0.0)
    assert p["paper_realized_usd"] == pytest.approx(973.98)
    assert p["primary_result_usd"] != p["paper_realized_usd"]


def test_epoch_parses_sqlite_timestamp_formats() -> None:
    assert _epoch_ms("2026-09-10 10:00:00") is not None
    assert _epoch_ms("2026-09-10 10:00:00.123456") is not None
    assert _epoch_ms("2026-09-10T10:00:00+00:00") is not None
    assert _epoch_ms("") is None
    assert _epoch_ms("not-a-date") is None


def test_qty_tolerance_is_representation_drift_only() -> None:
    assert _qty_close(0.001, 0.001 + 1e-15)
    assert not _qty_close(0.001, 0.002)


def test_unmatched_venue_fills_are_grouped() -> None:
    unmatched = [
        {"source": "venue_fill", "side": "BUY", "quantity": 1.0, "unmatched_quantity": 0.001, "dollar_discrepancy": 0.11, "exchange_order_id": "known-1"},
        {"source": "venue_fill", "side": "BUY", "quantity": 0.5, "unmatched_quantity": 0.2, "dollar_discrepancy": 22.0, "exchange_order_id": "known-2"},
        {"source": "venue_fill", "side": "BUY", "quantity": 0.001, "unmatched_quantity": 0.001, "dollar_discrepancy": 80.0, "exchange_order_id": "1837670272", "symbol": "BTC/USDT"},
        {"source": "venue_fill", "side": "SELL", "quantity": 0.02, "unmatched_quantity": 0.02, "dollar_discrepancy": 50.0, "exchange_order_id": "sell-1"},
        {"source": "local_recorded", "side": "SELL", "quantity": 1.0, "unmatched_quantity": 1.0, "dollar_discrepancy": 1.0},
    ]
    groups = classify_unmatched_venue_fills(unmatched, known_order_ids={"known-1", "known-2"})
    assert groups["summary"]["base_asset_fee_fragments"]["count"] == 1
    assert groups["summary"]["partial_fills_of_known_orders"]["count"] == 1
    assert groups["summary"]["full_buys_lacking_local"]["count"] == 1
    assert groups["summary"]["full_sells_lacking_local"]["count"] == 1
    assert groups["summary"]["genuine_unexplained"]["count"] == 0
    counted = sum(g["count"] for g in groups["summary"].values())
    assert counted == 4


def test_qty_coverage_does_not_double_count_one_fill() -> None:
    recorded = {("ETH/USDT", "BUY"): [{"qty": 0.02, "price": 2500.0, "ts": 1, "order_id": "1587098371", "exit_type": ""}]}
    venue = {
        "fills": [
            {"id": "a", "order": "1587098371", "side": "BUY", "qty": 0.01, "cost": 25.0, "ts": 1, "fee_cost": 0.0, "fee_ccy": "USDT"},
            {"id": "b", "order": "1587098371", "side": "BUY", "qty": 0.01, "cost": 25.0, "ts": 2, "fee_cost": 0.0, "fee_ccy": "USDT"},
        ]
    }
    rec = _reconcile_symbol("ETH/USDT", recorded, venue)
    assert rec.matched_recorded_rows == 1
    assert rec.unmatched_venue_fills == 0
    assert rec.qty_coverage_pct == 100.0


def test_day_symbol_universe_unchanged() -> None:
    assert DAY_SYMBOLS == ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")


def test_reconciliation_cache_serves_last_good_when_venue_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import backend.services.live_pnl_reconciliation as mod

    good = {"ok": True, "live_reconciled_usd": -27.88, "error": "", "notes": []}
    monkeypatch.setattr(mod, "_cache", {"at": mod.time.time(), "payload": dict(good)})

    async def failing(_db: str) -> mod.LivePnlReconciliation:
        return mod.LivePnlReconciliation(ok=False, error="venue fetch failed")

    monkeypatch.setattr(mod, "build_reconciliation", failing)
    out = asyncio.run(mod.get_reconciliation(str(tmp_path / "x.db"), force=True))
    assert out["stale"] is True, "stale result must be flagged, not silently shown as current"
    assert out["live_reconciled_usd"] == pytest.approx(-27.88)
    assert out["error"]


def test_cached_only_never_hits_the_venue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Latency-sensitive callers must not trigger four exchange round trips."""
    import backend.services.live_pnl_reconciliation as mod

    monkeypatch.setattr(mod, "_cache", {"at": 0.0, "payload": None})

    called = {"n": 0}

    async def should_not_run(_db: str) -> mod.LivePnlReconciliation:
        called["n"] += 1
        return mod.LivePnlReconciliation(ok=True)

    monkeypatch.setattr(mod, "build_reconciliation", should_not_run)
    out = asyncio.run(mod.get_reconciliation(str(tmp_path / "x.db"), cached_only=True))
    assert called["n"] == 0
    assert out["ok"] is False
    assert out["error"], "an unfetched reconciliation must not read as a real zero result"


def test_production_trailing_buy_timeout_is_900() -> None:
    """The 14400 value was an unproven strategy change and is reverted."""
    env = (REPO / "deploy" / "core_only_local.env").read_text()
    line = [ln for ln in env.splitlines() if ln.startswith("DAY_TRAILING_BUY_MAX_WAIT_SECONDS=")]
    assert line == ["DAY_TRAILING_BUY_MAX_WAIT_SECONDS=900"], line


def test_intent_age_follows_restored_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent staleness must track the configured wait, not a stale 4h value."""
    monkeypatch.setenv("DAY_TRAILING_BUY_MAX_WAIT_SECONDS", "900")
    import importlib

    from backend.config import day_entry_execution, day_setup_discovery

    importlib.reload(day_entry_execution)
    importlib.reload(day_setup_discovery)
    assert day_entry_execution.trailing_buy_max_wait_seconds() == 900
    assert day_setup_discovery.MAX_INTENT_AGE_SEC == 900
