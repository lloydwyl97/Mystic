"""Live, paper and dust P&L must stay separated."""

from __future__ import annotations

import sqlite3

from backend.services.live_fill_economics import sum_dust_adjustment_pnl, sum_realized_pnl_by_mode
from backend.services.live_pnl_reconciliation import presentation_fields


def _seed(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            side TEXT, pnl REAL, mode TEXT, is_synthetic INTEGER, exit_type TEXT, timestamp TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?)",
        [
            ("SELL", 5.83, "live", 0, "NET_PROFIT_EXIT", "2026-09-17"),
            ("SELL", 1104.81, "paper", 0, "NET_PROFIT_EXIT", "2026-08-17"),
            ("SELL", -56.78, "live", 0, "DUST_WRITEOFF", "2026-09-15"),
            ("SELL", -56.56, "live", 0, "DUST_WRITEOFF", "2026-09-15"),
        ],
    )
    conn.commit()
    conn.close()


def test_live_sum_excludes_dust_and_paper(tmp_path):
    db = str(tmp_path / "pnl.db")
    _seed(db)
    assert sum_realized_pnl_by_mode(db, mode="live") == 5.83
    assert sum_realized_pnl_by_mode(db, mode="paper") == 1104.81
    assert sum_dust_adjustment_pnl(db) == -113.34


def test_presentation_labels_dust_as_accounting_correction():
    fields = presentation_fields(
        {
            "ok": True,
            "live_reconciled_usd": 5.83,
            "live_recorded_usd": 5.83,
            "live_dust_writeoff_usd": -113.34,
            "paper_realized_usd": 1104.81,
            "legacy_mixed_total_usd": 961.55,
        },
        is_live=True,
    )
    assert fields["primary_result_usd"] == 5.83
    assert "ACCOUNTING CORRECTION" in fields["live_dust_writeoff_label"]
    assert fields["paper_is_not_live_performance"] is True
    assert "not live profit" in fields["legacy_mixed_total_label"]
    cash = presentation_fields(
        {
            "ok": True,
            "live_reconciled_usd": -34.92,
            "live_recorded_usd": -8.67,
            "live_dust_writeoff_usd": -144.07,
            "paper_realized_usd": 973.98,
            "legacy_mixed_total_usd": 965.31,
        },
        is_live=True,
        current_equity=223.92463088,
        forward_baseline_equity=228.06746265,
        reconciliation_adjustment_usd=0.94516971,
    )
    assert cash["lifetime_contributed_capital"] == "UNKNOWN"
    assert "not contributed principal" in cash["forward_baseline_label"]
    assert cash["live_economic_pnl_usd"] is None
    assert "not total-account P&L" in cash["live_economic_label"]
    assert abs(cash["forward_cash_change_usd"] - (223.92463088 - 228.06746265)) < 1e-9
    assert abs(cash["accounting_correction_total_usd"] - (-144.07 + 0.94516971)) < 1e-9
    assert cash["paper_realized_usd"] == 973.98
