import sqlite3

from backend.services.day_decision_label_contract import (
    TABLE_LABELS,
    normalize_label,
    write_outcome_label,
)
from backend.services.sqlite_large_table_retention import RETENTION_POLICIES


def test_label_provenance_and_offline_write(tmp_path):
    raw = {
        "decision_group_id": "daygrp_1",
        "symbol": "ETHUSDT",
        "provenance": "reconstructed",
        "markouts": {"15m": -3.0, "30m": -4.0, "1h": -5.0, "2h": 1.0, "4h": -8.0},
        "mfe_bps": 12.0,
        "mae_bps": -9.0,
        "cost_cover": False,
        "production_exit_net_bps": -12.0,
        "exit_reason": "DAY_4H_STRUCTURE_BREAK_EXIT",
        "regret_vs_hold_bps": -12.0,
    }
    row = normalize_label(raw)
    assert row["provenance"] == "reconstructed"
    db = str(tmp_path / "labels.db")
    write_outcome_label(db, raw)
    conn = sqlite3.connect(db)
    stored = conn.execute(f"SELECT provenance, mfe_bps, exit_reason FROM {TABLE_LABELS}").fetchone()
    conn.close()
    assert stored == ("reconstructed", 12.0, "DAY_4H_STRUCTURE_BREAK_EXIT")
    keep = {p.table: p.keep_days for p in RETENTION_POLICIES}
    assert keep["day_decision_outcome_labels"] == 90


def test_unknown_provenance_is_not_promoted():
    row = normalize_label({"decision_group_id": "g", "symbol": "HOLD", "provenance": "magic"})
    assert row["provenance"] == "unknown"
