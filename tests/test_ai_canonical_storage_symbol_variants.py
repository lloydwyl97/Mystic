"""Regression test (found during p28 restart verification): DAY writes rows
into `ai_outcome_training_rows` using ccxt slash form ("BTC/USDT") while
several new subsystems (multi_horizon_ev, ai_multi_target_regressors,
feature_family_ablation, /walk-forward endpoint) look rows up using plain
exchange form ("BTCUSDT"). Without bidirectional symbol matching, those
lookups silently return zero rows forever, even once real data exists."""

from __future__ import annotations

import sqlite3

from backend.services.ai_canonical_storage import (
    _symbol_variants_for_lookup,
    ensure_ai_canonical_tables,
    read_recent_outcome_training_rows,
)


def test_symbol_variants_include_both_slash_and_plain_forms():
    assert set(_symbol_variants_for_lookup("BTCUSDT")) == {"BTC/USDT", "BTCUSDT"}
    assert set(_symbol_variants_for_lookup("BTC/USDT")) == {"BTC/USDT", "BTCUSDT"}


def test_symbol_variants_do_not_cross_contaminate_other_symbols():
    btc = set(_symbol_variants_for_lookup("BTCUSDT"))
    eth = set(_symbol_variants_for_lookup("ETHUSDT"))
    assert not (btc & eth)


def _insert_row(db_path: str, symbol: str, strategy_id: str = "day") -> None:
    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        row = {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "opened_at_utc": "2026-08-01T00:00:00Z",
            "closed_at_utc": "2026-08-01T00:05:00Z",
            "features_json": "[1.0, 2.0, 3.0]",
            "net_pnl_pct": 0.01,
            "ingested_at_utc": "2026-08-01T00:05:00Z",
        }
        use = {k: v for k, v in row.items() if k in cols}
        conn.execute(
            f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
            list(use.values()),
        )
        conn.commit()


def test_lookup_with_plain_symbol_finds_rows_stored_in_slash_form(tmp_path):
    db = tmp_path / "canon.db"
    ensure_ai_canonical_tables(str(db))
    _insert_row(str(db), "BTC/USDT", strategy_id="day")

    rows = read_recent_outcome_training_rows(symbol="BTCUSDT", strategy_id="day", db_path=str(db))
    assert len(rows) == 1


def test_lookup_with_slash_symbol_finds_rows_stored_in_plain_form(tmp_path):
    db = tmp_path / "canon2.db"
    ensure_ai_canonical_tables(str(db))
    _insert_row(str(db), "BTCUSDT", strategy_id="scalp")

    rows = read_recent_outcome_training_rows(symbol="BTC/USDT", strategy_id="scalp", db_path=str(db))
    assert len(rows) == 1


def test_lookup_does_not_leak_across_symbols(tmp_path):
    db = tmp_path / "canon3.db"
    ensure_ai_canonical_tables(str(db))
    _insert_row(str(db), "BTC/USDT", strategy_id="day")
    _insert_row(str(db), "ETH/USDT", strategy_id="day")

    rows = read_recent_outcome_training_rows(symbol="BTCUSDT", strategy_id="day", db_path=str(db))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC/USDT"
