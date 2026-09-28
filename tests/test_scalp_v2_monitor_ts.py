"""Unit tests for scalp_v2_checkpoint_monitor internals.

Covers the timestamp normalisation fix (_normalise_ts_for_feature_ohlcv) and
the partial CF computation path (COMPUTED_TIME_STOP_ONLY) that was introduced
to fix the "all 13 givebacks UNAVAILABLE" bug.
"""

from __future__ import annotations

import sqlite3

import pytest

from scripts.scalp_v2_checkpoint_monitor import (
    _PROD_MAX_HOLD_SEC,
    _compute_cf,
    _load_post_exit_1m_bars,
    _normalise_ts_for_feature_ohlcv,
)

# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------


def test_normalise_iso_with_tz_offset():
    ts = "2026-09-23T01:36:06.830174+00:00"
    out = _normalise_ts_for_feature_ohlcv(ts)
    assert out == "2026-09-23 01:36:06"


def test_normalise_iso_z_suffix():
    ts = "2026-09-23T14:22:00.000000Z"
    out = _normalise_ts_for_feature_ohlcv(ts)
    assert out == "2026-09-23 14:22:00"


def test_normalise_already_space_format():
    ts = "2026-09-23 01:36:06.000000"
    out = _normalise_ts_for_feature_ohlcv(ts)
    assert out.startswith("2026-09-23 01:36:06")


# ---------------------------------------------------------------------------
# _load_post_exit_1m_bars: symbol format and timestamp matching
# ---------------------------------------------------------------------------


def _make_ohlcv_db(bars: list[tuple]) -> sqlite3.Connection:
    """Create an in-memory feature_ohlcv DB with 1m bars."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE feature_ohlcv (
            symbol TEXT, interval TEXT, ts TEXT,
            open REAL, high REAL, low REAL, close REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)",
        bars,
    )
    conn.commit()
    return conn


def test_load_bars_slash_symbol_mapped_to_hyphen():
    """BTC/USDT (paper_trades format) must resolve to BTC-USDT (feature_ohlcv format)."""
    conn = _make_ohlcv_db(
        [
            ("BTC-USDT", "1m", "2026-09-23 01:37:00.000000", 100.0, 101.0, 99.5, 100.5),
            ("BTC-USDT", "1m", "2026-09-23 01:38:00.000000", 100.5, 102.0, 100.0, 101.0),
        ]
    )
    bars = _load_post_exit_1m_bars(conn, "BTC/USDT", "2026-09-23T01:36:06.830174+00:00", limit=10)
    assert len(bars) == 2
    assert bars[0]["o"] == 100.0
    assert bars[1]["c"] == 101.0


def test_load_bars_returns_empty_when_all_bars_before_exit():
    """When all bars precede the exit timestamp, return empty list."""
    conn = _make_ohlcv_db(
        [
            ("BTC-USDT", "1m", "2026-09-23 01:30:00.000000", 99.0, 100.0, 98.5, 99.5),
            ("BTC-USDT", "1m", "2026-09-23 01:31:00.000000", 99.5, 100.5, 99.0, 100.0),
        ]
    )
    bars = _load_post_exit_1m_bars(conn, "BTC/USDT", "2026-09-23T01:36:06.000000+00:00", limit=10)
    assert bars == []


# ---------------------------------------------------------------------------
# _compute_cf: COMPUTED_TIME_STOP_ONLY when stop/target NULL but bars present
# ---------------------------------------------------------------------------


def _fake_trade(exit_ts: str, entry_ts: str, qty: float = 0.01, ep: float = 100.0, hold_sec: float = 600.0) -> dict:
    return {
        "trade_id": "t1",
        "symbol": "BTC/USDT",
        "quantity": qty,
        "entry_price": ep,
        "exit_timestamp": exit_ts,
        "entry_timestamp": entry_ts,
        "hold_time_seconds": hold_sec,
        "pnl_usd_net": -0.50,
        "stop_price": None,
        "take_profit_price": None,
        "exit_type": "GIVEBACK_EXIT",
        "fees_paid": 0.01,
        "slippage_cost": 0.005,
    }


def test_compute_cf_time_stop_only_when_stop_and_target_null():
    """When stop/target are NULL but bars exist, return COMPUTED_TIME_STOP_ONLY."""
    bars = [
        {"ts": "2026-09-23 01:37:00.000000", "o": 100.0, "h": 100.5, "l": 99.8, "c": 100.2},
        {"ts": "2026-09-23 01:38:00.000000", "o": 100.2, "h": 100.8, "l": 100.0, "c": 100.6},
    ]
    trade = _fake_trade(
        exit_ts="2026-09-23T01:36:06.000000+00:00",
        entry_ts="2026-09-23T01:00:00.000000+00:00",
    )
    result = _compute_cf(trade, bars)
    assert result["cf_status"] == "COMPUTED_TIME_STOP_ONLY"
    assert result["cf_missing_data_reason"] == "stop_and_target_unavailable_time_stop_only"
    assert result["cf_first_exit"] == "TIME_STOP_EXIT"
    assert result["cf_exit_price"] is not None
    assert result["cf_pnl_usd_net"] is not None


def test_compute_cf_unavailable_when_no_bars():
    """When no bars are available, CF must be UNAVAILABLE regardless of stop/target."""
    trade = _fake_trade(
        exit_ts="2026-09-23T01:36:06.000000+00:00",
        entry_ts="2026-09-23T01:00:00.000000+00:00",
    )
    result = _compute_cf(trade, [])
    assert result["cf_status"] == "UNAVAILABLE"


def test_compute_cf_computed_when_stop_present():
    """When stop is present and bar hits it, return COMPUTED status."""
    bars = [
        {"ts": "2026-09-23 01:37:00.000000", "o": 100.0, "h": 100.5, "l": 98.0, "c": 98.5},
    ]
    trade = {**_fake_trade("2026-09-23T01:36:06+00:00", "2026-09-23T01:00:00+00:00"), "stop_price": 99.0}
    result = _compute_cf(trade, bars)
    assert result["cf_status"] == "COMPUTED"
    assert result["cf_first_exit"] == "STOP_LOSS"
    assert result["cf_exit_price"] == 99.0


def test_compute_cf_remaining_hold_sec_correct():
    """cf_remaining_hold_sec must equal prod max minus actual hold time."""
    # Entry 10 minutes before exit → hold = 600s
    bars = [{"ts": "2026-09-23 01:11:00.000000", "o": 100.0, "h": 100.5, "l": 99.8, "c": 100.2}]
    trade = _fake_trade(
        exit_ts="2026-09-23T01:10:00.000000+00:00",
        entry_ts="2026-09-23T01:00:00.000000+00:00",
    )
    result = _compute_cf(trade, bars)
    assert result["cf_max_hold_sec"] == float(_PROD_MAX_HOLD_SEC)
    # Actual hold = 600s; remaining = PROD_MAX_HOLD_SEC - 600
    expected_remaining = _PROD_MAX_HOLD_SEC - 600.0
    assert result["cf_remaining_hold_sec"] == pytest.approx(expected_remaining, abs=2.0)


# ---------------------------------------------------------------------------
# Checkpoint report generation
# ---------------------------------------------------------------------------


def _report_db() -> sqlite3.Connection:
    from scripts.scalp_v2_checkpoint_monitor import _ensure_schema, _migrate_cf_schema

    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    _migrate_cf_schema(conn)
    conn.execute("CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY, principal REAL, cash_balance REAL, realized_pnl REAL, total_equity REAL)")
    conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, 228.07, 121.36, 0.0, 256.21)")
    return conn


def _closed(symbol: str, pnl: float, exit_reason: str) -> dict:
    return {
        "symbol": symbol,
        "pnl_usd_net": pnl,
        "fees_paid": 0.01,
        "hold_time_seconds": 600,
        "timestamp": "2026-09-27T23:00:00+00:00",
        "exit_reason": exit_reason,
        "scalp_opportunity_id": "opp-1",
    }


def test_generate_report_reads_giveback_rows_by_column_name():
    """Regression: columns came from PRAGMA table_info index (cid), so every
    giveback row was keyed 0..N and report generation crashed with
    KeyError: 'trade_id' at CHECKPOINT_100_REACHED (Ocean 2026-09-27)."""
    from scripts.scalp_v2_checkpoint_monitor import _generate_report

    conn = _report_db()
    conn.execute(
        "INSERT INTO scalp_v2_giveback_cf (trade_id, symbol, pnl_usd_net, cf_status, cf_pnl_usd_net, cf_first_exit, cf_saved_or_cost) VALUES (?,?,?,?,?,?,?)",
        ("mystic_sell_BTC/USDT_1", "BTC/USDT", -0.05, "COMPUTED", 0.02, "NET_PROFIT", -0.07),
    )
    trades = [_closed("BTC/USDT", 0.10, "NET_PROFIT_EXIT"), _closed("ETH/USDT", -0.05, "GIVEBACK_EXIT")]

    report = _generate_report(trades, conn, ":memory:")

    cf = report["giveback_counterfactual"]
    assert cf["cf_computed"] == 1
    assert cf["computed_actual_total_pnl"] == pytest.approx(-0.05)
    assert cf["rows"][0]["trade_id"] == "mystic_sell_BTC/USDT_1"
    assert cf["rows"][0]["symbol"] == "BTC/USDT"


def test_generate_report_marks_missing_trade_id_unavailable():
    from scripts.scalp_v2_checkpoint_monitor import _generate_report

    conn = _report_db()
    conn.execute("INSERT INTO scalp_v2_giveback_cf (trade_id, symbol, pnl_usd_net, cf_status) VALUES (NULL, 'XRP/USDT', -0.01, 'UNAVAILABLE')")

    report = _generate_report([_closed("XRP/USDT", -0.01, "GIVEBACK_EXIT")], conn, ":memory:")

    assert report["giveback_counterfactual"]["rows"][0]["trade_id"] == "UNAVAILABLE"
