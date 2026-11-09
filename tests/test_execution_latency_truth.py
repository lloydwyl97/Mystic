"""Execution latency shown to operators comes from stored live orders."""

from __future__ import annotations

import json
import sqlite3

from backend.services.protected_limit_execution import summarize_live_fill_latency


def _order(*, submit: float, ack_ms: float) -> str:
    ack_sec = submit + ack_ms / 1000.0
    return json.dumps(
        {
            "_mystic_latency": {
                "order_submit_timestamp": submit,
                "order_response_timestamp": ack_sec,
            },
            "timestamp": int(ack_sec * 1000),
            "info": {},
        }
    )


def test_latency_summary_uses_stored_orders_and_not_a_constant(tmp_path) -> None:
    db = tmp_path / "latency.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE live_exchange_fills (id INTEGER PRIMARY KEY, raw_json TEXT)")
    conn.executemany(
        "INSERT INTO live_exchange_fills (raw_json) VALUES (?)",
        [(_order(submit=1_700_000_000.0, ack_ms=ms),) for ms in (100.0, 200.0, 300.0, 400.0)],
    )
    conn.commit()
    conn.close()

    summary = summarize_live_fill_latency(str(db), limit=10)
    ack = summary["order_submit_to_ack_ms"]
    assert ack["samples"] == 4
    assert ack["avg"] == 250.0
    assert ack["p50"] == 300.0
    assert ack["max"] == 400.0
    assert summary["source"] == "live_exchange_fills"
    assert ack["avg"] != 5
    assert ack["p95"] != 15


def test_latency_summary_is_null_when_no_orders(tmp_path) -> None:
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE live_exchange_fills (id INTEGER PRIMARY KEY, raw_json TEXT)")
    conn.commit()
    conn.close()

    summary = summarize_live_fill_latency(str(db))
    assert summary["order_submit_to_ack_ms"]["samples"] == 0
    assert summary["order_submit_to_ack_ms"]["avg"] is None
    assert summary["order_submit_to_fill_ms"]["p95"] is None


def test_engine_latency_card_is_not_hardcoded() -> None:
    from pathlib import Path

    source = Path("backend/services/portfolio_engine.py").read_text()
    assert '{"avg": 5, "p95": 15, "samples": 0}' not in source
    assert "Paper mode - instant" not in source
