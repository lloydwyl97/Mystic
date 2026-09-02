"""Observability writes do not change DAY decisions or order lifecycle labels."""

from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace

from backend.services.day_decision_observability import (
    TABLE_CANDIDATES,
    TABLE_GROUPS,
    build_group_contract,
    classify_terminal_fill,
    observability_enabled,
    record_day_ranking_group,
    runtime_account_execution_mode,
    update_day_decision_lifecycle,
)
from backend.services.day_direct_path_ev_authority import select_action
from backend.services.day_gate_telemetry import record_day_decision


def _decision(**overrides):
    base = select_action(
        {
            "btc_path_ev": 0.0001,
            "eth_path_ev": 0.0008,
            "sol_path_ev": 0.0002,
            "xrp_path_ev": 0.0001,
            "path_net_status": "predicted",
            "path_net_model_id": "day_path_net_v1",
        },
        old_rank_nominee="BTCUSDT",
        old_rank_score=9.0,
    )
    base.update(overrides)
    return base


def test_runtime_mode_live(monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    assert runtime_account_execution_mode() == "live"


def test_runtime_mode_paper(monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "paper")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    assert runtime_account_execution_mode() == "paper"


def test_record_day_decision_stamps_live_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    db = str(tmp_path / "obs.db")
    record_day_decision(db, decision_id="d1", symbol="BTC/USDT", aw_valid=True, final_decision="execute")
    conn = sqlite3.connect(db)
    mode = conn.execute("SELECT mode FROM day_decision_records WHERE decision_id='d1'").fetchone()[0]
    conn.close()
    assert mode == "live"


def test_record_day_decision_stamps_paper_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "paper")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    db = str(tmp_path / "obs.db")
    record_day_decision(db, decision_id="d2", symbol="ETH/USDT", aw_valid=True, final_decision="reject")
    conn = sqlite3.connect(db)
    mode = conn.execute("SELECT mode FROM day_decision_records WHERE decision_id='d2'").fetchone()[0]
    conn.close()
    assert mode == "paper"


def test_builder_does_not_mutate_decision():
    dec = _decision()
    before = json.dumps(dec, sort_keys=True, default=str)
    cands = [
        SimpleNamespace(symbol="ETHUSDT", decision_data={"prob_buy": 0.6, "final_selection_score": 0.0008, "intelligence_rank_delta": 0.01}),
        SimpleNamespace(symbol="BTCUSDT", decision_data={"prob_buy": 0.4, "final_selection_score": 0.0001}),
    ]
    contract = build_group_contract(decision=dec, candidates=cands, bar_timestamp=100)
    after = json.dumps(dec, sort_keys=True, default=str)
    assert before == after
    assert contract["decision_group_id"] == "daygrp_100"
    assert contract["selected_action"] == "BUY_ETHUSDT"
    assert contract["selected_symbol"] == "ETHUSDT"
    assert {r["symbol"] for r in contract["candidates"]} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"}
    eth = next(r for r in contract["candidates"] if r["symbol"] == "ETHUSDT")
    assert eth["p_buy"] == 0.6
    assert eth["rank_deltas"]["intelligence_rank_delta"] == 0.01
    sol = next(r for r in contract["candidates"] if r["symbol"] == "SOLUSDT")
    assert sol["eligible"] is False
    assert sol["exclusion_reason"] == "NO_SCORED_CANDIDATE"


def test_record_and_lifecycle_update(tmp_path, monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "true")
    db = str(tmp_path / "obs.db")
    dec = _decision()
    gid = record_day_ranking_group(db, decision=dec, bar_timestamp=200)
    assert gid == "daygrp_200"
    update_day_decision_lifecycle(
        db,
        decision_group_id=gid,
        execute_authorized=True,
        lifecycle_state="filled",
        order_id="ex-9",
        fill_trade_id="fill-3",
        maker_taker="taker",
        commission=0.02,
        commission_asset="USDT",
    )
    conn = sqlite3.connect(db)
    row = conn.execute(
        f"SELECT account_execution_mode, lifecycle_state, order_id, fill_trade_id FROM {TABLE_GROUPS} WHERE decision_group_id=?",
        (gid,),
    ).fetchone()
    n_cands = conn.execute(f"SELECT COUNT(*) FROM {TABLE_CANDIDATES} WHERE decision_group_id=?", (gid,)).fetchone()[0]
    conn.close()
    assert row == ("live", "filled", "ex-9", "fill-3")
    assert n_cands == 5


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "false")
    assert observability_enabled() is False
    db = str(tmp_path / "obs.db")
    assert record_day_ranking_group(db, decision=_decision(), bar_timestamp=1) is None
    assert not os.path.exists(db) or sqlite3.connect(db).execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_GROUPS,)
    ).fetchone() is None


def test_terminal_fill_failure_only_for_zero_fill_cancel():
    assert classify_terminal_fill(status="canceled", filled_qty=0, requested_qty=1) == "terminal_fill_failure"
    assert classify_terminal_fill(status="rejected", filled_qty=0, requested_qty=1) == "terminal_fill_failure"
    assert classify_terminal_fill(status="filled", filled_qty=1, requested_qty=1) == "filled"
    assert classify_terminal_fill(status="filled", filled_qty=0.4, requested_qty=1) == "partial_fill"
    assert classify_terminal_fill(status="open", filled_qty=0, requested_qty=1) == "order_submitted"


def test_retention_learning_tables_are_90_days():
    from backend.services.sqlite_large_table_retention import RETENTION_POLICIES

    keep = {p.table: p.keep_days for p in RETENTION_POLICIES}
    assert keep["feature_ohlcv"] == 90
    assert keep["ai_inference_log"] == 90
    assert keep["day_decision_group_records"] == 90
    assert keep["day_decision_candidate_records"] == 90
    assert keep["ai_context_snapshots"] == 30
    assert keep["pipeline_decisions"] == 30
    assert keep["decision_book_tape"] == 14
    assert keep["strategy_runtime_audit"] == 3
    assert keep["paper_trades"] == 90
