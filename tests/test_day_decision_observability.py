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
    assert keep["day_decision_feature_artifacts"] == 90
    assert keep["day_decision_outcome_labels"] == 90
    assert keep["ai_context_snapshots"] == 30
    assert keep["pipeline_decisions"] == 30
    assert keep["decision_book_tape"] == 14
    assert keep["strategy_runtime_audit"] == 3
    assert keep["paper_trades"] == 90


def test_hold_is_explicit_and_rank_deltas_preserved():
    dec = _decision()
    cands = [
        SimpleNamespace(
            symbol="ETHUSDT",
            decision_data={
                "prob_buy": 0.6,
                "final_selection_score": 0.0008,
                "intelligence_rank_delta": 0.01,
                "quality_opinion_penalty": 0.02,
                "feature_vector": [float(i) for i in range(145)],
            },
        ),
        SimpleNamespace(symbol="BTCUSDT", decision_data={"prob_buy": 0.4, "final_selection_score": 0.0001}),
    ]
    contract = build_group_contract(decision=dec, candidates=cands, bar_timestamp=100)
    hold = next(r for r in contract["candidates"] if r["symbol"] == "HOLD")
    assert hold["gross_value"] == 0.0
    assert hold["net_value"] == 0.0
    assert hold["capital_usage"] == 0.0
    assert hold["path_ev"] == 0.0
    assert hold["proposed_notional"] == 0.0
    eth = next(r for r in contract["candidates"] if r["symbol"] == "ETHUSDT")
    assert eth["all_rank_deltas"]["intelligence_rank_delta"] == 0.01
    assert eth["all_haircuts"]["quality_opinion_penalty"] == 0.02
    assert eth["feature_artifact_id"]
    assert eth["rank_position"] == 1
    assert contract["strategy_id"] == "day"
    assert contract["runtime_trading_mode"] in {"paper", "live", "unknown"}
    assert contract["lifecycle_state"] == "ranking_selected"


def test_feature_artifact_persisted(tmp_path, monkeypatch):
    from backend.services.day_decision_observability import TABLE_FEATURE_ARTIFACTS

    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "true")
    db = str(tmp_path / "obs.db")
    vec = [float(i) * 0.001 for i in range(145)]
    cands = [SimpleNamespace(symbol="ETHUSDT", decision_data={"feature_vector": vec, "prob_buy": 0.7, "final_selection_score": 0.001})]
    gid = record_day_ranking_group(db, decision=_decision(), candidates=cands, bar_timestamp=300)
    conn = sqlite3.connect(db)
    art = conn.execute(f"SELECT feature_dim, feature_values_json FROM {TABLE_FEATURE_ARTIFACTS}").fetchone()
    cand = conn.execute(
        f"SELECT feature_hash, feature_json FROM {TABLE_CANDIDATES} WHERE decision_group_id=? AND symbol='ETHUSDT'",
        (gid,),
    ).fetchone()
    conn.close()
    assert art is not None
    assert art[0] == 145
    stored = json.loads(art[1])
    assert stored == vec
    meta = json.loads(cand[1])
    assert meta["feature_artifact_id"] == cand[0]
    assert "feature_vector" not in meta


def test_blocked_after_ranking_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    db = str(tmp_path / "obs.db")
    gid = record_day_ranking_group(db, decision=_decision(), bar_timestamp=400)
    update_day_decision_lifecycle(
        db,
        decision_group_id=gid,
        execute_authorized=False,
        lifecycle_state="blocked_after_ranking",
        block_reason="NO_SLOT_OR_CAPITAL",
    )
    conn = sqlite3.connect(db)
    payload = json.loads(
        conn.execute(f"SELECT contract_json FROM {TABLE_GROUPS} WHERE decision_group_id=?", (gid,)).fetchone()[0]
    )
    conn.close()
    assert payload["lifecycle_state"] == "blocked_after_ranking"
    assert payload["block_reason"] == "NO_SLOT_OR_CAPITAL"
    assert payload["execute_authorization"] is False


def test_partial_and_terminal_and_hold_lifecycle():
    from backend.services.day_decision_observability import classify_execute_lifecycle

    assert classify_execute_lifecycle(result=None, block_reason="SPREAD") == "blocked_after_ranking"
    assert (
        classify_execute_lifecycle(result={"status": "canceled", "filled_qty": 0, "requested_qty": 1}) == "terminal_fill_failure"
    )
    assert classify_execute_lifecycle(result={"status": "filled", "filled_qty": 0.2, "requested_qty": 1}) == "partial_fill"
    assert classify_execute_lifecycle(result={"order_id": "o1", "status": "new"}) == "order_submitted"
    assert classify_execute_lifecycle(result={"trade_id": "t1", "filled_qty": 1, "requested_qty": 1}) == "filled"
    hold = build_group_contract(decision={"selected_action": "HOLD"}, bar_timestamp=1)
    assert hold["lifecycle_state"] == "HOLD"
    assert hold["selected_symbol"] == "HOLD"


def test_mode_source_does_not_change_trading_decision(monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    live_dec = _decision()
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "paper")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    paper_dec = _decision()
    assert live_dec["selected_action"] == paper_dec["selected_action"]
    assert live_dec["selected_symbol"] == paper_dec["selected_symbol"]
    live_c = build_group_contract(decision=live_dec, bar_timestamp=1)
    paper_c = build_group_contract(decision=paper_dec, bar_timestamp=1)
    assert live_c["account_execution_mode"] == "paper"
    assert paper_c["account_execution_mode"] == "paper"
    assert live_c["selected_action"] == paper_c["selected_action"]


def test_storage_estimation_prefers_90_when_disk_allows():
    from backend.services.day_decision_observability import estimate_observability_storage

    est = estimate_observability_storage(
        groups_per_day=100.0,
        group_bytes=6000.0,
        candidate_bytes=400.0,
        artifact_bytes=9000.0,
        current_db_bytes=8_137_711_616,
        disk_free_bytes=5_800_000_000,
        reserve_bytes=2_000_000_000,
    )
    assert est["selected_retention_days"] == 90
    assert est["horizons"]["90"]["fits"] is True
    assert est["rows_per_day"] == 100.0
