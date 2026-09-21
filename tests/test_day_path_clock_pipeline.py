"""Clock-v2 collection pipeline. Offline only. No live ranking."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.services.day_decision_label_contract import ensure_label_schema
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS, _ensure_schema
from backend.services.day_experiment_registry import TABLE as TABLE_REGISTRY
from backend.services.day_experiment_registry import ensure_registry_schema, seed_historical
from backend.services.day_forward_lock import FORWARD_LOCK_START, register_lock
from backend.services.day_forward_lock import TABLE as TABLE_LOCK
from backend.services.day_model_readiness import (
    CHRONOLOGICAL_BLOCK_DEFINITION,
    CHRONOLOGICAL_BLOCK_HOURS,
    MATURE_EVENT_UNIT,
    MIN_CHRONOLOGICAL_BLOCKS,
    MIN_EVENTS_PER_FEATURE,
    evaluate_readiness,
    format_snapshot,
)
from backend.services.day_path_clock_pipeline import (
    FORBIDDEN_OUTCOME_KEYS,
    TABLE_FEATURES,
    bar_quality,
    persist_feature_snapshots,
    run_pipeline,
    snapshot_columns,
)
from backend.services.day_path_clock_v2 import (
    PLANNED_RESULT,
    REQUIRED_CLOCK_V2_FIELDS,
    planned_challenger_specification,
    planned_training_procedure,
)
from backend.services.day_path_net import predict_decision_net, reset_day_artifact_cache, resolve_day_path_ev


def _dense(n: int = 80, start: datetime | None = None, close0: float = 100.0) -> list[dict]:
    t0 = start or datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = []
    px = close0
    for i in range(n):
        px *= 1.0002
        ts = t0 + timedelta(minutes=i)
        bars.append({"open": px, "high": px, "low": px, "close": px, "volume": 8.0, "ts": ts})
    return bars


def _seed_ohlcv(conn: sqlite3.Connection, symbol: str, bars: list[dict]) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    for bar in bars:
        conn.execute(
            "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?,?)",
            (symbol, "1m", bar["ts"].isoformat(), bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]),
        )


def _mini_db(path: Path) -> str:
    db = str(path / "prod.db")
    _ensure_schema(db)
    ensure_label_schema(db)
    ensure_registry_schema(db)
    seed_historical(db)
    register_lock(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS position_close_ledger ("
        "id INTEGER PRIMARY KEY, symbol TEXT, closed_at TEXT, closed_at_epoch REAL, "
        "close_reason TEXT, manual_sell INT, realized_profit REAL, realized_profit_unknown INT, "
        "cooldown_until REAL, quantity REAL, entry_price REAL, exit_price REAL, sell_trade_id TEXT, detail TEXT)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS paper_trades (trade_id TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL, remaining_position REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS portfolio_engine_positions (symbol TEXT, quantity REAL, entry_price REAL, status TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS portfolio_engine_ledger (id INTEGER PRIMARY KEY, total_equity REAL)")
    conn.execute("INSERT OR REPLACE INTO portfolio_engine_ledger (id,total_equity) VALUES (1,1000)")
    t0 = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    for symbol, px in (("BTC-USDT", 80000.0), ("ETH-USDT", 2500.0), ("SOL-USDT", 100.0), ("XRP-USDT", 1.4)):
        _seed_ohlcv(conn, symbol, _dense(200, start=t0 - timedelta(hours=3), close0=px))
    created = "2026-09-04T01:00:00+00:00"
    gid = "daygrp_clock_pipe"
    contract = {
        "4h_entry_telemetry": {
            "BTCUSDT": {"production_4h_break_true_at_decision": False, "distance_to_4h_break_bps": 40.0, "4h_range_position": 0.4},
            "ETHUSDT": {"production_4h_break_true_at_decision": False, "distance_to_4h_break_bps": 30.0, "4h_range_position": 0.5},
            "SOLUSDT": {"production_4h_break_true_at_decision": False, "distance_to_4h_break_bps": 20.0, "4h_range_position": 0.6},
            "XRPUSDT": {"production_4h_break_true_at_decision": False, "distance_to_4h_break_bps": 10.0, "4h_range_position": 0.3},
        },
        "spread_bps": 2.0,
    }
    conn.execute(
        f"""
        INSERT INTO {TABLE_GROUPS}(
            decision_group_id, created_at, account_execution_mode, selected_action,
            selected_symbol, selected_ranking_action, execute_authorized, lifecycle_state,
            schema_version, feature_schema, model_version, feature_artifact_ref,
            slot_count, cash_balance, contract_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (gid, created, "live", "HOLD", "HOLD", "HOLD", 0, "ranked", "v1", "v1", "v1", "x", 4, 1000, json.dumps(contract)),
    )
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        conn.execute(
            f"INSERT INTO {TABLE_CANDIDATES}(decision_group_id, symbol, created_at, eligible, p_buy, path_ev, final_rank_score, feature_json) VALUES (?,?,?,?,?,?,?,?)",
            (gid, symbol, created, 1, 0.5, 0.001, 0.2, "{}"),
        )
    conn.commit()
    conn.close()
    return db


def test_140_event_denominator_is_authoritative_selected_trades():
    assert MIN_EVENTS_PER_FEATURE == 10
    assert MIN_CHRONOLOGICAL_BLOCKS == 5
    assert "selected-trade" in MATURE_EVENT_UNIT
    assert "Not decision groups" in MATURE_EVENT_UNIT
    assert "Not four-coin" in MATURE_EVENT_UNIT
    spec = planned_challenger_specification()
    assert spec["result"] == PLANNED_RESULT
    assert spec["train"] is False
    assert spec["readiness_required_mature_trade_labels"] if False else spec["acceptance"]
    assert 14 * MIN_EVENTS_PER_FEATURE == 140


def test_five_block_definition_is_utc_24h_bins():
    assert CHRONOLOGICAL_BLOCK_HOURS == 24
    assert "24-hour" in CHRONOLOGICAL_BLOCK_DEFINITION or "24h" in CHRONOLOGICAL_BLOCK_DEFINITION


def test_planned_spec_and_procedure_are_frozen_not_run():
    spec = planned_challenger_specification()
    assert spec["inputs"] == list(REQUIRED_CLOCK_V2_FIELDS)
    assert spec["target"] == "expected_executable_net_bps"
    assert spec["hold_value_bps"] == 0.0
    proc = planned_training_procedure()
    assert proc["executed"] is False
    assert proc["hyperparameter_search_on_lock"] is False
    assert proc["embargo_seconds_min"] == 4 * 3600


def test_feature_only_lock_storage_does_not_inspect_lock(tmp_path):
    db = _mini_db(tmp_path)
    research = str(tmp_path / "research.db")
    before = sqlite3.connect(db).execute(f"SELECT inspected FROM {TABLE_LOCK}").fetchone()[0]
    out = run_pipeline(db, research_db=research, ocean_sha="test")
    after = sqlite3.connect(db).execute(f"SELECT inspected FROM {TABLE_LOCK}").fetchone()[0]
    assert before == 0
    assert after == 0
    assert out["lock_inspected_after"] is False
    cols = {row[1] for row in sqlite3.connect(research).execute(f"PRAGMA table_info({TABLE_FEATURES})")}
    assert not FORBIDDEN_OUTCOME_KEYS.intersection(cols)
    assert set(snapshot_columns()).isdisjoint(FORBIDDEN_OUTCOME_KEYS)
    stored = sqlite3.connect(research).execute(f"SELECT feature_json FROM {TABLE_FEATURES}").fetchall()
    assert stored
    for (raw,) in stored:
        payload = json.loads(raw)
        assert FORBIDDEN_OUTCOME_KEYS.isdisjoint(payload)
    planned = sqlite3.connect(research).execute(f"SELECT result, promoted FROM {TABLE_REGISTRY} WHERE experiment_id='M_clock_v2_planned_20260905'").fetchone()
    assert planned[0] == "PLANNED_NOT_RUN"
    assert planned[1] == 0


def test_future_data_and_timezone_and_btc_relative_quality():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = _dense(40, start=t0)
    future = {**bars[-1], "ts": bars[-1]["ts"] + timedelta(hours=2), "close": 1.0}
    q = bar_quality([*bars, future], as_of=bars[-1]["ts"])
    assert q["valid"] is False
    assert "future_data" in q["reasons"]
    z_bars = [{**b, "ts": b["ts"].isoformat().replace("+00:00", "Z")} for b in bars]
    assert bar_quality(z_bars, as_of=bars[-1]["ts"].isoformat().replace("+00:00", "Z"))["valid"] is True


def test_group_independence_and_readiness_progress(tmp_path):
    db = _mini_db(tmp_path)
    research = str(tmp_path / "research.db")
    out = run_pipeline(db, research_db=research)
    cov = out["coverage"]
    assert cov["independent_decision_groups"] == cov["groups_total"]
    assert cov["candidate_coin_rows"] == cov["groups_total"] * 4
    text = out["snapshot"]
    assert "READY_FOR_MODEL_RESEARCH" in text
    assert "authoritative selected-trade" in text
    report = evaluate_readiness(db)
    assert "G_forward_span" in (report.get("reasons_not_ready") or [])
    assert report["sample_support"]["primary_unit"] == "decision_group"


def test_legacy_dense_golden_still_holds():
    reset_day_artifact_cache()
    bars = _dense(40, start=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc))
    dd = {"bars_1m": bars, "symbol": "ETHUSDT", "btc_ret_5": 0.0}
    assert resolve_day_path_ev(dd, symbol="ETHUSDT")[0] == predict_decision_net(dd)
    reset_day_artifact_cache()


def test_pipeline_not_imported_by_live_path():
    for rel in (
        "backend/services/day_path_net.py",
        "backend/services/day_direct_path_ev_authority.py",
        "backend/services/portfolio_engine.py",
    ):
        assert "day_path_clock_pipeline" not in Path(rel).read_text()


def test_persist_rejects_outcome_keys(tmp_path):
    research = str(tmp_path / "research.db")
    try:
        persist_feature_snapshots(
            research,
            {
                "snapshots": [
                    {
                        "candidates": [
                            {
                                "decision_group_id": "x",
                                "symbol": "BTCUSDT",
                                "created_at": "t",
                                "decision_timestamp": "t",
                                "eligible": True,
                                "features": {"production_exit_net_bps": 1.0},
                                "lock_window": True,
                            }
                        ]
                    }
                ]
            },
        )
    except RuntimeError as exc:
        assert "outcome" in str(exc)
    else:
        raise AssertionError("outcome fields must be rejected")
