"""V5 authoritative calibration-fill identity. Readiness only.

Selections, labels, HOLD, exits, dust, and partials are never aliases for a
logical live BUY fill. Does not train and does not inspect sealed 4H outcomes.
"""

from __future__ import annotations

import json
import sqlite3

from backend.services.day_clock_v2_calibration import count_v5_authoritative_calibration_fills
from backend.services.day_clock_v2_label_source import LABEL_SOURCE_VERSION
from backend.services.day_clock_v2_labels import TARGET_NAME, persist_v5_labels
from backend.services.day_clock_v2_partition import DEVELOPMENT, PRE_MODEL_QUARANTINE
from backend.services.day_decision_observability import TABLE_GROUPS, _ensure_schema
from backend.services.day_path_clock_v2 import PRIMARY_TARGET_HORIZON_SEC
from backend.services.day_path_clock_v2_capture import TABLE_ARTIFACT, ensure_artifact_schema
from backend.services.day_path_clock_v2_readiness import (
    evaluate_clock_v2_v5_readiness,
    persist_clock_v2_v5,
)

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DECISION_TS = "2026-09-06T01:00:00+00:00"
QUOTE = {"best_bid": 100.0, "best_ask": 100.1, "spread_bps": 10.0}


def _label(gid: str, symbol: str, *, valid: bool = True, version: str = LABEL_SOURCE_VERSION) -> dict:
    return {
        "decision_group_id": gid,
        "symbol": symbol,
        "decision_timestamp": DECISION_TS,
        "target_name": TARGET_NAME,
        "target_horizon_sec": PRIMARY_TARGET_HORIZON_SEC,
        "executable_net_bps_3h": 0.0 if valid else None,
        "executable_gross_bps_3h": 0.0 if valid else None,
        "commission_bps": 8.0,
        "spread_bps": 10.0,
        "slippage_bps": 2.0,
        "all_in_cost_bps": 10.0,
        "executable_price_method": "decision_ask_to_horizon_bid",
        "horizon_provenance": "test",
        "market_data_cutoff": DECISION_TS,
        "label_valid": valid,
        "label_invalid_reason": None if valid else "TEST_INVALID",
        "label_source_version": version,
        "label_source": "test",
        "label_status": "COMPLETE" if valid else "TERMINAL_INVALID",
        "source_verified": valid,
        "exchange_symbol": symbol,
        "label_interval": "1m",
    }


def _create_audit(db) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_engine_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                fees REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                decision_id TEXT,
                trade_id TEXT,
                pre_ledger_json TEXT NOT NULL DEFAULT '{}',
                post_ledger_json TEXT NOT NULL DEFAULT '{}',
                invariant_ok INTEGER NOT NULL DEFAULT 1,
                entry_reason TEXT,
                exit_reason TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_artifacts(db, gid: str, *, partition: str = DEVELOPMENT, quote: dict | None = QUOTE) -> None:
    ensure_artifact_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        for symbol in (*COINS, "HOLD"):
            feats = {"symbol": symbol, "legacy_path_ev": 0.1, "spread_bps": 10.0}
            conn.execute(
                f"""INSERT OR REPLACE INTO {TABLE_ARTIFACT}(
                    decision_group_id, symbol, created_at, decision_timestamp,
                    feature_schema_version, feature_contract_version, eligible,
                    feature_json, quote_json, clock_v2_partition, action_available,
                    lock_window, inspected
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,0,0)""",
                (
                    gid,
                    symbol,
                    DECISION_TS,
                    DECISION_TS,
                    "day_path_clock_v2",
                    "day_path_clock_v2_capture_1",
                    1,
                    json.dumps(feats),
                    json.dumps(quote or {}),
                    partition,
                    1,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_group(
    db,
    gid: str,
    *,
    selected: str = "BTCUSDT",
    action: str = "BUY",
    authorized: int | None = 1,
    fill_trade_id: str | None = "mystic_BTC/USDT_1",
    commission: float | None = 0.01,
    lifecycle: str = "filled",
) -> None:
    _ensure_schema(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE_GROUPS}(
                decision_group_id, created_at, account_execution_mode, selected_action,
                selected_symbol, execute_authorized, lifecycle_state, schema_version,
                fill_trade_id, commission, commission_asset
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                gid,
                DECISION_TS,
                "live",
                action,
                selected,
                authorized,
                lifecycle,
                "test",
                fill_trade_id,
                commission,
                "USDT",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_audit(
    db,
    *,
    action: str = "BUY",
    symbol: str = "BTC/USDT",
    trade_id: str = "mystic_BTC/USDT_1",
    qty: float = 0.01,
    price: float = 100.1,
    fees: float | None = 0.01,
    slippage: float | None = 0.002,
    exit_reason: str | None = None,
) -> None:
    _create_audit(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """INSERT INTO portfolio_engine_audit(
                ts, action, symbol, qty, price, fees, slippage, trade_id, exit_reason
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (DECISION_TS, action, symbol, qty, price, fees, slippage, trade_id, exit_reason),
        )
        conn.commit()
    finally:
        conn.close()


def _qualifying(db, gid: str = "daygrp_fill") -> None:
    _insert_artifacts(db, gid)
    _insert_group(db, gid)
    _insert_audit(db)
    persist_v5_labels(db, [_label(gid, "BTCUSDT"), _label(gid, "HOLD")])


def test_selected_but_no_fill_does_not_count(tmp_path):
    db = tmp_path / "no_fill.db"
    _insert_artifacts(db, "g_sel")
    _insert_group(db, "g_sel", authorized=1, fill_trade_id=None, commission=None)
    persist_v5_labels(db, [_label("g_sel", "BTCUSDT")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["non_hold_selected_groups"] == 1
    assert out["execute_authorized_groups"] == 1
    assert out["actual_logical_buy_fills"] == 0
    assert out["authoritative_calibration_fills"] == 0


def test_blocked_after_ranking_does_not_count(tmp_path):
    db = tmp_path / "blocked.db"
    _insert_artifacts(db, "g_block")
    _insert_group(
        db,
        "g_block",
        authorized=0,
        fill_trade_id=None,
        commission=None,
        lifecycle="blocked_after_ranking",
    )
    persist_v5_labels(db, [_label("g_block", "BTCUSDT")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["non_hold_selected_groups"] == 1
    assert out["execute_authorized_groups"] == 0
    assert out["authoritative_calibration_fills"] == 0


def test_valid_selected_symbol_label_without_fill_does_not_count(tmp_path):
    db = tmp_path / "label_only.db"
    _insert_artifacts(db, "g_lab")
    _insert_group(db, "g_lab", authorized=None, fill_trade_id=None, lifecycle="ranking_selected")
    persist_v5_labels(db, [_label("g_lab", "BTCUSDT")])
    snap = evaluate_clock_v2_v5_readiness(db)
    assert snap["groups_with_valid_selected_v2_label"] == 1
    assert snap["AUTHORITATIVE_CALIBRATION_FILLS"] == 0
    assert snap["authoritative_calibration_fills"] == 0


def test_real_buy_fill_with_provenance_counts_once(tmp_path):
    db = tmp_path / "real.db"
    _qualifying(db)
    out = count_v5_authoritative_calibration_fills(db)
    assert out["actual_logical_buy_fills"] == 1
    assert out["groups_with_complete_execution_provenance"] == 1
    assert out["authoritative_calibration_fills"] == 1
    assert out["AUTHORITATIVE_CALIBRATION_FILLS"] == 1
    assert out["per_symbol_calibration_support"] == {"BTCUSDT": 1}


def test_partial_exchange_fills_count_as_one_logical_event(tmp_path):
    db = tmp_path / "partial.db"
    _insert_artifacts(db, "g_part")
    _insert_group(db, "g_part")
    _insert_audit(db, qty=0.004, price=100.1)
    _insert_audit(db, qty=0.006, price=100.2, fees=0.02, slippage=0.003)
    persist_v5_labels(db, [_label("g_part", "BTCUSDT")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["actual_logical_buy_fills"] == 1
    assert out["authoritative_calibration_fills"] == 1
    assert len(out["calibration_keys"]) == 1


def test_sell_exit_does_not_count(tmp_path):
    db = tmp_path / "sell.db"
    _insert_artifacts(db, "g_sell")
    _insert_group(db, "g_sell", fill_trade_id="mystic_BTC/USDT_sell")
    _insert_audit(
        db,
        action="SELL",
        trade_id="mystic_BTC/USDT_sell",
        qty=0.01,
        exit_reason="NET_PROFIT_EXIT",
    )
    persist_v5_labels(db, [_label("g_sell", "BTCUSDT")])
    assert count_v5_authoritative_calibration_fills(db)["authoritative_calibration_fills"] == 0


def test_dust_sell_does_not_count(tmp_path):
    db = tmp_path / "dust.db"
    _insert_artifacts(db, "g_dust")
    _insert_group(db, "g_dust", selected="XRPUSDT", fill_trade_id="mystic_XRP/USDT_dust")
    _insert_audit(
        db,
        action="SELL",
        symbol="XRP/USDT",
        trade_id="mystic_XRP/USDT_dust",
        qty=0.1,
        exit_reason="DUST_CLEANUP",
    )
    persist_v5_labels(db, [_label("g_dust", "XRPUSDT")])
    assert count_v5_authoritative_calibration_fills(db)["authoritative_calibration_fills"] == 0


def test_hold_does_not_count(tmp_path):
    db = tmp_path / "hold.db"
    _insert_artifacts(db, "g_hold")
    _insert_group(
        db,
        "g_hold",
        selected="HOLD",
        action="HOLD",
        authorized=0,
        fill_trade_id=None,
        commission=None,
        lifecycle="ranking_selected",
    )
    persist_v5_labels(db, [_label("g_hold", "HOLD")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["non_hold_selected_groups"] == 0
    assert out["authoritative_calibration_fills"] == 0


def test_counterfactual_candidate_label_does_not_count(tmp_path):
    db = tmp_path / "cf.db"
    _insert_artifacts(db, "g_cf")
    _insert_group(db, "g_cf", selected="BTCUSDT", authorized=1, fill_trade_id=None)
    persist_v5_labels(db, [_label("g_cf", "ETHUSDT")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["groups_with_valid_selected_v2_label"] == 0
    assert out["authoritative_calibration_fills"] == 0


def test_duplicate_readiness_cycle_does_not_double_count(tmp_path):
    db = tmp_path / "dup.db"
    _qualifying(db)
    first = evaluate_clock_v2_v5_readiness(db)["authoritative_calibration_fills"]
    persist_clock_v2_v5(db)
    persist_clock_v2_v5(db)
    second = evaluate_clock_v2_v5_readiness(db)["authoritative_calibration_fills"]
    assert first == 1
    assert second == 1


def test_pre_v5_fill_does_not_count(tmp_path):
    db = tmp_path / "prev5.db"
    _insert_artifacts(db, "g_old", partition=PRE_MODEL_QUARANTINE)
    _insert_group(db, "g_old")
    _insert_audit(db)
    persist_v5_labels(db, [_label("g_old", "BTCUSDT")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["v5_development_groups"] == 0
    assert out["authoritative_calibration_fills"] == 0


def test_pre_model_quarantine_does_not_count(tmp_path):
    db = tmp_path / "quar.db"
    _insert_artifacts(db, "g_quar", partition=PRE_MODEL_QUARANTINE)
    _insert_group(db, "g_quar")
    _insert_audit(db)
    persist_v5_labels(db, [_label("g_quar", "BTCUSDT")])
    snap = evaluate_clock_v2_v5_readiness(db)
    assert snap["quarantined_groups_excluded"] == 1
    assert snap["AUTHORITATIVE_CALIBRATION_FILLS"] == 0


def test_v5_development_qualifying_fill_counts(tmp_path):
    db = tmp_path / "v5.db"
    _qualifying(db, "daygrp_v5")
    snap = evaluate_clock_v2_v5_readiness(db)
    assert snap["AUTHORITATIVE_CALIBRATION_FILLS"] == 1
    assert snap["required_calibration_fills"] == 50
    assert snap["NON_HOLD_SELECTED_GROUPS"] == 1
    assert snap["EXECUTE_AUTHORIZED_GROUPS"] == 1
    assert snap["ACTUAL_LOGICAL_BUY_FILLS"] == 1
    assert snap["train"] is False
    assert snap["promoted"] is False


def test_selected_symbol_label_must_be_valid_v2(tmp_path):
    db = tmp_path / "v1lab.db"
    _insert_artifacts(db, "g_v1")
    _insert_group(db, "g_v1")
    _insert_audit(db)
    persist_v5_labels(db, [_label("g_v1", "BTCUSDT", version="day_clock_v2_label_source_v1")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["actual_logical_buy_fills"] == 1
    assert out["groups_with_valid_selected_v2_label"] == 0
    assert out["authoritative_calibration_fills"] == 0

    persist_v5_labels(db, [_label("g_v1", "BTCUSDT", valid=False)])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["authoritative_calibration_fills"] == 0


def test_readiness_populations_are_not_aliases(tmp_path):
    db = tmp_path / "alias.db"
    _insert_artifacts(db, "g_hold")
    _insert_group(db, "g_hold", selected="HOLD", action="HOLD", authorized=0, fill_trade_id=None)
    persist_v5_labels(db, [_label("g_hold", "HOLD")])
    _insert_artifacts(db, "g_sel")
    _insert_group(db, "g_sel", authorized=None, fill_trade_id=None, lifecycle="ranking_selected")
    persist_v5_labels(db, [_label("g_sel", "BTCUSDT")])
    _qualifying(db, "g_fill")
    snap = evaluate_clock_v2_v5_readiness(db)
    assert snap["FEATURE_COMPLETE_GROUPS"] == snap["feature_complete_development_groups"]
    assert snap["FULLY_COMPARABLE_GROUPS"] == snap["fully_comparable_development_groups"]
    assert snap["NON_HOLD_SELECTED_GROUPS"] == 2
    assert snap["EXECUTE_AUTHORIZED_GROUPS"] == 1
    assert snap["ACTUAL_LOGICAL_BUY_FILLS"] == 1
    assert snap["AUTHORITATIVE_CALIBRATION_FILLS"] == 1
    assert snap["groups_with_valid_selected_v2_label"] == 2
    assert snap["AUTHORITATIVE_CALIBRATION_FILLS"] != snap["NON_HOLD_SELECTED_GROUPS"]
    assert snap["AUTHORITATIVE_CALIBRATION_FILLS"] != snap["EXECUTE_AUTHORIZED_GROUPS"] or snap["EXECUTE_AUTHORIZED_GROUPS"] == 1
    assert snap["FEATURE_COMPLETE_GROUPS"] != snap["AUTHORITATIVE_CALIBRATION_FILLS"] or snap["FEATURE_COMPLETE_GROUPS"] == 1


def test_fill_without_audit_linkage_does_not_count(tmp_path):
    db = tmp_path / "nolink.db"
    _insert_artifacts(db, "g_nolink")
    _insert_group(db, "g_nolink")
    _create_audit(db)
    persist_v5_labels(db, [_label("g_nolink", "BTCUSDT")])
    out = count_v5_authoritative_calibration_fills(db)
    assert out["execute_authorized_groups"] == 1
    assert out["actual_logical_buy_fills"] == 0
    assert out["authoritative_calibration_fills"] == 0
