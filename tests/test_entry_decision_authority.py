"""Provenance stamps and two-book classification. No trading-logic changes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.services.clean_acceptance import day_clean_rows, scalp_clean_rows
from backend.services.entry_decision_authority import (
    DAY_ACCEPTED_MODEL,
    DAY_POLICY,
    LEGACY_DIRECTION,
    LEGACY_SOFT_RANK,
    SCALP_ACCEPTED_MODEL,
    SCALP_POLICY,
    build_day_entry_provenance,
    build_scalp_entry_provenance,
    copy_entry_provenance,
    is_model_controlled,
)
from backend.services.portfolio_engine import TradeExplainability
from backend.services.validation_cutoff import replace_validation_cutoff


def test_day_predicted_path_is_model_controlled():
    prov = build_day_entry_provenance(
        decision_data={
            "path_net_status": "predicted",
            "forward_net_model_version": DAY_ACCEPTED_MODEL,
            "selected_net_expected_value": 0.0008,
            "p_positive_net": 0.61,
            "live_ai_strategy": "day",
            "setup_type": "HTF_TREND_PULLBACK",
        },
        symbol="SOL/USDT",
        decision_id="day_SOLUSDT_1",
        why_selected="highest_final_selection_score",
        direction_probability=0.62,
    )
    assert prov["entry_policy_version"] == DAY_POLICY
    assert prov["model_version"] == DAY_ACCEPTED_MODEL
    assert prov["selected_action"] == "BUY"
    assert prov["selection_reason"] == "PATH_NET_BEATS_HOLD"
    assert prov["hold_ev"] == 0.0
    assert prov["buy_ev"] == 0.0008
    assert is_model_controlled(prov, engine="day") is True


def test_day_missing_path_status_is_legacy_direction():
    prov = build_day_entry_provenance(
        decision_data={"selected_net_expected_value": 0.0007, "prob_buy": 0.57},
        symbol="XRP/USDT",
        decision_id="day_XRPUSDT_1",
        why_selected="highest_final_selection_score",
        direction_probability=0.57,
    )
    assert prov["entry_policy_version"] == LEGACY_DIRECTION
    assert prov["model_version"] == LEGACY_DIRECTION
    assert is_model_controlled(prov, engine="day") is False


def test_scalp_path_net_beats_hold():
    prov = build_scalp_entry_provenance(
        ranking_meta={
            "forward_net_model_version": SCALP_ACCEPTED_MODEL,
            "selected_expected_net_ev": 0.0002,
            "hold_action_ev": 0.0,
            "selected_predicted_prob_positive_net": 0.33,
            "rank_score": 0.28,
        },
        symbol="XRPUSDT",
        setup_name="vwap_ema_reclaim",
        strategy_passed=False,
        epoch=1786770558.592,
        opportunity_id="2087",
    )
    assert prov["entry_policy_version"] == SCALP_POLICY
    assert prov["soft_rank_entry"] is True
    assert prov["selected_action"] == "BUY_XRPUSDT"
    assert prov["selection_reason"] == "PATH_NET_BEATS_HOLD"
    assert is_model_controlled(prov, engine="scalp") is True


def test_scalp_without_model_is_legacy_soft_rank():
    prov = build_scalp_entry_provenance(
        ranking_meta={"rank_score": 0.95, "selection_reason": "rank=0.95"},
        symbol="ETHUSDT",
        setup_name="vwap_ema_reclaim",
        strategy_passed=False,
        epoch=1786753465.866,
    )
    assert prov["entry_policy_version"] == LEGACY_SOFT_RANK
    assert is_model_controlled(prov, engine="scalp") is False


def test_existing_partial_scalp_buy_is_not_model_controlled():
    partial = {
        "decision_policy_version": "scalp_path_aware_v1",
        "forward_net_model_version": "scalp_path_net_v1",
        "predicted_net_ev": 0.00027,
        "soft_rank_entry": True,
    }
    assert is_model_controlled(partial, engine="scalp") is False


def test_sell_copies_entry_provenance():
    entry = build_scalp_entry_provenance(
        ranking_meta={
            "forward_net_model_version": SCALP_ACCEPTED_MODEL,
            "selected_expected_net_ev": 0.0002,
            "hold_action_ev": 0.0,
        },
        symbol="BTCUSDT",
        setup_name="vwap_ema_reclaim",
        strategy_passed=False,
        epoch=1.0,
    )
    sell = copy_entry_provenance(entry, {"exit_reason": "PATH_EXECUTABLE_PROFIT"})
    assert sell["model_version"] == SCALP_ACCEPTED_MODEL
    assert sell["hold_ev"] == 0.0
    assert sell["exit_reason"] == "PATH_EXECUTABLE_PROFIT"


def test_explainability_to_dict_merges_provenance():
    ex = TradeExplainability(
        trade_id="t1",
        symbol="SOL/USDT",
        side="BUY",
        timestamp="2026-08-15T14:30:07+00:00",
    )
    ex.entry_provenance = build_day_entry_provenance(
        decision_data={
            "path_net_status": "predicted",
            "forward_net_model_version": DAY_ACCEPTED_MODEL,
            "selected_net_expected_value": 0.0006,
        },
        symbol="SOL/USDT",
        decision_id="day_SOLUSDT_1",
    )
    payload = ex.to_dict()
    assert payload["entry_policy_version"] == DAY_POLICY
    assert payload["model_version"] == DAY_ACCEPTED_MODEL
    assert payload["selected_action"] == "BUY"


def test_two_books_split_on_new_stamps_only(tmp_path: Path):
    day_db = str(tmp_path / "day.db")
    scalp_db = str(tmp_path / "scalp.db")
    conn = sqlite3.connect(day_db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT,
            pnl REAL, timestamp TEXT, entry_timestamp TEXT, exit_reason TEXT,
            explainability_json TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO paper_trades VALUES
        (991,'sell1','XRP/USDT','SELL',-2.08,'2026-08-15T07:30:20+00:00','2026-08-15T02:30:07+00:00','TIME_STOP_EXIT','{"buy_probability":0.57,"predicted_net_return":0.0007}')
        """
    )
    conn.execute(
        """
        INSERT INTO paper_trades VALUES
        (989,'buy1','XRP/USDT','BUY',NULL,'2026-08-15T02:30:07+00:00','2026-08-15T02:30:07+00:00',NULL,'{"buy_probability":0.57,"predicted_net_return":0.0007}')
        """
    )
    conn.commit()
    conn.close()
    replace_validation_cutoff(day_db, engine="day", cutoff_utc="2026-08-15T02:03:02.291353+00:00", label="test")

    sconn = sqlite3.connect(scalp_db)
    sconn.execute(
        """
        CREATE TABLE scalp_paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT,
            pnl_usd REAL, created_at TEXT, exit_reason TEXT, diagnostics_json TEXT
        )
        """
    )
    sconn.execute("CREATE TABLE scalp_paper_positions (symbol TEXT, status TEXT, entry_time TEXT, trade_id TEXT)")
    buy_diag = {
        "soft_rank_entry": True,
        "decision_policy_version": "scalp_path_aware_v1",
        "forward_net_model_version": "scalp_path_net_v1",
        "predicted_net_ev": 0.0002,
    }
    sconn.execute(
        "INSERT INTO scalp_paper_trades VALUES (1,'scalp_paper_XRPUSDT_1786770558592','XRPUSDT','BUY',NULL,'2026-08-15 05:09:18',NULL,?)",
        (json.dumps(buy_diag),),
    )
    sconn.execute(
        "INSERT INTO scalp_paper_trades VALUES (2,'scalp_paper_XRPUSDT_1786770558592_SELL','XRPUSDT','SELL',-0.02,'2026-08-15 05:29:24','MAX_HOLD_HARD_LIMIT',?)",
        (json.dumps({"soft_rank_entry": True}),),
    )
    sconn.commit()
    sconn.close()
    replace_validation_cutoff(scalp_db, engine="scalp", cutoff_utc="2026-08-15T02:03:02.291353+00:00", label="test")

    day = day_clean_rows(day_db)
    scalp = scalp_clean_rows(scalp_db)
    assert day["clean_runtime"]["n"] == 1
    assert day["model_controlled"]["n"] == 0
    assert scalp["clean_runtime"]["n"] == 1
    assert scalp["model_controlled"]["n"] == 0
