"""P1C: low-MFE STALL learning persistence + non-blocking outcome demotion."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_outcome_training_writer import record_outcome_training_row
from backend.services.day_controlled_exits import EXIT_STALL_DEAD
from backend.services.day_outcome_attribution import (
    build_attribution_payload,
    ensure_outcome_attribution_table,
    record_outcome_attribution,
)
from backend.services.day_trade_thesis import (
    SETUP_RANGE_BOUNCE,
    canonical_day_exit_reason,
    split_day_exit_reasons,
)
from backend.services.symbol_setup_outcome_penalty import (
    DAY_UNIVERSAL_PENALTY_SYMBOLS,
    apply_v3_outcome_ranking_to_decision_data,
    evaluate_low_mfe_stall_penalty,
    evaluate_outcome_penalty,
)


def test_1_stall_dead_preserved_for_learning_display_canonical():
    parts = split_day_exit_reasons(EXIT_STALL_DEAD)
    assert parts["raw_exit_reason"] == "STALL_EXIT_DEAD_NO_MFE"
    assert parts["canonical_exit_reason"] == "STALL_EXIT"
    assert parts["dead_trade_reason"] == "DEAD_NO_MFE"
    # Display path still collapses.
    assert canonical_day_exit_reason(EXIT_STALL_DEAD) == "STALL_EXIT"


def test_2_mfe_mae_persist_into_ai_outcome_training_rows(tmp_path: Path):
    db = str(tmp_path / "learn.db")
    ensure_ai_canonical_tables(db)
    rid = record_outcome_training_row(
        symbol="SOL/USDT",
        opened_at_utc="2026-08-03T21:15:06+00:00",
        closed_at_utc="2026-08-03T23:15:42+00:00",
        hold_seconds=7236.0,
        entry_price=73.78,
        exit_price=73.45,
        net_profit_usd=-10.0,
        net_profit_pct=-0.0049,
        gross_pnl_pct=-0.0049,
        close_reason="STALL_EXIT",
        strategy_id="day",
        explainability={
            "setup_type": SETUP_RANGE_BOUNCE,
            "entry_thesis": SETUP_RANGE_BOUNCE,
            "day_route_regime": "range",
            "final_selection_score": 0.34,
            "selected_net_expected_value": 0.075,
            "rank_score": 0.64,
            "prob_buy": 0.58,
            "prob_hold": 0.42,
            "prob_sell": 0.0,
            "ai_confidence": 0.41,
            "mfe_pct": 0.000669,
            "mae_pct": 0.005565,
            "raw_exit_reason": "STALL_EXIT_DEAD_NO_MFE",
            "canonical_exit_reason": "STALL_EXIT",
            "dead_trade_reason": "DEAD_NO_MFE",
        },
        db_path=db,
    )
    assert rid is not None
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT max_favorable_excursion, max_adverse_excursion, score_components_json FROM ai_outcome_training_rows ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["max_favorable_excursion"] is not None
    assert row["max_adverse_excursion"] is not None
    assert float(row["max_favorable_excursion"]) == pytest.approx(0.000669, rel=1e-6)
    assert float(row["max_adverse_excursion"]) == pytest.approx(0.005565, rel=1e-6)
    sc = json.loads(row["score_components_json"])
    assert sc["mfe_pct"] == pytest.approx(0.000669, rel=1e-6)
    assert sc["mae_pct"] == pytest.approx(0.005565, rel=1e-6)
    assert sc["exit_reason_raw"] == "STALL_EXIT_DEAD_NO_MFE"
    assert sc["exit_reason_canonical"] == "STALL_EXIT"
    assert sc["setup_type"] == SETUP_RANGE_BOUNCE


def test_3_rank_data_json_includes_mfe_mae_and_exit_split():
    """Mirror portfolio_engine learning rank_data enrichment contract."""
    mfe_pct = 0.000669
    mae_pct_abs = 0.005565
    parts = split_day_exit_reasons("STALL_EXIT_DEAD_NO_MFE")
    rank_data = {
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct_abs,
        "raw_exit_reason": parts["raw_exit_reason"],
        "canonical_exit_reason": parts["canonical_exit_reason"],
        "dead_trade_reason": parts["dead_trade_reason"],
        "setup_type_canonical": SETUP_RANGE_BOUNCE,
        "final_selection_score": 0.34,
    }
    assert "mfe_pct" in rank_data and rank_data["mfe_pct"] is not None
    assert "mae_pct" in rank_data and rank_data["mae_pct"] is not None
    assert rank_data["raw_exit_reason"] == "STALL_EXIT_DEAD_NO_MFE"
    assert rank_data["canonical_exit_reason"] == "STALL_EXIT"


def test_4_day_outcome_attribution_filled(tmp_path: Path):
    db = str(tmp_path / "attr.db")
    ensure_outcome_attribution_table(db)
    explain = {
        "setup_type": SETUP_RANGE_BOUNCE,
        "entry_thesis": SETUP_RANGE_BOUNCE,
        "setup_regime_remapped_from": "HTF_TREND_PULLBACK",
        "day_route_regime": "range",
        "final_selection_score": 0.3409612,
        "rank_score": 0.648653,
        "selected_net_expected_value": 0.07545451,
        "prob_buy": 0.579,
        "prob_hold": 0.421,
        "prob_sell": 0.0,
        "ai_confidence": 0.409,
        "buy_margin": 0.158,
        "setup_score": 0.69,
        "execution_quality_score": 0.84,
        "feature_health_score": 0.90,
        "feature_health_pass": True,
        "mfe_pct": 0.000669,
        "mae_pct": 0.005565,
        "raw_exit_reason": "STALL_EXIT_DEAD_NO_MFE",
        "canonical_exit_reason": "STALL_EXIT",
        "entry_timestamp": "2026-08-03T21:15:06+00:00",
    }
    payload = build_attribution_payload(
        trade_id="mystic_SOL/USDT_test",
        symbol="SOL/USDT",
        explainability=explain,
        net_profit_usd=-10.0,
        net_profit_pct=-0.0049,
        close_reason="STALL_EXIT",
        hold_seconds=7236.0,
    )
    assert payload["setup_thesis"] == SETUP_RANGE_BOUNCE
    assert payload["regime"] == "range"
    assert float(payload["final_selection_score"]) > 0
    assert payload["model_probabilities"]["prob_buy"] == 0.579
    assert payload["model_probabilities"]["prob_hold"] == 0.421
    assert payload["model_probabilities"]["prob_sell"] == 0.0
    assert payload["raw_exit_reason"] == "STALL_EXIT_DEAD_NO_MFE"
    assert payload["mfe_pct"] is not None

    rid = record_outcome_attribution(
        trade_id="mystic_SOL/USDT_test",
        symbol="SOL/USDT",
        explainability=explain,
        net_profit_usd=-10.0,
        net_profit_pct=-0.0049,
        close_reason="STALL_EXIT",
        hold_seconds=7236.0,
        db_path=db,
    )
    assert rid is not None
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT setup_thesis, regime, final_selection_score, model_probabilities_json FROM day_outcome_attribution WHERE trade_id=?",
            ("mystic_SOL/USDT_test",),
        ).fetchone()
    assert row["setup_thesis"] == SETUP_RANGE_BOUNCE
    assert row["regime"] == "range"
    assert float(row["final_selection_score"]) > 0
    probs = json.loads(row["model_probabilities_json"])
    assert probs["prob_buy"] is not None
    assert probs["prob_hold"] is not None
    assert probs["prob_sell"] is not None


def _seed_low_mfe_stall_sells(db: str, *, symbol: str, setup: str, n: int = 2) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                pnl REAL,
                exit_reason TEXT,
                timestamp TEXT,
                entry_price REAL,
                price REAL,
                explainability_json TEXT,
                hold_time_seconds REAL,
                is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        for i in range(n):
            explain = {
                "setup_type": setup,
                "entry_thesis": setup,
                "setup_type_canonical": setup,
                "day_route_regime": "range",
                "selected_net_expected_value": 0.07,
                "final_selection_score": 0.33,
                "mfe_pct": 0.0008,
                "mae_pct": 0.0040,
                "raw_exit_reason": "STALL_EXIT_DEAD_NO_MFE",
                "canonical_exit_reason": "STALL_EXIT",
                "dead_trade_reason": "DEAD_NO_MFE",
            }
            conn.execute(
                """
                INSERT INTO paper_trades (
                    symbol, side, pnl, exit_reason, timestamp, entry_price, price,
                    explainability_json, hold_time_seconds, is_synthetic
                ) VALUES (?, 'SELL', ?, 'STALL_EXIT', ?, 100.0, 99.5, ?, 7200, 0)
                """,
                (symbol, -5.0 - i, f"2026-08-03T2{i}:00:00+00:00", json.dumps(explain)),
            )
        conn.commit()


def test_6_sol_range_bounce_low_mfe_stall_applies_penalty(tmp_path: Path):
    db = str(tmp_path / "pen.db")
    _seed_low_mfe_stall_sells(db, symbol="SOL/USDT", setup=SETUP_RANGE_BOUNCE, n=2)
    # Legacy XRP-only evaluator must not be the only path.
    xrp_scoped = evaluate_outcome_penalty("SOL/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db)
    assert xrp_scoped.get("reason") == "not_xrp_penalty_scope"

    pen = evaluate_low_mfe_stall_penalty("SOL/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db)
    assert pen["applied"] is True
    assert pen["hard_block"] is False
    assert pen["reason"] != "not_xrp_penalty_scope"
    assert pen["rank_delta"] < 0
    assert pen["ev_factor"] < 1.0

    dd = {
        "setup_type": SETUP_RANGE_BOUNCE,
        "entry_thesis": SETUP_RANGE_BOUNCE,
        "day_route_regime": "range",
        "selected_net_expected_value": 0.10,
        "buy_margin": 0.12,
    }
    out = apply_v3_outcome_ranking_to_decision_data(dd, "SOL/USDT", raw_rank_score=0.55, buy_margin=0.12, db_path=db)
    assert out["outcome_penalty_applied"] is True
    assert out.get("candidate_eligible") is True
    assert out.get("outcome_penalty_hard_block") is False
    assert float(out["adjusted_ev"]) < float(out["raw_ev"])
    assert float(out["outcome_adjusted_rank_score"]) < 0.55
    assert out.get("final_selection_score") is not None


def test_7_all_top4_symbols_in_penalty_scope(tmp_path: Path):
    db = str(tmp_path / "scope.db")
    assert {"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"} == DAY_UNIVERSAL_PENALTY_SYMBOLS
    for sym in sorted(DAY_UNIVERSAL_PENALTY_SYMBOLS):
        _seed_low_mfe_stall_sells(db, symbol=sym, setup=SETUP_RANGE_BOUNCE, n=2)
        pen = evaluate_low_mfe_stall_penalty(sym, SETUP_RANGE_BOUNCE, "range", db_path=db)
        assert pen["applied"] is True, sym
        assert pen["hard_block"] is False, sym
        assert pen["reason"] != "symbol_not_in_day_penalty_scope", sym


def test_8_penalty_is_non_blocking(tmp_path: Path):
    db = str(tmp_path / "noblock.db")
    _seed_low_mfe_stall_sells(db, symbol="BTC/USDT", setup="HTF_TREND_PULLBACK", n=3)
    dd = {
        "setup_type": "HTF_TREND_PULLBACK",
        "day_route_regime": "bull",
        "selected_net_expected_value": 0.08,
        "buy_margin": 0.10,
    }
    out = apply_v3_outcome_ranking_to_decision_data(dd, "BTC/USDT", raw_rank_score=0.50, buy_margin=0.10, db_path=db)
    assert out.get("candidate_eligible") is True
    assert out.get("outcome_penalty_hard_block") is False
    assert out.get("hard_block") is False
    # Still has a usable final score — demoted, not removed (may be negative FSS).
    assert out.get("final_selection_score") is not None
    if out.get("outcome_penalty_applied"):
        assert float(out["adjusted_ev"]) <= float(out["raw_ev"])
        assert float(out["outcome_adjusted_rank_score"]) <= 0.50
