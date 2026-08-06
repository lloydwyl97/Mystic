"""Path A: DAY ranking engine restored — all setups ranked, none hard-blocked by fill gates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
    remap_setup_for_day_regime,
)
from backend.services.symbol_setup_outcome_penalty import (
    apply_v3_outcome_ranking_to_decision_data,
    evaluate_low_mfe_stall_penalty,
    should_defer_day_fbr_fill,
    should_defer_day_htf_fill,
    should_defer_low_mfe_stall_fill,
)


def _ensure_trades_table(db: str) -> None:
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
        conn.commit()


def _seed(db: str, symbol: str, setup: str, n: int = 3) -> None:
    _ensure_trades_table(db)
    for i in range(n):
        explain = {
            "setup_type": setup,
            "setup_type_canonical": setup,
            "mfe_pct": 0.0005,
            "mae_pct": 0.0045,
            "raw_exit_reason": "STALL_EXIT_DEAD_NO_MFE",
            "canonical_exit_reason": "STALL_EXIT",
            "dead_trade_reason": "DEAD_NO_MFE",
        }
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    symbol, side, pnl, exit_reason, timestamp, entry_price, price,
                    explainability_json, hold_time_seconds, is_synthetic
                ) VALUES (?, 'SELL', ?, 'STALL_EXIT', ?, 100.0, 99.4, ?, ?, 0)
                """,
                (symbol, -7.0 - i, f"2026-08-05T{11+i:02d}:00:00+00:00", json.dumps(explain), 7000.0),
            )
            conn.commit()


def test_buy_queue_source_has_no_hard_setup_defers():
    src = Path(__file__).resolve().parents[1] / "backend" / "services" / "portfolio_engine.py"
    text = src.read_text(encoding="utf-8")
    assert "BUY_DEFERRED_DAY_FBR" not in text
    assert "BUY_DEFERRED_DAY_HTF" not in text
    assert "BUY_DEFERRED_LOW_MFE_STALL_RANK" not in text
    assert "BUY_SKIPPED_LOW_MFE_STALL_DEFER" not in text
    assert "should_defer_day_fbr_fill" not in text
    assert "should_defer_day_htf_fill" not in text
    assert "should_defer_low_mfe_stall_fill" not in text


def test_range_breakout_allowed_by_normal_ranking_not_whitelist(tmp_path: Path):
    db = str(tmp_path / "rb.db")
    _seed(db, "BTC/USDT", SETUP_RANGE_BOUNCE, n=2)
    _seed(db, "ETH/USDT", SETUP_BREAKOUT_CONTINUATION, n=2)
    range_out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_RANGE_BOUNCE,
            "selected_net_expected_value": 0.11,
            "buy_margin": 0.02,
        },
        "BTC/USDT",
        raw_rank_score=0.72,
        buy_margin=0.02,
        db_path=db,
    )
    brk_out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_BREAKOUT_CONTINUATION,
            "selected_net_expected_value": 0.10,
            "buy_margin": 0.02,
        },
        "ETH/USDT",
        raw_rank_score=0.68,
        buy_margin=0.02,
        db_path=db,
    )
    assert range_out["candidate_eligible"] is True
    assert brk_out["candidate_eligible"] is True
    assert should_defer_low_mfe_stall_fill(range_out) is False
    assert should_defer_low_mfe_stall_fill(brk_out) is False


def test_selection_follows_rank_ev_ordering(tmp_path: Path):
    db = str(tmp_path / "ord.db")
    _seed(db, "ETH/USDT", SETUP_HTF_TREND_PULLBACK, n=4)
    demoted = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_HTF_TREND_PULLBACK,
            "selected_net_expected_value": 0.08,
            "buy_margin": 0.02,
        },
        "ETH/USDT",
        raw_rank_score=0.80,
        buy_margin=0.02,
        db_path=db,
    )
    clean = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_RANGE_BOUNCE,
            "selected_net_expected_value": 0.08,
            "buy_margin": 0.02,
        },
        "SOL/USDT",
        raw_rank_score=0.55,
        buy_margin=0.02,
        db_path=db,
    )
    assert demoted["outcome_low_mfe_stall_penalty_applied"] is True
    assert demoted["candidate_eligible"] is True
    assert clean["candidate_eligible"] is True
    # Soft demotion should pull toxic HTF below clean RANGE despite higher raw rank.
    assert float(demoted["final_selection_score"]) < float(clean["final_selection_score"])


def test_soft_penalty_apis_still_active(tmp_path: Path):
    db = str(tmp_path / "api.db")
    _seed(db, "SOL/USDT", SETUP_FAILED_BREAKDOWN_REVERSAL, n=3)
    pen = evaluate_low_mfe_stall_penalty(
        "SOL/USDT", SETUP_FAILED_BREAKDOWN_REVERSAL, "bear", db_path=db
    )
    assert pen["hard_block"] is False
    assert pen["candidate_eligible"] is True
    assert pen.get("applied") is True
    assert should_defer_day_fbr_fill({"setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL}) is False
    assert should_defer_day_htf_fill({"setup_type": SETUP_HTF_TREND_PULLBACK}) is False


def test_explainability_honest_when_htf_locked_in_range():
    out = apply_ml_locked_setup_override(
        {
            "setup_type": SETUP_HTF_TREND_PULLBACK,
            "allweather_setup": SETUP_HTF_TREND_PULLBACK,
            "entry_thesis": SETUP_HTF_TREND_PULLBACK,
            "regime": "range",
            "day_route_regime": "range",
        },
        current_price=42000.0,
        atr=200.0,
    )
    assert out["setup_type_raw"] == SETUP_HTF_TREND_PULLBACK
    assert out["setup_type_canonical"] == SETUP_HTF_TREND_PULLBACK
    assert remap_setup_for_day_regime(SETUP_HTF_TREND_PULLBACK, "range") == SETUP_HTF_TREND_PULLBACK
    assert "RANGE_BOUNCE" not in str(out.get("setup_type"))
