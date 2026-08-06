"""Path A: DAY hard setup fill gates removed — soft demotion only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import backend.services.symbol_setup_outcome_penalty as outcome_pen
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    apply_ml_locked_setup_override,
)
from backend.services.symbol_setup_outcome_penalty import (
    apply_v3_outcome_ranking_to_decision_data,
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


def _seed_stall_dead(db: str, *, symbol: str, setup: str, n: int = 4, pnl: float = -8.0) -> None:
    _ensure_trades_table(db)
    for i in range(n):
        explain = {
            "setup_type": setup,
            "entry_thesis": setup,
            "setup_type_canonical": setup,
            "day_route_regime": "range",
            "mfe_pct": 0.0006,
            "mae_pct": 0.005,
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
                ) VALUES (?, 'SELL', ?, 'STALL_EXIT', ?, 100.0, 99.5, ?, ?, 0)
                """,
                (
                    symbol,
                    pnl - i * 0.2,
                    f"2026-08-04T{10+i:02d}:00:00+00:00",
                    json.dumps(explain),
                    7200.0 + i,
                ),
            )
            conn.commit()


def test_fbr_eligible_soft_demoted(tmp_path: Path):
    db = str(tmp_path / "fbr.db")
    _seed_stall_dead(db, symbol="SOL/USDT", setup=SETUP_FAILED_BREAKDOWN_REVERSAL, n=4)
    out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL,
            "day_route_regime": "bear",
            "selected_net_expected_value": 0.08,
            "buy_margin": 0.02,
        },
        "SOL/USDT",
        raw_rank_score=0.60,
        buy_margin=0.02,
        db_path=db,
    )
    assert out["outcome_penalty_applied"] is True or out["outcome_low_mfe_stall_penalty_applied"] is True
    assert out["hard_block"] is False
    assert out["candidate_eligible"] is True
    assert should_defer_day_fbr_fill(out) is False
    assert out.get("day_fbr_fill_deferred") is False


def test_htf_eligible_soft_demoted(tmp_path: Path):
    db = str(tmp_path / "htf.db")
    _seed_stall_dead(db, symbol="ETH/USDT", setup=SETUP_HTF_TREND_PULLBACK, n=4)
    out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_HTF_TREND_PULLBACK,
            "day_route_regime": "bull",
            "selected_net_expected_value": 0.08,
            "buy_margin": 0.02,
        },
        "ETH/USDT",
        raw_rank_score=0.55,
        buy_margin=0.02,
        db_path=db,
    )
    assert out["outcome_low_mfe_stall_penalty_applied"] is True
    assert out["hard_block"] is False
    assert out["candidate_eligible"] is True
    assert should_defer_day_htf_fill(out) is False
    assert out.get("day_htf_fill_deferred") is False


def test_low_mfe_history_does_not_empty_buy_queue(tmp_path: Path):
    db = str(tmp_path / "queue.db")
    for sym, setup in (
        ("BTC/USDT", SETUP_HTF_TREND_PULLBACK),
        ("ETH/USDT", SETUP_FAILED_BREAKDOWN_REVERSAL),
        ("SOL/USDT", SETUP_RANGE_BOUNCE),
    ):
        _seed_stall_dead(db, symbol=sym, setup=setup, n=3)
    ranked = []
    for sym, setup, raw in (
        ("BTC/USDT", SETUP_HTF_TREND_PULLBACK, 0.50),
        ("ETH/USDT", SETUP_FAILED_BREAKDOWN_REVERSAL, 0.48),
        ("SOL/USDT", SETUP_RANGE_BOUNCE, 0.70),
    ):
        out = apply_v3_outcome_ranking_to_decision_data(
            {
                "setup_type": setup,
                "day_route_regime": "range",
                "selected_net_expected_value": 0.09,
                "buy_margin": 0.02,
            },
            sym,
            raw_rank_score=raw,
            buy_margin=0.02,
            db_path=db,
        )
        assert out["candidate_eligible"] is True
        assert should_defer_low_mfe_stall_fill(out) is False
        ranked.append((sym, float(out["final_selection_score"]), out))
    ranked.sort(key=lambda x: x[1], reverse=True)
    assert len(ranked) == 3
    assert ranked[0][0] in ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    assert all(r[2].get("low_mfe_stall_fill_deferred") is False for r in ranked)


def test_negative_fss_not_a_setup_defer_gate(tmp_path: Path):
    db = str(tmp_path / "neg.db")
    _seed_stall_dead(db, symbol="XRP/USDT", setup=SETUP_HTF_TREND_PULLBACK, n=5, pnl=-10.0)
    out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_HTF_TREND_PULLBACK,
            "day_route_regime": "range",
            "selected_net_expected_value": 0.07,
            "buy_margin": 0.02,
        },
        "XRP/USDT",
        raw_rank_score=0.52,
        buy_margin=0.02,
        db_path=db,
    )
    assert float(out["final_selection_score"]) < 0.0
    assert out["candidate_eligible"] is True
    assert should_defer_low_mfe_stall_fill(out) is False


def test_no_preferred_fill_whitelist():
    assert not hasattr(outcome_pen, "DAY_PREFERRED_FILL_SETUPS")


def test_remapped_htf_does_not_pretend_native_range():
    out = apply_ml_locked_setup_override(
        {
            "setup_type": SETUP_HTF_TREND_PULLBACK,
            "entry_thesis": SETUP_HTF_TREND_PULLBACK,
            "day_route_regime": "range",
            "regime": "range",
        },
        current_price=100.0,
        atr=1.0,
    )
    assert out["setup_type"] == SETUP_HTF_TREND_PULLBACK
    assert out["setup_type_canonical"] == SETUP_HTF_TREND_PULLBACK
    assert out["setup_type_raw"] == SETUP_HTF_TREND_PULLBACK
    assert out["setup_type"] != SETUP_RANGE_BOUNCE


def test_outcome_penalty_remains_non_blocking(tmp_path: Path):
    db = str(tmp_path / "soft.db")
    _seed_stall_dead(db, symbol="BTC/USDT", setup=SETUP_BREAKOUT_CONTINUATION, n=3)
    out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": SETUP_BREAKOUT_CONTINUATION,
            "day_route_regime": "bull",
            "selected_net_expected_value": 0.10,
            "buy_margin": 0.02,
        },
        "BTC/USDT",
        raw_rank_score=0.65,
        buy_margin=0.02,
        db_path=db,
    )
    assert out["hard_block"] is False
    assert out["candidate_eligible"] is True
