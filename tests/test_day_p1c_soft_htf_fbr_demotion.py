"""P1C: stronger soft HTF/FBR demotion (hard fill defer retired)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.services.symbol_setup_outcome_penalty import (
    apply_v3_outcome_ranking_to_decision_data,
    evaluate_low_mfe_stall_penalty,
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


def _insert_sell(
    db: str,
    *,
    symbol: str,
    setup: str,
    pnl: float,
    raw_exit: str,
    mfe: float,
    mae: float,
    hold_sec: float = 7200.0,
    ts: str = "2026-08-04T12:00:00+00:00",
) -> None:
    display = "STALL_EXIT" if "STALL" in raw_exit.upper() else raw_exit
    if "NET_PROFIT" in raw_exit.upper():
        display = "NET_PROFIT_EXIT"
    explain = {
        "setup_type": setup,
        "entry_thesis": setup,
        "setup_type_canonical": setup,
        "day_route_regime": "range",
        "selected_net_expected_value": 0.07,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "raw_exit_reason": raw_exit,
        "canonical_exit_reason": display,
        "dead_trade_reason": "DEAD_NO_MFE" if "DEAD" in raw_exit.upper() else None,
    }
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (
                symbol, side, pnl, exit_reason, timestamp, entry_price, price,
                explainability_json, hold_time_seconds, is_synthetic
            ) VALUES (?, 'SELL', ?, ?, ?, 100.0, 99.5, ?, ?, 0)
            """,
            (symbol, pnl, display, ts, json.dumps(explain), hold_sec),
        )
        conn.commit()


def _seed_stall_dead(db: str, *, symbol: str, setup: str, n: int, pnl: float = -7.0) -> None:
    _ensure_trades_table(db)
    for i in range(n):
        _insert_sell(
            db,
            symbol=symbol,
            setup=setup,
            pnl=pnl - i * 0.2,
            raw_exit="STALL_EXIT_DEAD_NO_MFE",
            mfe=0.0006,
            mae=0.005,
            hold_sec=7200.0 + i,
            ts=f"2026-08-04T{10+i:02d}:00:00+00:00",
        )


def test_p1c_eth_htf_stronger_than_mild_bucket(tmp_path: Path):
    db = str(tmp_path / "eth.db")
    _seed_stall_dead(db, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=3, pnl=-8.0)
    pen = evaluate_low_mfe_stall_penalty("ETH/USDT", "HTF_TREND_PULLBACK", "bull", db_path=db)
    assert pen["applied"] is True
    assert pen["hard_block"] is False
    assert pen["candidate_eligible"] is True
    assert pen["size_factor"] == 1.0
    assert pen["penalty_generation"] == "low_mfe_stall_p1c"
    assert pen["rank_delta"] <= -0.30
    assert pen["ev_factor"] <= 0.50
    assert pen.get("toxic_setup_boost") is True


def test_p1c_xrp_fbr_toxic_boost(tmp_path: Path):
    db = str(tmp_path / "fbr.db")
    _seed_stall_dead(db, symbol="XRP/USDT", setup="FAILED_BREAKDOWN_REVERSAL", n=4, pnl=-12.0)
    pen = evaluate_low_mfe_stall_penalty(
        "XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=db
    )
    assert pen["applied"] is True
    assert pen["rank_delta"] <= -0.40
    assert pen["ev_factor"] <= 0.40
    assert pen["hard_block"] is False


def test_p1c_severe_bleed_not_softened_by_latest3(tmp_path: Path):
    db_soft = str(tmp_path / "soft.db")
    db_hard = str(tmp_path / "hard.db")
    # 3 stalls then 3 wins — old P1B softened via latest_3; P1C must not for severe bleed
    _seed_stall_dead(db_soft, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=3, pnl=-6.0)
    for i, pnl in enumerate((5.0, 6.0, 7.0)):
        _insert_sell(
            db_soft,
            symbol="ETH/USDT",
            setup="HTF_TREND_PULLBACK",
            pnl=pnl,
            raw_exit="NET_PROFIT_EXIT",
            mfe=0.005,
            mae=0.001,
            ts=f"2026-08-04T{18+i:02d}:00:00+00:00",
        )
    _seed_stall_dead(db_hard, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=3, pnl=-6.0)

    soft = evaluate_low_mfe_stall_penalty("ETH/USDT", "HTF_TREND_PULLBACK", "range", db_path=db_soft)
    hard = evaluate_low_mfe_stall_penalty("ETH/USDT", "HTF_TREND_PULLBACK", "range", db_path=db_hard)
    assert soft["applied"] and hard["applied"]
    # Soft bucket is net positive overall so PF soften may apply, but latest_3 alone
    # must not erase toxic demotion when stalls remain in lookback.
    assert soft["rank_delta"] < -0.05
    assert hard["rank_delta"] <= soft["rank_delta"] or hard["ev_factor"] <= soft["ev_factor"]


def test_p1c_negative_fss_toxic_still_eligible_no_hard_defer(tmp_path: Path):
    db = str(tmp_path / "defer.db")
    _seed_stall_dead(db, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=4, pnl=-9.0)
    out = apply_v3_outcome_ranking_to_decision_data(
        {
            "setup_type": "HTF_TREND_PULLBACK",
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
    assert out["candidate_eligible"] is True
    assert out["hard_block"] is False
    assert float(out["final_selection_score"]) < 0.0


def test_p1c_trend_pullback_aliases_htf(tmp_path: Path):
    db = str(tmp_path / "alias.db")
    _seed_stall_dead(db, symbol="BTC/USDT", setup="TREND_PULLBACK", n=2)
    pen = evaluate_low_mfe_stall_penalty("BTC/USDT", "TREND_PULLBACK", "bull", db_path=db)
    assert pen["applied"] is True
    assert pen["setup"] == "HTF_TREND_PULLBACK"
