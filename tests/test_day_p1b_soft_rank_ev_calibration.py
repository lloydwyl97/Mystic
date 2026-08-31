"""P1B: stronger soft rank/EV calibration for low-MFE dead-trade history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.services.day_trade_thesis import SETUP_RANGE_BOUNCE
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
    display_exit: str | None = None,
) -> None:
    display = display_exit or ("STALL_EXIT" if "STALL" in raw_exit.upper() else raw_exit.split("_EXIT", maxsplit=1)[0] + "_EXIT" if "_EXIT" in raw_exit else raw_exit)
    if "GIVEBACK" in raw_exit.upper():
        display = "GIVEBACK_EXIT"
    if "NET_PROFIT" in raw_exit.upper():
        display = "NET_PROFIT_EXIT"
    explain = {
        "setup_type": setup,
        "entry_thesis": setup,
        "setup_type_canonical": setup,
        "day_route_regime": "range",
        "selected_net_expected_value": 0.07,
        "final_selection_score": 0.33,
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


def _seed_stall_dead(
    db: str,
    *,
    symbol: str,
    setup: str,
    n: int,
    mfe: float = 0.0008,
    mae: float = 0.0045,
    pnl: float = -6.0,
) -> None:
    _ensure_trades_table(db)
    for i in range(n):
        _insert_sell(
            db,
            symbol=symbol,
            setup=setup,
            pnl=pnl - i * 0.1,
            raw_exit="STALL_EXIT_DEAD_NO_MFE",
            mfe=mfe,
            mae=mae,
            hold_sec=7200.0 + i,
            ts=f"2026-08-04T{10 + i:02d}:00:00+00:00",
        )


def test_1_repeated_stall_dead_strengthens_penalty(tmp_path: Path):
    db2 = str(tmp_path / "c2.db")
    db4 = str(tmp_path / "c4.db")
    db6 = str(tmp_path / "c6.db")
    _seed_stall_dead(db2, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=2)
    _seed_stall_dead(db4, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=4)
    _seed_stall_dead(db6, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=6)

    p2 = evaluate_low_mfe_stall_penalty("ETH/USDT", "HTF_TREND_PULLBACK", "range", db_path=db2)
    p4 = evaluate_low_mfe_stall_penalty("ETH/USDT", "HTF_TREND_PULLBACK", "range", db_path=db4)
    p6 = evaluate_low_mfe_stall_penalty("ETH/USDT", "HTF_TREND_PULLBACK", "range", db_path=db6)

    assert p2["applied"] and p4["applied"] and p6["applied"]
    for p in (p2, p4, p6):
        assert p["hard_block"] is False
        assert p["candidate_eligible"] is True
        assert p["size_factor"] == 1.0
        assert p["rank_delta"] < 0
        assert p["ev_factor"] < 1.0

    assert p2["rank_delta"] > p4["rank_delta"]  # less negative than count-4
    # Count-6 is strictly stronger on at least one demotion axis, or both sit at P1C floors.
    at_floor = p4["rank_delta"] <= -0.74 and p6["rank_delta"] <= -0.74 and p4["ev_factor"] <= 0.23 and p6["ev_factor"] <= 0.23
    assert at_floor or (p4["rank_delta"] > p6["rank_delta"]) or (p4["ev_factor"] > p6["ev_factor"])
    assert p2["ev_factor"] > p4["ev_factor"]
    assert p4["ev_factor"] >= p6["ev_factor"]
    assert p2["low_mfe_stall_count"] == 2
    assert p4["low_mfe_stall_count"] == 4
    assert p6["low_mfe_stall_count"] == 6


def test_2_symbol_setup_bucket_stronger_than_setup_only(tmp_path: Path):
    db = str(tmp_path / "pair.db")
    _ensure_trades_table(db)
    # BTC/RANGE bad history
    _seed_stall_dead(db, symbol="BTC/USDT", setup=SETUP_RANGE_BOUNCE, n=3, mfe=0.0005, mae=0.005)
    # SOL/RANGE clean (one win) — should not inherit BTC damage as symbol/setup
    _insert_sell(
        db,
        symbol="SOL/USDT",
        setup=SETUP_RANGE_BOUNCE,
        pnl=8.0,
        raw_exit="NET_PROFIT_EXIT",
        mfe=0.005,
        mae=0.001,
        ts="2026-08-04T20:00:00+00:00",
    )

    btc = evaluate_low_mfe_stall_penalty("BTC/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db)
    sol = evaluate_low_mfe_stall_penalty("SOL/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db)

    assert btc["applied"] is True
    assert btc["rank_delta"] < -0.15
    # SOL may get mild setup-wide cluster demotion, but weaker than BTC pair bucket.
    if sol["applied"]:
        assert sol["rank_delta"] > btc["rank_delta"]  # less negative
        assert sol["ev_factor"] >= btc["ev_factor"]
    else:
        assert sol["applied"] is False


def test_3_htf_setup_cluster_applies_setup_level_penalty(tmp_path: Path):
    db = str(tmp_path / "htf.db")
    _ensure_trades_table(db)
    # Spread STALL_DEAD across XRP/ETH HTF → setup cluster
    _seed_stall_dead(db, symbol="XRP/USDT", setup="HTF_TREND_PULLBACK", n=2, mfe=0.0007)
    _seed_stall_dead(db, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=2, mfe=0.0006)
    # SOL HTF has no own pair stalls — still demoted via setup cluster
    sol = evaluate_low_mfe_stall_penalty("SOL/USDT", "HTF_TREND_PULLBACK", "chop", db_path=db)
    assert sol["applied"] is True
    assert sol.get("setup_cluster_applied") is True
    assert sol["hard_block"] is False
    assert sol["candidate_eligible"] is True
    assert sol["rank_delta"] < 0
    assert sol["ev_factor"] < 1.0

    dd = {
        "setup_type": "HTF_TREND_PULLBACK",
        "day_route_regime": "chop",
        "selected_net_expected_value": 0.12,
        "buy_margin": 0.10,
    }
    out = apply_v3_outcome_ranking_to_decision_data(dd, "SOL/USDT", raw_rank_score=0.60, buy_margin=0.10, db_path=db)
    assert out["outcome_penalty_applied"] is True
    assert float(out["final_selection_score"]) < float(out["final_selection_score_before_outcome_penalty"])


def test_4_profitable_bucket_softens_penalty(tmp_path: Path):
    db_bad = str(tmp_path / "bad.db")
    db_good = str(tmp_path / "good.db")
    _ensure_trades_table(db_bad)
    _ensure_trades_table(db_good)

    # Bad: only stalls
    _seed_stall_dead(db_bad, symbol="BTC/USDT", setup=SETUP_RANGE_BOUNCE, n=2, pnl=-8.0)

    # Good: 2 stalls but many NET_PROFIT → PF > 1.25 and net positive
    _seed_stall_dead(db_good, symbol="BTC/USDT", setup=SETUP_RANGE_BOUNCE, n=2, pnl=-5.0)
    for i, pnl in enumerate((11.0, 10.0, 9.0, 8.0)):
        _insert_sell(
            db_good,
            symbol="BTC/USDT",
            setup=SETUP_RANGE_BOUNCE,
            pnl=pnl,
            raw_exit="NET_PROFIT_EXIT",
            mfe=0.005,
            mae=0.001,
            ts=f"2026-08-04T{18 + i:02d}:00:00+00:00",
        )

    bad = evaluate_low_mfe_stall_penalty("BTC/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db_bad)
    good = evaluate_low_mfe_stall_penalty("BTC/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db_good)
    assert bad["applied"] and good["applied"]
    assert good["bucket_profit_factor"] > 1.25
    assert good["bucket_net_pnl"] > 0
    assert good["rank_delta"] > bad["rank_delta"]  # softened (less negative)
    assert good["ev_factor"] >= bad["ev_factor"]
    assert good.get("soften", 1.0) < 1.0


def test_5_giveback_secondary_not_stall_severity(tmp_path: Path):
    db_gb = str(tmp_path / "gb.db")
    db_stall = str(tmp_path / "stall.db")
    _ensure_trades_table(db_gb)
    _ensure_trades_table(db_stall)

    for i in range(3):
        _insert_sell(
            db_gb,
            symbol="XRP/USDT",
            setup="HTF_TREND_PULLBACK",
            pnl=-2.2,
            raw_exit="GIVEBACK_EXIT",
            mfe=0.0028,  # < 0.35%
            mae=0.0015,
            hold_sec=5000,
            ts=f"2026-08-04T1{i}:00:00+00:00",
        )
    _seed_stall_dead(db_stall, symbol="XRP/USDT", setup="HTF_TREND_PULLBACK", n=3, mfe=0.0005)

    gb = evaluate_low_mfe_stall_penalty("XRP/USDT", "HTF_TREND_PULLBACK", "range", db_path=db_gb)
    stall = evaluate_low_mfe_stall_penalty("XRP/USDT", "HTF_TREND_PULLBACK", "range", db_path=db_stall)
    assert gb["applied"] is True
    assert stall["applied"] is True
    assert gb["low_mfe_stall_count"] == 0
    assert gb["giveback_weak_count"] >= 2
    assert gb["rank_delta"] > stall["rank_delta"]  # milder than stall-dead
    assert gb["ev_factor"] > stall["ev_factor"]
    assert gb["rank_delta"] >= -0.15  # secondary cap region


def test_6_candidate_remains_executable(tmp_path: Path):
    db = str(tmp_path / "exec.db")
    _seed_stall_dead(db, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=4, mfe=0.0004, mae=0.006)
    dd = {
        "setup_type": "HTF_TREND_PULLBACK",
        "day_route_regime": "bear",
        "selected_net_expected_value": 0.09,
        "buy_margin": 0.11,
    }
    out = apply_v3_outcome_ranking_to_decision_data(dd, "ETH/USDT", raw_rank_score=0.55, buy_margin=0.11, db_path=db)
    assert out["outcome_penalty_applied"] is True
    assert out["candidate_eligible"] is True
    assert out["outcome_penalty_hard_block"] is False
    assert out.get("hard_block") is False
    assert "reject" not in str(out.get("penalty_reason") or "").lower()
    assert out.get("thesis_size_factor") in (None, 1.0) or float(out.get("thesis_size_factor") or 1.0) == pytest.approx(1.0)
    assert out["final_selection_score"] is not None


def test_7_explainability_fields_stamped(tmp_path: Path):
    db = str(tmp_path / "expl.db")
    _seed_stall_dead(db, symbol="ETH/USDT", setup="HTF_TREND_PULLBACK", n=3)
    dd = {
        "setup_type": "HTF_TREND_PULLBACK",
        "day_route_regime": "range",
        "selected_net_expected_value": 0.10,
        "buy_margin": 0.12,
    }
    out = apply_v3_outcome_ranking_to_decision_data(dd, "ETH/USDT", raw_rank_score=0.50, buy_margin=0.12, db_path=db)
    assert out["outcome_penalty_applied"] is True
    assert out.get("penalty_reason")
    assert "repeated_low_mfe_stall_losses" in str(out["penalty_reason"])
    assert out.get("low_mfe_stall_count", 0) >= 2
    assert out.get("bucket_net_pnl") is not None
    assert out.get("bucket_profit_factor") is not None
    assert out.get("outcome_penalty_rank_delta") is not None
    assert out.get("outcome_penalty_ev_factor") is not None
    assert out.get("rank_score_before_outcome_penalty") == pytest.approx(0.50)
    assert out.get("selected_net_expected_value_before_outcome_penalty") == pytest.approx(0.10)
    assert out.get("final_selection_score_before_outcome_penalty") is not None
    assert out["hard_block"] is False
    assert out["candidate_eligible"] is True
    eval_obj = out.get("outcome_low_mfe_stall_penalty_eval") or {}
    assert eval_obj.get("penalty_generation") == "low_mfe_stall_p1c"
    assert eval_obj.get("hard_block") is False
