"""DAY promote/kill Thompson bandit — winners rise, losers starve."""

from __future__ import annotations

import random
from pathlib import Path

from backend.services.day_outcome_bandit import (
    apply_bandit_to_decision_data,
    arm_key,
    get_arm_stats,
    record_bandit_outcome,
    sample_arm,
)
from backend.services.day_trade_thesis import (
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
)


def test_win_promotes_arm_mean(tmp_path: Path):
    db = str(tmp_path / "b.db")
    for _ in range(5):
        record_bandit_outcome(
            symbol="BTC/USDT",
            setup=SETUP_RANGE_BOUNCE,
            regime="range",
            pnl_usd=9.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    st = get_arm_stats("BTC/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db)
    assert st["wins"] == 5
    assert st["mean"] > 0.7
    assert st["starved"] is False


def test_losses_starve_arm(tmp_path: Path):
    db = str(tmp_path / "s.db")
    for _ in range(5):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup=SETUP_HTF_TREND_PULLBACK,
            regime="bull",
            pnl_usd=-8.0,
            exit_reason="STALL_EXIT",
            db_path=db,
        )
    st = get_arm_stats("ETH/USDT", SETUP_HTF_TREND_PULLBACK, "bull", db_path=db)
    assert st["losses"] == 5
    assert st["mean"] < 0.35
    assert st["starved"] is True
    samp = sample_arm("ETH/USDT", SETUP_HTF_TREND_PULLBACK, "bull", db_path=db, rng=random.Random(1))
    assert samp["size_factor"] <= 0.15
    assert samp["hard_block"] is False
    assert samp["candidate_eligible"] is True


def test_winner_outranks_starved_loser(tmp_path: Path):
    db = str(tmp_path / "r.db")
    for _ in range(4):
        record_bandit_outcome(
            symbol="SOL/USDT",
            setup=SETUP_RANGE_BOUNCE,
            regime="range",
            pnl_usd=10.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup=SETUP_FAILED_BREAKDOWN_REVERSAL,
            regime="bear",
            pnl_usd=-9.0,
            exit_reason="STALL_EXIT",
            db_path=db,
        )
    rng = random.Random(42)
    good = apply_bandit_to_decision_data(
        {
            "setup_type": SETUP_RANGE_BOUNCE,
            "day_route_regime": "range",
            "final_selection_score": 0.1,
        },
        "SOL/USDT",
        db_path=db,
        rng=rng,
    )
    bad = apply_bandit_to_decision_data(
        {
            "setup_type": SETUP_FAILED_BREAKDOWN_REVERSAL,
            "day_route_regime": "bear",
            "final_selection_score": 0.9,  # high soft score must not beat bandit starve
        },
        "ETH/USDT",
        db_path=db,
        rng=rng,
    )
    assert good["candidate_eligible"] is True
    assert bad["candidate_eligible"] is True
    assert bad["day_bandit_starved"] is True
    assert float(good["day_bandit_score"]) > float(bad["day_bandit_score"])
    assert float(good["final_selection_score"]) > float(bad["final_selection_score"])


def test_arm_key_normalizes_aliases():
    assert arm_key("BTCUSDT", "TREND_PULLBACK", "trending_up") == arm_key("BTC/USDT", "HTF_TREND_PULLBACK", "bull")


def test_no_hard_block_on_toxic(tmp_path: Path):
    db = str(tmp_path / "t.db")
    for _ in range(6):
        record_bandit_outcome(
            symbol="XRP/USDT",
            setup=SETUP_HTF_TREND_PULLBACK,
            regime="range",
            pnl_usd=-11.0,
            exit_reason="STALL_EXIT",
            db_path=db,
        )
    out = apply_bandit_to_decision_data(
        {"setup_type": SETUP_HTF_TREND_PULLBACK, "day_route_regime": "range", "final_selection_score": -0.2},
        "XRP/USDT",
        db_path=db,
        rng=random.Random(7),
    )
    assert out["hard_block"] is False
    assert out["candidate_eligible"] is True
    assert out["day_bandit_starved"] is True
