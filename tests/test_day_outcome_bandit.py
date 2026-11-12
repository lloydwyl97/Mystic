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


def test_restart_reloads_decayed_posterior_not_raw_counts(tmp_path: Path):
    import sqlite3

    from backend.services.day_outcome_bandit import ensure_bandit_schema

    db = str(tmp_path / "persist.db")
    ensure_bandit_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO day_outcome_bandit_arms (
                arm_key, symbol, setup, regime, alpha, beta, wins, losses, total_pnl,
                last_pnl, last_exit_reason, last_updated, n_obs, clean_terminal_n
            ) VALUES ('BTC/USDT|HTF_TREND_PULLBACK|bull', 'BTC/USDT', 'HTF_TREND_PULLBACK', 'bull',
                      1.2, 12.5, 14, 300, -40, -1, 'STALL', 1, 300, 0)
            """
        )
        conn.commit()
    st = get_arm_stats("BTC/USDT", "HTF_TREND_PULLBACK", "bull", db_path=db)
    assert st["alpha"] == 1.2
    assert st["beta"] == 12.5
    assert abs(st["mean"] - (1.2 / 13.7)) < 1e-9
    assert st["wins"] == 14 and st["losses"] == 300
    assert st["clean_terminal_n"] == 0


def test_legacy_count_does_not_unlock_the_size_ceiling(tmp_path: Path):
    import sqlite3

    from backend.services.day_outcome_bandit import MAX_SIZE_FRESH, ensure_bandit_schema

    db = str(tmp_path / "size.db")
    ensure_bandit_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO day_outcome_bandit_arms (
                arm_key, symbol, setup, regime, alpha, beta, wins, losses, total_pnl,
                last_pnl, last_exit_reason, last_updated, n_obs, clean_terminal_n
            ) VALUES ('ETH/USDT|BREAKOUT_CONTINUATION|bull', 'ETH/USDT', 'BREAKOUT_CONTINUATION', 'bull',
                      9.4, 4.1, 5, 3, 10, 1, 'NET_PROFIT', 1, 300, 0)
            """
        )
        conn.commit()
    legacy = sample_arm("ETH/USDT", "BREAKOUT_CONTINUATION", "bull", db_path=db, rng=random.Random(1))
    assert legacy["n_obs"] == 300
    assert legacy["clean_terminal_n"] == 0
    assert legacy["size_factor_cap"] == "fresh"
    assert legacy["size_factor"] <= MAX_SIZE_FRESH + 1e-6
    for _ in range(8):
        record_bandit_outcome(symbol="ETH/USDT", setup="BREAKOUT_CONTINUATION", regime="bull", pnl_usd=12.0, exit_reason="NET_PROFIT_EXIT", db_path=db)
    clean = sample_arm("ETH/USDT", "BREAKOUT_CONTINUATION", "bull", db_path=db, rng=random.Random(1))
    assert clean["clean_terminal_n"] == 8
    assert clean["size_factor_cap"] == "top"


def test_starved_arm_loses_every_seeded_comparison(tmp_path: Path):
    import sqlite3

    from backend.services.day_outcome_bandit import apply_bandit_to_decision_data, ensure_bandit_schema

    db = str(tmp_path / "starve.db")
    ensure_bandit_schema(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO day_outcome_bandit_arms (
                arm_key, symbol, setup, regime, alpha, beta, wins, losses, total_pnl,
                last_pnl, last_exit_reason, last_updated, n_obs, clean_terminal_n
            ) VALUES
            ('ETH/USDT|UNKNOWN|bull', 'ETH/USDT', 'UNKNOWN', 'bull', 1.0, 12.58, 14, 314, -20, -1, 'STALL', 1, 328, 0),
            ('BTC/USDT|RANGE_BOUNCE|range', 'BTC/USDT', 'RANGE_BOUNCE', 'range', 2.0, 2.0, 4, 4, 1, 1, 'NET_PROFIT', 1, 8, 8)
            """
        )
        conn.commit()
    wins = 0
    for seed in range(200):
        low = apply_bandit_to_decision_data({"final_selection_score": 50.0, "setup_type": "UNKNOWN", "day_route_regime": "bull"}, "ETH/USDT", db_path=db, rng=random.Random(seed))
        other = apply_bandit_to_decision_data({"final_selection_score": 0.0, "setup_type": "RANGE_BOUNCE", "day_route_regime": "range"}, "BTC/USDT", db_path=db, rng=random.Random(seed))
        if float(low["final_selection_score"]) > float(other["final_selection_score"]):
            wins += 1
    assert wins == 0


def test_three_sell_chunks_are_one_bandit_observation(tmp_path: Path):
    from backend.services.portfolio_engine import PortfolioEngine

    db = str(tmp_path / "chunks.db")

    class _Book:
        db_path = db
        coin_performance: dict = {}

    chunks = []
    for pnl in (-2.0, 4.0, 5.0):
        chunks.append(
            {
                "learn": {"engine": "DAY_V2", "symbol": "SOL/USDT", "setup": "HTF_TREND_PULLBACK", "regime": "bull", "exit_reason": "NET_PROFIT_EXIT"},
                "facts": {"net_profit_usd": pnl, "entry_price": 100.0, "quantity": 1.0, "highest_price": 101.0, "lowest_price": 99.0, "entry_timestamp": 1.0, "exit_timestamp": 2.0},
            }
        )
    PortfolioEngine._apply_completed_trade_quality(_Book(), "trade-1", chunks)
    st = get_arm_stats("SOL/USDT", "HTF_TREND_PULLBACK", "bull", db_path=db)
    assert st["n_obs"] == 1
    assert st["clean_terminal_n"] == 1
    assert st["total_pnl"] == 7.0


def test_observation_count_keeps_every_update(tmp_path: Path):
    from backend.services.day_outcome_bandit import SLIDING_WINDOW

    db = str(tmp_path / "decay.db")
    for _ in range(SLIDING_WINDOW + 1):
        record_bandit_outcome(symbol="BTC/USDT", setup=SETUP_RANGE_BOUNCE, regime="range", pnl_usd=1.0, exit_reason="NET_PROFIT_EXIT", db_path=db)
    st = get_arm_stats("BTC/USDT", SETUP_RANGE_BOUNCE, "range", db_path=db)
    assert st["n_obs"] == SLIDING_WINDOW + 1
    assert st["wins"] == SLIDING_WINDOW + 1


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
