"""Batch 1: DAY bandit observability + hygiene.

Guards:
1. TradeExplainability round-trips DAY bandit fields through to_dict() so
   paper_trades.explainability_json can be replayed offline.
2. _stamp_day_bandit_explain copies alpha/β/mean/starved/size/score from
   decision_data onto the dataclass exactly as apply_bandit_to_decision_data
   would have written them.
3. bootstrap_bandit_from_paper_trades ignores legacy sells whose setup label
   is empty or 'UNKNOWN'.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.services.day_outcome_bandit import (
    bootstrap_bandit_from_paper_trades,
    ensure_bandit_schema,
)
from backend.services.portfolio_engine import (
    TradeExplainability,
    _stamp_day_bandit_explain,
)


def test_explainability_class_carries_day_bandit_fields():
    ex = TradeExplainability(trade_id="t1", symbol="BTC/USDT", side="BUY", timestamp="ts")
    ex.day_bandit_enabled = True
    ex.day_bandit_arm_key = "BTC/USDT|HTF_TREND_PULLBACK|range"
    ex.day_bandit_alpha = 10.4
    ex.day_bandit_beta = 4.5
    ex.day_bandit_mean = 0.699
    ex.day_bandit_sample = 0.71
    ex.day_bandit_score = 0.577
    ex.day_bandit_n_obs = 8
    ex.day_bandit_wins = 5
    ex.day_bandit_losses = 3
    ex.day_bandit_total_pnl = 47.56
    ex.day_bandit_starved = False
    ex.day_bandit_size_factor = 1.24
    ex.final_selection_score_pre_bandit = 0.42

    payload = ex.to_dict()
    assert payload["day_bandit_enabled"] is True
    assert payload["day_bandit_arm_key"] == "BTC/USDT|HTF_TREND_PULLBACK|range"
    assert payload["day_bandit_alpha"] == 10.4
    assert payload["day_bandit_beta"] == 4.5
    assert payload["day_bandit_mean"] == 0.699
    assert payload["day_bandit_sample"] == 0.71
    assert payload["day_bandit_score"] == 0.577
    assert payload["day_bandit_n_obs"] == 8
    assert payload["day_bandit_wins"] == 5
    assert payload["day_bandit_losses"] == 3
    assert payload["day_bandit_total_pnl"] == 47.56
    assert payload["day_bandit_starved"] is False
    assert payload["day_bandit_size_factor"] == 1.24
    assert payload["final_selection_score_pre_bandit"] == 0.42
    # Round-trip json must not raise
    json.dumps(payload)


def test_stamp_day_bandit_explain_copies_all_fields():
    ex = TradeExplainability(trade_id="t2", symbol="ETH/USDT", side="BUY", timestamp="ts")
    dd = {
        "day_bandit_enabled": True,
        "day_bandit_arm_key": "ETH/USDT|HTF_TREND_PULLBACK|range",
        "day_bandit_alpha": 3.906,
        "day_bandit_beta": 14.873,
        "day_bandit_mean": 0.208,
        "day_bandit_sample": 0.11,
        "day_bandit_score": 0.06,
        "day_bandit_n_obs": 11,
        "day_bandit_wins": 2,
        "day_bandit_losses": 9,
        "day_bandit_total_pnl": -47.60,
        "day_bandit_starved": True,
        "day_bandit_size_factor": 0.08,
        "final_selection_score_pre_bandit": -0.31,
    }
    _stamp_day_bandit_explain(ex, dd)
    assert ex.day_bandit_enabled is True
    assert ex.day_bandit_arm_key == "ETH/USDT|HTF_TREND_PULLBACK|range"
    assert ex.day_bandit_alpha == 3.906
    assert ex.day_bandit_beta == 14.873
    assert ex.day_bandit_mean == 0.208
    assert ex.day_bandit_starved is True
    assert ex.day_bandit_size_factor == 0.08
    assert ex.day_bandit_n_obs == 11
    assert ex.day_bandit_wins == 2
    assert ex.day_bandit_losses == 9
    assert ex.final_selection_score_pre_bandit == -0.31


def test_stamp_day_bandit_explain_tolerates_missing_dd():
    ex = TradeExplainability(trade_id="t3", symbol="SOL/USDT", side="BUY", timestamp="ts")
    _stamp_day_bandit_explain(ex, None)
    assert ex.day_bandit_enabled is False
    assert ex.day_bandit_arm_key == ""
    assert ex.day_bandit_alpha is None
    assert ex.day_bandit_starved is False


def test_bootstrap_skips_unknown_setup_labels(tmp_path: Path):
    db = tmp_path / "boot.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                pnl REAL,
                exit_reason TEXT,
                strategy_id TEXT,
                explainability_json TEXT
            );
            """
        )
        rows = [
            (
                "BTC/USDT",
                12.0,
                "NET_PROFIT_EXIT",
                "day",
                json.dumps({"setup_type_canonical": "HTF_TREND_PULLBACK", "day_route_regime": "range"}),
            ),
            (
                "ETH/USDT",
                -5.0,
                "STALL_EXIT",
                "day",
                json.dumps({"setup_type": "UNKNOWN", "day_route_regime": "range"}),
            ),
            (
                "SOL/USDT",
                -3.0,
                "STALL_EXIT",
                "day",
                json.dumps({}),  # missing setup entirely
            ),
            (
                "XRP/USDT",
                8.0,
                "NET_PROFIT_EXIT",
                "day",
                json.dumps({"setup_type_canonical": "RANGE_BOUNCE", "day_route_regime": "range"}),
            ),
        ]
        for sym, pnl, exit_r, strat, ex in rows:
            conn.execute(
                "INSERT INTO paper_trades (symbol, side, pnl, exit_reason, strategy_id, explainability_json) VALUES (?, 'SELL', ?, ?, ?, ?)",
                (sym, pnl, exit_r, strat, ex),
            )
        conn.commit()

    ensure_bandit_schema(db)
    count = bootstrap_bandit_from_paper_trades(db, lookback=100)
    assert count == 2

    with sqlite3.connect(db) as conn:
        arms = conn.execute("SELECT setup FROM day_outcome_bandit_arms ORDER BY setup").fetchall()
    setups = sorted(r[0] for r in arms)
    assert setups == ["HTF_TREND_PULLBACK", "RANGE_BOUNCE"]
    assert "UNKNOWN" not in setups
