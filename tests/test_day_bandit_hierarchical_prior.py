"""Batch 2: hierarchical (partial-pooling) prior + deeper bootstrap.

- A fresh (symbol, setup, regime) arm should inherit peer evidence when
  other symbols already have observations under the same (setup, regime).
- Peer inheritance must be capped so it never overwhelms Beta(1,1).
- Environment variable DAY_BANDIT_HIERARCHICAL_PRIOR_ENABLED=false disables it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from backend.services.day_outcome_bandit import (
    apply_bandit_to_decision_data,
    get_arm_stats,
    record_bandit_outcome,
    sample_arm,
)


def _fresh_db(tmp_path: Path, name: str = "b.db") -> str:
    return str(tmp_path / name)


def test_fresh_arm_uniform_when_no_peers(tmp_path: Path):
    db = _fresh_db(tmp_path, "no_peers.db")
    st = get_arm_stats("BTC/USDT", "HTF_TREND_PULLBACK", "range", db_path=db)
    assert st["n_obs"] == 0
    assert st["mean"] == pytest.approx(0.5, abs=1e-6)
    assert st["prior_source"] == "uniform"
    assert st["peer_n_obs"] == 0
    samp = sample_arm(
        "BTC/USDT", "HTF_TREND_PULLBACK", "range",
        db_path=db, rng=random.Random(0),
    )
    assert samp["size_factor"] == pytest.approx(1.0, abs=1e-6)


def test_hierarchical_prior_shifts_fresh_arm_toward_peer_evidence(tmp_path: Path):
    db = _fresh_db(tmp_path, "hier.db")
    for _ in range(8):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="BREAKOUT_CONTINUATION",
            regime="range",
            pnl_usd=10.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    # Fresh SOL arm inherits ETH's BREAKOUT_CONTINUATION|range evidence
    st = get_arm_stats("SOL/USDT", "BREAKOUT_CONTINUATION", "range", db_path=db)
    assert st["n_obs"] == 0
    assert st["prior_source"] == "hierarchical"
    assert st["peer_n_obs"] >= 8
    assert st["mean"] > 0.5


def test_hierarchical_prior_penalizes_fresh_arm_in_bad_family(tmp_path: Path):
    db = _fresh_db(tmp_path, "hier_bad.db")
    for _ in range(8):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="FAILED_BREAKDOWN_REVERSAL",
            regime="range",
            pnl_usd=-10.0,
            exit_reason="STALL_EXIT",
            db_path=db,
        )
    st = get_arm_stats("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "range", db_path=db)
    assert st["prior_source"] == "hierarchical"
    assert st["mean"] < 0.5
    # Peer evidence must not fabricate a "starved" flag on a zero-obs arm
    assert st["starved"] is False


def test_peer_mass_is_capped_relative_to_prior(tmp_path: Path):
    """Even with tons of peer observations, α+β growth is bounded."""
    db = _fresh_db(tmp_path, "capped.db")
    for _ in range(60):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="HTF_TREND_PULLBACK",
            regime="range",
            pnl_usd=10.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    st = get_arm_stats("BTC/USDT", "HTF_TREND_PULLBACK", "range", db_path=db)
    total_mass = float(st["alpha"]) + float(st["beta"])
    # Prior mass (2) + capped peer contribution (≤ HIERARCHICAL_PRIOR_MAX_WEIGHT)
    assert total_mass < 2.0 + 2.5 + 0.5  # small slack


def test_hierarchical_prior_can_be_disabled(tmp_path: Path, monkeypatch):
    db = _fresh_db(tmp_path, "disabled.db")
    for _ in range(6):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="BREAKOUT_CONTINUATION",
            regime="range",
            pnl_usd=10.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    monkeypatch.setenv("DAY_BANDIT_HIERARCHICAL_PRIOR_ENABLED", "false")
    st = get_arm_stats("SOL/USDT", "BREAKOUT_CONTINUATION", "range", db_path=db)
    assert st["prior_source"] == "uniform"
    assert st["mean"] == pytest.approx(0.5, abs=1e-6)


def test_apply_bandit_stamps_prior_source_and_peer_n(tmp_path: Path):
    db = _fresh_db(tmp_path, "stamp.db")
    for _ in range(5):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="BREAKOUT_CONTINUATION",
            regime="range",
            pnl_usd=10.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    dd = apply_bandit_to_decision_data(
        {
            "setup_type": "BREAKOUT_CONTINUATION",
            "day_route_regime": "range",
            "final_selection_score": 0.0,
        },
        "SOL/USDT",
        db_path=db,
        rng=random.Random(1),
    )
    assert dd["day_bandit_prior_source"] == "hierarchical"
    assert dd["day_bandit_peer_n_obs"] >= 5


def test_empirical_arm_reports_empirical_source(tmp_path: Path):
    db = _fresh_db(tmp_path, "emp.db")
    for _ in range(3):
        record_bandit_outcome(
            symbol="BTC/USDT",
            setup="RANGE_BOUNCE",
            regime="range",
            pnl_usd=8.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    st = get_arm_stats("BTC/USDT", "RANGE_BOUNCE", "range", db_path=db)
    assert st["n_obs"] == 3
    assert st["prior_source"] == "empirical"
    assert st["peer_n_obs"] == 0


def test_bootstrap_default_lookback_is_deeper():
    """Deeper default lookback → more sells hydrated when arms table empty."""
    import inspect

    from backend.services.day_outcome_bandit import bootstrap_bandit_from_paper_trades

    sig = inspect.signature(bootstrap_bandit_from_paper_trades)
    default_lb = sig.parameters["lookback"].default
    assert int(default_lb) >= 240
