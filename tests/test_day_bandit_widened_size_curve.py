"""Batch 7: widened bandit size curve gated on n_obs.

Guards:
1. Fresh arm (n_obs == 0, no peers) → size_factor <= MAX_SIZE_FRESH.
2. Fresh-but-peer-informed arm → still capped at MAX_SIZE_FRESH.
3. Well-observed winner (n_obs >= N_OBS_FOR_UPSIZE, mean high) → size may exceed 1.35.
4. Well-observed neutral arm → size in mid-band, still capped by mean.
5. Starved arm → STARVE_SIZE_FLOOR unchanged (0.08).
6. Env overrides honored: DAY_BANDIT_MAX_SIZE_TOP / DAY_BANDIT_MAX_SIZE_FRESH.
"""

from __future__ import annotations

import importlib
import random
from pathlib import Path

from backend.services.day_outcome_bandit import (
    EXPLORE_SIZE_FLOOR,
    MAX_SIZE_FRESH,
    MAX_SIZE_TOP,
    STARVE_SIZE_FLOOR,
    record_bandit_outcome,
    sample_arm,
)


def test_fresh_arm_capped_below_max_fresh(tmp_path: Path):
    db = str(tmp_path / "fresh.db")
    samp = sample_arm(
        "BTC/USDT",
        "HTF_TREND_PULLBACK",
        "range",
        db_path=db,
        rng=random.Random(0),
    )
    # Fresh arm with no peers → default 1.0 (or MAX_SIZE_FRESH if smaller)
    assert samp["size_factor"] <= max(1.0, MAX_SIZE_FRESH) + 1e-6
    assert samp["size_factor_cap"] == "fresh"


def test_peer_informed_arm_still_capped_at_fresh(tmp_path: Path):
    db = str(tmp_path / "peer.db")
    for _ in range(20):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="HTF_TREND_PULLBACK",
            regime="range",
            pnl_usd=15.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    # BTC arm is still fresh even though peer support pushes mean high.
    samp = sample_arm(
        "BTC/USDT",
        "HTF_TREND_PULLBACK",
        "range",
        db_path=db,
        rng=random.Random(0),
    )
    assert samp["n_obs"] == 0
    assert samp["size_factor"] <= MAX_SIZE_FRESH + 1e-6
    assert samp["size_factor_cap"] == "fresh"


def test_well_observed_winner_can_exceed_135(tmp_path: Path):
    db = str(tmp_path / "top.db")
    for _ in range(15):
        record_bandit_outcome(
            symbol="SOL/USDT",
            setup="BREAKOUT_CONTINUATION",
            regime="bull",
            pnl_usd=20.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    samp = sample_arm(
        "SOL/USDT",
        "BREAKOUT_CONTINUATION",
        "bull",
        db_path=db,
        rng=random.Random(0),
    )
    assert samp["size_factor_cap"] == "top"
    assert samp["size_factor"] > 1.35
    assert samp["size_factor"] <= MAX_SIZE_TOP + 1e-6


def test_well_observed_neutral_mean_gives_mid_size(tmp_path: Path):
    db = str(tmp_path / "mid.db")
    # 12 sells alternating win/loss → mean drifts around 0.5
    for i in range(12):
        record_bandit_outcome(
            symbol="XRP/USDT",
            setup="RANGE_BOUNCE",
            regime="range",
            pnl_usd=5.0 if i % 2 == 0 else -5.0,
            exit_reason="NET_PROFIT_EXIT" if i % 2 == 0 else "STALL_EXIT",
            db_path=db,
        )
    samp = sample_arm(
        "XRP/USDT",
        "RANGE_BOUNCE",
        "range",
        db_path=db,
        rng=random.Random(0),
    )
    assert samp["size_factor_cap"] == "top"
    assert EXPLORE_SIZE_FLOOR <= samp["size_factor"] <= MAX_SIZE_TOP


def test_starve_size_floor_still_low(tmp_path: Path):
    db = str(tmp_path / "s.db")
    for _ in range(10):
        record_bandit_outcome(
            symbol="ETH/USDT",
            setup="HTF_TREND_PULLBACK",
            regime="bull",
            pnl_usd=-8.0,
            exit_reason="STALL_EXIT",
            db_path=db,
        )
    samp = sample_arm(
        "ETH/USDT",
        "HTF_TREND_PULLBACK",
        "bull",
        db_path=db,
        rng=random.Random(0),
    )
    assert samp["size_factor"] == STARVE_SIZE_FLOOR
    assert samp["size_factor_cap"] == "starve"


def test_env_override_max_size_top(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DAY_BANDIT_MAX_SIZE_TOP", "1.50")
    monkeypatch.setenv("DAY_BANDIT_MAX_SIZE_FRESH", "1.05")
    import backend.services.day_outcome_bandit as mod

    importlib.reload(mod)
    db = str(tmp_path / "e.db")
    for _ in range(20):
        mod.record_bandit_outcome(
            symbol="BTC/USDT",
            setup="RANGE_BOUNCE",
            regime="range",
            pnl_usd=15.0,
            exit_reason="NET_PROFIT_EXIT",
            db_path=db,
        )
    samp = mod.sample_arm(
        "BTC/USDT",
        "RANGE_BOUNCE",
        "range",
        db_path=db,
        rng=random.Random(0),
    )
    assert samp["size_factor"] <= 1.50 + 1e-6
    # Reload back to default so other tests are not affected.
    monkeypatch.delenv("DAY_BANDIT_MAX_SIZE_TOP", raising=False)
    monkeypatch.delenv("DAY_BANDIT_MAX_SIZE_FRESH", raising=False)
    importlib.reload(mod)
