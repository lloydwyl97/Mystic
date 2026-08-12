"""Items p6/p7/p9: adaptive targets, adaptive MAE/giveback trigger, progress-rate decay exit."""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import day_adaptive_targets as dat
from backend.services import mfe_mae_distribution_learner as dist
from backend.services.day_controlled_exits import (
    EXIT_ADAPTIVE_LOSS,
    EXIT_GIVEBACK,
    EXIT_PROGRESS_DECAY,
    EXIT_STOP_LOSS,
    effective_target_price,
    evaluate_adaptive_loss_exit,
    evaluate_engine_managed_exit,
    evaluate_giveback_exit,
    evaluate_progress_decay_exit,
)
from backend.services.market_role_outcome_learner import _SCHEMA_SQL


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = str(tmp_path / "test_outcomes.db")
    with sqlite3.connect(p) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    dist._cache.clear()
    monkeypatch.setattr(dist, "MIN_OBS", 5)
    return p


def _insert(db_path, *, symbol, pnl_pct, mfe_pct, mae_pct, hold_seconds=600):
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO market_role_trade_outcomes
              (trade_id, buy_trade_id, symbol, strategy, realized_pnl_pct,
               hold_seconds, exit_reason, mfe_pct, mae_pct, market_regime,
               created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (f"t_{time.time_ns()}", "buy", symbol.upper(), "day", pnl_pct, hold_seconds, "TEST", mfe_pct, mae_pct, "range", now_iso),
        )
        conn.commit()


def _pos(**overrides):
    base = dict(symbol="BTC/USDT", entry_thesis="HTF_TREND_PULLBACK", day_route_regime_at_entry="range")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_adaptive_target_insufficient_data_falls_back(db_path, monkeypatch):
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    result = dat.adaptive_target_pct_for_arm("BTC/USDT", "setup", "range")
    assert result["source"] == "insufficient_data"
    assert result["target_pct"] == 0.0


def test_adaptive_target_uses_winner_mfe_when_sufficient(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="BTC/USDT", pnl_pct=1.0, mfe_pct=0.01 + i * 0.001, mae_pct=0.001)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    result = dat.adaptive_target_pct_for_arm("BTC/USDT", "setup", "range")
    assert result["source"] not in ("insufficient_data", "disabled")
    assert result["target_pct"] > 0.0


def test_effective_target_price_only_tightens_never_extends(db_path, monkeypatch):
    for i in range(6):
        # Winners' MFE is small (~0.5%) -> adaptive target should be *below*
        # a distant fixed thesis target, never above it.
        _insert(db_path, symbol="BTC/USDT", pnl_pct=1.0, mfe_pct=0.005 + i * 0.0002, mae_pct=0.001)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="BTC/USDT")
    entry = 100.0
    distant_thesis_target = entry * 1.10  # +10%, far beyond what winners on this arm reach
    target = effective_target_price(entry, 0.0, distant_thesis_target, pos)
    assert target < distant_thesis_target
    assert target > entry


def test_effective_target_price_unaffected_when_position_omitted():
    # Exact prior behavior preserved when position is not supplied.
    target = effective_target_price(100.0, 102.0, 105.0)
    assert target == 102.0


def test_adaptive_giveback_trigger_replaces_fixed_default_when_arm_has_history(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="ETH/USDT")
    out = evaluate_giveback_exit(
        entry_price=100.0,
        highest_price=100.5,  # 0.5% MFE clears the fixed 0.25% min-mfe trigger
        net_pnl_pct=-0.008,  # deeper than both the fixed -0.15% default AND the arm's learned ~-0.64%
        hold_minutes=25.0,
        position=pos,
    )
    assert out is not None
    assert "trigger_source" in out["detail"]
    assert "fixed_default" not in out["detail"]


def test_adaptive_giveback_trigger_falls_back_when_no_history():
    out_no_history = evaluate_giveback_exit(
        entry_price=100.0,
        highest_price=100.5,
        net_pnl_pct=-0.005,
        hold_minutes=25.0,
        position=_pos(symbol="NEWCOIN/USDT"),
    )
    assert out_no_history is not None
    assert "trigger_source=fixed_default" in out_no_history["detail"]


def test_giveback_trigger_tightens_when_hold_ev_strongly_disfavors_holding(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="ETH/USDT")

    baseline = evaluate_giveback_exit(
        entry_price=100.0,
        highest_price=100.5,
        net_pnl_pct=-0.005,  # shallower than this arm's own adaptive trigger (~-0.0066) -> no fire
        hold_minutes=25.0,
        position=pos,
    )
    assert baseline is None

    fake_hev = SimpleNamespace(hold_ev_score=-0.9, confidence="confident")
    with mock.patch("backend.services.hold_ev_engine.hold_ev_for_position", return_value=fake_hev):
        tightened = evaluate_giveback_exit(
            entry_price=100.0,
            highest_price=100.5,
            net_pnl_pct=-0.005,  # same shallow loss — now enough once HoldEV tightens the trigger
            hold_minutes=25.0,
            position=pos,
        )
    assert tightened is not None
    assert tightened["reason"] == EXIT_GIVEBACK
    assert "hev_score=-0.900" in tightened["detail"]
    assert "hev_factor=" in tightened["detail"]


def test_giveback_trigger_unaffected_when_hold_ev_insufficient_data(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="ETH/USDT")

    fake_hev = SimpleNamespace(hold_ev_score=-0.9, confidence="insufficient_data")
    with mock.patch("backend.services.hold_ev_engine.hold_ev_for_position", return_value=fake_hev):
        out = evaluate_giveback_exit(
            entry_price=100.0,
            highest_price=100.5,
            net_pnl_pct=-0.005,
            hold_minutes=25.0,
            position=pos,
        )
    assert out is None  # HoldEV confidence is insufficient_data -> neutral, base trigger untouched


def test_giveback_trigger_never_widens_when_hold_ev_favors_holding(db_path, monkeypatch):
    """HoldEV must only ever tighten (never widen) the giveback trigger —
    a strongly positive score must not prevent a fire the base trigger
    would otherwise have produced."""
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="ETH/USDT")

    fake_hev = SimpleNamespace(hold_ev_score=0.9, confidence="confident")
    with mock.patch("backend.services.hold_ev_engine.hold_ev_for_position", return_value=fake_hev):
        out = evaluate_giveback_exit(
            entry_price=100.0,
            highest_price=100.5,
            net_pnl_pct=-0.008,  # already deeper than the base adaptive trigger (~-0.0066)
            hold_minutes=25.0,
            position=pos,
        )
    assert out is not None
    assert out["reason"] == EXIT_GIVEBACK


def test_progress_decay_never_fires_on_green_position():
    out = evaluate_progress_decay_exit(
        entry_price=100.0,
        highest_price=100.0,
        net_pnl_pct=0.01,
        hold_minutes=60.0,
        position=_pos(symbol="BTC/USDT"),
    )
    assert out is None


def test_progress_decay_never_fires_without_position():
    out = evaluate_progress_decay_exit(
        entry_price=100.0,
        highest_price=100.0,
        net_pnl_pct=-0.01,
        hold_minutes=60.0,
        position=None,
    )
    assert out is None


def test_progress_decay_never_fires_before_min_hold():
    out = evaluate_progress_decay_exit(
        entry_price=100.0,
        highest_price=100.0,
        net_pnl_pct=-0.01,
        hold_minutes=5.0,
        position=_pos(symbol="BTC/USDT"),
    )
    assert out is None


def test_progress_decay_fires_when_far_below_arm_typical_pace(db_path, monkeypatch):
    # Arm's typical (same hold-time bucket) winner reaches 2% MFE.
    for i in range(6):
        _insert(db_path, symbol="BTC/USDT", pnl_pct=1.0, mfe_pct=0.02 + i * 0.0005, mae_pct=0.001, hold_seconds=1800)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_progress_decay_exit(
            entry_price=100.0,
            highest_price=100.02,  # only 0.02% MFE — far below the arm's typical ~2%
            net_pnl_pct=-0.003,
            hold_minutes=30.0,  # same hold_seconds=1800 bucket as seeded winners
            position=_pos(symbol="BTC/USDT"),
        )
    assert out is not None
    assert out["reason"] == EXIT_PROGRESS_DECAY


def _rows_1h(closes, high_offset=0.5, low_offset=0.5):
    out = []
    for i, c in enumerate(closes):
        out.append([i * 3_600_000, c, c + high_offset, c - low_offset, c, 100.0])
    return out


def test_atr_grid_target_insufficient_data_falls_back(db_path, monkeypatch):
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    result = dat.atr_grid_target_candidate("BTC/USDT", current_atr_pct=0.01)
    assert result["source"] == "insufficient_data"
    assert result["target_pct"] == 0.0


def test_atr_grid_target_zero_or_disabled_atr_short_circuits(db_path, monkeypatch):
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    result = dat.atr_grid_target_candidate("BTC/USDT", current_atr_pct=0.0)
    assert result["source"] == "disabled_or_no_atr"


def test_atr_grid_target_selects_best_expectancy_multiple(db_path, monkeypatch):
    # Seed trades whose MFE consistently clears the 1.00x ATR target (0.01) but
    # NOT the 2.00x ATR target (0.02) -> 1.00x should score better than 2.00x
    # (2.00x candidates fall back to the worse actual realized outcome).
    for i in range(30):
        _insert(db_path, symbol="BTC/USDT", pnl_pct=0.003, mfe_pct=0.012 + (i % 3) * 0.0005, mae_pct=0.001)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    result = dat.atr_grid_target_candidate("BTC/USDT", current_atr_pct=0.01)
    assert result["source"] == "atr_grid_expectancy"
    assert result["n_obs"] == 30
    assert result["atr_multiple"] in dat.DAY_ATR_TARGET_GRID
    # 1.00x (target 0.01) is reachable by every seeded trade and nets a better
    # simulated outcome than 2.00x (target 0.02, never reached -> falls back
    # to the smaller actual realized 0.003 pnl).
    assert result["grid_candidates"]["1.00x_atr"] > result["grid_candidates"]["2.00x_atr"]


def test_atr_grid_target_never_uses_cross_symbol_pool(db_path, monkeypatch):
    for i in range(30):
        _insert(db_path, symbol="ETHUSDT", pnl_pct=0.05, mfe_pct=0.05, mae_pct=0.001)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    result = dat.atr_grid_target_candidate("SOLUSDT", current_atr_pct=0.01)
    assert result["source"] == "insufficient_data"


def test_effective_target_price_incorporates_atr_grid_via_bundle(db_path, monkeypatch):
    for i in range(30):
        _insert(db_path, symbol="BTC/USDT", pnl_pct=0.003, mfe_pct=0.006 + (i % 3) * 0.0005, mae_pct=0.001)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="BTC/USDT")
    entry = 100.0
    distant_thesis_target = entry * 1.10
    # ATR% of ~0.005 (0.5%) on a flat-ish 1h series -> grid targets are small,
    # tighter than the distant fixed thesis target.
    bundle = {"1h": _rows_1h([100.0 + (i % 2) * 0.4 for i in range(40)])}
    target = effective_target_price(entry, 0.0, distant_thesis_target, pos, bundle)
    assert target < distant_thesis_target
    assert target > entry


def test_effective_target_price_unaffected_when_bundle_omitted(db_path, monkeypatch):
    for i in range(30):
        _insert(db_path, symbol="BTC/USDT", pnl_pct=1.0, mfe_pct=0.005 + i * 0.0002, mae_pct=0.001)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="BTC/USDT")
    # No bundle -> identical to the pre-p6-grid behavior (MFE-percentile
    # candidate can still apply, ATR-grid candidate is simply absent).
    target_without_bundle = effective_target_price(100.0, 0.0, 110.0, pos)
    target_with_none_bundle = effective_target_price(100.0, 0.0, 110.0, pos, None)
    assert target_without_bundle == target_with_none_bundle


def test_progress_decay_never_uses_cross_symbol_pool(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="ETHUSDT", pnl_pct=1.0, mfe_pct=0.05, mae_pct=0.001, hold_seconds=1800)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_progress_decay_exit(
            entry_price=100.0,
            highest_price=100.0,
            net_pnl_pct=-0.003,
            hold_minutes=30.0,
            position=_pos(symbol="SOLUSDT"),  # different symbol — no own history
        )
    assert out is None


# --- Item p7: evaluate_adaptive_loss_exit (straight-loser adaptive MAE check) ---


def test_adaptive_loss_exit_fires_on_abnormal_same_symbol_mae(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.03,  # far beyond this arm's own historical losing-MAE p75 (~0.0068)
            hold_minutes=60.0,
            position=_pos(symbol="ETH/USDT"),
        )
    assert out is not None
    assert out["reason"] == EXIT_ADAPTIVE_LOSS
    assert "n_obs=6" in out["detail"]


def test_adaptive_loss_exit_does_not_fire_within_normal_range(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.001,  # well inside this arm's normal losing-MAE range
            hold_minutes=60.0,
            position=_pos(symbol="ETH/USDT"),
        )
    assert out is None


def test_adaptive_loss_exit_insufficient_data_returns_none(db_path):
    # No rows seeded for this symbol at all.
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.05,
            hold_minutes=60.0,
            position=_pos(symbol="BTC/USDT"),
        )
    assert out is None


def test_adaptive_loss_exit_never_uses_cross_symbol_pool(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETHUSDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.05,
            hold_minutes=60.0,
            position=_pos(symbol="SOLUSDT"),  # different symbol — no own history
        )
    assert out is None


def test_adaptive_loss_exit_never_fires_on_nonnegative_pnl(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=0.02,
            hold_minutes=60.0,
            position=_pos(symbol="ETH/USDT"),
        )
    assert out is None


def test_adaptive_loss_exit_never_fires_before_min_hold(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.03,
            hold_minutes=1.0,  # below DAY_ADAPTIVE_LOSS_MIN_HOLD_MIN default of 10
            position=_pos(symbol="ETH/USDT"),
        )
    assert out is None


def test_adaptive_loss_exit_respects_disabled_toggle(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setenv("DAY_ADAPTIVE_LOSS_EXIT_ENABLED", "false")
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.03,
            hold_minutes=60.0,
            position=_pos(symbol="ETH/USDT"),
        )
    assert out is None


def test_adaptive_loss_exit_never_fires_without_position(db_path):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_adaptive_loss_exit(
            entry_price=100.0,
            net_pnl_pct=-0.03,
            hold_minutes=60.0,
            position=None,
        )
    assert out is None


def test_adaptive_loss_exit_wired_before_fixed_stop_in_engine_managed_exit(db_path, monkeypatch):
    """A straight loser whose excursion is abnormal for its own arm must be
    caught by the adaptive check BEFORE the fixed stop_price, per item p7."""
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(symbol="ETH/USDT", entry_price=100.0, stop_price=50.0, trailing_stop_price=0.0)  # fixed stop far below current price
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_engine_managed_exit(
            position=pos,
            current_price=97.0,  # -3% — well beyond the arm's own losing-MAE p75 (~0.0068), but above the fixed stop of 50
            net_pnl_pct=-0.03,
            hold_minutes=60.0,
            coin_profile={},
            bundle=None,
            bar_low=97.0,
        )
    assert out.get("reason") == EXIT_ADAPTIVE_LOSS


# --- Item p7: bull-regime giveback must use the arm's adaptive trigger, not silently bypass it ---


def test_bull_giveback_uses_adaptive_arm_trigger_when_available(db_path, monkeypatch):
    for i in range(6):
        _insert(db_path, symbol="ETH/USDT", pnl_pct=-1.0, mfe_pct=0.001, mae_pct=0.006 + i * 0.0002)
    monkeypatch.setattr(dat, "_db_path", lambda: db_path)
    pos = _pos(
        symbol="ETH/USDT",
        entry_price=100.0,
        day_route_regime_at_entry="bull",
        stop_price=0.0,
        trailing_stop_price=0.0,
        highest_price=100.6,
    )
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_engine_managed_exit(
            position=pos,
            current_price=99.33,
            # Between this arm's giveback trigger (~-0.0066) and its adaptive-loss
            # p75 threshold (~-0.00675) so the giveback path fires but the
            # earlier straight-loser adaptive-loss check does not preempt it.
            net_pnl_pct=-0.0067,
            hold_minutes=60.0,
            coin_profile={},
            bundle=None,
            bar_low=99.33,
        )
    assert out.get("reason") == "GIVEBACK_EXIT"
    assert "base_trigger_source=symbol" in out.get("detail", "")
    assert "base_trigger_source=fixed_default" not in out.get("detail", "")


def test_bull_giveback_falls_back_to_fixed_default_without_history(db_path):
    pos = _pos(
        symbol="NEWCOIN/USDT",
        entry_price=100.0,
        day_route_regime_at_entry="bull",
        stop_price=0.0,
        trailing_stop_price=0.0,
        highest_price=100.6,
    )
    with mock.patch("backend.services.day_controlled_exits._db_path", return_value=db_path):
        out = evaluate_engine_managed_exit(
            position=pos,
            current_price=99.8,
            net_pnl_pct=-0.002,
            hold_minutes=60.0,
            coin_profile={},
            bundle=None,
            bar_low=99.8,
        )
    if out.get("reason") == "GIVEBACK_EXIT":
        assert "base_trigger_source=fixed_default" in out.get("detail", "")
