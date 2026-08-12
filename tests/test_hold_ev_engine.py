"""Item p8: HoldEV continuous hold-economics signal + its promotion to a
bounded, tighten-only exit-lever influence for DAY (giveback trigger) and
SCALP (scratch review-count patience)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.hold_ev_engine import (
    compute_hold_ev,
    hold_ev_for_position,
    hold_ev_giveback_tighten_factor,
    hold_ev_scratch_review_reduction,
)
from backend.services.mfe_mae_distribution_learner import ExpectedExcursion


def _mock_expected(mfe_p60=0.01, mae_p60=0.005, mfe_conf="confident", mae_conf="confident", mfe_n=20, mae_n=20):
    return ExpectedExcursion(
        symbol="BTCUSDT",
        strategy="day",
        expected_mfe_p60=mfe_p60,
        expected_mae_p60=mae_p60,
        mfe_confidence=mfe_conf,
        mae_confidence=mae_conf,
        mfe_stratum="symbol",
        mae_stratum="symbol",
        mfe_n_obs=mfe_n,
        mae_n_obs=mae_n,
    )


def test_score_always_bounded():
    with mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=_mock_expected()):
        result = compute_hold_ev(
            symbol="BTCUSDT",
            strategy="day",
            entry_price=100.0,
            current_price=110.0,
            highest_price=112.0,
            hold_minutes=60.0,
        )
    assert -1.0 <= result.hold_ev_score <= 1.0


def test_deep_adverse_excursion_pulls_score_negative():
    with mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=_mock_expected(mae_p60=0.005)):
        result = compute_hold_ev(
            symbol="BTCUSDT",
            strategy="day",
            entry_price=100.0,
            current_price=98.5,  # -1.5% vs typical loser MAE of 0.5%
            highest_price=100.0,
            hold_minutes=60.0,
        )
    assert result.excursion_component < 0
    assert result.hold_ev_score < 0
    assert result.recommendation in ("consider_exit", "monitor_closely")


def test_profit_beyond_typical_winner_ceiling_reduces_score():
    with mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=_mock_expected(mfe_p60=0.01)):
        near_target = compute_hold_ev(symbol="BTCUSDT", strategy="day", entry_price=100.0, current_price=100.3, highest_price=100.3, hold_minutes=30.0)
        beyond_target = compute_hold_ev(symbol="BTCUSDT", strategy="day", entry_price=100.0, current_price=101.5, highest_price=101.5, hold_minutes=30.0)
    assert beyond_target.excursion_component < near_target.excursion_component


def test_insufficient_data_reports_honest_confidence():
    empty = ExpectedExcursion(
        symbol="BTCUSDT",
        strategy="day",
        expected_mfe_p60=0.0,
        expected_mae_p60=0.0,
        mfe_confidence="insufficient_data",
        mae_confidence="insufficient_data",
        mfe_stratum="none",
        mae_stratum="none",
        mfe_n_obs=0,
        mae_n_obs=0,
    )
    with mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=empty):
        result = compute_hold_ev(symbol="BTCUSDT", strategy="day", entry_price=100.0, current_price=100.0, highest_price=100.0, hold_minutes=10.0)
    assert result.confidence == "insufficient_data"


def test_liquidity_damping_pulls_toward_neutral():
    with mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=_mock_expected()):
        tight = compute_hold_ev(symbol="BTCUSDT", strategy="day", entry_price=100.0, current_price=98.0, highest_price=100.0, hold_minutes=60.0, spread_pct=0.0001)
        wide = compute_hold_ev(symbol="BTCUSDT", strategy="day", entry_price=100.0, current_price=98.0, highest_price=100.0, hold_minutes=60.0, spread_pct=0.02)
    assert abs(wide.hold_ev_score) <= abs(tight.hold_ev_score)


def test_never_raises_when_microstructure_engine_unavailable():
    with (
        mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=_mock_expected()),
        mock.patch("backend.services.microstructure_engine.get_microstructure_ranking_delta", side_effect=RuntimeError("no data")),
    ):
        result = compute_hold_ev(symbol="BTCUSDT", strategy="day", entry_price=100.0, current_price=100.0, highest_price=100.0, hold_minutes=10.0)
    assert result.orderflow_component == 0.0


def test_hold_ev_for_position_wraps_day_position():
    pos = SimpleNamespace(symbol="BTC/USDT", entry_price=100.0, highest_price=101.0)
    with mock.patch("backend.services.hold_ev_engine.get_expected_mfe_mae", return_value=_mock_expected()):
        result = hold_ev_for_position(pos, current_price=100.5, hold_minutes=15.0)
    assert -1.0 <= result.hold_ev_score <= 1.0


# --- Item p8 promotion: DAY giveback-trigger tighten factor ---


def test_giveback_tighten_neutral_when_insufficient_data():
    assert hold_ev_giveback_tighten_factor(-0.9, "insufficient_data") == 1.0


def test_giveback_tighten_neutral_when_score_above_threshold():
    assert hold_ev_giveback_tighten_factor(-0.1, "confident") == 1.0
    assert hold_ev_giveback_tighten_factor(0.5, "confident") == 1.0


def test_giveback_tighten_shrinks_as_score_worsens():
    mild = hold_ev_giveback_tighten_factor(-0.4, "confident")
    severe = hold_ev_giveback_tighten_factor(-0.9, "confident")
    assert mild < 1.0
    assert severe < mild  # more negative HoldEV -> tighter (smaller) factor


def test_giveback_tighten_never_below_floor():
    factor = hold_ev_giveback_tighten_factor(-1.0, "confident")
    floor = float("0.6")
    assert factor >= floor - 1e-9
    assert factor <= 1.0


def test_giveback_tighten_never_exceeds_one():
    for score in (-1.0, -0.5, -0.35, 0.0, 1.0):
        assert hold_ev_giveback_tighten_factor(score, "confident") <= 1.0
        assert hold_ev_giveback_tighten_factor(score, "low_confidence") <= 1.0


# --- Item p8 promotion: SCALP scratch-review-count reduction ---


def test_scratch_reduction_zero_when_insufficient_data():
    assert hold_ev_scratch_review_reduction(-0.9, "insufficient_data") == 0


def test_scratch_reduction_zero_when_score_above_threshold():
    assert hold_ev_scratch_review_reduction(-0.2, "confident") == 0


def test_scratch_reduction_positive_when_score_strongly_negative():
    reduction = hold_ev_scratch_review_reduction(-0.9, "confident")
    assert reduction >= 1


def test_scratch_reduction_never_negative():
    for score in (-1.0, -0.6, -0.5, -0.1, 0.5):
        assert hold_ev_scratch_review_reduction(score, "confident") >= 0
