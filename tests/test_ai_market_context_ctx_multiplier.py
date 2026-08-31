"""Items p15/p16/p18/p19/p20 ranking promotion: _ctx_multiplier now folds in
feature-stack (momentum x same-symbol RVOL), BTC-lag, derivatives, and
cross-exchange signals alongside the pre-existing MTF/RS/depth/regime/
microstructure terms — still a bounded, non-gating additive multiplier."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import ai_decision_contract as adc
from backend.services.ai_market_context import _ctx_multiplier

_NEUTRAL_MTF = {"1h": {"bars": 20, "ema_align": 0.5}}


def _base_kwargs(**overrides):
    kwargs = {
        "own_mtf": _NEUTRAL_MTF,
        "rs_btc": 0.0,
        "rs_eth": 0.0,
        "depth_imbalance": 0.0,
        "market_regime": "ranging",
    }
    kwargs.update(overrides)
    return kwargs


def test_all_new_signals_default_to_zero_contribution():
    multiplier, audit = _ctx_multiplier(**_base_kwargs())
    assert multiplier == 1.0
    assert audit["feature_stack_term"] == 0.0
    assert audit["btc_lag_term"] == 0.0
    assert audit["derivatives_term"] == 0.0
    assert audit["cross_exchange_term"] == 0.0


def test_feature_stack_signal_moves_multiplier_by_its_weight():
    multiplier, audit = _ctx_multiplier(**_base_kwargs(feature_stack_signal=1.0))
    assert audit["feature_stack_term"] == adc.CTX_FEATURE_STACK_WEIGHT
    assert multiplier > 1.0


def test_btc_lag_signal_moves_multiplier_by_its_weight():
    multiplier, audit = _ctx_multiplier(**_base_kwargs(btc_lag_signal=-1.0))
    assert audit["btc_lag_term"] == -adc.CTX_BTC_LAG_WEIGHT
    assert multiplier < 1.0


def test_derivatives_signal_moves_multiplier_by_its_weight():
    _multiplier, audit = _ctx_multiplier(**_base_kwargs(derivatives_signal=0.5))
    assert audit["derivatives_term"] == 0.5 * adc.CTX_DERIVATIVES_WEIGHT


def test_cross_exchange_signal_moves_multiplier_by_its_weight():
    _multiplier, audit = _ctx_multiplier(**_base_kwargs(cross_exchange_signal=-0.5))
    assert audit["cross_exchange_term"] == -0.5 * adc.CTX_CROSS_EXCHANGE_WEIGHT


def test_all_signals_saturated_stays_within_total_cap():
    multiplier, audit = _ctx_multiplier(
        **_base_kwargs(
            rs_btc=1.0,
            rs_eth=1.0,
            depth_imbalance=1.0,
            market_regime="trending_up",
            microstructure_signal=1.0,
            feature_stack_signal=1.0,
            btc_lag_signal=1.0,
            derivatives_signal=1.0,
            cross_exchange_signal=1.0,
        )
    )
    assert multiplier <= 1.0 + adc.CTX_TOTAL_CAP + 1e-12
    assert abs(audit["total_signed"]) <= adc.CTX_TOTAL_CAP + 1e-12


def test_signals_are_clamped_before_weighting():
    # out-of-range inputs must not blow past their own weight contribution
    _multiplier, audit = _ctx_multiplier(**_base_kwargs(feature_stack_signal=5.0))
    assert audit["feature_stack_term"] == adc.CTX_FEATURE_STACK_WEIGHT
