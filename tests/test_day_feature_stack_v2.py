"""Items p15-p17, p19: momentum/RVOL/volatility/BTC-lag feature stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import day_feature_stack_v2 as fs


def _rows(closes, volumes=None, high_offset=0.5, low_offset=0.5):
    volumes = volumes or [100.0] * len(closes)
    out = []
    for i, c in enumerate(closes):
        out.append([i * 60_000, c, c + high_offset, c - low_offset, c, volumes[i]])
    return out


def test_momentum_pct_neutral_on_short_series():
    assert fs.momentum_pct(_rows([1.0, 2.0]), lookback=10) == 0.0


def test_momentum_pct_computes_real_change():
    closes = [100.0] * 10 + [110.0]
    val = fs.momentum_pct(_rows(closes), lookback=5)
    assert val == pytest_approx(0.10)


def pytest_approx(x, tol=1e-9):
    class _A(float):
        def __eq__(self, other):
            return abs(other - x) <= tol

        __hash__ = None

    return _A(x)


def test_momentum_multi_horizon_stable_keys_on_missing_tf():
    bundle = {"5m": _rows([100.0] * 10)}
    out = fs.momentum_multi_horizon(bundle)
    assert set(out.keys()) == set(fs.DEFAULT_MOMENTUM_HORIZONS.keys())
    assert out["1d"] == 0.0  # missing TF -> neutral, not omitted


def test_same_symbol_rvol_neutral_on_insufficient_history():
    assert fs.same_symbol_rvol(_rows([100.0] * 5, [10.0] * 5)) == 1.0


def test_same_symbol_rvol_detects_volume_spike():
    vols = [10.0] * 20 + [50.0]
    val = fs.same_symbol_rvol(_rows([100.0] * 21, vols))
    assert val == pytest_approx(5.0)


def test_same_symbol_rvol_neutral_on_zero_baseline_volume():
    vols = [0.0] * 20 + [50.0]
    assert fs.same_symbol_rvol(_rows([100.0] * 21, vols)) == 1.0


def test_atr_pct_multi_period_zero_on_flat_series():
    rows = [[i * 60_000, 100.0, 100.0, 100.0, 100.0, 10.0] for i in range(40)]
    out = fs.atr_pct_multi_period(rows)
    assert set(out.keys()) == set(fs.DEFAULT_ATR_PERIODS)
    for v in out.values():
        assert v == 0.0


def test_realized_vol_pct_zero_on_insufficient_bars():
    assert fs.realized_vol_pct(_rows([100.0, 101.0]), window=20) == 0.0


def test_realized_vol_pct_nonzero_on_noisy_series():
    rng = np.random.default_rng(42)
    closes = list(100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 50)))
    val = fs.realized_vol_pct(_rows(closes), window=20)
    assert val > 0.0


def test_vol_percentile_neutral_on_insufficient_history():
    assert fs.vol_percentile(_rows([100.0] * 30)) == 0.5


def test_vol_percentile_ranks_current_relative_to_own_history():
    rng = np.random.default_rng(7)
    n = 14 + 90 + 5
    closes = list(100.0 * np.cumprod(1.0 + rng.normal(0, 0.003, n)))
    rows = _rows(closes)
    val = fs.vol_percentile(rows, atr_period=14, lookback=90)
    assert 0.0 <= val <= 1.0


def test_lagged_cross_correlation_insufficient_data():
    result = fs.lagged_cross_correlation_vs_btc(_rows([100.0] * 5), _rows([100.0] * 5))
    assert result.confidence == "insufficient_data"
    assert result.best_lag_bars == 0
    assert result.corr_at_best_lag == 0.0


def test_lagged_cross_correlation_detects_symbol_lagging_btc():
    # Symbol repeats BTC's move from `lag` bars ago -> symbol trails/LAGS BTC,
    # which is best_lag_bars == -lag under this module's lead/lag sign convention
    # (positive == symbol leads, negative == symbol lags).
    rng = np.random.default_rng(1)
    n = 120
    btc_rets = rng.normal(0, 0.01, n)
    btc_closes = list(100.0 * np.cumprod(1.0 + btc_rets))
    lag = 2
    sym_rets = np.concatenate([rng.normal(0, 0.001, lag), btc_rets[: n - lag]])
    sym_closes = list(50.0 * np.cumprod(1.0 + sym_rets))
    result = fs.lagged_cross_correlation_vs_btc(_rows(sym_closes), _rows(btc_closes), max_lag_bars=5, min_obs=20)
    assert result.confidence in ("confident", "low_confidence")
    assert result.best_lag_bars == -lag


def test_compute_feature_stack_snapshot_never_raises_on_empty_bundle():
    snap = fs.compute_feature_stack_snapshot("XYZUSDT", {})
    d = snap.to_dict()
    assert d["symbol"] == "XYZUSDT"
    assert d["vol_percentile_1h"] == 0.5
    assert d["btc_lag"] is None


def test_compute_feature_stack_snapshot_skips_lag_for_btc_itself():
    bundle = {"1m": _rows([100.0] * 50)}
    snap = fs.compute_feature_stack_snapshot("BTCUSDT", bundle, btc_bundle=bundle)
    assert snap.btc_lag is None


def test_compute_feature_stack_snapshot_full_pipeline():
    rng = np.random.default_rng(3)
    n = 200
    closes = list(100.0 * np.cumprod(1.0 + rng.normal(0, 0.005, n)))
    vols = list(np.abs(rng.normal(10.0, 2.0, n)))
    bundle = {tf: _rows(closes, vols) for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d")}
    btc_bundle = {"1m": _rows(closes, vols)}
    snap = fs.compute_feature_stack_snapshot("ETHUSDT", bundle, btc_bundle=btc_bundle)
    d = snap.to_dict()
    assert set(d["momentum"].keys()) == set(fs.DEFAULT_MOMENTUM_HORIZONS.keys())
    assert set(d["rvol"].keys()) == set(fs.DEFAULT_RVOL_TFS)
    assert d["btc_lag"] is not None


# ---------------------------------------------------------------------------
# Items p15/p16/p19 ranking-promotion signals (ctx_multiplier terms)
# ---------------------------------------------------------------------------


def test_momentum_rvol_confirmation_signal_neutral_on_empty_momentum():
    assert fs.momentum_rvol_confirmation_signal({}, {"1h": 2.0}) == 0.0


def test_momentum_rvol_confirmation_signal_bullish_with_volume_confirmation():
    momentum = {"1h": 0.03, "4h": 0.03}  # at scale -> signal 1.0 before confirmation
    high_rvol = fs.momentum_rvol_confirmation_signal(momentum, {"1h": 3.0})
    low_rvol = fs.momentum_rvol_confirmation_signal(momentum, {"1h": 0.0})
    assert high_rvol > low_rvol > 0.0
    assert -1.0 <= high_rvol <= 1.0
    assert -1.0 <= low_rvol <= 1.0


def test_momentum_rvol_confirmation_signal_bearish_direction_preserved():
    momentum = {"1h": -0.03}
    val = fs.momentum_rvol_confirmation_signal(momentum, {"1h": 2.0})
    assert val < 0.0


def test_btc_lag_predictive_signal_zero_when_not_confident():
    lag = fs.LagCorrelationResult(tf="1m", best_lag_bars=-2, corr_at_best_lag=0.8, same_bar_corr=0.1, n_obs=40, confidence="low_confidence")
    assert fs.btc_lag_predictive_signal(lag, 0.02) == 0.0


def test_btc_lag_predictive_signal_zero_when_symbol_leads_or_simultaneous():
    lag = fs.LagCorrelationResult(tf="1m", best_lag_bars=2, corr_at_best_lag=0.8, same_bar_corr=0.1, n_obs=200, confidence="confident")
    assert fs.btc_lag_predictive_signal(lag, 0.02) == 0.0
    lag0 = fs.LagCorrelationResult(tf="1m", best_lag_bars=0, corr_at_best_lag=0.8, same_bar_corr=0.8, n_obs=200, confidence="confident")
    assert fs.btc_lag_predictive_signal(lag0, 0.02) == 0.0


def test_btc_lag_predictive_signal_fires_when_symbol_confidently_lags():
    lag = fs.LagCorrelationResult(tf="1m", best_lag_bars=-3, corr_at_best_lag=0.8, same_bar_corr=0.1, n_obs=200, confidence="confident")
    val = fs.btc_lag_predictive_signal(lag, 0.01)  # BTC just moved +1% over the lag window
    assert val == pytest_approx(0.8)


def test_btc_lag_predictive_signal_none_lag_is_neutral():
    assert fs.btc_lag_predictive_signal(None, 0.05) == 0.0
