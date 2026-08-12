"""Item p18: derivatives (OI/funding/basis) public reference feed."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend import derivatives_monitor as dm


def setup_function(_fn):
    dm._DERIV_CACHE.clear()


def test_open_interest_no_longer_requires_api_key(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    calls = []

    def fake_http_get(url, params=None, headers=None):
        calls.append((url, params, headers))
        return {"openInterest": "12345.0", "time": 1700000000000}

    with mock.patch.object(dm, "_http_get", side_effect=fake_http_get), mock.patch.object(dm, "_fetch_futures_ticker_data", return_value={}):
        result = dm.fetch_binance_open_interest("BTCUSDT")

    assert result["open_interest"] == 12345.0
    # public endpoint: no API key header attached
    assert "X-MBX-APIKEY" not in calls[0][2]


def test_open_interest_decoupled_from_execution_exchange(monkeypatch):
    # Even though EXCHANGE_ID / execution venue is binance_us (no futures),
    # the reference feed must still attempt the GLOBAL futures public API.
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(dm, "_http_get", return_value={"openInterest": "1.0", "time": 1}), mock.patch.object(dm, "_fetch_futures_ticker_data", return_value={}):
        result = dm.fetch_binance_open_interest("BTCUSDT")
    assert result


def test_reference_feed_kill_switch(monkeypatch):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "0")
    assert dm.fetch_binance_open_interest("BTCUSDT") == {}
    assert dm.fetch_binance_funding_and_basis("BTCUSDT") == {}


def test_funding_and_basis_computation(monkeypatch):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(
        dm,
        "_http_get",
        return_value={
            "markPrice": "101.0",
            "indexPrice": "100.0",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1700000000000,
        },
    ):
        result = dm.fetch_binance_funding_and_basis("ETHUSDT")
    assert abs(result["basis_pct"] - 0.01) < 1e-9
    assert result["funding_rate"] == 0.0001


def test_funding_and_basis_degrades_on_bad_payload(monkeypatch):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(dm, "_http_get", return_value={}):
        assert dm.fetch_binance_funding_and_basis("ETHUSDT") == {}


def test_reference_snapshot_honest_degraded_state_on_api_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(dm, "fetch_binance_open_interest", return_value={}), mock.patch.object(dm, "fetch_binance_funding_and_basis", return_value={}):
        result = dm.derivatives_reference_snapshot("DOGEUSDT", db_path=str(tmp_path / "t.db"))
    assert result["available"] is False
    assert "degraded_reason" in result


def test_reference_snapshot_available_and_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    oi_payload = {
        "symbol": "BTCUSDT",
        "open_interest": 5000.0,
        "positioning_signals": {
            "open_interest_volume_ratio": 10.0,
            "volatility_expectation": 0.2,
            "momentum_alignment": 0.1,
            "positioning_bias": "bullish",
            "bias_strength": 0.7,
            "extreme_positioning": False,
        },
        "timestamp": 1,
    }
    fb_payload = {"basis_pct": 0.002, "funding_rate": 0.0001}
    calls = {"oi": 0, "fb": 0}
    db_path = str(tmp_path / "t.db")

    def _oi(_sym):
        calls["oi"] += 1
        return oi_payload

    def _fb(_sym):
        calls["fb"] += 1
        return fb_payload

    with mock.patch.object(dm, "fetch_binance_open_interest", side_effect=_oi), mock.patch.object(dm, "fetch_binance_funding_and_basis", side_effect=_fb):
        first = dm.derivatives_reference_snapshot("BTCUSDT", db_path=db_path)
        second = dm.derivatives_reference_snapshot("BTCUSDT", db_path=db_path)  # should hit cache

    assert first["available"] is True
    assert first["oi_available"] is True
    assert first["funding_basis_available"] is True
    assert first["degraded_reason"] is None
    assert first["positioning_bias"] == "bullish"
    assert first["basis_pct"] == 0.002
    assert second == first
    assert calls["oi"] == 1  # cached on second call, no duplicate fetch
    assert calls["fb"] == 1


def test_reference_snapshot_partial_success_is_not_zero_filled(monkeypatch, tmp_path):
    """Item p18 honesty fix: if funding/basis fails but OI succeeds, funding_rate
    and basis_pct must be None (never a fabricated 0.0 indistinguishable from a
    genuinely flat funding rate)."""
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    oi_payload = {
        "open_interest": 5000.0,
        "positioning_signals": {"positioning_bias": "bullish", "bias_strength": 0.7},
    }
    with mock.patch.object(dm, "fetch_binance_open_interest", return_value=oi_payload), mock.patch.object(dm, "fetch_binance_funding_and_basis", return_value={}):
        result = dm.derivatives_reference_snapshot("BTCUSDT", db_path=str(tmp_path / "t.db"))

    assert result["available"] is True
    assert result["oi_available"] is True
    assert result["funding_basis_available"] is False
    assert result["degraded_reason"] == "funding_basis_unavailable"
    assert result["funding_rate"] is None
    assert result["basis_pct"] is None
    assert result["open_interest"] == 5000.0


def test_reference_snapshot_partial_success_other_direction(monkeypatch, tmp_path):
    """OI fails, funding/basis succeeds -> OI fields None, not 0.0."""
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    fb_payload = {"basis_pct": 0.003, "funding_rate": 0.0002}
    with mock.patch.object(dm, "fetch_binance_open_interest", return_value={}), mock.patch.object(dm, "fetch_binance_funding_and_basis", return_value=fb_payload):
        result = dm.derivatives_reference_snapshot("ETHUSDT", db_path=str(tmp_path / "t.db"))

    assert result["available"] is True
    assert result["oi_available"] is False
    assert result["funding_basis_available"] is True
    assert result["degraded_reason"] == "oi_unavailable"
    assert result["open_interest"] is None
    assert result["oi_volume_ratio"] is None
    assert result["funding_rate"] == 0.0002


def test_reference_snapshot_history_change_and_percentile(monkeypatch, tmp_path):
    """Item p18 gap-closure: funding change / percentile / OI change / basis
    change+zscore computed from real persisted history."""
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    db_path = str(tmp_path / "t.db")
    dm._ensure_history_schema(db_path)

    # Seed 10 prior snapshots with a range of funding/basis/OI values.
    for i in range(10):
        dm._persist_history_row(
            "BTCUSDT",
            open_interest=1000.0 + i * 10,
            funding_rate=0.0001 * i,
            basis_pct=0.001 * i,
            db_path=db_path,
        )

    oi_payload = {
        "open_interest": 2000.0,
        "positioning_signals": {"positioning_bias": "bullish", "bias_strength": 0.5},
    }
    fb_payload = {"basis_pct": 0.02, "funding_rate": 0.002}
    with mock.patch.object(dm, "fetch_binance_open_interest", return_value=oi_payload), mock.patch.object(dm, "fetch_binance_funding_and_basis", return_value=fb_payload):
        result = dm.derivatives_reference_snapshot("BTCUSDT", db_path=db_path)

    assert result["history_sample_count"] == 10
    # New OI (2000) is far above the last seeded value (1090) -> positive change
    assert result["open_interest_change_pct"] > 0
    # New funding (0.002) is higher than all 10 seeded values -> percentile == 1.0 (all <= value)
    assert result["funding_rate_percentile"] == 1.0
    assert result["funding_rate_change"] is not None
    assert result["basis_pct_change"] is not None
    assert result["basis_pct_zscore"] is not None


def test_reference_snapshot_history_stats_none_when_insufficient_samples(monkeypatch, tmp_path):
    """Honesty: with < _MIN_HISTORY_SAMPLES_FOR_PERCENTILE prior rows, percentile
    and z-score must be None, not a fabricated neutral value."""
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    db_path = str(tmp_path / "t.db")
    oi_payload = {"open_interest": 100.0, "positioning_signals": {}}
    fb_payload = {"basis_pct": 0.001, "funding_rate": 0.0001}
    with mock.patch.object(dm, "fetch_binance_open_interest", return_value=oi_payload), mock.patch.object(dm, "fetch_binance_funding_and_basis", return_value=fb_payload):
        result = dm.derivatives_reference_snapshot("SOLUSDT", db_path=db_path)

    assert result["history_sample_count"] == 0
    assert result["funding_rate_percentile"] is None
    assert result["basis_pct_zscore"] is None


def test_reference_snapshot_kill_switch_reports_disabled(monkeypatch):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "0")
    result = dm.derivatives_reference_snapshot("XRPUSDT")
    assert result == {"available": False, "degraded_reason": "reference_feed_disabled"}


def test_derivatives_positioning_signal_zero_when_unavailable():
    assert dm.derivatives_positioning_signal({"available": False}) == 0.0
    assert dm.derivatives_positioning_signal({}) == 0.0


def test_derivatives_positioning_signal_bullish_bias():
    snapshot = {
        "available": True,
        "oi_available": True,
        "funding_basis_available": False,
        "positioning_bias": "bullish",
        "bias_strength": 0.8,
        "funding_rate_percentile": None,
    }
    val = dm.derivatives_positioning_signal(snapshot)
    assert 0.0 < val <= 1.0


def test_derivatives_positioning_signal_contrarian_on_crowded_long_funding():
    snapshot = {
        "available": True,
        "oi_available": False,
        "funding_basis_available": True,
        "positioning_bias": None,
        "bias_strength": 0.0,
        "funding_rate_percentile": 1.0,  # funding at all-time-high in the sample window
    }
    val = dm.derivatives_positioning_signal(snapshot)
    assert val < 0.0  # crowded-long contrarian tilt is bearish


def test_derivatives_positioning_signal_bounded():
    snapshot = {
        "available": True,
        "oi_available": True,
        "funding_basis_available": True,
        "positioning_bias": "bullish",
        "bias_strength": 1.0,
        "funding_rate_percentile": 0.0,
    }
    val = dm.derivatives_positioning_signal(snapshot)
    assert -1.0 <= val <= 1.0


def test_signal_check_uses_prefetched_data_no_double_call(monkeypatch):
    monkeypatch.setenv("DERIVATIVES_REFERENCE_FEED_ENABLED", "1")
    oi_payload = {
        "open_interest": 100.0,
        "positioning_signals": {"positioning_bias": "neutral", "bias_strength": 0.0},
    }
    with mock.patch.object(dm, "fetch_binance_open_interest") as mocked:
        result = dm.derivatives_signal_check("BTCUSDT", _data=oi_payload)
    mocked.assert_not_called()
    assert result["positioning_bias"] == "neutral"
