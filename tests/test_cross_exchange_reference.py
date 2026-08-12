"""Item p20: cross-exchange informational reference layer (Coinbase public feed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import cross_exchange_reference as cer


def setup_function(_fn):
    cer._CROSS_EX_CACHE.clear()


def test_product_id_mapping_known_top4():
    assert cer._coinbase_product_id("BTCUSDT") == "BTC-USD"
    assert cer._coinbase_product_id("XRPUSDT") == "XRP-USD"


def test_product_id_mapping_generic_fallback():
    assert cer._coinbase_product_id("DOGEUSDT") == "DOGE-USD"


def test_fetch_ticker_degrades_on_unreachable(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(cer, "_http_get", return_value={}):
        assert cer.fetch_coinbase_ticker("BTCUSDT") == {}


def test_fetch_ticker_kill_switch(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "0")
    assert cer.fetch_coinbase_ticker("BTCUSDT") == {}


def test_fetch_ticker_parses_valid_payload(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(cer, "_http_get", return_value={"price": "100.5", "volume": "1000.0"}):
        result = cer.fetch_coinbase_ticker("BTCUSDT")
    assert result["price"] == 100.5
    assert result["volume_24h"] == 1000.0


def test_snapshot_honest_degraded_state(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(cer, "fetch_coinbase_ticker", return_value={}):
        result = cer.cross_exchange_snapshot("BTCUSDT", own_price=100.0)
    assert result["available"] is False
    assert "degraded_reason" in result


def test_snapshot_computes_dislocation_and_volume_ratio(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "1")
    with mock.patch.object(cer, "fetch_coinbase_ticker", return_value={"price": 100.0, "volume_24h": 500.0}):
        result = cer.cross_exchange_snapshot("BTCUSDT", own_price=101.0, own_volume_24h=1000.0)
    assert result["available"] is True
    assert abs(result["dislocation_pct"] - 0.01) < 1e-9
    assert result["volume_ratio_vs_coinbase"] == 2.0


def test_snapshot_caches_and_recomputes_dislocation_with_fresh_own_price(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "1")
    calls = {"n": 0}

    def _fetch(_sym):
        calls["n"] += 1
        return {"price": 100.0, "volume_24h": 500.0}

    with mock.patch.object(cer, "fetch_coinbase_ticker", side_effect=_fetch):
        first = cer.cross_exchange_snapshot("ETHUSDT", own_price=100.0)
        second = cer.cross_exchange_snapshot("ETHUSDT", own_price=102.0)

    assert calls["n"] == 1  # coinbase fetch cached, not re-fetched
    assert first["dislocation_pct"] == 0.0
    assert abs(second["dislocation_pct"] - 0.02) < 1e-9


def test_snapshot_kill_switch_reports_disabled(monkeypatch):
    monkeypatch.setenv("CROSS_EXCHANGE_REFERENCE_FEED_ENABLED", "0")
    result = cer.cross_exchange_snapshot("SOLUSDT", own_price=50.0)
    assert result["available"] is False
    assert result["degraded_reason"] == "reference_feed_disabled"


def test_dislocation_signal_zero_when_unavailable():
    assert cer.cross_exchange_dislocation_signal({"available": False}) == 0.0
    assert cer.cross_exchange_dislocation_signal({}) == 0.0


def test_dislocation_signal_mean_reversion_direction():
    # own price trading RICH vs Coinbase -> mean-reversion signal is negative (bearish tilt)
    rich = cer.cross_exchange_dislocation_signal({"available": True, "dislocation_pct": 0.002})
    cheap = cer.cross_exchange_dislocation_signal({"available": True, "dislocation_pct": -0.002})
    assert rich < 0.0
    assert cheap > 0.0
    assert abs(rich - (-1.0)) < 1e-9


def test_dislocation_signal_bounded_at_extreme_dislocation():
    val = cer.cross_exchange_dislocation_signal({"available": True, "dislocation_pct": 0.05})
    assert val == -1.0
