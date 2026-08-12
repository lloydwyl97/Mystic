"""Item p25: SCALP live order/exchange-constraints fail-safe behavior."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp import exchange_constraints as ec
from backend.services.binance_scalp import scalp_order_bridge as sob


def setup_function(_fn):
    ec._CACHE.clear()
    ec._CACHE_TS = 0.0


def test_get_symbol_constraints_labels_fallback_when_refresh_fails(monkeypatch):
    monkeypatch.setattr(ec, "_refresh_cache", lambda: None)  # cache stays empty
    result = ec.get_symbol_constraints("BTCUSDT")
    assert result["is_fallback"] is True
    assert result["min_qty"] == ec._FALLBACK["min_qty"]


def test_get_symbol_constraints_labels_real_data_as_not_fallback(monkeypatch):
    def _fake_refresh():
        ec._CACHE["BTCUSDT"] = {"min_qty": 0.001, "max_qty": 100.0, "step_size": 0.001, "min_notional": 5.0}
        ec._CACHE_TS = time.time()

    monkeypatch.setattr(ec, "_refresh_cache", _fake_refresh)
    result = ec.get_symbol_constraints("BTCUSDT")
    assert result["is_fallback"] is False
    assert result["min_qty"] == 0.001


def test_sign_and_post_raises_on_zero_executed_qty():
    bridge = sob.ScalpOrderBridge("key", "secret")
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "executedQty": "0.0",
        "status": "EXPIRED",
        "orderId": "123",
        "transactTime": 1700000000000,
        "fills": [],
    }
    with mock.patch.object(sob, "_requests") as mocked_requests:
        mocked_requests.post.return_value = fake_response
        with pytest.raises(RuntimeError, match="unfilled"):
            bridge._sign_and_post({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT_IOC"})


def test_sign_and_post_succeeds_on_nonzero_fill():
    bridge = sob.ScalpOrderBridge("key", "secret")
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "executedQty": "0.01",
        "status": "FILLED",
        "orderId": "124",
        "transactTime": 1700000000000,
        "fills": [{"qty": "0.01", "price": "50000.0", "commission": "0.5", "commissionAsset": "USDT"}],
    }
    with mock.patch.object(sob, "_requests") as mocked_requests:
        mocked_requests.post.return_value = fake_response
        fill = bridge._sign_and_post({"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"})
    assert fill.qty == 0.01
    assert fill.fill_price == 50000.0


@pytest.mark.asyncio
async def test_place_buy_returns_none_on_unfilled_order():
    bridge = sob.ScalpOrderBridge("key", "secret")
    bridge.arm()
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "executedQty": "0.0",
        "status": "EXPIRED",
        "orderId": "125",
        "transactTime": 1700000000000,
        "fills": [],
    }
    with mock.patch.object(sob, "_requests") as mocked_requests:
        mocked_requests.post.return_value = fake_response
        fill = await bridge.place_buy("BTCUSDT", 100.0, 50000.0)
    assert fill is None
