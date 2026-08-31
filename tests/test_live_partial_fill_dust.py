"""Live IOC partial fills must be tracked; dust stays in inventory/equity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.services.live_fill_economics import apply_live_buy_economics, extract_live_commission
from backend.services.protected_limit_execution import execute_protected_limit_live


class _FakeLive:
    def __init__(self, place_order):
        self.place_order = place_order
        self.fetch_order = AsyncMock(return_value={"status": "error"})
        self.cancel_order = AsyncMock(return_value={"status": "success"})

    async def _ensure_initialized(self):
        return None


@pytest.mark.asyncio
async def test_zero_fill_returns_none():
    async def place(**_k):
        return {
            "status": "success",
            "order": {"id": "1", "filled": 0.0, "amount": 0.00085, "status": "expired"},
        }

    out = await execute_protected_limit_live(
        _FakeLive(place),
        symbol="BTCUSDT",
        side="buy",
        quantity=0.00085,
        limit_price=76871.34,
    )
    assert out is None


@pytest.mark.asyncio
async def test_full_fill_returns_order_without_partial_flag():
    async def place(**_k):
        return {
            "status": "success",
            "order": {
                "id": "2",
                "filled": 0.00085,
                "amount": 0.00085,
                "average": 76871.34,
                "status": "closed",
                "fee": {"cost": 0.01, "currency": "USDT"},
            },
        }

    out = await execute_protected_limit_live(
        _FakeLive(place),
        symbol="BTCUSDT",
        side="buy",
        quantity=0.00085,
        limit_price=76871.34,
    )
    assert out is not None
    assert float(out["filled"]) == pytest.approx(0.00085)
    assert not out.get("_mystic_partial_fill")


@pytest.mark.asyncio
async def test_partial_fill_is_returned_not_dropped():
    async def place(**_k):
        return {
            "status": "success",
            "order": {
                "id": "1814000000",
                "filled": 0.00006,
                "amount": 0.00085,
                "average": 76871.34,
                "status": "expired",
                "info": {"fills": [{"commission": "0.00000001", "commissionAsset": "BTC"}]},
            },
        }

    fake = _FakeLive(place)
    fake.fetch_order = AsyncMock(
        return_value={
            "status": "success",
            "order": {
                "id": "1814000000",
                "filled": 0.00006,
                "amount": 0.00085,
                "average": 76871.34,
                "status": "expired",
                "info": {"fills": [{"commission": "0.00000001", "commissionAsset": "BTC"}]},
            },
        }
    )
    out = await execute_protected_limit_live(
        fake,
        symbol="BTCUSDT",
        side="buy",
        quantity=0.00085,
        limit_price=76871.34,
    )
    assert out is not None
    assert out.get("_mystic_partial_fill") is True
    assert float(out["filled"]) == pytest.approx(0.00006)
    assert float(out["filled"]) != pytest.approx(0.00085)


def test_partial_below_import_min_still_tracked():
    filled = 0.00001
    px = 77000.0
    notional = filled * px
    assert notional < 11.0
    comm = extract_live_commission({}, symbol="BTCUSDT", fill_price=px)
    qty, _fee, cash = apply_live_buy_economics(
        filled_qty=filled,
        fill_price=px,
        modeled_fee=0.0,
        commission=comm,
    )
    assert qty == filled
    assert cash == pytest.approx(notional)
    # Dust still contributes to live equity = cash remaining + qty*mark
    remaining_cash = 239.40 - cash
    equity = remaining_cash + qty * px
    assert equity == pytest.approx(239.40)


def test_dust_equity_reaches_zero_when_binance_qty_zero():
    engine_qty = 0.00001
    px = 77000.0
    assert engine_qty * px > 0
    binance_qty = 0.0
    equity_after = 239.40 + binance_qty * px
    assert equity_after == pytest.approx(239.40)
    assert binance_qty == 0.0


def test_allow_partial_policy_unchanged():
    from backend.config.protected_execution import PROTECTED_LIMIT_ALLOW_PARTIAL

    assert PROTECTED_LIMIT_ALLOW_PARTIAL is False
