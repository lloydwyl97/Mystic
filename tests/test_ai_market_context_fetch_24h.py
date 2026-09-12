"""ctx_change_24h_pct must refresh from a live 24h ticker, not a frozen cache."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.ai_market_context import AIMarketContextService


@pytest.mark.asyncio
async def test_fetch_24h_force_refreshes_ticker(monkeypatch):
    svc = AIMarketContextService(symbols=["BTCUSDT"])
    ticker = mock.AsyncMock(
        return_value={
            "percentage": -0.19,
            "change_24h": -0.19,
            "volume_24h": 480000.0,
        }
    )
    monkeypatch.setattr(
        "backend.services.ai_market_context.live_market_data_service.get_ticker",
        ticker,
    )
    out = await svc._fetch_24h("BTCUSDT")
    ticker.assert_awaited_once()
    assert ticker.await_args.kwargs.get("force_refresh") is True
    assert out["change_24h_pct"] == pytest.approx(-0.0019)
    assert out["volume_24h_usd"] == 480000.0


@pytest.mark.asyncio
async def test_fetch_24h_uses_1m_change_when_ticker_missing(monkeypatch):
    svc = AIMarketContextService(symbols=["SOLUSDT"])
    monkeypatch.setattr(
        "backend.services.ai_market_context.live_market_data_service.get_ticker",
        mock.AsyncMock(return_value=None),
    )
    monkeypatch.setattr(svc, "_change_and_volume_from_1m", mock.AsyncMock(return_value=(-0.00187, 490000.0)))
    out = await svc._fetch_24h("SOLUSDT")
    assert out["change_24h_pct"] == pytest.approx(-0.00187)
    assert out["volume_24h_usd"] == 490000.0
