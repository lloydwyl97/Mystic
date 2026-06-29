"""Canonical mark must ignore stale redis market:{base} and align with Binance REST."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.canonical_mark_price import fetch_canonical_mark


@pytest.mark.asyncio
async def test_canonical_mark_prefers_binance_over_stale_redis():
    book = {"bid": 1.1088, "ask": 1.1089, "mid": 1.10885}

    with patch(
        "backend.services.canonical_mark_price._fetch_binance_book_ticker",
        new=AsyncMock(return_value=book),
    ):
        with patch(
            "backend.services.canonical_mark_price._fetch_binance_1m_close",
            new=AsyncMock(return_value=1.1088),
        ):
            mark = await fetch_canonical_mark("XRP/USDT", use_cache=False)

    assert mark is not None
    assert mark.mark == pytest.approx(1.10885)
    assert mark.source == "binance_book_ticker_mid"
    assert mark.fresh is True
    assert mark.symbol_format == "XRPUSDT"


@pytest.mark.asyncio
async def test_stale_redis_mark_string_not_used_when_binance_available():
    """Regression: redis market:XRP=1.1002 must not override live book mid ~1.1088."""
    book = {"bid": 1.1088, "ask": 1.1089, "mid": 1.10885}

    with patch(
        "backend.services.canonical_mark_price._fetch_binance_book_ticker",
        new=AsyncMock(return_value=book),
    ):
        with patch(
            "backend.services.canonical_mark_price._fetch_binance_1m_close",
            new=AsyncMock(return_value=1.1088),
        ):
            mark = await fetch_canonical_mark("XRP/USDT", use_cache=False)

    assert mark is not None
    assert abs(mark.mark - 1.1002) > 0.005
