"""Canonical candle API — one contract for every supported timeframe."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from backend.config.canonical_candle_intervals import CANONICAL_CANDLE_INTERVALS, CANONICAL_SYMBOLS
from backend.services.canonical_candle_pipeline import canonical_candle_pipeline, get_canonical_candles
from backend.services.canonical_candle_store import refuse_research_table_read

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/market/candles")
async def market_candles(
    symbol: str = Query(..., description="BTCUSDT / BTC-USDT / BTC/USDT"),
    interval: str = Query("1m"),
    limit: int = Query(300, ge=1, le=2000),
    include_forming: bool = Query(True),
) -> dict[str, Any]:
    payload = await get_canonical_candles(symbol, interval.strip().lower(), limit=limit, include_forming=include_forming)
    status = 200 if payload.get("success") else 404
    payload["http_status"] = status
    return payload


@router.get("/api/market/candles/status")
async def market_candles_status(full: bool = Query(False)) -> dict[str, Any]:
    matrix = await canonical_candle_pipeline.status_matrix(full=full)
    return {"success": True, **matrix, "research_table": refuse_research_table_read()}


@router.get("/api/market/candles/intervals")
async def market_candle_intervals() -> dict[str, Any]:
    return {
        "success": True,
        "symbols": list(CANONICAL_SYMBOLS),
        "intervals": list(CANONICAL_CANDLE_INTERVALS),
        "research_table": refuse_research_table_read(),
    }
