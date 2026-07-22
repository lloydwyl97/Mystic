"""
ScalpOrderBridge — live order placement for SCALP strategy.
ONLY active when SCALP_LIVE=true and SCALP_LIVE_ARMED=true.
Paper fills bypass this entirely.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import requests as _requests

logger = logging.getLogger(__name__)

SCALP_ORDER_TIMEOUT_SEC = float(os.getenv("SCALP_ORDER_TIMEOUT_SEC", "5.0"))
SCALP_ORDER_TYPE = os.getenv("SCALP_ORDER_TYPE", "MARKET")  # MARKET or LIMIT_IOC
SCALP_SLIPPAGE_BPS = float(os.getenv("SCALP_SLIPPAGE_BPS", "5"))  # 0.05% limit price offset

_BINANCE_US_BASE = "https://api.binance.us"


@dataclass
class ScalpFill:
    symbol: str
    side: str          # BUY or SELL
    qty: float
    fill_price: float
    fee_usdt: float
    order_id: str
    timestamp_ms: int
    raw_response: dict


class ScalpOrderBridge:
    """Places real Binance.US orders for the SCALP strategy."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._armed = False

    def arm(self) -> None:
        """Arm the bridge for live order placement. Requires explicit call."""
        self._armed = True
        logger.warning("[SCALP_LIVE] ScalpOrderBridge ARMED — real orders will be placed")

    def disarm(self) -> None:
        self._armed = False
        logger.warning("[SCALP_LIVE] ScalpOrderBridge disarmed")

    async def place_buy(
        self, symbol: str, notional_usdt: float, ref_price: float
    ) -> Optional[ScalpFill]:
        """Place a scalp BUY. Returns ScalpFill on success, None on failure."""
        if not self._armed:
            logger.error("[SCALP_LIVE] place_buy called but bridge not armed")
            return None
        import asyncio
        try:
            return await asyncio.to_thread(self._sync_place_buy, symbol, notional_usdt, ref_price)
        except Exception as e:
            logger.error("[SCALP_LIVE] place_buy failed %s: %s", symbol, e, exc_info=True)
            return None

    async def place_sell(
        self, symbol: str, qty: float, ref_price: float
    ) -> Optional[ScalpFill]:
        """Place a scalp SELL. Returns ScalpFill on success, None on failure."""
        if not self._armed:
            logger.error("[SCALP_LIVE] place_sell called but bridge not armed")
            return None
        import asyncio
        try:
            return await asyncio.to_thread(self._sync_place_sell, symbol, qty, ref_price)
        except Exception as e:
            logger.error("[SCALP_LIVE] place_sell failed %s: %s", symbol, e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Synchronous REST helpers (run in thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _sync_place_buy(self, symbol: str, notional: float, ref_price: float) -> ScalpFill:
        """Market or LIMIT_IOC BUY using quoteOrderQty (notional) for MARKET,
        or base qty for LIMIT_IOC."""
        from backend.services.binance_scalp.exchange_constraints import round_qty_to_step

        params: dict = {"symbol": symbol, "side": "BUY", "type": SCALP_ORDER_TYPE}
        if SCALP_ORDER_TYPE == "MARKET":
            params["quoteOrderQty"] = str(round(notional, 2))
        else:
            qty = round_qty_to_step(symbol, notional / ref_price)
            params["quantity"] = str(qty)
            params["price"] = str(self._limit_price("BUY", ref_price))
            params["timeInForce"] = "IOC"
        return self._sign_and_post(params)

    def _sync_place_sell(self, symbol: str, qty: float, ref_price: float) -> ScalpFill:
        """Market or LIMIT_IOC SELL using base qty."""
        from backend.services.binance_scalp.exchange_constraints import round_qty_to_step

        qty_rounded = round_qty_to_step(symbol, qty)
        params: dict = {
            "symbol": symbol,
            "side": "SELL",
            "type": SCALP_ORDER_TYPE,
            "quantity": str(qty_rounded),
        }
        if SCALP_ORDER_TYPE == "LIMIT_IOC":
            params["price"] = str(self._limit_price("SELL", ref_price))
            params["timeInForce"] = "IOC"
        return self._sign_and_post(params)

    def _limit_price(self, side: str, ref: float) -> float:
        offset = ref * SCALP_SLIPPAGE_BPS / 10_000
        return round(ref + offset if side == "BUY" else ref - offset, 2)

    def _sign_and_post(self, params: dict) -> ScalpFill:
        """Add timestamp + HMAC-SHA256 signature, POST to /api/v3/order, parse fill."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urllib.parse.urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        headers = {"X-MBX-APIKEY": self._api_key}
        resp = _requests.post(
            f"{_BINANCE_US_BASE}/api/v3/order",
            params=params,
            headers=headers,
            timeout=SCALP_ORDER_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data: dict = resp.json()

        fills = data.get("fills") or []
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            avg_price = (
                sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                if total_qty > 0
                else 0.0
            )
        else:
            avg_price = float(data.get("price") or 0.0)

        fee_usdt = sum(
            float(f["commission"])
            for f in fills
            if f.get("commissionAsset") == "USDT"
        )

        return ScalpFill(
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            qty=float(data["executedQty"]),
            fill_price=avg_price,
            fee_usdt=fee_usdt,
            order_id=str(data["orderId"]),
            timestamp_ms=int(data["transactTime"]),
            raw_response=data,
        )
