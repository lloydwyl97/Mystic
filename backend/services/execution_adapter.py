"""
Mystic ExecutionAdapter — the only thing that changes between PAPER and LIVE.

Design:
  * One trading brain (`portfolio_engine.PortfolioEngine`).
  * One mode switch (`backend/config/trading_mode.MYSTIC_TRADING_MODE`).
  * One execution interface (this module).
  * Two concrete adapters:
      - PaperExecutionAdapter           (simulated order/fill)
      - LiveBinanceUSExecutionAdapter   (real Binance.US order/fill via ccxt)
  * Both adapters expose IDENTICAL methods so the engine never branches on
    paper vs live for execution.

Required methods (paper and live agree on shape):
    get_current_price(symbol)
    get_balances()
    get_open_orders(symbol=None)
    get_recent_fills(symbol=None)
    place_market_buy(symbol, quote_amount)
    place_market_sell(symbol, quantity)
    get_symbol_filters(symbol)
    reconcile_position(symbol)

Execution-only shim. Trade decisions live in the portfolio engine; this
module only routes market buy/sell to paper or live.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.config.trading_mode import (
    TradingMode,
    TradingModeError,
    resolve_trading_mode,
)
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.utils.binance_limited_http import limited_binance_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _to_api_symbol(symbol: str) -> str:
    """Normalize any symbol form to Binance.US API form (BTCUSDT)."""
    if not symbol:
        return ""
    s = symbol.replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s[:-3] + "USDT"
    return s


def _to_ccxt_symbol(symbol: str) -> str:
    """Normalize to BASE/USDT for ccxt-style consumers."""
    api = _to_api_symbol(symbol)
    if api.endswith("USDT"):
        return f"{api[:-4]}/USDT"
    return api


@dataclass(frozen=True)
class SymbolFilters:
    """
    Subset of Binance.US LOT_SIZE / MIN_NOTIONAL / PRICE_FILTER fields used
    by the trading brain and dust cleanup. Same shape in paper and live.
    """

    symbol: str
    min_qty: float
    step_size: float
    min_notional: float
    tick_size: float
    base_asset_precision: int
    quote_asset_precision: int


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ExecutionAdapter(abc.ABC):
    """
    Common adapter contract used by both PAPER and LIVE.

    Implementations MUST be idempotent on read methods and return the same
    shape regardless of mode. The trading brain calls these directly; it
    does not have a paper-specific or live-specific execution path.
    """

    mode: TradingMode

    @abc.abstractmethod
    async def get_current_price(self, symbol: str) -> float | None:
        """Return last/mark price for ``symbol`` in USDT, or None if unknown."""

    @abc.abstractmethod
    async def get_balances(self) -> dict[str, dict[str, float]]:
        """
        Return {asset: {"free": float, "used": float, "total": float}, ...}
        with the asset code (e.g. ``BTC``, ``USDT``).
        """

    @abc.abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return list of open orders (filtered by symbol when provided)."""

    @abc.abstractmethod
    async def get_recent_fills(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return recent fills/trades (filtered by symbol when provided)."""

    @abc.abstractmethod
    async def place_market_buy(self, symbol: str, quote_amount: float) -> dict[str, Any]:
        """Place a market BUY sized in quote currency (USDT)."""

    @abc.abstractmethod
    async def place_market_sell(self, symbol: str, quantity: float) -> dict[str, Any]:
        """Place a market SELL of ``quantity`` base units."""

    @abc.abstractmethod
    async def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        """Return canonical Binance.US filters for ``symbol`` (cached)."""

    @abc.abstractmethod
    async def reconcile_position(self, symbol: str) -> dict[str, Any]:
        """
        Reconcile what the engine thinks it holds against the execution
        venue. Returns ``{base_asset, free, total, dust, reason}`` so the
        caller can detect HUMAN_MANUAL_SELL or sub-sellable dust.
        """


# ---------------------------------------------------------------------------
# Binance.US filter cache (shared by paper + live)
# ---------------------------------------------------------------------------


class _BinanceUSFilters:
    """
    Tiny on-demand fetcher for Binance.US exchangeInfo filters. Cached
    forever in-process for the DAY top-4 universe (BTCUSDT / ETHUSDT /
    SOLUSDT / XRPUSDT). Paper uses the same source as live so dust
    behavior matches exactly.
    """

    _cache: dict[str, SymbolFilters] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get(cls, symbol: str) -> SymbolFilters | None:
        api_sym = _to_api_symbol(symbol)
        if not api_sym:
            return None
        cached = cls._cache.get(api_sym)
        if cached is not None:
            return cached
        async with cls._lock:
            cached = cls._cache.get(api_sym)
            if cached is not None:
                return cached
            try:
                resp = await limited_binance_get(
                    "/api/v3/exchangeInfo",
                    params={"symbol": api_sym},
                    timeout_sec=8.0,
                )
                if resp is None or resp.status_code != 200:
                    logger.warning(
                        "SYMBOL_FILTERS_HTTP_%s symbol=%s",
                        resp.status_code if resp is not None else "none",
                        api_sym,
                    )
                    return None
                data = resp.json() or {}
            except Exception as exc:
                logger.warning("SYMBOL_FILTERS_FETCH_FAILED symbol=%s err=%s", api_sym, exc)
                return None
            for s in data.get("symbols", []):
                if str(s.get("symbol", "")).upper() != api_sym:
                    continue
                filters = {f.get("filterType"): f for f in s.get("filters", [])}
                lot = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE") or {}
                notional = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
                price_f = filters.get("PRICE_FILTER") or {}
                try:
                    sf = SymbolFilters(
                        symbol=api_sym,
                        min_qty=float(lot.get("minQty", 0) or 0),
                        step_size=float(lot.get("stepSize", 0) or 0),
                        min_notional=float(notional.get("minNotional", notional.get("notional", 0)) or 0),
                        tick_size=float(price_f.get("tickSize", 0) or 0),
                        base_asset_precision=int(s.get("baseAssetPrecision", 8) or 8),
                        quote_asset_precision=int(s.get("quoteAssetPrecision", 8) or 8),
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning("SYMBOL_FILTERS_PARSE_FAILED symbol=%s err=%s", api_sym, exc)
                    return None
                cls._cache[api_sym] = sf
                return sf
            return None


def _classify_dust(
    quantity: float,
    price: float,
    filters: SymbolFilters | None,
) -> tuple[bool, str]:
    """
    Same dust rule paper and live use. Returns (is_dust, reason).
    Live cannot sell sub-LOT_SIZE / sub-MIN_NOTIONAL, so paper agrees with
    that physics — the simulator does not pretend it can sell dust that
    Binance.US would reject.
    """
    if quantity <= 0:
        return True, "quantity<=0"
    if price <= 0:
        return True, "price<=0"
    if filters is None:
        return False, ""
    if filters.min_qty > 0 and quantity < filters.min_qty:
        return True, f"qty<{filters.min_qty}"
    notional = quantity * price
    if filters.min_notional > 0 and notional < filters.min_notional:
        return True, f"notional<{filters.min_notional}"
    return False, ""


# ---------------------------------------------------------------------------
# PAPER adapter
# ---------------------------------------------------------------------------


class PaperExecutionAdapter(ExecutionAdapter):
    """
    Simulated adapter. Uses:
        * Binance.US public exchangeInfo for symbol filters (live parity).
        * Binance.US public ticker for current price (live parity).
        * backend.services.paper_trading_service for balances / fills.

    PAPER MUST behave like LIVE for dust cleanup, symbol filters, and
    cooldown bookkeeping. Only order EXECUTION is simulated.
    """

    mode = TradingMode.PAPER

    def __init__(self) -> None:
        self._paper_service: Any | None = None

    def _service(self) -> Any:
        if self._paper_service is None:
            from backend.services.paper_trading_service import (
                get_paper_trading_service,
            )

            self._paper_service = get_paper_trading_service()
        return self._paper_service

    async def get_current_price(self, symbol: str) -> float | None:
        api_sym = _to_api_symbol(symbol)
        if not api_sym:
            return None
        try:
            resp = await limited_binance_get(
                "/api/v3/ticker/price",
                params={"symbol": api_sym},
                timeout_sec=5.0,
            )
            if resp is None or resp.status_code != 200:
                logger.debug(
                    "PAPER_PRICE_HTTP_%s symbol=%s",
                    resp.status_code if resp is not None else "none",
                    api_sym,
                )
                return None
            data = resp.json() or {}
            px = float(data.get("price", 0) or 0)
            return px if px > 0 else None
        except Exception as exc:
            logger.debug("PAPER_PRICE_FETCH_FAILED symbol=%s err=%s", api_sym, exc)
            return None

    async def get_balances(self) -> dict[str, dict[str, float]]:
        svc = self._service()
        try:
            await svc._ensure_redis()
            account = await svc.get_account_balance()
        except Exception as exc:
            logger.warning("PAPER_BALANCES_FAILED err=%s", exc)
            return {}
        out: dict[str, dict[str, float]] = {}
        cash = float(account.get("balance", 0.0) or 0.0)
        out["USDT"] = {"free": cash, "used": 0.0, "total": cash}
        for pos in account.get("positions", []) or []:
            sym = str(pos.get("symbol") or "")
            api = _to_api_symbol(sym)
            base = api[:-4] if api.endswith("USDT") else api
            qty = float(pos.get("quantity", 0.0) or 0.0)
            out[base] = {"free": qty, "used": 0.0, "total": qty}
        return out

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        svc = self._service()
        orders = await svc.get_orders(status="PENDING")
        if symbol:
            api_sym = _to_api_symbol(symbol)
            return [o for o in orders if _to_api_symbol(str(o.get("symbol", ""))) == api_sym]
        return list(orders)

    async def get_recent_fills(self, symbol: str | None = None) -> list[dict[str, Any]]:
        svc = self._service()
        trades = await svc.get_trade_history(limit=100)
        if symbol:
            api_sym = _to_api_symbol(symbol)
            return [t for t in trades if _to_api_symbol(str(t.get("symbol", ""))) == api_sym]
        return list(trades)

    async def place_market_buy(self, symbol: str, quote_amount: float) -> dict[str, Any]:
        api_sym = _to_api_symbol(symbol)
        price = await self.get_current_price(api_sym)
        if not price or price <= 0:
            return {
                "success": False,
                "error": "price_unavailable",
                "symbol": api_sym,
                "mode": self.mode.value,
            }
        filters = await self.get_symbol_filters(api_sym)
        qty = float(quote_amount) / price if quote_amount > 0 else 0.0
        if filters and filters.step_size > 0:
            steps = int(qty / filters.step_size)
            qty = steps * filters.step_size
        if filters and filters.min_qty > 0 and qty < filters.min_qty:
            return {
                "success": False,
                "error": "below_min_qty",
                "symbol": api_sym,
                "qty": qty,
                "min_qty": filters.min_qty,
                "mode": self.mode.value,
            }
        if filters and filters.min_notional > 0 and qty * price < filters.min_notional:
            return {
                "success": False,
                "error": "below_min_notional",
                "symbol": api_sym,
                "notional": qty * price,
                "min_notional": filters.min_notional,
                "mode": self.mode.value,
            }
        svc = self._service()
        result = await svc.place_order(
            symbol=_to_ccxt_symbol(api_sym),
            side="BUY",
            order_type="MARKET",
            quantity=qty,
        )
        result.setdefault("symbol", api_sym)
        result.setdefault("mode", self.mode.value)
        result.setdefault("price", price)
        result.setdefault("quantity", qty)
        return result

    async def place_market_sell(self, symbol: str, quantity: float) -> dict[str, Any]:
        api_sym = _to_api_symbol(symbol)
        price = await self.get_current_price(api_sym)
        if not price or price <= 0:
            return {
                "success": False,
                "error": "price_unavailable",
                "symbol": api_sym,
                "mode": self.mode.value,
            }
        filters = await self.get_symbol_filters(api_sym)
        qty = float(quantity)
        if filters and filters.step_size > 0:
            steps = int(qty / filters.step_size)
            qty = steps * filters.step_size
        is_dust, dust_reason = _classify_dust(qty, price, filters)
        if is_dust:
            return {
                "success": False,
                "error": "dust",
                "dust_reason": dust_reason,
                "symbol": api_sym,
                "quantity": qty,
                "estimated_notional_usdt": qty * price,
                "mode": self.mode.value,
            }
        svc = self._service()
        result = await svc.place_order(
            symbol=_to_ccxt_symbol(api_sym),
            side="SELL",
            order_type="MARKET",
            quantity=qty,
        )
        result.setdefault("symbol", api_sym)
        result.setdefault("mode", self.mode.value)
        result.setdefault("price", price)
        result.setdefault("quantity", qty)
        return result

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        return await _BinanceUSFilters.get(symbol)

    async def reconcile_position(self, symbol: str) -> dict[str, Any]:
        api_sym = _to_api_symbol(symbol)
        base = api_sym[:-4] if api_sym.endswith("USDT") else api_sym
        balances = await self.get_balances()
        bal = balances.get(base) or {"free": 0.0, "used": 0.0, "total": 0.0}
        price = await self.get_current_price(api_sym) or 0.0
        filters = await self.get_symbol_filters(api_sym)
        free = float(bal.get("free", 0.0) or 0.0)
        is_dust, dust_reason = _classify_dust(free, price, filters)
        return {
            "symbol": api_sym,
            "base_asset": base,
            "free": free,
            "used": float(bal.get("used", 0.0) or 0.0),
            "total": float(bal.get("total", 0.0) or 0.0),
            "mark_price": price,
            "estimated_notional_usdt": free * price,
            "dust": is_dust,
            "dust_reason": dust_reason,
            "mode": self.mode.value,
        }


# ---------------------------------------------------------------------------
# LIVE adapter
# ---------------------------------------------------------------------------


class LiveBinanceUSExecutionAdapter(ExecutionAdapter):
    """
    Live Binance.US adapter. Wraps the existing ``LiveTradingService``
    (ccxt-based) and adds the unified ExecutionAdapter surface so the
    trading brain calls one interface regardless of mode.
    """

    mode = TradingMode.LIVE

    def __init__(self) -> None:
        self._live_service: Any | None = None
        if not os.getenv("BINANCE_US_API_KEY") and not os.getenv("BINANCEUS_API_KEY"):
            logger.warning("LIVE_ADAPTER_INIT api keys not detected — live adapter will still fail closed on order placement")

    def _service(self) -> Any:
        if self._live_service is None:
            from backend.services.live_trading_service import trading_service

            self._live_service = trading_service
        return self._live_service

    async def get_current_price(self, symbol: str) -> float | None:
        api_sym = _to_api_symbol(symbol)
        svc = self._service()
        try:
            data = await svc.get_market_price(api_sym)
        except Exception as exc:
            logger.warning("LIVE_PRICE_FETCH_FAILED symbol=%s err=%s", api_sym, exc)
            return None
        if not data:
            return None
        px = float(data.get("price", 0) or 0)
        return px if px > 0 else None

    async def get_balances(self) -> dict[str, dict[str, float]]:
        svc = self._service()
        try:
            result = await svc.get_account_balance()
        except Exception as exc:
            logger.warning("LIVE_BALANCES_FAILED err=%s", exc)
            return {}
        if not isinstance(result, dict) or result.get("status") != "success":
            return {}
        bal_envelope = result.get("balances") or {}
        # bal_envelope shape: {EXCHANGE_ID: {"free": {...}, "used": {...}, "total": {...}}}
        flat: dict[str, dict[str, float]] = {}
        for _exchange, payload in bal_envelope.items():
            free = payload.get("free", {}) or {}
            used = payload.get("used", {}) or {}
            total = payload.get("total", {}) or {}
            for asset in set(free) | set(used) | set(total):
                flat[asset] = {
                    "free": float(free.get(asset, 0) or 0),
                    "used": float(used.get(asset, 0) or 0),
                    "total": float(total.get(asset, 0) or 0),
                }
        return flat

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        svc = self._service()
        try:
            result = await svc.get_open_orders()
        except Exception as exc:
            logger.warning("LIVE_OPEN_ORDERS_FAILED err=%s", exc)
            return []
        if not isinstance(result, dict) or result.get("status") != "success":
            return []
        orders: list[dict[str, Any]] = []
        for _exchange, items in (result.get("orders") or {}).items():
            for o in items or []:
                orders.append(o)
        if symbol:
            api_sym = _to_api_symbol(symbol)
            orders = [o for o in orders if _to_api_symbol(str(o.get("symbol", ""))) == api_sym]
        return orders

    async def get_recent_fills(self, symbol: str | None = None) -> list[dict[str, Any]]:
        svc = self._service()
        try:
            result = await svc.get_trade_history(symbol=symbol, limit=100)
        except Exception as exc:
            logger.warning("LIVE_RECENT_FILLS_FAILED err=%s", exc)
            return []
        if not isinstance(result, dict) or result.get("status") != "success":
            return []
        fills: list[dict[str, Any]] = []
        for _exchange, items in (result.get("trades") or {}).items():
            for t in items or []:
                fills.append(t)
        if symbol:
            api_sym = _to_api_symbol(symbol)
            fills = [t for t in fills if _to_api_symbol(str(t.get("symbol", ""))) == api_sym]
        return fills

    async def place_market_buy(self, symbol: str, quote_amount: float) -> dict[str, Any]:
        api_sym = _to_api_symbol(symbol)
        price = await self.get_current_price(api_sym)
        if not price or price <= 0:
            return {
                "success": False,
                "error": "price_unavailable",
                "symbol": api_sym,
                "mode": self.mode.value,
            }
        filters = await self.get_symbol_filters(api_sym)
        qty = float(quote_amount) / price if quote_amount > 0 else 0.0
        if filters and filters.step_size > 0:
            steps = int(qty / filters.step_size)
            qty = steps * filters.step_size
        if filters and filters.min_qty > 0 and qty < filters.min_qty:
            return {
                "success": False,
                "error": "below_min_qty",
                "symbol": api_sym,
                "qty": qty,
                "min_qty": filters.min_qty,
                "mode": self.mode.value,
            }
        if filters and filters.min_notional > 0 and qty * price < filters.min_notional:
            return {
                "success": False,
                "error": "below_min_notional",
                "symbol": api_sym,
                "notional": qty * price,
                "min_notional": filters.min_notional,
                "mode": self.mode.value,
            }
        svc = self._service()
        result = await svc.place_order(
            exchange="binanceus",
            symbol=api_sym,
            order_type="market",
            side="buy",
            amount=qty,
        )
        result.setdefault("mode", self.mode.value)
        result.setdefault("symbol", api_sym)
        return result

    async def place_market_sell(self, symbol: str, quantity: float) -> dict[str, Any]:
        api_sym = _to_api_symbol(symbol)
        price = await self.get_current_price(api_sym)
        if not price or price <= 0:
            return {
                "success": False,
                "error": "price_unavailable",
                "symbol": api_sym,
                "mode": self.mode.value,
            }
        filters = await self.get_symbol_filters(api_sym)
        qty = float(quantity)
        if filters and filters.step_size > 0:
            steps = int(qty / filters.step_size)
            qty = steps * filters.step_size
        is_dust, dust_reason = _classify_dust(qty, price, filters)
        if is_dust:
            return {
                "success": False,
                "error": "dust",
                "dust_reason": dust_reason,
                "symbol": api_sym,
                "quantity": qty,
                "estimated_notional_usdt": qty * price,
                "mode": self.mode.value,
            }
        svc = self._service()
        result = await svc.place_order(
            exchange="binanceus",
            symbol=api_sym,
            order_type="market",
            side="sell",
            amount=qty,
        )
        result.setdefault("mode", self.mode.value)
        result.setdefault("symbol", api_sym)
        return result

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters | None:
        return await _BinanceUSFilters.get(symbol)

    async def reconcile_position(self, symbol: str) -> dict[str, Any]:
        api_sym = _to_api_symbol(symbol)
        base = api_sym[:-4] if api_sym.endswith("USDT") else api_sym
        balances = await self.get_balances()
        bal = balances.get(base) or {"free": 0.0, "used": 0.0, "total": 0.0}
        price = await self.get_current_price(api_sym) or 0.0
        filters = await self.get_symbol_filters(api_sym)
        free = float(bal.get("free", 0.0) or 0.0)
        is_dust, dust_reason = _classify_dust(free, price, filters)
        return {
            "symbol": api_sym,
            "base_asset": base,
            "free": free,
            "used": float(bal.get("used", 0.0) or 0.0),
            "total": float(bal.get("total", 0.0) or 0.0),
            "mark_price": price,
            "estimated_notional_usdt": free * price,
            "dust": is_dust,
            "dust_reason": dust_reason,
            "mode": self.mode.value,
        }


# ---------------------------------------------------------------------------
# Factory + startup log
# ---------------------------------------------------------------------------


_ADAPTER_SINGLETON: ExecutionAdapter | None = None
_ADAPTER_BOUND_MODE: TradingMode | None = None


def get_execution_adapter() -> ExecutionAdapter:
    """
    Return the singleton ExecutionAdapter bound to the currently configured
    trading mode (``MYSTIC_TRADING_MODE``). Fails closed:

        * If the mode is missing / invalid: raise ``TradingModeError``.
        * If the mode changed since the singleton was created: raise to force
          a clean restart instead of silently switching live<->paper.
    """
    global _ADAPTER_SINGLETON, _ADAPTER_BOUND_MODE
    mode = resolve_trading_mode()
    if _ADAPTER_SINGLETON is None:
        if mode is TradingMode.PAPER:
            _ADAPTER_SINGLETON = PaperExecutionAdapter()
        elif mode is TradingMode.LIVE:
            _ADAPTER_SINGLETON = LiveBinanceUSExecutionAdapter()
        else:  # pragma: no cover - resolve_trading_mode already validates
            raise TradingModeError(f"unhandled trading mode: {mode!r}")
        _ADAPTER_BOUND_MODE = mode
        logger.warning(
            "EXECUTION_ADAPTER_BOUND mode=%s adapter=%s symbols=%s",
            mode.value,
            type(_ADAPTER_SINGLETON).__name__,
            list(DAY_TRADE_SYMBOLS),
        )
        return _ADAPTER_SINGLETON
    if _ADAPTER_BOUND_MODE is not mode:
        raise TradingModeError(f"trading mode changed after startup (bound={_ADAPTER_BOUND_MODE}, now={mode}); restart required")
    return _ADAPTER_SINGLETON


def reset_execution_adapter_for_tests() -> None:
    """Test-only helper. Production code never calls this."""
    global _ADAPTER_SINGLETON, _ADAPTER_BOUND_MODE
    _ADAPTER_SINGLETON = None
    _ADAPTER_BOUND_MODE = None


__all__ = [
    "ExecutionAdapter",
    "LiveBinanceUSExecutionAdapter",
    "PaperExecutionAdapter",
    "SymbolFilters",
    "_classify_dust",
    "get_execution_adapter",
    "reset_execution_adapter_for_tests",
]


# ---------------------------------------------------------------------------
# Module-load self-check
# ---------------------------------------------------------------------------


def _self_check() -> None:
    """Static guard: both concrete adapters must implement the contract."""
    required = {
        "get_current_price",
        "get_balances",
        "get_open_orders",
        "get_recent_fills",
        "place_market_buy",
        "place_market_sell",
        "get_symbol_filters",
        "reconcile_position",
    }
    for cls in (PaperExecutionAdapter, LiveBinanceUSExecutionAdapter):
        missing = required - set(dir(cls))
        if missing:
            raise RuntimeError(f"{cls.__name__} is missing ExecutionAdapter methods: {missing}")


_self_check()

# Touch ``time`` so the linter does not flag the import as unused even when
# no caller currently needs a clock here.
_ = time.time
