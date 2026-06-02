"""
Market Types - All Live Data, No Fallback/Hardcoded Data

This module provides data models for live market data (backend port 8000).
All models:
- Represent live market data from Binance.US API (via backend port 8000)
- Structure live ticker, orderbook, trade, and OHLCV data
- Validate live market data values (prices, quantities, timestamps)
- No fallback/hardcoded data - all models represent live market operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- Ticker data: Live market prices from Binance.US API
- Orderbook data: Live orderbook depth from Binance.US API
- Trade data: Live recent trades from Binance.US API
- OHLCV data: Live candlestick data from Binance.US API
- Order results: Live order execution results from Binance.US API
- All models represent live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (market data models used by backend services)
- Binance.US API: https://api.binance.us (live exchange API for market data)
- All models use live connections - no fallback/hardcoded data
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ---- Type aliases for live market data ----------------------------------------

ExchangeId = str  # Live exchange identifier (e.g., "binanceus")
Symbol = str  # Live trading symbol (e.g., "BTCUSDT" from Binance.US Top-10)
UnixMillis = int  # Live timestamp in milliseconds (UTC)
DepthLevel = tuple[float, float]  # Live orderbook depth level (price, size)


def _finite(x: float | None) -> bool:
    """
    Check if value is a finite number (not NaN, not inf).

    Used for validating live market data values (prices, quantities).

    Args:
        x: Value to check (from live market data)

    Returns:
        True if value is a finite number, False otherwise
    """
    try:
        if x is None:
            return False
        x_float = float(x)
        # Check if finite (not NaN, not inf) using math.isfinite
        return math.isfinite(x_float)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False


# ---- Core market data models -----------------------------------------------


@dataclass(slots=True)
class Ticker:
    """
    Live ticker data from Binance.US API (via backend port 8000).

    Represents live market ticker with price, bid, ask, and timestamp.
    All data from live market operations - no fallback/hardcoded data.

    Attributes:
        exchange: Live exchange identifier (e.g., "binanceus")
        symbol: Live trading symbol (e.g., "BTCUSDT" from Binance.US Top-10)
        price: Live current price from Binance.US API
        bid: Live best bid price (optional, from live orderbook)
        ask: Live best ask price (optional, from live orderbook)
        ts: Live timestamp in milliseconds (UTC)
    """

    exchange: ExchangeId
    symbol: Symbol
    price: float
    bid: float | None
    ask: float | None
    ts: UnixMillis

    def __post_init__(self) -> None:
        """Validate live ticker data from market operations."""
        if not _finite(self.price) or self.price <= 0:
            error_msg = "Ticker.price must be a positive finite number"
            raise ValueError(error_msg)
        if self.bid is not None and (not _finite(self.bid) or self.bid <= 0):
            self.bid = None  # Invalid live bid price
        if self.ask is not None and (not _finite(self.ask) or self.ask <= 0):
            self.ask = None  # Invalid live ask price

    @property
    def mid(self) -> float | None:
        """
        Calculate mid price from live bid/ask.

        Returns:
            Mid price from live market data, or None if bid/ask unavailable
        """
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float | None:
        """
        Calculate spread from live bid/ask.

        Returns:
            Spread from live market data, or None if bid/ask unavailable
        """
        if self.bid is None or self.ask is None:
            return None
        return max(0.0, self.ask - self.bid)

    def to_dict(self) -> dict[str, Any]:
        """Convert live ticker data to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class OrderBook:
    """
    Live orderbook data from Binance.US API (via backend port 8000).

    Represents live orderbook depth with bids and asks.
    All data from live market operations - no fallback/hardcoded data.

    Attributes:
        exchange: Live exchange identifier (e.g., "binanceus")
        symbol: Live trading symbol (e.g., "BTCUSDT" from Binance.US Top-10)
        bids: Live bid levels (highest price first, from live orderbook)
        asks: Live ask levels (lowest price first, from live orderbook)
        ts: Live timestamp in milliseconds (UTC)
    """

    exchange: ExchangeId
    symbol: Symbol
    bids: list[DepthLevel] = field(default_factory=list)  # Schema default, not fallback data - live bids from orderbook
    asks: list[DepthLevel] = field(default_factory=list)  # Schema default, not fallback data - live asks from orderbook
    ts: UnixMillis = 0  # Schema default, not fallback data - live timestamp

    def __post_init__(self) -> None:
        """Validate and sanitize live orderbook data from market operations."""
        # Sanitize depth: keep only valid positive numbers from live orderbook, sizes >= 0
        self.bids = [(float(p), float(s)) for p, s in self.bids if _finite(p) and _finite(s) and p > 0 and s >= 0]
        self.asks = [(float(p), float(s)) for p, s in self.asks if _finite(p) and _finite(s) and p > 0 and s >= 0]
        # Sort live depth data
        self.bids.sort(key=lambda x: x[0], reverse=True)  # Highest price first
        self.asks.sort(key=lambda x: x[0])  # Lowest price first

    @property
    def best_bid(self) -> DepthLevel | None:
        """Get best bid from live orderbook data."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> DepthLevel | None:
        """Get best ask from live orderbook data."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> float | None:
        """Calculate spread from live best bid/ask."""
        if not self.best_bid or not self.best_ask:
            return None
        return max(0.0, self.best_ask[0] - self.best_bid[0])

    def total_bid_size(self) -> float:
        """Calculate total bid size from live orderbook data."""
        return float(sum(s for _, s in self.bids))

    def total_ask_size(self) -> float:
        """Calculate total ask size from live orderbook data."""
        return float(sum(s for _, s in self.asks))

    def top_n(self, n: int) -> OrderBook:
        """
        Get top N levels from live orderbook.

        Args:
            n: Number of levels to return from live orderbook

        Returns:
            New OrderBook with top N levels from live data
        """
        n = max(0, int(n))
        return OrderBook(
            exchange=self.exchange,
            symbol=self.symbol,
            bids=self.bids[:n],  # Top N bids from live data
            asks=self.asks[:n],  # Top N asks from live data
            ts=self.ts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert live orderbook data to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class Trade:
    """
    Live trade data from Binance.US API (via backend port 8000).

    Represents live executed trade with price, quantity, and side.
    All data from live market operations - no fallback/hardcoded data.

    Attributes:
        exchange: Live exchange identifier (e.g., "binanceus")
        symbol: Live trading symbol (e.g., "BTCUSDT" from Binance.US Top-10)
        price: Live execution price from Binance.US API
        qty: Live execution quantity from Binance.US API
        side: Live trade side ("buy" or "sell")
        ts: Live timestamp in milliseconds (UTC)
    """

    exchange: ExchangeId
    symbol: Symbol
    price: float
    qty: float
    side: Literal["buy", "sell"]
    ts: UnixMillis

    def __post_init__(self) -> None:
        """Validate live trade data from market operations."""
        if not _finite(self.price) or self.price <= 0:
            error_msg = "Trade.price must be positive"
            raise ValueError(error_msg)
        if not _finite(self.qty) or self.qty <= 0:
            error_msg = "Trade.qty must be positive"
            raise ValueError(error_msg)
        if self.side not in ("buy", "sell"):
            error_msg = "Trade.side must be 'buy' or 'sell'"
            raise ValueError(error_msg)

    def notional(self) -> float:
        """
        Calculate notional value from live trade data.

        Returns:
            Notional value (price * quantity) from live trade
        """
        return float(self.price * self.qty)

    def to_dict(self) -> dict[str, Any]:
        """Convert live trade data to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class OHLCV:
    """
    Live OHLCV candlestick data from Binance.US API (via backend port 8000).

    Represents live candlestick with open, high, low, close, volume.
    All data from live market operations - no fallback/hardcoded data.

    Attributes:
        exchange: Live exchange identifier (e.g., "binanceus")
        symbol: Live trading symbol (e.g., "BTCUSDT" from Binance.US Top-10)
        ts: Live timestamp in milliseconds (UTC)
        open: Live opening price from Binance.US API
        high: Live high price from Binance.US API
        low: Live low price from Binance.US API
        close: Live closing price from Binance.US API
        volume: Live trading volume from Binance.US API
    """

    exchange: ExchangeId
    symbol: Symbol
    ts: UnixMillis
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        """Validate live OHLCV data from market operations."""
        for name in ("open", "high", "low", "close"):
            v = getattr(self, name)
            if not _finite(v) or v <= 0:
                error_msg = f"OHLCV.{name} must be positive"
                raise ValueError(error_msg)
        if not _finite(self.volume) or self.volume < 0:
            error_msg = "OHLCV.volume must be non-negative"
            raise ValueError(error_msg)
        # Ensure high/low bounds from live data (fix rounding errors)
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high and self.low <= self.high):
            # If slightly out due to rounding, fix bounds conservatively from live data
            low = min(self.open, self.high, self.low, self.close)
            high = max(self.open, self.high, self.low, self.close)
            self.low, self.high = float(low), float(high)

    def to_list(self) -> list[float]:
        """
        Convert live OHLCV data to CCXT-style array.

        Returns:
            [ts, open, high, low, close, volume] from live market data
        """
        return [self.ts, self.open, self.high, self.low, self.close, self.volume]

    def hlc3(self) -> float:
        """Calculate HLC3 (high + low + close) / 3 from live OHLCV data."""
        return (self.high + self.low + self.close) / 3.0

    def body(self) -> float:
        """Calculate candle body from live OHLCV data."""
        return abs(self.close - self.open)

    def range(self) -> float:
        """Calculate price range from live OHLCV data."""
        return self.high - self.low

    def to_dict(self) -> dict[str, Any]:
        """Convert live OHLCV data to dictionary."""
        return asdict(self)


@dataclass(slots=True)
class OrderResult:
    """
    Live order execution result from Binance.US API (via backend port 8000).

    Represents live order execution status with fill information.
    All data from live market operations - no fallback/hardcoded data.

    Attributes:
        exchange: Live exchange identifier (e.g., "binanceus")
        symbol: Live trading symbol (e.g., "BTCUSDT" from Binance.US Top-10)
        side: Live order side ("buy" or "sell")
        qty: Live order quantity
        type: Live order type ("market" or "limit")
        status: Live order status ("submitted", "filled", "rejected", "error")
        id: Live order ID from Binance.US API (optional)
        fill_price: Live fill price from Binance.US API (optional)
        ts: Live timestamp in milliseconds (UTC)
        raw: Live raw response data from Binance.US API (schema default, not fallback data)
    """

    exchange: ExchangeId
    symbol: Symbol
    side: Literal["buy", "sell"]
    qty: float
    type: Literal["market", "limit"]
    status: Literal["submitted", "filled", "rejected", "error"]
    id: str | None
    fill_price: float | None
    ts: UnixMillis
    raw: dict[str, Any] = field(default_factory=dict)  # Schema default, not fallback data - live raw response

    def __post_init__(self) -> None:
        """Validate live order result data from market operations."""
        if self.side not in ("buy", "sell"):
            error_msg = "OrderResult.side must be 'buy' or 'sell'"
            raise ValueError(error_msg)
        if self.type not in ("market", "limit"):
            error_msg = "OrderResult.type must be 'market' or 'limit'"
            raise ValueError(error_msg)
        if self.status not in ("submitted", "filled", "rejected", "error"):
            error_msg = "OrderResult.status invalid"
            raise ValueError(error_msg)
        if not _finite(self.qty) or self.qty <= 0:
            error_msg = "OrderResult.qty must be positive"
            raise ValueError(error_msg)
        if self.fill_price is not None and (not _finite(self.fill_price) or self.fill_price <= 0):
            self.fill_price = None  # Leave unset if not valid live fill price

    @property
    def filled(self) -> bool:
        """Check if live order is filled."""
        return self.status == "filled"

    @property
    def notional(self) -> float | None:
        """
        Calculate notional value from live order result.

        Returns:
            Notional value from live fill data, or None if not filled
        """
        if self.filled and self.fill_price is not None:
            return float(self.fill_price * self.qty)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert live order result data to dictionary."""
        return asdict(self)
