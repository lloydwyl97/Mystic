"""
Canonical Symbol Formatter - Single Source of Truth
====================================================

This module is the ONE symbol formatter for the entire system.
All symbol conversions MUST go through these functions.

Canonical internal/production format is Binance.US API:
- "BTCUSDT" (no separator) — this is what `normalize_symbol` returns.

Other formats are derived and used only when crossing an external boundary:
- CCXT (external library): "BTC/USDT" - returned only by `to_ccxt`.
- Display (user-facing UI only): "BTC-USD" - returned only by `to_display`.
- Base asset alone: "BTC".

Redis Key Standards:
- AI decisions: "ai_decision:BTCUSDT" (Exchange format)
- Market data: "market:BTC" (Base format)
- Prices: "price:BTC" (Base format)
"""

from __future__ import annotations

import re

# Valid quote currencies (Binance.US only supports USDT for crypto pairs)
VALID_QUOTES = {"USDT", "USD", "BTC", "ETH"}
DEFAULT_QUOTE = "USDT"


class SymbolFormatError(ValueError):
    """Raised when symbol format is invalid or cannot be parsed"""

    pass


class CanonicalSymbolFormatter:
    """
    Canonical symbol formatter with strict validation.

    This class ensures all symbols are properly formatted and validated
    before being used in the system.
    """

    @staticmethod
    def parse_symbol(symbol: str) -> tuple[str, str]:
        """
        Parse any symbol format into (base, quote) tuple.

        Args:
            symbol: Any format - "BTCUSDT", "BTC/USDT", "BTC-USD", "BTC USDT"

        Returns:
            (base, quote) tuple - ("BTC", "USDT")

        Raises:
            SymbolFormatError: If symbol is invalid or cannot be parsed
        """
        if not symbol or not isinstance(symbol, str):
            raise SymbolFormatError(f"Invalid symbol: {symbol}")

        # Clean and uppercase
        s = str(symbol).strip().upper()

        if not s:
            raise SymbolFormatError("Empty symbol after cleaning")

        # Handle slash format: BTC/USDT or BTC/USD
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                raise SymbolFormatError(f"Invalid slash format: {symbol} (expected BASE/QUOTE)")

            base, quote = parts[0].strip(), parts[1].strip()

            # Critical validation: detect malformed symbols like "BTCUSDT/USDT"
            if base.endswith("USDT") and quote == "USDT":
                # Double USDT - extract real base
                base = base[:-4]
            elif base.endswith("USD") and quote in ("USD", "USDT"):
                # Double USD - extract real base
                base = base[:-3]

            # Normalize quote
            if quote == "USD":
                quote = "USDT"  # Binance.US uses USDT

            if not base or not quote:
                raise SymbolFormatError(f"Empty base or quote in: {symbol}")

            return base, quote

        # Handle dash format: BTC-USD or BTC-USDT
        if "-" in s:
            parts = s.split("-")
            if len(parts) != 2:
                raise SymbolFormatError(f"Invalid dash format: {symbol}")

            base, quote = parts[0].strip(), parts[1].strip()

            # Normalize USD to USDT
            if quote == "USD":
                quote = "USDT"

            if not base or not quote:
                raise SymbolFormatError(f"Empty base or quote in: {symbol}")

            return base, quote

        # Handle space format: "BTC USDT"
        if " " in s:
            parts = s.split()
            if len(parts) != 2:
                raise SymbolFormatError(f"Invalid space format: {symbol}")

            base, quote = parts[0], parts[1]

            if quote == "USD":
                quote = "USDT"

            return base, quote

        # Handle underscore format: BTC_USDT
        if "_" in s:
            parts = s.split("_")
            if len(parts) != 2:
                raise SymbolFormatError(f"Invalid underscore format: {symbol}")

            base, quote = parts[0].strip(), parts[1].strip()

            if quote == "USD":
                quote = "USDT"

            return base, quote

        # Handle concatenated format: BTCUSDT (most complex)
        # Try USDT suffix first (4 chars)
        if s.endswith("USDT") and len(s) > 4:
            base = s[:-4]
            # Validate base doesn't accidentally contain USDT again
            if base.endswith("USDT"):
                raise SymbolFormatError(f"Malformed symbol with double USDT: {symbol}")
            return base, "USDT"

        # Try USD suffix (3 chars)
        if s.endswith("USD") and len(s) > 3:
            base = s[:-3]
            # Make sure we didn't extract USDT partially (like "BTC" from "BTCUSDT")
            # Check if adding T would make USDT
            if s.endswith("USDT"):
                # This was actually USDT, handle above
                base = s[:-4]
                return base, "USDT"
            return base, "USDT"  # Normalize to USDT

        # Try BTC suffix (3 chars) for BTC pairs
        if s.endswith("BTC") and len(s) > 3:
            base = s[:-3]
            return base, "BTC"

        # Try ETH suffix (3 chars) for ETH pairs
        if s.endswith("ETH") and len(s) > 3:
            base = s[:-3]
            return base, "ETH"

        # Bare symbol (just base) - assume USDT quote
        # Validate it's a reasonable base symbol (2-10 alphanumeric chars)
        if re.match(r"^[A-Z0-9]{2,10}$", s):
            return s, DEFAULT_QUOTE

        raise SymbolFormatError(f"Cannot parse symbol format: {symbol}")

    @staticmethod
    def to_ccxt(symbol: str) -> str:
        """
        Convert to CCXT format: BTC/USDT

        Args:
            symbol: Any format

        Returns:
            CCXT format string
        """
        base, quote = CanonicalSymbolFormatter.parse_symbol(symbol)
        return f"{base}/{quote}"

    @staticmethod
    def to_exchange(symbol: str) -> str:
        """
        Convert to Exchange/Binance format: BTCUSDT

        Args:
            symbol: Any format

        Returns:
            Exchange format string (no separator)
        """
        base, quote = CanonicalSymbolFormatter.parse_symbol(symbol)
        return f"{base}{quote}"

    @staticmethod
    def to_display(symbol: str) -> str:
        """
        Convert to Display format: BTC-USD

        Args:
            symbol: Any format

        Returns:
            Display format string (USDT shown as USD)
        """
        base, quote = CanonicalSymbolFormatter.parse_symbol(symbol)
        # Display USDT as USD for user-facing interfaces
        display_quote = "USD" if quote == "USDT" else quote
        return f"{base}-{display_quote}"

    @staticmethod
    def to_base(symbol: str) -> str:
        """
        Extract base currency: BTC

        Args:
            symbol: Any format

        Returns:
            Base currency only
        """
        base, _ = CanonicalSymbolFormatter.parse_symbol(symbol)
        return base

    @staticmethod
    def validate(symbol: str) -> bool:
        """
        Validate if symbol can be parsed.

        Args:
            symbol: Symbol to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            base, quote = CanonicalSymbolFormatter.parse_symbol(symbol)
            # Additional validation
            if len(base) < 2 or len(base) > 10:
                return False
            return quote in VALID_QUOTES
        except (SymbolFormatError, ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def normalize_for_redis_ai_decision(symbol: str) -> str:
        """
        Normalize symbol for Redis ai_decision keys.

        Redis key format: "ai_decision:BTCUSDT"

        Args:
            symbol: Any format

        Returns:
            Exchange format (BTCUSDT)
        """
        return CanonicalSymbolFormatter.to_exchange(symbol)

    @staticmethod
    def normalize_for_redis_market(symbol: str) -> str:
        """
        Normalize symbol for Redis market data keys.

        Redis key format: "market:BTC"

        Args:
            symbol: Any format

        Returns:
            Base currency only
        """
        return CanonicalSymbolFormatter.to_base(symbol)

    @staticmethod
    def normalize_for_redis_price(symbol: str) -> str:
        """
        Normalize symbol for Redis price keys.

        Redis key format: "price:BTC"

        Args:
            symbol: Any format

        Returns:
            Base currency only
        """
        return CanonicalSymbolFormatter.to_base(symbol)


# Convenience functions for backward compatibility
def to_ccxt_symbol(symbol: str) -> str:
    """Convert symbol to CCXT format (BTC/USDT)"""
    return CanonicalSymbolFormatter.to_ccxt(symbol)


def to_exchange_symbol(symbol: str) -> str:
    """Convert symbol to Exchange format (BTCUSDT)"""
    return CanonicalSymbolFormatter.to_exchange(symbol)


def to_display_symbol(symbol: str) -> str:
    """Convert symbol to Display format (BTC-USD)"""
    return CanonicalSymbolFormatter.to_display(symbol)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to canonical internal/exchange format (BTCUSDT)."""
    return CanonicalSymbolFormatter.to_exchange(symbol)


def parse_symbol(symbol: str) -> tuple[str, str]:
    """Parse symbol into (base, quote) tuple"""
    return CanonicalSymbolFormatter.parse_symbol(symbol)


def validate_symbol(symbol: str) -> bool:
    """Validate if symbol format is correct"""
    return CanonicalSymbolFormatter.validate(symbol)


# Export the formatter class and convenience functions
__all__ = [
    "CanonicalSymbolFormatter",
    "SymbolFormatError",
    "normalize_symbol",
    "parse_symbol",
    "to_ccxt_symbol",
    "to_display_symbol",
    "to_exchange_symbol",
    "validate_symbol",
]
