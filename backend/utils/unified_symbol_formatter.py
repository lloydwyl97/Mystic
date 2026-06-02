"""
BUG #31 FIX: Unified Symbol Formatter

Centralized symbol normalization to prevent lookup failures.
All code should use this formatter instead of scattered normalization logic.
"""

import logging

logger = logging.getLogger(__name__)


class SymbolFormatter:
    """Centralized symbol formatting with multiple output formats"""

    @staticmethod
    def normalize(symbol: str) -> str:
        """
        Normalize symbol to canonical format: SYMBOL/USDT

        Examples:
            "BTC" -> "BTC/USDT"
            "BTCUSDT" -> "BTC/USDT"
            "BTC/USDT" -> "BTC/USDT"
            "BTC/USD" -> "BTC/USDT" (normalized to USDT)
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        symbol = symbol.upper().strip()

        # Already in correct format
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2 and parts[1] in ("USDT", "USD", "BUSD"):
                return f"{parts[0]}/USDT"
            return symbol

        # Remove common suffixes and add /USDT
        for suffix in ("USDT", "USD", "BUSD"):
            if symbol.endswith(suffix):
                base = symbol[: -len(suffix)]
                return f"{base}/USDT"

        # Default: add /USDT
        return f"{symbol}/USDT"

    @staticmethod
    def to_binance(symbol: str) -> str:
        """
        Convert to Binance format (no slash): SYMBOLUSDT

        Examples:
            "BTC" -> "BTCUSDT"
            "BTC/USDT" -> "BTCUSDT"
        """
        normalized = SymbolFormatter.normalize(symbol)
        return normalized.replace("/", "")

    @staticmethod
    def to_ccxt(symbol: str) -> str:
        """
        Convert to CCXT format: SYMBOL/USDT

        Examples:
            "BTC" -> "BTC/USDT"
            "BTCUSDT" -> "BTC/USDT"
        """
        return SymbolFormatter.normalize(symbol)

    @staticmethod
    def parse_base_quote(symbol: str) -> tuple[str, str]:
        """
        Parse symbol into (base, quote) components

        Examples:
            "BTC/USDT" -> ("BTC", "USDT")
            "BTCUSDT" -> ("BTC", "USDT")
        """
        normalized = SymbolFormatter.normalize(symbol)
        parts = normalized.split("/")
        return (parts[0], parts[1])

    @staticmethod
    def get_base(symbol: str) -> str:
        """Get base asset from symbol"""
        base, _ = SymbolFormatter.parse_base_quote(symbol)
        return base

    @staticmethod
    def is_valid_format(symbol: str) -> bool:
        """Check if symbol is in valid format"""
        try:
            SymbolFormatter.normalize(symbol)
            return True
        except Exception:
            return False


# Export main function for convenience
def normalize_symbol(symbol: str) -> str:
    """Convenience function - main API for symbol normalization"""
    return SymbolFormatter.normalize(symbol)


if __name__ == "__main__":
    # Quick tests
    test_symbols = [
        "BTC",
        "BTCUSDT",
        "BTC/USDT",
        "ETH",
        "ETHUSDT",
        "ETH/USDT",
        "BTC/USD",
    ]

    for sym in test_symbols:
        normalized = SymbolFormatter.normalize(sym)
        binance = SymbolFormatter.to_binance(sym)
        ccxt = SymbolFormatter.to_ccxt(sym)
        logger.info(f"{sym:15} -> normalize: {normalized:12} binance: {binance:10} ccxt: {ccxt:12}")
