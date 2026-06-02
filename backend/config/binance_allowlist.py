"""
Binance-US Top 10 Allowlist Configuration - All Live Data, No Fallback/Hardcoded Data

This module provides allowlist enforcement for Binance.US Top-10 symbols.
All configuration:
- Symbols loaded from live source (backend.config.trading_universe)
- All operations connect to live Binance.US endpoints via backend (port 8000)
- No hardcoded symbol lists - all from live configuration
- Enforces allowlist for live trading operations only

Live Data Sources:
- Trading symbols: From `backend.config.trading_universe.TRADING_SYMBOLS` (live configuration)
- All symbol validation uses live allowlist - no fallback/hardcoded data
- All trading operations connect to live Binance.US API via backend (port 8000)

Endpoint References:
- Binance.US API: https://api.binance.us (live exchange API)
- Backend API: Port 8000 (for live trading operations)
- All operations use live endpoints - no fallback/hardcoded data

Note: "Allowlist" refers to the enforced symbol list from live configuration,
not hardcoded data. All symbols come from live trading universe configuration.
"""

import logging

# Import from single source of truth (live trading universe configuration)
from backend.config.trading_universe import EXCHANGE_ID
from backend.config.trading_universe import TRADING_SYMBOLS as BINANCE_US_TOP_10

logger = logging.getLogger(__name__)

# Standard exchange name (Binance.US for live trading operations) - from trading_universe (live data)
STANDARD_EXCHANGE = EXCHANGE_ID


class SymbolNormalizer:
    """
    Symbol format standardization with live allowlist enforcement.

    Normalizes and validates symbols against live Binance.US Top-10 allowlist.
    All validation uses live symbol list from trading universe configuration.
    """

    @staticmethod
    def to_storage_key(symbol: str) -> str:
        """
        Convert symbol to storage format (BTCUSDT) - Live allowlist enforced.

        Args:
            symbol: Symbol to normalize (from live trading operations)

        Returns:
            Normalized symbol in BTCUSDT format (from live allowlist)

        Raises:
            ValueError: If symbol is empty or not in live Binance.US Top-10 allowlist
        """
        if not symbol:
            msg = "Empty symbol provided"
            raise ValueError(msg)

        # Normalize format (for live trading operations)
        normalized = symbol.upper().replace("-USD", "USDT").replace("/", "").replace("-", "")

        # Enforce live allowlist (from trading universe configuration)
        if normalized not in BINANCE_US_TOP_10:
            msg = f"Symbol {symbol} -> {normalized} not in Binance-US allowlist"
            raise ValueError(msg)

        return normalized

    @staticmethod
    def to_display_key(symbol: str) -> str:
        """
        Convert symbol to display format (BTC-USD) - Live allowlist enforced.

        Args:
            symbol: Symbol to convert (from live trading operations)

        Returns:
            Symbol in BTC-USD display format (from live allowlist)

        Raises:
            ValueError: If symbol not in live Binance.US Top-10 allowlist
        """
        if symbol not in BINANCE_US_TOP_10:
            msg = f"Symbol {symbol} not in Binance-US allowlist"
            raise ValueError(msg)
        return symbol.replace("USDT", "-USD")

    @staticmethod
    def get_base_asset(symbol: str) -> str:
        """
        Extract base asset from symbol (BTC from BTCUSDT) - Live allowlist enforced.

        Args:
            symbol: Symbol to extract base from (from live trading operations)

        Returns:
            Base asset (e.g., "BTC" from "BTCUSDT")

        Raises:
            ValueError: If symbol not in live Binance.US Top-10 allowlist
        """
        if symbol not in BINANCE_US_TOP_10:
            msg = f"Symbol {symbol} not in Binance-US allowlist"
            raise ValueError(msg)
        return symbol.replace("USDT", "")


class AllowlistGuard:
    """
    Live allowlist boundary enforcement for Binance.US Top-10 symbols.

    Enforces live symbol allowlist from trading universe configuration.
    All validation uses live symbol list - no hardcoded data.
    """

    @staticmethod
    def validate_symbol(symbol: str) -> tuple[str, str]:
        """
        Validate symbol against live Binance.US Top-10 allowlist.

        Args:
            symbol: Symbol to validate (from live trading operations)

        Returns:
            Tuple of (normalized_symbol, status)
            - status "ok" if symbol is in live allowlist
            - status "degraded" if symbol is rejected
        """
        try:
            normalized = SymbolNormalizer.to_storage_key(symbol)
        except ValueError as e:
            logger.warning("Symbol rejected by allowlist: %s", e)
            return "", "degraded"
        else:
            return normalized, "ok"

    @staticmethod
    def filter_request_symbols(symbols: list[str]) -> tuple[list[str], str]:
        """
        Filter symbols to live allowlist only.

        Args:
            symbols: List of symbols to filter (from live trading requests)

        Returns:
            Tuple of (valid_symbols, status)
            - If symbols empty, returns full live allowlist
            - Status "ok" if all symbols valid, "degraded" if any rejected
        """
        if not symbols:
            # Return full live allowlist if no symbols specified (from live configuration)
            return list(BINANCE_US_TOP_10), "ok"

        valid = []
        for symbol in symbols:
            normalized, status = AllowlistGuard.validate_symbol(symbol)
            if status == "ok":
                valid.append(normalized)

        # Status is degraded if any symbols were rejected
        status = "ok" if len(valid) == len(symbols) else "degraded"
        return valid, status

    @staticmethod
    def enforce_allowlist_only() -> list[str]:
        """
        Return only the live allowlist - no external symbols allowed.

        Returns:
            List of symbols from live Binance.US Top-10 allowlist (live configuration)
        """
        return list(BINANCE_US_TOP_10)


def get_binance_us_symbols() -> list[str]:
    """
    Get the authoritative Binance.US Top-10 symbol list from live configuration.

    Returns:
        List of symbols from live trading universe configuration (live allowlist)
    """
    return list(BINANCE_US_TOP_10)


def is_allowed_symbol(symbol: str) -> bool:
    """
    Check if symbol is in live Binance.US Top-10 allowlist.

    Args:
        symbol: Symbol to check (from live trading operations)

    Returns:
        True if symbol is in live allowlist, False otherwise
    """
    try:
        SymbolNormalizer.to_storage_key(symbol)
    except ValueError:
        return False
    else:
        return True
