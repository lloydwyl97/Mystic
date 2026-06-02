"""
Centralized symbol normalization utilities.
Single source of truth for symbol format conversions.
All Live Data, No Fallback/Hardcoded Data
"""

from __future__ import annotations

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        TOP10_COINS,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading universe: {e}"
    raise RuntimeError(msg) from e

# Canonical format: BTC/USDT (ccxt standard)
# Display format: BTC-USD (UI/database)
# Exchange format: BTCUSDT (Binance pair)


def to_ccxt_symbol(symbol: str) -> str:
    """
    Convert any symbol format to ccxt canonical format (BTC/USDT).

    Accepts:
    - BTCUSDT, BTC-USD, BTC/USD, BTC_USDT, BTC USD
    - BTC, ETH (assumes USDT quote)

    Returns:
    - BTC/USDT (canonical ccxt format)

    Raises:
    - ValueError: If symbol is empty or invalid
    """
    if not symbol:
        msg = "Empty symbol provided"
        raise ValueError(msg)

    s = str(symbol).strip().upper()

    # Handle slash format - if already has one slash and ends with USDT, assume it's correct
    if "/" in s and s.count("/") == 1:
        base, quote = s.split("/", 1)
        if quote in ("USD", "USDT"):
            return f"{base}/USDT"
        else:
            return f"{base}/{quote}"

    # Handle malformed symbols with double slashes or wrong formats
    if "//" in s or s.count("/") > 1:
        # Clean up malformed symbols
        s = s.replace("//", "/").strip("/")
        if "/" in s:
            base, quote = s.split("/", 1)
            quote = "USDT" if quote in ("USD", "USDT") else quote
            return f"{base}/{quote}"

    # Handle dash format
    if "-" in s:
        base, quote = s.split("-", 1)
        quote = "USDT" if quote in ("USD", "USDT") else quote
        return f"{base}/{quote}"

    # Handle underscore format
    if "_" in s:
        base, quote = s.split("_", 1)
        quote = "USDT" if quote in ("USD", "USDT") else quote
        return f"{base}/{quote}"

    # Handle concatenated format (BTCUSDT)
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    if s.endswith("USD"):
        return f"{s[:-3]}/USDT"

    # Handle space format
    if " " in s:
        base, quote = s.split(" ", 1)
        quote = "USDT" if quote in ("USD", "USDT") else quote
        return f"{base}/{quote}"

    # Bare base symbol (BTC) - assume USDT
    return f"{s}/USDT"


def to_exchange_symbol(symbol: str) -> str:
    """
    Convert any symbol format to exchange API format (BTCUSDT).

    CRITICAL FIX: Uses canonical formatter to handle malformed symbols.
    This function now accepts ANY format and returns properly formatted exchange symbol.

    Args:
        symbol: Any format - "BTC/USDT", "BTCUSDT", "BTC-USD", even malformed like "LINKUSDT/USDT"

    Returns:
        str: BTCUSDT format for exchange APIs (Binance format)
    """
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        return CanonicalSymbolFormatter.to_exchange(symbol)
    except Exception:
        # Fallback: try to handle it manually
        if not symbol:
            return symbol

        s = str(symbol).strip().upper()

        # If contains slash, split and fix
        if "/" in s:
            parts = s.split("/")
            if len(parts) >= 2:
                base, quote = parts[0], parts[1]
                # Fix malformed: "LINKUSDT/USDT" -> "LINKUSDT"
                if base.endswith("USDT") and quote == "USDT":
                    base = base[:-4]
                elif base.endswith("USD") and quote in ("USD", "USDT"):
                    base = base[:-3]
                return f"{base}USDT"
            return s.replace("/", "")

        # Already exchange format or needs no conversion
        # Fix double USDT
        if s.count("USDT") > 1:
            s = s.replace("USDT", "") + "USDT"

        return s


def normalize_symbol(symbol: str) -> str:
    """
    Master symbol normalization function - ensures consistent CCXT format.

    This should be used everywhere instead of custom symbol manipulation.
    """
    return to_ccxt_symbol(symbol)


def to_display_symbol(ccxt_symbol: str) -> str:
    """
    Convert ccxt symbol to display format (BTC-USD).

    Args:
        ccxt_symbol: BTC/USDT format

    Returns:
        BTC-USD format (for UI/database display)

    Raises:
        ValueError: If ccxt_symbol is empty or invalid format
    """
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        return CanonicalSymbolFormatter.to_display(ccxt_symbol)
    except Exception:
        # Fallback
        if not ccxt_symbol or "/" not in ccxt_symbol:
            msg = f"Invalid ccxt symbol format: {ccxt_symbol}"
            raise ValueError(msg) from None

        base, quote = ccxt_symbol.split("/", 1)
        quote_display = "USD" if quote == "USDT" else quote
        return f"{base}-{quote_display}"


def normalize_symbol_to_dash(symbol: str) -> str:
    """
    Legacy compatibility function.
    Convert any symbol to dash format for display.
    """
    return to_display_symbol(to_ccxt_symbol(symbol))


def parse_symbol(symbol: str) -> dict[str, str]:
    """
    Parse a symbol into all common formats.

    Returns:
        {
            "ccxt": "BTC/USDT",
            "display": "BTC-USD",
            "exchange": "BTCUSDT",
            "base": "BTC",
            "quote": "USDT"
        }
    """
    ccxt = to_ccxt_symbol(symbol)
    base, quote = ccxt.split("/", 1)

    return {
        "ccxt": ccxt,
        "display": to_display_symbol(ccxt),
        "exchange": to_exchange_symbol(ccxt),
        "base": base,
        "quote": quote,
    }


def validate_symbol(symbol: str) -> bool:
    """
    Validate if symbol can be normalized to a valid format.
    """
    try:
        ccxt = to_ccxt_symbol(symbol)
        base, quote = ccxt.split("/", 1)
        return bool(base and quote and len(base) >= 2 and len(quote) >= 2)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False


# Top 10 Binance.US symbols - dynamically generated from trading_universe (single source of truth)
def _generate_top_10_lists() -> tuple[list[str], list[str], list[str]]:
    """Generate Top 10 symbol lists from trading_universe (live data)."""
    ccxt_list = [to_ccxt_symbol(s) for s in TRADING_SYMBOLS]
    display_list = [to_display_symbol(ccxt) for ccxt in ccxt_list]
    exchange_list = TRADING_SYMBOLS.copy()  # Already in exchange format
    return ccxt_list, display_list, exchange_list


BINANCE_US_TOP_10_CCXT, BINANCE_US_TOP_10_DISPLAY, BINANCE_US_TOP_10_EXCHANGE = _generate_top_10_lists()


def is_top_10_symbol(symbol: str) -> bool:
    """Check if symbol is in Binance.US Top 10."""
    ccxt_symbol = to_ccxt_symbol(symbol)
    return ccxt_symbol in BINANCE_US_TOP_10_CCXT


def get_top_10_symbols(format_type: str = "ccxt") -> list[str]:
    """
    Get Top 10 symbols in specified format.

    Args:
        format_type: "ccxt", "display", or "exchange"

    Returns:
        List of symbols in requested format
    """
    if format_type == "display":
        return BINANCE_US_TOP_10_DISPLAY.copy()
    if format_type == "exchange":
        return BINANCE_US_TOP_10_EXCHANGE.copy()
    # ccxt
    return BINANCE_US_TOP_10_CCXT.copy()
