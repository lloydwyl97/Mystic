"""
Symbol utilities for CCXT format conversion
"""


def _to_ccxt_symbol(symbol: str) -> str:
    """
    Convert symbol to CCXT format (BASE/QUOTE)

    Args:
        symbol: Symbol in various formats (BTCUSDT, BTC-USDT, etc.)

    Returns:
        Symbol in CCXT format (BTC/USDT)
    """
    if not symbol:
        return symbol

    # CRITICAL FIX: Use canonical formatter for proper validation and parsing
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        return CanonicalSymbolFormatter.to_ccxt(symbol)
    except Exception:
        pass  # Fallback to old logic below

    # If already in CCXT format (contains /), validate it properly
    if "/" in symbol:
        s = symbol.upper()
        parts = s.split("/")
        if len(parts) == 2:
            base, quote = parts[0].strip(), parts[1].strip()
            # CRITICAL: Detect malformed symbols like "BTCUSDT/USDT"
            if base.endswith("USDT") and quote == "USDT":
                # Double USDT detected - extract real base
                base = base[:-4]
            elif base.endswith("USD") and quote in ("USD", "USDT"):
                base = base[:-3]
            return f"{base}/{quote}"
        return symbol.upper()  # Malformed but has slash

    # Remove common separators
    clean_symbol = symbol.replace("-", "").replace("_", "").upper()

    # Handle USDT pairs
    if clean_symbol.endswith("USDT"):
        base = clean_symbol[:-4]
        return f"{base}/USDT"

    # Handle USD pairs
    if clean_symbol.endswith("USD"):
        base = clean_symbol[:-3]
        return f"{base}/USD"

    # Handle BTC pairs
    if clean_symbol.endswith("BTC"):
        base = clean_symbol[:-3]
        return f"{base}/BTC"

    # Handle ETH pairs
    if clean_symbol.endswith("ETH"):
        base = clean_symbol[:-3]
        return f"{base}/ETH"

    # If no known quote currency, assume USDT
    return f"{clean_symbol}/USDT"


def _from_ccxt_symbol(ccxt_symbol: str) -> str:
    """
    Convert CCXT format symbol to exchange format

    Args:
        ccxt_symbol: Symbol in CCXT format (BTC/USDT)

    Returns:
        Symbol in exchange format (BTCUSDT)
    """
    if not ccxt_symbol or "/" not in ccxt_symbol:
        return ccxt_symbol

    base, quote = ccxt_symbol.split("/", 1)
    return f"{base}{quote}"
