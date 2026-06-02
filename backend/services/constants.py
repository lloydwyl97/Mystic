"""
Centralized constants for backend services.
Single source of truth for exchange IDs and symbol normalization helpers.
"""

from __future__ import annotations

from typing import Final

# Import and re-export from single source of truth
from backend.config.trading_universe import EXCHANGE_ID

# Exchange ID - standardized naming (ONLY use this ID everywhere) - from trading_universe (live data)
# EXCHANGE_ID is now imported from trading_universe above

# Supported exchanges mapping (normalized to our EXCHANGE_ID)
EXCHANGE_MAPPING: Final[dict[str, str]] = {
    "binance": EXCHANGE_ID,
    "binanceus": EXCHANGE_ID,
    "binance_us": EXCHANGE_ID,
    "binance.us": EXCHANGE_ID,
}


def normalize_exchange_id(exchange_name: str | None) -> str:
    """Normalize any exchange name to our standard binance_us format."""
    if not exchange_name:
        return EXCHANGE_ID
    normalized = exchange_name.strip().lower().replace("-", "_").replace(".", "_")
    return EXCHANGE_MAPPING.get(normalized, EXCHANGE_ID)


def _to_ccxt_symbol(sym: str) -> str:
    """
    Normalize to ccxt symbol: BTCUSDT/BTC_USDT/BTC-USDT/BTC USDT/BTC -> BTC/USDT.
    - If there's no quote, default to USDT.
    - Only returns 'BASE/QUOTE' form.
    """
    s = (sym or "").strip().upper().replace("-", "/").replace("_", "/").replace(" ", "/")
    if "/" not in s:
        # Infer base/quote if missing slash
        if s.endswith(("USDT", "USD")):
            base = s[:-4]
            quote = s[-4:]
            return f"{base}/{quote}"
        return f"{s}/USDT"
    base, quote = s.split("/", 1)
    return f"{base}/{quote}"


# Allowed quotes for Binance.US spot pairs (UI/ccxt normalized)
ALLOWED_QUOTES: Final[set[str]] = {"USDT", "USD"}
