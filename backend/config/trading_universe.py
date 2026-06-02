"""
DAY-only trading universe configuration.

SINGLE SOURCE OF TRUTH for the active production symbols.
Mystic is DAY trading only and trades only the configured top 4 Binance.US USDT
markets in exact Binance.US API format. No slashes, no dashes, no lowercase,
no USD/USDT mixing, no auto-expansion, no fallback universe.
"""

from typing import Final

# Coin universe is fixed; no auto-expansion, no fallback universe, no dynamic
# discovery. Any code that needs the live trading universe MUST import from
# this module via DAY_TRADE_SYMBOLS / TRADING_SYMBOLS.
COIN_UNIVERSE_FIXED: Final[bool] = True
# Max share of total capital per coin (%). Log warning when approached or exceeded.
MAX_COIN_CONCENTRATION_PCT: Final[float] = 30.0
CONCENTRATION_WARN_PCT: Final[float] = 25.0

# AUTHORITATIVE BINANCE.US TOP 4 (DAY-only live universe).
# Exact Binance.US API symbol format: no slash, no dash, no underscore,
# no lowercase, no USD-only pairs. USDT pairs only.
DAY_TRADE_SYMBOLS: Final[list[str]] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

# Back-compat alias. All callers should treat this as identical to
# DAY_TRADE_SYMBOLS; both names point at the same top-4 list.
TRADING_SYMBOLS: Final[list[str]] = DAY_TRADE_SYMBOLS

# Base symbols (without USDT suffix) derived from the active live universe.
TOP4_BASE_COINS: Final[list[str]] = ["BTC", "ETH", "SOL", "XRP"]
# Back-compat alias for legacy importers; identical list.
TOP10_COINS: Final[list[str]] = TOP4_BASE_COINS

# Exchange identifier (Binance.US only - live trading exchange)
# All operations connect to Binance.US for live trading (backend port 8000)
EXCHANGE_ID: Final[str] = "binance_us"
FEATURED_EXCHANGE: Final[str] = "binance_us"


def get_trading_symbols() -> list[str]:
    """Return the live DAY top-4 trading symbols (Binance.US API format)."""
    return DAY_TRADE_SYMBOLS.copy()


def get_base_symbols() -> list[str]:
    """Return the live DAY top-4 base coin symbols (without USDT suffix)."""
    return TOP4_BASE_COINS.copy()


def get_symbol_count() -> int:
    """Return the live DAY trading symbol count (always 4)."""
    return len(DAY_TRADE_SYMBOLS)


def is_valid_symbol(symbol: str) -> bool:
    """Return True only if symbol is in the DAY top-4 (USDT or base form)."""
    return symbol in DAY_TRADE_SYMBOLS or symbol in TOP4_BASE_COINS
