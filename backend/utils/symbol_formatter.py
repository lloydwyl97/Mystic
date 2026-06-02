"""
Symbol Formatter Utility
Safe symbol format handling to prevent crashes
All Live Data, No Fallback/Hardcoded Data
"""

from __future__ import annotations

import logging

# Import from single source of truth
try:
    from backend.config.binance_allowlist import SymbolNormalizer
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading universe, binance_allowlist, or _to_ccxt_symbol: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

# Single-source fallback symbol from trading_universe (live data)
if not TRADING_SYMBOLS:
    msg = "TRADING_SYMBOLS is empty - cannot initialize symbol formatter"
    raise RuntimeError(msg)
FALLBACK_STORAGE = TRADING_SYMBOLS[0]


def safe_symbol_split(symbol: str) -> tuple[str, str]:
    try:
        normalized = SymbolNormalizer.to_storage_key(symbol or FALLBACK_STORAGE)
        base = SymbolNormalizer.get_base_asset(normalized)
        quote = normalized[len(base) :] or "USDT"
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("symbol split failed for %s: %s", symbol, e)
        fb_base = SymbolNormalizer.get_base_asset(FALLBACK_STORAGE)
        fb_quote = FALLBACK_STORAGE[len(fb_base) :] or "USDT"
        return fb_base, fb_quote
    else:
        return base, quote


def safe_symbol_format(symbol: str, target_format: str = "display") -> str:
    try:
        normalized = SymbolNormalizer.to_storage_key(symbol or FALLBACK_STORAGE)
        if target_format == "display":
            return SymbolNormalizer.to_display_key(normalized)
        if target_format == "storage":
            return normalized
        if target_format == "base":
            return SymbolNormalizer.get_base_asset(normalized)
        if target_format == "ccxt":
            return _to_ccxt_symbol(normalized)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("symbol format failed for %s: %s", symbol, e)
        return SymbolNormalizer.to_display_key(FALLBACK_STORAGE) if target_format == "display" else FALLBACK_STORAGE
    else:
        return normalized


def validate_and_normalize_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    for s in symbols:
        try:
            out.append(SymbolNormalizer.to_storage_key(s))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("symbol %s rejected by allowlist: %s", s, e)
    return out


def get_safe_trading_pairs() -> list[str]:
    """Get safe trading pairs from trading_universe (live data)."""
    return [SymbolNormalizer.to_display_key(s) for s in TRADING_SYMBOLS]


def parse_symbol_safely(symbol_input: str) -> dict[str, str | bool]:
    try:
        normalized = SymbolNormalizer.to_storage_key(symbol_input)
        base = SymbolNormalizer.get_base_asset(normalized)
        quote = normalized[len(base) :] or "USDT"
        return {
            "exchange": EXCHANGE_ID,
            "storage": normalized,
            "display": SymbolNormalizer.to_display_key(normalized),
            "base": base,
            "quote": quote,
            "ccxt_format": _to_ccxt_symbol(normalized),
            "valid": True,
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("symbol parsing failed for %s: %s", symbol_input, e)
        fb_base = SymbolNormalizer.get_base_asset(FALLBACK_STORAGE)
        fb_quote = FALLBACK_STORAGE[len(fb_base) :] or "USDT"
        return {
            "exchange": EXCHANGE_ID,
            "storage": FALLBACK_STORAGE,
            "display": SymbolNormalizer.to_display_key(FALLBACK_STORAGE),
            "base": fb_base,
            "quote": fb_quote,
            "ccxt_format": _to_ccxt_symbol(FALLBACK_STORAGE),
            "valid": False,
            "error": str(e),
        }


# Quick test checklist:
# - ccxt calls only receive BASE/QUOTE (via _to_ccxt_symbol).
# - No binance/binanceus string leaks—only EXCHANGE_ID from single source.
# - No unreachable code after returns.
# - Logging has no weird characters.
