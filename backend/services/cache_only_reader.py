"""
Cache-Only Reader Service
Enforces cache-only reads with no REST fallbacks
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Import AllowlistGuard and SymbolNormalizer from binance_allowlist (they use trading_universe internally)
from backend.config.binance_allowlist import AllowlistGuard, SymbolNormalizer
from backend.modules.ai.persistent_cache import get_persistent_cache

# Use TRADING_SYMBOLS from trading_universe (live data)
BINANCE_US_TOP_10 = list(TRADING_SYMBOLS)
STANDARD_EXCHANGE = EXCHANGE_ID

logger = logging.getLogger(__name__)


class CacheOnlyReader:
    """
    Read market data exclusively from cache (PersistentCache) — NO REST fallbacks.

    - Symbols are validated against the allowlist.
    - Returns prices in a stable, predictable order (allowlist order by default).
    - Optionally enriches rows with 24h stats if present in cache.
    - Includes overall cache freshness and hydration counts for health reporting.
    """

    def __init__(self) -> None:
        self.cache = get_persistent_cache()
        self.exchange_name = STANDARD_EXCHANGE

        # Optional getter for 24h ticker snapshot if the cache provides it.
        # The writer (authoritative writer) uses `set_ticker_24h(...)`; read via `get_ticker_24h(...)` if available.
        self._get_ticker_24h: Callable[[str, str], dict[str, Any] | None] | None = getattr(self.cache, "get_ticker_24h", None)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_price_safe(self, symbol: str) -> float:
        """
        Strictly read from cache. Return 0.0 if missing/invalid.
        """
        try:
            px = self.cache.get_latest_price(self.exchange_name, symbol)
            return float(px) if px is not None else 0.0
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return 0.0

    def _read_ticker24h_safe(self, symbol: str) -> dict[str, Any]:
        """
        Read 24h stats from cache, if available. Returns an empty dict if not present.
        Fields expected (if writer populated them):
          last_price, volume_24h, change_24h, high_24h, low_24h, timestamp
        """
        try:
            if not self._get_ticker_24h:
                return {}
            d = self._get_ticker_24h(self.exchange_name, symbol)  # type: ignore[call-arg]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return {}
        else:
            return d or {}

    async def get_prices(
        self,
        symbols: list[str] | None = None,
        include_24h: bool = True,
    ) -> dict[str, Any]:
        """
        CACHE ONLY — No REST fallbacks on GET paths.

        Args:
            symbols: Optional list of symbols (e.g., ["BTCUSDT", "ETHUSDT"]).
                     If None, uses BINANCE_US_TOP_10 from allowlist.
            include_24h: If True, enrich rows with 24h cache fields when available.

        Returns:
            {
                "prices": {
                    "BTC-USD": {...},
                    ...
                },
                "status": "<allowlist status text>",
                "source": "cache_only",
                "hydrated_count": <int>,
                "requested_count": <int>,
                "allowlist_enforced": True,
                "timestamp": <iso>
            }
        """
        try:
            # Stable target order: if no symbols provided, stick to allowlist order
            target_symbols = (symbols or BINANCE_US_TOP_10)[:]

            # Validate/normalize via allowlist
            valid_symbols, status = AllowlistGuard.filter_request_symbols(target_symbols)

            prices: dict[str, Any] = {}
            hydrated_count = 0

            # Preserve order based on provided list (or allowlist) while skipping invalids
            for raw in target_symbols:
                if raw not in valid_symbols:
                    continue

                sym = raw.upper()
                price = self._read_price_safe(sym)
                if price > 0:
                    hydrated_count += 1

                display_symbol = SymbolNormalizer.to_display_key(sym)
                row: dict[str, Any] = {
                    "symbol": display_symbol,
                    "exchange": self.exchange_name,
                    "price": float(price) if price > 0 else None,
                    "timestamp": self._now_iso(),
                    "source": "cache_only",
                }

                if include_24h:
                    t24 = self._read_ticker24h_safe(sym)
                    if t24:
                        # Only include fields that exist in the cache snapshot
                        for k in (
                            "last_price",
                            "volume_24h",
                            "change_24h",
                            "high_24h",
                            "low_24h",
                            "timestamp",
                        ):
                            if k in t24:
                                row[k] = t24[k]

                prices[display_symbol] = row

            out_obj = {
                "prices": prices,
                "status": status,
                "source": "cache_only",
                "hydrated_count": hydrated_count,
                "requested_count": len(target_symbols),
                "allowlist_enforced": True,
                "timestamp": self._now_iso(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[ERROR] Cache reader error: %s", e)
            return {
                "prices": {},
                "status": "error",
                "source": "cache_only",
                "hydrated_count": 0,
                "requested_count": len(symbols or BINANCE_US_TOP_10),
                "allowlist_enforced": True,
                "error": str(e),
                "timestamp": self._now_iso(),
            }
        else:
            return out_obj

    async def get_live_data(self) -> dict[str, Any]:
        """
        Get live market data formatted for /market/live endpoint.
        Uses the full allowlist and cache-only reads.
        """
        try:
            prices_data = await self.get_prices(BINANCE_US_TOP_10, include_24h=True)

            # Convert to list format for compatibility and keep stable ordering
            prices_list = list(prices_data["prices"].values())

            # Global freshness gate (writer freshness)
            try:
                freshness = float(self.cache.get_last_update_age())
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                freshness = 1e9

            hydrated_count = int(prices_data.get("hydrated_count", 0))
            # Maintain your original health thresholds
            is_fresh = freshness < 30
            is_hydrated = hydrated_count >= 10

            status = "live" if (is_fresh and is_hydrated) else "degraded"

            out_obj = {
                "timestamp": self._now_iso(),
                "live_data": {"source": "binance_us_cache"},
                "prices": prices_list,
                "trends": {"source": "cache_only"},
                "status": status,
                "hydrated_count": hydrated_count,
                "freshness_sec": freshness,
                "cache_only": True,
                "allowlist_enforced": True,
                "version": "2.0.0",
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[ERROR] Live data error: %s", e)
            return {
                "timestamp": self._now_iso(),
                "live_data": {},
                "prices": [],
                "trends": {},
                "status": "error",
                "error": str(e),
                "cache_only": True,
                "allowlist_enforced": True,
            }
        else:
            return out_obj

    def get_health_status(self) -> dict[str, Any]:
        """
        Health probe for monitoring dashboards.
        - Counts hydrated symbols that have a >0 cached price
        - Uses writer freshness for liveness
        - Keeps allowlist enforcement visible
        """
        try:
            hydrated_symbols: list[str] = []
            for sym in BINANCE_US_TOP_10:
                px = self._read_price_safe(sym)
                if px > 0:
                    hydrated_symbols.append(sym)

            try:
                freshness = float(self.cache.get_last_update_age())
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                freshness = 1e9

            is_healthy = len(hydrated_symbols) >= 10 and freshness < 30
            dashboard_data_mode = "live" if is_healthy else "degraded"

            out_obj = {
                "hydrated_symbols": hydrated_symbols,
                "hydrated_count": len(hydrated_symbols),
                "writer_freshness_sec": freshness,
                "dashboard_data_mode": dashboard_data_mode,
                "allowlist_enforced": True,
                "cache_only_reads": True,
                "target_symbols": BINANCE_US_TOP_10,
                "healthy": is_healthy,
                "timestamp": self._now_iso(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[ERROR] Health status error: %s", e)
            return {
                "hydrated_symbols": [],
                "hydrated_count": 0,
                "writer_freshness_sec": 1e9,
                "dashboard_data_mode": "error",
                "allowlist_enforced": True,
                "cache_only_reads": True,
                "target_symbols": BINANCE_US_TOP_10,
                "healthy": False,
                "error": str(e),
                "timestamp": self._now_iso(),
            }
        else:
            return out_obj


# Global reader instance
_cache_reader: CacheOnlyReader | None = None


# Cache reader state - using dict to avoid global keyword
_cache_reader_state: dict[str, CacheOnlyReader | None] = {"instance": None}


def get_cache_reader() -> CacheOnlyReader:
    """Get the global cache reader instance (singleton)."""
    if _cache_reader_state["instance"] is None:
        _cache_reader_state["instance"] = CacheOnlyReader()
    return _cache_reader_state["instance"]
