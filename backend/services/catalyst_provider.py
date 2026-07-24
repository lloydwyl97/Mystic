"""
Catalyst Provider — modular interface for news and catalyst data.

Design goals:
  - Completely decoupled from trading execution.
  - Catalyst score is a RANKING input only, never a trade blocker.
  - Missing / unavailable data returns honest null values with source metadata.
  - Providers can be swapped without touching trading code.
  - Duplicate detection and freshness/decay are enforced at this layer.

Built-in providers:
  NullCatalystProvider  — always returns unavailable (default; no API key required)
  NewsSentimentProvider — wraps the existing news_sentiment.py if NEWS_API_KEY is set

Adding a new provider:
  1. Subclass CatalystProvider.
  2. Implement get_catalyst_score(symbol) -> CatalystResult | None.
  3. Register via register_provider(name, provider_instance).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Catalyst categories (open-ended string, these are suggestions)
CATALYST_CATEGORIES = frozenset({
    "regulatory",
    "etf_flow",
    "institutional",
    "protocol_upgrade",
    "partnership",
    "macro_event",
    "negative_event",
    "unknown",
})


@dataclass(frozen=True)
class CatalystResult:
    """Structured result from a catalyst provider."""
    symbol: str
    score: float               # [0, 1] relevance/impact score
    source: str                # provider name
    freshness_sec: int         # age of the underlying data
    confidence: float = 1.0   # how confident the score is [0, 1]
    category: str | None = None         # catalyst type
    direction: str | None = None        # "positive" / "negative" / "neutral"
    headline: str | None = None         # brief summary
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    is_stale: bool = False


class CatalystProvider(ABC):
    """Abstract base for all catalyst providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def get_catalyst_score(self, symbol: str) -> CatalystResult | None: ...

    async def is_available(self) -> bool:
        """Return True if the provider has valid credentials / connectivity."""
        return False


class NullCatalystProvider(CatalystProvider):
    """
    Default provider — always returns None.
    Used when no catalyst API key is configured.
    Returns None so callers know to show "unavailable" rather than 0.
    """

    @property
    def name(self) -> str:
        return "unavailable"

    async def get_catalyst_score(self, symbol: str) -> CatalystResult | None:
        return None

    async def is_available(self) -> bool:
        return False


class NewsSentimentProvider(CatalystProvider):
    """
    Wraps the existing Mystic news_sentiment.py infrastructure.
    Requires NEWS_API_KEY in environment.

    Symbol→topic mapping:
      BTCUSDT → bitcoin
      ETHUSDT → ethereum
      SOLUSDT → solana
      XRPUSDT → xrp ripple
    """

    _SYMBOL_TOPICS: dict[str, str] = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
        "XRPUSDT": "xrp ripple",
    }
    _CACHE: dict[str, tuple[CatalystResult, float]] = {}
    _CACHE_TTL_SEC = 300  # 5-minute cache

    @property
    def name(self) -> str:
        return "news_sentiment"

    async def is_available(self) -> bool:
        return bool(os.getenv("NEWS_API_KEY"))

    async def get_catalyst_score(self, symbol: str) -> CatalystResult | None:
        sym = symbol.upper().replace("/", "")
        cached = self._CACHE.get(sym)
        if cached and (time.time() - cached[1]) < self._CACHE_TTL_SEC:
            return cached[0]

        topic = self._SYMBOL_TOPICS.get(sym)
        if not topic:
            return None

        score: float | None = None
        direction: str | None = None
        headline: str | None = None
        freshness = 3600

        try:
            from backend.services.news_sentiment import get_news_sentiment

            result = await get_news_sentiment(topic)
            if isinstance(result, dict):
                raw = result.get("sentiment") or result.get("score") or result.get("value")
                if raw is not None:
                    score = float(raw)
                    # Normalise to [0, 1] from whatever scale news_sentiment uses
                    if score < 0:
                        score = max(0.0, 0.5 + score * 0.5)
                    else:
                        score = min(1.0, 0.5 + score * 0.5)
                direction = result.get("direction") or ("positive" if score and score > 0.55 else "negative" if score and score < 0.45 else "neutral")
                headline = str(result.get("headline") or result.get("top_headline") or "")[:120] or None
                freshness = int(result.get("age_sec") or result.get("freshness_sec") or 3600)
        except Exception as exc:
            logger.debug("NewsSentimentProvider %s failed: %s", symbol, exc)
            return None

        if score is None:
            return None

        cat = "unknown"
        if sym == "XRPUSDT" and direction == "positive":
            cat = "regulatory"

        result_obj = CatalystResult(
            symbol=sym,
            score=round(score, 4),
            source=self.name,
            freshness_sec=freshness,
            confidence=0.7,
            category=cat,
            direction=direction,
            headline=headline,
            expires_at=time.time() + self._CACHE_TTL_SEC,
        )
        self._CACHE[sym] = (result_obj, time.time())
        return result_obj


class AggregateCatalystProvider(CatalystProvider):
    """
    Combine multiple providers.  Returns highest-confidence non-None result.
    """

    def __init__(self, providers: list[CatalystProvider]) -> None:
        self._providers = providers

    @property
    def name(self) -> str:
        return "aggregate"

    async def get_catalyst_score(self, symbol: str) -> CatalystResult | None:
        results: list[CatalystResult] = []
        for p in self._providers:
            with contextlib.suppress(Exception):
                r = await p.get_catalyst_score(symbol)
                if r is not None:
                    results.append(r)
        if not results:
            return None
        return max(results, key=lambda r: r.confidence)

    async def is_available(self) -> bool:
        for p in self._providers:
            if await p.is_available():
                return True
        return False


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_providers: dict[str, CatalystProvider] = {}
_default_provider: CatalystProvider | None = None


def register_provider(name: str, provider: CatalystProvider) -> None:
    _providers[name] = provider
    logger.info("CATALYST_PROVIDER registered: %s", name)


def get_default_provider() -> CatalystProvider:
    global _default_provider
    if _default_provider is not None:
        return _default_provider

    providers: list[CatalystProvider] = []
    if os.getenv("NEWS_API_KEY"):
        providers.append(NewsSentimentProvider())
        logger.info("CATALYST_PROVIDER: NewsSentimentProvider activated (NEWS_API_KEY present)")
    else:
        logger.info("CATALYST_PROVIDER: NEWS_API_KEY not set — using NullCatalystProvider")

    if providers:
        _default_provider = AggregateCatalystProvider(providers)
    else:
        _default_provider = NullCatalystProvider()

    return _default_provider


__all__ = [
    "CATALYST_CATEGORIES",
    "AggregateCatalystProvider",
    "CatalystProvider",
    "CatalystResult",
    "NewsSentimentProvider",
    "NullCatalystProvider",
    "get_default_provider",
    "register_provider",
]
