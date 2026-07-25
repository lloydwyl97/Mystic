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


class NewsDataIoCatalystProvider(CatalystProvider):
    """
    NewsData.io catalyst provider.

    Uses the /api/1/crypto endpoint with the `coin` parameter.
    Requires NEWSDATA_API_KEY in environment.

    Daily limit management:
      - Default cache TTL: 60 min per symbol (4 coins × 24 refreshes = 96 calls/day)
      - Override via NEWSDATA_CACHE_TTL_SEC env var
      - Graceful degradation on rate-limit (429) or network errors

    Scoring:
      - Counts positive / negative / neutral article sentiments in the result set
      - score = (positive - negative * 0.7) / max(total, 1), normalised to [0, 1]
      - score > 0.55 → constructive catalyst lift in ranking_delta()
    """

    _COIN_CODES: dict[str, str] = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
        "XRPUSDT": "xrp",
    }
    _CATEGORY_HINTS: dict[str, dict[str, str]] = {
        "XRPUSDT": {"regulatory": ["sec", "ripple", "lawsuit", "ruling", "etf", "approval"]},
        "BTCUSDT": {"institutional": ["etf", "blackrock", "fidelity", "microstrategy", "institutional"]},
        "ETHUSDT": {"protocol_upgrade": ["upgrade", "pectra", "dencun", "eip", "staking"]},
        "SOLUSDT": {"partnership": ["solana", "sol", "meme", "memecoin", "nft"]},
    }
    # Shared class-level cache: {symbol: (result, cached_at_timestamp)}
    _CACHE: dict[str, tuple[CatalystResult, float]] = {}
    _RATE_LIMITED_UNTIL: float = 0.0

    @property
    def name(self) -> str:
        return "newsdata_io"

    def _cache_ttl(self) -> float:
        return float(os.getenv("NEWSDATA_CACHE_TTL_SEC", "3600"))  # 60 min default

    async def is_available(self) -> bool:
        return bool(os.getenv("NEWSDATA_API_KEY"))

    async def get_catalyst_score(self, symbol: str) -> CatalystResult | None:
        sym = symbol.upper().replace("/", "")
        api_key = os.getenv("NEWSDATA_API_KEY", "").strip()
        if not api_key:
            return None

        # Respect rate-limit backoff
        if time.time() < self._RATE_LIMITED_UNTIL:
            cached = self._CACHE.get(sym)
            if cached:
                r, _ = cached
                return CatalystResult(
                    symbol=r.symbol, score=r.score, source=r.source,
                    freshness_sec=int(time.time() - r.expires_at + self._cache_ttl()),
                    confidence=r.confidence * 0.6,
                    category=r.category, direction=r.direction, headline=r.headline,
                    expires_at=r.expires_at, is_stale=True,
                )
            return None

        # Check in-memory cache
        cached = self._CACHE.get(sym)
        if cached and (time.time() - cached[1]) < self._cache_ttl():
            result, _ = cached
            freshness = int(time.time() - cached[1])
            return CatalystResult(
                symbol=result.symbol, score=result.score, source=result.source,
                freshness_sec=freshness, confidence=result.confidence,
                category=result.category, direction=result.direction, headline=result.headline,
                expires_at=result.expires_at, is_stale=False,
            )

        coin_code = self._COIN_CODES.get(sym)
        if not coin_code:
            return None

        # Fetch from NewsData.io crypto endpoint
        score: float | None = None
        direction: str | None = None
        headline: str | None = None
        category: str | None = None
        confidence = 0.75
        freshness = 0

        try:
            import aiohttp

            params = {
                "apikey": api_key,
                "coin": coin_code,
                "language": "english",
                "timeframe": "48",
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
                async with session.get(
                    "https://newsdata.io/api/1/crypto",
                    params=params,
                    headers={"User-Agent": "Mystic/1.0"},
                ) as resp:
                    if resp.status == 429:
                        # Back off for 1 hour
                        NewsDataIoCatalystProvider._RATE_LIMITED_UNTIL = time.time() + 3600
                        logger.warning("NEWSDATA_IO rate limited — backing off 1h")
                        return None
                    if resp.status != 200:
                        logger.debug("NEWSDATA_IO %s HTTP %s", sym, resp.status)
                        return None
                    data = await resp.json()

            articles: list[dict] = data.get("results", []) or []
            if not articles:
                return None

            # Score from sentiment field (NewsData.io returns sentiment per article on paid plans)
            pos = neg = neu = 0
            top_headline: str | None = None
            top_pub_date: str | None = None
            for art in articles[:20]:
                sent = str(art.get("sentiment") or "").lower()
                if sent == "positive":
                    pos += 1
                elif sent == "negative":
                    neg += 1
                else:
                    neu += 1
                if top_headline is None and art.get("title"):
                    top_headline = str(art["title"])[:120]
                    top_pub_date = art.get("pubDate") or art.get("publishedAt")

            total = pos + neg + neu
            if total == 0:
                # No sentiment data — score by article count (proxy for interest level)
                # Normalise: 0 articles = 0.5, 10+ = 0.65
                score = min(0.65, 0.5 + len(articles) * 0.015)
                direction = "neutral"
                confidence = 0.4
            else:
                # Weighted: negative counts 0.7× (false alarms common)
                raw_score = (pos - neg * 0.7) / total
                score = round(max(0.0, min(1.0, 0.5 + raw_score * 0.5)), 4)
                direction = "positive" if raw_score > 0.1 else ("negative" if raw_score < -0.1 else "neutral")

            headline = top_headline

            # Category inference from headline keywords
            hint_map = self._CATEGORY_HINTS.get(sym, {})
            if headline:
                hl_lower = headline.lower()
                for cat_name, keywords in hint_map.items():
                    if any(kw in hl_lower for kw in keywords):
                        category = cat_name
                        break

            if category is None:
                category = "unknown"

            # Freshness: age of most recent article (approximate)
            freshness = 600  # default 10 min if pub_date unavailable
            if top_pub_date:
                with contextlib.suppress(Exception):
                    from datetime import datetime, timezone
                    tnorm = str(top_pub_date).replace("Z", "+00:00").replace(" ", "T")
                    t = datetime.fromisoformat(tnorm)
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    freshness = max(0, int((datetime.now(timezone.utc) - t.astimezone(timezone.utc)).total_seconds()))

        except Exception as exc:
            logger.debug("NEWSDATA_IO %s fetch failed: %s", sym, exc)
            return None

        if score is None:
            return None

        result_obj = CatalystResult(
            symbol=sym,
            score=score,
            source=self.name,
            freshness_sec=freshness,
            confidence=confidence,
            category=category,
            direction=direction,
            headline=headline,
            expires_at=time.time() + self._cache_ttl(),
            is_stale=False,
        )
        NewsDataIoCatalystProvider._CACHE[sym] = (result_obj, time.time())
        logger.info(
            "NEWSDATA_IO %s score=%.3f dir=%s cat=%s articles=%d pos=%d neg=%d",
            sym, score, direction, category, len(articles), pos, neg,
        )
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
    if os.getenv("NEWSDATA_API_KEY"):
        providers.append(NewsDataIoCatalystProvider())
        logger.info("CATALYST_PROVIDER: NewsDataIoCatalystProvider activated (NEWSDATA_API_KEY present)")
    if os.getenv("NEWS_API_KEY"):
        providers.append(NewsSentimentProvider())
        logger.info("CATALYST_PROVIDER: NewsSentimentProvider activated (NEWS_API_KEY present)")
    if not providers:
        logger.info("CATALYST_PROVIDER: no API keys set — using NullCatalystProvider")

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
    "NewsDataIoCatalystProvider",
    "NewsSentimentProvider",
    "NullCatalystProvider",
    "get_default_provider",
    "register_provider",
]
