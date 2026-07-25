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

    Uses the /api/1/crypto endpoint with the `coin` parameter (free plan).
    Requires NEWSDATA_API_KEY in environment.

    Daily limit management (free plan = 200 calls/day):
      - Cache TTL: 3600s (1h) per symbol
      - 4 coins × 24 refreshes/day = 96 calls/day — safely within limit
      - Override via NEWSDATA_CACHE_TTL_SEC
      - Graceful 429 backoff with 1h rate-limit guard using stale cache

    Scoring (sentiment field requires paid plan — not available on free tier):
      - Filter out duplicate articles (duplicate=true)
      - Filter to articles actually tagged with this coin in their `coin` array
      - Score titles using bullish/bearish keyword lists
      - Weigh high-priority sources (lower source_priority number) more heavily
      - score = (bullish_weighted - bearish_weighted * 0.7) / total_weight → [0, 1]
      - score > 0.55 → constructive catalyst lift in ranking_delta()

    Category inference:
      - Keyword matching on title + description + keywords array
      - Per-coin category hint tables (regulatory for XRP, institutional for BTC, etc.)
    """

    # NewsData.io `coin` parameter values (lowercase coin code)
    _COIN_PARAMS: dict[str, str] = {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
        "SOLUSDT": "sol",
        "XRPUSDT": "xrp",
    }
    # Uppercase coin codes as they appear in the article `coin` array
    _COIN_TAGS: dict[str, str] = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "SOLUSDT": "SOL",
        "XRPUSDT": "XRP",
    }
    _BULLISH_WORDS: frozenset[str] = frozenset({
        "surge", "rally", "breakout", "bullish", "gains", "gained", "high", "target",
        "approval", "approved", "etf", "institutional", "inflow", "inflows", "buy",
        "bought", "accumulate", "launch", "upgrade", "partnership", "milestone",
        "record", "adoption", "outperform", "strong", "support", "recover", "reclaim",
        "positive", "optimistic", "growth", "rise", "rising", "jumped", "explode",
        "moon", "upside", "momentum", "demand", "interest", "listing",
    })
    _BEARISH_WORDS: frozenset[str] = frozenset({
        "crash", "drop", "fall", "fell", "sell", "bearish", "concern", "risk",
        "warning", "exploit", "hack", "breach", "loss", "lose", "lawsuit", "ban",
        "rejected", "decline", "dump", "plunge", "collapse", "vulnerability", "fraud",
        "scam", "fear", "panic", "outflow", "outflows", "withdraw", "negative",
        "pessimistic", "weak", "resistance", "struggle", "failed", "failure",
    })
    _CATEGORY_HINTS: dict[str, list[tuple[str, list[str]]]] = {
        "XRPUSDT": [
            ("regulatory", ["sec", "ripple", "lawsuit", "ruling", "etf", "approval", "rlusd", "regulatory"]),
            ("etf_flow", ["etf", "inflow", "fund", "approval"]),
        ],
        "BTCUSDT": [
            ("institutional", ["etf", "blackrock", "fidelity", "microstrategy", "institutional", "saylor", "strategy"]),
            ("macro_event", ["fed", "interest rate", "inflation", "macro", "treasury", "reserve"]),
        ],
        "ETHUSDT": [
            ("protocol_upgrade", ["upgrade", "pectra", "dencun", "eip", "staking", "merge", "layer"]),
            ("etf_flow", ["etf", "inflow", "fund"]),
        ],
        "SOLUSDT": [
            ("protocol_upgrade", ["upgrade", "v2", "network", "congestion", "validator"]),
            ("partnership", ["partnership", "integration", "launch", "nft", "defi", "meme"]),
        ],
    }
    _CACHE: dict[str, tuple[CatalystResult, float]] = {}
    _RATE_LIMITED_UNTIL: float = 0.0

    @property
    def name(self) -> str:
        return "newsdata_io"

    def _cache_ttl(self) -> float:
        return float(os.getenv("NEWSDATA_CACHE_TTL_SEC", "3600"))

    async def is_available(self) -> bool:
        return bool(os.getenv("NEWSDATA_API_KEY"))

    def _score_title(self, text: str) -> tuple[float, float]:
        """Return (bullish_weight, bearish_weight) for a title string."""
        words = set(text.lower().split())
        bull = sum(1.0 for w in words if w in self._BULLISH_WORDS)
        bear = sum(1.0 for w in words if w in self._BEARISH_WORDS)
        return bull, bear

    def _infer_category(self, sym: str, title: str, desc: str, keywords: list) -> str | None:
        combined = " ".join(filter(None, [title, desc, " ".join(keywords or [])])).lower()
        for cat_name, kw_list in self._CATEGORY_HINTS.get(sym, []):
            if any(kw in combined for kw in kw_list):
                return cat_name
        return "unknown"

    async def get_catalyst_score(self, symbol: str) -> CatalystResult | None:
        sym = symbol.upper().replace("/", "")
        api_key = os.getenv("NEWSDATA_API_KEY", "").strip()
        if not api_key:
            return None

        # Rate-limit backoff: return stale cache or None
        if time.time() < self._RATE_LIMITED_UNTIL:
            cached = self._CACHE.get(sym)
            if cached:
                r, _ = cached
                return CatalystResult(
                    symbol=r.symbol, score=r.score, source=r.source,
                    freshness_sec=int(time.time() - cached[1]),
                    confidence=round(r.confidence * 0.5, 4),
                    category=r.category, direction=r.direction, headline=r.headline,
                    expires_at=r.expires_at, is_stale=True,
                )
            return None

        # In-memory cache hit
        cached = self._CACHE.get(sym)
        if cached and (time.time() - cached[1]) < self._cache_ttl():
            r, cached_at = cached
            return CatalystResult(
                symbol=r.symbol, score=r.score, source=r.source,
                freshness_sec=int(time.time() - cached_at),
                confidence=r.confidence, category=r.category,
                direction=r.direction, headline=r.headline,
                expires_at=r.expires_at, is_stale=False,
            )

        coin_param = self._COIN_PARAMS.get(sym)
        coin_tag = self._COIN_TAGS.get(sym)
        if not coin_param:
            return None

        score: float | None = None
        direction: str | None = None
        headline: str | None = None
        category: str | None = None
        freshness = 600

        try:
            import aiohttp

            params = {
                "apikey": api_key,
                "coin": coin_param,
                "language": "en",
                "timezone": "America/Chicago",
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(
                    "https://newsdata.io/api/1/crypto",
                    params=params,
                    headers={"User-Agent": "Mystic/1.0"},
                ) as resp:
                    if resp.status == 429:
                        NewsDataIoCatalystProvider._RATE_LIMITED_UNTIL = time.time() + 3600
                        logger.warning("NEWSDATA_IO rate limited — backoff 1h, sym=%s", sym)
                        return None
                    if resp.status != 200:
                        logger.debug("NEWSDATA_IO %s HTTP %s", sym, resp.status)
                        return None
                    data = await resp.json()

            all_articles: list[dict] = data.get("results", []) or []

            # Filter: non-duplicate only
            non_dup = [a for a in all_articles[:60] if not a.get("duplicate")]

            # Sort by coin specificity: articles with fewer coins in their `coin` array
            # are more specifically about this coin → surface them first
            def _coin_specificity(art: dict) -> float:
                tagged = a.get("coin") or []
                if coin_tag not in tagged:
                    return 999.0   # deprioritize articles not tagged with this coin
                return float(len(tagged))   # fewer coins = more specific = lower score = sorts first

            articles_tagged = [a for a in non_dup if coin_tag in (a.get("coin") or [])]
            articles_tagged.sort(key=lambda a: len(a.get("coin") or []))

            # Fall back to all non-dup if coin-tag filter is too strict (e.g., SOL)
            articles = articles_tagged if articles_tagged else non_dup[:20]
            if not articles:
                articles = all_articles[:20]

            # Source priority weighting: lower source_priority = higher authority
            # Scale: priority 1000 → weight 2.0, priority 5_000_000 → weight 0.5
            def _source_weight(art: dict) -> float:
                prio = int(art.get("source_priority") or 5_000_000)
                return max(0.5, min(2.0, 5_000_000 / max(prio, 1_000)))

            total_bull = total_bear = total_w = 0.0
            top_headline: str | None = None
            top_pub_date: str | None = None

            for art in articles:
                title = str(art.get("title") or "")
                w = _source_weight(art)
                bull, bear = self._score_title(title)
                total_bull += bull * w
                total_bear += bear * w
                total_w += w
                if top_headline is None and title:
                    top_headline = title[:120]
                    top_pub_date = art.get("pubDate")

            headline = top_headline

            if total_w == 0:
                score = 0.5
                direction = "neutral"
                confidence = 0.3
            else:
                raw = (total_bull - total_bear * 0.7) / total_w
                score = round(max(0.0, min(1.0, 0.5 + raw * 0.25)), 4)
                direction = "positive" if raw > 0.15 else ("negative" if raw < -0.15 else "neutral")
                # Confidence scales with article count (more coverage = more reliable)
                confidence = round(min(0.85, 0.55 + len(articles) * 0.005), 4)

            # Category inference
            if top_headline:
                first_art = articles[0] if articles else {}
                category = self._infer_category(
                    sym,
                    top_headline,
                    str(first_art.get("description") or ""),
                    first_art.get("keywords") or [],
                )

            # Freshness from most recent article
            if top_pub_date:
                with contextlib.suppress(Exception):
                    from datetime import datetime, timezone
                    tnorm = str(top_pub_date).replace(" ", "T")
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
            category=category or "unknown",
            direction=direction,
            headline=headline,
            expires_at=time.time() + self._cache_ttl(),
            is_stale=False,
        )
        NewsDataIoCatalystProvider._CACHE[sym] = (result_obj, time.time())
        logger.info(
            "NEWSDATA_IO %s score=%.3f dir=%s cat=%s articles=%d bull=%.2f bear=%.2f fresh=%ds",
            sym, score, direction, category, len(articles), total_bull, total_bear, freshness,
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
