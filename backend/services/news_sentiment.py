"""NewsAPI crypto sentiment — real articles only, no fabrication. Single writer: ActiveSentimentCollector."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

REDIS_NEWS_CORPUS_KEY = "news_sentiment:corpus"
REDIS_NEWS_COOLDOWN_KEY = "news_sentiment:rate_limit_until"
REDIS_NEWS_STATUS_KEY = "news_sentiment:status"
_NEWS_BASES = ("BTC", "ETH", "SOL", "XRP")
_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "bitcoin"),
    "ETH": ("eth", "ethereum"),
    "SOL": ("sol", "solana"),
    "XRP": ("xrp", "ripple"),
}
_DEFAULT_RSS_FEEDS: tuple[str, ...] = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)


def _score_text(text: str) -> float:
    from backend.services.crypto_sentiment_vader import score_crypto_text

    return score_crypto_text(text)


def _base_from_symbol(symbol: str) -> str:
    s = (symbol or "BTC").upper().replace("/USDT", "").replace("USDT", "").strip()
    return s or "BTC"


def _news_enabled() -> bool:
    return bool((os.getenv("NEWS_API_KEY") or "").strip())


def _rss_only_mode() -> bool:
    """When true, never call NewsAPI — RSS feeds only."""
    return str(os.getenv("NEWS_RSS_ONLY", "")).strip().lower() in ("1", "true", "yes", "on")


def _rss_fetch_interval_sec() -> int:
    raw = os.getenv("NEWS_RSS_FETCH_INTERVAL_SEC") or os.getenv("NEWS_FETCH_INTERVAL_SEC") or "300"
    return max(120, int(str(raw).split()[0] or 300))


def _api_fetch_interval_sec() -> int:
    return max(3600, int(str(os.getenv("NEWS_API_FETCH_INTERVAL_SEC", "21600")).split()[0] or 21600))


def _api_max_calls_per_day() -> int:
    return max(1, min(100, int(str(os.getenv("NEWS_API_MAX_CALLS_PER_DAY", "90")).split()[0] or 90)))


def _corpus_age_sec(corpus: NewsCorpus | None) -> float | None:
    if corpus is None or not corpus.ts_utc:
        return None
    try:
        dt = datetime.fromisoformat(str(corpus.ts_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except (TypeError, ValueError):
        return None


def _article_text(art: dict[str, Any]) -> str:
    title = str(art.get("title") or "")
    desc = str(art.get("description") or "")
    return f"{title} {desc}".strip()


def article_matches_base(text: str, base: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    coin = _base_from_symbol(base)
    aliases = _SYMBOL_ALIASES.get(coin, (coin.lower(),))
    for alias in aliases:
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return True
    return f"{coin.lower()}usdt" in t.replace("/", "").replace("-", "")


@dataclass
class NewsCorpus:
    articles: list[dict[str, Any]] = field(default_factory=list)
    fetch_ok: bool = False
    error: str | None = None
    ts_utc: str = ""
    from_cache: bool = False
    source: str = ""


def _cache_ttl_sec() -> int:
    return max(120, int(str(os.getenv("NEWS_CACHE_TTL_SEC", "300")).split()[0] or 300))


def _cooldown_sec() -> int:
    return max(600, int(str(os.getenv("NEWS_RATE_LIMIT_COOLDOWN_SEC", "1800")).split()[0] or 1800))


async def _publish_news_status(redis_client: Any | None, payload: dict[str, Any]) -> None:
    if redis_client is None:
        return
    try:
        await redis_client.set(
            REDIS_NEWS_STATUS_KEY,
            json.dumps(payload, separators=(",", ":")),
            ex=max(300, _cache_ttl_sec()),
        )
    except Exception as exc:
        logger.debug("news status redis write failed: %s", exc)


def score_news_corpus_for_base(corpus: NewsCorpus | list[dict[str, Any]], base: str) -> tuple[float, list[dict[str, Any]]] | None:
    """Return (score, matched_articles) or None if no matching articles."""
    articles = corpus.articles if isinstance(corpus, NewsCorpus) else list(corpus)
    coin = _base_from_symbol(base)
    matched: list[dict[str, Any]] = []
    scores: list[float] = []
    for art in articles:
        text = str(art.get("text") or _article_text(art))
        if not article_matches_base(text, coin):
            continue
        sc = float(art.get("score")) if art.get("score") is not None else _score_text(text)
        scores.append(sc)
        matched.append({**art, "score": sc, "tier": art.get("tier") or "symbol_specific"})
    if not scores:
        return None
    avg = max(-1.0, min(1.0, sum(scores) / len(scores)))
    return avg, matched[:25]


def _normalize_article(art: dict[str, Any], *, tier: str = "market_wide") -> dict[str, Any] | None:
    if not isinstance(art, dict):
        return None
    title = str(art.get("title") or "")
    text = _article_text(art)
    if not text:
        return None
    sc = _score_text(text)
    return {
        "title": title[:240],
        "text": text,
        "score": sc,
        "publishedAt": art.get("publishedAt"),
        "source": (art.get("source") or {}).get("name") if isinstance(art.get("source"), dict) else None,
        "tier": tier,
    }


async def _load_corpus_cache(redis_client: Any | None) -> NewsCorpus | None:
    if redis_client is None:
        return None
    try:
        raw = await redis_client.get(REDIS_NEWS_CORPUS_KEY)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        arts = [a for a in data.get("articles") or [] if isinstance(a, dict)]
        if not arts:
            return None
        return NewsCorpus(
            articles=arts,
            fetch_ok=True,
            error=None,
            ts_utc=str(data.get("ts_utc") or ""),
            from_cache=True,
            source=str(data.get("source") or "cache"),
        )
    except Exception:
        return None


async def _save_corpus_cache(redis_client: Any | None, corpus: NewsCorpus) -> None:
    if redis_client is None or not corpus.articles:
        return
    ttl = _cache_ttl_sec()
    payload = {
        "ts_utc": corpus.ts_utc or datetime.now(timezone.utc).isoformat(),
        "articles": corpus.articles,
        "source": corpus.source or "newsapi",
    }
    try:
        await redis_client.set(REDIS_NEWS_CORPUS_KEY, json.dumps(payload, separators=(",", ":")), ex=ttl)
    except Exception as exc:
        logger.debug("news corpus cache write failed: %s", exc)


async def _rate_limit_active(redis_client: Any | None) -> bool:
    if redis_client is None:
        return False
    try:
        raw = await redis_client.get(REDIS_NEWS_COOLDOWN_KEY)
        return bool(raw)
    except Exception:
        return False


def _api_calls_key() -> str:
    return f"news_sentiment:api_calls:{datetime.now(timezone.utc).date().isoformat()}"


async def _api_calls_today(redis_client: Any | None) -> int:
    if redis_client is None:
        return 0
    try:
        raw = await redis_client.get(_api_calls_key())
        if not raw:
            return 0
        return int(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        return 0


async def _record_api_call(redis_client: Any | None) -> None:
    if redis_client is None:
        return
    key = _api_calls_key()
    try:
        count = await _api_calls_today(redis_client)
        await redis_client.set(key, str(count + 1), ex=86400 * 2)
        await redis_client.set("news_sentiment:api_last_fetch_epoch", str(time.time()), ex=86400 * 7)
    except Exception as exc:
        logger.debug("news api call counter write failed: %s", exc)


async def _should_call_newsapi(redis_client: Any | None) -> bool:
    if not _news_enabled():
        return False
    if await _rate_limit_active(redis_client):
        return False
    if await _api_calls_today(redis_client) >= _api_max_calls_per_day():
        return False
    if redis_client is None:
        return True
    try:
        raw = await redis_client.get("news_sentiment:api_last_fetch_epoch")
        if not raw:
            return True
        last = float(raw.decode() if isinstance(raw, bytes) else raw)
        return (time.time() - last) >= float(_api_fetch_interval_sec())
    except Exception:
        return True


def _merge_articles(primary: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for art in primary + extra:
        title = str(art.get("title") or "").lower()[:120]
        if not title or title in seen:
            continue
        seen.add(title)
        merged.append(art)
    return merged[:100]


async def _set_rate_limit_cooldown(redis_client: Any | None) -> None:
    if redis_client is None:
        return
    cooldown = _cooldown_sec()
    try:
        await redis_client.set(REDIS_NEWS_COOLDOWN_KEY, "1", ex=cooldown)
        await _publish_news_status(
            redis_client,
            {
                "available": "no",
                "reason": "news_rate_limited_cooldown",
                "cooldown_sec": cooldown,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        logger.debug("news rate limit cooldown set failed: %s", exc)


def _rss_feed_urls() -> list[str]:
    raw = (os.getenv("NEWS_RSS_FEEDS") or "").strip()
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    return list(_DEFAULT_RSS_FEEDS)


async def _fetch_rss_corpus(*, http_client: httpx.AsyncClient) -> NewsCorpus:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        import feedparser
    except ImportError:
        return NewsCorpus([], fetch_ok=False, error="rss_feedparser_missing", ts_utc=ts, source="rss")

    seen: set[str] = set()
    articles: list[dict[str, Any]] = []
    for url in _rss_feed_urls():
        try:
            resp = await http_client.get(url, timeout=12.0)
            if resp.status_code != 200:
                continue
            feed = feedparser.parse(resp.text)
            for entry in feed.entries[:40]:
                if not isinstance(entry, dict):
                    continue
                title = str(getattr(entry, "title", "") or "")
                summary = str(getattr(entry, "summary", "") or getattr(entry, "description", "") or "")
                text = f"{title} {summary}".strip()
                if not text:
                    continue
                key = title.lower()[:120]
                if key in seen:
                    continue
                seen.add(key)
                norm = _normalize_article({"title": title, "description": summary}, tier="rss")
                if norm:
                    norm["source"] = str(getattr(entry, "source", None) or url)
                    articles.append(norm)
        except Exception as exc:
            logger.debug("rss fetch failed url=%s: %s", url, exc)
            continue

    if not articles:
        return NewsCorpus([], fetch_ok=False, error="rss_no_articles", ts_utc=ts, source="rss")
    return NewsCorpus(articles=articles[:80], fetch_ok=True, error=None, ts_utc=ts, from_cache=False, source="rss")


async def read_cached_news_corpus(redis_client: Any | None) -> NewsCorpus | None:
    """Read-only corpus lookup — never calls NewsAPI."""
    return await _load_corpus_cache(redis_client)


async def read_news_sentiment_from_collector(
    redis_client: Any | None,
    base_symbol: str,
) -> tuple[float | None, str]:
    """
    Score news for ``base_symbol`` from collector cache only.
    Returns (score_or_none, path_label). Never hits NewsAPI.
    """
    base = _base_from_symbol(base_symbol)
    corpus = await read_cached_news_corpus(redis_client)
    if corpus is None or not corpus.articles:
        if await _rate_limit_active(redis_client):
            return None, "news_unavailable_rate_limited"
        return None, "news_cache_empty"
    if not corpus.fetch_ok and corpus.error:
        if "rate_limit" in str(corpus.error):
            return None, "news_unavailable_rate_limited"
    scored = score_news_corpus_for_base(corpus, base)
    if scored is None:
        return None, "news_ok_no_symbol_match"
    score, _arts = scored
    src = corpus.source or ("cache" if corpus.from_cache else "corpus")
    return float(score), f"news_collector_{src}"


async def _fetch_newsapi_corpus(
    *,
    http_client: httpx.AsyncClient,
    redis_client: Any | None,
) -> NewsCorpus | None:
    """Optional NewsAPI enrich — returns None when skipped or failed."""
    if not _news_enabled():
        return None
    key = (os.getenv("NEWS_API_KEY") or "").strip()
    url = (os.getenv("NEWS_API_URL") or "https://newsapi.org/v2/everything").strip()
    query = (os.getenv("NEWS_BATCH_QUERY") or "bitcoin OR ethereum OR solana OR ripple OR cryptocurrency").strip()
    page_size = max(10, min(100, int(str(os.getenv("NEWS_BATCH_PAGE_SIZE", "50")).split()[0] or 50)))
    ts = datetime.now(timezone.utc).isoformat()
    try:
        resp = await http_client.get(
            url,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": key,
            },
        )
        await _record_api_call(redis_client)
        if resp.status_code == 429:
            await _set_rate_limit_cooldown(redis_client)
            logger.info("NewsAPI 429 — daily quota hit; RSS corpus continues until cooldown")
            return None
        if resp.status_code != 200:
            logger.debug("NewsAPI http_%s — RSS corpus continues", resp.status_code)
            return None
        seen: set[str] = set()
        articles: list[dict[str, Any]] = []
        for art in resp.json().get("articles") or []:
            norm = _normalize_article(art if isinstance(art, dict) else {})
            if norm is None:
                continue
            key_title = norm["title"].lower()
            if key_title in seen:
                continue
            seen.add(key_title)
            articles.append(norm)
        if not articles:
            return None
        return NewsCorpus(articles=articles, fetch_ok=True, error=None, ts_utc=ts, from_cache=False, source="newsapi")
    except Exception as exc:
        logger.debug("NewsAPI fetch failed: %s", exc)
        return None


async def fetch_news_corpus(
    *,
    http_client: httpx.AsyncClient | None = None,
    redis_client: Any | None = None,
    force_refresh: bool = False,
) -> NewsCorpus:
    """
    Sole news writer for Mystic.

    RSS refresh runs on ``NEWS_RSS_FETCH_INTERVAL_SEC`` (default 300s) for server freshness.
    NewsAPI is optional enrichment on ``NEWS_API_FETCH_INTERVAL_SEC`` (default 6h), capped by
    ``NEWS_API_MAX_CALLS_PER_DAY`` (default 90) to stay within the free 100/day tier.
    """
    ts = datetime.now(timezone.utc).isoformat()
    cached = await _load_corpus_cache(redis_client)
    cache_age = _corpus_age_sec(cached)

    if not force_refresh and cached is not None and cache_age is not None:
        if cache_age < float(_rss_fetch_interval_sec()):
            cached.from_cache = True
            await _publish_news_status(
                redis_client,
                {
                    "available": "yes",
                    "reason": "cache_hit",
                    "article_count": len(cached.articles),
                    "source": cached.source or "cache",
                    "cache_age_sec": round(cache_age, 1),
                    "ts_utc": cached.ts_utc or ts,
                },
            )
            return cached

    own = http_client is None
    client = http_client or httpx.AsyncClient(timeout=14.0)
    try:
        rss = await _fetch_rss_corpus(http_client=client)
        articles = list(rss.articles) if rss.fetch_ok and rss.articles else []
        source = rss.source if articles else "rss"
        error = None if articles else (rss.error or "rss_no_articles")

        api_used = False
        if not _rss_only_mode() and await _should_call_newsapi(redis_client):
            api_corpus = await _fetch_newsapi_corpus(http_client=client, redis_client=redis_client)
            if api_corpus is not None and api_corpus.articles:
                articles = _merge_articles(articles, api_corpus.articles)
                source = "rss+newsapi" if rss.articles else "newsapi"
                api_used = True

        if not articles and cached is not None:
            cached.from_cache = True
            cached.error = error or cached.error
            return cached

        if not articles:
            await _publish_news_status(
                redis_client,
                {
                    "available": "no",
                    "reason": error or "news_corpus_empty",
                    "source": "rss",
                    "ts_utc": ts,
                },
            )
            return NewsCorpus([], fetch_ok=False, error=error or "news_corpus_empty", ts_utc=ts, source="rss")

        corpus = NewsCorpus(
            articles=articles,
            fetch_ok=True,
            error=None,
            ts_utc=ts,
            from_cache=False,
            source=source,
        )
        await _save_corpus_cache(redis_client, corpus)
        await _publish_news_status(
            redis_client,
            {
                "available": "yes",
                "reason": "rss_refresh_ok" if not api_used else "rss_and_newsapi_ok",
                "article_count": len(articles),
                "source": source,
                "api_calls_today": await _api_calls_today(redis_client),
                "api_max_per_day": _api_max_calls_per_day(),
                "rss_interval_sec": _rss_fetch_interval_sec(),
                "api_interval_sec": _api_fetch_interval_sec(),
                "ts_utc": ts,
            },
        )
        return corpus
    except Exception as exc:
        logger.debug("fetch_news_corpus failed: %s", exc)
        if cached is not None:
            cached.error = str(exc)[:200]
            return cached
        return NewsCorpus([], fetch_ok=False, error=str(exc)[:200], ts_utc=ts, source="rss")
    finally:
        if own:
            await client.aclose()


async def crypto_news_articles(base_symbol: str, *, http_client: httpx.AsyncClient | None = None) -> tuple[float, list[dict[str, Any]]]:
    """
    Read-only scoring from cached corpus. Does not call NewsAPI (use collector fetch_news_corpus).
    """
    from backend.config.redis_config import get_shared_redis_async

    redis_client = await get_shared_redis_async()
    score, _path = await read_news_sentiment_from_collector(redis_client, base_symbol)
    if score is None:
        return 0.0, []
    corpus = await read_cached_news_corpus(redis_client)
    if corpus is None:
        return float(score), []
    scored = score_news_corpus_for_base(corpus, base_symbol)
    if scored is None:
        return float(score), []
    avg, arts = scored
    return avg, arts


def score_text(text: str) -> float:
    return _score_text(text)


__all__ = [
    "REDIS_NEWS_COOLDOWN_KEY",
    "REDIS_NEWS_CORPUS_KEY",
    "REDIS_NEWS_STATUS_KEY",
    "NewsCorpus",
    "article_matches_base",
    "crypto_news_articles",
    "fetch_news_corpus",
    "read_cached_news_corpus",
    "read_news_sentiment_from_collector",
    "score_news_corpus_for_base",
    "score_text",
]
