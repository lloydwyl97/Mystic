"""Reddit OAuth social polarity — real posts only."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_REDDIT_BASES = ("BTC", "ETH", "SOL", "XRP")


def _base(symbol: str) -> str:
    s = (symbol or "BTC").upper().replace("/USDT", "").replace("USDT", "").strip()
    return s or "BTC"


def _score_title(title: str) -> float:
    from backend.services.crypto_sentiment_vader import score_crypto_text

    return score_crypto_text(title)


def _reddit_enabled() -> bool:
    return os.getenv("ENABLE_REDDIT", "true").strip().lower() not in ("0", "false", "no", "off")


async def _reddit_token(client: httpx.AsyncClient) -> str | None:
    cid = (os.getenv("REDDIT_CLIENT_ID") or "").strip()
    secret = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
    if not cid or not secret:
        return None
    ua = (os.getenv("REDDIT_USER_AGENT") or "MysticTrader/1.0").strip()
    try:
        resp = await client.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
            headers={"User-Agent": ua},
        )
        if resp.status_code != 200:
            return None
        return str(resp.json().get("access_token") or "") or None
    except Exception as exc:
        logger.debug("reddit token failed: %s", exc)
        return None


@dataclass
class RedditCorpus:
    titles: list[str]
    fetch_ok: bool = False
    error: str | None = None


def score_reddit_corpus_for_base(corpus: RedditCorpus | list[str], base: str) -> tuple[float, int] | None:
    titles = corpus.titles if isinstance(corpus, RedditCorpus) else list(corpus)
    coin = _base(base)
    scores: list[float] = []
    for title in titles:
        if coin.lower() not in title.lower() and coin not in ("BTC", "ETH"):
            continue
        scores.append(_score_title(title))
    if not scores:
        return None
    return max(-1.0, min(1.0, sum(scores) / len(scores))), len(scores)


async def fetch_reddit_corpus(*, http_client: httpx.AsyncClient | None = None) -> RedditCorpus:
    """
    Fetch Reddit titles once per sentiment pass (hot subs + per-base search).
    """
    if not _reddit_enabled():
        return RedditCorpus([], fetch_ok=False, error="reddit_disabled")
    ua = (os.getenv("REDDIT_USER_AGENT") or "MysticTrader/1.0").strip()
    own = http_client is None
    client = http_client or httpx.AsyncClient(timeout=14.0)
    titles: list[str] = []
    try:
        token = await _reddit_token(client)
        if not token:
            return RedditCorpus([], fetch_ok=False, error="reddit_token_failed")
        headers = {"Authorization": f"bearer {token}", "User-Agent": ua}
        subs = ["CryptoCurrency", "CryptoMarkets", "altcoin"]
        for sub in subs:
            try:
                resp = await client.get(
                    f"https://oauth.reddit.com/r/{sub}/hot",
                    params={"limit": 25},
                    headers=headers,
                )
                if resp.status_code != 200:
                    continue
                for child in resp.json().get("data", {}).get("children") or []:
                    if not isinstance(child, dict):
                        continue
                    title = str((child.get("data") or {}).get("title") or "")
                    if title:
                        titles.append(title)
            except Exception as exc:
                logger.debug("reddit hot scan failed sub=%s: %s", sub, exc)
        for base in _REDDIT_BASES:
            try:
                resp = await client.get(
                    "https://oauth.reddit.com/search",
                    params={"q": base, "limit": 20, "sort": "new", "restrict_sr": "false"},
                    headers=headers,
                )
                if resp.status_code != 200:
                    continue
                for child in resp.json().get("data", {}).get("children") or []:
                    title = str((child.get("data") or {}).get("title") or "")
                    if title:
                        titles.append(title)
            except Exception as exc:
                logger.debug("reddit search failed base=%s: %s", base, exc)
        if not titles:
            return RedditCorpus([], fetch_ok=True, error="reddit_no_posts")
        return RedditCorpus(titles, fetch_ok=True, error=None)
    except Exception as exc:
        return RedditCorpus([], fetch_ok=False, error=str(exc)[:200])
    finally:
        if own:
            await client.aclose()


async def fetch_reddit_social_polarity(symbol: str, *, http_client: httpx.AsyncClient | None = None) -> tuple[float, int] | None:
    """Backward-compatible single-symbol fetch."""
    corpus = await fetch_reddit_corpus(http_client=http_client)
    if not corpus.fetch_ok or not corpus.titles:
        return None
    return score_reddit_corpus_for_base(corpus, symbol)


__all__ = ["RedditCorpus", "fetch_reddit_corpus", "fetch_reddit_social_polarity", "score_reddit_corpus_for_base"]
