"""
Active multi-source sentiment collector for DAY top-4 engine.

Sources: Reddit, Telegram, News API, Fear & Greed (alternative.me), Discord bot read.
Discord webhook remains outbound-only for alerts; bot token + channel ID required for read.
Twitter/X intentionally disabled.

Never fabricates scores. Failed/missing sources are marked inactive.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.services.ai_decision_contract import REDIS_KEY_AI_SENTIMENT
from backend.services.market_regime import regime_score

logger = logging.getLogger(__name__)

REDIS_SENTIMENT_SYMBOL_PREFIX = "ai_sentiment:"
REDIS_SENTIMENT_STATUS_KEY = "ai_sentiment:status"


def _env_interval_sec(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(str(raw).split()[0] or default))
    except (TypeError, ValueError):
        return default


def sentiment_fetch_intervals() -> dict[str, int]:
    from backend.services.ai_decision_contract import AI_SENTIMENT_LOOP_SEC

    return {
        "discord": _env_interval_sec("DISCORD_FETCH_INTERVAL_SEC", 120),
        "reddit": _env_interval_sec("REDDIT_FETCH_INTERVAL_SEC", 180),
        "news": _env_interval_sec("NEWS_RSS_FETCH_INTERVAL_SEC", _env_interval_sec("NEWS_FETCH_INTERVAL_SEC", 300)),
        "telegram": _env_interval_sec("TELEGRAM_FETCH_INTERVAL_SEC", 60),
        "fear_greed": _env_interval_sec("FEAR_GREED_FETCH_INTERVAL_SEC", AI_SENTIMENT_LOOP_SEC),
        "collector_tick": _env_interval_sec("SENTIMENT_COLLECT_INTERVAL_SEC", 30),
    }


def collector_loop_interval_sec() -> int:
    cfg = sentiment_fetch_intervals()
    return max(15, min(cfg.values()))


def _source_fetch_due(last_ts: float, interval_sec: int, *, has_cache: bool) -> bool:
    if not has_cache:
        return True
    return (time.time() - last_ts) >= float(interval_sec)


@dataclass
class _SourceFetchCache:
    discord_messages: list[dict[str, Any]] = field(default_factory=list)
    discord_meta: dict[str, Any] = field(default_factory=lambda: {"fetch_ok": False, "error": "discord_fetch_not_run", "message_count": 0})
    discord_ts: float = 0.0
    reddit_corpus: Any = None
    reddit_ts: float = 0.0
    news_corpus: Any = None
    news_ts: float = 0.0
    telegram_messages: list[dict[str, Any]] = field(default_factory=list)
    telegram_meta: dict[str, Any] = field(default_factory=lambda: {"fetch_ok": False, "error": "telegram_fetch_not_run"})
    telegram_ts: float = 0.0
    fear_greed_score: float | None = None
    fear_greed_err: str | None = None
    fear_greed_ts: float = 0.0


_fetch_cache = _SourceFetchCache()


def _truthy(v: str | None, default: bool = True) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _base(symbol_bus: str) -> str:
    s = (symbol_bus or "BTCUSDT").upper().replace("/USDT", "").replace("USDT", "").strip()
    return s or "BTC"


def _configured_chat_id_masked() -> str:
    raw = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not raw:
        return ""
    if raw.lstrip("-").isdigit():
        return f"{raw[:3]}***{raw[-2:]}" if len(raw) > 5 else "***"
    name = raw.lstrip("@")
    return f"@{name[:2]}***" if len(name) > 2 else "@***"


def env_source_config(*, discord_fetch_ok: bool | None = None) -> dict[str, Any]:
    from backend.services.discord_social_sentiment_live import discord_readiness

    reddit_on = _truthy(os.getenv("ENABLE_REDDIT"), True) and bool(os.getenv("REDDIT_CLIENT_ID", "").strip()) and bool(os.getenv("REDDIT_CLIENT_SECRET", "").strip())
    discord = discord_readiness()
    discord_bot = discord["bot_configured"]
    discord_channel = discord["channel_configured"]
    discord_webhook = discord["webhook_only"] or bool(os.getenv("DISCORD_WEBHOOK", "").strip() or os.getenv("DISCORD_WEBHOOK_URL", "").strip())
    read_ready = discord["read_ready"]
    read_active = read_ready and discord_fetch_ok is True
    telegram_on = _truthy(os.getenv("ENABLE_TELEGRAM"), True) and bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
    news_on = bool(os.getenv("NEWS_API_KEY", "").strip())
    twitter_disabled = not _truthy(os.getenv("ENABLE_TWITTER"), False)
    return {
        "reddit": {"enabled": reddit_on, "configured": reddit_on},
        "discord": {
            "enabled": read_ready,
            "configured": discord["configured"],
            "bot_configured": discord_bot,
            "channel_configured": discord_channel,
            "read_ready": read_ready,
            "read_active": read_active,
            "webhook_only": discord_webhook and not discord_bot,
        },
        "telegram": {
            "enabled": telegram_on,
            "configured": telegram_on and bool(os.getenv("TELEGRAM_CHAT_ID", "").strip()),
        },
        "news": {"enabled": news_on, "configured": news_on},
        "fear_greed": {"enabled": True, "configured": True},
        "twitter": {"enabled": False, "configured": False, "disabled_intentionally": twitter_disabled},
    }


async def _fetch_fear_greed() -> tuple[float | None, str | None]:
    try:
        s = await regime_score()
        return max(-1.0, min(1.0, float(s))), None
    except Exception as exc:
        return None, str(exc)


async def collect_symbol_sentiment(
    symbol_bus: str,
    *,
    http_client: httpx.AsyncClient | None = None,
    discord_messages: list[dict[str, Any]] | None = None,
    discord_meta: dict[str, Any] | None = None,
    reddit_corpus: Any | None = None,
    telegram_messages: list[dict[str, Any]] | None = None,
    telegram_meta: dict[str, Any] | None = None,
    news_corpus: Any | None = None,
    fear_greed_score: float | None = None,
    fear_greed_err: str | None = None,
) -> dict[str, Any]:
    base = _base(symbol_bus)
    cfg = env_source_config(discord_fetch_ok=(discord_meta or {}).get("fetch_ok"))
    ts = datetime.now(timezone.utc).isoformat()
    sources_active: list[str] = []
    sources_missing: list[str] = []
    breakdown: dict[str, Any] = {}

    own = http_client is None
    client = http_client or httpx.AsyncClient(timeout=14.0)

    if fear_greed_score is not None:
        fg, fg_err = fear_greed_score, None
    elif fear_greed_err is not None:
        fg, fg_err = None, fear_greed_err
    else:
        fg, fg_err = await _fetch_fear_greed()
    if fg is not None:
        breakdown["fear_greed_index"] = fg
        sources_active.append("fear_greed")
    else:
        sources_missing.append("fear_greed")
        breakdown["fear_greed_error"] = fg_err

    reddit_score = None
    if cfg["reddit"]["enabled"]:
        try:
            from backend.services.reddit_social_sentiment_live import score_reddit_corpus_for_base
            from backend.services.sentiment_source_status import (
                SOURCE_FETCH_FAILED,
                SOURCE_OK_MATCHED,
                SOURCE_OK_NO_SYMBOL_MATCH,
                apply_source_status,
            )

            if reddit_corpus is not None and getattr(reddit_corpus, "fetch_ok", False):
                tup = score_reddit_corpus_for_base(reddit_corpus, base)
                if tup is not None:
                    reddit_score, n_posts = tup
                    apply_source_status(
                        source_name="reddit",
                        status=SOURCE_OK_MATCHED,
                        breakdown=breakdown,
                        sources_active=sources_active,
                        sources_missing=sources_missing,
                        score_fields={"reddit_sentiment_score": reddit_score, "reddit_post_count": n_posts},
                    )
                else:
                    apply_source_status(
                        source_name="reddit",
                        status=SOURCE_OK_NO_SYMBOL_MATCH,
                        breakdown=breakdown,
                        sources_active=sources_active,
                        sources_missing=sources_missing,
                        error=getattr(reddit_corpus, "error", None) or "reddit_no_matching_posts",
                    )
            else:
                err = getattr(reddit_corpus, "error", None) if reddit_corpus is not None else "reddit_fetch_not_run"
                apply_source_status(
                    source_name="reddit",
                    status=SOURCE_FETCH_FAILED,
                    breakdown=breakdown,
                    sources_active=sources_active,
                    sources_missing=sources_missing,
                    error=str(err) if err else "reddit_fetch_failed",
                )
        except Exception as exc:
            from backend.services.sentiment_source_status import SOURCE_FETCH_FAILED, apply_source_status

            apply_source_status(
                source_name="reddit",
                status=SOURCE_FETCH_FAILED,
                breakdown=breakdown,
                sources_active=sources_active,
                sources_missing=sources_missing,
                error=str(exc)[:200],
            )
    else:
        sources_missing.append("reddit")

    news_score = None
    if cfg["news"]["enabled"]:
        try:
            from backend.services.news_sentiment import score_news_corpus_for_base
            from backend.services.sentiment_source_status import (
                SOURCE_FETCH_FAILED,
                SOURCE_OK_MATCHED,
                SOURCE_OK_NO_SYMBOL_MATCH,
                apply_source_status,
            )

            if news_corpus is not None and getattr(news_corpus, "fetch_ok", False) and news_corpus.articles:
                scored = score_news_corpus_for_base(news_corpus, base)
                if scored is not None:
                    news_score, arts = scored
                    apply_source_status(
                        source_name="news",
                        status=SOURCE_OK_MATCHED,
                        breakdown=breakdown,
                        sources_active=sources_active,
                        sources_missing=sources_missing,
                        score_fields={
                            "news_sentiment_score": news_score,
                            "news_article_count": len(arts),
                            "news_source": getattr(news_corpus, "source", "") or "",
                        },
                    )
                else:
                    apply_source_status(
                        source_name="news",
                        status=SOURCE_OK_NO_SYMBOL_MATCH,
                        breakdown=breakdown,
                        sources_active=sources_active,
                        sources_missing=sources_missing,
                        error="news_no_matching_articles",
                    )
            else:
                err = getattr(news_corpus, "error", None) if news_corpus is not None else "news_fetch_not_run"
                err_s = str(err or "news_fetch_not_run")
                if "rate_limit" in err_s or err_s in ("news_rate_limited_unavailable", "news_http_429"):
                    breakdown["news_unavailable"] = "yes"
                apply_source_status(
                    source_name="news",
                    status=SOURCE_FETCH_FAILED,
                    breakdown=breakdown,
                    sources_active=sources_active,
                    sources_missing=sources_missing,
                    error=err_s,
                )
        except Exception as exc:
            from backend.services.sentiment_source_status import SOURCE_FETCH_FAILED, apply_source_status

            apply_source_status(
                source_name="news",
                status=SOURCE_FETCH_FAILED,
                breakdown=breakdown,
                sources_active=sources_active,
                sources_missing=sources_missing,
                error=str(exc)[:200],
            )
    else:
        sources_missing.append("news")

    telegram_score = None
    tg_meta = telegram_meta or {}
    if cfg["telegram"]["enabled"]:
        from backend.services.sentiment_source_status import (
            SOURCE_FETCH_FAILED,
            SOURCE_OK_MATCHED,
            SOURCE_OK_NO_SYMBOL_MATCH,
            apply_source_status,
        )

        if tg_meta.get("fetch_ok"):
            try:
                from backend.services.telegram_social_sentiment_live import score_cached_messages_for_base

                msgs = telegram_messages or []
                breakdown["telegram_message_count"] = len(msgs)
                breakdown["telegram_ts_utc"] = tg_meta.get("ts_utc") or ts
                breakdown["telegram_chat_id"] = _configured_chat_id_masked()
                if not msgs:
                    apply_source_status(
                        source_name="telegram",
                        status=SOURCE_OK_NO_SYMBOL_MATCH,
                        breakdown=breakdown,
                        sources_active=sources_active,
                        sources_missing=sources_missing,
                        error="telegram_no_channel_posts",
                        score_fields={"telegram_matched_count": 0},
                    )
                else:
                    scored = score_cached_messages_for_base(msgs, base)
                    if scored is not None:
                        telegram_score, _total, matched = scored
                        apply_source_status(
                            source_name="telegram",
                            status=SOURCE_OK_MATCHED,
                            breakdown=breakdown,
                            sources_active=sources_active,
                            sources_missing=sources_missing,
                            score_fields={
                                "telegram_sentiment_score": telegram_score,
                                "telegram_matched_count": matched,
                            },
                        )
                    else:
                        apply_source_status(
                            source_name="telegram",
                            status=SOURCE_OK_NO_SYMBOL_MATCH,
                            breakdown=breakdown,
                            sources_active=sources_active,
                            sources_missing=sources_missing,
                            error="telegram_no_matching_messages",
                            score_fields={"telegram_matched_count": 0},
                        )
            except Exception as exc:
                apply_source_status(
                    source_name="telegram",
                    status=SOURCE_FETCH_FAILED,
                    breakdown=breakdown,
                    sources_active=sources_active,
                    sources_missing=sources_missing,
                    error=str(exc)[:200],
                )
        else:
            apply_source_status(
                source_name="telegram",
                status=SOURCE_FETCH_FAILED,
                breakdown=breakdown,
                sources_active=sources_active,
                sources_missing=sources_missing,
                error=str(tg_meta.get("error") or "telegram_fetch_failed")[:200],
            )
    else:
        sources_missing.append("telegram")

    discord_score = None
    meta = discord_meta or {}
    if cfg["discord"]["bot_configured"]:
        from backend.services.sentiment_source_status import (
            SOURCE_FETCH_FAILED,
            SOURCE_OK_MATCHED,
            SOURCE_OK_NO_SYMBOL_MATCH,
            apply_source_status,
        )

        if not cfg["discord"]["channel_configured"]:
            apply_source_status(
                source_name="discord",
                status=SOURCE_FETCH_FAILED,
                breakdown=breakdown,
                sources_active=sources_active,
                sources_missing=sources_missing,
                error="discord_channel_missing",
            )
            breakdown["discord_stale"] = "yes"
        elif meta.get("error") and not meta.get("fetch_ok"):
            apply_source_status(
                source_name="discord",
                status=SOURCE_FETCH_FAILED,
                breakdown=breakdown,
                sources_active=sources_active,
                sources_missing=sources_missing,
                error=str(meta["error"])[:200],
            )
            breakdown["discord_stale"] = "yes"
            if meta.get("ts_utc"):
                breakdown["discord_ts_utc"] = meta["ts_utc"]
        elif meta.get("fetch_ok"):
            msgs = discord_messages or []
            breakdown["discord_message_count"] = len(msgs)
            breakdown["discord_ts_utc"] = meta.get("ts_utc") or ts
            breakdown["discord_stale"] = "no"
            try:
                from backend.services.discord_social_sentiment_live import score_messages_for_base

                if not msgs:
                    apply_source_status(
                        source_name="discord",
                        status=SOURCE_OK_NO_SYMBOL_MATCH,
                        breakdown=breakdown,
                        sources_active=sources_active,
                        sources_missing=sources_missing,
                        error="discord_no_messages_in_corpus",
                        score_fields={"discord_matched_count": 0},
                    )
                else:
                    scored = score_messages_for_base(msgs, base)
                    if scored is not None:
                        discord_score, _total, matched = scored
                        apply_source_status(
                            source_name="discord",
                            status=SOURCE_OK_MATCHED,
                            breakdown=breakdown,
                            sources_active=sources_active,
                            sources_missing=sources_missing,
                            score_fields={
                                "discord_sentiment_score": discord_score,
                                "discord_matched_count": matched,
                            },
                        )
                    else:
                        apply_source_status(
                            source_name="discord",
                            status=SOURCE_OK_NO_SYMBOL_MATCH,
                            breakdown=breakdown,
                            sources_active=sources_active,
                            sources_missing=sources_missing,
                            error="discord_no_matching_messages",
                            score_fields={"discord_matched_count": 0},
                        )
            except Exception as exc:
                apply_source_status(
                    source_name="discord",
                    status=SOURCE_FETCH_FAILED,
                    breakdown=breakdown,
                    sources_active=sources_active,
                    sources_missing=sources_missing,
                    error=str(exc)[:200],
                )
                breakdown["discord_stale"] = "yes"
        else:
            apply_source_status(
                source_name="discord",
                status=SOURCE_FETCH_FAILED,
                breakdown=breakdown,
                sources_active=sources_active,
                sources_missing=sources_missing,
                error="discord_fetch_not_run",
            )
            breakdown["discord_stale"] = "yes"
    else:
        sources_missing.append("discord")
        if cfg["discord"].get("webhook_only"):
            breakdown["discord_error"] = "discord_bot_not_configured"
        else:
            breakdown["discord_error"] = "discord_not_configured"
        breakdown["discord_stale"] = "yes"

    social_parts = [v for v in (reddit_score, telegram_score, discord_score) if v is not None]
    social_composite = max(-1.0, min(1.0, sum(social_parts) / len(social_parts))) if social_parts else None

    payload: dict[str, Any] = {
        "symbol": symbol_bus,
        "base": base,
        "sentiment_ts_utc": ts,
        "sentiment_stale": "no",
        "fear_greed_index": breakdown.get("fear_greed_index"),
        "reddit_sentiment_score": breakdown.get("reddit_sentiment_score"),
        "discord_sentiment_score": breakdown.get("discord_sentiment_score"),
        "discord_message_count": breakdown.get("discord_message_count"),
        "discord_matched_count": breakdown.get("discord_matched_count"),
        "discord_ts_utc": breakdown.get("discord_ts_utc"),
        "discord_stale": breakdown.get("discord_stale"),
        "discord_error": breakdown.get("discord_error"),
        "telegram_sentiment_score": breakdown.get("telegram_sentiment_score"),
        "telegram_message_count": breakdown.get("telegram_message_count"),
        "telegram_matched_count": breakdown.get("telegram_matched_count"),
        "telegram_ts_utc": breakdown.get("telegram_ts_utc"),
        "telegram_error": breakdown.get("telegram_error"),
        "news_sentiment_score": breakdown.get("news_sentiment_score"),
        "reddit_status": breakdown.get("reddit_status"),
        "news_status": breakdown.get("news_status"),
        "telegram_status": breakdown.get("telegram_status"),
        "discord_status": breakdown.get("discord_status"),
        "social_sentiment_score": social_composite,
        "sentiment_sources_active": ",".join(sources_active),
        "sentiment_sources_missing": ",".join(sources_missing),
        "breakdown_json": json.dumps(breakdown, separators=(",", ":")),
        "twitter_disabled": "yes",
    }
    if own:
        await client.aclose()
    return payload


async def publish_symbol_sentiment_to_redis(symbol_bus: str, payload: dict[str, Any], redis_client: Any) -> None:
    if redis_client is None:
        return
    key = f"{REDIS_SENTIMENT_SYMBOL_PREFIX}{symbol_bus}"
    mapping = {k: str(v) if v is not None else "" for k, v in payload.items()}
    ttl = max(120, int(float(os.getenv("SENTIMENT_REDIS_TTL_SEC", "600"))))
    pipe = redis_client.pipeline(transaction=True)
    pipe.hset(key, mapping=mapping)
    pipe.expire(key, ttl)
    await pipe.execute()
    if payload.get("fear_greed_index") is not None:
        with contextlib.suppress(Exception):
            await redis_client.set(REDIS_KEY_AI_SENTIMENT, str(payload["fear_greed_index"]), ex=ttl * 3)


async def publish_sentiment_status(
    redis_client: Any,
    per_symbol: dict[str, dict[str, Any]],
    *,
    discord_meta: dict[str, Any] | None = None,
) -> None:
    if redis_client is None:
        return
    meta = discord_meta or {}
    cfg = env_source_config(discord_fetch_ok=meta.get("fetch_ok"))
    discord_cfg = cfg["discord"]
    intervals = sentiment_fetch_intervals()
    cache = _fetch_cache
    status = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "fetch_intervals_json": json.dumps(intervals, separators=(",", ":")),
        "collector_tick_sec": str(collector_loop_interval_sec()),
        "source_last_fetch_epoch_json": json.dumps(
            {
                "reddit": cache.reddit_ts,
                "news": cache.news_ts,
                "telegram": cache.telegram_ts,
                "discord": cache.discord_ts,
                "fear_greed": cache.fear_greed_ts,
            },
            separators=(",", ":"),
        ),
        "reddit_active": "yes" if cfg["reddit"]["enabled"] else "no",
        "discord_active": "yes" if discord_cfg.get("read_active") else "no",
        "discord_read_ready": "yes" if discord_cfg.get("read_ready") else "no",
        "discord_bot_configured": "yes" if discord_cfg.get("bot_configured") else "no",
        "discord_channel_configured": "yes" if discord_cfg.get("channel_configured") else "no",
        "discord_webhook_only": "yes" if discord_cfg.get("webhook_only") else "no",
        "discord_fetch_ok": "yes" if meta.get("fetch_ok") else "no",
        "discord_last_error": str(meta.get("error") or ""),
        "discord_channel_source": str(meta.get("channel_source") or ""),
        "discord_ts_utc": str(meta.get("ts_utc") or ""),
        "discord_message_count": str(meta.get("message_count") or 0),
        "telegram_active": "yes" if cfg["telegram"]["enabled"] and cfg["telegram"]["configured"] else "no",
        "news_active": "yes" if cfg["news"]["enabled"] else "no",
        "fear_greed_active": "yes",
        "twitter_disabled": "yes",
        "symbols_json": json.dumps({k: {"active": v.get("sentiment_sources_active"), "missing": v.get("sentiment_sources_missing")} for k, v in per_symbol.items()}),
    }
    ttl = max(120, int(float(os.getenv("SENTIMENT_REDIS_TTL_SEC", "600"))))
    await redis_client.hset(REDIS_SENTIMENT_STATUS_KEY, mapping={k: str(v) for k, v in status.items()})
    await redis_client.expire(REDIS_SENTIMENT_STATUS_KEY, ttl)


async def run_sentiment_collection_pass(symbols: list[str] | None = None, redis_client: Any = None) -> dict[str, dict[str, Any]]:
    syms = list(symbols or TRADING_SYMBOLS)
    out: dict[str, dict[str, Any]] = {}
    cache = _fetch_cache
    intervals = sentiment_fetch_intervals()
    now = time.time()
    async with httpx.AsyncClient(timeout=14.0) as client:
        discord_messages = cache.discord_messages
        discord_meta = dict(cache.discord_meta)
        if _source_fetch_due(cache.discord_ts, intervals["discord"], has_cache=bool(cache.discord_ts)):
            try:
                from backend.services.discord_social_sentiment_live import fetch_recent_channel_messages

                fetch_result = await fetch_recent_channel_messages(http_client=client)
                discord_messages = fetch_result.messages
                discord_meta = {
                    "fetch_ok": fetch_result.fetch_ok,
                    "error": fetch_result.error,
                    "ts_utc": fetch_result.ts_utc,
                    "message_count": fetch_result.message_count,
                    "channel_source": fetch_result.channel_source,
                }
                cache.discord_messages = discord_messages
                cache.discord_meta = discord_meta
                cache.discord_ts = now
            except Exception as exc:
                logger.debug("discord batch fetch failed: %s", exc)
                discord_meta = {
                    "fetch_ok": False,
                    "error": str(exc)[:200],
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "message_count": 0,
                }
                cache.discord_meta = discord_meta
                cache.discord_ts = now

        reddit_corpus = cache.reddit_corpus
        if _source_fetch_due(cache.reddit_ts, intervals["reddit"], has_cache=cache.reddit_corpus is not None):
            try:
                from backend.services.reddit_social_sentiment_live import fetch_reddit_corpus

                reddit_corpus = await fetch_reddit_corpus(http_client=client)
                cache.reddit_corpus = reddit_corpus
                cache.reddit_ts = now
            except Exception as exc:
                logger.debug("reddit batch fetch failed: %s", exc)
                from backend.services.reddit_social_sentiment_live import RedditCorpus

                reddit_corpus = RedditCorpus([], fetch_ok=False, error=str(exc)[:200])
                cache.reddit_corpus = reddit_corpus
                cache.reddit_ts = now

        news_corpus = cache.news_corpus
        if _source_fetch_due(cache.news_ts, intervals["news"], has_cache=cache.news_corpus is not None):
            try:
                from backend.services.news_sentiment import fetch_news_corpus

                news_corpus = await fetch_news_corpus(http_client=client, redis_client=redis_client)
                cache.news_corpus = news_corpus
                cache.news_ts = now
            except Exception as exc:
                logger.debug("news batch fetch failed: %s", exc)
                from backend.services.news_sentiment import NewsCorpus

                news_corpus = NewsCorpus([], fetch_ok=False, error=str(exc)[:200])
                cache.news_corpus = news_corpus
                cache.news_ts = now

        telegram_messages = cache.telegram_messages
        telegram_meta = dict(cache.telegram_meta)
        if _source_fetch_due(cache.telegram_ts, intervals["telegram"], has_cache=bool(cache.telegram_ts)):
            try:
                from backend.services.telegram_social_sentiment_live import refresh_telegram_message_cache

                telegram_messages, telegram_meta = await refresh_telegram_message_cache(
                    http_client=client,
                    redis_client=redis_client,
                )
                cache.telegram_messages = telegram_messages
                cache.telegram_meta = telegram_meta
                cache.telegram_ts = now
            except Exception as exc:
                logger.debug("telegram batch fetch failed: %s", exc)
                telegram_meta = {
                    "fetch_ok": False,
                    "error": str(exc)[:200],
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                }
                cache.telegram_meta = telegram_meta
                cache.telegram_ts = now

        fear_greed_score = cache.fear_greed_score
        fear_greed_err = cache.fear_greed_err
        if _source_fetch_due(cache.fear_greed_ts, intervals["fear_greed"], has_cache=cache.fear_greed_ts > 0):
            fg, fg_err = await _fetch_fear_greed()
            cache.fear_greed_score = fg
            cache.fear_greed_err = fg_err
            cache.fear_greed_ts = now
            fear_greed_score = fg
            fear_greed_err = fg_err

        for sym in syms:
            try:
                payload = await collect_symbol_sentiment(
                    sym,
                    http_client=client,
                    discord_messages=discord_messages,
                    discord_meta=discord_meta,
                    reddit_corpus=reddit_corpus,
                    telegram_messages=telegram_messages,
                    telegram_meta=telegram_meta,
                    news_corpus=news_corpus,
                    fear_greed_score=fear_greed_score,
                    fear_greed_err=fear_greed_err,
                )
                out[sym] = payload
                await publish_symbol_sentiment_to_redis(sym, payload, redis_client)
            except Exception as exc:
                logger.debug("sentiment collect failed %s: %s", sym, exc)
    await publish_sentiment_status(redis_client, out, discord_meta=discord_meta)
    return out


class ActiveSentimentCollector:
    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = list(symbols or TRADING_SYMBOLS)
        self.is_running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop(), name="active_sentiment_collector:loop")
        logger.info("ActiveSentimentCollector started for %d symbols", len(self.symbols))

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        from backend.config.redis_config import get_shared_redis_async

        while self.is_running:
            t0 = time.time()
            try:
                r = await get_shared_redis_async()
                await run_sentiment_collection_pass(self.symbols, r)
            except Exception as exc:
                logger.warning("ActiveSentimentCollector pass failed: %s", exc)
            interval = collector_loop_interval_sec()
            elapsed = time.time() - t0
            await asyncio.sleep(max(5.0, interval - elapsed))


_collector: ActiveSentimentCollector | None = None


def get_active_sentiment_collector() -> ActiveSentimentCollector:
    global _collector
    if _collector is None:
        _collector = ActiveSentimentCollector()
    return _collector


__all__ = [
    "REDIS_SENTIMENT_STATUS_KEY",
    "REDIS_SENTIMENT_SYMBOL_PREFIX",
    "ActiveSentimentCollector",
    "collect_symbol_sentiment",
    "collector_loop_interval_sec",
    "env_source_config",
    "get_active_sentiment_collector",
    "run_sentiment_collection_pass",
    "sentiment_fetch_intervals",
]
