"""Telegram bot channel read with Redis-backed message cache — real messages only."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

REDIS_OFFSET_KEY = "telegram_sentiment:update_offset"
REDIS_CACHE_KEY = "telegram_sentiment:message_cache"
_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "btcusdt", "$btc", "#btc", "bitcoin"),
    "ETH": ("eth", "ethusdt", "$eth", "#eth", "ethereum"),
    "SOL": ("sol", "solusdt", "$sol", "#sol", "solana"),
    "XRP": ("xrp", "xrpusdt", "$xrp", "#xrp", "ripple"),
}


def _truthy(v: str | None, default: bool = True) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _base(symbol: str) -> str:
    s = (symbol or "BTC").upper().replace("/USDT", "").replace("USDT", "").strip()
    return s or "BTC"


def _configured_chat() -> str:
    return (os.getenv("TELEGRAM_CHAT_ID") or "").strip()


def _bot_token() -> str:
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def telegram_readiness() -> dict[str, Any]:
    token = bool(_bot_token())
    chat = bool(_configured_chat())
    enabled = _truthy(os.getenv("ENABLE_TELEGRAM"), True)
    return {
        "enabled": enabled and token and chat,
        "bot_configured": token,
        "chat_configured": chat,
    }


def _chat_matches(chat_obj: dict[str, Any], configured: str) -> bool:
    cfg = configured.strip().lstrip("@").lower()
    if not cfg:
        return False
    chat_id = str(chat_obj.get("id") or "")
    username = str(chat_obj.get("username") or "").lower()
    title = str(chat_obj.get("title") or "").lower()
    if cfg.lstrip("-").isdigit():
        return chat_id == configured.strip() or chat_id == cfg
    return cfg in (username, title, chat_id.lower())


def _message_text(msg: dict[str, Any]) -> str:
    return str(msg.get("text") or msg.get("caption") or "").strip()


def message_matches_base(text: str, base: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    aliases = _SYMBOL_ALIASES.get(base.upper(), (base.lower(), f"{base.lower()}usdt"))
    for alias in aliases:
        if alias.startswith("$") or alias.startswith("#"):
            if alias in t:
                return True
            continue
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return True
    pair = f"{base.lower()}usdt"
    compact = t.replace("/", "").replace("-", "").replace(" ", "")
    if pair in compact:
        return True
    return f"{base.lower()}/usdt" in t


def score_cached_messages_for_base(messages: list[dict[str, Any]], base: str) -> tuple[float, int, int] | None:
    from backend.services.news_sentiment import score_text

    matched: list[float] = []
    for msg in messages:
        text = str(msg.get("text") or "")
        if not message_matches_base(text, base):
            continue
        matched.append(score_text(text))
    if not matched:
        return None
    score = max(-1.0, min(1.0, sum(matched) / len(matched)))
    return score, len(messages), len(matched)


async def _load_cache(redis_client: Any) -> list[dict[str, Any]]:
    if redis_client is None:
        return []
    try:
        raw = await redis_client.get(REDIS_CACHE_KEY)
        if not raw:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
        return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []
    except Exception:
        return []


async def _save_cache(redis_client: Any, messages: list[dict[str, Any]]) -> None:
    if redis_client is None:
        return
    ttl = max(300, int(str(os.getenv("TELEGRAM_CACHE_TTL_SEC", "3600")).split()[0] or 3600))
    cap = max(50, int(str(os.getenv("TELEGRAM_CACHE_MAX_MESSAGES", "200")).split()[0] or 200))
    trimmed = messages[-cap:]
    await redis_client.set(REDIS_CACHE_KEY, json.dumps(trimmed, separators=(",", ":")), ex=ttl)


async def refresh_telegram_message_cache(
    *,
    http_client: httpx.AsyncClient,
    redis_client: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Poll Telegram getUpdates with persisted offset, append channel/group messages to Redis cache.
    Returns (cached_messages, meta).
    """
    ts = datetime.now(timezone.utc).isoformat()
    meta: dict[str, Any] = {"ts_utc": ts, "fetch_ok": False, "error": None, "polled_updates": 0, "cache_size": 0}
    token = _bot_token()
    chat = _configured_chat()
    if not token or not chat or not _truthy(os.getenv("ENABLE_TELEGRAM"), True):
        meta["error"] = "telegram_disabled_or_unconfigured"
        return [], meta

    cache = await _load_cache(redis_client)
    offset = 0
    if redis_client is not None:
        with __import__("contextlib").suppress(Exception):
            raw_off = await redis_client.get(REDIS_OFFSET_KEY)
            if raw_off:
                offset = int(raw_off.decode() if isinstance(raw_off, bytes) else raw_off)

    try:
        polled = 0
        for _ in range(3):
            resp = await http_client.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"limit": 100, "timeout": 0, "offset": offset},
                timeout=15.0,
            )
            if resp.status_code != 200:
                meta["error"] = f"telegram_http_{resp.status_code}"
                break
            data = resp.json()
            if not data.get("ok"):
                meta["error"] = "telegram_api_not_ok"
                break
            updates = data.get("result") or []
            if not updates:
                meta["fetch_ok"] = True
                break
            for upd in updates:
                if not isinstance(upd, dict):
                    continue
                upd_id = int(upd.get("update_id") or 0)
                offset = max(offset, upd_id + 1)
                polled += 1
                msg = upd.get("message") or upd.get("channel_post") or upd.get("edited_message") or upd.get("edited_channel_post") or {}
                if not isinstance(msg, dict):
                    continue
                chat_obj = msg.get("chat") or {}
                if not _chat_matches(chat_obj, chat):
                    continue
                text = _message_text(msg)
                if not text:
                    continue
                cache.append(
                    {
                        "text": text,
                        "chat_id": str(chat_obj.get("id") or ""),
                        "ts_utc": datetime.fromtimestamp(int(msg.get("date") or 0), tz=timezone.utc).isoformat() if msg.get("date") else ts,
                    }
                )
            meta["fetch_ok"] = True
            if len(updates) < 100:
                break

        meta["polled_updates"] = polled
        await _save_cache(redis_client, cache)
        if redis_client is not None and offset > 0:
            with __import__("contextlib").suppress(Exception):
                await redis_client.set(REDIS_OFFSET_KEY, str(offset))
        meta["cache_size"] = len(cache)
        if meta["fetch_ok"] and not cache:
            meta["error"] = "telegram_no_channel_posts"
            meta["match_status"] = "no_match"
        elif meta["fetch_ok"]:
            meta["match_status"] = "corpus_available"
        return cache, meta
    except Exception as exc:
        logger.debug("telegram cache refresh failed: %s", exc)
        meta["error"] = str(exc)[:200]
        return cache, meta


__all__ = [
    "REDIS_CACHE_KEY",
    "REDIS_OFFSET_KEY",
    "message_matches_base",
    "refresh_telegram_message_cache",
    "score_cached_messages_for_base",
    "telegram_readiness",
]
