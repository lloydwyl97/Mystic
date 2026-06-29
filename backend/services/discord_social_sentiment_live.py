"""Discord bot channel read — real messages only, no fabrication."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DISCORD_API = "https://discord.com/api/v10"
_CHANNEL_ID_RE = re.compile(r"^\d{17,20}$")
_WEBHOOK_URL_RE = re.compile(r"/webhooks/(\d+)/([^/?#\s]+)")
_resolved_channel_cache: dict[str, Any] = {"channel_id": "", "source": "", "ts": 0.0}
_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("btc", "btcusdt", "$btc", "#btc", "bitcoin"),
    "ETH": ("eth", "ethusdt", "$eth", "#eth", "ethereum"),
    "SOL": ("sol", "solusdt", "$sol", "#sol", "solana"),
    "XRP": ("xrp", "xrpusdt", "$xrp", "#xrp", "ripple"),
}


def _base(symbol: str) -> str:
    s = (symbol or "BTC").upper().replace("/USDT", "").replace("USDT", "").strip()
    return s or "BTC"


def get_discord_bot_token() -> str:
    return (os.getenv("DISCORD_BOT_TOKEN") or os.getenv("DISCORD_TOKEN") or "").strip()


def _bot_id_from_token(token: str) -> str:
    part = (token or "").split(".", 1)[0]
    if not part:
        return ""
    try:
        import base64

        pad = part + "=" * (-len(part) % 4)
        return base64.b64decode(pad).decode("utf-8")
    except Exception:
        return ""


def get_discord_channel_id() -> str:
    raw = (os.getenv("DISCORD_CHANNEL_ID") or "").strip()
    if not _CHANNEL_ID_RE.match(raw):
        return ""
    bot_id = _bot_id_from_token(get_discord_bot_token())
    if bot_id and raw == bot_id:
        return ""
    return raw


def get_discord_webhook_url() -> str:
    return (os.getenv("DISCORD_WEBHOOK") or os.getenv("DISCORD_WEBHOOK_URL") or "").strip()


def parse_webhook_url(url: str) -> tuple[str, str] | None:
    m = _WEBHOOK_URL_RE.search(url or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def discord_readiness() -> dict[str, Any]:
    bot = bool(get_discord_bot_token())
    channel_env = bool(get_discord_channel_id())
    webhook = bool(get_discord_webhook_url())
    webhook_parsed = parse_webhook_url(get_discord_webhook_url()) is not None
    channel_target = channel_env or webhook_parsed
    return {
        "bot_configured": bot,
        "channel_configured": channel_target,
        "channel_env_configured": channel_env,
        "channel_webhook_resolvable": webhook_parsed,
        "configured": bot or webhook,
        "read_ready": bot and channel_target,
        "webhook_only": webhook and not bot,
    }


@dataclass
class DiscordFetchResult:
    messages: list[dict[str, Any]]
    message_count: int
    ts_utc: str
    error: str | None = None
    fetch_ok: bool = False
    channel_id: str = ""
    channel_source: str = ""


def _message_text(msg: dict[str, Any]) -> str:
    content = str(msg.get("content") or "")
    parts = [content]
    for emb in msg.get("embeds") or []:
        if not isinstance(emb, dict):
            continue
        parts.append(str(emb.get("title") or ""))
        parts.append(str(emb.get("description") or ""))
    return " ".join(p for p in parts if p).strip()


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
    return pair in t.replace("/", "").replace("-", "")


def score_messages_for_base(messages: list[dict[str, Any]], base: str) -> tuple[float, int, int] | None:
    """Return (score, total_message_count, matched_count) or None if no matches."""
    from backend.services.news_sentiment import score_text

    matched_scores: list[float] = []
    for msg in messages:
        text = _message_text(msg)
        if not message_matches_base(text, base):
            continue
        matched_scores.append(score_text(text))
    if not matched_scores:
        return None
    score = max(-1.0, min(1.0, sum(matched_scores) / len(matched_scores)))
    return score, len(messages), len(matched_scores)


async def _discord_get(
    client: httpx.AsyncClient,
    path: str,
    *,
    token: str,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    headers = {"Authorization": f"Bot {token}"}
    url = f"{_DISCORD_API}{path}"
    resp = await client.get(url, headers=headers, params=params or {}, timeout=15.0)
    if resp.status_code == 429:
        retry_after = float(resp.headers.get("Retry-After") or resp.json().get("retry_after", 1.0))
        await asyncio.sleep(min(max(retry_after, 0.5), 5.0))
        resp = await client.get(url, headers=headers, params=params or {}, timeout=15.0)
    return resp


async def _bot_user_id(client: httpx.AsyncClient, token: str) -> str:
    resp = await _discord_get(client, "/users/@me", token=token)
    if resp.status_code != 200:
        return ""
    return str(resp.json().get("id") or "")


async def _channel_readable(client: httpx.AsyncClient, token: str, channel_id: str) -> tuple[bool, int, str | None]:
    resp = await _discord_get(client, f"/channels/{channel_id}/messages", token=token, params={"limit": 1})
    if resp.status_code == 200:
        return True, 200, None
    if resp.status_code == 403:
        return False, 403, "discord_permission_denied"
    if resp.status_code == 404:
        return False, 404, "discord_channel_not_found"
    if resp.status_code == 401:
        return False, 401, "discord_bot_unauthorized"
    return False, resp.status_code, f"discord_http_{resp.status_code}"


async def _resolve_channel_from_webhook(client: httpx.AsyncClient) -> tuple[str, str | None]:
    parsed = parse_webhook_url(get_discord_webhook_url())
    if not parsed:
        return "", "discord_webhook_unparseable"
    webhook_id, webhook_token = parsed
    try:
        resp = await client.get(f"{_DISCORD_API}/webhooks/{webhook_id}/{webhook_token}", timeout=15.0)
    except Exception as exc:
        return "", str(exc)[:200]
    if resp.status_code != 200:
        return "", f"discord_webhook_lookup_{resp.status_code}"
    channel_id = str(resp.json().get("channel_id") or "")
    if not _CHANNEL_ID_RE.match(channel_id):
        return "", "discord_webhook_no_channel_id"
    return channel_id, None


async def resolve_discord_channel_id(
    client: httpx.AsyncClient,
    token: str,
    *,
    use_cache: bool = True,
) -> tuple[str, str, str | None]:
    """
    Resolve the readable channel id for bot message fetch.
    Returns (channel_id, source, error).
    source: env | webhook | cache | none
    """
    cache_ttl = max(30, int(float(str(os.getenv("DISCORD_CHANNEL_CACHE_SEC", "300")).split()[0] or 300)))
    now = datetime.now(timezone.utc).timestamp()
    if use_cache:
        cached_id = str(_resolved_channel_cache.get("channel_id") or "")
        cached_source = str(_resolved_channel_cache.get("source") or "")
        cached_ts = float(_resolved_channel_cache.get("ts") or 0.0)
        if cached_id and cached_source and (now - cached_ts) < cache_ttl:
            return cached_id, "cache", None

    bot_id = await _bot_user_id(client, token)
    env_channel = get_discord_channel_id()
    if env_channel:
        if bot_id and env_channel == bot_id:
            logger.info("DISCORD_CHANNEL_ID matches bot user id; resolving channel from webhook")
        else:
            ok, _status, err = await _channel_readable(client, token, env_channel)
            if ok:
                _resolved_channel_cache.update({"channel_id": env_channel, "source": "env", "ts": now})
                return env_channel, "env", None
            if err == "discord_permission_denied":
                return "", "env", err

    webhook_channel, webhook_err = await _resolve_channel_from_webhook(client)
    if webhook_channel:
        ok, _status, err = await _channel_readable(client, token, webhook_channel)
        if ok:
            _resolved_channel_cache.update({"channel_id": webhook_channel, "source": "webhook", "ts": now})
            return webhook_channel, "webhook", None
        return "", "webhook", err or webhook_err

    if env_channel and bot_id and env_channel == bot_id:
        return "", "env", webhook_err or "discord_channel_id_is_bot_user_id"
    if env_channel:
        return "", "env", "discord_channel_not_found"
    return "", "none", webhook_err or "discord_channel_missing"


async def fetch_recent_channel_messages(
    *,
    http_client: httpx.AsyncClient | None = None,
) -> DiscordFetchResult:
    """
    Fetch recent messages using DISCORD_BOT_TOKEN.
    Channel resolution order:
      1) DISCORD_CHANNEL_ID when readable
      2) webhook URL channel (when env id missing/wrong/bot user id)
    """
    ts = datetime.now(timezone.utc).isoformat()
    token = get_discord_bot_token()
    if not token:
        return DiscordFetchResult([], 0, ts, error="discord_bot_missing", fetch_ok=False)
    if not discord_readiness().get("read_ready"):
        return DiscordFetchResult([], 0, ts, error="discord_channel_missing", fetch_ok=False)

    limit = max(1, min(100, int(str(os.getenv("DISCORD_MESSAGE_LIMIT", "50")).split()[0] or 50)))
    own = http_client is None
    client = http_client or httpx.AsyncClient(timeout=15.0)
    try:
        channel_id, source, resolve_err = await resolve_discord_channel_id(client, token)
        if not channel_id:
            return DiscordFetchResult([], 0, ts, error=resolve_err or "discord_channel_missing", fetch_ok=False, channel_source=source)

        resp = await _discord_get(
            client,
            f"/channels/{channel_id}/messages",
            token=token,
            params={"limit": limit},
        )
        if resp.status_code == 403:
            return DiscordFetchResult([], 0, ts, error="discord_permission_denied", fetch_ok=False, channel_id=channel_id, channel_source=source)
        if resp.status_code == 404:
            return DiscordFetchResult([], 0, ts, error="discord_channel_not_found", fetch_ok=False, channel_id=channel_id, channel_source=source)
        if resp.status_code == 401:
            return DiscordFetchResult([], 0, ts, error="discord_bot_unauthorized", fetch_ok=False, channel_id=channel_id, channel_source=source)
        if resp.status_code != 200:
            return DiscordFetchResult([], 0, ts, error=f"discord_http_{resp.status_code}", fetch_ok=False, channel_id=channel_id, channel_source=source)
        raw = resp.json()
        if not isinstance(raw, list):
            return DiscordFetchResult([], 0, ts, error="discord_invalid_response", fetch_ok=False, channel_id=channel_id, channel_source=source)
        messages = [m for m in raw if isinstance(m, dict)]
        return DiscordFetchResult(
            messages,
            len(messages),
            ts,
            error=None,
            fetch_ok=True,
            channel_id=channel_id,
            channel_source=source,
        )
    except Exception as exc:
        logger.debug("discord fetch failed: %s", exc)
        return DiscordFetchResult([], 0, ts, error=str(exc)[:200], fetch_ok=False)
    finally:
        if own:
            await client.aclose()


__all__ = [
    "DiscordFetchResult",
    "discord_readiness",
    "fetch_recent_channel_messages",
    "get_discord_bot_token",
    "get_discord_channel_id",
    "get_discord_webhook_url",
    "message_matches_base",
    "parse_webhook_url",
    "resolve_discord_channel_id",
    "score_messages_for_base",
]
