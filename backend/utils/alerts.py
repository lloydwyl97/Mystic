"""
Alerts - Discord + Telegram Live Push

Sends real-time alerts to Discord and Telegram.

Quick test checklist:
- No exchange strings here; EXCHANGE_ID/_to_ccxt_symbol are not needed in this module.
- No unreachable code after returns.
- Logging has no non-ASCII characters.
- No references to streamlit, docker, coinbase, coingecko, kraken, or similar.
- Python 3.12 compatible, Windows PowerShell friendly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("alerts")

# Configuration
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Platform limits (safety truncation)
_DISCORD_MAX = 2000
_TELEGRAM_MAX = 4096


def _truncate(msg: str, limit: int) -> str:
    """Truncate message to platform-safe length."""
    return msg if len(msg) <= limit else (msg[: max(0, limit - 3)] + "...")


async def send_discord(message: str, http_client: httpx.AsyncClient | None = None) -> bool:
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK:
        logger.debug("Discord webhook not configured")
        return False

    try:
        if http_client:
            # Use shared HTTP client
            payload: dict[str, Any] = {
                "content": _truncate(message, _DISCORD_MAX),
                "username": "Mystic Trading Bot",
            }
            response = await http_client.post(DISCORD_WEBHOOK, json=payload)
            # Discord webhooks commonly return 204 on success; accept any 2xx.
            if 200 <= response.status_code < 300:
                logger.debug("Discord message sent")
                return True
            text = response.text
            logger.error("Discord API error: status=%s body=%s", response.status_code, text[:200])
            return False
        # Create our own client
        timeout = httpx.Timeout(10)
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload: dict[str, Any] = {
                "content": _truncate(message, _DISCORD_MAX),
                "username": "Mystic Trading Bot",
            }
            response = await client.post(DISCORD_WEBHOOK, json=payload)
            # Discord webhooks commonly return 204 on success; accept any 2xx.
            if 200 <= response.status_code < 300:
                logger.debug("Discord message sent")
                return True
            text = response.text
            logger.error("Discord API error: status=%s body=%s", response.status_code, text[:200])
            return False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Discord send error: %s", e)
        return False


async def send_telegram(message: str, http_client: httpx.AsyncClient | None = None) -> bool:
    """Send message to Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram bot not configured")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": _truncate(message, _TELEGRAM_MAX),
            # We send plain text; no special formatting required for alerts.
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        if http_client:
            # Use shared HTTP client
            response = await http_client.post(url, data=payload)
            ok = False
            body = ""
            try:
                data = response.json()
                ok = bool(data.get("ok"))
                body = str(data)[:200]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                body = response.text[:200]
            if ok and response.status_code == 200:
                logger.debug("Telegram message sent")
                return True
            logger.error("Telegram API error: status=%s body=%s", response.status_code, body)
            return False
        # Create our own client
        timeout = httpx.Timeout(10)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, data=payload)
            ok = False
            body = ""
            try:
                data = response.json()
                ok = bool(data.get("ok"))
                body = str(data)[:200]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                body = response.text[:200]
            if ok and response.status_code == 200:
                logger.debug("Telegram message sent")
                return True
            logger.error("Telegram API error: status=%s body=%s", response.status_code, body)
            return False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Telegram send error: %s", e)
        return False


async def broadcast_alert(message: str) -> bool:
    """Send alert to all configured channels."""
    try:
        msg = (message or "").strip()
        if not msg:
            logger.warning("Empty alert message, skipping broadcast")
            return False

        discord_sent = await send_discord(msg)
        telegram_sent = await send_telegram(msg)

        if discord_sent or telegram_sent:
            logger.info(
                "Alert broadcast sent (discord=%s, telegram=%s)",
                discord_sent,
                telegram_sent,
            )
            return True

        logger.warning("No alert channels configured or send failed")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Broadcast error: %s", e)
        return False
    else:
        return False


async def send_trade_alert(
    symbol: str,
    action: str,
    price: float,
    amount: float,
    pnl: float | None = None,
) -> None:
    """Send formatted trade alert."""
    try:
        action_u = (action or "").upper()
        total = price * amount
        if action_u == "BUY":
            message = f"BUY ORDER\nSymbol: {symbol}\nPrice: ${price:.4f}\nAmount: {amount:.4f}\nTotal: ${total:.2f}"
        elif action_u == "SELL":
            pnl_text = f"P&L: {pnl:+.2f}%" if pnl is not None else ""
            message = (f"SELL ORDER\nSymbol: {symbol}\nPrice: ${price:.4f}\nAmount: {amount:.4f}\nTotal: ${total:.2f}\n{pnl_text}").strip()
        else:
            message = f"{action_u}\nSymbol: {symbol}\nPrice: ${price:.4f}"

        await broadcast_alert(message)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Trade alert error: %s", e)


async def send_market_alert(alert_type: str, data: dict[str, Any]) -> None:
    """Send formatted market alert."""
    try:
        t = (alert_type or "").upper()
        if t == "BREAKOUT":
            message = f"BREAKOUT DETECTED\nSymbol: {data.get('symbol', 'Unknown')}\nChange: {float(data.get('change', 0)):,.2f}%\nPrice: ${float(data.get('price', 0)):,.4f}"
        elif t == "PUMP":
            message = f"PUMP DETECTED\nSymbol: {data.get('symbol', 'Unknown')}\nVolume: ${float(data.get('volume', 0)):,.0f}\nRank: #{int(data.get('rank', 0))}"
        elif t == "MYSTIC":
            raw = float(data.get("confidence", 0) or 0)
            try:
                from backend.services.confidence_normalizer import ConfidenceNormalizer

                pct = ConfidenceNormalizer.normalize(raw) * 100
            except Exception as ex:
                logger.debug("ConfidenceNormalizer unavailable: %s", ex)
                pct = raw * 100 if raw <= 1 else raw
            message = f"MYSTIC SIGNAL\nMessage: {data.get('message', 'Unknown')!s}\nConfidence: {pct:,.2f}%"
        else:
            # Safe, compact fallback
            preview = str(data)[:300]
            message = f"{t}\nData: {preview}"

        await broadcast_alert(message)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Market alert error: %s", e)


def get_alert_status() -> dict[str, Any]:
    """Get current alert configuration status."""
    discord_ok = bool(DISCORD_WEBHOOK)
    telegram_ok = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    return {
        "discord_configured": discord_ok,
        "telegram_configured": telegram_ok,
        "total_channels": int(discord_ok) + int(telegram_ok),
    }
