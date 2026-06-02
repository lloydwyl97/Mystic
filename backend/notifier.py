import logging
import os
from typing import Any

import httpx

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe or _to_ccxt_symbol: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)


DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def _http_post_json(url: str, payload: dict[str, Any]) -> bool:
    try:
        resp = httpx.post(
            url,
            json=payload,
            timeout=10,
            headers={"User-Agent": "mystic-notifier/1.0"},
        )
        resp.raise_for_status()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"HTTP POST failed: {e}")
        return False
    else:
        return True


def _http_post_form(url: str, data: dict[str, Any]) -> bool:
    try:
        resp = httpx.post(
            url,
            data=data,
            timeout=10,
            headers={"User-Agent": "mystic-notifier/1.0"},
        )
        resp.raise_for_status()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"HTTP POST failed: {e}")
        return False
    else:
        return True


def send_alert(message: str) -> None:
    if DISCORD_WEBHOOK:
        ok = _http_post_json(DISCORD_WEBHOOK, {"content": message})
        if not ok:
            logger.warning("Discord notification failed")
    else:
        logger.debug("DISCORD_WEBHOOK not configured")

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        ok = _http_post_form(url, payload)
        if not ok:
            logger.warning("Telegram notification failed")
    else:
        logger.debug("Telegram credentials not configured")


def send_trade_alert(symbol: str, action: str, price: float, profit: float) -> None:
    message = f"AI Trade: {action} {symbol} @ ${price:.2f} | Profit: ${profit:.2f}"
    send_alert(message)


def send_performance_alert(avg_profit: float, total_trades: int) -> None:
    message = f"AI Performance: {total_trades} trades | Avg Profit: ${avg_profit:.2f}"
    send_alert(message)
