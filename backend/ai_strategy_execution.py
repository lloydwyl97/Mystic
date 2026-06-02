import contextlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx
from dotenv import load_dotenv  # type: ignore[import-untyped]

from backend.config import settings
from backend.config.redis_config import get_shared_redis_sync
from backend.services.binance_rest_client import BinanceREST
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = tuple(TRADING_SYMBOLS)

load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))

if not Path("logs").is_dir():
    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_strategy_execution.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BINANCE_KEY = settings.exchange.binance_us_api_key
BINANCE_SECRET = settings.exchange.binance_us_secret_key
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def _validate_symbol(symbol: str) -> str:
    s = str(symbol).upper()
    if s not in ALLOWED_SYMBOLS:
        msg = f"symbol not allowed: {s}"
        raise ValueError(msg)
    return s


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def send_alert(msg: str) -> None:
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                    timeout=10,
                )
        if DISCORD_WEBHOOK_URL and DISCORD_WEBHOOK_URL != "your_discord_webhook_url_here":
            async with httpx.AsyncClient() as client:
                await client.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
        logger.info(f"[{EXCHANGE_ID}] alert sent")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[{EXCHANGE_ID}] alert failed: {e}")


async def get_binance_price(symbol: str | None = None) -> float | None:
    # All Live Data, No Fallback/Hardcoded Data
    try:
        if not symbol:
            if not TRADING_SYMBOLS:
                logger.error(f"[{EXCHANGE_ID}] No trading symbols available - symbol required")
                return None
            symbol = TRADING_SYMBOLS[0]
        s = _validate_symbol(symbol)
        try:
            r = get_shared_redis_sync()
            if r:
                raw = r.hget(f"price:{s}", "v")
                if raw:
                    try:
                        return float(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
        except Exception as e:
            logger.info(f"[{EXCHANGE_ID}] redis price cache unavailable: {e}")

        limiter = await BinanceWeightLimiter.create()
        client = BinanceREST(limiter)

        data = await client.price(s)
        if data and data.get("price") is not None:
            try:
                price = float(data["price"])
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                price = None
        else:
            price = None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[{EXCHANGE_ID}] get price failed for {symbol}: {e}")
        return None
    else:
        return price


async def binance_market_buy(symbol: str, quoteOrderQty: float = 50) -> dict[str, Any]:
    # All Live Data, No Fallback/Hardcoded Data - symbol is required
    try:
        s = _validate_symbol(symbol)
        q = float(quoteOrderQty)
        limiter = await BinanceWeightLimiter.create()
        client = BinanceREST(limiter)

        result = await client.order_market(s, "BUY", quoteOrderQty=str(q))
        if result is None:
            msg = f"[{EXCHANGE_ID}] order_market returned no result for {s}"
            await send_alert(msg)
            logger.error(msg)
            return {"error": "order_market failed"}
        await send_alert(f"[{EXCHANGE_ID}] buy executed: {s} | ${q} | {result}")
        logger.info(f"[{EXCHANGE_ID}] buy executed: {s} ${q}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"[{EXCHANGE_ID}] buy error: {e}"
        try:
            await send_alert(msg)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("failed to send alert for buy error")
        logger.exception(msg)
        return {"error": str(e)}
    else:
        return result


def execute_ai_strategy_signal(symbol_binance: str, usd_amount: float, signal: bool) -> dict[str, Any] | None:
    try:
        if os.getenv("AI_CANONICAL_EXECUTION_ONLY", "true").strip().lower() in ("1", "true", "yes", "on"):
            logger.warning("[%s] blocked direct strategy execution for %s", EXCHANGE_ID, symbol_binance)
            return {"error": "CANONICAL_PATH_REQUIRED"}
        if not signal:
            logger.info(f"[{EXCHANGE_ID}] no signal, skipping")
            return None
        s = _validate_symbol(symbol_binance)
        logger.info(f"[{EXCHANGE_ID}] executing signal: {s} ${usd_amount}")
        price = anyio.run(get_binance_price, s)
        if price is None:
            logger.error(f"[{EXCHANGE_ID}] price unavailable")
            return None
        logger.info(f"[{EXCHANGE_ID}] using price ${price}")
        return anyio.run(binance_market_buy, s, usd_amount)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"[{EXCHANGE_ID}] strategy execution error: {e}"
        try:
            anyio.run(send_alert, msg)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("failed to send alert for strategy execution error")
        logger.exception(msg)
        return {"error": str(e)}


def test_connections() -> None:
    logger.info(f"[{EXCHANGE_ID}] testing connections")
    try:
        # Use first symbol from trading_universe for test (live data)
        test_symbol = TRADING_SYMBOLS[0] if TRADING_SYMBOLS else None
        if not test_symbol:
            logger.error(f"[{EXCHANGE_ID}] No trading symbols available for connection test")
            return
        p = anyio.run(get_binance_price, test_symbol)
        if p:
            logger.info(f"[{EXCHANGE_ID}] connection ok - {test_symbol} price ${p}")
        else:
            logger.error(f"[{EXCHANGE_ID}] connection failed")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[{EXCHANGE_ID}] connection error: {e}")
    try:
        anyio.run(send_alert, f"[{EXCHANGE_ID}] connection test {_now_iso()}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("failed to send connection test alert")


if __name__ == "__main__":
    test_connections()
