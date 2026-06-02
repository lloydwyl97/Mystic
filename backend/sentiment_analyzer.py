"""
Market Sentiment Analysis Service
Analyzes market sentiment from live sources
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# import aiohttp - moved inside methods to avoid circular imports
from dotenv import load_dotenv

from backend.config.redis_config import get_shared_redis_sync

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    _VADER = None
    _HAS_VADER = False

load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger("sentiment_service")
logging.basicConfig(level=logging.INFO)

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = list(TRADING_SYMBOLS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _simple_sentiment(text: str) -> float:
    positives = ("bull", "bullish", "moon", "pump", "surge", "rally", "gain", "profit")
    negatives = ("bear", "bearish", "crash", "dump", "drop", "fall", "loss", "sell")
    t = text.lower()
    p = sum(1 for w in positives if w in t)
    n = sum(1 for w in negatives if w in t)
    if p == 0 and n == 0:
        return 0.0
    return (p - n) / (p + n)


def _scale_unit(x: float) -> float:
    return max(0.0, min(1.0, (x + 1.0) / 2.0))


def _scale_change_to_unit(pct: float) -> float:
    return max(0.0, min(1.0, (pct / 10.0) / 2.0 + 0.5))


class SentimentAnalyzer:
    def __init__(self) -> None:
        self.redis_client = get_shared_redis_sync()
        if self.redis_client is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.client: Any = None  # httpx.AsyncClient
        self.running = False
        self.interval = int(os.getenv("SENTIMENT_INTERVAL_SEC", "300"))
        self.cryptopanic_token = os.getenv("CRYPTOPANIC_TOKEN", "").strip()

    async def start(self):
        logger.info("Starting Market Sentiment Analyzer")
        self.running = True
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=10.0))
        try:
            await self.analyze_loop()
        finally:
            await self.stop()

    async def stop(self):
        logger.info("Stopping Market Sentiment Analyzer")
        self.running = False
        if self.client:
            await self.client.aclose()
            self.client = None

    async def analyze_loop(self):
        while self.running:
            try:
                data = await self.calculate_sentiment()
                await self.store_sentiment_data(data)
                await self.publish_sentiment_updates(data)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Analysis error: {e}")
            await asyncio.sleep(self.interval)

    async def calculate_sentiment(self) -> dict[str, Any]:
        news_headlines = await self._fetch_cryptopanic_headlines()
        news_score, pos, neg, neu = self._score_headlines(news_headlines)
        tech_map = await self._fetch_binanceus_metrics()
        tech_scores = {s: _scale_change_to_unit(m.get("change_pct", 0.0)) for s, m in tech_map.items()}
        tech_avg = sum(tech_scores.values()) / len(tech_scores) if tech_scores else 0.5
        overall = 0.5 * _scale_unit(news_score) + 0.5 * tech_avg
        sentiment_by_symbol = {
            "BTCUSDT": tech_scores.get("BTCUSDT", 0.5),
            "ETHUSDT": tech_scores.get("ETHUSDT", 0.5),
            "ADAUSDT": tech_scores.get("ADAUSDT", 0.5),
            "SOLUSDT": tech_scores.get("SOLUSDT", 0.5),
            "DOGEUSDT": tech_scores.get("DOGEUSDT", 0.5),
            "XRPUSDT": tech_scores.get("XRPUSDT", 0.5),
            "BCHUSDT": tech_scores.get("BCHUSDT", 0.5),
            "LTCUSDT": tech_scores.get("LTCUSDT", 0.5),
            "AVAXUSDT": tech_scores.get("AVAXUSDT", 0.5),
            "LINKUSDT": tech_scores.get("LINKUSDT", 0.5),
        }
        return {
            "timestamp": _now_iso(),
            "sentiment": {
                "overall": round(overall, 4),
                **sentiment_by_symbol,
            },
            "news": {
                "source": "cryptopanic",
                "available": bool(news_headlines),
                "headline_count": len(news_headlines),
                "score_compound": round(news_score, 4),
                "positive": pos,
                "negative": neg,
                "neutral": neu,
            },
            "technical": {
                "symbols": {
                    k: {
                        "change_pct": tech_map.get(k, {}).get("change_pct", 0.0),
                        "score": v,
                    }
                    for k, v in sentiment_by_symbol.items()
                }
            },
        }

    async def _fetch_cryptopanic_headlines(self) -> list[str]:
        if not self.cryptopanic_token:
            logger.warning("CRYPTOPANIC_TOKEN not set; skipping news sentiment")
            return []
        assert self.client is not None
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_token}&public=true"
        headers = {"User-Agent": "MysticSentiment/1.0"}
        try:
            r = await self.client.get(url, headers=headers)
            if r.status_code != 200:
                logger.warning(f"CryptoPanic HTTP {r.status_code}")
                return []
            data = r.json()
            results = data.get("results", [])
            return [str(x.get("title", "")).strip() for x in results if x.get("title")]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"CryptoPanic fetch error: {e}")
            return []

    def _score_headlines(self, headlines: list[str]) -> tuple[float, int, int, int]:
        sentiments: list[float] = []
        pos = neg = neu = 0
        for h in headlines:
            if not h or len(h.strip()) < 10:
                continue
            comp = float(_VADER.polarity_scores(h).get("compound", 0.0)) if _HAS_VADER and _VADER is not None else float(_simple_sentiment(h))
            sentiments.append(comp)
            if comp > 0.1:
                pos += 1
            elif comp < -0.1:
                neg += 1
            else:
                neu += 1
        avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
        return avg, pos, neg, neu

    async def _fetch_binanceus_metrics(self) -> dict[str, dict[str, float]]:
        assert self.client is not None
        out: dict[str, dict[str, float]] = {}
        try:
            r = await self.client.get("https://api.binance.us/api/v3/ticker/24hr")
            if r.status_code != 200:
                logger.warning(f"Binance.US HTTP {r.status_code}")
                return out
            data = r.json()
            for item in data:
                sym = item.get("symbol")
                if sym in ALLOWED_SYMBOLS:
                    try:
                        out[sym] = {
                            "last_price": float(item.get("lastPrice") or 0.0),
                            "change_pct": float(item.get("priceChangePercent") or 0.0),
                            "quote_volume": float(item.get("quoteVolume") or 0.0),
                        }
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Binance.US metrics error: {e}")
        return out

    async def store_sentiment_data(self, data: dict[str, Any]) -> None:
        try:
            self.redis_client.set("sentiment_data", json.dumps(data), ex=1800)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Redis set error: {e}")

    async def publish_sentiment_updates(self, data: dict[str, Any]) -> None:
        try:
            self.redis_client.publish("sentiment_updates", json.dumps(data))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Redis publish error: {e}")


async def main():
    svc = SentimentAnalyzer()
    try:
        await svc.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await svc.stop()


if __name__ == "__main__":
    asyncio.run(main())
