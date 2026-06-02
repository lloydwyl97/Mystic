import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    _VADER = None
    _HAS_VADER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_monitor")

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

INTERVAL = 600
SENTIMENT_DB = os.getenv("SENTIMENT_DB", "./data/sentiment_history.db")
ALERT_THRESHOLD = 0.5
# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = set(TRADING_SYMBOLS)

Path("./data").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)


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


class SentimentDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    headline_count INTEGER NOT NULL,
                    positive_count INTEGER NOT NULL,
                    negative_count INTEGER NOT NULL,
                    neutral_count INTEGER NOT NULL,
                    market_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()

    def save_sentiment(self, data: dict) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sentiment_data
                (timestamp, source, sentiment_score, headline_count, positive_count, negative_count, neutral_count, market_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["timestamp"],
                    data["source"],
                    float(data["sentiment_score"]),
                    int(data["headline_count"]),
                    int(data["positive_count"]),
                    int(data["negative_count"]),
                    int(data["neutral_count"]),
                    json.dumps(data.get("market_data", {})),
                ),
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()

    def save_alert(self, alert_type: str, sentiment_score: float, message: str) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sentiment_alerts (timestamp, alert_type, sentiment_score, message)
                VALUES (?, ?, ?, ?)
                """,
                (_now_iso(), alert_type, float(sentiment_score), message),
            )
            conn.commit()
        finally:
            if conn is not None:
                conn.close()


async def fetch_headlines(token: str) -> list[str]:
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={token}&public=true"
    headers = {"User-Agent": "MysticSentiment/1.0"}
    headlines: list[str] = []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        headlines = [str(x.get("title", "")).strip() for x in results if x.get("title")]
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Headline fetch error")
    return headlines


async def get_market_data() -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=10)) as client:
            r = await client.get("https://api.binance.us/api/v3/ticker/24hr")
            r.raise_for_status()
            data = r.json()
            for item in data:
                sym = item.get("symbol")
                if sym in ALLOWED_SYMBOLS:
                    out[sym] = {
                        "price": float(item.get("lastPrice") or 0.0),
                        "change": float(item.get("priceChangePercent") or 0.0),
                        "volume": float(item.get("quoteVolume") or 0.0),
                    }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Market data fetch error")
    return out


def score_headlines(headlines: list[str]) -> tuple[float, int, int, int]:
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


async def analyze_once(db: SentimentDatabase, token: str) -> None:
    headlines = await fetch_headlines(token)
    avg, pos, neg, neu = score_headlines(headlines)
    market = await get_market_data()
    rec = {
        "timestamp": _now_iso(),
        "source": "cryptopanic",
        "sentiment_score": avg,
        "headline_count": len(headlines),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "market_data": market,
    }
    db.save_sentiment(rec)
    if abs(avg) > ALERT_THRESHOLD:
        a_type = "extreme_positive" if avg > 0 else "extreme_negative"
        msg = f"Extreme sentiment {avg:.3f} across {len(headlines)} headlines"
        db.save_alert(a_type, avg, msg)
        logger.warning(msg)
    # Use first symbol from TRADING_SYMBOLS (live data)
    first_symbol = TRADING_SYMBOLS[0] if TRADING_SYMBOLS else None
    if not first_symbol:
        logger.warning("No trading symbols available - using default")
        btc_change = 0.0
    else:
        btc_change = market.get(first_symbol, {}).get("change", 0.0)
    corr = "positive" if (avg > 0 and btc_change > 0) or (avg < 0 and btc_change < 0) else "negative"
    logger.info(f"Sentiment {avg:.3f} ({len(headlines)} headlines) | BTC {btc_change:+.2f}% | Corr: {corr}")


async def main() -> None:
    token = os.environ.get("CRYPTOPANIC_TOKEN", "").strip()
    if not token:
        logger.error("Missing CRYPTOPANIC_TOKEN")
        sys.exit(1)
    db = SentimentDatabase(SENTIMENT_DB)
    while True:
        try:
            await analyze_once(db, token)
        except KeyboardInterrupt:
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Loop error")
        await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
