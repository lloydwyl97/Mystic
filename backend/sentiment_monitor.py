"""
Sentiment AI Feed Parser - Real-time Crypto News Sentiment Analysis
Monitors crypto news feeds and analyzes sentiment for trading signals.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVAL = 600
PING_FILE = "./logs/sentiment_monitor.ping"
SENTIMENT_THRESHOLD = 0.3

Path("./logs").mkdir(parents=True, exist_ok=True)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    analyzer = None
    VADER_AVAILABLE = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def simple_sentiment_analysis(text: str) -> float:
    positive_words = [
        "bull",
        "bullish",
        "moon",
        "pump",
        "surge",
        "rally",
        "gain",
        "profit",
    ]
    negative_words = [
        "bear",
        "bearish",
        "crash",
        "dump",
        "drop",
        "fall",
        "loss",
        "sell",
    ]
    t = text.lower()
    pos = sum(1 for w in positive_words if w in t)
    neg = sum(1 for w in negative_words if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def create_ping_file(status: str, sentiment_score: float, headline_count: int) -> None:
    try:
        ping_path = Path(PING_FILE)
        ping_path.parent.mkdir(parents=True, exist_ok=True)
        with ping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": status,
                    "last_update": _now_iso(),
                    "sentiment_score": float(sentiment_score),
                    "headline_count": int(headline_count),
                },
                f,
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Ping file error: {e}")


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
        headlines.extend([str(x.get("title", "")).strip() for x in results if x.get("title")])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Headline fetch error: {e}")
    return headlines


def analyze_sentiment(headlines: list[str]) -> float:
    sentiments: list[float] = []
    for h in headlines:
        if not h or len(h.strip()) <= 10:
            continue
        if VADER_AVAILABLE and analyzer is not None:
            try:
                scores = analyzer.polarity_scores(h)
                sentiments.append(float(scores.get("compound", 0.0)))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                sentiments.append(0.0)
        else:
            try:
                sentiments.append(float(simple_sentiment_analysis(h)))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                sentiments.append(0.0)
    if not sentiments:
        return 0.0
    return sum(sentiments) / len(sentiments)


def main() -> int:
    token = os.environ.get("CRYPTOPANIC_TOKEN", "").strip()
    if not token:
        logger.error("Missing CRYPTOPANIC_TOKEN environment variable")
        create_ping_file("error", 0.0, 0)
        return 1

    logger.info("Sentiment Monitor starting")
    logger.info(f"Interval: {INTERVAL} seconds")

    while True:
        try:
            # Detect if we're already running inside an event loop
            try:
                _ = asyncio.get_running_loop()
                logger.warning("Already in async context, skipping sentiment monitoring")
                break
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                headlines = asyncio.run(fetch_headlines(token))

            if not headlines:
                logger.warning("No headlines fetched")
                create_ping_file("online", 0.0, 0)
            else:
                avg = analyze_sentiment(headlines)
                pos = 0
                neg = 0
                if VADER_AVAILABLE and analyzer is not None:
                    for x in headlines:
                        try:
                            score = float(analyzer.polarity_scores(x).get("compound", 0.0))
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            score = 0.0
                        if score > 0.1:
                            pos += 1
                        elif score < -0.1:
                            neg += 1
                    engine = "vader"
                else:
                    for x in headlines:
                        try:
                            score = float(simple_sentiment_analysis(x))
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            score = 0.0
                        if score > 0.1:
                            pos += 1
                        elif score < -0.1:
                            neg += 1
                    engine = "simple"

                logger.info(f"Sentiment: {avg:.3f} ({len(headlines)} headlines, +{pos}/-{neg})")
                create_ping_file("online", avg, len(headlines))
                log_rec = {
                    "timestamp": _now_iso(),
                    "sentiment_score": round(avg, 6),
                    "headline_count": len(headlines),
                    "positive_count": pos,
                    "negative_count": neg,
                    "engine": engine,
                }
                try:
                    sentiment_log_path = Path("./logs/sentiment_log.jsonl")
                    sentiment_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with sentiment_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(log_rec) + "\n")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Log write error: {e}")
        except KeyboardInterrupt:
            logger.info("Sentiment monitor stopped")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Main loop error: {e}")
            create_ping_file("error", 0.0, 0)
        time.sleep(INTERVAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
