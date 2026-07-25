"""
Shared VADER sentiment scorer for all social/news text sources (Reddit, News/RSS,
Telegram, Discord all route through this — see news_sentiment.score_text and
reddit_social_sentiment_live._score_title).

Replaces the old naive keyword-counting scorers (crude presence/absence counts with
no negation, intensity, or degree handling — "not bullish" scored identical to
"bullish") with VADER (Valence Aware Dictionary and sEntiment Reasoner), a real
lexicon+rule-based sentiment engine that handles negation ("not going to moon"),
intensifiers ("very bullish" vs "bullish"), degree modifiers, punctuation emphasis
("!!!"), and capitalization emphasis ("MOON") out of the box.

VADER's general-purpose lexicon doesn't know crypto slang, so it's extended once at
import time with hand-tuned valence scores (-4..+4 scale, matching VADER's own
lexicon range) for the terms the old keyword lists covered plus common crypto slang
VADER would otherwise score as neutral (0.0).
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

# VADER lexicon convention: valence scores roughly in [-4, 4]. Values chosen to be
# comparable to VADER's own built-in entries (e.g. "great"=3.1, "terrible"=-3.4).
_CRYPTO_LEXICON: dict[str, float] = {
    # bullish / positive
    "moon": 2.8,
    "mooning": 2.9,
    "moonshot": 2.6,
    "bullish": 2.7,
    "bull": 1.8,
    "pump": 1.8,
    "pumping": 1.9,
    "surge": 2.2,
    "surging": 2.2,
    "rally": 2.2,
    "rallying": 2.2,
    "breakout": 2.0,
    "hodl": 1.5,
    "hodling": 1.5,
    "accumulate": 1.2,
    "accumulating": 1.2,
    "long": 0.8,
    "ath": 2.0,  # all-time high
    "gem": 1.6,
    "undervalued": 1.4,
    "soar": 2.3,
    "soaring": 2.3,
    "gains": 1.8,
    "green": 1.0,
    "buy": 0.8,
    "buying": 0.8,
    "diamond": 1.2,  # "diamond hands"
    "hands": 0.0,  # neutral alone; "diamond hands" phrase handled by "diamond"
    # bearish / negative
    "rekt": -2.8,
    "dump": -2.2,
    "dumping": -2.3,
    "crash": -3.0,
    "crashing": -3.0,
    "bearish": -2.7,
    "bear": -1.6,
    "plunge": -2.6,
    "plunging": -2.6,
    "capitulation": -2.6,
    "capitulate": -2.4,
    "fud": -1.8,
    "scam": -3.2,
    "rugpull": -3.4,
    "rug": -2.2,
    "liquidated": -2.6,
    "liquidation": -2.4,
    "sell": -0.8,
    "selling": -0.9,
    "short": -0.8,
    "shorting": -0.9,
    "red": -1.0,
    "bagholder": -1.8,
    "bagholding": -1.8,
    "decline": -1.6,
    "declining": -1.6,
    "sink": -1.8,
    "sinking": -1.8,
    "slump": -1.8,
    "slumping": -1.8,
    "overvalued": -1.4,
    "bankruptcy": -3.0,
    "insolvent": -2.8,
    "hack": -2.6,
    "hacked": -2.8,
    "exploit": -2.0,
    "exploited": -2.2,
}

_lock = threading.Lock()
_analyzer_instance = None


def _build_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(_CRYPTO_LEXICON)
    return analyzer


def _get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        with _lock:
            if _analyzer_instance is None:
                _analyzer_instance = _build_analyzer()
    return _analyzer_instance


@lru_cache(maxsize=4096)
def _score_cached(text: str) -> float:
    analyzer = _get_analyzer()
    return float(analyzer.polarity_scores(text)["compound"])


def score_crypto_text(text: str) -> float:
    """
    Score arbitrary crypto-related text in [-1, 1] using VADER + crypto lexicon.
    Returns 0.0 (neutral) on empty input or if VADER is unavailable for any reason —
    same safe-default contract the old keyword scorers had.
    """
    t = (text or "").strip()
    if not t:
        return 0.0
    try:
        # Cap length fed to the analyzer/cache — sentiment signal saturates well
        # before typical article/post lengths, and this bounds cache memory.
        return max(-1.0, min(1.0, _score_cached(t[:2000])))
    except Exception as exc:
        logger.debug("crypto_sentiment_vader score failed, defaulting neutral: %s", exc)
        return 0.0


__all__ = ["score_crypto_text"]
