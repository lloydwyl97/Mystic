"""Source-of-truth timeframe consumer matrix.

Completed-candle features use completed candles only. Trailing Buy uses the
fresh executable book, not a candle close. 4H is telemetry-only.
"""

from __future__ import annotations

from typing import Final

from backend.config.canonical_candle_intervals import TELEMETRY_ONLY_NO_TRADE_AUTHORITY

TIMEFRAME_CONSUMER_MATRIX: Final[dict[str, dict[str, str]]] = {
    "trailing_buy_executable_bid_ask": {
        "timeframe": "none",
        "source": "order book / bookTicker / decision_book_tape",
        "completed_only": "n/a",
        "notes": "Fresh executable bid/ask. No candle substitution.",
    },
    "ranking": {
        "timeframe": "BuyCandidate.rank_score: confidence + buy-margin + non-4H thesis_rank_delta",
        "source": "ranked-candidate stream → trailing buy; 4H bundle fields are schema/telemetry only",
        "completed_only": "yes for candle-shape features",
        "notes": "rank_score does not read 4h. 4H context dims stay in the artifact vector for compatibility and have no live rank/order authority.",
    },
    "candle_shape_body_wick": {
        "timeframe": "15m",
        "source": "canonical 15m completed candles",
        "completed_only": "yes",
        "notes": "Deployed 1m→15m candle-shape substitution is intentional for DAY shape features, not a 1m/3m/5m repair.",
    },
    "volume_features": {
        "timeframe": "native TF of the feature (1m primary, HTF from bundle)",
        "source": "canonical OHLCV volume (zero-volume bars kept)",
        "completed_only": "yes",
        "notes": "Missing volume is not coerced to zero.",
    },
    "volatility": {
        "timeframe": "1m ATR / HTF ATR from bundle",
        "source": "completed canonical candles",
        "completed_only": "yes",
        "notes": "",
    },
    "slope_trend": {
        "timeframe": "bundle TFs (1h/1d/…)",
        "source": "completed canonical candles",
        "completed_only": "yes",
        "notes": "No silent TF substitution.",
    },
    "market_context": {
        "timeframe": "DAY bundle + 24h ticker",
        "source": "ai_context hashes + canonical candles",
        "completed_only": "yes for candle fields",
        "notes": "",
    },
    "learning": {
        "timeframe": "production-realized labels + completed candles",
        "source": "trade_learning_outcomes / feature_ohlcv aligned",
        "completed_only": "yes",
        "notes": "day_research_klines is retired.",
    },
    "dashboard_charts": {
        "timeframe": "every CANONICAL_CANDLE_INTERVALS value",
        "source": "GET /api/market/candles",
        "completed_only": "completed + explicit forming overlay",
        "notes": "Empty successful responses are forbidden for supported TFs.",
    },
    "telemetry_only_4h": {
        "timeframe": "4h",
        "source": "canonical 4h store",
        "completed_only": "yes",
        "notes": TELEMETRY_ONLY_NO_TRADE_AUTHORITY,
    },
}

__all__ = ["TIMEFRAME_CONSUMER_MATRIX"]
