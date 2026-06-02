"""
Anomaly Guardian - Market Anomaly Detection System
Detects unusual price movements and market behavior using machine learning.
Windows 11 Home + PowerShell compatible.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest  # type: ignore[import-not-found]

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TOP10_COINS,
        TRADING_SYMBOLS,
    )
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe or _to_ccxt_symbol: {e}"
    raise RuntimeError(msg) from e

# Lazy import for optional live market data service (may not be available in all deployments)
try:
    from backend.services.live_market_data import live_market_data_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    live_market_data_service = None  # type: ignore[assignment, misc]


# Configure logging (ASCII only)
Path("./logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s anomaly_guardian: %(message)s",
    handlers=[
        logging.FileHandler("logs/anomaly_guardian.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
CHECK_INTERVAL = 300  # 5 minutes
# Use TRADING_SYMBOLS from trading_universe (live data)
# Convert trading symbols (BTCUSDT) to tuples (BTC, USDT)
SYMBOLS: list[tuple[str, str]] = []
for symbol in TRADING_SYMBOLS:
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        SYMBOLS.append((base, "USDT"))
PING_FILE = "./logs/anomaly_guardian.ping"

# Optional ML dependency
try:
    SKLEARN_AVAILABLE = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    logger.warning("scikit-learn not available, falling back to statistical detection")
    SKLEARN_AVAILABLE = False


def create_ping_file(anomaly_count: int, symbols_checked: int) -> None:
    """Create ping file for dashboard monitoring."""
    try:
        ping_path = Path(PING_FILE)
        ping_path.parent.mkdir(parents=True, exist_ok=True)
        with ping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "anomaly_count": anomaly_count,
                    "symbols_checked": symbols_checked,
                },
                f,
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Ping file write error: {e!s}")


def _normalize_to_df(raw: Any) -> pd.DataFrame:
    """
    Normalize various raw formats to a DataFrame with 'close' column.
    Accepts:
      - DataFrame with 'close' or 'price'
      - list[dict] with keys 'close' | 'c' | 'price'
      - list[list/tuple] where index 4 is close (kline format)
      - list[float] or list[int]
    """
    try:
        if raw is None:
            return pd.DataFrame(columns=["close"])

        # Avoid treating strings/bytes as generic iterables of items
        if isinstance(raw, (str, bytes)):
            return pd.DataFrame(columns=["close"])

        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
            if "close" not in df.columns:
                if "price" in df.columns:
                    df["close"] = df["price"]
                elif "c" in df.columns:
                    df["close"] = df["c"]
            # Ensure only the 'close' column returned
            if "close" in df.columns:
                return df[["close"]].dropna()
            return pd.DataFrame(columns=["close"])

        closes: list[float] = []
        if isinstance(raw, Iterable):
            for item in raw:
                val = None  # type: Optional[float]
                if isinstance(item, dict):
                    v = item.get("close", item.get("c", item.get("price")))
                    if v is not None:
                        try:
                            val = float(v)
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            val = None
                elif isinstance(item, (list, tuple)):
                    if len(item) >= 5:
                        try:
                            val = float(item[4])
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            val = None
                elif isinstance(item, (int, float, np.floating, np.integer)):
                    try:
                        val = float(item)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        val = None
                if val is not None and np.isfinite(val):
                    closes.append(val)

        if closes:
            return pd.DataFrame({"close": closes})
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.debug("Failed to normalize raw data to DataFrame", exc_info=True)
    return pd.DataFrame(columns=["close"])


def get_live_price_data(base: str, quote: str, limit: int = 100) -> pd.DataFrame:
    """
    Fetch live historical price data via the platform's market data service.
    Tries CCXT-style symbol first, then a compact variant if needed.
    Returns a DataFrame with a 'close' column or empty DataFrame on failure.
    """
    symbol_ccxt = _to_ccxt_symbol(base, quote)
    compact = symbol_ccxt.replace("/", "")
    try:
        if live_market_data_service is not None:
            raw = live_market_data_service.get_price_history(symbol_ccxt, limit)  # Prefer BASE/QUOTE
            df = _normalize_to_df(raw)
            if not df.empty:
                return df

            # Fallback: compact variant if service expects it
            raw2 = live_market_data_service.get_price_history(compact, limit)
            return _normalize_to_df(raw2)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Market data fetch error for {symbol_ccxt}: {e!s}")
    return pd.DataFrame(columns=["close"])


def statistical_anomaly_detection(df: pd.DataFrame) -> bool:
    """Simple z-score outlier detection on the latest close."""
    if df.empty or len(df) < 10:
        return False
    prices = df["close"].to_numpy(dtype=float)
    hist = prices[:-1]
    if hist.size == 0:
        return False
    mean_price = float(np.mean(hist))
    std_price = float(np.std(hist))
    if std_price == 0.0:
        return False
    z_score = abs(float(prices[-1]) - mean_price) / std_price
    return z_score > 3.0


def ml_anomaly_detection(df: pd.DataFrame) -> bool:
    """
    IsolationForest detection on basic derived features.
    Returns True if the most recent point is classified as an outlier.
    """
    if not SKLEARN_AVAILABLE:
        return statistical_anomaly_detection(df)
    if df.empty or len(df) < 50:
        return statistical_anomaly_detection(df)

    try:
        closes = df["close"].astype(float)
        returns = closes.pct_change().fillna(0.0)
        vol = returns.rolling(10, min_periods=1).std().fillna(0.0)
        bb_mean = closes.rolling(20, min_periods=1).mean()
        bb_std = closes.rolling(20, min_periods=1).std().replace(0.0, np.nan).bfill().ffill()
        # Avoid division by zero / NaN; result may contain NaN which will be dropped later
        bb_z = (closes - bb_mean) / bb_std.replace(0.0, np.nan)

        feats = pd.DataFrame(
            {
                "ret": returns,
                "vol": vol,
                "bb_z": bb_z.replace([np.inf, -np.inf], 0.0).fillna(0.0),
            }
        ).dropna()

        if len(feats) < 30:
            return statistical_anomaly_detection(df)

        model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        preds = model.fit_predict(feats.to_numpy(dtype=float))
        # preds align with rows of feats; choose last prediction
        return int(preds[-1]) == -1
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"ML anomaly detection error: {e!s}")
        return statistical_anomaly_detection(df)


def main() -> None:
    """Main execution loop."""
    logger.info("Anomaly Guardian started")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    logger.info(f"Monitoring symbols: {[f'{b}/{q}' for b, q in SYMBOLS]}")

    anomaly_count = 0

    while True:
        try:
            symbols_checked = 0
            current_anomalies = 0

            for base, quote in SYMBOLS:
                try:
                    df = get_live_price_data(base, quote, 100)
                    if not df.empty:
                        if ml_anomaly_detection(df):
                            current_anomalies += 1
                        symbols_checked += 1
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Symbol processing error for {base}/{quote}: {e!s}")

            anomaly_count += current_anomalies
            create_ping_file(anomaly_count, symbols_checked)

            if current_anomalies > 0:
                logger.info(f"Detected {current_anomalies} anomaly/anomalies this cycle")
            else:
                logger.info(f"No anomalies detected. Checked {symbols_checked} symbols.")
        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Main loop error: {e!s}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
