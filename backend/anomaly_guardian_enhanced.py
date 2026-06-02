from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from backend.services.binance_rest_client import BinanceREST
from backend.services.confidence_normalizer import ConfidenceNormalizer
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

try:
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import _to_ccxt_symbol from binance_data_fetcher: {e}"
    raise RuntimeError(msg) from e


# Alerts (optional)
try:
    from alerts import send_discord_alert
except (ImportError, ModuleNotFoundError, AttributeError):  # alerts module may not exist in all deployments
    send_discord_alert = None  # type: ignore[assignment]

# Configuration
ANOMALY_DB = os.getenv("ANOMALY_DB", "./data/anomaly_detection.db")
ALERT_THRESHOLD = float(os.getenv("ANOMALY_ALERT_THRESHOLD", "0.8"))
CHECK_INTERVAL = int(os.getenv("ANOMALY_CHECK_INTERVAL", "300"))  # seconds
LOOKBACK_PERIODS = [1, 4, 24]  # hours
# Use TRADING_SYMBOLS from trading_universe (live data)
# Environment variable override if needed, otherwise use trading_universe
env_symbols = os.getenv("ANOMALY_SYMBOLS")
if env_symbols:
    SYMBOLS: list[str] = [s.strip() for s in env_symbols.split(",") if s.strip()]
else:
    SYMBOLS: list[str] = list(TRADING_SYMBOLS)

logger = logging.getLogger(__name__)


# ---------- SQLite Helpers ----------
def _ensure_db_dir(path: str) -> None:
    try:
        d = str(Path(path).resolve().parent)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] failed to ensure DB dir: %s", EXCHANGE_ID, e)


class AnomalyDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        _ensure_db_dir(self.db_path)
        self.init_database()

    def init_database(self) -> None:
        """Initialize anomaly database (idempotent)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS price_anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        anomaly_score REAL NOT NULL,
                        anomaly_type TEXT NOT NULL,
                        price_change REAL NOT NULL,
                        volume_change REAL NOT NULL,
                        timeframe TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                )
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS volume_anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        volume_ratio REAL NOT NULL,
                        avg_volume REAL NOT NULL,
                        current_volume REAL NOT NULL,
                        anomaly_score REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                )
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pattern_anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        price_level REAL NOT NULL,
                        volume_level REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                )
                conn.commit()
            logger.info("[anomaly][%s] database initialized at %s", EXCHANGE_ID, self.db_path)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[anomaly][%s] init_database failed: %s", EXCHANGE_ID, e)

    def save_price_anomaly(self, data: dict[str, Any]) -> None:
        """Persist a price anomaly."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO price_anomalies
                    (timestamp, symbol, anomaly_score, anomaly_type, price_change, volume_change, timeframe, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(data["timestamp"]),
                        str(data["symbol"]),
                        float(data["anomaly_score"]),
                        str(data["anomaly_type"]),
                        float(data["price_change"]),
                        float(data["volume_change"]),
                        str(data["timeframe"]),
                        ConfidenceNormalizer.normalize(float(data["confidence"])),
                    ),
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[anomaly][%s] save_price_anomaly failed: %s", EXCHANGE_ID, e)

    def save_volume_anomaly(self, data: dict[str, Any]) -> None:
        """Persist a volume anomaly."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO volume_anomalies
                    (timestamp, symbol, volume_ratio, avg_volume, current_volume, anomaly_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(data["timestamp"]),
                        str(data["symbol"]),
                        float(data["volume_ratio"]),
                        float(data["avg_volume"]),
                        float(data["current_volume"]),
                        float(data["anomaly_score"]),
                    ),
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[anomaly][%s] save_volume_anomaly failed: %s", EXCHANGE_ID, e)

    def save_pattern_anomaly(self, data: dict[str, Any]) -> None:
        """Persist a pattern anomaly."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO pattern_anomalies
                    (timestamp, symbol, pattern_type, confidence, price_level, volume_level)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(data["timestamp"]),
                        str(data["symbol"]),
                        str(data["pattern_type"]),
                        ConfidenceNormalizer.normalize(float(data["confidence"])),
                        float(data["price_level"]),
                        float(data["volume_level"]),
                    ),
                )
                conn.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[anomaly][%s] save_pattern_anomaly failed: %s", EXCHANGE_ID, e)


# ---------- Live Market Data ----------
def get_historical_data(symbol: str, hours: int) -> pd.DataFrame:
    """
    Get live historical price data via BinanceREST wrapper.
    Uses interval 1h if hours <= 24 else 4h. Limit is min(hours, 1000).
    """
    try:
        interval = "1h" if hours <= 24 else "4h"
        limit = max(50, min(hours, 1000))  # ensure enough bars for indicators

        async def _kl():
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
            return await client.klines(symbol, interval=interval, limit=limit)

        klines = anyio.run(_kl) or []
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] historical fetch error for %s: %s", EXCHANGE_ID, symbol, e)
        return pd.DataFrame()


# ---------- Technical Indicators (minimal safe implementation) ----------
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a minimal set of indicators required by the anomaly detectors.
    Adds columns: price_change, volume_ratio, rsi, bb_width, volatility
    Returns a cleaned DataFrame.
    """
    if df.empty:
        return df
    try:
        df = df.copy()
        # price change: percent change vs previous close
        df["price_change"] = df["close"].pct_change().fillna(0.0)

        # rolling average volume and ratio
        df["avg_volume_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
        # avoid division by zero
        df["volume_ratio"] = df.apply(
            lambda row: float(row["volume"]) / float(row["avg_volume_20"]) if row["avg_volume_20"] > 0 else 0.0,
            axis=1,
        )

        # RSI (14)
        delta = df["close"].diff()
        up = delta.clip(lower=0.0)
        down = -1.0 * delta.clip(upper=0.0)
        roll_up = up.rolling(window=14, min_periods=14).mean()
        roll_down = down.rolling(window=14, min_periods=14).mean()
        rs = roll_up / roll_down.replace(0, pd.NA)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["rsi"] = df["rsi"].fillna(50.0)

        # Bollinger Band width (20, 2)
        ma = df["close"].rolling(window=20, min_periods=20).mean()
        std = df["close"].rolling(window=20, min_periods=20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        df["bb_width"] = ((upper - lower) / ma).replace([pd.NA, float("inf")], 0.0).fillna(0.0)

        # volatility: rolling std of returns (20)
        df["volatility"] = df["close"].pct_change().rolling(window=20, min_periods=5).std().fillna(0.0)

        # Drop helper column
        df = df.drop(columns=["avg_volume_20"], errors="ignore")

        # Keep relevant columns
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "price_change", "volume_ratio", "rsi", "bb_width", "volatility"]
        return df[keep_cols].dropna()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] technical indicator calc error: %s", EXCHANGE_ID, e)
        return pd.DataFrame()


def detect_price_anomalies(df: pd.DataFrame, symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """Detect price anomalies using Isolation Forest on engineered features."""
    anomalies: list[dict[str, Any]] = []
    if df.empty or len(df) < 60:
        return anomalies
    try:
        feats = ["price_change", "volume_ratio", "rsi", "bb_width", "volatility"]
        feat_df = df[feats].dropna()
        if len(feat_df) < 40:
            return anomalies

        scaler = StandardScaler()
        X = scaler.fit_transform(feat_df)

        iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
        iso.fit(X)

        # Evaluate latest point only for alerting (we store only the latest decision)
        latest_vec = X[-1:].reshape(1, -1)
        latest_raw = iso.decision_function(latest_vec)[0]  # higher is more normal
        latest_row = feat_df.iloc[-1]
        is_anom = iso.predict(latest_vec)[0] == -1

        if is_anom:
            anomaly_type = "price_spike" if float(latest_row["price_change"] or 0) > 0.05 else "price_crash"
            score = float(-latest_raw)  # invert so bigger = more anomalous (>=0)
            anomalies.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "anomaly_score": score,
                    "anomaly_type": anomaly_type,
                    "price_change": float(latest_row["price_change"]),
                    "volume_change": float(latest_row["volume_ratio"]),
                    "timeframe": timeframe,
                    "confidence": float(min(score * 2.0, 1.0)),
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] price anomaly error %s: %s", EXCHANGE_ID, symbol, e)
        return anomalies
    else:
        return anomalies


def detect_volume_anomalies(df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
    """Detect large volume spikes versus rolling average."""
    out: list[dict[str, Any]] = []
    if df.empty or len(df) < 25:
        return out
    try:
        current_vol = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].rolling(window=20, min_periods=20).mean().iloc[-1])
        if avg_vol <= 0:
            return out
        ratio = current_vol / avg_vol
        if ratio > 3.0:
            out.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "volume_ratio": ratio,
                    "avg_volume": avg_vol,
                    "current_volume": current_vol,
                    "anomaly_score": float(min(ratio / 5.0, 1.0)),
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] volume anomaly error %s: %s", EXCHANGE_ID, symbol, e)
        return out
    else:
        return out


def detect_pattern_anomalies(df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
    """Simple pattern signals: double top, support/resistance breaks."""
    out: list[dict[str, Any]] = []
    if df.empty or len(df) < 60:
        return out
    try:
        highs = df["high"].rolling(window=5, min_periods=5).max()
        current_price = float(df["close"].iloc[-1])
        support = float(df["low"].tail(20).min())
        resistance = float(df["high"].tail(20).max())

        # Double top heuristic: at least two recent highs in top 20% of recent range
        if len(highs.dropna()) >= 20:
            recent_highs = highs.tail(20)
            threshold = float(recent_highs.quantile(0.8))
            if (recent_highs > threshold).sum() >= 2:
                out.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "pattern_type": "double_top",
                        "confidence": 0.7,
                        "price_level": current_price,
                        "volume_level": float(df["volume"].iloc[-1]),
                    },
                )

        # Support / resistance breaks
        if support > 0 and current_price < support * 0.98:
            out.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "pattern_type": "support_break",
                    "confidence": 0.8,
                    "price_level": current_price,
                    "volume_level": float(df["volume"].iloc[-1]),
                },
            )
        elif resistance > 0 and current_price > resistance * 1.02:
            out.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "pattern_type": "resistance_break",
                    "confidence": 0.8,
                    "price_level": current_price,
                    "volume_level": float(df["volume"].iloc[-1]),
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] pattern anomaly error %s: %s", EXCHANGE_ID, symbol, e)
        return out
    else:
        return out


# ---------- Orchestration ----------
def _maybe_send_alert(anomaly: dict[str, Any]) -> None:
    """Dispatch alert when severity passes threshold."""
    try:
        if send_discord_alert is None:
            return
        # determine score key (canonical 0-1 for threshold comparison)
        raw = float(anomaly.get("confidence", anomaly.get("anomaly_score", 0.0)) or 0.0)
        score = ConfidenceNormalizer.normalize(raw)
        if score < ALERT_THRESHOLD:
            return

        title = "Anomaly Detected"
        desc = f"{anomaly.get('symbol', 'UNKNOWN')} - {anomaly.get('anomaly_type') or anomaly.get('pattern_type')}"
        color = 0xD32F2F
        fields = [
            {
                "name": "Timeframe",
                "value": str(anomaly.get("timeframe", "n/a")),
                "inline": True,
            },
            {"name": "Score", "value": f"{score:.3f}", "inline": True},
        ]
        if "price_change" in anomaly:
            fields.append(
                {
                    "name": "Price Change",
                    "value": f"{float(anomaly.get('price_change', 0.0)):.4f}",
                    "inline": True,
                }
            )
        if "volume_change" in anomaly:
            fields.append(
                {
                    "name": "Volume Ratio",
                    "value": f"{float(anomaly.get('volume_change', 0.0)):.3f}",
                    "inline": True,
                }
            )
        if "pattern_type" in anomaly:
            fields.append(
                {
                    "name": "Pattern",
                    "value": str(anomaly.get("pattern_type")),
                    "inline": True,
                }
            )

        send_discord_alert(
            f"ANOMALY ALERT\n{desc}",
            {
                "title": title,
                "description": desc,
                "color": color,
                "fields": fields,
            },
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] alert dispatch error: %s", EXCHANGE_ID, e)


def monitor_anomalies_enhanced() -> list[dict[str, Any]]:
    """Run anomaly scan across symbols and lookback windows; persist + alert."""
    db = AnomalyDatabase(ANOMALY_DB)
    collected: list[dict[str, Any]] = []

    for symbol in SYMBOLS:
        for hours in LOOKBACK_PERIODS:
            df = get_historical_data(symbol, hours)
            if df.empty:
                continue

            df = calculate_technical_indicators(df)
            if df.empty:
                continue

            timeframe = "1h" if hours <= 24 else "4h"

            # Price anomalies
            pa = detect_price_anomalies(df, symbol, timeframe)
            for a in pa:
                db.save_price_anomaly(a)
                collected.append(a)
                _maybe_send_alert(a)

            # Volume anomalies
            va = detect_volume_anomalies(df, symbol)
            for a in va:
                db.save_volume_anomaly(a)
                collected.append(a)
                _maybe_send_alert(
                    {
                        **a,
                        "anomaly_type": "volume_spike",
                        "timeframe": timeframe,
                        "confidence": float(a.get("anomaly_score", 0.0)),
                    },
                )

            # Pattern anomalies
            pta = detect_pattern_anomalies(df, symbol)
            for a in pta:
                db.save_pattern_anomaly(a)
                collected.append(a)
                _maybe_send_alert(
                    {
                        **a,
                        "timeframe": timeframe,
                    },
                )

    logger.info("[anomaly][%s] scan complete: %d anomalies", EXCHANGE_ID, len(collected))
    return collected


def main() -> None:
    """Main entry point for anomaly guardian"""
    run_once_env = os.getenv("ANOMALY_RUN_ONCE", "true").strip().lower()
    run_once = run_once_env in ("1", "true", "yes")

    logger.info("[anomaly][%s] starting monitor (run_once=%s)", EXCHANGE_ID, run_once)
    try:
        if run_once:
            monitor_anomalies_enhanced()
        else:
            # Run periodically until interrupted
            while True:
                monitor_anomalies_enhanced()
                time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("[anomaly][%s] interrupted by user", EXCHANGE_ID)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[anomaly][%s] runtime error in main: %s", EXCHANGE_ID, e)


if __name__ == "__main__":
    main()
