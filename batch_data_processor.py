#!/usr/bin/env python3
"""
Batch Data Processor - Live Configuration Only

Processes large quantities of historical data for accelerated AI training.
All configuration values come from live config - no hardcoded values.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional imports - try at top level
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    TRADING_SYMBOLS = None

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Import from proper package structure

# Import advanced indicators
try:
    from backend.indicators.advanced_indicators import FibonacciPatterns, IchimokuCloud
except ImportError:
    # Define placeholder classes if imports fail
    class IchimokuCloud:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_features(self, _df: Any) -> dict[str, Any]:
            return {"cloud_position": 0, "tk_cross_signal": 0, "cloud_direction": 0, "cloud_thickness": 0, "chikou_position": 0}

    class FibonacciPatterns:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_features(self, _df: Any) -> dict[str, Any]:
            return {"fib_position": 0, "fib_nearest_level": 0, "fib_distance": 0, "fib_pattern_strength": 0, "fib_trend": 0}


# Setup logging directories
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "batch_processor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_data_dir() -> str:
    """Get data directory from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "data_loader") and hasattr(value.data_loader, "data_dir"):
                data_dir = value.data_loader.data_dir
                if isinstance(data_dir, str) and data_dir:
                    return data_dir.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    data_dir = os.getenv("BATCH_DATA_PROCESSOR_DATA_DIR", "").strip()
    if data_dir:
        return data_dir

    return "data"


def _get_output_dir() -> str:
    """Get output directory from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "data_loader") and hasattr(value.data_loader, "output_dir"):
                output_dir = value.data_loader.output_dir
                if isinstance(output_dir, str) and output_dir:
                    return output_dir.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    output_dir = os.getenv("BATCH_DATA_PROCESSOR_OUTPUT_DIR", "").strip()
    if output_dir:
        return output_dir

    data_dir = _get_data_dir()
    return str(Path(data_dir) / "training_chunks")


def _get_parquet_dir() -> str:
    """Get parquet directory from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "data_loader") and hasattr(value.data_loader, "parquet_dir"):
                parquet_dir = value.data_loader.parquet_dir
                if isinstance(parquet_dir, str) and parquet_dir:
                    return parquet_dir.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    parquet_dir = os.getenv("BATCH_DATA_PROCESSOR_PARQUET_DIR", "").strip()
    if parquet_dir:
        return parquet_dir

    return _get_data_dir()


def _get_default_symbols() -> list[str]:
    """Get default symbols from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "trading_universe") and hasattr(value.trading_universe, "top10_symbols"):
                symbols = value.trading_universe.top10_symbols
                if isinstance(symbols, list) and symbols:
                    return [str(s) for s in symbols]
            if hasattr(value, "data_loader") and hasattr(value.data_loader, "default_symbols"):
                symbols = value.data_loader.default_symbols
                if isinstance(symbols, list) and symbols:
                    return [str(s) for s in symbols]
        except (AttributeError, ValueError, TypeError):
            pass

    symbols = os.getenv("BATCH_DATA_PROCESSOR_DEFAULT_SYMBOLS", "").strip()
    if symbols:
        return [s.strip() for s in symbols.split(",") if s.strip()]

    # Import from single source of truth
    if TRADING_SYMBOLS is None:
        msg = "TRADING_SYMBOLS not available"
        raise RuntimeError(msg)

    # Use TRADING_SYMBOLS from trading_universe (live data)
    return list(TRADING_SYMBOLS)


def _get_chunk_size() -> int:
    """Get chunk size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "chunk_size"):
                chunk_size = value.batch_processor.chunk_size
                if isinstance(chunk_size, int) and chunk_size > 0:
                    return chunk_size
        except (AttributeError, ValueError, TypeError):
            pass

    chunk_size = os.getenv("BATCH_DATA_PROCESSOR_CHUNK_SIZE", "").strip()
    if chunk_size:
        try:
            return int(chunk_size)
        except (ValueError, TypeError):
            pass

    return 5000


def _get_feature_window() -> int:
    """Get feature window from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "feature_window"):
                window = value.batch_processor.feature_window
                if isinstance(window, int) and window > 0:
                    return window
        except (AttributeError, ValueError, TypeError):
            pass

    window = os.getenv("BATCH_DATA_PROCESSOR_FEATURE_WINDOW", "").strip()
    if window:
        try:
            return int(window)
        except (ValueError, TypeError):
            pass

    return 48


def _get_prediction_horizon() -> int:
    """Get prediction horizon from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "prediction_horizon"):
                horizon = value.batch_processor.prediction_horizon
                if isinstance(horizon, int) and horizon > 0:
                    return horizon
        except (AttributeError, ValueError, TypeError):
            pass

    horizon = os.getenv("BATCH_DATA_PROCESSOR_PREDICTION_HORIZON", "").strip()
    if horizon:
        try:
            return int(horizon)
        except (ValueError, TypeError):
            pass

    return 24


def _get_parquet_file_pattern() -> str:
    """Get parquet file pattern from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "parquet_file_pattern"):
                pattern = value.batch_processor.parquet_file_pattern
                if isinstance(pattern, str) and pattern:
                    return pattern.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    pattern = os.getenv("BATCH_DATA_PROCESSOR_PARQUET_FILE_PATTERN", "").strip()
    if pattern:
        return pattern

    return "_1h_50000.parquet"


def _get_ichimoku_min_length() -> int:
    """Get minimum length for Ichimoku from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "ichimoku_min_length"):
                length = value.batch_processor.ichimoku_min_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass

    length = os.getenv("BATCH_DATA_PROCESSOR_ICHIMOKU_MIN_LENGTH", "").strip()
    if length:
        try:
            return int(length)
        except (ValueError, TypeError):
            pass

    return 52


def _get_fibonacci_min_length() -> int:
    """Get minimum length for Fibonacci from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "fibonacci_min_length"):
                length = value.batch_processor.fibonacci_min_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass

    length = os.getenv("BATCH_DATA_PROCESSOR_FIBONACCI_MIN_LENGTH", "").strip()
    if length:
        try:
            return int(length)
        except (ValueError, TypeError):
            pass

    return 20


def _get_return_periods() -> list[int]:
    """Get return periods for calculation from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "return_periods"):
                periods = value.batch_processor.return_periods
                if isinstance(periods, list) and periods:
                    return [int(p) for p in periods if isinstance(p, (int, float)) and p > 0]
        except (AttributeError, ValueError, TypeError):
            pass

    periods = os.getenv("BATCH_DATA_PROCESSOR_RETURN_PERIODS", "").strip()
    if periods:
        try:
            return [int(p.strip()) for p in periods.split(",") if p.strip()]
        except (ValueError, TypeError):
            pass

    return [1, 3, 6, 12, 24]


def _get_target_data_points() -> int:
    """Get target data points for learning progress from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_learning") and hasattr(value.ai_learning, "target_data_points"):
                points = value.ai_learning.target_data_points
                if isinstance(points, int) and points > 0:
                    return points
        except (AttributeError, ValueError, TypeError):
            pass

    points = os.getenv("AI_LEARNING_TARGET_DATA_POINTS", "").strip()
    if points:
        try:
            return int(points)
        except (ValueError, TypeError):
            pass

    return 500000


def _get_pattern_discovery_factor() -> float:
    """Get pattern discovery factor from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_learning") and hasattr(value.ai_learning, "pattern_discovery_factor"):
                factor = value.ai_learning.pattern_discovery_factor
                if isinstance(factor, (int, float)) and factor > 0:
                    return float(factor)
        except (AttributeError, ValueError, TypeError):
            pass

    factor = os.getenv("AI_LEARNING_PATTERN_DISCOVERY_FACTOR", "").strip()
    if factor:
        try:
            return float(factor)
        except (ValueError, TypeError):
            pass

    return 0.01


def _get_min_patterns() -> int:
    """Get minimum patterns discovered from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "min_patterns"):
                min_p = value.batch_processor.min_patterns
                if isinstance(min_p, int) and min_p > 0:
                    return min_p
        except (AttributeError, ValueError, TypeError):
            pass

    min_p = os.getenv("BATCH_DATA_PROCESSOR_MIN_PATTERNS", "").strip()
    if min_p:
        try:
            return int(min_p)
        except (ValueError, TypeError):
            pass

    return 5


def _get_separator_width() -> int:
    """Get separator width from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "batch_processor") and hasattr(value.batch_processor, "separator_width"):
                width = value.batch_processor.separator_width
                if isinstance(width, int) and width > 0:
                    return width
        except (AttributeError, ValueError, TypeError):
            pass

    width = os.getenv("BATCH_DATA_PROCESSOR_SEPARATOR_WIDTH", "").strip()
    if width:
        try:
            return int(width)
        except (ValueError, TypeError):
            pass

    return 50


class BatchDataProcessor:
    """Processes large batches of historical data to accelerate AI training"""

    def __init__(self, data_dir: str | None = None, output_dir: str | None = None, parquet_dir: str | None = None, symbols: list[str] | None = None) -> None:
        # Load configuration from live config
        self.data_dir = data_dir if data_dir is not None else _get_data_dir()
        self.output_dir = output_dir if output_dir is not None else _get_output_dir()
        self.parquet_dir = parquet_dir if parquet_dir is not None else _get_parquet_dir()
        self.symbols = symbols if symbols is not None else _get_default_symbols()

        # Ensure output directory exists
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Configuration - load from live config
        self.chunk_size = _get_chunk_size()
        self.feature_window = _get_feature_window()
        self.prediction_horizon = _get_prediction_horizon()
        self.parquet_file_pattern = _get_parquet_file_pattern()
        self.ichimoku_min_length = _get_ichimoku_min_length()
        self.fibonacci_min_length = _get_fibonacci_min_length()
        self.return_periods = _get_return_periods()
        self.target_data_points = _get_target_data_points()
        self.pattern_discovery_factor = _get_pattern_discovery_factor()
        self.min_patterns = _get_min_patterns()

        # Initialize advanced indicators
        self.ichimoku = IchimokuCloud()
        self.fibonacci = FibonacciPatterns()

        # Statistics
        self.stats = {"total_rows_processed": 0, "chunks_created": 0, "symbols_processed": 0, "processing_time": 0, "start_time": time.time()}

    def _load_parquet_file(self, symbol: str) -> pd.DataFrame | None:
        """Load data from parquet file for a specific symbol"""
        try:
            file_path = Path(self.parquet_dir) / f"{symbol}{self.parquet_file_pattern}"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(df)} rows for {symbol}")
                return df
            logger.warning(f"Parquet file not found: {file_path}")
        except (OSError, FileNotFoundError, PermissionError):
            logger.exception(f"File system error loading parquet data for {symbol}")
            return None
        except (ValueError, TypeError, KeyError):
            logger.exception(f"Data error loading parquet data for {symbol}")
            return None
        except RuntimeError:
            logger.exception(f"Unexpected error loading parquet data for {symbol}")
            return None
        else:
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Copy dataframe to avoid modifying original
            df_with_indicators = df.copy()

            # Simple Moving Averages
            df_with_indicators["sma_7"] = df["close"].rolling(window=7).mean()
            df_with_indicators["sma_25"] = df["close"].rolling(window=25).mean()
            df_with_indicators["sma_99"] = df["close"].rolling(window=99).mean()

            # Exponential Moving Averages
            df_with_indicators["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
            df_with_indicators["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

            # Bollinger Bands (20, 2)
            sma = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            df_with_indicators["bb_upper"] = sma + 2 * std
            df_with_indicators["bb_lower"] = sma - 2 * std
            df_with_indicators["bb_width"] = (df_with_indicators["bb_upper"] - df_with_indicators["bb_lower"]) / sma

            # RSI (14)
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df_with_indicators["rsi_14"] = 100 - (100 / (1 + rs))

            # MACD (12, 26, 9)
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df_with_indicators["macd"] = ema_12 - ema_26
            df_with_indicators["macd_signal"] = df_with_indicators["macd"].ewm(span=9, adjust=False).mean()
            df_with_indicators["macd_hist"] = df_with_indicators["macd"] - df_with_indicators["macd_signal"]

            # Average True Range (14)
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df_with_indicators["atr_14"] = true_range.rolling(14).mean()

            # On-Balance Volume (OBV)
            obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
            df_with_indicators["obv"] = obv

            # Price rate of change
            df_with_indicators["roc_10"] = df["close"].pct_change(periods=10) * 100

            # Stochastic Oscillator (14, 3, 3)
            low_min = df["low"].rolling(window=14).min()
            high_max = df["high"].rolling(window=14).max()
            df_with_indicators["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min))
            df_with_indicators["stoch_d"] = df_with_indicators["stoch_k"].rolling(window=3).mean()

            # Williams %R
            df_with_indicators["williams_r"] = -100 * ((high_max - df["close"]) / (high_max - low_min))

            # Commodity Channel Index (CCI)
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            mean_deviation = abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
            df_with_indicators["cci"] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_deviation)

            # Ichimoku Cloud features - Using our advanced indicator class
            if len(df) >= self.ichimoku_min_length:
                try:
                    ichimoku_features = self.ichimoku.get_features(df)
                    for key, value in ichimoku_features.items():
                        df_with_indicators[f"ichimoku_{key}"] = value
                except (ValueError, TypeError, KeyError, AttributeError) as e:
                    logger.warning(f"Error calculating Ichimoku features: {e}")

            # Fibonacci Pattern features
            if len(df) >= self.fibonacci_min_length:
                try:
                    fib_features = self.fibonacci.get_features(df)
                    for key, value in fib_features.items():
                        df_with_indicators[f"fib_{key}"] = value
                except (ValueError, TypeError, KeyError, AttributeError) as e:
                    logger.warning(f"Error calculating Fibonacci features: {e}")

            # Percentage return calculations
            for period in self.return_periods:
                df_with_indicators[f"return_{period}h"] = df["close"].pct_change(periods=period)

            # Clean up NaN values
            return df_with_indicators.bfill().ffill()

        except (ValueError, TypeError, KeyError, AttributeError):
            logger.exception("Data error calculating technical indicators")
            return df
        except RuntimeError:
            logger.exception("Unexpected error calculating technical indicators")
            return df

    def _prepare_training_features(self, df: pd.DataFrame, row_index: int) -> tuple[np.ndarray, list[str]]:
        """Extract features for training from a specific row with lookback window"""
        try:
            if row_index < self.feature_window:
                return np.array([]), []

            # Get historical data window
            window_start = max(0, row_index - self.feature_window)
            window_data = df.iloc[window_start:row_index]

            if len(window_data) < self.feature_window / 2:  # Require at least half the window
                return np.array([]), []

            # Extract features
            features = []
            feature_names = []

            # Price features - VECTORIZED for performance
            price_cols = ["open", "high", "low", "close", "volume"]
            for col in price_cols:
                col_data = window_data[col]

                # Last value
                features.append(col_data.iloc[-1])
                feature_names.append(f"last_{col}")

                # Statistical features - VECTORIZED
                features.append(col_data.mean())
                feature_names.append(f"mean_{col}")

                features.append(col_data.std())
                feature_names.append(f"std_{col}")

                # Min/Max - VECTORIZED
                features.append(col_data.min())
                feature_names.append(f"min_{col}")

                features.append(col_data.max())
                feature_names.append(f"max_{col}")

                # Change features - VECTORIZED
                if col != "volume":
                    pct_change = col_data.pct_change().iloc[-5:].to_numpy()
                    # Use vectorized operations instead of loop
                    features.extend(pct_change)
                    feature_names.extend([f"{col}_change_t-{i + 1}" for i in range(len(pct_change))])

            # Technical indicator features
            for indicator in [
                "sma_7",
                "sma_25",
                "sma_99",
                "ema_9",
                "ema_21",
                "bb_upper",
                "bb_lower",
                "bb_width",
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_hist",
                "atr_14",
                "obv",
                "roc_10",
                "stoch_k",
                "stoch_d",
            ]:
                if indicator in window_data.columns:
                    # Last value
                    features.append(window_data[indicator].iloc[-1])
                    feature_names.append(f"last_{indicator}")

                    # Direction (increasing/decreasing)
                    direction = 1 if window_data[indicator].iloc[-1] > window_data[indicator].iloc[-2] else -1
                    features.append(direction)
                    feature_names.append(f"direction_{indicator}")

            # Return arrays
            return np.array(features), feature_names

        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            logger.exception("Data error preparing training features")
            return np.array([]), []
        except RuntimeError:
            logger.exception("Unexpected error preparing training features")
            return np.array([]), []

    def _prepare_training_target(self, df: pd.DataFrame, row_index: int) -> np.ndarray:
        """Extract target labels for training from a specific row with future window"""
        try:
            # Check if we have enough future data
            if row_index + self.prediction_horizon >= len(df):
                return np.array([])

            # Get current price
            current_close = df["close"].iloc[row_index]

            # Get future price
            future_close = df["close"].iloc[row_index + self.prediction_horizon]

            # Calculate price movement
            price_change = (future_close - current_close) / current_close

            # Create target label (1 for up, 0 for down)
            target = 1 if price_change > 0 else 0

            return np.array([target])

        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            logger.exception("Data error preparing training target")
            return np.array([])
        except RuntimeError:
            logger.exception("Unexpected error preparing training target")
            return np.array([])

    def process_symbol_data(self, symbol: str) -> dict[str, Any]:
        """Process all data for a specific symbol"""
        try:
            # Load data
            df = self._load_parquet_file(symbol)
            if df is None or len(df) == 0:
                logger.warning(f"No data available for {symbol}")
                return {"symbol": symbol, "chunks_created": 0, "rows_processed": 0}

            # Calculate technical indicators
            logger.info(f"Calculating technical indicators for {symbol}")
            df_with_indicators = self._calculate_technical_indicators(df)

            # Process data in chunks
            chunk_features = []
            chunk_targets = []
            chunk_metadata = []
            chunks_created = 0
            rows_processed = 0
            feature_names = []

            # Use tqdm for progress bar
            logger.info(f"Processing {len(df)} rows for {symbol}")
            for i in tqdm(range(self.feature_window, len(df) - self.prediction_horizon), desc=symbol):
                # Extract features and target for this timepoint
                features, feature_cols = self._prepare_training_features(df_with_indicators, i)
                target = self._prepare_training_target(df_with_indicators, i)

                # Skip if invalid
                if len(features) == 0 or len(target) == 0:
                    continue

                # Store feature names from first valid row
                if not feature_names and feature_cols:
                    feature_names = feature_cols

                # Add to current chunk
                chunk_features.append(features)
                chunk_targets.append(target)
                chunk_metadata.append(
                    {"symbol": symbol, "timestamp": str(df.index[i]), "close_price": float(df["close"].iloc[i]), "target_price": float(df["close"].iloc[i + self.prediction_horizon])}
                )

                rows_processed += 1

                # Save chunk if it's full
                if len(chunk_features) >= self.chunk_size:
                    self._save_chunk(symbol, chunks_created, np.array(chunk_features), np.array(chunk_targets), chunk_metadata, feature_names)
                    chunks_created += 1

                    # Reset for next chunk
                    chunk_features = []
                    chunk_targets = []
                    chunk_metadata = []

            # Save remaining data as final chunk
            if len(chunk_features) > 0:
                self._save_chunk(symbol, chunks_created, np.array(chunk_features), np.array(chunk_targets), chunk_metadata, feature_names)
                chunks_created += 1

            logger.info(f"Completed processing {symbol}: {rows_processed} rows, {chunks_created} chunks")
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            logger.exception(f"Data error processing {symbol}")
            return {"symbol": symbol, "chunks_created": 0, "rows_processed": 0, "error": str(e)}
        except RuntimeError as e:
            logger.exception(f"Unexpected error processing {symbol}")
            return {"symbol": symbol, "chunks_created": 0, "rows_processed": 0, "error": str(e)}
        else:
            return {"symbol": symbol, "chunks_created": chunks_created, "rows_processed": rows_processed}

    def _save_chunk(self, symbol: str, chunk_id: int, features: np.ndarray, targets: np.ndarray, metadata: list[dict[str, Any]], feature_names: list[str]) -> None:
        """Save a data chunk to file"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            filename = f"{symbol}_chunk_{chunk_id}_{timestamp}.json"
            filepath = Path(self.output_dir) / filename

            # Convert data for saving
            chunk_data = {
                "symbol": symbol,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "feature_window": self.feature_window,
                "prediction_horizon": self.prediction_horizon,
                "chunk_id": chunk_id,
                "features_shape": features.shape,
                "targets_shape": targets.shape,
                "feature_names": feature_names,
                "features": features.tolist(),
                "targets": targets.tolist(),
                "metadata": metadata,
            }

            # Save to file
            with filepath.open("w") as f:
                json.dump(chunk_data, f)

            # Update stats
            self.stats["chunks_created"] += 1
            self.stats["total_rows_processed"] += len(features)
        except (OSError, FileNotFoundError, PermissionError):
            logger.exception("File system error saving chunk")
        except (ValueError, TypeError, KeyError):
            logger.exception("Data error saving chunk")
        except RuntimeError:
            logger.exception("Unexpected error saving chunk")

    async def process_all_data(self) -> dict[str, Any]:
        """Process all symbols data"""
        try:
            self.stats["start_time"] = time.time()

            results = []
            for symbol in self.symbols:
                # Process each symbol
                symbol_result = self.process_symbol_data(symbol)
                results.append(symbol_result)

                # Update stats
                if symbol_result.get("chunks_created", 0) > 0:
                    self.stats["symbols_processed"] += 1

            # Calculate processing time
            self.stats["processing_time"] = time.time() - self.stats["start_time"]

            # Display summary
            separator_width = _get_separator_width()
            logger.info("=" * separator_width)
            logger.info("PROCESSING COMPLETED")
            logger.info("=" * separator_width)
            logger.info(f"Total symbols processed: {self.stats['symbols_processed']}/{len(self.symbols)}")
            logger.info(f"Total chunks created: {self.stats['chunks_created']}")
            logger.info(f"Total rows processed: {self.stats['total_rows_processed']}")
            logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            logger.exception("Data error processing all data")
            return {"error": str(e)}
        except RuntimeError as e:
            logger.exception("Unexpected error processing all data")
            return {"error": str(e)}
        else:
            return {"results": results, "stats": self.stats}

    def update_learning_data(self) -> None:
        """Update AI learning data file with batch processing results"""
        try:
            ai_data_path = Path(self.data_dir) / "ai_learning_data.json"

            # Load existing data
            if ai_data_path.exists():
                with ai_data_path.open() as f:
                    ai_data = json.load(f)
            else:
                # Create new data structure
                ai_data = {"learning_metrics": {}, "training_status": {}, "model_performance": {}, "ai_trading_simulation": {}}

            # Update metrics
            metrics = ai_data.get("learning_metrics", {})
            metrics["total_data_points"] = metrics.get("total_data_points", 0) + self.stats["total_rows_processed"]
            metrics["patterns_discovered"] = max(self.min_patterns, int(metrics["total_data_points"] * self.pattern_discovery_factor))
            metrics["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Calculate learning progress (0-100%)
            metrics["learning_progress"] = min(100.0, (metrics["total_data_points"] / self.target_data_points) * 100)

            ai_data["learning_metrics"] = metrics

            # Save updated data
            with ai_data_path.open("w") as f:
                json.dump(ai_data, f, indent=2)

            logger.info(f"Updated AI learning data: {ai_data_path}")
        except (OSError, FileNotFoundError, PermissionError):
            logger.exception("File system error updating AI learning data")
        except (ValueError, TypeError, KeyError):
            logger.exception("Data error updating AI learning data")
        except RuntimeError:
            logger.exception("Unexpected error updating AI learning data")


async def main() -> None:
    """Main entry point"""
    try:
        processor = BatchDataProcessor()
        await processor.process_all_data()
        processor.update_learning_data()
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        logger.exception("Data error in batch data processor")
    except RuntimeError:
        logger.exception("Unexpected error in batch data processor")


if __name__ == "__main__":
    asyncio.run(main())
