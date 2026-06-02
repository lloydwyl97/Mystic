"""
AI Auto-Retrain Service - Live Configuration Only

Automatic model retraining and optimization system.
All configuration values come from live config - no hardcoded values.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

import redis

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.services.binance_rest_client import BinanceREST
from backend.services.live_market_data import live_market_data_service
from backend.utils.binance_weight_limiter import BinanceWeightLimiter
from backend.utils.path_helpers import (
    ensure_model_directories,
    get_model_file_path,
    get_scaler_file_path,
)
from utils.redis_helpers import to_str, to_str_list

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from project root (single source of truth)
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=str(root_env))

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_redis_host() -> str:
    """Get Redis host from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "host"):
                host = value.redis.host
                if isinstance(host, str) and host:
                    return host.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    host = os.getenv("REDIS_HOST", "").strip()
    if host:
        return host

    return "localhost"


def _get_redis_port() -> int:
    """Get Redis port from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "port"):
                port = value.redis.port
                if isinstance(port, int) and 1 <= port <= 65535:
                    return port
        except (AttributeError, ValueError, TypeError):
            pass

    port = os.getenv("REDIS_PORT", "").strip()
    if port:
        try:
            port_val = int(port)
            if 1 <= port_val <= 65535:
                return port_val
        except (ValueError, TypeError):
            pass

    return 6379


def _get_redis_db() -> int:
    """Get Redis DB number from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "redis") and hasattr(value.redis, "db"):
                db = value.redis.db
                if isinstance(db, int) and db >= 0:
                    return db
        except (AttributeError, ValueError, TypeError):
            pass

    db = os.getenv("REDIS_DB", "").strip()
    if db:
        try:
            db_val = int(db)
            if db_val >= 0:
                return db_val
        except (ValueError, TypeError):
            pass

    return 0


def _get_retrain_threshold() -> float:
    """Get retrain threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "retrain_threshold"):
                threshold = value.ai_auto_retrain.retrain_threshold
                if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("AI_AUTO_RETRAIN_THRESHOLD", "").strip()
    if threshold:
        try:
            threshold_val = float(threshold)
            if 0.0 <= threshold_val <= 1.0:
                return threshold_val
        except (ValueError, TypeError):
            pass

    return 0.05


def _get_min_data_points() -> int:
    """Get minimum data points from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "min_data_points"):
                points = value.ai_auto_retrain.min_data_points
                if isinstance(points, int) and points > 0:
                    return points
        except (AttributeError, ValueError, TypeError):
            pass

    points = os.getenv("AI_AUTO_RETRAIN_MIN_DATA_POINTS", "").strip()
    if points:
        try:
            points_val = int(points)
            if points_val > 0:
                return points_val
        except (ValueError, TypeError):
            pass

    return 1000


def _get_retrain_interval_hours() -> int:
    """Get retrain interval in hours from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "retrain_interval_hours"):
                hours = value.ai_auto_retrain.retrain_interval_hours
                if isinstance(hours, int) and hours > 0:
                    return hours
        except (AttributeError, ValueError, TypeError):
            pass

    hours = os.getenv("AI_AUTO_RETRAIN_INTERVAL_HOURS", "").strip()
    if hours:
        try:
            hours_val = int(hours)
            if hours_val > 0:
                return hours_val
        except (ValueError, TypeError):
            pass

    return 24


def _get_performance_window_days() -> int:
    """Get performance window in days from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "performance_window_days"):
                days = value.ai_auto_retrain.performance_window_days
                if isinstance(days, int) and days > 0:
                    return days
        except (AttributeError, ValueError, TypeError):
            pass

    days = os.getenv("AI_AUTO_RETRAIN_PERFORMANCE_WINDOW_DAYS", "").strip()
    if days:
        try:
            days_val = int(days)
            if days_val > 0:
                return days_val
        except (ValueError, TypeError):
            pass

    return 7


def _get_monitoring_interval_seconds() -> int:
    """Get monitoring interval in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "monitoring_interval_seconds"):
                interval = value.ai_auto_retrain.monitoring_interval_seconds
                if isinstance(interval, int) and interval > 0:
                    return interval
        except (AttributeError, ValueError, TypeError):
            pass

    interval = os.getenv("AI_AUTO_RETRAIN_MONITORING_INTERVAL_SECONDS", "").strip()
    if interval:
        try:
            interval_val = int(interval)
            if interval_val > 0:
                return interval_val
        except (ValueError, TypeError):
            pass

    return 300


def _get_error_sleep_seconds() -> int:
    """Get error sleep interval in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "error_sleep_seconds"):
                sleep = value.ai_auto_retrain.error_sleep_seconds
                if isinstance(sleep, int) and sleep > 0:
                    return sleep
        except (AttributeError, ValueError, TypeError):
            pass

    sleep = os.getenv("AI_AUTO_RETRAIN_ERROR_SLEEP_SECONDS", "").strip()
    if sleep:
        try:
            sleep_val = int(sleep)
            if sleep_val > 0:
                return sleep_val
        except (ValueError, TypeError):
            pass

    return 600


def _get_model_configs() -> dict[str, dict[str, Any]]:
    """Get model configurations from live config."""
    configs = {
        "lstm": {},
        "transformer": {},
    }

    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "model_configs"):
                live_configs = value.ai_auto_retrain.model_configs
                if isinstance(live_configs, dict):
                    configs.update(live_configs)

            # Get LSTM config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "lstm"):
                lstm = value.ai_auto_retrain.lstm
                if isinstance(lstm, dict):
                    configs["lstm"].update(lstm)

            # Get Transformer config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "transformer"):
                transformer = value.ai_auto_retrain.transformer
                if isinstance(transformer, dict):
                    configs["transformer"].update(transformer)
        except (AttributeError, ValueError, TypeError):
            pass

    # Set defaults if not provided
    if "input_size" not in configs["lstm"]:
        configs["lstm"]["input_size"] = 10
    if "hidden_size" not in configs["lstm"]:
        configs["lstm"]["hidden_size"] = 128
    if "num_layers" not in configs["lstm"]:
        configs["lstm"]["num_layers"] = 3
    if "output_size" not in configs["lstm"]:
        configs["lstm"]["output_size"] = 3
    if "sequence_length" not in configs["lstm"]:
        configs["lstm"]["sequence_length"] = 60

    if "input_size" not in configs["transformer"]:
        configs["transformer"]["input_size"] = 10
    if "d_model" not in configs["transformer"]:
        configs["transformer"]["d_model"] = 128
    if "nhead" not in configs["transformer"]:
        configs["transformer"]["nhead"] = 8
    if "num_layers" not in configs["transformer"]:
        configs["transformer"]["num_layers"] = 4
    if "output_size" not in configs["transformer"]:
        configs["transformer"]["output_size"] = 3
    if "sequence_length" not in configs["transformer"]:
        configs["transformer"]["sequence_length"] = 60

    return configs


def _get_default_learning_rate() -> float:
    """Get default learning rate from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "default_learning_rate"):
                lr = value.ai_auto_retrain.default_learning_rate
                if isinstance(lr, (int, float)) and lr > 0:
                    return float(lr)
        except (AttributeError, ValueError, TypeError):
            pass

    lr = os.getenv("AI_AUTO_RETRAIN_DEFAULT_LEARNING_RATE", "").strip()
    if lr:
        try:
            lr_val = float(lr)
            if lr_val > 0:
                return lr_val
        except (ValueError, TypeError):
            pass

    return 0.001


def _get_default_epochs() -> int:
    """Get default epochs from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "default_epochs"):
                epochs = value.ai_auto_retrain.default_epochs
                if isinstance(epochs, int) and epochs > 0:
                    return epochs
        except (AttributeError, ValueError, TypeError):
            pass

    epochs = os.getenv("AI_AUTO_RETRAIN_DEFAULT_EPOCHS", "").strip()
    if epochs:
        try:
            epochs_val = int(epochs)
            if epochs_val > 0:
                return epochs_val
        except (ValueError, TypeError):
            pass

    return 50


def _get_default_batch_size() -> int:
    """Get default batch size from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "default_batch_size"):
                batch = value.ai_auto_retrain.default_batch_size
                if isinstance(batch, int) and batch > 0:
                    return batch
        except (AttributeError, ValueError, TypeError):
            pass

    batch = os.getenv("AI_AUTO_RETRAIN_DEFAULT_BATCH_SIZE", "").strip()
    if batch:
        try:
            batch_val = int(batch)
            if batch_val > 0:
                return batch_val
        except (ValueError, TypeError):
            pass

    return 32


def _get_train_split_ratio() -> float:
    """Get train/test split ratio from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "train_split_ratio"):
                ratio = value.ai_auto_retrain.train_split_ratio
                if isinstance(ratio, (int, float)) and 0.0 < ratio < 1.0:
                    return float(ratio)
        except (AttributeError, ValueError, TypeError):
            pass

    ratio = os.getenv("AI_AUTO_RETRAIN_TRAIN_SPLIT_RATIO", "").strip()
    if ratio:
        try:
            ratio_val = float(ratio)
            if 0.0 < ratio_val < 1.0:
                return ratio_val
        except (ValueError, TypeError):
            pass

    return 0.8


def _get_labeling_buy_threshold() -> float:
    """Get buy labeling threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "labeling_buy_threshold"):
                threshold = value.ai_auto_retrain.labeling_buy_threshold
                if isinstance(threshold, (int, float)) and threshold > 0:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("AI_AUTO_RETRAIN_LABELING_BUY_THRESHOLD", "").strip()
    if threshold:
        try:
            threshold_val = float(threshold)
            if threshold_val > 0:
                return threshold_val
        except (ValueError, TypeError):
            pass

    return 0.01


def _get_labeling_sell_threshold() -> float:
    """Get sell labeling threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "labeling_sell_threshold"):
                threshold = value.ai_auto_retrain.labeling_sell_threshold
                if isinstance(threshold, (int, float)) and threshold > 0:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("AI_AUTO_RETRAIN_LABELING_SELL_THRESHOLD", "").strip()
    if threshold:
        try:
            threshold_val = float(threshold)
            if threshold_val > 0:
                return float(threshold_val)
        except (ValueError, TypeError):
            pass

    return 0.01


def _get_redis_expiration_seconds() -> int:
    """Get Redis expiration time in seconds from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "redis_expiration_seconds"):
                exp = value.ai_auto_retrain.redis_expiration_seconds
                if isinstance(exp, int) and exp > 0:
                    return exp
        except (AttributeError, ValueError, TypeError):
            pass

    exp = os.getenv("AI_AUTO_RETRAIN_REDIS_EXPIRATION_SECONDS", "").strip()
    if exp:
        try:
            exp_val = int(exp)
            if exp_val > 0:
                return exp_val
        except (ValueError, TypeError):
            pass

    return 86400


def _get_klines_limit() -> int:
    """Get klines limit from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "klines_limit"):
                limit = value.ai_auto_retrain.klines_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass

    limit = os.getenv("AI_AUTO_RETRAIN_KLINES_LIMIT", "").strip()
    if limit:
        try:
            limit_val = int(limit)
            if limit_val > 0:
                return limit_val
        except (ValueError, TypeError):
            pass

    return 8760


def _get_default_symbol() -> str:
    """Get default symbol from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "trading_universe") and hasattr(value.trading_universe, "top10_symbols"):
                symbols = value.trading_universe.top10_symbols
                if isinstance(symbols, list) and symbols:
                    return str(symbols[0])
        except (AttributeError, ValueError, TypeError, IndexError):
            pass

    symbol = os.getenv("AI_AUTO_RETRAIN_DEFAULT_SYMBOL", "").strip()
    if symbol:
        return symbol

    # Use first symbol from TRADING_SYMBOLS (live data) - convert to CCXT format
    if not TRADING_SYMBOLS:
        msg = "No trading symbols available - TRADING_SYMBOLS must be configured"
        raise RuntimeError(msg)
    # Convert BTCUSDT to BTC/USDT format
    base_symbol = TRADING_SYMBOLS[0]
    if base_symbol.endswith("USDT"):
        return f"{base_symbol[:-4]}/USDT"
    return f"{base_symbol}/USDT"


def _get_rsi_period() -> int:
    """Get RSI period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "rsi_period"):
                period = value.ai_auto_retrain.rsi_period
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass

    period = os.getenv("AI_AUTO_RETRAIN_RSI_PERIOD", "").strip()
    if period:
        try:
            period_val = int(period)
            if period_val > 0:
                return period_val
        except (ValueError, TypeError):
            pass

    return 14


def _get_macd_fast() -> int:
    """Get MACD fast period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "macd_fast"):
                fast = value.ai_auto_retrain.macd_fast
                if isinstance(fast, int) and fast > 0:
                    return fast
        except (AttributeError, ValueError, TypeError):
            pass

    fast = os.getenv("AI_AUTO_RETRAIN_MACD_FAST", "").strip()
    if fast:
        try:
            fast_val = int(fast)
            if fast_val > 0:
                return fast_val
        except (ValueError, TypeError):
            pass

    return 12


def _get_macd_slow() -> int:
    """Get MACD slow period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "macd_slow"):
                slow = value.ai_auto_retrain.macd_slow
                if isinstance(slow, int) and slow > 0:
                    return slow
        except (AttributeError, ValueError, TypeError):
            pass

    slow = os.getenv("AI_AUTO_RETRAIN_MACD_SLOW", "").strip()
    if slow:
        try:
            slow_val = int(slow)
            if slow_val > 0:
                return slow_val
        except (ValueError, TypeError):
            pass

    return 26


def _get_bollinger_period() -> int:
    """Get Bollinger Bands period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "bollinger_period"):
                period = value.ai_auto_retrain.bollinger_period
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass

    period = os.getenv("AI_AUTO_RETRAIN_BOLLINGER_PERIOD", "").strip()
    if period:
        try:
            period_val = int(period)
            if period_val > 0:
                return period_val
        except (ValueError, TypeError):
            pass

    return 20


def _get_bollinger_std_dev() -> float:
    """Get Bollinger Bands standard deviation from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "bollinger_std_dev"):
                std_dev = value.ai_auto_retrain.bollinger_std_dev
                if isinstance(std_dev, (int, float)) and std_dev > 0:
                    return float(std_dev)
        except (AttributeError, ValueError, TypeError):
            pass

    std_dev = os.getenv("AI_AUTO_RETRAIN_BOLLINGER_STD_DEV", "").strip()
    if std_dev:
        try:
            std_dev_val = float(std_dev)
            if std_dev_val > 0:
                return std_dev_val
        except (ValueError, TypeError):
            pass

    return 2.0


def _get_sma_short_period() -> int:
    """Get short SMA period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "sma_short_period"):
                period = value.ai_auto_retrain.sma_short_period
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass

    period = os.getenv("AI_AUTO_RETRAIN_SMA_SHORT_PERIOD", "").strip()
    if period:
        try:
            period_val = int(period)
            if period_val > 0:
                return period_val
        except (ValueError, TypeError):
            pass

    return 20


def _get_sma_long_period() -> int:
    """Get long SMA period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "sma_long_period"):
                period = value.ai_auto_retrain.sma_long_period
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass

    period = os.getenv("AI_AUTO_RETRAIN_SMA_LONG_PERIOD", "").strip()
    if period:
        try:
            period_val = int(period)
            if period_val > 0:
                return period_val
        except (ValueError, TypeError):
            pass

    return 50


def _get_volume_sma_period() -> int:
    """Get volume SMA period from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "volume_sma_period"):
                period = value.ai_auto_retrain.volume_sma_period
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass

    period = os.getenv("AI_AUTO_RETRAIN_VOLUME_SMA_PERIOD", "").strip()
    if period:
        try:
            period_val = int(period)
            if period_val > 0:
                return period_val
        except (ValueError, TypeError):
            pass

    return 20


def _get_volatility_window() -> int:
    """Get volatility rolling window from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_auto_retrain") and hasattr(value.ai_auto_retrain, "volatility_window"):
                window = value.ai_auto_retrain.volatility_window
                if isinstance(window, int) and window > 0:
                    return window
        except (AttributeError, ValueError, TypeError):
            pass

    window = os.getenv("AI_AUTO_RETRAIN_VOLATILITY_WINDOW", "").strip()
    if window:
        try:
            window_val = int(window)
            if window_val > 0:
                return window_val
        except (ValueError, TypeError):
            pass

    return 20


class AutoRetrainService:
    def __init__(self) -> None:
        """Initialize Auto-Retrain Service with live configuration."""
        self.redis_client = redis.Redis(
            host=_get_redis_host(),
            port=_get_redis_port(),
            db=_get_redis_db(),
            decode_responses=True,
        )
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Retraining parameters from live config
        self.retrain_threshold = _get_retrain_threshold()
        self.min_data_points = _get_min_data_points()
        self.retrain_interval_hours = _get_retrain_interval_hours()
        self.performance_window_days = _get_performance_window_days()

        # Ensure model directories exist
        ensure_model_directories()

        # Model configurations from live config
        self.model_configs = _get_model_configs()

    async def start(self):
        """Start the Auto-Retrain Service"""
        logger.info("START: Starting Auto-Retrain Service...")
        self.running = True

        # Start monitoring and retraining
        await self.monitor_and_retrain()

    async def monitor_and_retrain(self):
        """Monitor model performance and trigger retraining"""
        logger.info("👀 Monitoring model performance...")

        while self.running:
            try:
                # Check all active models
                active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))

                for model_id in active_models:
                    await self.check_model_performance(model_id)

                # Check for retraining requests
                retrain_request = to_str(self.redis_client.lpop("retrain_queue"))
                if retrain_request:
                    request_data = json.loads(retrain_request)
                    await self.process_retrain_request(request_data)

                monitoring_interval = _get_monitoring_interval_seconds()
                await asyncio.sleep(monitoring_interval)

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("ERROR: Error in monitoring")
                error_sleep = _get_error_sleep_seconds()
                await asyncio.sleep(error_sleep)

    async def check_model_performance(self, model_id: str):
        """Check if model needs retraining"""
        try:
            model_data = self.redis_client.get(f"ai_strategy:{model_id}")
            if not model_data:
                return

            model = json.loads(model_data)

            # Get recent performance data
            recent_performance = await self.get_recent_performance(model_id)

            if not recent_performance:
                return

            # Check if performance has degraded
            if await self.should_retrain(model, recent_performance):
                logger.info(f"RETRAIN: Model {model_id} needs retraining")

                # Add to retrain queue
                retrain_request = {
                    "model_id": model_id,
                    "reason": "performance_degradation",
                    "current_performance": recent_performance,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                self.redis_client.lpush("retrain_queue", json.dumps(retrain_request))

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error checking model performance")

    async def get_recent_performance(self, _model_id: str) -> dict[str, Any] | None:
        """Get recent performance data for model from live sources."""
        try:
            # Get performance data from configured window
            end_date = datetime.now(timezone.utc)
            end_date - timedelta(days=self.performance_window_days)

            # Performance data should come from live sources (database, Redis, etc.)
            # Returning None to indicate no data available (no hardcoded fallback)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error getting recent performance")
            return None
        else:
            return None

    async def should_retrain(self, model: dict[str, Any], recent_performance: dict[str, Any]) -> bool:
        """Determine if model should be retrained"""
        try:
            # Get baseline performance
            baseline_performance = model.get("performance", {})

            if not baseline_performance:
                return False

            # Check accuracy degradation
            baseline_acc = baseline_performance.get("accuracy", 0)
            current_acc = recent_performance.get("accuracy", 0)

            if baseline_acc - current_acc > self.retrain_threshold:
                return True

            # Check return degradation
            baseline_return = baseline_performance.get("total_return", 0)
            current_return = recent_performance.get("total_return", 0)

            if baseline_return - current_return > self.retrain_threshold:
                return True

            # Check if enough time has passed since last retrain
            last_retrain = model.get("last_retrain")
            if last_retrain:
                last_retrain_time = datetime.fromisoformat(last_retrain).replace(tzinfo=timezone.utc)
                time_since_retrain = datetime.now(timezone.utc) - last_retrain_time

                result = time_since_retrain.total_seconds() > self.retrain_interval_hours * 3600
            else:
                result = False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error checking if should retrain")
            return False
        else:
            return result

    async def process_retrain_request(self, request_data: dict[str, Any]):
        """Process retraining request"""
        try:
            model_id = request_data.get("model_id")
            reason = request_data.get("reason", "unknown")

            logger.info(f"PROCESS: Processing retrain request for {model_id} - Reason: {reason}")

            # Get model data
            model_data = self.redis_client.get(f"ai_strategy:{model_id}")
            if not model_data:
                logger.warning(f"ERROR: Model {model_id} not found")
                return

            model = json.loads(model_data)

            # Retrain model
            retrained_model = await self.retrain_model(model)

            if retrained_model:
                # Update model
                await self.update_model(model_id, retrained_model)

                # Notify versioning service
                self.redis_client.lpush("new_models_queue", json.dumps(retrained_model))

                logger.info(f"SUCCESS: Successfully retrained model {model_id}")
            else:
                logger.warning(f"ERROR: Failed to retrain model {model_id}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error processing retrain request")

    async def retrain_model(self, model: dict[str, Any]) -> dict[str, Any] | None:
        """Retrain a model with new data"""
        try:
            model_type = model.get("type", "lstm")
            default_symbol = _get_default_symbol()
            symbol = model.get("symbol", default_symbol)

            logger.info(f"RETRAIN: Retraining {model_type} model for {symbol}")

            # Get new training data
            new_data = await self.get_training_data(symbol)

            if new_data.empty:
                logger.warning("ERROR: No new training data available")
                return None

            # Prepare features
            features = self.prepare_features(new_data)

            if len(features[0]) == 0:
                logger.warning("ERROR: Failed to prepare features")
                return None

            # Train new model
            new_model, new_scaler = await self.train_model(model_type, features, model.get("parameters", {}))

            if new_model is None:
                logger.warning("ERROR: Failed to train new model")
                return None

            # Evaluate new model
            performance = await self.evaluate_model(new_model, new_scaler, new_data)

            # Create retrained model object
            retrained_model = {
                "id": (f"{model['id']}_RETRAINED_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"),
                "name": f"{model['name']} (Retrained)",
                "type": model_type,
                "symbol": symbol,
                "model_type": model_type,
                "status": "ACTIVE",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parameters": model.get("parameters", {}),
                "performance": performance,
                "model_path": get_model_file_path(
                    model_type,
                    symbol,
                    f"_retrained_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                ),
                "scaler_path": get_scaler_file_path(
                    model_type,
                    symbol,
                    f"_retrained_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                ),
                "parent_model": model["id"],
                "retrain_reason": "performance_degradation",
                "last_retrain": datetime.now(timezone.utc).isoformat(),
            }

            # Save new model and scaler
            await self.save_model(new_model, new_scaler, retrained_model)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error retraining model")
            return None
        else:
            return retrained_model

    async def get_training_data(self, symbol: str) -> pd.DataFrame:
        """Get training data for retraining from live sources only"""
        try:
            # Use live market data instead of generated data
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)

            # Get klines data from live config
            klines_limit = _get_klines_limit()
            klines = await client.klines(symbol, "1h", klines_limit)

            if not klines or len(klines) == 0:
                logger.warning(f"No live training data available for {symbol}")
                return pd.DataFrame()

            # Convert klines to DataFrame
            data = []
            for kline in klines:
                try:
                    data.append(
                        {
                            "timestamp": datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                        },
                    )
                except (ValueError, IndexError):
                    continue

            if not data:
                logger.warning(f"Failed to parse live training data for {symbol}")
                return pd.DataFrame()

            return pd.DataFrame(data).set_index("timestamp")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception(f"Error getting live training data for {symbol}")
            # Try fallback to live market data service
            try:
                klines_limit = _get_klines_limit()
                coin_data = await live_market_data_service.get_historical_data(symbol, "1h", klines_limit)

                if coin_data and coin_data.price_history:
                    data = []
                    for price_point in coin_data.price_history:
                        data.append(
                            {
                                "timestamp": datetime.fromisoformat(price_point.get("timestamp", datetime.now(timezone.utc).isoformat())),
                                "open": float(price_point.get("price", 0)),
                                "high": float(price_point.get("price", 0)),
                                "low": float(price_point.get("price", 0)),
                                "close": float(price_point.get("price", 0)),
                                "volume": float(price_point.get("volume", 0)),
                            },
                        )

                    if data:
                        return pd.DataFrame(data).set_index("timestamp")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception(f"Fallback also failed for {symbol}")

            return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any]]:
        """Prepare features for model training"""
        try:
            # Calculate technical indicators using live config
            sma_short = _get_sma_short_period()
            sma_long = _get_sma_long_period()
            volume_sma_period = _get_volume_sma_period()
            volatility_window = _get_volatility_window()

            data["sma_20"] = data["close"].rolling(window=sma_short).mean()
            data["sma_50"] = data["close"].rolling(window=sma_long).mean()
            data["rsi"] = self.calculate_rsi(data["close"])
            data["macd"] = self.calculate_macd(data["close"])
            data["bb_upper"], data["bb_middle"], data["bb_lower"] = self.calculate_bollinger_bands(data["close"])
            data["volume_sma"] = data["volume"].rolling(window=volume_sma_period).mean()
            data["price_change"] = data["close"].pct_change()
            data["volatility"] = data["price_change"].rolling(window=volatility_window).std()

            # Create features
            feature_columns = [
                "close",
                "volume",
                "sma_20",
                "sma_50",
                "rsi",
                "macd",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "volatility",
            ]

            features = data[feature_columns].fillna(0).to_numpy()

            # Create labels (simplified: 0=Hold, 1=Buy, 2=Sell) using live config
            labels = np.zeros(len(features))

            buy_threshold = _get_labeling_buy_threshold()
            sell_threshold = _get_labeling_sell_threshold()

            # Simple labeling logic
            sequence_length = self.model_configs.get("lstm", {}).get("sequence_length", 60)
            for i in range(sequence_length, len(features)):
                future_return = (data["close"].iloc[i + 1] - data["close"].iloc[i]) / data["close"].iloc[i] if i + 1 < len(data) else 0

                if future_return > buy_threshold:
                    labels[i] = 1  # Buy
                elif future_return < -sell_threshold:
                    labels[i] = 2  # Sell
                else:
                    labels[i] = 0  # Hold
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error preparing features")
            return np.array([]), np.array([])
        else:
            return features, labels

    async def train_model(
        self,
        strategy_type: str,
        features: NDArray[Any],
        parameters: dict[str, Any] | None = None,
    ) -> tuple[nn.Module | None, MinMaxScaler | None]:
        """Train the AI model"""
        try:
            if len(features) == 0:
                return None, None

            # Prepare data
            x_features, y_labels = features

            # Normalize features
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_features)

            # Create sequences
            sequence_length = self.model_configs[strategy_type]["sequence_length"]
            x_sequences, y_sequences = self.create_sequences(x_scaled, y_labels, sequence_length)

            if len(x_sequences) == 0:
                return None, None

            # Split data using live config
            train_split_ratio = _get_train_split_ratio()
            split_idx = int(train_split_ratio * len(x_sequences))
            x_train, x_test = x_sequences[:split_idx], x_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

            # Create model
            if strategy_type == "lstm":
                config = self.model_configs["lstm"]
                model = nn.LSTM(
                    input_size=config["input_size"],
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    batch_first=True,
                    dropout=0.2,
                )
                model.fc = nn.Linear(config["hidden_size"], config["output_size"])
            elif strategy_type == "transformer":
                config = self.model_configs["transformer"]
                model = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(config["d_model"], config["nhead"]),
                    config["num_layers"],
                )
                model.input_projection = nn.Linear(config["input_size"], config["d_model"])
                model.output_projection = nn.Linear(config["d_model"], config["output_size"])
            else:
                return None, None

            model.to(self.device)

            # Training parameters from live config
            default_lr = _get_default_learning_rate()
            default_epochs = _get_default_epochs()
            default_batch_size = _get_default_batch_size()

            learning_rate = parameters.get("learning_rate", default_lr) if parameters else default_lr
            epochs = parameters.get("epochs", default_epochs) if parameters else default_epochs
            _ = parameters.get("batch_size", default_batch_size) if parameters else default_batch_size

            # Train model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Convert to tensors
            x_train_tensor = torch.FloatTensor(x_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            x_test_tensor = torch.FloatTensor(x_test).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device)

            # Training loop
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(x_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Evaluate model
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                accuracy = accuracy_score(y_test_tensor.cpu(), test_predictions.cpu())
                logger.info(f"Retrained model accuracy: {accuracy:.4f}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error training model")
            return None, None
        else:
            return model, scaler

    def create_sequences(self, x_data: NDArray[Any], y_data: NDArray[Any], sequence_length: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Create sequences for time series prediction"""
        try:
            x_sequences, y_sequences = [], []

            for i in range(sequence_length, len(x_data)):
                x_sequences.append(x_data[i - sequence_length : i])
                y_sequences.append(y_data[i])

            return np.array(x_sequences), np.array(y_sequences)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error creating sequences")
            return np.array([]), np.array([])

    async def evaluate_model(self, model: nn.Module, scaler: MinMaxScaler, data: pd.DataFrame) -> dict[str, Any]:
        """Evaluate model performance"""
        try:
            # Prepare test data
            features = self.prepare_features(data)
            if len(features[0]) == 0:
                return {}

            x_data, y_data = features
            x_scaled = scaler.transform(x_data)

            # Create sequences using live config
            sequence_length = self.model_configs.get("lstm", {}).get("sequence_length", 60)
            x_sequences, y_sequences = self.create_sequences(x_scaled, y_data, sequence_length)

            if len(x_sequences) == 0:
                return {}

            # Test model
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x_sequences).to(self.device)
                outputs = model(x_tensor)
                predictions = torch.argmax(outputs, dim=1)

                # Calculate metrics
                accuracy = accuracy_score(y_sequences, predictions.cpu())
                precision = precision_score(
                    y_sequences,
                    predictions.cpu(),
                    average="weighted",
                    zero_division=0,
                )
                recall = recall_score(
                    y_sequences,
                    predictions.cpu(),
                    average="weighted",
                    zero_division=0,
                )
                f1 = f1_score(
                    y_sequences,
                    predictions.cpu(),
                    average="weighted",
                    zero_division=0,
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error evaluating model")
            return {}
        else:
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "total_return": 0.0,  # Would be calculated from backtest
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

    async def save_model(
        self,
        model: nn.Module,
        scaler: MinMaxScaler,
        model_data: dict[str, Any],
    ):
        """Save model and scaler"""
        try:
            # Ensure model directories exist (handled by path helpers)
            ensure_model_directories()

            # Save model
            torch.save(model.state_dict(), model_data["model_path"])

            # Save scaler
            joblib.dump(scaler, model_data["scaler_path"])

            logger.info(f"SUCCESS: Saved retrained model: {model_data['model_path']}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error saving model")

    async def update_model(self, model_id: str, new_model_data: dict[str, Any]):
        """Update model in Redis"""
        try:
            # Update model data using live config
            redis_expiration = _get_redis_expiration_seconds()
            self.redis_client.set(f"ai_strategy:{model_id}", json.dumps(new_model_data), ex=redis_expiration)

            # Update last retrain time
            self.redis_client.set(
                f"last_retrain:{model_id}",
                datetime.now(timezone.utc).isoformat(),
                ex=redis_expiration,
            )

            # Broadcast model metrics update
            await self.broadcast_model_metrics()

            # Broadcast retrain status update
            await self.broadcast_retrain_status()

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error updating model")

    async def broadcast_model_metrics(self):
        """Broadcast model performance metrics"""
        try:
            # Get all active models
            active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))
            models_data = []

            for model_id in active_models:
                model_data = self.redis_client.get(f"ai_strategy:{model_id}")
                if model_data:
                    model = json.loads(model_data)
                    models_data.append(model)

            # Create metrics payload
            metrics_payload = {
                "models": models_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store in Redis for dashboard access
            self.redis_client.set("model_metrics", json.dumps(metrics_payload), ex=300)

            # Publish to Redis channel
            self.redis_client.publish("model_metrics", json.dumps(metrics_payload))

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error broadcasting model metrics")

    async def broadcast_retrain_status(self):
        """Broadcast retrain status and queue"""
        try:
            # Get retrain queue
            queue_data = to_str_list(self.redis_client.lrange("retrain_queue", 0, -1))
            queue = [json.loads(item) for item in queue_data]

            # Get current retrain status
            status = {
                "currently_retraining": None,
                "retrain_progress": 0.0,
                "estimated_completion": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Check if any model is currently retraining
            for item in queue:
                if item.get("status") == "retraining":
                    status["currently_retraining"] = item["model_id"]
                    status["retrain_progress"] = item.get("progress", 0.0)
                    status["estimated_completion"] = item.get("estimated_completion")
                    break

            # Create status payload
            status_payload = {
                "queue": queue,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store in Redis for dashboard access
            self.redis_client.set("retrain_status", json.dumps(status_payload), ex=300)

            # Publish to Redis channel
            self.redis_client.publish("retrain_status", json.dumps(status_payload))

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error broadcasting retrain status")

    # Technical indicator calculations
    def calculate_rsi(self, prices: pd.Series, period: int | None = None) -> pd.Series:
        """Calculate RSI using live config."""
        if period is None:
            period = _get_rsi_period()
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int | None = None, slow: int | None = None) -> pd.Series:
        """Calculate MACD using live config."""
        if fast is None:
            fast = _get_macd_fast()
        if slow is None:
            slow = _get_macd_slow()
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(self, prices: pd.Series, period: int | None = None, std_dev: float | None = None) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands using live config."""
        if period is None:
            period = _get_bollinger_period()
        if std_dev is None:
            std_dev = _get_bollinger_std_dev()
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    async def stop(self):
        """Stop the Auto-Retrain Service"""
        logger.info("STOP: Stopping Auto-Retrain Service...")
        self.running = False


async def main():
    """Main function"""
    retrain_service = AutoRetrainService()

    try:
        await retrain_service.start()
    except KeyboardInterrupt:
        logger.info("STOP: Received interrupt signal")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("ERROR: Error in main")
    finally:
        await retrain_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
