"""
AI Strategy Generator Service
Advanced neural network-based strategy generation and signal optimization
"""

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from joblib import dump as joblib_dump
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import redis
from backend.config.redis_config import get_shared_redis_sync
from backend.services.task_manager import task_manager

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.modules.ai.persistent_cache import get_persistent_cache
from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
from backend.services.binance_rest_client import BinanceREST
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = tuple(TRADING_SYMBOLS)
QUEUE_KEY = "strategy_queue"

if not Path("logs").is_dir():
    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_strategy_generator_core.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_strategy_generator_core")

load_dotenv()

app = FastAPI(title="AI Strategy Generator", version="1.0.0")

# Health endpoint DELETED


@app.get("/")
async def root():
    return {"message": "AI Strategy Generator is running"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_symbol(s: str) -> str:
    return str(s).replace("/", "").upper()


def _validate_symbol(s: str) -> str:
    sym = _normalize_symbol(s)
    if sym not in ALLOWED_SYMBOLS:
        msg = f"symbol not allowed: {sym}"
        raise ValueError(msg)
    return sym


def _redis_from_env() -> redis.Redis:
    client = get_shared_redis_sync()
    if client is None:
        msg = "Shared Redis client unavailable"
        raise RuntimeError(msg)
    return client


class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, batch_first=False)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.output_projection(x[:, -1, :])


class AIStrategyGenerator:
    def __init__(self) -> None:
        self.redis_client = _redis_from_env()
        self.running = False
        self.models: dict[str, nn.Module] = {}
        self.scalers: dict[str, MinMaxScaler] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_configs = {
            "lstm": {
                "input_size": 10,
                "hidden_size": 128,
                "num_layers": 3,
                "output_size": 3,
                "sequence_length": 60,
            },
            "transformer": {
                "input_size": 10,
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "output_size": 3,
                "sequence_length": 60,
            },
        }
        try:
            self.redis_client.ping()
            logger.info(f"[{EXCHANGE_ID}] redis connected")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] redis connection failed: {e}")

    async def start(self):
        logger.info(f"[{EXCHANGE_ID}] starting generator")
        self.running = True
        await self.generate_strategies()

    async def generate_strategies(self):
        logger.info(f"[{EXCHANGE_ID}] strategy loop started")
        while self.running:
            try:
                item = self.redis_client.lpop(QUEUE_KEY)
                if item:
                    try:
                        request_data = json.loads(item)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        request_data = {}
                    await self.process_strategy_request(request_data)
                else:
                    await self.generate_periodic_strategies()
                await asyncio.sleep(300)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"[{EXCHANGE_ID}] loop error: {e}")
                await asyncio.sleep(600)

    async def process_strategy_request(self, request_data: dict[str, Any]):
        # Validate required fields before entering try block
        strategy_type = str(request_data.get("type") or request_data.get("strategy_type") or "lstm")
        symbol_input = request_data.get("symbol")
        if not symbol_input:
            msg = "symbol is required in request_data - no fallback/hardcoded symbol"
            raise ValueError(msg)
        symbol = _validate_symbol(symbol_input)

        try:
            parameters = dict(request_data.get("parameters") or {})
            logger.info(f"[{EXCHANGE_ID}] generating {strategy_type} for {symbol}")
            strategy = await self.create_ai_strategy(strategy_type, symbol, parameters)
            if strategy:
                await self.store_strategy(strategy)
                await self.publish_strategy(strategy)
                logger.info(f"[{EXCHANGE_ID}] generated strategy {strategy['id']}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] process request failed: {e}")

    async def generate_periodic_strategies(self):
        try:
            # Use all Top-10 symbols from trading_universe (live data)
            symbols = list(TRADING_SYMBOLS)
            strategy_types = ["lstm", "transformer"]
            for sym in symbols:
                for st in strategy_types:
                    if await self.should_generate_strategy(sym, st):
                        strategy = await self.create_ai_strategy(st, sym)
                        if strategy:
                            await self.store_strategy(strategy)
                            await self.publish_strategy(strategy)
                            logger.info(f"[{EXCHANGE_ID}] generated periodic {strategy['id']}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] periodic generation failed: {e}")

    async def create_ai_strategy(self, strategy_type: str, symbol: str, parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        try:
            data = await self.get_historical_data(symbol)
            if data.empty:
                logger.warning(f"[{EXCHANGE_ID}] no data for {symbol}")
                return None
            X, y = self.prepare_features(data)
            if X.size == 0 or y.size == 0:
                logger.warning(f"[{EXCHANGE_ID}] feature prep failed for {symbol}")
                return None
            model, scaler, metrics = await self.train_model(strategy_type, X, y, parameters or {})
            if model is None or scaler is None:
                return None
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            model_path = f"models/{strategy_type}_{symbol}_{ts}.pth"
            scaler_path = f"scalers/{strategy_type}_{symbol}_{ts}.pkl"
            strategy = {
                "id": f"AI_{strategy_type.upper()}_{ts}",
                "name": f"AI {strategy_type.upper()} Strategy",
                "type": strategy_type,
                "symbol": symbol,
                "model_type": strategy_type,
                "status": "ACTIVE",
                "created_at": _now_iso(),
                "parameters": self.generate_strategy_config(strategy_type, parameters or {}),
                "performance": metrics,
                "model_path": model_path,
                "scaler_path": scaler_path,
            }
            await self.save_model(model, scaler, model_path, scaler_path)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] create strategy failed: {e}")
            return None
        else:
            return strategy

    async def train_model(self, strategy_type: str, X: np.ndarray, y: np.ndarray, parameters: dict[str, Any]) -> tuple[Any, Any, dict[str, float]]:
        try:
            if len(X) == 0 or len(y) == 0:
                return (
                    None,
                    None,
                    {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
                )
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            seq_len = self.model_configs[strategy_type]["sequence_length"]
            X_seq, y_seq = self.create_sequences(X_scaled, y, seq_len)
            if X_seq.size == 0 or y_seq.size == 0:
                return (
                    None,
                    None,
                    {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
                )
            split = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]
            if strategy_type == "lstm":
                cfg = self.model_configs["lstm"]
                model = LSTMPredictor(
                    cfg["input_size"],
                    cfg["hidden_size"],
                    cfg["num_layers"],
                    cfg["output_size"],
                )
            elif strategy_type == "transformer":
                cfg = self.model_configs["transformer"]
                model = TransformerPredictor(
                    cfg["input_size"],
                    cfg["d_model"],
                    cfg["nhead"],
                    cfg["num_layers"],
                    cfg["output_size"],
                )
            else:
                return (
                    None,
                    None,
                    {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
                )
            model.to(self.device)
            lr = float(parameters.get("learning_rate", 0.001))
            epochs = int(parameters.get("epochs", 30))
            batch_size = int(parameters.get("batch_size", 64))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_ds = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
            # test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))  # Unused
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            model.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb_tensor = xb.to(self.device)
                    yb_tensor = yb.to(self.device)
                    optimizer.zero_grad()
                    out = model(xb_tensor)
                    loss = criterion(out, yb_tensor)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                Xtest_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                logits = model(Xtest_t)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            acc = float(accuracy_score(y_test, preds))
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
            metrics = {
                "accuracy": acc,
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
            }
            logger.info(f"[{EXCHANGE_ID}] trained {strategy_type} acc={acc:.3f} f1={float(f1):.3f}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] train error: {e}")
            return (
                None,
                None,
                {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0},
            )
        else:
            return model, scaler, metrics

    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            if len(X) <= sequence_length:
                return np.array([]), np.array([])
            seqs: list[np.ndarray] = []
            labels: list[int] = []
            for i in range(sequence_length, len(X)):
                seqs.append(X[i - sequence_length : i])
                labels.append(int(y[i]))
            return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.int64)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] sequence error: {e}")
            return np.array([]), np.array([])

    def generate_strategy_config(self, strategy_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_type": strategy_type,
            "sequence_length": self.model_configs[strategy_type]["sequence_length"],
            "confidence_threshold": float(parameters.get("confidence_threshold", 0.7)),
            "position_size": float(parameters.get("position_size", 0.1)),
            "stop_loss": float(parameters.get("stop_loss", 0.05)),
            "take_profit": float(parameters.get("take_profit", 0.1)),
            "max_positions": int(parameters.get("max_positions", 3)),
        }

    async def save_model(self, model: nn.Module, scaler: MinMaxScaler, model_path: str, scaler_path: str):
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            joblib_dump(scaler, scaler_path)
            logger.info(f"[{EXCHANGE_ID}] saved model {model_path}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] save model failed: {e}")

    async def should_generate_strategy(self, symbol: str, strategy_type: str) -> bool:
        try:
            key = f"last_strategy_generation:{symbol}:{strategy_type}"
            last = self.redis_client.get(key)
            if last is None:
                return True
            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return True
            return (datetime.now(timezone.utc) - last_dt).total_seconds() > 86400
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] should_generate error: {e}")
            return False

    async def store_strategy(self, strategy: dict[str, Any]):
        try:
            self.redis_client.set(f"ai_strategy:{strategy['id']}", json.dumps(strategy), ex=86400)
            self.redis_client.lpush("ai_strategies", strategy["id"])
            self.redis_client.ltrim("ai_strategies", 0, 99)
            self.redis_client.set(
                f"last_strategy_generation:{strategy['symbol']}:{strategy['type']}",
                _now_iso(),
                ex=86400,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] store strategy failed: {e}")

    async def publish_strategy(self, strategy: dict[str, Any]):
        try:
            self.redis_client.publish("ai_strategies", json.dumps(strategy))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] publish failed: {e}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    async def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data for training from live sources only"""
        # All Live Data, No Fallback/Hardcoded Data
        try:
            # Convert symbol to CCXT format for live data fetching
            ccxt_symbol = _to_ccxt_symbol(symbol)

            # Try to get live data from canonical cache or market data service
            try:
                cache = get_persistent_cache()
                if cache:
                    # Get OHLCV data from cache (live data)
                    ohlcv = cache.get_ohlcv(ccxt_symbol, timeframe="1h", limit=600)
                    if ohlcv and len(ohlcv) > 0:
                        data = []
                        for candle in ohlcv:
                            if len(candle) >= 5:
                                data.append(
                                    {
                                        "timestamp": datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                                        "open": float(candle[1]),
                                        "high": float(candle[2]),
                                        "low": float(candle[3]),
                                        "close": float(candle[4]),
                                        "volume": float(candle[5]) if len(candle) > 5 else 0.0,
                                    }
                                )
                        if data:
                            df = pd.DataFrame(data)
                            return df.set_index("timestamp")
            except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
                # Cache not available, try direct API call
                pass

            # Fallback to direct API call if cache unavailable
            try:
                limiter = await BinanceWeightLimiter.create()
                client = BinanceREST(limiter)
                klines = await client.klines(ccxt_symbol, "1h", 600)
                if klines and len(klines) > 0:
                    data = []
                    for kline in klines:
                        try:
                            if len(kline) >= 5:
                                data.append(
                                    {
                                        "timestamp": datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                                        "open": float(kline[1]),
                                        "high": float(kline[2]),
                                        "low": float(kline[3]),
                                        "close": float(kline[4]),
                                        "volume": float(kline[5]) if len(kline) > 5 else 0.0,
                                    }
                                )
                        except (ValueError, IndexError, TypeError):
                            continue
                    if data:
                        df = pd.DataFrame(data)
                        return df.set_index("timestamp")
            except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
                logger.exception(f"[{EXCHANGE_ID}] Failed to fetch live historical data for {symbol}: {e}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] get_historical_data failed for {symbol}: {e}")
            msg = f"Failed to get historical data for {symbol}: {e}"
            raise RuntimeError(msg) from e

        # No live data available - raise error instead of returning empty DataFrame
        # Move raise outside try block to avoid TRY301
        msg = f"No live historical data available for {symbol} - all data sources failed"
        raise RuntimeError(msg)

    def prepare_features(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels from historical data"""
        try:
            if data.empty or "close" not in data.columns:
                return np.array([]), np.array([])

            closes = data["close"]
            rsi = self.calculate_rsi(closes)
            macd = self.calculate_macd(closes)
            upper, middle, lower = self.calculate_bollinger_bands(closes)

            # Create feature matrix
            features = pd.DataFrame(
                {
                    "close": closes,
                    "rsi": rsi,
                    "macd": macd,
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "volume": data.get("volume", pd.Series([0.0] * len(closes))),
                    "high": data.get("high", closes),
                    "low": data.get("low", closes),
                    "open": data.get("open", closes),
                }
            )

            # Fill NaN values
            features = features.bfill().ffill().fillna(0.0)

            # Create labels: 0 = sell, 1 = hold, 2 = buy
            # Simple strategy: buy if price goes up 2% in next 5 periods, sell if down 2%
            labels = []
            for i in range(len(closes)):
                if i + 5 >= len(closes):
                    labels.append(1)  # hold
                else:
                    future_return = (closes.iloc[i + 5] - closes.iloc[i]) / closes.iloc[i]
                    if future_return > 0.02:
                        labels.append(2)  # buy
                    elif future_return < -0.02:
                        labels.append(0)  # sell
                    else:
                        labels.append(1)  # hold

            X = features.to_numpy().astype(np.float32)
            y = np.array(labels, dtype=np.int64)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] prepare_features failed: {e}")
            return np.array([]), np.array([])
        else:
            return X, y

    async def stop(self):
        logger.info(f"[{EXCHANGE_ID}] stopping generator")
        self.running = False


async def main():
    generator = AIStrategyGenerator()

    async def run_generator():
        try:
            await generator.start()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[{EXCHANGE_ID}] generator error: {e}")

    bg_task = await task_manager.create_task(run_generator(), name="ai_strategy_generator:run_generator")
    port = int(os.getenv("SERVICE_PORT", "8000"))
    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info(f"[{EXCHANGE_ID}] interrupt received")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[{EXCHANGE_ID}] main error: {e}")
    finally:
        await generator.stop()
        bg_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bg_task


if __name__ == "__main__":
    asyncio.run(main())
