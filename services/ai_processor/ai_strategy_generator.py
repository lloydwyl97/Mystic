"""
AI Strategy Generator Service
Advanced neural network-based strategy generation and signal optimization
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from backend.config.redis_config import get_shared_redis_sync
from utils.redis_helpers import to_str

# Live service imports - only imported when needed to avoid circular dependencies
try:
    from backend.services.binance_rest_client import BinanceREST
    from backend.utils.binance_weight_limiter import BinanceWeightLimiter
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    BinanceREST = None  # type: ignore[assignment,misc]
    BinanceWeightLimiter = None  # type: ignore[assignment,misc]

try:
    from backend.services.live_market_data import live_market_data_service
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    live_market_data_service = None  # type: ignore[assignment,misc]

# Load environment variables from project root (single source of truth)
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=str(root_env))


class LSTMPredictor(nn.Module):
    """LSTM Neural Network for price prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class TransformerPredictor(nn.Module):
    """Transformer-based model for market analysis"""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4),
            num_layers,
        )
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        return self.output_projection(x[:, -1, :])  # Take last sequence output


class AIStrategyGenerator:
    def __init__(self):
        """Initialize AI Strategy Generator"""
        self.redis_client = get_shared_redis_sync()
        if self.redis_client is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.running = False
        self.models = {}
        self.scalers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configurations
        self.model_configs = {
            "lstm": {
                "input_size": 10,
                "hidden_size": 128,
                "num_layers": 3,
                "output_size": 3,  # Buy, Hold, Sell
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

    async def start(self):
        """Start the AI Strategy Generator"""
        print("Starting AI Strategy Generator...")
        self.running = True

        # Start strategy generation
        await self.generate_strategies()

    async def generate_strategies(self):
        """Generate AI strategies continuously"""
        print("🧠 Starting AI strategy generation...")

        while self.running:
            try:
                # Check for strategy generation requests
                request = to_str(self.redis_client.lpop("ai_strategy_queue"))

                if request:
                    request_data = json.loads(request)
                    await self.process_strategy_request(request_data)
                else:
                    # Generate periodic strategies
                    await self.generate_periodic_strategies()

                await asyncio.sleep(300)  # Check every 5 minutes

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                print(f"ERROR: Error in strategy generation: {e}")
                await asyncio.sleep(600)

    async def process_strategy_request(self, request_data: dict[str, Any]):
        """Process individual strategy generation request"""
        try:
            strategy_type = request_data.get("type", "lstm")
            symbol = request_data.get("symbol", "BTC/USDT")
            parameters = request_data.get("parameters", {})

            print(f"Generating {strategy_type} strategy for {symbol}")

            # Generate strategy
            strategy = await self.create_ai_strategy(strategy_type, symbol, parameters)

            if strategy:
                # Store strategy
                await self.store_strategy(strategy)

                # Publish strategy
                await self.publish_strategy(strategy)

                print(f"SUCCESS: Generated strategy: {strategy['id']}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"ERROR: Error processing strategy request: {e}")

    async def generate_periodic_strategies(self):
        """Generate strategies periodically"""
        try:
            symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"]
            strategy_types = ["lstm", "transformer"]

            for symbol in symbols:
                for strategy_type in strategy_types:
                    # Check if we need a new strategy
                    if await self.should_generate_strategy(symbol, strategy_type):
                        strategy = await self.create_ai_strategy(strategy_type, symbol)
                        if strategy:
                            await self.store_strategy(strategy)
                            await self.publish_strategy(strategy)
                            print(f"SUCCESS: Generated periodic strategy: {strategy['id']}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"ERROR: Error in periodic strategy generation: {e}")

    async def create_ai_strategy(
        self,
        strategy_type: str,
        symbol: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create an AI strategy"""
        try:
            # Get historical data
            data = await self.get_historical_data(symbol)

            if data.empty:
                return None

            # Prepare features
            features = self.prepare_features(data)

            # Train model
            model, scaler = await self.train_model(strategy_type, features, parameters)

            if model is None:
                return None

            # Generate strategy configuration
            strategy_config = self.generate_strategy_config(strategy_type, parameters)

            # Create strategy object
            strategy = {
                "id": (f"AI_{strategy_type.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"),
                "name": f"AI {strategy_type.upper()} Strategy",
                "type": strategy_type,
                "symbol": symbol,
                "model_type": strategy_type,
                "status": "ACTIVE",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parameters": strategy_config,
                "performance": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "model_path": (f"models/{strategy_type}_{symbol.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.pth"),
                "scaler_path": (f"scalers/{strategy_type}_{symbol.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.pkl"),
            }

            # Save model and scaler
            await self.save_model(model, scaler, strategy)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"ERROR: Error creating AI strategy: {e}")
            return None
        else:
            return strategy

    async def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data for training from live sources only"""
        # Use live market data instead of generated data
        if BinanceREST is None or BinanceWeightLimiter is None:
            print("BinanceREST or BinanceWeightLimiter not available")
            return pd.DataFrame()

        try:
            # Get live historical data from Binance
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)

            # Get klines data (1 hour intervals for last 365 days)
            klines = await client.klines(symbol, "1h", 8760)  # 365 days * 24 hours

            if not klines or len(klines) == 0:
                print(f"No live historical data available for {symbol}")
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
                print(f"Failed to parse live historical data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            return df.set_index("timestamp")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error getting live historical data for {symbol}: {e}")
            # Try fallback to live market data service
            try:
                if live_market_data_service is None:
                    print("live_market_data_service not available")
                    return pd.DataFrame()

                coin_data = await live_market_data_service.get_historical_data(symbol, "1h", 8760)

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
                        df = pd.DataFrame(data)
                        return df.set_index("timestamp")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as fallback_error:
                print(f"Fallback also failed for {symbol}: {fallback_error}")

            return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any]]:
        """Prepare features for model training"""
        # Constants for labeling thresholds
        buy_threshold = 0.01  # 1% gain
        sell_threshold = -0.01  # 1% loss

        try:
            # Calculate technical indicators
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["sma_50"] = data["close"].rolling(window=50).mean()
            data["rsi"] = self.calculate_rsi(data["close"])
            data["macd"] = self.calculate_macd(data["close"])
            data["bb_upper"], data["bb_middle"], data["bb_lower"] = self.calculate_bollinger_bands(data["close"])
            data["volume_sma"] = data["volume"].rolling(window=20).mean()
            data["price_change"] = data["close"].pct_change()
            data["volatility"] = data["price_change"].rolling(window=20).std()

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

            # Create labels (simplified: 0=Hold, 1=Buy, 2=Sell)
            labels = np.zeros(len(features))

            # Simple labeling logic
            for i in range(60, len(features)):
                future_return = (data["close"].iloc[i + 1] - data["close"].iloc[i]) / data["close"].iloc[i] if i + 1 < len(data) else 0

                if future_return > buy_threshold:
                    labels[i] = 1  # Buy
                elif future_return < sell_threshold:
                    labels[i] = 2  # Sell
                else:
                    labels[i] = 0  # Hold
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error preparing features: {e}")
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
            x_data, y_data = features

            # Normalize features
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x_data)

            # Create sequences
            sequence_length = self.model_configs[strategy_type]["sequence_length"]
            x_sequences, y_sequences = self.create_sequences(x_scaled, y_data, sequence_length)

            if len(x_sequences) == 0:
                return None, None

            # Split data
            split_idx = int(0.8 * len(x_sequences))
            x_train, x_test = x_sequences[:split_idx], x_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

            # Create model
            if strategy_type == "lstm":
                config = self.model_configs["lstm"]
                model = LSTMPredictor(
                    input_size=config["input_size"],
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    output_size=config["output_size"],
                )
            elif strategy_type == "transformer":
                config = self.model_configs["transformer"]
                model = TransformerPredictor(
                    input_size=config["input_size"],
                    d_model=config["d_model"],
                    nhead=config["nhead"],
                    num_layers=config["num_layers"],
                    output_size=config["output_size"],
                )
            else:
                return None, None

            model.to(self.device)

            # Training parameters
            learning_rate = parameters.get("learning_rate", 0.001) if parameters else 0.001
            epochs = parameters.get("epochs", 50) if parameters else 50
            parameters.get("batch_size", 32) if parameters else 32

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
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Evaluate model
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                accuracy = accuracy_score(y_test_tensor.cpu(), test_predictions.cpu())
                print(f"Model accuracy: {accuracy:.4f}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error training model: {e}")
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

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

    def generate_strategy_config(self, strategy_type: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate strategy configuration"""
        return {
            "model_type": strategy_type,
            "sequence_length": self.model_configs[strategy_type]["sequence_length"],
            "confidence_threshold": (parameters.get("confidence_threshold", 0.7) if parameters else 0.7),
            "position_size": (parameters.get("position_size", 0.1) if parameters else 0.1),
            "stop_loss": (parameters.get("stop_loss", 0.05) if parameters else 0.05),
            "take_profit": (parameters.get("take_profit", 0.1) if parameters else 0.1),
            "max_positions": (parameters.get("max_positions", 3) if parameters else 3),
        }

    async def save_model(self, model: nn.Module, scaler: MinMaxScaler, strategy: dict[str, Any]):
        """Save model and scaler"""
        try:
            # Create directories if they don't exist
            Path("models").mkdir(parents=True, exist_ok=True)
            Path("scalers").mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(model.state_dict(), strategy["model_path"])

            # Save scaler
            joblib.dump(scaler, strategy["scaler_path"])

            print(f"SUCCESS: Saved model: {strategy['model_path']}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error saving model: {e}")

    async def should_generate_strategy(self, symbol: str, strategy_type: str) -> bool:
        """Check if we should generate a new strategy"""
        # Constants
        seconds_per_day = 86400

        try:
            # Check last strategy generation time
            last_generation = self.redis_client.get(f"last_strategy_generation:{symbol}:{strategy_type}")

            if last_generation is None:
                return True

            last_time = datetime.fromisoformat(last_generation)
            time_diff = datetime.now(timezone.utc) - last_time

            # Generate new strategy every 24 hours
            return time_diff.total_seconds() > seconds_per_day

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error checking strategy generation: {e}")
            return False

    async def store_strategy(self, strategy: dict[str, Any]):
        """Store strategy in Redis"""
        seconds_per_day = 86400
        try:
            self.redis_client.set(f"ai_strategy:{strategy['id']}", json.dumps(strategy), ex=seconds_per_day)
            self.redis_client.lpush("ai_strategies", strategy["id"])
            self.redis_client.ltrim("ai_strategies", 0, 99)  # Keep only 100 strategies

            # Update last generation time
            self.redis_client.set(
                f"last_strategy_generation:{strategy['symbol']}:{strategy['type']}",
                datetime.now(timezone.utc).isoformat(),
                ex=seconds_per_day,
            )

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error storing strategy: {e}")

    async def publish_strategy(self, strategy: dict[str, Any]):
        """Publish strategy to Redis channels"""
        try:
            self.redis_client.publish("ai_strategies", json.dumps(strategy))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            print(f"Error publishing strategy: {e}")

    # Technical indicator calculations
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    async def stop(self):
        """Stop the AI Strategy Generator"""
        print("STOP: Stopping AI Strategy Generator...")
        self.running = False


async def main():
    """Main function"""
    generator = AIStrategyGenerator()

    try:
        await generator.start()
    except KeyboardInterrupt:
        print("STOP: Received interrupt signal")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        print(f"ERROR: Error in main: {e}")
    finally:
        await generator.stop()


if __name__ == "__main__":
    asyncio.run(main())
