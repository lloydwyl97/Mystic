"""
AI Auto-Retrain Service
Automatic model retraining and optimization system
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from backend.config.redis_config import get_shared_redis_sync
from backend.utils.path_helpers import (
    ensure_model_directories,
    get_model_file_path,
    get_scaler_file_path,
)
from utils.redis_helpers import to_str, to_str_list

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------
load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))


# ---------------------------------------------------------------------
# Simple classifier heads
# ---------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)  # out: (batch, seq, hidden)
        last = out[:, -1, :]  # (batch, hidden)
        return self.fc(last)  # (batch, classes)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        x = self.input_proj(x)  # (batch, seq, d_model)
        enc = self.encoder(x)  # (batch, seq, d_model)
        last = enc[:, -1, :]  # (batch, d_model)
        return self.out(last)  # (batch, classes)


# ---------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------
class AutoRetrainService:
    def __init__(self) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        self.redis_client = get_shared_redis_sync()
        if self.redis_client is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Retraining parameters
        self.retrain_threshold = 0.05  # relative degradation
        self.min_data_points = 1000
        self.retrain_interval_hours = 24
        self.performance_window_days = 7

        ensure_model_directories()

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

    async def start(self) -> None:
        logger.info("Starting Auto-Retrain Service")
        self.running = True
        await self.monitor_and_retrain()

    async def stop(self) -> None:
        logger.info("Stopping Auto-Retrain Service")
        self.running = False

    async def monitor_and_retrain(self) -> None:
        logger.info("Monitoring model performance")
        while self.running:
            try:
                active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))
                for model_id in active_models:
                    await self.check_model_performance(model_id)

                retrain_request = to_str(self.redis_client.lpop("retrain_queue"))
                if retrain_request:
                    request_data = json.loads(retrain_request)
                    await self.process_retrain_request(request_data)

                await asyncio.sleep(300)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Monitor loop error: %s", e)
                await asyncio.sleep(600)

    async def check_model_performance(self, model_id: str) -> None:
        try:
            model_data = self.redis_client.get(f"ai_strategy:{model_id}")
            if not model_data:
                return

            model = json.loads(model_data)
            recent = await self.get_recent_performance(model_id)
            if not recent:
                return

            if await self.should_retrain(model, recent):
                logger.info("Model %s flagged for retrain", model_id)
                retrain_request = {
                    "model_id": model_id,
                    "reason": "performance_degradation",
                    "current_performance": recent,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
                self.redis_client.lpush("retrain_queue", json.dumps(retrain_request))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("check_model_performance error: %s", e)

    async def get_recent_performance(self, model_id: str) -> dict[str, Any] | None:
        try:
            _end = datetime.now(tz=timezone.utc)
            _start = _end - timedelta(days=self.performance_window_days)

            perf_key = f"performance:{model_id}"
            perf_data = to_str_list(self.redis_client.lrange(perf_key, 0, -1))
            if not perf_data:
                return None

            accuracy: list[float] = []
            total_return: list[float] = []
            sharpe_ratio: list[float] = []
            win_rate: list[float] = []
            total_trades: list[int] = []
            for entry in perf_data:
                try:
                    record = json.loads(entry)
                    ts = record.get("timestamp")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(str(ts))
                            if not (_start <= dt <= _end):
                                continue
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            pass
                    accuracy.append(float(record.get("accuracy", 0)))
                    total_return.append(float(record.get("total_return", 0)))
                    sharpe_ratio.append(float(record.get("sharpe_ratio", 0)))
                    win_rate.append(float(record.get("win_rate", 0)))
                    total_trades.append(int(record.get("total_trades", 0)))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue

            if not accuracy:
                return None

            return {
                "accuracy": float(np.mean(accuracy)),
                "total_return": float(np.mean(total_return)),
                "sharpe_ratio": float(np.mean(sharpe_ratio)),
                "win_rate": float(np.mean(win_rate)),
                "total_trades": int(np.sum(total_trades)),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("get_recent_performance error: %s", e)
            return None

    async def should_retrain(self, model: dict[str, Any], recent: dict[str, Any]) -> bool:
        try:
            baseline = model.get("performance", {})
            if not baseline:
                return False

            base_acc = float(baseline.get("accuracy", 0))
            curr_acc = float(recent.get("accuracy", 0))
            if base_acc - curr_acc > self.retrain_threshold:
                return True

            base_ret = float(baseline.get("total_return", 0))
            curr_ret = float(recent.get("total_return", 0))
            if base_ret - curr_ret > self.retrain_threshold:
                return True

            last_retrain = model.get("last_retrain")
            if last_retrain:
                try:
                    last_dt = datetime.fromisoformat(last_retrain)
                    # Do not retrain too frequently
                    if (datetime.now(tz=timezone.utc) - last_dt).total_seconds() < (self.retrain_interval_hours * 3600):
                        return False
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass

            # check minimum data requirement from recent stats
            total_trades = int(recent.get("total_trades", 0))
            result = not (total_trades < self.min_data_points)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("should_retrain error: %s", e)
            return False
        else:
            return result

    async def process_retrain_request(self, request_data: dict[str, Any]) -> None:
        try:
            model_id = request_data.get("model_id")
            if not model_id:
                logger.error("process_retrain_request missing model_id")
                return

            logger.info("Processing retrain request for model %s", model_id)

            # Fetch model/strategy metadata
            strategy_raw = self.redis_client.get(f"ai_strategy:{model_id}")
            if not strategy_raw:
                logger.error("No strategy found for model %s", model_id)
                return
            strategy = json.loads(strategy_raw)

            # Attempt to retrieve training data from Redis or strategy payload
            data_key_candidates = [
                f"training_data:{model_id}",
                f"historical:{model_id}",
                f"historical_data:{model_id}",
            ]
            df = None
            for key in data_key_candidates:
                raw = self.redis_client.get(key)
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            df = pd.DataFrame(parsed)
                        elif isinstance(parsed, dict):
                            # single dict maybe with 'data' key
                            df = pd.DataFrame(parsed["data"]) if "data" in parsed and isinstance(parsed["data"], list) else pd.DataFrame([parsed])
                        break
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        continue

            # fallback to any inline training_data in strategy
            if df is None and "training_data" in strategy:
                try:
                    td = strategy.get("training_data")
                    if isinstance(td, list):
                        df = pd.DataFrame(td)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    df = None

            if df is None or df.empty:
                logger.error("No training data available for model %s", model_id)
                return

            # Prepare features/labels
            X, y = self.prepare_features(df)
            if X.size == 0 or y.size == 0:
                logger.error("Insufficient processed features for model %s", model_id)
                return

            # split train/test
            split_idx = int(0.8 * X.shape[0])
            if split_idx < 1 or X.shape[0] - split_idx < 1:
                logger.error("Not enough data to split train/test for model %s", model_id)
                return
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # scaling
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # sequence length from config
            strategy_type = strategy.get("type", "lstm")
            cfg = self.model_configs.get(strategy_type, self.model_configs["lstm"])
            seq_len = int(cfg.get("sequence_length", 60))

            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, seq_len)
            X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test, seq_len)

            if X_train_seq.size == 0 or X_test_seq.size == 0:
                logger.error("Not enough sequence data for model %s", model_id)
                return

            parameters = request_data.get("parameters", {})
            model, used_scaler = await self.train_model(
                strategy_type=strategy_type,
                X_train=X_train_seq,
                y_train=y_train_seq,
                X_test=X_test_seq,
                y_test=y_test_seq,
                scaler=scaler,
                parameters=parameters,
            )

            if model is None or used_scaler is None:
                logger.error("Training failed for model %s", model_id)
                return

            eval_metrics = await self.evaluate_model(model, used_scaler, df)

            # Prepare file paths and save
            model_path = get_model_file_path(model_id)
            scaler_path = get_scaler_file_path(model_id)
            model_data = {
                "model_id": model_id,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "performance": eval_metrics,
                "last_retrain": datetime.now(tz=timezone.utc).isoformat(),
                "type": strategy.get("type", "lstm"),
            }

            await self.save_model(model, used_scaler, model_data)
            await self.update_model(model_id, model_data)
            logger.info("Retraining complete for model %s", model_id)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("process_retrain_request error: %s", e)

    def prepare_features(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare features X and labels y from a DataFrame.
        Expects a 'label' or 'target' column for supervised learning.
        If not present, returns empty arrays.
        """
        try:
            if data is None or data.empty:
                return np.array([]), np.array([])

            df = data.copy()
            # Standardize column names
            if "label" in df.columns:
                y = df["label"].to_numpy()
                X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).fillna(0).to_numpy()
                return X, y
            if "target" in df.columns:
                y = df["target"].to_numpy()
                X = df.drop(columns=["target"]).select_dtypes(include=[np.number]).fillna(0).to_numpy()
                return X, y

            # If 'close' present we can build a simple supervised task predicting next-step movement
            if "close" in df.columns:
                prices = pd.Series(df["close"].astype(float)).reset_index(drop=True)
                # compute simple features
                rsi = self.calculate_rsi(prices).fillna(0)
                macd = self.calculate_macd(prices).fillna(0)
                upper, sma, lower = self.calculate_bollinger_bands(prices)
                returns = prices.pct_change().fillna(0)

                Xdf = pd.DataFrame(
                    {
                        "close": prices,
                        "rsi": rsi,
                        "macd": macd,
                        "sma": sma,
                        "upper": upper,
                        "lower": lower,
                        "returns": returns,
                    }
                ).fillna(0)
                # label: 0 = down, 1 = neutral/small, 2 = up (based on future return)
                future_returns = prices.shift(-1).pct_change().fillna(0)
                labels = []
                for fr in future_returns:
                    if fr > 0.001:
                        labels.append(2)
                    elif fr < -0.001:
                        labels.append(0)
                    else:
                        labels.append(1)
                X = Xdf.to_numpy()
                y = np.array(labels[: X.shape[0]])
                return X, y

            # otherwise no usable features
            return np.array([]), np.array([])
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("prepare_features error: %s", e)
            return np.array([]), np.array([])

    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D feature matrix X and label vector y into sequences of length seq_len.
        The label for each sequence is the label at the last timestep of the sequence.
        """
        try:
            if X is None or y is None:
                return np.array([]), np.array([])
            if len(X) < seq_len or len(y) < seq_len:
                return np.array([]), np.array([])
            seqs = []
            labs = []
            for i in range(len(X) - seq_len + 1):
                seq = X[i : i + seq_len]
                label = y[i + seq_len - 1]
                seqs.append(seq)
                labs.append(label)
            return np.array(seqs), np.array(labs)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("create_sequences error: %s", e)
            return np.array([]), np.array([])

    async def train_model(
        self,
        strategy_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: MinMaxScaler,
        parameters: dict | None = None,
    ) -> tuple[nn.Module | None, MinMaxScaler | None]:
        try:
            cfg = self.model_configs.get(strategy_type, self.model_configs["lstm"])

            # infer input_size from features dimension
            if X_train.ndim != 3:
                logger.error("train_model expected 3D X_train (samples, seq, features), got %s", X_train.shape)
                return None, None
            input_size = X_train.shape[2]

            if strategy_type == "lstm":
                model = LSTMClassifier(
                    input_size=input_size,
                    hidden_size=cfg["hidden_size"],
                    num_layers=cfg["num_layers"],
                    output_size=cfg["output_size"],
                )
            elif strategy_type == "transformer":
                model = TransformerClassifier(
                    input_size=input_size,
                    d_model=cfg["d_model"],
                    nhead=cfg["nhead"],
                    num_layers=cfg["num_layers"],
                    output_size=cfg["output_size"],
                )
            else:
                return None, None

            model.to(self.device)

            learning_rate = (parameters or {}).get("learning_rate", 0.001)
            epochs = int((parameters or {}).get("epochs", 50))
            batch_size = int((parameters or {}).get("batch_size", 32))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
            y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

            model.train()
            n = X_train_t.shape[0]
            for epoch in range(epochs):
                perm = torch.randperm(n, device=self.device)
                X_shuffled = X_train_t[perm]
                y_shuffled = y_train_t[perm]
                epoch_loss = 0.0

                for i in range(0, n, batch_size):
                    xb = X_shuffled[i : i + batch_size]
                    yb = y_shuffled[i : i + batch_size]

                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if epoch % 10 == 0 or epoch == epochs - 1:
                    logger.info("Epoch %d/%d | loss=%.6f", epoch + 1, epochs, epoch_loss)

            # quick eval
            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.device)
                logits = model(X_test_t)
                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(y_test_t.cpu().numpy(), preds.cpu().numpy())
                logger.info("Retrained model accuracy=%.4f", acc)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("train_model error: %s", e)
            return None, None
        else:
            return model, scaler

    async def evaluate_model(self, model: nn.Module, scaler: MinMaxScaler, data: pd.DataFrame) -> dict[str, Any]:
        try:
            X, y = self.prepare_features(data)
            if X.size == 0 or y.size == 0:
                return {}

            X_scaled = scaler.transform(X)
            seq_len = 60
            X_seq, y_seq = self.create_sequences(X_scaled, y, seq_len)
            if X_seq.size == 0:
                return {}

            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
                logits = model(X_t)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            acc = accuracy_score(y_seq, preds)
            prec = precision_score(y_seq, preds, average="weighted", zero_division=0)
            rec = recall_score(y_seq, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_seq, preds, average="weighted", zero_division=0)

            return {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("evaluate_model error: %s", e)
            return {}

    async def save_model(self, model: nn.Module, scaler: MinMaxScaler, model_data: dict[str, Any]) -> None:
        try:
            ensure_model_directories()
            torch.save(model.state_dict(), model_data["model_path"])
            joblib.dump(scaler, model_data["scaler_path"])
            logger.info(
                "Saved retrained model | model=%s scaler=%s",
                model_data["model_path"],
                model_data["scaler_path"],
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("save_model error: %s", e)

    async def update_model(self, model_id: str, new_model_data: dict[str, Any]) -> None:
        try:
            self.redis_client.set(f"ai_strategy:{model_id}", json.dumps(new_model_data), ex=86400)
            self.redis_client.set(f"last_retrain:{model_id}", datetime.now(tz=timezone.utc).isoformat(), ex=86400)
            await self.broadcast_model_metrics()
            await self.broadcast_retrain_status()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("update_model error: %s", e)

    async def broadcast_model_metrics(self) -> None:
        try:
            active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))
            models_data = []
            for model_id in active_models:
                data = self.redis_client.get(f"ai_strategy:{model_id}")
                if data:
                    models_data.append(json.loads(data))

            payload = {"models": models_data, "timestamp": datetime.now(tz=timezone.utc).isoformat()}
            self.redis_client.set("model_metrics", json.dumps(payload), ex=300)
            self.redis_client.publish("model_metrics", json.dumps(payload))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("broadcast_model_metrics error: %s", e)

    async def broadcast_retrain_status(self) -> None:
        try:
            queue_data = to_str_list(self.redis_client.lrange("retrain_queue", 0, -1))
            queue = []
            for item in queue_data:
                try:
                    queue.append(json.loads(item))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue

            status = {
                "currently_retraining": None,
                "retrain_progress": 0.0,
                "estimated_completion": None,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            for item in queue:
                if item.get("status") == "retraining":
                    status["currently_retraining"] = item.get("model_id")
                    status["retrain_progress"] = float(item.get("progress", 0.0))
                    status["estimated_completion"] = item.get("estimated_completion")
                    break

            payload = {
                "queue": queue,
                "status": status,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            self.redis_client.set("retrain_status", json.dumps(payload), ex=300)
            self.redis_client.publish("retrain_status", json.dumps(payload))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("broadcast_retrain_status error: %s", e)

    # -----------------------------------------------------------------
    # Technical indicators
    # -----------------------------------------------------------------
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = (sma + (std * std_dev)).fillna(0)
        lower = (sma - (std * std_dev)).fillna(0)
        return upper, sma.fillna(0), lower


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
async def main() -> None:
    service = AutoRetrainService()
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Service main error: %s", e)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
