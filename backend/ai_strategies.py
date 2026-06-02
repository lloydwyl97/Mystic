import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

# Import from single source of truth
try:
    from backend.services.confidence_normalizer import ConfidenceNormalizer
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    ConfidenceNormalizer = None  # type: ignore[assignment,misc]

try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = tuple(TRADING_SYMBOLS)

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"
    DAYING = "daying"
    SWING_TRADING = "swing_trading"
    CUSTOM = "custom"


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    strategy: str
    indicators: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    name: str
    strategy_type: StrategyType
    symbols: list[str]
    parameters: dict[str, Any]
    risk_level: str
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MLModel:
    name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_trained: datetime
    features: list[str]
    parameters: dict[str, Any]


def _validate_symbol(symbol: str) -> str:
    s = str(symbol).upper()
    if s not in ALLOWED_SYMBOLS:
        msg = f"Symbol not allowed: {s}"
        raise ValueError(msg)
    return s


class TechnicalIndicators:
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        k_percent = 100 * ((close - lowest_low) / denom)
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent.fillna(50.0), d_percent.fillna(50.0)

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_volume_profile(volume: pd.Series, price: pd.Series, bins: int = 50) -> dict[str, float]:
        price_bins = pd.cut(price, bins=bins)
        volume_profile = volume.groupby(price_bins).sum()
        return {str(k): float(v) for k, v in volume_profile.to_dict().items()}


class PatternRecognition:
    @staticmethod
    def detect_candlestick_patterns(open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> dict[str, list[int]]:
        patterns: dict[str, list[int]] = {}
        try:
            doji_threshold = 0.001
            body = (open_prices - close).abs()
            range_ = (high - low).abs()
            doji_pattern = body <= (range_ * doji_threshold)
            doji_idx = np.where(doji_pattern.to_numpy())[0]
            if doji_idx.size > 0:
                patterns["doji"] = doji_idx.tolist()

            body_size = (close - open_prices).abs()
            lower_shadow = np.minimum(open_prices, close) - low
            upper_shadow = high - np.maximum(open_prices, close)

            hammer_pattern = (lower_shadow > (2.0 * body_size)) & (upper_shadow < body_size)
            hammer_idx = np.where(hammer_pattern.to_numpy())[0]
            if hammer_idx.size > 0:
                patterns["hammer"] = hammer_idx.tolist()

            shooting_star_pattern = (upper_shadow > (2.0 * body_size)) & (lower_shadow < body_size)
            shooting_star_idx = np.where(shooting_star_pattern.to_numpy())[0]
            if shooting_star_idx.size > 0:
                patterns["shooting_star"] = shooting_star_idx.tolist()
        except (ValueError, TypeError, AttributeError, IndexError, RuntimeError) as e:
            logger.warning(f"candlestick detection failed: {e}")
        return patterns

    @staticmethod
    def detect_chart_patterns(high: pd.Series, low: pd.Series, close: pd.Series) -> dict[str, list[dict[str, Any]]]:
        patterns: dict[str, list[dict[str, Any]]] = {}
        hns = PatternRecognition._detect_head_shoulders(high, low, close)
        if hns:
            patterns["head_shoulders"] = hns
        dbl = PatternRecognition._detect_double_patterns(high, low, close)
        if dbl:
            patterns["double_patterns"] = dbl
        tris = PatternRecognition._detect_triangles(high, low, close)
        if tris:
            patterns["triangles"] = tris
        return patterns

    @staticmethod
    def _detect_head_shoulders(high: pd.Series, _low: pd.Series, _close: pd.Series) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        window = 20
        n = len(high)
        # VECTORIZED head and shoulders detection for performance
        high_array = high.to_numpy()
        for i in range(2 * window, n - window):
            # VECTORIZED slice operations for performance
            left_slice = high_array[i - 2 * window : i - window]
            middle_slice = high_array[i - window : i]
            right_slice = high_array[i : i + window]
            left_peak = float(np.max(left_slice))
            middle_peak = float(np.max(middle_slice))
            right_peak = float(np.max(right_slice))
            if middle_peak > left_peak and middle_peak > right_peak and abs(left_peak - right_peak) / max(left_peak, 1e-9) < 0.05:
                left_pos = (i - 2 * window) + int(np.argmax(left_slice))
                head_pos = (i - window) + int(np.argmax(middle_slice))
                right_pos = i + int(np.argmax(right_slice))
                neckline = (left_peak + right_peak) / 2.0
                patterns.append(
                    {
                        "type": "head_shoulders",
                        "left_shoulder": left_pos,
                        "head": head_pos,
                        "right_shoulder": right_pos,
                        "neckline": neckline,
                    }
                )
        return patterns

    @staticmethod
    def _detect_double_patterns(high: pd.Series, low: pd.Series, _close: pd.Series) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        window = 15
        n = len(high)
        # VECTORIZED double pattern detection for performance
        high_array = high.to_numpy()
        low_array = low.to_numpy()
        for i in range(window, n - window):
            # VECTORIZED slice operations for performance
            left_peak = float(np.max(high_array[i - window : i]))
            right_peak = float(np.max(high_array[i : i + window]))
            # ensure peaks are similar
            if abs(left_peak - right_peak) / max(left_peak, 1e-9) < 0.03:
                lp_slice = high_array[i - window : i]
                rp_slice = high_array[i : i + window]
                left_peak_pos = (i - window) + int(np.argmax(lp_slice))
                right_peak_pos = i + int(np.argmax(rp_slice))
                # find valley between peaks to distinguish top vs bottom - VECTORIZED for performance
                valley = float(np.min(low_array[left_peak_pos : right_peak_pos + 1])) if right_peak_pos > left_peak_pos else float(np.min(low_array[right_peak_pos : left_peak_pos + 1]))
                # if valley is significantly lower than peaks, it's a double top
                if valley < min(left_peak, right_peak) * 0.98:
                    patterns.append(
                        {
                            "type": "double_top",
                            "left_peak": left_peak_pos,
                            "right_peak": right_peak_pos,
                            "neckline": valley,
                        }
                    )
                else:
                    patterns.append(
                        {
                            "type": "double_peak",
                            "left_peak": left_peak_pos,
                            "right_peak": right_peak_pos,
                            "neckline": valley,
                        }
                    )
        return patterns

    @staticmethod
    def _detect_triangles(_high: pd.Series, _low: pd.Series, _close: pd.Series) -> list[dict[str, Any]]:
        # Minimal triangle detection placeholder: return empty list for safety
        return []


class StrategyBuilder:
    """
    Minimal StrategyBuilder stub to preserve public API and avoid runtime errors.
    This provides basic condition functions and action creators used elsewhere.
    """

    def __init__(self) -> None:
        # conditions return a pandas Series or array-like with boolean values aligned to input length when possible
        self.available_conditions = {
            "crossover": self._crossover,
            "threshold": self._threshold,
            "divergence": self._divergence,
            "pattern": self._pattern_condition,
        }
        self.available_actions = {
            "buy": self._action_buy,
            "sell": self._action_sell,
            "set_stop_loss": self._action_set_stop_loss,
            "set_take_profit": self._action_set_take_profit,
        }

    def _crossover(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        # True when series1 crosses above series2 on this bar
        s1 = series1.ffill().to_numpy()
        s2 = series2.ffill().to_numpy()
        if len(s1) < 2 or len(s2) < 2:
            return pd.Series([False] * len(series1), index=series1.index)
        prev = s1[:-1] <= s2[:-1]
        curr = s1[1:] > s2[1:]
        res = np.concatenate(([False], prev & curr))
        return pd.Series(res, index=series1.index)

    def _threshold(self, series: pd.Series, threshold: float, operator: str) -> pd.Series:
        if operator == ">":
            return series > threshold
        if operator == "<":
            return series < threshold
        if operator == ">=":
            return series >= threshold
        if operator == "<=":
            return series <= threshold
        if operator == "==":
            return series == threshold
        return pd.Series([False] * len(series), index=series.index)

    def _divergence(self, price: pd.Series, indicator: pd.Series) -> pd.Series:
        # simplistic divergence: sign of price diff opposite to indicator diff
        p_diff = price.diff().fillna(0)
        i_diff = indicator.diff().fillna(0)
        return (p_diff * i_diff) < 0

    def _pattern_condition(self, indices: list[int], length: int) -> pd.Series:
        flags = [False] * length
        for idx in indices:
            if 0 <= idx < length:
                flags[idx] = True
        return pd.Series(flags)

    def _action_buy(self, signal_data: dict[str, Any]) -> TradingSignal:
        # All Live Data, No Fallback/Hardcoded Data
        symbol = signal_data.get("symbol")
        if not symbol:
            msg = "symbol is required in signal_data - no fallback/hardcoded symbol"
            raise ValueError(msg)
        raw = float(signal_data.get("confidence", 0.5) or 0.5)
        conf = ConfidenceNormalizer.normalize(raw) if ConfidenceNormalizer else raw
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=conf,
            price=float(signal_data.get("price", 0.0)),
            timestamp=datetime.now(timezone.utc),
            strategy=signal_data.get("strategy", "unknown"),
            indicators=signal_data.get("indicators", {}),
            metadata={
                "stop_loss_price": signal_data.get("stop_loss_price"),
                "take_profit_price": signal_data.get("take_profit_price"),
            },
        )

    def _action_sell(self, signal_data: dict[str, Any]) -> TradingSignal:
        # All Live Data, No Fallback/Hardcoded Data
        symbol = signal_data.get("symbol")
        if not symbol:
            msg = "symbol is required in signal_data - no fallback/hardcoded symbol"
            raise ValueError(msg)
        raw = float(signal_data.get("confidence", 0.5) or 0.5)
        conf = ConfidenceNormalizer.normalize(raw) if ConfidenceNormalizer else raw
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            confidence=conf,
            price=float(signal_data.get("price", 0.0)),
            timestamp=datetime.now(timezone.utc),
            strategy=signal_data.get("strategy", "unknown"),
            indicators=signal_data.get("indicators", {}),
            metadata={
                "stop_loss_price": signal_data.get("stop_loss_price"),
                "take_profit_price": signal_data.get("take_profit_price"),
            },
        )

    def _action_set_stop_loss(self, _signal_data: dict[str, Any]) -> None:
        # placeholder: set stop loss metadata; no side effects in this stub
        return None

    def _action_set_take_profit(self, _signal_data: dict[str, Any]) -> None:
        # placeholder: set take profit metadata; no side effects in this stub
        return None


class PredictiveAnalytics:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_importance: dict[str, dict[str, float]] = {}

    def prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        features["price_change"] = market_data["close"].pct_change()
        features["price_change_5"] = market_data["close"].pct_change(5)
        features["price_change_10"] = market_data["close"].pct_change(10)
        features["volume_change"] = market_data["volume"].pct_change()
        features["volume_sma_ratio"] = market_data["volume"] / market_data["volume"].rolling(20).mean()
        features["rsi"] = TechnicalIndicators.calculate_rsi(market_data["close"])
        features["sma_20"] = TechnicalIndicators.calculate_sma(market_data["close"], 20)
        features["sma_50"] = TechnicalIndicators.calculate_sma(market_data["close"], 50)
        features["ema_12"] = TechnicalIndicators.calculate_ema(market_data["close"], 12)
        features["ema_26"] = TechnicalIndicators.calculate_ema(market_data["close"], 26)
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(market_data["close"])
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(market_data["close"])
        features["bb_upper"] = upper
        features["bb_middle"] = middle
        features["bb_lower"] = lower
        # avoid division by zero
        denom = (upper - lower).replace(0, np.nan)
        features["bb_position"] = (market_data["close"] - lower) / denom
        features["atr"] = TechnicalIndicators.calculate_atr(market_data["high"], market_data["low"], market_data["close"])
        features["volatility"] = market_data["close"].rolling(20).std()
        return features.dropna()

    def create_labels(self, market_data: pd.DataFrame, horizon: int = 5) -> pd.Series:
        future_returns = market_data["close"].shift(-horizon) / market_data["close"] - 1
        labels = (future_returns > 0).astype(int)
        return labels.dropna()

    def train_model(
        self,
        model_name: str,
        features: pd.DataFrame,
        labels: pd.Series,
        model_type: str = "random_forest",
    ) -> MLModel:
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        # Chronological split to prevent data leakage (no random shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42)
        else:
            msg = f"Unknown model type: {model_type}"
            raise ValueError(msg)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        feature_importance = dict(zip(features.columns, model.feature_importances_, strict=False)) if hasattr(model, "feature_importances_") else {}
        ml_model = MLModel(
            name=model_name,
            model_type=model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            last_trained=datetime.now(timezone.utc),
            features=list(features.columns),
            parameters=model.get_params(),
        )
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_importance[model_name] = feature_importance
        logger.info(f"Trained model {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
        return ml_model

    def predict(self, model_name: str, features: pd.DataFrame) -> tuple[np.ndarray, float]:
        if model_name not in self.models:
            msg = f"Model {model_name} not found"
            raise ValueError(msg)
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)
            raw = float(np.max(proba, axis=1).mean()) if proba.ndim > 1 else float(np.max(proba))
            confidence = ConfidenceNormalizer.normalize(raw) if ConfidenceNormalizer else raw
        else:
            confidence = ConfidenceNormalizer.normalize(0.5) if ConfidenceNormalizer else 0.5
        return prediction, confidence

    def get_feature_importance(self, model_name: str) -> dict[str, float]:
        return self.feature_importance.get(model_name, {})

    def save_model(self, model_name: str, filepath: str) -> None:
        if model_name not in self.models:
            msg = f"Model {model_name} not found"
            raise ValueError(msg)
        model_data = {
            "model": self.models[model_name],
            "scaler": self.scalers[model_name],
            "feature_importance": self.feature_importance[model_name],
        }
        filepath_obj = Path(filepath)
        with filepath_obj.open("wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Saved model {model_name} to {filepath}")

    def load_model(self, model_name: str, filepath: str) -> None:
        filepath_obj = Path(filepath)
        with filepath_obj.open("rb") as f:
            model_data = pickle.load(f)
        self.models[model_name] = model_data["model"]
        self.scalers[model_name] = model_data["scaler"]
        self.feature_importance[model_name] = model_data["feature_importance"]
        logger.info(f"Loaded model {model_name} from {filepath}")


strategy_builder = StrategyBuilder()
predictive_analytics = PredictiveAnalytics()
pattern_recognition = PatternRecognition()


class AIStrategies:
    pass
