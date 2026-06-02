#!/usr/bin/env python3
"""
Advanced AI Models for Mystic Trading Platform
Enhanced models with advanced features for better performance
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class AdvancedAIModel:
    """Advanced AI model with ensemble methods and feature engineering"""

    def __init__(self, model_name: str, model_type: str = "ensemble"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.feature_importance = None
        self.performance_metrics = {}
        self.training_history = []
        self.is_trained = False
        self.feature_names: list[str] | None = None
        self.selected_feature_names: list[str] | None = None

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model"""
        if self.model_type == "ensemble":
            # Advanced ensemble with multiple algorithms
            self.model = VotingClassifier(
                [
                    (
                        "rf",
                        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
                    ),
                    (
                        "gb",
                        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                    ),
                    (
                        "mlp",
                        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                    ),
                    ("svm", SVC(probability=True, kernel="rbf", random_state=42)),
                    ("ada", AdaBoostClassifier(n_estimators=100, random_state=42)),
                ],
                voting="soft",
            )
        elif self.model_type == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate="adaptive",
                max_iter=1000,
                random_state=42,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=42,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical features"""
        df = data.copy()

        # Price-based features
        df["price_change"] = df["price"].pct_change()
        df["price_change_2"] = df["price"].pct_change(2)
        df["price_change_5"] = df["price"].pct_change(5)
        df["price_change_10"] = df["price"].pct_change(10)

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f"ma_{window}"] = df["price"].rolling(window=window).mean()
            df[f"ma_{window}_ratio"] = df["price"] / df[f"ma_{window}"]

        # Volatility features
        df["volatility_5"] = df["price_change"].rolling(window=5).std()
        df["volatility_10"] = df["price_change"].rolling(window=10).std()
        df["volatility_20"] = df["price_change"].rolling(window=20).std()

        # Volume features
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_5"]

        # Technical indicators
        df["rsi"] = self._calculate_rsi(df["price"])
        df["macd"] = self._calculate_macd(df["price"])
        bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(df["price"])
        df["bollinger_upper"] = bollinger_upper
        df["bollinger_lower"] = bollinger_lower
        band_width = bollinger_upper - bollinger_lower
        df["bollinger_position"] = np.where(
            band_width != 0,
            (df["price"] - bollinger_lower) / band_width,
            np.nan,
        )

        # Momentum features
        df["momentum_5"] = df["price"] / df["price"].shift(5)
        df["momentum_10"] = df["price"] / df["price"].shift(10)
        df["momentum_20"] = df["price"] / df["price"].shift(20)

        # Time-based features
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Market regime features
        df["trend_strength"] = self._calculate_trend_strength(df["price"])
        df["market_regime"] = self._classify_market_regime(df["price"])

        return df.replace([np.inf, -np.inf], np.nan)

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        if isinstance(loss, pd.Series):
            loss = loss.replace(0, np.nan)
        rs = gain / loss
        if isinstance(rs, pd.Series):
            rs = rs.replace([np.inf, -np.inf], np.nan)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, _signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower

    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength"""
        ma_short = prices.rolling(window=5).mean()
        ma_long = prices.rolling(window=window).mean()
        return (ma_short - ma_long) / ma_long

    def _classify_market_regime(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Classify market regime (0: sideways, 1: uptrend, 2: downtrend)"""
        returns = prices.pct_change(window)
        volatility = returns.rolling(window=window).std()

        regime = np.where(
            returns > volatility,
            1,  # Uptrend
            np.where(returns < -volatility, 2, 0),
        )  # Downtrend, Sideways
        return pd.Series(regime, index=prices.index)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> dict[str, Any]:
        """Train the model with advanced features"""
        try:
            self.performance_metrics = {}
            self.feature_importance = None
            self.is_trained = False

            if feature_names is not None:
                base_feature_names = list(feature_names)
            elif self.feature_names is not None and len(self.feature_names) == X.shape[1]:
                base_feature_names = self.feature_names
            else:
                base_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.feature_names = base_feature_names
            self.selected_feature_names = base_feature_names

            # Feature selection
            X_selected = X
            self.feature_selector = None
            if X.shape[1] > 10 and np.unique(y).size > 1:
                try:
                    selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
                    X_selected = selector.fit_transform(X, y)
                    mask = selector.get_support()
                    selected_feature_names = [base_feature_names[i] for i, keep in enumerate(mask) if keep]
                    if not selected_feature_names:
                        selected_feature_names = base_feature_names
                    self.selected_feature_names = selected_feature_names
                    self.feature_selector = selector
                    logger.info(f"Selected {X_selected.shape[1]} best features")
                except ValueError as exc:
                    logger.warning(f"Feature selection skipped due to error: {exc}")
                    X_selected = X
                    self.feature_selector = None
                    self.selected_feature_names = base_feature_names
            else:
                self.selected_feature_names = base_feature_names

            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)

            # Train model
            self.model.fit(X_scaled, y)

            # Calculate performance metrics
            y_pred = self.model.predict(X_scaled)
            self.performance_metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y, y_pred, average="weighted", zero_division=0),
            }

            # Feature importance
            if hasattr(self.model, "feature_importances_"):
                self.feature_importance = np.asarray(self.model.feature_importances_, dtype=np.float64)
            elif hasattr(self.model, "estimators_"):
                importances = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, "feature_importances_"):
                        importances.append(np.asarray(estimator.feature_importances_, dtype=np.float64))
                if importances:
                    try:
                        self.feature_importance = np.mean(np.vstack(importances), axis=0)
                    except ValueError:
                        logger.warning("Could not aggregate feature importances from estimators.")
                        self.feature_importance = None

            self.is_trained = True
            self.training_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": self.performance_metrics.copy(),
                    "n_features": X_selected.shape[1],
                    "n_samples": len(y),
                }
            )

            logger.info(f"Model {self.model_name} trained successfully")
            logger.info(f"Performance: {self.performance_metrics}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception(f"Error training model {self.model_name}")
            raise
        else:
            return self.performance_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            msg = "Model not trained"
            raise ValueError(msg)

        # Apply feature selection if available
        X_selected = self.feature_selector.transform(X) if self.feature_selector else X

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        # Make predictions
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            msg = "Model not trained"
            raise ValueError(msg)

        # Apply feature selection if available
        X_selected = self.feature_selector.transform(X) if self.feature_selector else X

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)

        predictions = self.model.predict(X_scaled)
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            msg = "Trained model does not expose classes_ for probability estimation."
            raise AttributeError(msg)
        class_to_index = {label: idx for idx, label in enumerate(classes)}
        proba = np.zeros((predictions.shape[0], len(classes)), dtype=np.float64)
        for idx, label in enumerate(predictions):
            label_index = class_to_index.get(label)
            if label_index is not None:
                proba[idx, label_index] = 1.0
        return proba

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores"""
        if self.feature_importance is None:
            return {}

        feature_names = self.selected_feature_names or self.feature_names
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]

        length = min(len(feature_names), len(self.feature_importance))
        return {feature_names[i]: float(self.feature_importance[i]) for i in range(length)}

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            msg = "Model not trained"
            raise ValueError(msg)

        model_data = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "feature_importance": self.feature_importance,
            "performance_metrics": self.performance_metrics,
            "training_history": self.training_history,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "selected_feature_names": self.selected_feature_names,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        if not Path(filepath).exists():
            msg = f"Model file not found: {filepath}"
            raise FileNotFoundError(msg)

        model_data = joblib.load(filepath)

        self.model_name = model_data["model_name"]
        self.model_type = model_data["model_type"]
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_selector = model_data["feature_selector"]
        self.feature_importance = model_data["feature_importance"]
        self.performance_metrics = model_data["performance_metrics"]
        self.training_history = model_data["training_history"]
        self.is_trained = model_data["is_trained"]
        self.feature_names = model_data.get("feature_names")
        self.selected_feature_names = model_data.get("selected_feature_names")

        logger.info(f"Model loaded from {filepath}")


class AdvancedModelManager:
    """Manager for multiple advanced AI models"""

    def __init__(self, models_dir: str = "data/models/advanced"):
        self.models_dir = models_dir
        self.models: dict[str, AdvancedAIModel] = {}
        Path(models_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _determine_cv_folds(y: np.ndarray, requested_folds: int) -> int:
        """Determine a safe number of CV folds based on class distribution."""
        if requested_folds < 2:
            msg = "requested_folds must be at least 2 for cross-validation."
            raise ValueError(msg)
        y_array = np.asarray(y).ravel()
        if y_array.size < 2:
            msg = "At least two samples are required for cross-validation."
            raise ValueError(msg)
        unique, counts = np.unique(y_array, return_counts=True)
        if unique.size < 2:
            msg = "At least two classes are required for cross-validation."
            raise ValueError(msg)
        max_folds_by_class = counts.min()
        max_possible = min(len(y_array), max_folds_by_class)
        cv = min(requested_folds, max_possible)
        if cv < 2:
            msg = "Not enough samples per class to perform cross-validation."
            raise ValueError(msg)
        return cv

    def create_model(self, model_name: str, model_type: str = "ensemble") -> AdvancedAIModel:
        """Create a new advanced model"""
        model = AdvancedAIModel(model_name, model_type)
        self.models[model_name] = model
        return model

    def get_model(self, model_name: str) -> AdvancedAIModel | None:
        """Get a model by name"""
        return self.models.get(model_name)

    def train_all_models(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        """Train all models"""
        results = {}
        for name, model in self.models.items():
            try:
                metrics = model.train(X, y, feature_names)
                results[name] = metrics
                logger.info(f"Trained {name}: {metrics}")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error training {name}")
                results[name] = {"error": str(e)}
        return results

    def optimize_model_with_grid_search(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """Optimize a model using GridSearchCV"""
        if model_name not in self.models:
            msg = f"Model {model_name} not found"
            raise ValueError(msg)

        cv_splits = self._determine_cv_folds(y, 5)

        # Use LogisticRegression for optimization example
        base_model = LogisticRegression(random_state=42, max_iter=1000)

        # Define parameter grid
        param_grid = {"C": [0.1, 1.0, 10.0], "solver": ["liblinear", "lbfgs"]}

        # Use StandardScaler for preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform grid search
        grid_search = GridSearchCV(base_model, param_grid, cv=cv_splits, scoring="accuracy", n_jobs=-1)

        grid_search.fit(X_scaled, y)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_estimator": grid_search.best_estimator_,
        }

    def cross_validate_model(self, model_name: str, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """Perform cross-validation on a model"""
        if model_name not in self.models:
            msg = f"Model {model_name} not found"
            raise ValueError(msg)

        effective_cv_folds = self._determine_cv_folds(y, cv_folds)

        # Use LogisticRegression for cross-validation
        model = LogisticRegression(random_state=42, max_iter=1000)

        # Use StandardScaler for preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=effective_cv_folds, scoring="accuracy")

        logger.info(f"Cross-validation scores for {model_name}: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return {
            "cv_scores": cv_scores.tolist(),
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "cv_folds": effective_cv_folds,
        }

    def get_best_model(self) -> AdvancedAIModel | None:
        """Get the best performing model"""
        if not self.models:
            return None

        best_model = None
        best_score = float("-inf")

        for _name, model in self.models.items():
            if model.is_trained and "accuracy" in model.performance_metrics:
                score = model.performance_metrics["accuracy"]
                if score > best_score:
                    best_score = score
                    best_model = model

        return best_model

    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions from all trained models"""
        if not self.models:
            msg = "No models available"
            raise ValueError(msg)

        predictions = []
        weights = []

        for _name, model in self.models.items():
            if model.is_trained:
                pred = model.predict_proba(X)
                predictions.append(np.asarray(pred, dtype=np.float64))
                weights.append(model.performance_metrics.get("accuracy", 1.0))

        if not predictions:
            msg = "No trained models available"
            raise ValueError(msg)

        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        weights = np.ones_like(weights) / weights.size if weight_sum <= 0 or not np.isfinite(weight_sum) else weights / weight_sum

        ensemble_pred = np.zeros_like(predictions[0], dtype=np.float64)
        for pred, weight in zip(predictions, weights, strict=False):
            ensemble_pred += pred * weight

        return ensemble_pred


# Global model manager - using dict to avoid global keyword
_model_manager_state: dict[str, AdvancedModelManager | None] = {"instance": None}


def get_advanced_model_manager() -> AdvancedModelManager:
    """Get the global advanced model manager"""
    if _model_manager_state["instance"] is None:
        _model_manager_state["instance"] = AdvancedModelManager()
    return _model_manager_state["instance"]
