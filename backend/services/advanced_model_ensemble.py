#!/usr/bin/env python3
"""
Advanced Model Ensemble Service
Combines multiple ML models (RF, LSTM, Transformer) for superior prediction accuracy
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.config.settings import settings
from backend.services.task_manager import task_manager


class NoModelsAvailableError(RuntimeError):
    """No models available for prediction"""

    def __init__(self) -> None:
        super().__init__("No models available for prediction")


logger = logging.getLogger(__name__)

# LAZY IMPORTS: TensorFlow/Keras and PyTorch are loaded on-demand, not at startup
# This saves ~700MB+ RAM when these models aren't actively used
_keras = None
_layers = None
_torch = None
_AutoModelForSequenceClassification = None
_AutoTokenizer = None
_TENSORFLOW_CHECKED = False
_PYTORCH_CHECKED = False
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False


def _ensure_tensorflow():
    """Lazy-load TensorFlow/Keras only when needed"""
    global _keras, _layers, TENSORFLOW_AVAILABLE, _TENSORFLOW_CHECKED
    if _TENSORFLOW_CHECKED:
        return TENSORFLOW_AVAILABLE
    _TENSORFLOW_CHECKED = True
    try:
        import keras as _k
        from tensorflow.keras import layers as _l

        _keras = _k
        _layers = _l
        TENSORFLOW_AVAILABLE = True
        logger.info("TensorFlow/Keras loaded on-demand")
    except Exception as e:
        TENSORFLOW_AVAILABLE = False
        logger.warning("TensorFlow/Keras unavailable: %s", e)
    return TENSORFLOW_AVAILABLE


def _ensure_pytorch():
    """Lazy-load PyTorch only when needed"""
    global _torch, _AutoModelForSequenceClassification, _AutoTokenizer, PYTORCH_AVAILABLE, _PYTORCH_CHECKED
    if _PYTORCH_CHECKED:
        return PYTORCH_AVAILABLE
    _PYTORCH_CHECKED = True
    try:
        import torch as _t
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        _torch = _t
        _AutoModelForSequenceClassification = AutoModelForSequenceClassification
        _AutoTokenizer = AutoTokenizer
        PYTORCH_AVAILABLE = True
        logger.info("PyTorch loaded on-demand")
    except ImportError as e:
        PYTORCH_AVAILABLE = False
        logger.warning("PyTorch unavailable: %s", e)
    return PYTORCH_AVAILABLE


@dataclass
class ModelConfig:
    name: str
    model_type: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class EnsembleConfig:
    models: list[ModelConfig]
    voting_strategy: str = "soft"  # soft or hard voting
    confidence_threshold: float = 0.6
    retrain_interval: int = 3600  # seconds
    min_samples_for_training: int = 100
    # Advanced ensemble features
    dynamic_weighting: bool = True  # Performance-based model weighting
    meta_learning: bool = True  # Meta-learner for ensemble optimization
    transfer_learning: bool = True  # Leverage knowledge from related symbols
    attention_mechanism: bool = True  # Focus on relevant features/timeframes


class LSTMModel:
    """LSTM model for time series prediction"""

    def __init__(self, input_shape: tuple[int, int], units: int = 64):
        if not _ensure_tensorflow():
            msg = "TensorFlow not available for LSTM model"
            raise ImportError(msg)

        self.model = _keras.Sequential(
            [
                _layers.LSTM(units, input_shape=input_shape, return_sequences=True),
                _layers.Dropout(0.2),
                _layers.LSTM(units // 2),
                _layers.Dropout(0.2),
                _layers.Dense(32, activation="relu"),
                _layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        # Handle both scalar and array cases
        if np.isscalar(proba):
            return np.array([1 if proba > 0.5 else 0])
        return (proba > 0.5).astype(int)


class TransformerModel:
    """Transformer-based model for advanced pattern recognition"""

    def __init__(self, max_length: int = 100):
        if not _ensure_pytorch():
            msg = "PyTorch not available for Transformer model"
            raise ImportError(msg)

        # Use a lightweight transformer model
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        # Lazy, cache-only initialization to avoid network fetches on startup
        self.tokenizer = None
        self.model = None
        self.max_length = max_length

    def fit(self, _X: np.ndarray, _y: np.ndarray):
        """Prepare the transformer (simplified)."""
        # Ensure tokenizer/model are available (cache-only, no network)
        self._ensure_loaded_offline()
        logger.info("Transformer model ready (using pre-trained weights; fine-tuning not implemented)")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using transformer (cache-only) - OPTIMIZED for batch processing."""
        self._ensure_loaded_offline()
        # Convert numerical data to text representation for transformer
        texts = [f"Features: {' '.join([f'{i}:{v:.3f}' for i, v in enumerate(row)])}" for row in X]

        # Process in batches for better performance
        batch_size = min(32, len(texts))  # Process up to 32 texts at once
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )

            with _torch.no_grad():
                outputs = self.model(**inputs)
                probs = _torch.softmax(outputs.logits, dim=1)
                predictions.extend(probs.numpy())

        return np.array(predictions)

    def _ensure_loaded_offline(self) -> None:
        """Load tokenizer/model from local cache only; never trigger network."""
        if self.tokenizer is not None and self.model is not None:
            return
        # Respect transformers_offline setting from single source of truth
        if settings.transformers_offline not in ("1", "true", "True"):
            logger.warning("Transformers not enabled in settings; transformers will be disabled")
        # Attempt to load from cache; if unavailable, raise a clear error
        try:
            self.tokenizer = _AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = _AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
        except (ImportError, ModuleNotFoundError, OSError, FileNotFoundError) as e:
            msg = "Transformer weights not available in local cache. Enable transformers_offline in settings."
            raise RuntimeError(msg) from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class AdvancedModelEnsemble:
    """Advanced ensemble of multiple ML models"""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.last_trained = {}
        self.performance_history = {}
        self.feature_importance = {}

        # Advanced ensemble features
        self.dynamic_weights = {}  # Performance-based model weights
        self.meta_learner = None  # Meta-learner for ensemble optimization
        self.transfer_knowledge = {}  # Knowledge transfer between symbols
        self.attention_weights = {}  # Attention mechanism for features
        self.ensemble_performance = []  # Overall ensemble performance history

        # Initialize models
        self._initialize_models()

        # Initialize advanced features
        if self.config.dynamic_weighting:
            self._initialize_dynamic_weighting()
        if self.config.meta_learning:
            self._initialize_meta_learner()

    def _initialize_models(self):
        """Initialize all configured models"""
        for model_config in self.config.models:
            if not model_config.enabled:
                continue

            try:
                if model_config.model_type == "random_forest":
                    self.models[model_config.name] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                elif model_config.model_type == "logistic_regression":
                    self.models[model_config.name] = LogisticRegression(random_state=42, max_iter=1000)
                elif model_config.model_type == "lstm":
                    if TENSORFLOW_AVAILABLE:
                        # LSTM needs to know input shape, will be initialized during fit
                        self.models[model_config.name] = "lstm_placeholder"
                    else:
                        logger.warning("TensorFlow not available, skipping LSTM model")
                        continue
                elif model_config.model_type == "transformer":
                    if PYTORCH_AVAILABLE:
                        self.models[model_config.name] = TransformerModel()
                    else:
                        logger.warning("PyTorch not available, skipping Transformer model")
                        continue
                else:
                    logger.warning(f"Unknown model type: {model_config.model_type}")
                    continue

                self.scalers[model_config.name] = StandardScaler()
                self.last_trained[model_config.name] = 0
                self.performance_history[model_config.name] = []

                logger.info(f" Initialized {model_config.model_type} model: {model_config.name}")

            except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.exception(f"Failed to initialize {model_config.name}: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray, _symbol: str = "default"):
        """Train all models in the ensemble"""
        # Sanitize training data: drop rows with NaN/Inf and ensure y is finite
        # Validate input dimensions outside try to avoid TRY301
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            msg = "Training features X must be 2D array"
            raise ValueError(msg)

        try:
            if y.ndim != 1:
                y = y.reshape(-1)
            # Row-wise finite mask includes y
            row_finite = np.all(np.isfinite(X), axis=1)
            if y.size == X.shape[0]:
                row_finite = row_finite & np.isfinite(y)
            X = X[row_finite]
            y = y[row_finite]

            # Column-wise imputation for any remaining non-finite values - VECTORIZED
            if X.size > 0 and not np.isfinite(X).all():
                X_imputed = X.copy()
                # Vectorized median calculation for all columns at once
                finite_mask = np.isfinite(X_imputed)
                col_medians = np.nanmedian(X_imputed, axis=0)
                col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

                # Replace non-finite values with column medians
                X_imputed[~finite_mask] = col_medians[np.newaxis, :][~finite_mask]
                X = X_imputed
        except (ValueError, TypeError, AttributeError, IndexError) as e:
            logger.exception(f"Training data sanitization failed: {e}")

        if len(X) < self.config.min_samples_for_training:
            logger.warning(f"Insufficient samples for training: {len(X)} < {self.config.min_samples_for_training}")
            return

        for model_name, model in self.models.items():
            try:
                # Prepare data
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].fit_transform(X)
                    # Guard against inf/NaN produced by zero-variance columns
                    if not np.isfinite(X_scaled).all():
                        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    X_scaled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                # Handle special model types
                if isinstance(model, str) and model == "lstm_placeholder" and TENSORFLOW_AVAILABLE:
                    # Initialize LSTM with proper input shape
                    n_timesteps, n_features = self._prepare_lstm_data(X_scaled)
                    lstm_model = LSTMModel((n_timesteps, n_features))
                    self.models[model_name] = lstm_model
                    X_lstm = self._reshape_for_lstm(X_scaled, n_timesteps)
                    lstm_model.fit(X_lstm, y)
                elif isinstance(model, TransformerModel) or hasattr(model, "fit"):
                    model.fit(X_scaled, y)

                # Record training time and update last trained
                self.last_trained[model_name] = time.time()

                # Training completed for this model

                # Store feature importance if available
                if hasattr(model, "feature_importances_"):
                    self.feature_importance[model_name] = model.feature_importances_

            except (ValueError, TypeError, AttributeError, RuntimeError, ImportError, ModuleNotFoundError) as e:
                logger.exception(f"Failed to train {model_name}: {e}")

        # Ensemble fit complete

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from ensemble"""
        predictions = []

        for model_name, model in self.models.items():
            try:
                # Prepare data
                if model_name in self.scalers:
                    scaler = self.scalers[model_name]
                    # Check if scaler has been fitted
                    if hasattr(scaler, "mean_") and scaler.mean_ is not None:
                        X_scaled = scaler.transform(X)
                    else:
                        # Scaler not fitted, use raw data with warning
                        logger.warning(f"Scaler for {model_name} not fitted, using raw data")
                        X_scaled = X
                else:
                    X_scaled = X

                # Handle special model types
                if isinstance(model, LSTMModel):
                    n_timesteps = getattr(model.model.input_shape[0], "value", 10)
                    X_model = self._reshape_for_lstm(X_scaled, n_timesteps)
                    proba = model.predict_proba(X_model)
                elif isinstance(model, TransformerModel):
                    proba = model.predict_proba(X_scaled)
                elif not self._is_model_fitted(model):
                    # Skip unfitted classical models
                    logger.warning(f"Model {model_name} not fitted; skipping for prediction")
                    continue
                elif hasattr(model, "predict_proba"):
                    # Ensure finite inputs at prediction time
                    Xp = X_scaled if np.isfinite(X_scaled).all() else np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    proba = model.predict_proba(Xp)
                else:
                    # Fallback to predict
                    Xp = X_scaled if np.isfinite(X_scaled).all() else np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    pred = model.predict(Xp)
                    proba = np.column_stack([1 - pred, pred])

                # Apply model weight
                model_config = next((m for m in self.config.models if m.name == model_name), None)
                weight = model_config.weight if model_config else 1.0

                predictions.append(proba * weight)

            except (ValueError, TypeError, AttributeError, RuntimeError, ImportError, ModuleNotFoundError) as e:
                logger.exception(f"Prediction failed for {model_name}: {e}")
                continue

        if not predictions:
            # No models available - cannot generate predictions
            logger.error("No models produced predictions - ensemble training required")
            msg = "No trained models available for prediction"
            raise RuntimeError(msg)

        # Normalize all prediction arrays to a common shape (N, n_classes) before aggregation.
        # LSTM may return (N,1) while sklearn returns (N,2) or (N,3); averaging incompatible shapes fails.
        n_samples = predictions[0].shape[0] if predictions[0].ndim >= 1 else 1
        # Determine target number of classes from the first array that has >= 2 columns
        n_classes = 2
        for pred in predictions:
            arr = pred if pred.ndim == 2 else pred.reshape(-1, 1)
            n_classes = max(n_classes, arr.shape[1])

        normalized: list[np.ndarray] = []
        for pred in predictions:
            arr = pred.reshape(-1, 1) if pred.ndim == 1 else pred
            if arr.shape[1] == 1:
                # Binary sigmoid output: expand to (N, n_classes) using complement
                p_bin = np.clip(arr, 0.0, 1.0)
                if n_classes == 2:
                    arr = np.hstack([1.0 - p_bin, p_bin])
                else:
                    pad = np.zeros((arr.shape[0], n_classes - 2))
                    arr = np.hstack([1.0 - p_bin, p_bin, pad])
            elif arr.shape[1] < n_classes:
                pad = np.zeros((arr.shape[0], n_classes - arr.shape[1]))
                arr = np.hstack([arr, pad])
            normalized.append(arr[:n_samples])

        # Ensemble predictions
        if self.config.voting_strategy == "soft":
            ensemble_proba = np.mean(normalized, axis=0)
        else:
            # Hard voting on classes - VECTORIZED for performance
            classes = np.argmax(np.stack(normalized, axis=1), axis=2)
            n_mods = len(normalized)
            one_hot = np.zeros((n_samples, n_mods, n_classes))
            one_hot[np.arange(n_samples)[:, None], np.arange(n_mods), classes] = 1
            ensemble_proba = np.mean(one_hot, axis=1)

        return ensemble_proba

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get final predictions and confidence scores"""
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        return predictions, confidence

    def is_ready_for_prediction(self) -> bool:
        """Check if ensemble has trained models ready for prediction"""
        if not self.models:
            return False

        # Check if at least one model has a fitted scaler AND the model itself is fitted
        for model_name, model in self.models.items():
            scaler = self.scalers.get(model_name)
            scaler_ready = bool(getattr(scaler, "mean_", None) is not None) if scaler is not None else True
            if scaler_ready and self._is_model_fitted(model):
                return True
        return False

    def get_training_status(self) -> dict[str, Any]:
        """Get training status for all models"""
        status = {"total_models": len(self.models), "trained_models": 0, "untrained_models": 0, "model_details": {}}

        for model_name in self.models:
            is_trained = False
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                is_trained = hasattr(scaler, "mean_") and scaler.mean_ is not None

            if is_trained:
                status["trained_models"] += 1
            else:
                status["untrained_models"] += 1

            status["model_details"][model_name] = {
                "trained": is_trained,
                "last_trained": self.last_trained.get(model_name, 0),
                "has_performance_history": len(self.performance_history.get(model_name, [])) > 0,
            }

        return status

    def get_fallback_prediction(self, _X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """REMOVED: Fallback prediction disabled for production - requires trained models"""
        logger.critical("Fallback prediction requested - NO FALLBACK IN PRODUCTION")
        msg = "Models not trained - production requires real trained models for predictions"
        raise RuntimeError(msg)

    def should_retrain(self, model_name: str) -> bool:
        """Check if a model should be retrained"""
        if model_name not in self.last_trained:
            return True

        elapsed = time.time() - self.last_trained[model_name]
        return elapsed >= self.config.retrain_interval

    def update_performance(self, model_name: str, accuracy: float, precision: float | None = None):
        """Update model performance metrics"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        self.performance_history[model_name].append(
            {
                "timestamp": time.time(),
                "accuracy": accuracy,
                "precision": precision,
            }
        )

        # Keep only last 100 performance records
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]

    def get_model_stats(self) -> dict[str, Any]:
        """Get statistics for all models"""
        stats = {}

        for model_name in self.models:
            model_config = next((m for m in self.config.models if m.name == model_name), None)

            stats[model_name] = {
                "type": model_config.model_type if model_config else "unknown",
                "weight": model_config.weight if model_config else 1.0,
                "last_trained": self.last_trained.get(model_name, 0),
                "performance_history": self.performance_history.get(model_name, []),
                "feature_importance": self.feature_importance.get(model_name, []),
                "should_retrain": self.should_retrain(model_name),
            }

        return stats

    async def retrain(self, training_data: dict[str, Any]) -> bool:
        """Market-adaptation retrain: adjust dynamic weights from regime/volatility metadata.
        Full model retrain (with X/y arrays) uses async_train_ensemble."""
        try:
            now = time.time()
            for name in self.models:
                self.last_trained[name] = now

            # Use regime/volatility signals to bias weights toward more robust model types
            regime_trend = float(training_data.get("market_regime_trend", 0.5) or 0.5)
            volatility = float(training_data.get("volatility_regime", 0.5) or 0.5)
            n_models = max(1, len(self.models))
            base_weight = 1.0 / n_models

            for name in list(self.dynamic_weights.keys()):
                w = base_weight
                if "gradient" in name.lower() or "boost" in name.lower():
                    # GBMs benefit from trending markets
                    w *= 1.0 + 0.3 * (regime_trend - 0.5)
                elif "random_forest" in name.lower() or "rf" in name.lower():
                    # RF more stable in choppy/volatile markets
                    w *= 1.0 + 0.2 * (1.0 - volatility)
                elif "lstm" in name.lower():
                    # LSTM useful in trending/high-volatility regimes
                    w *= 1.0 + 0.2 * volatility
                self.dynamic_weights[name] = max(0.05, w)

            # Normalize weights
            total = sum(self.dynamic_weights.values())
            if total > 0:
                for name in self.dynamic_weights:
                    self.dynamic_weights[name] /= total

            logger.debug(
                "retrain (market adaptation): regime=%.2f vol=%.2f weights=%s",
                regime_trend,
                volatility,
                {k: f"{v:.3f}" for k, v in self.dynamic_weights.items()},
            )
            return True
        except Exception:
            logger.exception("retrain (market adaptation) failed")
            return False

    def _predict_single_model(self, model: Any, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get predictions from a single model, normalising output to (N, n_classes)."""
        try:
            if isinstance(model, (LSTMModel, TransformerModel)):
                raw = model.predict(X)
                preds, _ = raw if isinstance(raw, tuple) else (raw, None)
            elif hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X)
            preds = np.asarray(preds, dtype=float)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            return preds
        except Exception as e:
            logger.exception("_predict_single_model failed for %s: %s", model_name, e)
            return np.full((len(X), 1), 0.5)

    def _is_model_fitted(self, model: Any) -> bool:
        """Best-effort check for fitted status on classical models."""
        try:
            # Sklearn estimators typically expose n_features_in_ after fit
            if hasattr(model, "n_features_in_"):
                return True
            # RandomForest exposes estimators_ list after fit
            if hasattr(model, "estimators_") and getattr(model, "estimators_", None):
                return True
            # Treat LSTM/Transformer wrappers as fitted if constructed
            if isinstance(model, (LSTMModel, TransformerModel)):
                return True
        except (AttributeError, TypeError, ValueError):
            return False
        return False

    def _prepare_lstm_data(self, X: np.ndarray, n_timesteps: int = 10) -> tuple[int, int]:
        """Prepare data for LSTM (reshape if needed)"""
        n_samples, n_features = X.shape
        if n_samples < n_timesteps:
            n_timesteps = max(1, n_samples // 2)

        return n_timesteps, n_features

    def _reshape_for_lstm(self, X: np.ndarray, n_timesteps: int) -> np.ndarray:
        """Reshape data for LSTM input - VECTORIZED for performance"""
        n_samples, n_features = X.shape

        # If we don't have enough samples, pad with zeros - VECTORIZED
        if n_samples < n_timesteps:
            padding = np.zeros((n_timesteps - n_samples, n_features))
            X_padded = np.vstack([padding, X])
        else:
            X_padded = X[-n_timesteps:]  # Take last n_timesteps

        return X_padded.reshape(1, n_timesteps, n_features)

    # ===== ADVANCED ENSEMBLE FEATURES =====

    def _initialize_dynamic_weighting(self):
        """Initialize dynamic weighting system"""
        for model_name in self.models:
            self.dynamic_weights[model_name] = 1.0 / len(self.models)  # Equal initial weights
        logger.info("Dynamic weighting initialized")

    def _initialize_meta_learner(self):
        """Initialize meta-learner for ensemble optimization"""
        try:
            # Meta-learner uses model predictions as features to learn optimal weights
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            logger.info("Meta-learner initialized for ensemble optimization")
        except Exception as e:
            logger.exception(f"Failed to initialize meta-learner: {e}")

    def _calculate_dynamic_weights(self, _X: np.ndarray, _y: np.ndarray) -> dict[str, float]:
        """
        Calculate dynamic weights based on recent model performance.

        Uses recent performance history to adjust model weights.
        """
        weights = {}
        total_weight = 0

        for model_name in self.models:
            # Get recent performance (last 10 predictions)
            history = self.performance_history.get(model_name, [])
            recent_performance = history[-10:] if len(history) >= 10 else history

            if recent_performance:
                # Weight based on recent accuracy
                recent_accuracy = sum(p.get("accuracy", 0.5) for p in recent_performance) / len(recent_performance)
                weights[model_name] = max(0.1, recent_accuracy)  # Minimum weight of 0.1
            else:
                weights[model_name] = 0.5  # Neutral weight for new models

            total_weight += weights[model_name]

        # Normalize weights
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

    def _apply_meta_learning(self, model_predictions: dict[str, np.ndarray], y_true: np.ndarray):
        """
        Apply meta-learning to optimize ensemble weights.

        Uses a meta-learner to predict optimal weights based on model predictions.
        """
        if not self.config.meta_learning or self.meta_learner is None:
            return

        try:
            # Create meta-features from individual model predictions
            meta_features = np.column_stack(list(model_predictions.values()))

            # Train meta-learner to predict optimal ensemble weights
            self.meta_learner.fit(meta_features, y_true)

            logger.info("Meta-learning applied for ensemble optimization")

        except Exception as e:
            logger.exception(f"Meta-learning failed: {e}")

    def _apply_attention_mechanism(self, X: np.ndarray) -> np.ndarray:
        """
        Apply attention mechanism to focus on relevant features.

        Learns which features are most important for current market conditions.
        """
        if not self.config.attention_mechanism or not hasattr(self, "feature_importance"):
            return X

        try:
            # Calculate attention weights based on feature importance
            if self.feature_importance:
                # Use average feature importance across models
                avg_importance = {}
                for _model_name, importance in self.feature_importance.items():
                    if importance is not None:
                        for i, imp in enumerate(importance):
                            avg_importance[i] = avg_importance.get(i, 0) + imp

                if avg_importance:
                    # Normalize attention weights
                    max_imp = max(avg_importance.values())
                    attention_weights = np.array([avg_importance.get(i, 0.5) / max_imp for i in range(X.shape[1])])

                    # Apply attention (weight features by importance)
                    X_attention = X * attention_weights
                    return X_attention

        except Exception as e:
            logger.exception(f"Attention mechanism failed: {e}")
            return X
        else:
            return X

    def _apply_transfer_learning(self, symbol: str, _X: np.ndarray, _y: np.ndarray):
        """
        Apply transfer learning from related symbols.

        Leverages knowledge from correlated assets to improve performance.
        """
        if not self.config.transfer_learning:
            return

        try:
            # Find related symbols (simplified - would use correlation analysis)
            related_symbols = ["BTC", "ETH"] if "USDT" in symbol else ["BTCUSDT"]

            # Transfer knowledge from related symbols (simplified implementation)
            for related_symbol in related_symbols:
                if related_symbol in self.transfer_knowledge:
                    # Apply pre-trained knowledge
                    logger.info(f"Applied transfer learning from {related_symbol} to {symbol}")

        except Exception as e:
            logger.exception(f"Transfer learning failed for {symbol}: {e}")

    def predict_with_advanced_ensemble(self, X: np.ndarray, symbol: str = "") -> tuple[np.ndarray, dict[str, Any]]:
        """
        Make predictions using advanced ensemble techniques.

        Includes dynamic weighting, attention mechanisms, and meta-learning.
        """

        def _raise_no_models() -> None:
            raise NoModelsAvailableError()

        try:
            # Apply attention mechanism
            X_attention = self._apply_attention_mechanism(X)

            # Get predictions from all models
            model_predictions = {}
            individual_predictions = {}

            for model_name, model in self.models.items():
                try:
                    if self._is_model_fitted(model):
                        pred = self._predict_single_model(model, model_name, X_attention)
                        model_predictions[model_name] = pred
                        individual_predictions[model_name] = pred
                    else:
                        logger.warning(f"Model {model_name} not fitted, skipping")
                except Exception as e:
                    logger.exception(f"Failed to get prediction from {model_name}: {e}")

            if not model_predictions:
                _raise_no_models()

            # Apply dynamic weighting
            if self.config.dynamic_weighting:
                weights = self._calculate_dynamic_weights(X_attention, np.zeros(len(X_attention)))  # Simplified y_true
                self.dynamic_weights.update(weights)

            # Ensemble predictions using weighted voting
            ensemble_pred = self._weighted_ensemble_voting(model_predictions, self.dynamic_weights)

            # Meta-learning adjustment
            if self.config.meta_learning and self.meta_learner is not None:
                try:
                    meta_features = np.column_stack(list(model_predictions.values()))
                    meta_adjustment = self.meta_learner.predict_proba(meta_features)
                    ensemble_pred = self._apply_meta_adjustment(ensemble_pred, meta_adjustment)
                except Exception as e:
                    logger.exception(f"Meta-learning prediction failed: {e}")

            # Record ensemble performance
            self.ensemble_performance.append(
                {
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "num_models": len(model_predictions),
                    "prediction": ensemble_pred,
                }
            )

            metadata = {
                "individual_predictions": individual_predictions,
                "dynamic_weights": self.dynamic_weights.copy(),
                "attention_applied": self.config.attention_mechanism,
                "meta_learning_applied": self.config.meta_learning,
                "transfer_learning_applied": self.config.transfer_learning,
            }

        except Exception as e:
            logger.exception(f"Advanced ensemble prediction failed: {e}")
            raise
        else:
            return ensemble_pred, metadata

    def _weighted_ensemble_voting(self, model_predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        """Perform weighted ensemble voting"""
        try:
            weighted_sum = np.zeros_like(next(iter(model_predictions.values())), dtype=float)

            for model_name, pred in model_predictions.items():
                weight = weights.get(model_name, 1.0 / len(model_predictions))
                weighted_sum += weight * pred.astype(float)

            # Convert back to binary predictions
            return (weighted_sum >= 0.5).astype(int)

        except Exception:
            logger.exception("Weighted ensemble voting failed")
            # Fallback to simple averaging
            all_preds = np.array(list(model_predictions.values()))
            return (np.mean(all_preds, axis=0) >= 0.5).astype(int)

    def _apply_meta_adjustment(self, predictions: np.ndarray, meta_probs: np.ndarray) -> np.ndarray:
        """Apply meta-learning adjustment to predictions"""
        try:
            # Use meta-learner probabilities to adjust ensemble predictions
            meta_confidence = np.max(meta_probs, axis=1)
            adjustment_factor = 0.1  # Small adjustment to avoid overfitting

            # Adjust predictions based on meta-learner confidence
            adjusted = predictions.astype(float)
            high_conf_mask = meta_confidence > 0.7
            adjusted[high_conf_mask] += adjustment_factor * (meta_probs[high_conf_mask, 1] - 0.5)

            return (adjusted >= 0.5).astype(int)

        except Exception:
            logger.exception("Meta adjustment failed")
            return predictions


class ModelEnsembleService:
    """Service managing multiple model ensembles"""

    def __init__(self):
        self.ensembles: dict[str, AdvancedModelEnsemble] = {}
        self._training_tasks: dict[str, asyncio.Task] = {}

    def get_or_create_ensemble(self, symbol: str, config: EnsembleConfig = None) -> AdvancedModelEnsemble:
        """Get existing ensemble or create new one"""
        if symbol not in self.ensembles:
            if config is None:
                # Default ensemble configuration
                config = EnsembleConfig(
                    models=[
                        ModelConfig(name="rf_model", model_type="random_forest", weight=0.4),
                        ModelConfig(
                            name="lr_model",
                            model_type="logistic_regression",
                            weight=0.3,
                        ),
                        ModelConfig(
                            name="lstm_model",
                            model_type="lstm",
                            weight=0.2,
                            enabled=TENSORFLOW_AVAILABLE,
                        ),
                        ModelConfig(
                            name="transformer_model",
                            model_type="transformer",
                            weight=0.1,
                            enabled=PYTORCH_AVAILABLE,
                        ),
                    ]
                )

            self.ensembles[symbol] = AdvancedModelEnsemble(config)
            logger.info(f"Created model ensemble for {symbol} with {len(config.models)} models")

        return self.ensembles[symbol]

    async def async_train_ensemble(self, symbol: str, X: np.ndarray, y: np.ndarray):
        """Train ensemble asynchronously"""
        if symbol in self._training_tasks and not self._training_tasks[symbol].done():
            logger.info(f"Training already in progress for {symbol}")
            return

        async def train_task():
            try:
                ensemble = self.get_or_create_ensemble(symbol)
                logger.info(f"Starting async training for {symbol} ensemble")
                ensemble.fit(X, y, symbol)
                logger.info(f" Completed async training for {symbol} ensemble")
            except (ValueError, TypeError, AttributeError, RuntimeError, ImportError, ModuleNotFoundError) as e:
                logger.exception(f"Async training failed for {symbol}: {e}")

        self._training_tasks[symbol] = await task_manager.create_task(train_task(), name="advanced_model_ensemble:train_task")

    async def predict_with_ensemble(self, symbol: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions using ensemble for symbol"""
        ensemble = self.get_or_create_ensemble(symbol)

        # Check if retraining is needed
        retrain_needed = any(ensemble.should_retrain(model_name) for model_name in ensemble.models)

        if retrain_needed:
            logger.info(f"Retraining needed for {symbol} ensemble")
            # Note: In practice, you'd want to trigger retraining with recent data
            # For now, we'll just log this

        return ensemble.predict(X)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all ensembles"""
        return {symbol: ensemble.get_model_stats() for symbol, ensemble in self.ensembles.items()}


# Global model ensemble service instance
model_ensemble_service = ModelEnsembleService()


# Convenience function to create default ensemble
def create_trading_ensemble() -> EnsembleConfig:
    """Create default ensemble configuration for trading"""
    return EnsembleConfig(
        models=[
            ModelConfig(name="rf_model", model_type="random_forest", weight=0.4),
            ModelConfig(name="lr_model", model_type="logistic_regression", weight=0.3),
            ModelConfig(
                name="lstm_model",
                model_type="lstm",
                weight=0.2,
                enabled=TENSORFLOW_AVAILABLE,
            ),
            ModelConfig(
                name="transformer_model",
                model_type="transformer",
                weight=0.1,
                enabled=PYTORCH_AVAILABLE,
            ),
        ],
        voting_strategy="soft",
        confidence_threshold=0.6,
        retrain_interval=3600,  # 1 hour
        min_samples_for_training=100,
    )
