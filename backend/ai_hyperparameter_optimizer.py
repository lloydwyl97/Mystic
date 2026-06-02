#!/usr/bin/env python3
"""
AI Hyperparameter Tuning with Optuna
"""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class AIHyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""

    def __init__(self) -> None:
        self.study_cache: dict[str, optuna.Study] = {}
        self.best_params_cache: dict[str, dict[str, Any]] = {}
        self.optimization_history: list[dict[str, Any]] = []

        # Optimization settings
        self.max_trials = 100
        self.timeout = 3600  # 1 hour timeout
        self.cv_folds = 5
        self.random_state = 42

    def optimize_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        optimization_type: str = "classification",
    ) -> dict[str, Any]:
        """Optimize hyperparameters for a specific model"""
        # Validate model type before entering try block
        if model_name.lower() not in ("random_forest", "gradient_boosting", "svm", "logistic_regression"):
            msg = f"Unknown model type: {model_name}"
            raise ValueError(msg)

        try:
            logger.info(f"Starting hyperparameter optimization for {model_name}")

            # Create or get existing study
            study_name = f"{model_name}_{optimization_type}"
            if study_name not in self.study_cache:
                self.study_cache[study_name] = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(seed=self.random_state),
                    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
                )

            study = self.study_cache[study_name]

            # Define objective function based on model type
            if model_name.lower() == "random_forest":
                objective = self._create_rf_objective(X, y)
            elif model_name.lower() == "gradient_boosting":
                objective = self._create_gb_objective(X, y)
            elif model_name.lower() == "svm":
                objective = self._create_svm_objective(X, y)
            elif model_name.lower() == "logistic_regression":
                objective = self._create_lr_objective(X, y)

            # Optimize
            # show_progress_bar is available in newer optuna versions; attempt to use it,
            # but fall back if not supported.
            try:
                study.optimize(
                    objective,
                    n_trials=self.max_trials,
                    timeout=self.timeout,
                    show_progress_bar=True,
                )
            except TypeError:
                study.optimize(
                    objective,
                    n_trials=self.max_trials,
                    timeout=self.timeout,
                )

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            # Cache results
            self.best_params_cache[model_name] = best_params

            # Record optimization history
            optimization_record = {
                "model_name": model_name,
                "best_score": best_score,
                "best_params": best_params,
                "n_trials": len(study.trials),
                "optimization_time": datetime.now(timezone.utc).isoformat(),
                "study_name": study_name,
            }
            self.optimization_history.append(optimization_record)

            logger.info(f"Optimization complete for {model_name}: {best_score:.4f}")

            return {
                "status": "success",
                "model_name": model_name,
                "best_score": best_score,
                "best_params": best_params,
                "n_trials": len(study.trials),
                "optimization_time": optimization_record["optimization_time"],
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error optimizing {model_name}: {e}")
            return {"status": "error", "model_name": model_name, "error": str(e)}

    def _create_rf_objective(self, X: np.ndarray, y: np.ndarray):
        """Create objective function for Random Forest optimization"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": self.random_state,
            }

            model = RandomForestClassifier(**params)
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring="accuracy")
            return scores.mean()

        return objective

    def _create_gb_objective(self, X: np.ndarray, y: np.ndarray):
        """Create objective function for Gradient Boosting optimization"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": self.random_state,
            }

            model = GradientBoostingClassifier(**params)
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring="accuracy")
            return scores.mean()

        return objective

    def _create_svm_objective(self, X: np.ndarray, y: np.ndarray):
        """Create objective function for SVM optimization"""

        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            # Choose gamma type: either a string option or a float value
            gamma_choice = trial.suggest_categorical("gamma_choice", ["scale", "auto", "float"])
            gamma = trial.suggest_float("gamma", 0.001, 1.0, log=True) if gamma_choice == "float" else gamma_choice

            params = {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "gamma": gamma,
                "kernel": kernel,
                "random_state": self.random_state,
            }

            model = SVC(**params)
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring="accuracy")
            return scores.mean()

        return objective

    def _create_lr_objective(self, X: np.ndarray, y: np.ndarray):
        """Create objective function for Logistic Regression optimization"""

        def objective(trial):
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            # propose solver but will override if incompatible
            solver_choice = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
            params: dict[str, Any] = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "penalty": penalty,
                "solver": solver_choice,
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "random_state": self.random_state,
            }

            # Adjust solver and extra params based on penalty
            if penalty == "elasticnet":
                # elasticnet requires saga
                params["solver"] = "saga"
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
            elif penalty == "l1":
                # prefer liblinear for l1, but saga also supports it
                if params["solver"] == "lbfgs":
                    params["solver"] = "liblinear"

            model = LogisticRegression(**params)
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring="accuracy")
            return scores.mean()

        return objective

    def optimize_all_models(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Optimize hyperparameters for all supported models"""
        try:
            logger.info("Starting comprehensive model optimization")

            models_to_optimize = [
                "random_forest",
                "gradient_boosting",
                "svm",
                "logistic_regression",
            ]

            results = {}
            best_overall_score = 0
            best_overall_model = None

            for model_name in models_to_optimize:
                logger.info(f"Optimizing {model_name}...")
                result = self.optimize_model(model_name, X, y)
                results[model_name] = result

                if result.get("status") == "success":
                    score = result.get("best_score", 0)
                    try:
                        if score is not None and score > best_overall_score:
                            best_overall_score = score
                            best_overall_model = model_name
                    except TypeError:
                        # ignore non-comparable scores
                        pass

            return {
                "status": "success",
                "best_model": best_overall_model,
                "best_score": best_overall_score,
                "all_results": results,
                "optimization_summary": {
                    "total_models": len(models_to_optimize),
                    "successful_optimizations": sum(1 for r in results.values() if r.get("status") == "success"),
                    "best_model": best_overall_model,
                    "best_score": best_overall_score,
                },
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in comprehensive optimization: {e}")
            return {"status": "error", "error": str(e)}

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history

    def get_best_params(self, model_name: str) -> dict[str, Any] | None:
        """Get best parameters for a specific model"""
        return self.best_params_cache.get(model_name)

    def create_optimized_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Create and train a model with optimized hyperparameters"""
        try:
            best_params = self.get_best_params(model_name)
            if not best_params:
                return {
                    "status": "error",
                    "message": f"No optimized parameters found for {model_name}",
                }

            # Create model with best parameters
            if model_name.lower() == "random_forest":
                model = RandomForestClassifier(**best_params)
            elif model_name.lower() == "gradient_boosting":
                model = GradientBoostingClassifier(**best_params)
            elif model_name.lower() == "svm":
                model = SVC(**best_params)
            elif model_name.lower() == "logistic_regression":
                model = LogisticRegression(**best_params)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown model type: {model_name}",
                }

            # Train the model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)

            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            y_pred = model.predict(X_test)

            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            return {
                "status": "success",
                "model_name": model_name,
                "model": model,
                "train_score": train_score,
                "test_score": test_score,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_params": best_params,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error creating optimized model: {e}")
            return {"status": "error", "error": str(e)}


# Global optimizer instance
hyperparameter_optimizer = AIHyperparameterOptimizer()


def get_hyperparameter_optimizer() -> AIHyperparameterOptimizer:
    """Get the global hyperparameter optimizer instance"""
    return hyperparameter_optimizer


if __name__ == "__main__":
    # Test the optimizer
    logger.info("AI Hyperparameter Optimizer Test")
    logger.info("=" * 40)

    # Create sample data
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 20))
    y = rng.integers(0, 3, 1000)

    optimizer = get_hyperparameter_optimizer()

    # Test single model optimization
    logger.info("Testing Random Forest optimization...")
    result = optimizer.optimize_model("random_forest", X, y)
    logger.info(f"Result: {result}")

    logger.info("Testing comprehensive optimization...")
    results = optimizer.optimize_all_models(X, y)
    logger.info(f"Best model: {results.get('best_model')}")
    best_score_val = results.get("best_score", 0) or 0.0
    try:
        logger.info(f"Best score: {best_score_val:.4f}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.info(f"Best score: {best_score_val}")

    logger.info("[OK] Hyperparameter optimizer test complete!")
