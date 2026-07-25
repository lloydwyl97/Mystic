"""
ai_blended_classifier — genuine algorithmic diversity for DAY/SCALP model training.

Background: the old "ensemble" (LSTM/Transformer/chart-pattern/FearGreedAgent) was
permanently removed from the live decision path — see ai_signal_generator.py's
"CANONICAL DECISION" comment — because torch/tensorflow/transformers are not
installed and those components were stubs, not real models. The per-coin
RandomForestClassifier has been the sole model since.

This module adds real diversity with zero new heavy dependencies: scikit-learn
already ships HistGradientBoostingClassifier, a fundamentally different algorithm
family (gradient-boosted trees vs RF's bagged trees) trained on the same features.
BlendedClassifier duck-types the sklearn estimator interface (predict,
predict_proba, classes_) so it is a drop-in replacement for artifact["model"] —
every existing consumer (ai_signal_generator.py, ai_model_promotion_holdout.py,
tiered-holdout comparison) keeps working unmodified, and the promotion/holdout
gate applies to the blended model exactly as it did to the plain RF.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MIN_BLEND_WEIGHT_FLOOR = 0.2
MIN_GBM_TRAIN_SAMPLES = 30


class BlendedClassifier:
    """predict_proba = w_rf * rf.predict_proba(X) + w_gbm * gbm.predict_proba(X).

    Weights are normalized to sum to 1 at construction time. classes_ is taken
    from ``rf`` and used as the canonical column order; gbm's columns are
    reindexed to match if its classes_ ordering ever differs (defensive — both
    are fit on the same y in practice so this is normally a no-op).
    """

    def __init__(self, rf: Any, gbm: Any, w_rf: float, w_gbm: float):
        self.rf = rf
        self.gbm = gbm
        total = max(1e-9, float(w_rf) + float(w_gbm))
        self.w_rf = float(w_rf) / total
        self.w_gbm = float(w_gbm) / total
        self.classes_ = np.asarray(rf.classes_)

    def _gbm_proba_aligned(self, X: Any) -> np.ndarray:
        gbm_proba = self.gbm.predict_proba(X)
        gbm_classes = np.asarray(self.gbm.classes_)
        if np.array_equal(gbm_classes, self.classes_):
            return gbm_proba
        order = [int(np.where(gbm_classes == c)[0][0]) for c in self.classes_]
        return gbm_proba[:, order]

    def predict_proba(self, X: Any) -> np.ndarray:
        rf_proba = self.rf.predict_proba(X)
        if self.w_gbm <= 0.0:
            return rf_proba
        gbm_proba = self._gbm_proba_aligned(X)
        return self.w_rf * rf_proba + self.w_gbm * gbm_proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, X: Any, y: Any) -> float:
        preds = self.predict(X)
        return float(np.mean(np.asarray(preds) == np.asarray(y)))

    def disagreement(self, X: Any) -> np.ndarray:
        """Total-variation distance between the RF and GBM class-probability
        vectors, in [0, 1] per row. 0 = the two algorithm families fully agree
        on this instance; 1 = maximally split. Free — both models are already
        fit and scored every training cycle, this just surfaces the spread
        between them at inference time instead of throwing it away after
        blending. See compute_disagreement() for the safe wrapper used by
        callers that may be holding a plain (non-blended) fallback model."""
        if self.w_gbm <= 0.0:
            return np.zeros(len(X) if hasattr(X, "__len__") else X.shape[0])
        rf_proba = self.rf.predict_proba(X)
        gbm_proba = self._gbm_proba_aligned(X)
        return np.sum(np.abs(rf_proba - gbm_proba), axis=1) / 2.0


def build_blended_classifier(
    rf: Any,
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    *,
    min_gbm_samples: int = MIN_GBM_TRAIN_SAMPLES,
    weight_floor: float = MIN_BLEND_WEIGHT_FLOOR,
) -> tuple[Any, dict[str, Any]]:
    """Fit a HistGradientBoostingClassifier alongside the already-fit ``rf`` and
    blend by relative held-out validation accuracy. Blend weights are floored at
    ``weight_floor`` each (then renormalized) so a temporarily-weaker model still
    contributes real diversity instead of being blended out to ~0.

    Falls back to the plain ``rf`` unchanged (w_rf=1.0) if data is too thin or the
    GBM fit fails for any reason — safe degradation, never blocks training.
    """
    rf_val_acc = float(rf.score(X_val_s, y_val))
    telemetry: dict[str, Any] = {
        "rf_val_acc": round(rf_val_acc, 4),
        "gbm_val_acc": None,
        "blend_w_rf": 1.0,
        "blend_w_gbm": 0.0,
        "blend_status": "rf_only",
    }

    if len(X_train_s) < min_gbm_samples or len(np.unique(y_train)) < 2:
        telemetry["blend_status"] = "rf_only_insufficient_data"
        return rf, telemetry

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        gbm = HistGradientBoostingClassifier(
            max_depth=6,
            max_iter=150,
            learning_rate=0.08,
            random_state=42,
        )
        gbm.fit(X_train_s, y_train, sample_weight=w_train)
        gbm_val_acc = float(gbm.score(X_val_s, y_val))
    except Exception as exc:
        logger.debug("BLENDED_CLASSIFIER_GBM_FIT_FAILED: %s", exc)
        telemetry["blend_status"] = "rf_only_gbm_fit_failed"
        return rf, telemetry

    total_acc = rf_val_acc + gbm_val_acc
    if total_acc <= 0:
        w_rf, w_gbm = 0.5, 0.5
    else:
        w_rf, w_gbm = rf_val_acc / total_acc, gbm_val_acc / total_acc
    w_rf = max(weight_floor, w_rf)
    w_gbm = max(weight_floor, w_gbm)
    norm = w_rf + w_gbm
    w_rf, w_gbm = w_rf / norm, w_gbm / norm

    blended = BlendedClassifier(rf, gbm, w_rf, w_gbm)
    telemetry = {
        "rf_val_acc": round(rf_val_acc, 4),
        "gbm_val_acc": round(gbm_val_acc, 4),
        "blend_w_rf": round(w_rf, 4),
        "blend_w_gbm": round(w_gbm, 4),
        "blend_status": "blended",
    }
    logger.info(
        "BLENDED_CLASSIFIER_FIT: rf_val_acc=%.4f gbm_val_acc=%.4f w_rf=%.3f w_gbm=%.3f",
        rf_val_acc,
        gbm_val_acc,
        w_rf,
        w_gbm,
    )
    return blended, telemetry


def compute_disagreement(model: Any, X: Any) -> np.ndarray:
    """Safe wrapper: returns model.disagreement(X) for a BlendedClassifier, or
    an all-zero array (perfect agreement / no discount) for any other model —
    e.g. a plain RandomForestClassifier fallback from build_blended_classifier's
    safe-degradation path, or an older pickled artifact predating this feature."""
    n = len(X) if hasattr(X, "__len__") else X.shape[0]
    if isinstance(model, BlendedClassifier):
        try:
            return model.disagreement(X)
        except Exception as exc:
            logger.debug("BLENDED_CLASSIFIER_DISAGREEMENT_FAILED: %s", exc)
            return np.zeros(n)
    return np.zeros(n)


__all__ = ["BlendedClassifier", "build_blended_classifier", "compute_disagreement"]
