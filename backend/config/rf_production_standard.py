"""
[MYSTIC_CORE_TAG: LEGACY_NOT_LIVE] Describes an older multi-symbol / scalers-dict artifact
shape. Live per-coin packs use ``models/active/{SYMBOL}_direction.pkl`` with a single scaler;
do not treat this doc as the on-disk contract without verifying keys.

Random Forest — production readiness for **live ensemble arbitration**.

This is the explicit Mystic standard: the RF head must not be treated as controlling
BUY/HOLD quality across the full universe until the artifact and training contract below are met.

Training contract (see ``backend.ai_training_pipeline``):
  - Every persisted sample used for RF training carries a canonical ``symbol`` (``TRADING_SYMBOLS`` form).
  - Next-bar labels are computed **only within each symbol's time-ordered series** (no cross-asset shuffling).
  - One shared ``RandomForestClassifier`` is trained on rows that were scaled with **that row's symbol's**
    ``StandardScaler`` (fit on training data for that symbol), then pooled.
  - The saved RF pickle must include a ``scalers`` mapping with an entry for **each** ``TRADING_SYMBOLS``
    symbol (values may reference a shared global scaler only when that symbol had insufficient rows at
    train time — ops should verify retrain coverage).

Inference contract:
  - ``ai_signal_generator`` applies ``scalers[symbol]`` for RF ``predict_proba`` only; deep models stay on raw features.

Operational policy:
  - Until ``rf_live_artifact_production_grade`` returns True for the loaded pack, keep ``AI_ENSEMBLE_RF_WEIGHT``
    at the default demotion (see ``resolve_ai_ensemble_weights``). Raising RF weight without meeting this
    standard risks the prior failure mode (nominal price tier / global scaler distortion).
"""

from __future__ import annotations

from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS


def rf_live_artifact_production_grade(artifact: dict[str, Any]) -> tuple[bool, str]:
    """
    Return (True, "ok") if the on-disk RF pack matches the production scaler contract for the
    configured universe. Legacy ``{"model", "scaler"}`` packs are **not** production-grade for
    cross-asset live arbitration.
    """
    if not isinstance(artifact, dict):
        return False, "artifact is not a dict"
    if artifact.get("model") is None:
        return False, "missing model"
    sm = artifact.get("scalers")
    if not isinstance(sm, dict) or len(sm) == 0:
        return False, "missing or empty scalers dict (legacy single-scaler pack)"
    missing = [s for s in TRADING_SYMBOLS if s not in sm]
    if missing:
        return False, f"scalers incomplete (missing {len(missing)} symbol(s), e.g. {missing[:2]})"
    return True, "ok"
