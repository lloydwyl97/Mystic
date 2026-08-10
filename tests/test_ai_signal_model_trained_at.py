"""Batch 8: model_trained_at + model_accuracy round-trip in the AI signal payload.

Guards:
1. AISignalGenerator has model_trained_ats / model_accuracies dicts and
   clears them on reinit.
2. The signal_data payload dict template includes model_trained_at and
   model_accuracy keys — asserted via source-code contract check so the
   test does not need a live Redis or model file.
3. The read side (portfolio_engine._stamp_low_mfe_outcome_explain and the
   BUY explainability path) has a working path for model_trained_at.
"""

from __future__ import annotations

import inspect
from pathlib import Path


def test_signal_generator_declares_trained_at_and_accuracy_dicts():
    from backend.services.ai_signal_generator import RealTimeAISignalGenerator

    src = Path(inspect.getsourcefile(RealTimeAISignalGenerator)).read_text()
    assert "self.model_trained_ats: dict[str, str]" in src
    assert "self.model_accuracies: dict[str, float]" in src
    # Cleared on reinit
    assert "self.model_trained_ats.clear()" in src
    assert "self.model_accuracies.clear()" in src


def test_signal_payload_includes_model_trained_at_and_accuracy():
    """Ensure signal_data emitted to Redis carries the new keys."""
    from backend.services.ai_signal_generator import RealTimeAISignalGenerator

    src = Path(inspect.getsourcefile(RealTimeAISignalGenerator)).read_text()
    assert '"model_trained_at":' in src
    assert '"model_accuracy":' in src
    assert "self.model_trained_ats.get(slot" in src
    assert "self.model_accuracies.get(slot" in src


def test_trained_at_landed_in_pickle_metadata():
    """Confirm the trainer stamps trained_at into the per-coin model pickle.

    The trainer writes model_data with a 'trained_at' key. Assert this key
    is present in the source so the runtime keeps writing it.
    """
    from backend import ai_training_pipeline

    src = Path(inspect.getsourcefile(ai_training_pipeline)).read_text()
    assert '"trained_at": _now_iso()' in src


def test_portfolio_engine_reads_model_trained_at_from_dd():
    """BUY-persistence must map decision_data.model_trained_at → explainability.

    The code was refactored into a helper (`_stamp_model_metadata_explain`) that
    reads `_dd.get("model_trained_at")` and assigns it via `explainability.model_trained_at = _trained`,
    with Redis-hash fallback for empty values. Assert both halves of that
    wiring still exist so the batch-8 metadata fix does not silently regress.
    """
    from backend.services import portfolio_engine as pe

    src = Path(inspect.getsourcefile(pe)).read_text()
    assert '_dd.get("model_trained_at")' in src, "dd.model_trained_at read missing"
    assert "explainability.model_trained_at = _trained" in src, "explainability.model_trained_at assignment missing"
