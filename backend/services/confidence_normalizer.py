"""
CRITICAL #2 FIX: Centralized Confidence Normalization

Prevents double-scaling and ensures consistent confidence semantics
across all signal processing and position sizing.

CANONICAL SCALE: 0.0-1.0
- 0.0 = no confidence, 1.0 = full confidence
- All internal gates, sizing, and Redis writes use this scale
- Inputs: accepts both 0-1 floats and 0-100 percentages (auto-converts)
- Use ConfidenceNormalizer.normalize() as the single entry point
"""

import logging
import os

from backend.config.signal_thresholds import min_confidence_buy

logger = logging.getLogger(__name__)


class ConfidenceNormalizer:
    """
    Single source of truth for confidence scaling.

    Ensures that confidence values are normalized consistently:
    - No double application of multipliers
    - Unified min/max bounds
    - Config-driven adjustments
    """

    # Upper clamp only by default; lower clamp stays 0 so weak model probs are not lifted to "pass" gates.
    CLAMP_MIN = float(os.getenv("CONFIDENCE_NORMALIZER_CLAMP_MIN", "0.0"))
    MAX_CONFIDENCE = float(os.getenv("MAX_CONFIDENCE", "0.95"))

    # Feature-specific boost (124-feature models)
    BOOST_124_ENABLED = os.getenv("CONFIDENCE_BOOST_124_FEATURES", "true").lower() == "true"
    BOOST_124_FACTOR = float(os.getenv("CONFIDENCE_BOOST_124_FACTOR", "1.05"))
    BOOST_124_CAP = float(os.getenv("CONFIDENCE_BOOST_124_CAP", "0.95"))

    # Canonical scale: 0.0 = no confidence, 1.0 = full confidence. All internal processing uses this.
    # Accepts both 0-1 floats and 0-100 percentages; normalizes to 0-1 before clamping.

    @classmethod
    def normalize(
        cls,
        raw_confidence: float,
        already_scaled: bool = False,
    ) -> float:
        """
        Normalize a scalar to canonical 0.0-1.0 scale (e.g. winner-class probability).

        124-feature boost is not applied here (see portfolio_engine sizing).

        Args:
            raw_confidence: Raw value (0.0-1.0 OR 0-100 percentage)
            already_scaled: If True, skip percentage auto-scaling

        Returns:
            Value clamped to [CLAMP_MIN, MAX_CONFIDENCE]
        """
        if not isinstance(raw_confidence, (int, float)):
            logger.warning(f"CRITICAL #2: Invalid confidence type {type(raw_confidence)}, using CLAMP_MIN")
            return cls.CLAMP_MIN

        confidence = float(raw_confidence)

        if not already_scaled:
            # Accept 0-100 scale: if > 1, treat as percentage
            if confidence > 1.0 and confidence <= 100.0:
                confidence = confidence / 100.0

        # Clamp to bounds (do not inflate low probabilities to MIN_CONFIDENCE_BUY)
        confidence = max(cls.CLAMP_MIN, min(cls.MAX_CONFIDENCE, confidence))

        # NOTE: 124-feature boost is intentionally NOT applied here.
        # It is handled in position sizing (portfolio_engine.py) to avoid double-scaling.

        return confidence

    @classmethod
    def is_above_threshold(cls, confidence: float) -> bool:
        """Legacy winner-probability check vs MIN_CONFIDENCE_BUY (not used for BUY admission; see buy_admission)."""
        return confidence >= min_confidence_buy()

    @classmethod
    def log_config(cls):
        """Log configuration at startup for transparency."""
        logger.info(f"CRITICAL #2: Confidence clamp: min={cls.CLAMP_MIN}, max={cls.MAX_CONFIDENCE}, buy_floor={min_confidence_buy()}")
        if cls.BOOST_124_ENABLED:
            logger.info(f"  124-feature boost: factor={cls.BOOST_124_FACTOR}, cap={cls.BOOST_124_CAP}")
        else:
            logger.info("  124-feature boost: DISABLED")
