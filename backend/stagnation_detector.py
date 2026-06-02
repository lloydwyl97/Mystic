import json
import logging
from pathlib import Path

from ai_auto_learner import AIAutoLearner
from notifier import send_alert

logger = logging.getLogger(__name__)

STATE_FILE = "ai_model_state.json"


def detect_stagnation() -> bool:
    if not Path(STATE_FILE).exists():
        return False
    try:
        state_path = Path(STATE_FILE)
        with state_path.open(encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False
    try:
        adjustments = int(state.get("adjustment_count", 0) or 0)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        adjustments = 0
    if adjustments >= 10:
        state["confidence_threshold"] = 0.75
        state["adjustment_count"] = 0
        try:
            state_path = Path(STATE_FILE)
            with state_path.open("w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return False
        try:
            send_alert("[WARN] AI strategy auto-reset due to stagnation.")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to send stagnation alert.")
        logger.info("[Stagnation] AI auto-reset triggered.")
        return True
    return False


def check_performance_plateau() -> bool:
    try:
        learner = AIAutoLearner()
        summary = {}
        if hasattr(learner, "logger") and hasattr(learner.logger, "get_summary"):
            summary = learner.logger.get_summary() or {}
        if not isinstance(summary, dict):
            summary = {}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False
    try:
        total_trades = int(summary.get("total_trades", 0) or 0)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False
    if total_trades > 50:
        avg_profit = summary.get("avg_profit", 0)
        try:
            if abs(float(avg_profit)) < 0.1:
                try:
                    send_alert("[DOWN] AI performance plateau detected. Consider strategy review.")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Failed to send performance plateau alert.")
                logger.info("[Stagnation] Performance plateau detected.")
                return True
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return False
    return False
