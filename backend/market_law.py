import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def validate_trade(agent_a: str, agent_b: str, terms: Mapping) -> bool:
    try:
        logger.info("[LAW] Reviewing trade between %s and %s", agent_a, agent_b)
        if not isinstance(terms, Mapping):
            logger.warning("[LAW] Invalid terms type")
            return False

        fairness = terms.get("fairness", None)
        try:
            fairness_val = float(fairness)
        except (TypeError, ValueError):
            logger.warning("[LAW] Missing or non-numeric fairness")
            return False

        if not (0.0 <= fairness_val <= 1.0):
            logger.warning("[LAW] Fairness out of range: %s", fairness_val)
            return False

        if fairness_val > 0.8:
            logger.info("[LAW] Approved")
            result = True
        else:
            logger.info("[LAW] Denied")
            result = False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[LAW] Validation error: %s", e)
        return False
    else:
        return result
