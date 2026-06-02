import logging

logger = logging.getLogger(__name__)


def send_global_signal(code="UNITY", message="AI Sovereign Civilization Online"):
    logger.info(f"[BROADCAST] {code} → {message}")
