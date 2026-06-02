import logging
import os
from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS as TOP10
from backend.services.ai_live_models_store import AILiveSignal, SessionLocal  # type: ignore[import-not-found]

from .app_factory import app

logger = logging.getLogger("main")


def _hget_redis(redis_obj: Any, key: str, field: str) -> str | None:
    try:
        raw = redis_obj.hget(key, field)
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _count_ai_signals_pending() -> int:
    try:
        with SessionLocal() as s:  # type: ignore[misc]
            return int(s.query(AILiveSignal).filter_by(consumed=False).count())
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return 0


def _get_model_version() -> str:
    try:
        return os.getenv("AI_MODEL_VERSION", "unknown")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return "unknown"


# Health endpoints removed - consolidated to 3 critical endpoints in app_factory.py


@app.get("/api/version")
async def get_version() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "build_date": "2024-06-22",
        "environment": "production",
        "features": [
            "real-time trading",
            "AI-powered analytics",
            "social trading",
            "mobile PWA support",
            "advanced order types",
            "risk management",
            "auto-trading bots",
        ],
        "universe": TOP10,
    }
