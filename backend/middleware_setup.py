import logging

from backend.middleware.manager import get_middleware_manager

from .app_factory import app

logger = logging.getLogger("main")

try:
    middleware_manager = get_middleware_manager()
    if middleware_manager is not None:
        register_all = getattr(middleware_manager, "register_all", None)
        if callable(register_all):
            register_all(app)
            logger.info("Middleware registered successfully")
        else:
            logger.info("Middleware manager has no 'register_all' method; skipping middleware registration")
    else:
        logger.info("No middleware manager available; skipping middleware registration")
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    logger.exception("Middleware registration failed")

__all__ = ["app"]
