"""Configuration module for the trading system."""

try:
    from .settings import settings
except ImportError as e:
    import logging

    logging.warning(f"Failed to import settings: {e}")
    settings = None  # type: ignore[assignment]

__all__ = ["settings"]
