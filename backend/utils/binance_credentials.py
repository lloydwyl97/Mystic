"""Canonical Binance.US credential resolution.

Single source of truth: ``BINANCE_US_API_KEY`` / ``BINANCE_US_SECRET_KEY``.
Legacy aliases (``BINANCEUS_*``, ``BINANCE_API_KEY`` / ``BINANCE_SECRET``) remain
read-only fallbacks so older env files keep working until removed.
"""

from __future__ import annotations

import os


def get_binance_us_api_key() -> str:
    return (
        (os.getenv("BINANCE_US_API_KEY") or "").strip()
        or (os.getenv("BINANCEUS_API_KEY") or "").strip()
        or (os.getenv("BINANCE_API_KEY") or "").strip()
    )


def get_binance_us_secret_key() -> str:
    return (
        (os.getenv("BINANCE_US_SECRET_KEY") or "").strip()
        or (os.getenv("BINANCE_US_API_SECRET") or "").strip()
        or (os.getenv("BINANCEUS_API_SECRET") or "").strip()
        or (os.getenv("BINANCE_SECRET") or "").strip()
        or (os.getenv("BINANCE_SECRET_KEY") or "").strip()
    )


def get_binance_us_credentials() -> tuple[str, str]:
    return get_binance_us_api_key(), get_binance_us_secret_key()


def sync_binance_us_env_aliases() -> None:
    """Publish canonical credentials into legacy alias env vars for old readers."""
    api_key, secret = get_binance_us_credentials()
    if api_key:
        os.environ["BINANCE_US_API_KEY"] = api_key
        os.environ["BINANCEUS_API_KEY"] = api_key
        # Only fill generic alias when empty or already equal — avoid clobbering a distinct key.
        generic = (os.getenv("BINANCE_API_KEY") or "").strip()
        if not generic or generic == api_key:
            os.environ["BINANCE_API_KEY"] = api_key
    if secret:
        os.environ["BINANCE_US_SECRET_KEY"] = secret
        os.environ["BINANCEUS_API_SECRET"] = secret
        generic_s = (os.getenv("BINANCE_SECRET") or "").strip()
        if not generic_s or generic_s == secret:
            os.environ["BINANCE_SECRET"] = secret
            os.environ["BINANCE_SECRET_KEY"] = secret


__all__ = [
    "get_binance_us_api_key",
    "get_binance_us_secret_key",
    "get_binance_us_credentials",
    "sync_binance_us_env_aliases",
]
