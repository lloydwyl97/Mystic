from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any
from urllib.parse import urlencode

from backend.utils.enhanced_logging import log_operation_performance

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

RECV_WINDOW_MS_DEFAULT = 5000


@log_operation_performance("create_signature")
def create_signature(api_secret: str, message: str) -> str:
    """
    Binance.US HMAC-SHA256 signature (hex digest) over the query string.
    """
    try:
        return hmac.new(api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Error creating signature: {e}"
        raise ValueError(msg) from e  # no unreachable code after raise/return


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _ensure_recv_window(params: dict[str, Any], recv_window_ms: int | None) -> None:
    if "recvWindow" not in params:
        params["recvWindow"] = int(recv_window_ms if recv_window_ms is not None else RECV_WINDOW_MS_DEFAULT)


def _canonical_query(params: dict[str, Any]) -> str:
    # Binance requires URL-encoded query string in natural key order (python dict preserves insertion order)
    # Values must be str/int/float/bool
    enc_ready: dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            enc_ready[k] = "true" if v else "false"
        else:
            enc_ready[k] = str(v)
    return urlencode(enc_ready, doseq=True, safe="")  # ccxt calls only receive BASE/QUOTE elsewhere


def build_signed_params(
    api_key: str,
    api_secret: str,
    params: dict[str, Any] | None = None,
    *,
    add_timestamp: bool = True,
    recv_window_ms: int | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Build signed query params and required headers for Binance.US signed endpoints.
    Returns (headers, params_with_signature).
    """
    if not api_key or not api_secret:
        msg = "API key and secret are required"
        raise ValueError(msg)

    p: dict[str, Any] = dict(params or {})
    if add_timestamp:
        p["timestamp"] = _timestamp_ms()
    _ensure_recv_window(p, recv_window_ms)

    query = _canonical_query(p)
    sig = create_signature(api_secret, query)
    p["signature"] = sig

    headers = {"X-MBX-APIKEY": api_key}
    return headers, p


class SignatureManager:
    """
    Helper for Binance.US request signing.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        if not api_key or not api_secret:
            msg = "API key and secret are required"
            raise ValueError(msg)
        self.api_key = api_key
        self.api_secret = api_secret

    def headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def sign(self, params: dict[str, Any] | None = None, *, recv_window_ms: int | None = None) -> dict[str, Any]:
        _, signed = build_signed_params(
            self.api_key,
            self.api_secret,
            params=params,
            add_timestamp=True,
            recv_window_ms=recv_window_ms,
        )
        return signed


# Quick test checklist:
# - No binance/binanceus string leaks—only binance_us via EXCHANGE_ID.
# - No unreachable code after returns.
# - Logging has no weird characters.
