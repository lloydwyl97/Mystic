import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e


def _to_ccxt_symbol(base: str, quote: str) -> str:
    return f"{base.upper()}/{quote.upper()}"


def _http_get(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        resp = httpx.get(url, params=params or {}, headers=headers or {}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning("Unexpected JSON response type")
            result = {}
        else:
            result = data
    except httpx.HTTPStatusError as e:
        logger.exception(f"HTTP error response: {e}")
        return {}
    except httpx.RequestError as e:
        logger.exception(f"HTTP request failed: {e}")
        return {}
    except ValueError as e:
        logger.exception(f"JSON decode failed: {e}")
        return {}
    else:
        return result


def fetch_eth_gas_price() -> int:
    # Use Etherscan API V2 endpoint
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        logger.warning("ETHERSCAN_API_KEY not set")
        return 0

    url = f"https://api.etherscan.io/v2/api?chainid=1&module=gastracker&action=gasoracle&apikey={api_key}"
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

        # Check API status
        if data.get("status") != "1":
            logger.warning(f"Etherscan API error: {data.get('message', 'unknown')}")
            return 0

        result = data.get("result", {})
        if not isinstance(result, dict):
            logger.warning(f"Etherscan API returned non-dict result: {result}")
            return 0

        raw = result.get("ProposeGasPrice")
        if raw is not None:
            # V2 API returns ProposeGasPrice in Gwei as decimal (e.g., "1.35")
            try:
                price_gwei = int(float(raw))
                return max(1, price_gwei)  # At least 1 Gwei
            except (TypeError, ValueError):
                return 0
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Error parsing Etherscan gas price: {e}")
        return 0
    except Exception as e:
        logger.warning(f"Unexpected error fetching gas price: {e}")
        return 0
    else:
        return 0


def fetch_whale_alerts() -> list[dict[str, Any]]:
    api_key = os.getenv("WHALE_ALERT_API_KEY", "")
    if not api_key:
        logger.debug("WHALE_ALERT_API_KEY not set; returning empty whale alerts")
        return []

    try:
        url = "https://api.whale-alert.io/v1/transactions"
        params = {
            "api_key": api_key,
            "min_value": 1_000_000,
        }
        data = _http_get(url, params=params, headers={"User-Agent": "mystic-trading/1.0"})

        # Handle error responses
        if not isinstance(data, dict):
            logger.warning(f"Whale Alert API returned non-dict response: {type(data)}")
            return []

        txs = data.get("transactions")
        return txs if isinstance(txs, list) else []
    except Exception as e:
        logger.warning(f"Error fetching whale alerts: {e}")
        return []


def onchain_signal_check() -> dict[str, int]:
    gas = fetch_eth_gas_price()
    whales = fetch_whale_alerts()
    whale_count = len(whales)
    logger.info(f"On-chain signals: gas={gas} gwei, whale_txs={whale_count}")
    return {"gas": gas, "whales": whale_count}
