from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe or _to_ccxt_symbol: {e}"
    raise RuntimeError(msg) from e


PORTFOLIO_FILE = "./data/portfolio.json"
THRESHOLD = 0.7
INTERVAL = 600
PING_FILE = "./logs/portfolio_ai_balance.ping"
BALANCE_LOG_FILE = "./logs/balance_log.jsonl"
# Use MYSTIC_BACKEND environment variable or default
BACKEND_BASE_URL = os.getenv("MYSTIC_BACKEND", "http://localhost:8000")

Path("./data").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("portfolio_ai_balance")


def create_ping_file(stablecoin_ratio: float, total_value: float, rebalance_needed: bool) -> None:
    try:
        ping_path = Path(PING_FILE)
        ping_path.parent.mkdir(parents=True, exist_ok=True)
        with ping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "stablecoin_ratio": stablecoin_ratio,
                    "total_value": total_value,
                    "rebalance_needed": rebalance_needed,
                },
                f,
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Ping file error: %s", e)


def load_portfolio() -> dict:
    try:
        portfolio_path = Path(PORTFOLIO_FILE)
        if portfolio_path.exists():
            with portfolio_path.open(encoding="utf-8") as f:
                data = json.load(f)
                if "coins" in data and isinstance(data["coins"], dict):
                    return data
        empty = {
            "coins": {"USDT": 0.0},
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        save_portfolio(empty)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error loading portfolio: %s", e)
        return {
            "coins": {"USDT": 0.0},
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
    else:
        return empty


def save_portfolio(portfolio: dict) -> None:
    try:
        portfolio["last_updated"] = datetime.now(timezone.utc).isoformat()
        portfolio_path = Path(PORTFOLIO_FILE)
        portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        with portfolio_path.open("w", encoding="utf-8") as f:
            json.dump(portfolio, f, indent=2)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error saving portfolio: %s", e)


async def _fetch_price_usdt(base_asset: str) -> float:
    try:
        symbol = f"{base_asset.upper()}USDT"
        url = f"{BACKEND_BASE_URL}/api/live/market-data/{symbol}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
        if resp.status_code == 200:
            payload = resp.json()
            price = float(payload.get("price", 0.0) or 0.0)
            return max(price, 0.0)
        logger.warning("Live price fetch non-200 for %s: %s", symbol, resp.status_code)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("Live price fetch error for %s: %s", base_asset, e)
    return 0.0


async def _portfolio_total_value_usdt(
    positions: dict[str, float],
) -> tuple[float, float]:
    usdt_balance = float(positions.get("USDT", 0.0) or 0.0)
    total_value = usdt_balance
    for asset, qty in positions.items():
        if str(asset).upper() == "USDT":
            continue
        try:
            quantity = float(qty or 0.0)
            if quantity <= 0.0:
                continue
            price = await _fetch_price_usdt(str(asset).upper())
            if price <= 0.0:
                continue
            total_value += quantity * price
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Valuation error for %s: %s", asset, e)
    return total_value, usdt_balance


async def evaluate_portfolio() -> tuple[float, float, bool, dict[str, float]]:
    try:
        portfolio = load_portfolio()
        coins = portfolio.get("coins", {})
        if not isinstance(coins, dict):
            coins = {"USDT": 0.0}
        total_value, usdt_value = await _portfolio_total_value_usdt(coins)
        if total_value <= 0.0:
            logger.info("No portfolio value available for evaluation")
            _log_balance(0.0, 0.0, False, 1 - THRESHOLD)
            return 0.0, 0.0, False, {"target_usdt": 0.0, "usdt_delta": 0.0}
        stablecoin_ratio = usdt_value / total_value
        target_ratio = 1 - THRESHOLD
        rebalance_needed = stablecoin_ratio < target_ratio
        plan: dict[str, float] = {"target_usdt": 0.0, "usdt_delta": 0.0}
        if rebalance_needed:
            target_usdt = total_value * target_ratio
            usdt_delta = target_usdt - usdt_value
            plan["target_usdt"] = round(target_usdt, 8)
            plan["usdt_delta"] = round(usdt_delta, 8)
        _log_balance(stablecoin_ratio, total_value, rebalance_needed, target_ratio, plan)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Portfolio evaluation error: %s", e)
        return 0.0, 0.0, False, {"target_usdt": 0.0, "usdt_delta": 0.0}
    else:
        return stablecoin_ratio, total_value, rebalance_needed, plan


def _log_balance(
    stablecoin_ratio: float,
    total_value: float,
    rebalance_needed: bool,
    target_ratio: float,
    plan: dict[str, float] | None = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stablecoin_ratio": stablecoin_ratio,
        "total_value": total_value,
        "rebalance_needed": rebalance_needed,
        "target_ratio": target_ratio,
    }
    if plan:
        record["plan"] = plan
    try:
        balance_log_path = Path(BALANCE_LOG_FILE)
        balance_log_path.parent.mkdir(parents=True, exist_ok=True)
        with balance_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error writing balance log: %s", e)
    level = logging.INFO
    if rebalance_needed:
        level = logging.WARNING
    logger.log(
        level,
        "Evaluation | total_value_usdt=%.4f | stablecoin_ratio=%.4f | target_ratio=%.4f | rebalance=%s",
        total_value,
        stablecoin_ratio,
        target_ratio,
        rebalance_needed,
    )
    if plan and rebalance_needed:
        logger.warning(
            "Rebalance plan | target_usdt=%.4f | usdt_delta=%.4f",
            plan.get("target_usdt", 0.0),
            plan.get("usdt_delta", 0.0),
        )


def main() -> None:
    logger.info("Portfolio AI Balance started")
    logger.info("Evaluation interval: %s seconds", INTERVAL)
    logger.info(
        "Rebalancing threshold (non-stable): %.3f | Target stable ratio: %.3f",
        THRESHOLD,
        1 - THRESHOLD,
    )
    while True:
        try:
            ratio, total, rebalance, _ = asyncio.run(evaluate_portfolio())
            create_ping_file(ratio, total, rebalance)
        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Main loop error: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
