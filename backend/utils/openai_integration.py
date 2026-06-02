"""
OpenAI Integration for Strategy Descriptions

Generates concise, human-readable descriptions for evolved strategies using OpenAI's chat models.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


async def generate_openai_description(
    strategy: dict[str, Any],
    parent: str = "",
    backtest_results: dict[str, Any] | None = None,
) -> str:
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured; using fallback description")
        return generate_fallback_description(strategy, parent)

    try:
        prompt = create_strategy_prompt(strategy, parent, backtest_results)
        response = await call_openai_api(prompt)
        if response and isinstance(response.get("choices"), list) and response["choices"]:
            msg = response["choices"][0].get("message", {})
            content = (msg.get("content") or "").strip()
            if content:
                return content
            logger.error("OpenAI response had no content")
            return generate_fallback_description(strategy, parent)
        logger.error("OpenAI API returned no choices")
        return generate_fallback_description(strategy, parent)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"OpenAI description generation failed: {e}")
        return generate_fallback_description(strategy, parent)


def create_strategy_prompt(
    strategy: dict[str, Any],
    parent: str,
    backtest_results: dict[str, Any] | None,
) -> str:
    strategy_type = str(strategy.get("strategy_type", "unknown"))
    params = strategy.get("parameters", strategy)

    parent_line = f"Parent Strategy: {parent}" if parent else "This is a newly generated strategy (no parent)."

    if backtest_results:
        win_rate = float(backtest_results.get("win_rate", 0.0))
        total_profit = float(backtest_results.get("total_profit", 0.0))
        num_trades = int(backtest_results.get("num_trades", 0))
        backtest_line = f"Backtest Results: Win Rate: {win_rate:.1%}, Profit: {total_profit:.2f}%, Trades: {num_trades}"
    else:
        backtest_line = "Backtest Results: N/A"

    return (
        "You are an expert quantitative trading analyst. Analyze the following trading strategy and write a brief, "
        "professional description suitable for technical documentation.\n\n"
        f"Strategy Type: {strategy_type}\n"
        f"Parameters (JSON): {json.dumps(params, indent=2)}\n"
        f"{parent_line}\n"
        f"{backtest_line}\n\n"
        "Return 2-3 sentences covering:\n"
        "1) What the strategy does and its core logic\n"
        "2) The key parameters and why they matter\n"
        "3) The market conditions where it tends to perform well\n"
        "If there is a parent, mention how this version improves upon it. Be concise and precise."
    )


async def call_openai_api(prompt: str) -> dict[str, Any] | None:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert quantitative trading analyst specializing in algorithmic trading strategies.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 220,
        "temperature": 0.6,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"OpenAI API request failed: {e}")
        return None


def generate_fallback_description(strategy: dict[str, Any], parent: str = "") -> str:
    strategy_type = str(strategy.get("strategy_type", "unknown"))
    params = strategy.get("parameters", strategy)

    parts: list[str] = []
    if strategy_type == "breakout":
        parts.append(
            f"This breakout strategy scans a lookback window of {params.get('lookback_period', 'N/A')} periods "
            f"and triggers entries when price exceeds an {params.get('entry_threshold', 'N/A')}x threshold.",
        )
        parts.append(f"Risk controls include a stop loss at {params.get('stop_loss', 'N/A')}% and take profit at {params.get('take_profit', 'N/A')}%.")
    elif strategy_type == "ema_crossover":
        parts.append(f"This EMA crossover strategy compares a fast EMA ({params.get('fast_ema', 'N/A')}) to a slow EMA ({params.get('slow_ema', 'N/A')}) to capture trend shifts.")
        parts.append(f"It applies risk management with a {params.get('stop_loss', 'N/A')}% stop loss and {params.get('take_profit', 'N/A')}% take profit.")
    elif strategy_type == "rsi_threshold":
        parts.append(f"This RSI threshold strategy buys when RSI falls below {params.get('rsi_buy', 'N/A')} and sells when RSI rises above {params.get('rsi_sell', 'N/A')}.")
        parts.append(f"It uses a lookback period of {params.get('lookback_period', 'N/A')} for stability.")
    else:
        kv = ", ".join(f"{k}={v}" for k, v in params.items())
        parts.append(f"This {strategy_type} strategy runs with parameters: {kv}.")

    if parent:
        parts.append(f"It evolves from {parent} with refined parameters for more robust execution.")

    parts.append("Description generated without OpenAI (fallback).")
    return " ".join(parts)


def is_openai_available() -> bool:
    return bool(OPENAI_API_KEY)


def generate_strategy_description(
    strategy: dict[str, Any],
    parent: str = "",
    backtest_results: dict[str, Any] | None = None,
) -> str:
    return generate_openai_description(strategy, parent, backtest_results)
