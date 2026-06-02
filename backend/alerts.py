from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe or _to_ccxt_symbol: {e}"
    raise RuntimeError(msg) from e


# ---- Logging ----
logger = logging.getLogger(__name__)

# ---- Env ----
load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


# ---- Internal utils ----
def _sanitize_text(v: Any) -> str:
    """
    Convert to clean ASCII-ish text for logs and payloads (no weird characters).
    """
    s = str(v) if v is not None else ""
    try:
        # keep only basic printable range; replace others with '?'
        return "".join(ch if 32 <= ord(ch) <= 126 else "?" for ch in s)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return s


def _fmt_pct(x: Any, default: str = "0.0%") -> str:
    try:
        return f"{float(x):.1%}"
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _fmt_num(x: Any, digits: int = 2, default: str = "0.00") -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _embed_fields_from_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in pairs:
        out.append(
            {
                "name": _sanitize_text(p.get("name", "")),
                "value": _sanitize_text(p.get("value", "")),
                "inline": bool(p.get("inline", False)),
            },
        )
    return out


def _post_discord(payload: dict[str, Any]) -> bool:
    if not DISCORD_WEBHOOK_URL or not DISCORD_WEBHOOK_URL.startswith("https://"):
        logger.warning("Discord webhook URL not configured correctly")
        return False
    try:
        with httpx.Client() as client:
            resp = client.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if 200 <= resp.status_code < 300:
            result = True
        else:
            logger.error("Discord webhook error %s: %s", resp.status_code, _sanitize_text(resp.text))
            result = False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Discord webhook exception: %s", _sanitize_text(e))
        return False
    else:
        return result


# ---- Public API ----
def send_discord_alert(message: str, embed_data: dict[str, Any] | None = None) -> bool:
    """
    Send a Discord alert via webhook.

    Args:
        message: Plain text message content.
        embed_data: Optional embed dict with keys title, description, color, fields, footer, timestamp.

    Returns:
        True on success, False otherwise.
    """
    content = _sanitize_text(message)
    payload: dict[str, Any] = {"content": content}

    if embed_data:
        title = _sanitize_text(embed_data.get("title", "Alert"))
        description = _sanitize_text(embed_data.get("description", ""))
        try:
            color = int(embed_data.get("color", 0x0080FF))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            color = 0x0080FF
        fields = _embed_fields_from_pairs(embed_data.get("fields", []))
        footer_text = _sanitize_text((embed_data.get("footer", {}) or {}).get("text", f"Mystic AI Trading • {EXCHANGE_ID}"))
        embed: dict[str, Any] = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": footer_text},
        }
        ts = embed_data.get("timestamp")
        if isinstance(ts, str) and ts:
            embed["timestamp"] = _sanitize_text(ts)
        payload["embeds"] = [embed]

    ok = _post_discord(payload)
    if ok:
        logger.info("Discord alert sent: %s", content[:80])
    else:
        logger.error("Discord alert failed to send")
    return ok


def alert_strategy_mutation(mutation_info: dict[str, Any]) -> bool:
    """
    Notify on new strategy mutation.
    Expected keys: name, parent_strategy, parent_win_rate, parent_avg_profit
    """
    name = _sanitize_text(mutation_info.get("name", "unknown"))
    parent = _sanitize_text(mutation_info.get("parent_strategy", "unknown"))
    win_rate = _fmt_pct(mutation_info.get("parent_win_rate"))
    avg_profit = _fmt_num(mutation_info.get("parent_avg_profit"))

    message = f"NEW STRATEGY MUTATION\nName: {name}\nParent: {parent}\nParent Win Rate: {win_rate}\nParent Avg Profit: {avg_profit}"

    embed_data = {
        "title": "Strategy Evolution",
        "description": f"New mutation created from {parent}",
        "color": 0x00A65A,
        "fields": _embed_fields_from_pairs(
            [
                {"name": "Mutation Name", "value": name, "inline": True},
                {"name": "Parent Strategy", "value": parent, "inline": True},
                {
                    "name": "Parent Performance",
                    "value": f"Win Rate: {win_rate}\nAvg Profit: {avg_profit}",
                    "inline": False,
                },
            ],
        ),
    }
    return send_discord_alert(message, embed_data)


def alert_strategy_deactivation(strategy_info: dict[str, Any]) -> bool:
    """
    Notify on strategy deactivation.
    Expected keys: name, win_rate, avg_profit, reason
    """
    name = _sanitize_text(strategy_info.get("name", "unknown"))
    win_rate = _fmt_pct(strategy_info.get("win_rate"))
    avg_profit = _fmt_num(strategy_info.get("avg_profit"))
    reason = _sanitize_text(strategy_info.get("reason", "unspecified"))

    message = f"STRATEGY DEACTIVATED\nName: {name}\nWin Rate: {win_rate}\nAvg Profit: {avg_profit}\nReason: {reason}"

    embed_data = {
        "title": "Strategy Deactivation",
        "description": f"Strategy {name} has been deactivated",
        "color": 0xD32F2F,
        "fields": _embed_fields_from_pairs(
            [
                {"name": "Strategy Name", "value": name, "inline": True},
                {
                    "name": "Performance",
                    "value": f"Win Rate: {win_rate}\nAvg Profit: {avg_profit}",
                    "inline": True,
                },
                {"name": "Deactivation Reason", "value": reason, "inline": False},
            ],
        ),
    }
    return send_discord_alert(message, embed_data)


def alert_trade_execution(trade_info: dict[str, Any]) -> bool:
    """
    Notify on trade execution.
    Expected keys: coin, strategy_name, entry_price, exit_price, profit, success
    """
    coin = _sanitize_text(trade_info.get("coin", "UNKNOWN/USDT"))
    pair = _to_ccxt_symbol(coin)
    strategy_name = _sanitize_text(trade_info.get("strategy_name", "unknown"))
    entry = _fmt_num(trade_info.get("entry_price"))
    exitp = _fmt_num(trade_info.get("exit_price"))
    profit = _fmt_num(trade_info.get("profit"))
    success = bool(trade_info.get("success", False))

    status = "SUCCESS" if success else "FAILED"
    try:
        profit_val = float(trade_info.get("profit", 0) or 0)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        profit_val = 0.0
    direction = "GAIN" if profit_val > 0 else "LOSS"

    message = f"TRADE EXECUTED\nStatus: {status}\nPair: {pair}\nStrategy: {strategy_name}\nEntry: {entry}\nExit: {exitp}\nProfit: {profit} ({direction})"

    color = 0x00A65A if success else 0xD32F2F
    embed_data = {
        "title": "Trade Execution",
        "description": f"Trade completed for {pair}",
        "color": color,
        "fields": _embed_fields_from_pairs(
            [
                {"name": "Trading Pair", "value": pair, "inline": True},
                {"name": "Strategy", "value": strategy_name, "inline": True},
                {"name": "Entry Price", "value": entry, "inline": True},
                {"name": "Exit Price", "value": exitp, "inline": True},
                {"name": "Profit/Loss", "value": profit, "inline": True},
                {
                    "name": "Success",
                    "value": ("Yes" if success else "No"),
                    "inline": True,
                },
            ],
        ),
    }
    return send_discord_alert(message, embed_data)


def alert_daily_summary(summary_data: dict[str, Any]) -> bool:
    """
    Send daily trading summary.
    Expected keys: total_trades, win_rate, total_profit, active_strategies, top_performer, worst_performer
    """
    total_trades = int(summary_data.get("total_trades", 0) or 0)
    win_rate = _fmt_pct(summary_data.get("win_rate"))
    total_profit = _fmt_num(summary_data.get("total_profit"))
    active_strategies = int(summary_data.get("active_strategies", 0) or 0)
    top_perf = _sanitize_text(summary_data.get("top_performer", "N/A"))
    worst_perf = _sanitize_text(summary_data.get("worst_performer", "N/A"))

    message = f"DAILY TRADING SUMMARY\nTotal Trades: {total_trades}\nWin Rate: {win_rate}\nTotal Profit: {total_profit}\nActive Strategies: {active_strategies}"

    embed_data = {
        "title": "Daily Trading Summary",
        "description": "End of day trading performance report",
        "color": 0x1976D2,
        "fields": _embed_fields_from_pairs(
            [
                {"name": "Total Trades", "value": str(total_trades), "inline": True},
                {"name": "Win Rate", "value": win_rate, "inline": True},
                {"name": "Total Profit", "value": total_profit, "inline": True},
                {
                    "name": "Active Strategies",
                    "value": str(active_strategies),
                    "inline": True,
                },
                {"name": "Top Performer", "value": top_perf, "inline": True},
                {"name": "Worst Performer", "value": worst_perf, "inline": True},
            ],
        ),
    }
    return send_discord_alert(message, embed_data)


def alert_evolution_cycle(evolution_data: dict[str, Any]) -> bool:
    """
    Notify on evolution cycle completion.
    Expected keys: total_new_strategies, mutations_created, crossovers_created,
                   random_strategies_created, population_stats, deactivated_strategies
    """
    new_total = int(evolution_data.get("total_new_strategies", 0) or 0)
    muts = int(evolution_data.get("mutations_created", 0) or 0)
    xovers = int(evolution_data.get("crossovers_created", 0) or 0)
    rnd = int(evolution_data.get("random_strategies_created", 0) or 0)
    active = int((evolution_data.get("population_stats", {}) or {}).get("active_strategies", 0) or 0)
    deactivated_count = len(evolution_data.get("deactivated_strategies", []) or [])

    message = f"EVOLUTION CYCLE COMPLETED\nNew Strategies: {new_total}\nMutations: {muts}\nCrossovers: {xovers}\nRandom: {rnd}\nActive Population: {active}"

    embed_data = {
        "title": "Strategy Evolution Cycle",
        "description": "New strategies created through evolution",
        "color": 0x6A1B9A,
        "fields": _embed_fields_from_pairs(
            [
                {"name": "New Strategies", "value": str(new_total), "inline": True},
                {"name": "Mutations", "value": str(muts), "inline": True},
                {"name": "Crossovers", "value": str(xovers), "inline": True},
                {"name": "Random Strategies", "value": str(rnd), "inline": True},
                {"name": "Active Population", "value": str(active), "inline": True},
                {
                    "name": "Deactivated",
                    "value": str(deactivated_count),
                    "inline": True,
                },
            ],
        ),
    }
    return send_discord_alert(message, embed_data)


def alert_system_health(health_data: dict[str, Any]) -> bool:
    """
    Send system health alert.
    Expected keys: status, database, api_connections, active_bots, last_update
    """
    status = str(health_data.get("status", "unknown")).lower()
    # Color map
    color_map = {"healthy": 0x00A65A, "warning": 0xFBC02D, "error": 0xD32F2F}
    color = color_map.get(status, 0x616161)

    message = (
        "SYSTEM HEALTH CHECK\n"
        f"Status: {status.upper()}\n"
        f"Database: {_sanitize_text(health_data.get('database', 'unknown'))}\n"
        f"API Connections: {_sanitize_text(health_data.get('api_connections', 'unknown'))}\n"
        f"Active Bots: {int(health_data.get('active_bots', 0) or 0)}"
    )

    embed_data = {
        "title": "System Health Status",
        "description": f"Current system status: {status.upper()}",
        "color": color,
        "fields": _embed_fields_from_pairs(
            [
                {"name": "Status", "value": status.upper(), "inline": True},
                {
                    "name": "Database",
                    "value": _sanitize_text(health_data.get("database", "unknown")),
                    "inline": True,
                },
                {
                    "name": "API Connections",
                    "value": _sanitize_text(health_data.get("api_connections", "unknown")),
                    "inline": True,
                },
                {
                    "name": "Active Bots",
                    "value": str(int(health_data.get("active_bots", 0) or 0)),
                    "inline": True,
                },
                {
                    "name": "Last Update",
                    "value": _sanitize_text(health_data.get("last_update", "unknown")),
                    "inline": True,
                },
            ],
        ),
    }
    return send_discord_alert(message, embed_data)


def test_discord_connection() -> bool:
    """
    Test Discord webhook connectivity.
    """
    test_message = "DISCORD INTEGRATION TEST\nThis is a test message from Mystic AI Trading."
    ok = send_discord_alert(
        test_message,
        {
            "title": "Connectivity Test",
            "description": "Verification ping",
            "color": 0x1976D2,
        },
    )
    if ok:
        logger.info("Discord connection test successful")
    else:
        logger.error("Discord connection test failed")
    return ok
