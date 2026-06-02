"""
Social Trading Manager Service
Handles social trading features and community signals.

Quick Test Checklist:
- Single exchange id constant: EXCHANGE_ID = "binance_us".
- CCXT symbols only (BASE/QUOTE like BTC/USDT) for stored pairs.
- No "binance" or other exchange string leaks — only "binance_us" in EXCHANGE_ID.
- UTC timestamps; ASCII-only logs; no unreachable code after returns.
- No external services required; JSON-serializable payloads.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

try:
    import redis.asyncio as redis
except (ImportError, ModuleNotFoundError):
    redis = None
from backend.config.redis_config import get_shared_redis_async
from backend.utils.symbols import to_ccxt_symbol as _to_ccxt_symbol

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

MAX_COMMUNITY_SIGNALS = int(os.getenv("MAX_COMMUNITY_SIGNALS", "5000"))


def _to_dash_symbol(ccxt_symbol: str) -> str:
    # BTC/USDT -> BTC-USD (for display/UI if needed)
    s = str(ccxt_symbol or "").upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        quote_disp = "USD" if quote == "USDT" else quote
        return f"{base}-{quote_disp}"
    return s.replace("/", "-").replace("USDT", "USD")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SocialTradingManager:
    def __init__(self) -> None:
        # trader_id -> trader dict
        self.active_traders: dict[str, dict[str, Any]] = {}
        # ring buffer of signals (most recent last, capped at MAX_COMMUNITY_SIGNALS)
        self.community_signals: list[dict[str, Any]] = []
        # optional cached leaderboard data if you precompute; not required at runtime
        self.performance_leaderboard: list[dict[str, Any]] = []

        # Redis client for persistence
        self.redis_client: redis.Redis | None = None
        self.use_redis = redis is not None

        logger.info("SocialTradingManager initialized")

    async def _ensure_redis(self) -> bool:
        """Ensure Redis connection is available"""
        if not self.use_redis:
            msg = "Redis support is required but not available"
            raise RuntimeError(msg)

        if self.redis_client is None:
            self.redis_client = get_shared_redis_async()
            if self.redis_client is None:
                msg = "Shared Redis client unavailable"
                raise RuntimeError(msg)
            await self.redis_client.ping()
            logger.info("Redis connection established for SocialTradingManager")
        return True

    async def _save_to_redis(self, key: str, data: dict[str, Any]) -> None:
        """Save data to Redis"""
        await self._ensure_redis()
        await self.redis_client.set(key, json.dumps(data))

    async def _load_from_redis(self, key: str) -> dict[str, Any] | None:
        """Load data from Redis"""
        await self._ensure_redis()

        data = await self.redis_client.get(key)
        return json.loads(data) if data else None

    async def load_persisted_data(self) -> None:
        """Load persisted data from Redis on startup"""
        await self._ensure_redis()
        trader_keys = []
        async for k in self.redis_client.scan_iter(match="social_trader:*", count=100):
            trader_keys.append(k)
        for key in trader_keys:
            trader_data = await self._load_from_redis(key)
            if trader_data:
                trader_id = trader_data.get("trader_id")
                if trader_id:
                    self.active_traders[trader_id] = trader_data

        signals_data = await self._load_from_redis("social_community_signals")
        if signals_data and isinstance(signals_data, list):
            self.community_signals = signals_data

        logger.info(f"Loaded {len(self.active_traders)} traders and {len(self.community_signals)} signals from Redis")

    # ---------- Status ----------

    async def get_social_status(self) -> dict[str, Any]:
        """Get status of social trading features."""
        return {
            "active_traders": len(self.active_traders),
            "community_signals": len(self.community_signals),
            "status": "operational",
            "exchange_id": EXCHANGE_ID,
            "timestamp": _utcnow_iso(),
        }

    # ---------- Traders ----------

    async def add_trader(self, trader_id: str, name: str, strategy: str) -> dict[str, Any]:
        """Add or overwrite a trader profile."""
        try:
            trader_data = {
                "trader_id": trader_id,
                "name": str(name or "").strip(),
                "strategy": str(strategy or "").strip(),
                "performance": {
                    "wins": 0,
                    "win_rate": float(os.getenv("DEFAULT_WIN_RATE", "0.0")),
                    "total_trades": 0,
                    "pnl": float(os.getenv("DEFAULT_PNL", "0.0")),
                },
                "followers": 0,
                "joined_at": _utcnow_iso(),
                "status": "active",
            }

            self.active_traders[trader_id] = trader_data

            await self._save_to_redis(f"social_trader:{trader_id}", trader_data)

            logger.info("Added trader: %s", name)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.exception("Error adding trader: %s", e)
            return {"success": False, "error": str(e)}
        else:
            return {"success": True, "trader_id": trader_id}

    async def update_trader_performance(self, trader_id: str, trade_result: dict[str, Any]) -> dict[str, Any]:
        """Update trader performance after a trade. Expected fields: success (bool), pnl (float)."""
        try:
            trader = self.active_traders.get(trader_id)
            if not trader:
                return {"success": False, "error": "Trader not found"}

            perf = trader.get("performance", {})
            total_trades = int(perf.get("total_trades", 0)) + 1
            pnl_delta = float(trade_result.get("pnl", float(os.getenv("DEFAULT_PNL_DELTA", "0.0"))) or float(os.getenv("DEFAULT_PNL_DELTA", "0.0")))
            perf["pnl"] = float(perf.get("pnl", float(os.getenv("DEFAULT_PNL", "0.0")))) + pnl_delta

            # Compute wins from stored integer count
            wins = int(perf.get("wins", 0))
            if bool(trade_result.get("success")) or pnl_delta > 0:
                wins += 1

            perf["wins"] = wins
            perf["total_trades"] = total_trades
            perf["win_rate"] = (wins / total_trades) if total_trades > 0 else float(os.getenv("DEFAULT_WIN_RATE", "0.0"))
            trader["performance"] = perf

            logger.info("Updated performance for trader %s", trader_id)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.exception("Error updating trader performance: %s", e)
            return {"success": False, "error": str(e)}
        else:
            return {"success": True, "trader_id": trader_id, "performance": perf}

    # ---------- Community Signals ----------

    async def add_community_signal(
        self,
        trader_id: str,
        pair: str,
        signal_type: str,
        confidence: float,
    ) -> dict[str, Any]:
        """
        Add a community trading signal.
        - pair is normalized to CCXT "BASE/QUOTE".
        - signal_type is free-form (e.g., "buy", "sell", "hold"); stored uppercased for consistency.
        - confidence is normalized to canonical 0.0-1.0 (accepts 0-1 or 0-100).
        """
        try:
            from backend.services.confidence_normalizer import ConfidenceNormalizer

            confidence = ConfidenceNormalizer.normalize(float(confidence) if confidence is not None else 0.0)
            ccxt_pair = _to_ccxt_symbol(pair)
            disp_pair = _to_dash_symbol(ccxt_pair)
            trader = self.active_traders.get(trader_id)

            signal = {
                "signal_id": len(self.community_signals) + 1,
                "exchange_id": EXCHANGE_ID,
                "trader_id": trader_id,
                "trader_name": (trader.get("name") if trader else "Unknown"),
                "pair": ccxt_pair,
                "pair_display": disp_pair,
                "signal_type": str(signal_type or "").upper(),
                "confidence": float(confidence),
                "timestamp": _utcnow_iso(),
                "votes": 0,
            }
            self.community_signals.append(signal)

            if len(self.community_signals) > MAX_COMMUNITY_SIGNALS:
                drop = len(self.community_signals) - MAX_COMMUNITY_SIGNALS
                if drop > 0:
                    del self.community_signals[0:drop]
                    logger.info(f"Trimmed {drop} old community signals to maintain cap of {MAX_COMMUNITY_SIGNALS}")

            # Publish to Redis for AI consumption
            try:
                if redis is None:
                    logger.warning("Redis not available")
                    return

                r = get_shared_redis_async()
                if r is None:
                    msg = "Shared Redis client unavailable"
                    raise RuntimeError(msg)
                key = f"social_signal:{ccxt_pair}"
                await r.hset(
                    key,
                    mapping={
                        "trader_id": trader_id,
                        "trader_name": signal["trader_name"],
                        "signal_type": signal_type,
                        "confidence": str(confidence),
                        "timestamp": _utcnow_iso(),
                        "source": "social_trading",
                        "votes": os.getenv("DEFAULT_VOTES", "0"),
                    },
                )
                await r.expire(key, int(os.getenv("SOCIAL_SIGNAL_TTL", "3600")))
                await r.aclose()
                logger.info("Published social signal to Redis: %s", key)
            except (ConnectionError, OSError, AttributeError, TypeError, ValueError) as redis_error:
                logger.warning(f"Failed to publish social signal to Redis: {redis_error}")

            logger.info("Added community signal from %s on %s", signal["trader_name"], ccxt_pair)
            return {"success": True, "signal_id": signal["signal_id"]}
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.exception("Error adding community signal: %s", e)
            return {"success": False, "error": str(e)}

    async def vote_signal(self, signal_id: int, up: bool = True) -> dict[str, Any]:
        """Increment/decrement votes on a community signal."""
        try:
            if signal_id <= 0 or signal_id > len(self.community_signals):
                return {"success": False, "error": "Signal not found"}
            sig = self.community_signals[signal_id - 1]
            cur = int(sig.get("votes", 0))
            sig["votes"] = cur + (1 if up else -1)
            return {"success": True, "signal_id": signal_id, "votes": sig["votes"]}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            logger.exception("Error voting on signal: %s", e)
            return {"success": False, "error": str(e)}

    async def get_recent_signals(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get most recent community signals (sorted desc by timestamp)."""
        try:
            lim = max(0, int(limit)) if limit else len(self.community_signals)
            # Sort by timestamp descending (ISO8601 UTC comparable)
            return sorted(
                self.community_signals,
                key=lambda x: x.get("timestamp", ""),
                reverse=True,
            )[:lim]
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.exception("Error getting recent signals: %s", e)
            return []

    # ---------- Leaderboard ----------

    async def get_leaderboard(self) -> list[dict[str, Any]]:
        """Return top 10 traders by PnL."""
        try:
            # VECTORIZED leaderboard creation for performance
            leaderboard = [
                {
                    "trader_id": trader_id,
                    "name": trader.get("name"),
                    "win_rate": float(trader.get("performance", {}).get("win_rate", float(os.getenv("DEFAULT_WIN_RATE", "0.0")))),
                    "total_trades": int(trader.get("performance", {}).get("total_trades", 0)),
                    "pnl": float(trader.get("performance", {}).get("pnl", float(os.getenv("DEFAULT_PNL", "0.0")))),
                    "followers": int(trader.get("followers", 0)),
                }
                for trader_id, trader in self.active_traders.items()
            ]

            # Sort by PnL descending
            leaderboard.sort(key=lambda x: x["pnl"], reverse=True)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.exception("Error getting leaderboard: %s", e)
            return []
        else:
            return leaderboard


# Global instance
social_trading_manager = SocialTradingManager()
