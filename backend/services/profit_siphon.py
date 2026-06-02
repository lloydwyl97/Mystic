#!/usr/bin/env python3
"""
Profit Siphon Service
Permanently locks gains out of risk to prevent long-term ruin
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from backend.config.redis_config import get_shared_redis_sync
from backend.database_schema import get_state, set_state

logger = logging.getLogger(__name__)


class ProfitSiphon:
    """
    Profit Siphon - permanently locks 40% of realized gains over $50
    Prevents long-term ruin by creating a ratcheting profit machine
    """

    # Siphon configuration - hardcoded for safety
    SIPHON_ENABLED = True
    SIPHON_PERCENT = 0.40  # 40% of gains
    SIPHON_TRIGGER_PROFIT = 50.0  # Triggers on $50+ gains
    SIPHON_BUCKET_KEY = "locked_profit_reserve"

    def __init__(self):
        self.redis_client = get_shared_redis_sync()
        self.total_siphoned = 0.0
        self.siphon_count = 0

        # Initialize from SQLite (authoritative source)
        self._load_from_sqlite()
        logger.info("[PROFIT SIPHON] Initialized - 40% of $50+ gains permanently locked")

    def _load_from_sqlite(self) -> None:
        """Load siphon state from SQLite (authoritative source)"""
        try:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                logger.warning("[PROFIT SIPHON] Async context detected; skip sync load to avoid deadlock (use async load if needed)")
                return

            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            siphon_data = loop.run_until_complete(get_state("locked_profit_reserve"))

            if siphon_data:
                self.total_siphoned = siphon_data.get("total_locked", 0.0)
                self.siphon_count = siphon_data.get("siphon_count", 0)
                logger.info(f"[PROFIT SIPHON] Loaded from SQLite: ${self.total_siphoned:.2f} locked, {self.siphon_count} siphons")

                # Populate Redis cache from SQLite
                self._populate_redis_cache(siphon_data)
            else:
                # Initialize if no data exists
                initial_data = {
                    "total_locked": 0.0,
                    "siphon_count": 0,
                    "last_siphon": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "description": "Locked profit reserve - permanently removed from trading risk",
                }
                loop.run_until_complete(set_state("locked_profit_reserve", initial_data))
                self._populate_redis_cache(initial_data)
                logger.info("[PROFIT SIPHON] Initialized new siphon state in SQLite")

        except Exception as e:
            logger.warning(f"[PROFIT SIPHON] Failed to load from SQLite: {e}")

    def _populate_redis_cache(self, siphon_data: dict) -> None:
        """Populate Redis cache from SQLite data"""
        try:
            if self.redis_client:
                self.redis_client.set(self.SIPHON_BUCKET_KEY, json.dumps(siphon_data))
                logger.debug("[PROFIT SIPHON] Populated Redis cache from SQLite")
        except Exception as e:
            logger.warning(f"[PROFIT SIPHON] Failed to populate Redis cache: {e}")

    def process_realized_pnl(self, realized_pnl_increment: float, current_portfolio_value: float) -> dict[str, Any]:
        """
        Process realized PnL increment through the siphon
        Called whenever realized profits are recorded

        Args:
            realized_pnl_increment: The amount of realized profit from recent trades
            current_portfolio_value: Current total portfolio value

        Returns:
            Dict with siphon results and adjusted values
        """
        # Prevent unused parameter warning
        _ = current_portfolio_value

        if not self.SIPHON_ENABLED:
            return {"siphoned": False, "siphon_amount": 0.0, "remaining_profit": realized_pnl_increment, "total_locked": self.total_siphoned}

        try:
            # Check if siphon should trigger
            if realized_pnl_increment >= self.SIPHON_TRIGGER_PROFIT:
                siphon_amount = realized_pnl_increment * self.SIPHON_PERCENT
                remaining_profit = realized_pnl_increment - siphon_amount

                # Execute the siphon
                success = self._execute_siphon(siphon_amount)

                if success:
                    self.total_siphoned += siphon_amount
                    self.siphon_count += 1

                    logger.info(
                        f"[PROFIT SIPHON] TRIGGERED | "
                        f"Gain: ${realized_pnl_increment:.2f} | "
                        f"Siphoned: ${siphon_amount:.2f} (40%) | "
                        f"Remaining: ${remaining_profit:.2f} | "
                        f"Total Locked: ${self.total_siphoned:.2f}"
                    )

                    return {"siphoned": True, "siphon_amount": siphon_amount, "remaining_profit": remaining_profit, "total_locked": self.total_siphoned, "siphon_count": self.siphon_count}
                else:
                    logger.warning(f"[PROFIT SIPHON] FAILED to execute siphon of ${siphon_amount:.2f}")
                    return {"siphoned": False, "siphon_amount": 0.0, "remaining_profit": realized_pnl_increment, "total_locked": self.total_siphoned, "error": "siphon_execution_failed"}
            else:
                # No siphon triggered
                return {
                    "siphoned": False,
                    "siphon_amount": 0.0,
                    "remaining_profit": realized_pnl_increment,
                    "total_locked": self.total_siphoned,
                    "reason": f"gain_below_threshold (${realized_pnl_increment:.2f} < ${self.SIPHON_TRIGGER_PROFIT:.2f})",
                }

        except Exception as e:
            logger.exception(f"[PROFIT SIPHON] Error processing realized PnL: {e}")
            return {"siphoned": False, "siphon_amount": 0.0, "remaining_profit": realized_pnl_increment, "total_locked": self.total_siphoned, "error": str(e)}

    def _execute_siphon(self, siphon_amount: float) -> bool:
        """
        Execute the actual siphoning - permanently remove from tradable capital
        SQLite authoritative, Redis cache
        """
        try:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                logger.warning("[PROFIT SIPHON] Async context detected; skip sync siphon to avoid deadlock")
                return False

            loop = asyncio.get_event_loop()
            current_data = loop.run_until_complete(get_state("locked_profit_reserve"))

            if not current_data:
                current_data = {"total_locked": 0.0, "siphon_count": 0, "last_siphon": None, "created_at": datetime.now(timezone.utc).isoformat()}

            # Update data
            current_data["total_locked"] += siphon_amount
            current_data["siphon_count"] += 1
            current_data["last_siphon"] = datetime.now(timezone.utc).isoformat()

            # CRITICAL: Write SQLite first (authoritative)
            loop.run_until_complete(set_state("locked_profit_reserve", current_data))

            # Update in-memory state
            self.total_siphoned = current_data["total_locked"]
            self.siphon_count = current_data["siphon_count"]

            # Then update Redis cache (if available)
            if self.redis_client:
                try:
                    self.redis_client.set(self.SIPHON_BUCKET_KEY, json.dumps(current_data))
                except Exception as redis_e:
                    logger.warning(f"[PROFIT SIPHON] Redis cache update failed (SQLite succeeded): {redis_e}")

            logger.info(f"[PROFIT SIPHON] Siphoned ${siphon_amount:.2f} | Total locked: ${current_data['total_locked']:.2f}")
            return True

        except Exception as e:
            logger.exception(f"[PROFIT SIPHON] Error executing siphon: {e}")
            return False

    def get_siphon_status(self) -> dict[str, Any]:
        """Get current siphon status and locked profit information"""
        try:
            if self.redis_client:
                bucket_data = self.redis_client.get(self.SIPHON_BUCKET_KEY)
                if bucket_data:
                    data = json.loads(bucket_data)
                    return {
                        "enabled": self.SIPHON_ENABLED,
                        "siphon_percent": self.SIPHON_PERCENT,
                        "trigger_threshold": self.SIPHON_TRIGGER_PROFIT,
                        "total_locked": data.get("total_locked", 0.0),
                        "siphon_count": data.get("siphon_count", 0),
                        "last_siphon": data.get("last_siphon"),
                        "created_at": data.get("created_at"),
                        "description": "40% of $50+ realized gains permanently locked from risk",
                    }
                else:
                    return {
                        "enabled": self.SIPHON_ENABLED,
                        "siphon_percent": self.SIPHON_PERCENT,
                        "trigger_threshold": self.SIPHON_TRIGGER_PROFIT,
                        "total_locked": self.total_siphoned,
                        "siphon_count": self.siphon_count,
                        "error": "redis_unavailable",
                    }
            else:
                return {
                    "enabled": self.SIPHON_ENABLED,
                    "siphon_percent": self.SIPHON_PERCENT,
                    "trigger_threshold": self.SIPHON_TRIGGER_PROFIT,
                    "total_locked": self.total_siphoned,
                    "siphon_count": self.siphon_count,
                    "error": "redis_unavailable",
                }

        except Exception as e:
            logger.exception(f"[PROFIT SIPHON] Error getting status: {e}")
            return {
                "enabled": self.SIPHON_ENABLED,
                "siphon_percent": self.SIPHON_PERCENT,
                "trigger_threshold": self.SIPHON_TRIGGER_PROFIT,
                "total_locked": self.total_siphoned,
                "siphon_count": self.siphon_count,
                "error": str(e),
            }

    def emergency_unlock(self, unlock_amount: float, reason: str) -> bool:
        """
        EMERGENCY ONLY: Unlock siphoned profits (should never be used)
        This breaks the ratcheting mechanism - use only in absolute emergencies
        """
        logger.critical(f"[PROFIT SIPHON] EMERGENCY UNLOCK REQUESTED: ${unlock_amount:.2f} | Reason: {reason}")
        logger.critical("[PROFIT SIPHON] This breaks the permanent ratcheting mechanism!")

        # This method intentionally does nothing - siphoned profits are meant to be permanent
        # If you need to implement emergency unlocking, add explicit override logic here
        return False


# Global singleton instance
_siphon_instance = None


def get_profit_siphon() -> ProfitSiphon:
    """Get singleton instance of profit siphon"""
    global _siphon_instance
    if _siphon_instance is None:
        _siphon_instance = ProfitSiphon()
    return _siphon_instance
