"""
Budget Tracker
Tracks budget changes, growth milestones, and performance metrics.

Rules honored:
- Single source of truth exchange constants (even if unused here, kept consistent).
- No ad-hoc exchange strings; only "binance_us".
- ASCII-only logging (no emojis).
- Python 3.12, Windows/PowerShell friendly.

ERROR CONTRACT:
- Success: Returns {"success": True, "data": ..., "error": None}
- Error: Returns {"success": False, "data": None, "error": {"code": str, "message": str}}
- All methods follow this consistent structure for predictable API responses
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

# Direct imports for production
import redis

from backend.config.redis_config import get_shared_redis_sync
from backend.services.redis_service import get_redis_service

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.micro_account_manager import get_micro_account_manager

# ---------- Single-source-of-truth exchange constants ----------
# Use TRADING_SYMBOLS from trading_universe (live data)
TOP10_BINANCEUS: set[str] = set(TRADING_SYMBOLS)

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks budget history and milestone achievements."""

    def __init__(self, redis_client: redis.Redis | None = None) -> None:
        """
        Initialize budget tracker.

        If no client is provided, attempts to connect using REDIS_URL or localhost.
        """
        if redis is None:
            logger.error("Redis not available - BudgetTracker disabled")
            self.redis_client = None
            return

        if redis_client is not None:
            self.redis_client = redis_client
        else:
            self.redis_client = get_shared_redis_sync()
            if self.redis_client is None:
                # Fallback to existing service if shared client unavailable
                self.redis_client = get_redis_service()
                if self.redis_client is None:
                    msg = "Shared Redis client unavailable"
                    raise RuntimeError(msg)

        # Namespaced keys
        namespace = os.getenv("BUDGET_TRACKER_NAMESPACE", "mystic")
        self.history_key = f"{namespace}:budget_history"
        self.milestones_key = f"{namespace}:achieved_milestones"
        self.performance_key = f"{namespace}:budget_performance"

        # Configurable milestones
        milestones_env = os.getenv("BUDGET_MILESTONES", "100,200,500,1000,2500,5000,10000,25000,50000,100000")
        try:
            self.milestones = sorted({int(x.strip()) for x in milestones_env.split(",") if x.strip()})
        except ValueError:
            self.milestones = [
                100,
                200,
                500,
                1000,
                2500,
                5000,
                10000,
                25000,
                50000,
                100000,
            ]
            logger.warning("Invalid BUDGET_MILESTONES format, using defaults")

        # Configurable history cap
        self.history_max = int(os.getenv("BUDGET_HISTORY_MAX", "1000"))
        if self.history_max <= 0:
            self.history_max = 1000
            logger.warning("Invalid BUDGET_HISTORY_MAX, using default 1000")

        # Validate connection early (fail fast, clear message in logs)
        if self.redis_client:
            try:
                self.redis_client.ping()
                logger.info("BudgetTracker connected to Redis.")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("BudgetTracker failed to connect to Redis: %s", e)
                # Keep instance usable; methods will handle errors gracefully.

    # -------- Core operations --------

    def track_budget_change(self, old_budget: float, new_budget: float, reason: str = "unknown") -> dict[str, Any]:
        """Track a budget change event."""
        try:
            # Input validation
            old_budget = float(old_budget)
            new_budget = float(new_budget)
            if old_budget < 0 or new_budget < 0:
                return {
                    "success": False,
                    "data": None,
                    "error": {
                        "code": "INVALID_BUDGET",
                        "message": "Budgets must be non-negative",
                    },
                }
            if not math.isfinite(old_budget) or not math.isfinite(new_budget):
                return {
                    "success": False,
                    "data": None,
                    "error": {
                        "code": "INVALID_BUDGET",
                        "message": "Budgets must be finite numbers",
                    },
                }

            # Validate and truncate reason
            reason = str(reason)[:512]  # Cap at 512 chars to prevent bloated Redis entries

            change = new_budget - old_budget
            change_pct = (change / old_budget * 100.0) if old_budget > 0 else 0.0

            change_event: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old_budget": old_budget,
                "new_budget": new_budget,
                "change": change,
                "change_pct": change_pct,
                "reason": reason,
                "milestone_achieved": self.check_milestone_achievement(old_budget, new_budget),
            }

            self.add_to_history(change_event)
            self.update_performance_metrics(new_budget)

            logger.info(
                "Budget change tracked: $%.2f -> $%.2f (%s)",
                old_budget,
                new_budget,
                reason,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error tracking budget change: %s", e)
            return {
                "success": False,
                "data": None,
                "error": {"code": "TRACKING_ERROR", "message": str(e)},
            }
        else:
            return {"success": True, "data": change_event, "error": None}

    # -------- History / Storage --------

    def add_to_history(self, event: dict[str, Any]) -> None:
        """Add event to budget history."""
        if not self.redis_client:
            return

        try:
            # Use Redis list operations for atomicity
            event_json = json.dumps(event)
            self.redis_client.lpush(self.history_key, event_json)

            # Trim to keep only last N events (configurable)
            self.redis_client.ltrim(self.history_key, 0, self.history_max - 1)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error adding to history: %s", e)

    def get_budget_history(self, limit: int = 100) -> dict[str, Any]:
        """Get budget change history."""
        if not self.redis_client:
            return {
                "success": False,
                "data": None,
                "error": {
                    "code": "REDIS_UNAVAILABLE",
                    "message": "Redis not available",
                },
            }

        try:
            # Get from Redis list (most recent first)
            history_strs = self.redis_client.lrange(self.history_key, 0, limit - 1)
            history: list[dict[str, Any]] = []
            for h_str in history_strs:
                try:
                    history.append(json.loads(h_str))
                except json.JSONDecodeError:
                    continue
            # Reverse to get chronological order
            return {"success": True, "data": list(reversed(history)), "error": None}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting budget history: %s", e)
            return {
                "success": False,
                "data": None,
                "error": {"code": "HISTORY_ERROR", "message": str(e)},
            }

    # -------- Milestones --------

    def check_milestone_achievement(self, old_budget: float, new_budget: float) -> int | None:
        """Check if a milestone was achieved."""
        for milestone in self.milestones:
            if old_budget < milestone <= new_budget:
                self.record_milestone_achievement(milestone, new_budget)
                return milestone
        return None

    def record_milestone_achievement(self, milestone: int, budget_at_achievement: float) -> None:
        """Record a milestone achievement with deduplication."""
        if not self.redis_client:
            return

        try:
            milestones_str = self.redis_client.get(self.milestones_key)
            achieved: list[dict[str, Any]] = json.loads(milestones_str) if milestones_str else []

            # Check if milestone already recorded
            for rec in achieved:
                try:
                    if int(rec.get("milestone", -1)) == int(milestone):
                        # Already recorded
                        return
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue

            record = {
                "milestone": int(milestone),
                "achieved_at": datetime.now(timezone.utc).isoformat(),
                "budget": float(budget_at_achievement),
            }
            achieved.append(record)
            try:
                self.redis_client.set(self.milestones_key, json.dumps(achieved))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Failed to persist achieved milestones: %s", e)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error recording milestone achievement: %s", e)

    # Performance and analysis helpers

    def update_performance_metrics(self, latest_budget: float) -> None:
        """Update lightweight performance metrics in Redis."""
        if not self.redis_client:
            return
        try:
            # Compute average daily growth based on history
            avg_daily = self.calculate_average_daily_growth(latest_budget)
            perf = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "last_budget": float(latest_budget),
                "avg_daily_growth": float(avg_daily),
            }
            try:
                self.redis_client.set(self.performance_key, json.dumps(perf))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Failed to persist performance metrics: %s", e)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error updating performance metrics: %s", e)

    def get_best_performance(self) -> dict[str, Any] | None:
        """Return the best (largest positive) single change event."""
        try:
            history_result = self.get_budget_history(limit=self.history_max)
            if not history_result["success"] or not history_result["data"]:
                return None

            history = history_result["data"]
            if not history:
                return None

            best = max(history, key=lambda x: float(x.get("change", 0.0)))
            return {
                "date": str(best.get("timestamp", ""))[:10],
                "change": float(best.get("change", 0.0)),
                "change_pct": float(best.get("change_pct", 0.0)),
                "reason": str(best.get("reason", "unknown")),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None

    def get_worst_performance(self) -> dict[str, Any] | None:
        """Return the worst (most negative) single change event."""
        try:
            history_result = self.get_budget_history(limit=self.history_max)
            if not history_result["success"] or not history_result["data"]:
                return None

            history = history_result["data"]
            if not history:
                return None

            worst = min(history, key=lambda x: float(x.get("change", 0.0)))
            return {
                "date": str(worst.get("timestamp", ""))[:10],
                "change": float(worst.get("change", 0.0)),
                "change_pct": float(worst.get("change_pct", 0.0)),
                "reason": str(worst.get("reason", "unknown")),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None

    def calculate_average_daily_growth(self, _current_budget: float | None = None, days: int = 30) -> float:
        """
        Calculate average daily growth over the past `days` days.
        Simple approach: use earliest and latest budget in history within window and divide by days span.
        """
        try:
            history_result = self.get_budget_history(limit=self.history_max)
            if not history_result["success"] or not history_result["data"]:
                return 0.0

            history = history_result["data"]
            if not history:
                return 0.0

            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            # Filter events after cutoff
            filtered = []
            for ev in history:
                ts = str(ev.get("timestamp", ""))
                parsed = self._parse_timestamp(ts)
                if parsed is None:
                    continue
                if parsed >= cutoff:
                    filtered.append((parsed, float(ev.get("new_budget", 0.0))))

            if not filtered:
                return 0.0

            # Sort by time
            filtered.sort(key=lambda x: x[0])
            first_time, first_budget = filtered[0]
            last_time, last_budget = filtered[-1]

            # If there is no range in time, cannot compute meaningful daily growth
            days_span = max(1, (last_time.date() - first_time.date()).days)
            growth = last_budget - first_budget
            avg_daily = growth / days_span
            if not math.isfinite(avg_daily):
                return 0.0
            return float(avg_daily)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error calculating average daily growth: %s", e)
            return 0.0

    # -------- Reporting helpers --------

    def get_growth_chart_data(self, days: int = 30) -> dict[str, Any]:
        """Get data for growth chart for the last N days."""
        try:
            history_result = self.get_budget_history()
            if not history_result["success"]:
                return history_result

            history = history_result["data"]
            if not history:
                return {"success": True, "data": [], "error": None}

            cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
            out: list[dict[str, Any]] = []

            for event in history:
                ts = str(event.get("timestamp", ""))
                if not ts:
                    continue

                parsed_time = self._parse_timestamp(ts)
                if parsed_time is None:
                    continue  # Skip invalid timestamps

                if parsed_time >= cutoff:
                    out.append(
                        {
                            "date": ts[:10],
                            "budget": float(event.get("new_budget", 0.0)),
                            "change": float(event.get("change", 0.0)),
                            "change_pct": float(event.get("change_pct", 0.0)),
                        },
                    )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting growth chart data: %s", e)
            return {
                "success": False,
                "data": None,
                "error": {"code": "CHART_ERROR", "message": str(e)},
            }
        else:
            return {"success": True, "data": out, "error": None}

    def get_milestone_progress(self) -> dict[str, Any]:
        """Get progress toward next milestone."""
        try:
            # Safe access to current budget via lazy-initialized manager
            current_budget = 0.0
            try:
                mgr = get_micro_account_manager()
                current_budget = float(getattr(mgr, "current_budget", 0))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return {
                    "success": False,
                    "data": None,
                    "error": {
                        "code": "BUDGET_UNAVAILABLE",
                        "message": "Current budget not available",
                    },
                }

            next_milestone = None
            for m in self.milestones:
                if current_budget < m:
                    next_milestone = m
                    break

            if next_milestone is None:
                return {
                    "success": True,
                    "data": {"message": "All major milestones achieved!"},
                    "error": None,
                }

            progress_pct = (current_budget / next_milestone) * 100.0
            remaining = next_milestone - current_budget

            return {
                "success": True,
                "data": {
                    "current_budget": current_budget,
                    "next_milestone": next_milestone,
                    "progress_pct": round(progress_pct, 1),
                    "remaining": round(remaining, 2),
                    "estimated_time": self.estimate_time_to_milestone(remaining),
                },
                "error": None,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error getting milestone progress: %s", e)
            return {
                "success": False,
                "data": None,
                "error": {"code": "MILESTONE_PROGRESS_ERROR", "message": str(e)},
            }

    def estimate_time_to_milestone(self, remaining_amount: float) -> dict[str, Any]:
        """Estimate time to reach next milestone with unit information."""
        try:
            # Safe access to current budget via lazy-initialized manager
            current_budget = 0.0
            try:
                mgr = get_micro_account_manager()
                current_budget = float(getattr(mgr, "current_budget", 0))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return {"value": "Unable to estimate", "unit": "unknown"}

            avg_daily_growth = self.calculate_average_daily_growth(current_budget)
            if avg_daily_growth <= 0 or not math.isfinite(avg_daily_growth):
                return {"value": "Unable to estimate", "unit": "unknown"}

            days_needed = remaining_amount / avg_daily_growth

            if not math.isfinite(days_needed):
                return {"value": "Unable to estimate", "unit": "unknown"}

            if days_needed < 1:
                return {"value": "Less than 1", "unit": "day"}
            if days_needed < 30:
                return {"value": int(days_needed), "unit": "days"}
            if days_needed < 365:
                return {"value": int(days_needed / 30), "unit": "months"}
            return {"value": int(days_needed / 365), "unit": "years"}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return {"value": "Unable to estimate", "unit": "unknown"}

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------
    def _parse_timestamp(self, timestamp_str: str) -> datetime | None:
        """Parse timestamp string to datetime, handling various formats. Returns None for invalid timestamps."""
        try:
            # Normalize Z suffix to +00:00
            normalized = timestamp_str.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)

            # Validate the parsed timestamp is reasonable (not too far in past/future)
            now = datetime.now(timezone.utc)
            if parsed < now - timedelta(days=365 * 10) or parsed > now + timedelta(days=365):
                logger.warning(
                    "Timestamp %s appears invalid (too far in past/future), skipping",
                    timestamp_str,
                )
                return None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.warning("Failed to parse timestamp %s, skipping event", timestamp_str)
            return None
        else:
            return parsed

    def get_status(self) -> dict[str, Any]:
        """Get lightweight status for dashboard."""
        try:
            # Single read of history to avoid double reads
            history_result = self.get_budget_history(limit=1)
            last_update = None
            history_length = 0

            if history_result["success"] and history_result["data"]:
                history = history_result["data"]
                last_update = history[0]["timestamp"] if history else None

                # Get full history length
                full_history_result = self.get_budget_history()
                if full_history_result["success"]:
                    history_length = len(full_history_result["data"])

            milestone_progress_result = self.get_milestone_progress()

            return {
                "success": True,
                "data": {
                    "last_update": last_update,
                    "history_length": history_length,
                    "milestone_progress": milestone_progress_result["data"] if milestone_progress_result["success"] else None,
                    "redis_available": self.redis_client is not None,
                    "last_error": None,  # Could track this if needed
                },
                "error": None,
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            return {
                "success": False,
                "data": None,
                "error": {"code": "STATUS_ERROR", "message": str(e)},
            }


# Global instance (lazy creation to avoid import-time Redis connection)
budget_tracker: BudgetTracker | None = None


# Budget tracker state - using dict to avoid global keyword
_budget_tracker_state: dict[str, BudgetTracker | None] = {"instance": None}


def get_budget_tracker() -> BudgetTracker | None:
    """Get the global budget tracker instance, creating it if needed."""
    if _budget_tracker_state["instance"] is None:
        _budget_tracker_state["instance"] = BudgetTracker()
    return _budget_tracker_state["instance"]


# ---------------- Quick test checklist (manual) ----------------
# - Import succeeds (py -3 budget_tracker.py should do nothing).
# - Redis available and reachable (uses REDIS_URL if set).
# - Logging lines are ASCII only.
# - No unreachable code after returns.
# - No exchange string leaks beyond binance_us constant.
