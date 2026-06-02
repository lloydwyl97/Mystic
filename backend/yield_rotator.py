#!/usr/bin/env python3
"""
Yield Rotation Engine - Live Configuration Only

Manages idle capital by parking it in yield-generating protocols
and rotating back to trading.
All configuration values come from live config - no hardcoded values.

Windows-friendly (no emoji), Python 3.12 compatible.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
from pathlib import Path
from threading import RLock
from typing import Any

import httpx

# Module logger (no basicConfig; host app configures handlers)
log = logging.getLogger("yield_rotator")

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Money precision - from live config


def _get_decimal_precision() -> int:
    """Get decimal precision from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "decimal_precision"):
                prec = value.decimal_precision
                if isinstance(prec, int) and 1 <= prec <= 50:
                    return prec
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("YIELD_DECIMAL_PRECISION", "28"))
        return max(1, min(50, value))
    except (ValueError, TypeError):
        return 28


getcontext().prec = _get_decimal_precision()
getcontext().rounding = ROUND_HALF_UP


@dataclass
class ParkingRecord:
    id: str
    amount: str  # Decimal serialized as string (native units or currency per config)
    protocol_id: str
    protocol_name: str
    apy: str  # Decimal as string; APY as fraction per provider (e.g., 0.12 for 12%)
    lock_period_days: int
    start_time_iso: str
    end_time_iso: str
    status: str  # "active" | "withdrawn"
    earned_yield: str = "0"
    withdrawal_time_iso: str | None = None


class YieldRotator:
    """
    Advanced yield rotation engine for capital efficiency.
    All configuration values come from live config.
    """

    def __init__(self, min_park_amount: float | None = None, max_park_percentage: float | None = None) -> None:
        """
        Args:
            min_park_amount: Minimum dollar amount to park in yield (overrides live config if provided).
            max_park_percentage: Maximum percentage of total capital to park (overrides live config if provided).
        """
        # Load from live config if not provided
        if min_park_amount is None:
            min_park_amount = _get_min_park_amount()
        if max_park_percentage is None:
            max_park_percentage = _get_max_park_percentage()

        self.min_park_amount = float(min_park_amount)
        self.max_park_percentage = float(max_park_percentage)

        # In-memory state
        self.parked_capital: dict[str, ParkingRecord] = {}
        self.yield_history: list[dict[str, Any]] = []
        self._lock = RLock()

        # Protocol registry (live config). No hardcoded APYs.
        self.yield_protocols: dict[str, dict[str, Any]] = {}
        self._protocols_ttl_sec = _get_protocols_ttl_sec()
        self._protocols_last_fetch: float = 0.0
        self._protocols_endpoint = _get_protocols_endpoint()
        self._load_protocols_from_env()

        # Persistence (SQLite)
        self._db_path = _get_yield_db_path()
        self._init_db()

    def _reload_config(self) -> None:
        """Reload configuration values from live config."""
        self.min_park_amount = _get_min_park_amount()
        self.max_park_percentage = _get_max_park_percentage()
        self._protocols_ttl_sec = _get_protocols_ttl_sec()
        self._protocols_endpoint = _get_protocols_endpoint()

    # --------------------------------------------------------------------------
    # Core helpers
    # --------------------------------------------------------------------------
    def calculate_optimal_park_amount(self, total_capital: float, idle_percentage: float | None = None) -> float:
        """Compute how much to park given idle capital and the cap."""
        total_capital = float(total_capital)
        if idle_percentage is None:
            idle_percentage = _get_default_idle_percentage()
        idle_percentage = float(idle_percentage)

        idle_amount = total_capital * idle_percentage
        max_park = total_capital * self.max_park_percentage
        park_amount = min(idle_amount, max_park)

        if park_amount < self.min_park_amount:
            return 0.0
        return round(park_amount, 2)

    def _matches_risk_tolerance(self, protocol_risk: str, user_risk: str) -> bool:
        """Allow protocols with risk <= user tolerance."""
        levels = {"low": 1, "medium": 2, "high": 3}
        return levels.get(protocol_risk, 2) <= levels.get(user_risk, 2)

    def select_yield_protocol(
        self,
        amount: float,
        risk_tolerance: str | None = None,
        lock_period: int | None = None,
    ) -> dict[str, Any]:
        """
        Choose the best protocol by APY that meets lock period & risk tolerance.
        """
        if risk_tolerance is None:
            risk_tolerance = _get_default_risk_tolerance()
        if lock_period is None:
            lock_period = _get_default_lock_period()
        candidates: list[tuple[str, dict[str, Any]]] = []
        for pid, p in self.yield_protocols.items():
            try:
                min_lock = int(p.get("min_lock", 0))
                max_lock = int(p.get("max_lock", 0))
                risk = str(p.get("risk_level", "medium"))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue
            if min_lock <= lock_period <= max_lock and self._matches_risk_tolerance(risk, risk_tolerance):
                candidates.append((pid, p))

        if not candidates:
            return {
                "error": "no_protocol_match",
                "message": "No protocol matches lock/risk constraints",
            }

        candidates.sort(key=lambda x: Decimal(str(x[1].get("apy", "0"))), reverse=True)
        pid, p = candidates[0]
        return {
            "protocol_id": pid,
            "protocol": p,
            "amount": amount,
            "lock_period": lock_period,
        }

    # ----------------------------------------------------------------------------
    # Protocols / Oracle / Portfolio wiring (live fetch with TTL)
    # ----------------------------------------------------------------------------
    def _load_protocols_from_env(self) -> None:
        reg = os.getenv("YIELD_PROTOCOLS_JSON", "").strip()
        if reg:
            try:
                self.yield_protocols = json.loads(reg)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                log.exception("Failed to parse YIELD_PROTOCOLS_JSON: %s", e)

    def _refresh_protocols_if_needed(self) -> None:
        if not getattr(self, "_protocols_endpoint", ""):
            return

        now = time.time()
        if now - getattr(self, "_protocols_last_fetch", 0.0) < getattr(self, "_protocols_ttl_sec", 300):
            return
        try:
            timeout = _get_protocols_fetch_timeout()
            with httpx.Client() as client:
                r = client.get(self._protocols_endpoint, timeout=timeout)
                if r.status_code in range(200, 300):
                    data = r.json()
                    if isinstance(data, dict):
                        self.yield_protocols = data
                        self._protocols_last_fetch = now
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            log.warning("Protocol fetch failed: %s", e)

    # ----------------------------------------------------------------------------
    # Persistence (SQLite)
    # ----------------------------------------------------------------------------
    def _init_db(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS yield_parking (
                        id TEXT PRIMARY KEY,
                        amount TEXT,
                        protocol_id TEXT,
                        protocol_name TEXT,
                        apy TEXT,
                        lock_period_days INTEGER,
                        start_time_iso TEXT,
                        end_time_iso TEXT,
                        status TEXT,
                        earned_yield TEXT,
                        withdrawal_time_iso TEXT
                    )
                    """,
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS yield_history (
                        ts TEXT,
                        action TEXT,
                        parking_id TEXT,
                        payload TEXT
                    )
                    """,
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            log.exception("Yield DB init failed: %s", e)
        finally:
            if conn is not None:
                conn.close()

    def _db_insert_record(self, rec: ParkingRecord) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO yield_parking (
                        id, amount, protocol_id, protocol_name, apy, lock_period_days,
                        start_time_iso, end_time_iso, status, earned_yield, withdrawal_time_iso
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.id,
                        rec.amount,
                        rec.protocol_id,
                        rec.protocol_name,
                        rec.apy,
                        rec.lock_period_days,
                        rec.start_time_iso,
                        rec.end_time_iso,
                        rec.status,
                        rec.earned_yield,
                        rec.withdrawal_time_iso,
                    ),
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            log.exception("Yield DB insert record failed: %s", e)
        finally:
            if conn is not None:
                conn.close()

    def _db_update_record(self, rec: ParkingRecord) -> None:
        self._db_insert_record(rec)

    def _db_insert_history(self, entry: dict[str, Any]) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            with conn:
                ts = entry.get("timestamp", datetime.now(timezone.utc).isoformat())
                action = entry.get("action", "")
                parking_id = entry.get("parking_id")
                payload = json.dumps(entry)
                conn.execute(
                    "INSERT INTO yield_history (ts, action, parking_id, payload) VALUES (?, ?, ?, ?)",
                    (ts, action, parking_id, payload),
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            log.debug("Yield DB insert history failed: %s", e)
        finally:
            if conn is not None:
                conn.close()

    def park_capital(
        self,
        amount: float,
        protocol_id: str | None = None,
        lock_period: int | None = None,
        risk_tolerance: str | None = None,
    ) -> dict[str, Any]:
        """
        Park capital into a protocol. If protocol_id is None, select the best one.
        """
        if lock_period is None:
            lock_period = _get_default_lock_period()
        if risk_tolerance is None:
            risk_tolerance = _get_default_risk_tolerance()
        quantize_level = _get_decimal_quantize_level()
        try:
            amt_dec = Decimal(str(amount)).quantize(Decimal(quantize_level))
        except (InvalidOperation, Exception):
            return {"success": False, "error": "invalid_amount"}

        self._refresh_protocols_if_needed()

        selected_protocol = None
        selected_pid = None
        if protocol_id:
            p = self.yield_protocols.get(protocol_id)
            if not p:
                return {"success": False, "error": "protocol_not_found"}
            selected_protocol = p
            selected_pid = protocol_id
        else:
            sel = self.select_yield_protocol(float(amt_dec), risk_tolerance, lock_period)
            if sel.get("error"):
                return {"success": False, "error": sel.get("error"), "message": sel.get("message")}
            selected_pid = sel["protocol_id"]
            selected_protocol = sel["protocol"]

        now = datetime.now(timezone.utc)
        end = now + timedelta(days=int(lock_period))

        apy_str = str(selected_protocol.get("apy", "0"))
        protocol_name = str(selected_protocol.get("name", selected_pid))

        rec = ParkingRecord(
            id=uuid.uuid4().hex,
            amount=str(amt_dec),
            protocol_id=str(selected_pid),
            protocol_name=protocol_name,
            apy=apy_str,
            lock_period_days=int(lock_period),
            start_time_iso=now.isoformat(),
            end_time_iso=end.isoformat(),
            status="active",
            earned_yield="0",
            withdrawal_time_iso=None,
        )

        with self._lock:
            self.parked_capital[rec.id] = rec
            self._db_insert_record(rec)

        self._log_parking_action(rec)

        log.info("Parked %s into %s (apy=%s, lock_days=%s)", rec.amount, rec.protocol_name, rec.apy, rec.lock_period_days)

        return {"success": True, "parking_id": rec.id, "record": asdict(rec)}

    # The following method was present in the original file (truncated in prompt).
    # Reconstructed here with corrections.
    def withdraw_capital(self, parking_id: str, force: bool = False) -> dict[str, Any]:
        """Withdraw principal (+ yield if mature). If forced early, apply penalty."""
        with self._lock:
            rec = self.parked_capital.get(parking_id)
        if not rec:
            return {"success": False, "error": f"Parking record {parking_id} not found"}

        end_time = datetime.fromisoformat(rec.end_time_iso)
        now = datetime.now(timezone.utc)
        mature = now >= end_time

        if not mature and not force:
            return {
                "success": False,
                "error": f"Lock period not ended. Ends at {end_time.isoformat()}",
            }

        # Precise days calculation
        start_dt = datetime.fromisoformat(rec.start_time_iso)
        days_parked = max((now - start_dt).total_seconds() / 86400.0, 0.0)

        # Earned yield to date (Decimal)
        try:
            amt = Decimal(rec.amount)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            amt = Decimal("0")
        try:
            apy = Decimal(rec.apy)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            apy = Decimal("0")
        quantize_level = _get_decimal_quantize_level()
        accrued_yield_dec = (amt * apy * Decimal(str(days_parked)) / Decimal(365)).quantize(Decimal(quantize_level))

        penalty_dec = Decimal(0)
        if not mature and force:
            # Protocol-provided early exit penalty percent if present
            proto = self.yield_protocols.get(rec.protocol_id, {})
            default_penalty_pct = _get_default_early_exit_penalty_pct()
            try:
                pen_pct = Decimal(str(proto.get("early_exit_penalty_pct", str(default_penalty_pct)))) / Decimal(100)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pen_pct = Decimal(str(default_penalty_pct)) / Decimal(100)
            penalty_dec = (pen_pct * amt).quantize(Decimal(quantize_level))

        total_return_dec = (amt + accrued_yield_dec - penalty_dec).quantize(Decimal(quantize_level))

        # Update record
        rec.status = "withdrawn"
        rec.earned_yield = str(accrued_yield_dec)
        rec.withdrawal_time_iso = now.isoformat()
        with self._lock:
            self.parked_capital[parking_id] = rec
            self._db_update_record(rec)

        # Log with numeric penalty
        numeric_penalty = float(penalty_dec)
        self._log_withdrawal_action(rec, penalty=numeric_penalty)

        log.info(
            "Withdrew %s (yield=%s, penalty=%s, total=%s) from %s",
            rec.amount,
            rec.earned_yield,
            str(penalty_dec),
            str(total_return_dec),
            rec.protocol_name,
        )

        # Safely convert values for return
        try:
            earned_yield_float = float(accrued_yield_dec)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            earned_yield_float = 0.0
        try:
            amt_float = float(amt)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            amt_float = 0.0

        return {
            "success": True,
            "amount": amt_float,
            "earned_yield": earned_yield_float,
            "penalty": float(penalty_dec),
            "total_returned": float(total_return_dec),
            "record": asdict(rec),
        }

    # --------------------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------------------
    def get_parked_capital_summary(self) -> dict[str, Any]:
        """Summary across all active positions with up-to-date accrued yield."""
        if not self.parked_capital:
            return {
                "total_parked": 0.0,
                "total_earned": 0.0,
                "active_positions": 0,
                "protocols": {},
            }

        total_parked = 0.0
        total_earned = 0.0
        active_positions = 0
        protocols: dict[str, dict[str, Any]] = {}

        now = datetime.now(timezone.utc)
        for rec in self.parked_capital.values():
            if rec.status == "active":
                total_parked += float(Decimal(rec.amount))
                active_positions += 1

                start_dt = datetime.fromisoformat(rec.start_time_iso)
                days_parked = max((now - start_dt).total_seconds() / 86400.0, 0.0)
                quantize_level = _get_decimal_quantize_level()
                earned = float((Decimal(rec.amount) * Decimal(rec.apy) * Decimal(str(days_parked)) / Decimal(365)).quantize(Decimal(quantize_level)))
                total_earned += earned

                agg = protocols.setdefault(rec.protocol_id, {"amount": 0.0, "positions": 0, "avg_apy": 0.0})
                agg["amount"] += float(Decimal(rec.amount))
                agg["positions"] += 1
                agg["avg_apy"] = float(Decimal(rec.apy))
            else:
                # already withdrawn; count recorded earned yield
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    total_earned += float(Decimal(rec.earned_yield))

        return {
            "total_parked": round(total_parked, 2),
            "total_earned": round(total_earned, 2),
            "active_positions": active_positions,
            "protocols": protocols,
        }

    def get_yield_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the last N actions."""
        if limit is None:
            limit = _get_yield_history_default_limit()
        return self.yield_history[-int(limit) :]

    # --------------------------------------------------------------------------
    # Rotation logic
    # --------------------------------------------------------------------------
    def auto_rotate_capital(
        self,
        total_capital: float,
        trading_signal_strength: float,
        idle_percentage: float | None = None,
    ) -> dict[str, Any]:
        """
        Automatically rotate capital based on trading signal strength.
        Thresholds and lock periods come from live configuration.
        """
        if idle_percentage is None:
            idle_percentage = _get_default_idle_percentage()
        strong_signal_threshold = _get_strong_signal_threshold()
        weak_signal_threshold = _get_weak_signal_threshold()
        short_lock_period = _get_short_lock_period()
        actions: list[dict[str, Any]] = []

        if trading_signal_strength > strong_signal_threshold:
            # Withdraw everything (force early if necessary)
            for pid, rec in list(self.parked_capital.items()):
                if rec.status == "active":
                    result = self.withdraw_capital(pid, force=True)
                    if result.get("success"):
                        actions.append(
                            {
                                "action": "withdraw",
                                "parking_id": pid,
                                "amount": result["total_returned"],
                                "reason": "strong_trading_signal",
                            }
                        )

        elif trading_signal_strength < weak_signal_threshold:
            park_amount = self.calculate_optimal_park_amount(total_capital, idle_percentage)
            if park_amount > 0:
                result = self.park_capital(park_amount, lock_period=short_lock_period)
                if result.get("success"):
                    actions.append(
                        {
                            "action": "park",
                            "parking_id": result["parking_id"],
                            "amount": park_amount,
                            "reason": "weak_trading_signal",
                        }
                    )

        return {
            "actions_taken": len(actions),
            "actions": actions,
            "trading_signal_strength": trading_signal_strength,
        }

    # --------------------------------------------------------------------------
    # Internal logging (kept simple and JSON-like)
    # --------------------------------------------------------------------------
    def _log_parking_action(self, rec: ParkingRecord) -> None:
        self.yield_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "park",
                "parking_id": rec.id,
                "amount": rec.amount,
                "protocol": rec.protocol_name,
                "apy": rec.apy,
                "lock_period_days": rec.lock_period_days,
            },
        )
        self._db_insert_history(self.yield_history[-1])

    def _log_withdrawal_action(self, rec: ParkingRecord, penalty: float = 0.0) -> None:
        # Ensure numeric values are safe
        try:
            earned = float(Decimal(rec.earned_yield))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            earned = 0.0
        try:
            amount = float(Decimal(rec.amount))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            amount = 0.0

        total_returned = round(amount + earned - float(penalty), 2)

        self.yield_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "withdraw",
                "parking_id": rec.id,
                "amount": rec.amount,
                "earned_yield": earned,
                "penalty": round(float(penalty), 2),
                "total_returned": total_returned,
            },
        )
        self._db_insert_history(self.yield_history[-1])


# ------------------------------------------------------------------------------
# Shared instance wrappers (stateful)
# ------------------------------------------------------------------------------
_rotator_singleton: YieldRotator | None = None


def get_yield_rotator() -> YieldRotator:
    """Get the shared yield rotator instance with live configuration."""
    global _rotator_singleton
    if _rotator_singleton is None:
        _rotator_singleton = YieldRotator()
    else:
        # Reload config to ensure live values
        _rotator_singleton._reload_config()
    return _rotator_singleton


def park_in_yield(amount: float, protocol_id: str | None = None) -> dict[str, Any]:
    return get_yield_rotator().park_capital(amount, protocol_id)


def exit_yield(parking_id: str) -> dict[str, Any]:
    return get_yield_rotator().withdraw_capital(parking_id)


def auto_rotate_yield(total_capital: float, trading_signal: float) -> dict[str, Any]:
    return get_yield_rotator().auto_rotate_capital(total_capital, trading_signal)


# ------------------------------------------------------------------------------
# Example usage (runs locally with plain ASCII logging)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Configuration helpers (live config)
# ------------------------------------------------------------------------------
def _get_min_park_amount() -> float:
    """Get minimum park amount from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "min_park_amount"):
                amount = value.min_park_amount
                if isinstance(amount, (int, float)) and amount > 0:
                    return float(amount)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_MIN_PARK_AMOUNT", "100.0"))
        return max(0.01, value)
    except (ValueError, TypeError):
        return 100.0


def _get_max_park_percentage() -> float:
    """Get maximum park percentage from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "max_park_percentage"):
                pct = value.max_park_percentage
                if isinstance(pct, (int, float)) and 0 < pct <= 1:
                    return float(pct)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_MAX_PARK_PERCENTAGE", "0.30"))
        return max(0.01, min(1.0, value))
    except (ValueError, TypeError):
        return 0.30


def _get_default_idle_percentage() -> float:
    """Get default idle percentage from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "default_idle_percentage"):
                pct = value.default_idle_percentage
                if isinstance(pct, (int, float)) and 0 < pct <= 1:
                    return float(pct)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_DEFAULT_IDLE_PERCENTAGE", "0.20"))
        return max(0.01, min(1.0, value))
    except (ValueError, TypeError):
        return 0.20


def _get_default_risk_tolerance() -> str:
    """Get default risk tolerance from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "default_risk_tolerance"):
                risk = value.default_risk_tolerance
                if isinstance(risk, str) and risk in ("low", "medium", "high"):
                    return risk
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    risk = os.getenv("YIELD_DEFAULT_RISK_TOLERANCE", "medium").strip().lower()
    if risk in ("low", "medium", "high"):
        return risk
    return "medium"


def _get_default_lock_period() -> int:
    """Get default lock period from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "default_lock_period_days"):
                period = value.default_lock_period_days
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("YIELD_DEFAULT_LOCK_PERIOD_DAYS", "30"))
        return max(1, value)
    except (ValueError, TypeError):
        return 30


def _get_short_lock_period() -> int:
    """Get short lock period for auto-rotation from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "short_lock_period_days"):
                period = value.short_lock_period_days
                if isinstance(period, int) and period > 0:
                    return period
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("YIELD_SHORT_LOCK_PERIOD_DAYS", "7"))
        return max(1, value)
    except (ValueError, TypeError):
        return 7


def _get_strong_signal_threshold() -> float:
    """Get strong signal threshold from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "strong_signal_threshold"):
                threshold = value.strong_signal_threshold
                if isinstance(threshold, (int, float)) and 0 < threshold <= 1:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_STRONG_SIGNAL_THRESHOLD", "0.7"))
        return max(0.01, min(1.0, value))
    except (ValueError, TypeError):
        return 0.7


def _get_weak_signal_threshold() -> float:
    """Get weak signal threshold from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "weak_signal_threshold"):
                threshold = value.weak_signal_threshold
                if isinstance(threshold, (int, float)) and 0 < threshold <= 1:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_WEAK_SIGNAL_THRESHOLD", "0.3"))
        return max(0.01, min(1.0, value))
    except (ValueError, TypeError):
        return 0.3


def _get_yield_history_default_limit() -> int:
    """Get default yield history limit from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "history_default_limit"):
                limit = value.history_default_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("YIELD_HISTORY_DEFAULT_LIMIT", "50"))
        return max(1, value)
    except (ValueError, TypeError):
        return 50


def _get_protocols_ttl_sec() -> int:
    """Get protocols TTL from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "protocols_ttl_sec"):
                ttl = value.protocols_ttl_sec
                if isinstance(ttl, int) and ttl > 0:
                    return ttl
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("YIELD_PROTOCOLS_TTL_SEC", "300"))
        return max(1, value)
    except (ValueError, TypeError):
        return 300


def _get_protocols_endpoint() -> str:
    """Get protocols endpoint from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "protocols_endpoint"):
                endpoint = value.protocols_endpoint
                if isinstance(endpoint, str) and endpoint:
                    return endpoint.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    return os.getenv("YIELD_PROTOCOLS_ENDPOINT", "").strip()


def _get_protocols_fetch_timeout() -> float:
    """Get protocols fetch timeout from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "protocols_fetch_timeout_sec"):
                timeout = value.protocols_fetch_timeout_sec
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_PROTOCOLS_FETCH_TIMEOUT_SEC", "5.0"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 5.0


def _get_yield_db_path() -> str:
    """Get yield database path from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "db_path"):
                db_path = value.db_path
                if isinstance(db_path, str) and db_path:
                    return db_path.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable or canonical DB
    db_path = os.getenv("YIELD_DB_PATH", "").strip()
    if db_path:
        return db_path
    try:
        from backend.database_schema import DATABASE_PATH

        return DATABASE_PATH
    except ImportError:
        fn = os.getenv("MYSTIC_DB", "mystic_trading.db")
        return str(Path(__file__).parent.parent / fn)


def _get_default_early_exit_penalty_pct() -> float:
    """Get default early exit penalty percentage from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "default_early_exit_penalty_pct"):
                penalty = value.default_early_exit_penalty_pct
                if isinstance(penalty, (int, float)) and 0 <= penalty <= 100:
                    return float(penalty)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("YIELD_DEFAULT_EARLY_EXIT_PENALTY_PCT", "1.0"))
        return max(0.0, min(100.0, value))
    except (ValueError, TypeError):
        return 1.0


def _get_decimal_quantize_level() -> str:
    """Get decimal quantize level from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "yield", None)
            if value and hasattr(value, "decimal_quantize_level"):
                level = value.decimal_quantize_level
                if isinstance(level, str) and level:
                    return level
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    return os.getenv("YIELD_DECIMAL_QUANTIZE_LEVEL", "0.01")


if __name__ == "__main__":
    log.info("Yield Rotation Engine module loaded.")
