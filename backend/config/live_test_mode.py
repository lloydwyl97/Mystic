"""
Live tiny-test safety caps — apply only in live execution context.

Paper mode is never affected. Full live without LIVE_TEST_MODE requires
FULL_LIVE_CONFIRMED=true at startup and before live orders.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("LIVE_TEST env %s=%r invalid; using %s", name, raw, default)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("LIVE_TEST env %s=%r invalid; using %s", name, raw, default)
        return int(default)


def _parse_symbol_allowlist(raw: str) -> frozenset[str]:
    if not raw or not raw.strip():
        return frozenset()
    out: set[str] = set()
    for part in raw.split(","):
        token = part.strip().upper().replace("/", "")
        if token:
            out.add(token)
    return frozenset(out)


LIVE_TEST_MODE: Final[bool] = _env_bool("LIVE_TEST_MODE", False)
LIVE_TEST_MAX_NOTIONAL: Final[float] = _env_float("LIVE_TEST_MAX_NOTIONAL", 25.0)
LIVE_TEST_MAX_OPEN_POSITIONS: Final[int] = _env_int("LIVE_TEST_MAX_OPEN_POSITIONS", 1)
LIVE_TEST_SYMBOL_ALLOWLIST: Final[frozenset[str]] = _parse_symbol_allowlist(os.getenv("LIVE_TEST_SYMBOL_ALLOWLIST", "BTCUSDT,ETHUSDT"))
LIVE_TEST_REQUIRE_MANUAL_ARM: Final[bool] = _env_bool("LIVE_TEST_REQUIRE_MANUAL_ARM", True)
LIVE_TEST_MANUAL_ARM: Final[bool] = _env_bool("LIVE_TEST_MANUAL_ARM", False)
FULL_LIVE_CONFIRMED: Final[bool] = _env_bool("FULL_LIVE_CONFIRMED", False)


@dataclass(frozen=True)
class LiveTestModeSnapshot:
    live_test_mode: bool
    live_test_max_notional: float
    live_test_max_open_positions: int
    live_test_symbol_allowlist: tuple[str, ...]
    live_test_require_manual_arm: bool
    live_test_manual_arm: bool
    full_live_confirmed: bool
    live_execution_context: bool
    live_test_mode_active: bool
    live_orders_permitted: bool
    live_orders_block_reason: str


def _execution_mode_value() -> str:
    return (os.getenv("EXECUTION_MODE") or "paper").strip().lower()


def _live_trades_allowed_value() -> bool:
    raw = os.getenv("LIVE_TRADES_ALLOWED", "false")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _live_execution_flag_value() -> bool:
    return _env_bool(os.getenv("LIVE_EXECUTION", "false"), default=False)


def is_live_execution_context() -> bool:
    """True when EXECUTION_MODE=live or LIVE_EXECUTION=true."""
    return _execution_mode_value() == "live" or _live_execution_flag_value()


def is_live_test_mode_active() -> bool:
    return bool(LIVE_TEST_MODE and is_live_execution_context())


def _normalize_exchange_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "")


def is_symbol_allowed_for_live_test(symbol: str) -> bool:
    if not LIVE_TEST_SYMBOL_ALLOWLIST:
        return True
    return _normalize_exchange_symbol(symbol) in LIVE_TEST_SYMBOL_ALLOWLIST


def live_test_max_open_positions_limit() -> int | None:
    if not is_live_test_mode_active():
        return None
    return max(1, int(LIVE_TEST_MAX_OPEN_POSITIONS))


def can_place_live_orders_sync() -> tuple[bool, str]:
    """Gate real exchange orders (BUY/SELL placement). Paper is unaffected."""
    from backend.services.execution_mode_service import is_live_execution_allowed_sync

    if not is_live_execution_allowed_sync():
        return False, "LIVE_ORDERS_NOT_ALLOWED"

    if not is_live_execution_context():
        return False, "NOT_LIVE_EXECUTION_CONTEXT"

    if is_live_test_mode_active():
        if LIVE_TEST_REQUIRE_MANUAL_ARM and not LIVE_TEST_MANUAL_ARM:
            return False, "LIVE_TEST_MANUAL_ARM_REQUIRED"
        return True, "LIVE_TEST_MODE"

    if not FULL_LIVE_CONFIRMED:
        return False, "FULL_LIVE_CONFIRMED_REQUIRED"
    return True, "FULL_LIVE"


def check_full_live_readiness_requirements() -> list[str]:
    """Extra readiness failures for full-live vs tiny-test paths."""
    failures: list[str] = []
    mode = _execution_mode_value()
    allowed = _live_trades_allowed_value()
    if mode != "live" and not _live_execution_flag_value():
        return failures

    if LIVE_TEST_MODE:
        if LIVE_TEST_MAX_NOTIONAL <= 0:
            failures.append("LIVE_TEST_MAX_NOTIONAL must be > 0 when LIVE_TEST_MODE=true")
        if LIVE_TEST_MAX_OPEN_POSITIONS < 1:
            failures.append("LIVE_TEST_MAX_OPEN_POSITIONS must be >= 1 when LIVE_TEST_MODE=true")
        if not LIVE_TEST_SYMBOL_ALLOWLIST:
            failures.append("LIVE_TEST_SYMBOL_ALLOWLIST must be non-empty when LIVE_TEST_MODE=true")
        return failures

    if mode == "live" and allowed and not FULL_LIVE_CONFIRMED:
        failures.append("FULL_LIVE_CONFIRMED=true required when LIVE_TEST_MODE=false")
    return failures


def assert_full_live_safety_at_startup() -> None:
    mode = _execution_mode_value()
    allowed = _live_trades_allowed_value()
    if mode == "live" and allowed and not LIVE_TEST_MODE and not FULL_LIVE_CONFIRMED:
        msg = (
            "Refusing full live trading: EXECUTION_MODE=live, LIVE_TRADES_ALLOWED=true, "
            "LIVE_TEST_MODE=false, and FULL_LIVE_CONFIRMED is not true. "
            "Set FULL_LIVE_CONFIRMED=true for intentional full live, or enable LIVE_TEST_MODE."
        )
        logger.error(msg)
        raise RuntimeError(msg)


def get_live_test_api_fields() -> dict[str, Any]:
    permitted, block_reason = can_place_live_orders_sync()
    return {
        "live_test_mode": LIVE_TEST_MODE,
        "live_test_max_notional": LIVE_TEST_MAX_NOTIONAL,
        "live_test_max_open_positions": LIVE_TEST_MAX_OPEN_POSITIONS,
        "live_test_symbol_allowlist": sorted(LIVE_TEST_SYMBOL_ALLOWLIST),
        "live_test_require_manual_arm": LIVE_TEST_REQUIRE_MANUAL_ARM,
        "live_test_manual_arm": LIVE_TEST_MANUAL_ARM,
        "full_live_confirmed": FULL_LIVE_CONFIRMED,
        "live_execution_context": is_live_execution_context(),
        "live_test_mode_active": is_live_test_mode_active(),
        "live_orders_permitted": permitted,
        "live_orders_block_reason": block_reason if not permitted else "",
    }


def log_live_test_mode_at_startup() -> LiveTestModeSnapshot:
    permitted, block_reason = can_place_live_orders_sync()
    snap = LiveTestModeSnapshot(
        live_test_mode=LIVE_TEST_MODE,
        live_test_max_notional=LIVE_TEST_MAX_NOTIONAL,
        live_test_max_open_positions=LIVE_TEST_MAX_OPEN_POSITIONS,
        live_test_symbol_allowlist=tuple(sorted(LIVE_TEST_SYMBOL_ALLOWLIST)),
        live_test_require_manual_arm=LIVE_TEST_REQUIRE_MANUAL_ARM,
        live_test_manual_arm=LIVE_TEST_MANUAL_ARM,
        full_live_confirmed=FULL_LIVE_CONFIRMED,
        live_execution_context=is_live_execution_context(),
        live_test_mode_active=is_live_test_mode_active(),
        live_orders_permitted=permitted,
        live_orders_block_reason=block_reason if not permitted else "",
    )
    logger.warning(
        "LIVE_TEST_MODE enabled=%s max_notional=%s max_open_positions=%s allowlist=%s "
        "require_manual_arm=%s manual_arm=%s full_live_confirmed=%s live_context=%s "
        "active=%s live_orders_permitted=%s block_reason=%s",
        snap.live_test_mode,
        snap.live_test_max_notional,
        snap.live_test_max_open_positions,
        list(snap.live_test_symbol_allowlist),
        snap.live_test_require_manual_arm,
        snap.live_test_manual_arm,
        snap.full_live_confirmed,
        snap.live_execution_context,
        snap.live_test_mode_active,
        snap.live_orders_permitted,
        snap.live_orders_block_reason or "none",
    )
    return snap


NormalizeFn = Callable[[str, float, float, str], tuple[float, str, float]]


def enforce_live_order_buy_gates(
    *,
    symbol: str,
    quantity: float,
    price: float,
    normalize_amount: NormalizeFn,
) -> tuple[bool, str, float]:
    """
    Live-only BUY gates: full-live interlock, manual arm, allowlist, notional cap.
    Call only when is_live_execution_allowed_sync() is true.
    """
    permitted, block_reason = can_place_live_orders_sync()
    if not permitted:
        return False, block_reason, quantity

    if not is_live_test_mode_active():
        return True, "", quantity

    if not is_symbol_allowed_for_live_test(symbol):
        allow = ",".join(sorted(LIVE_TEST_SYMBOL_ALLOWLIST))
        return False, f"LIVE_TEST_SYMBOL_NOT_ALLOWED:{_normalize_exchange_symbol(symbol)} not in [{allow}]", quantity

    if price <= 0:
        return False, "LIVE_TEST_INVALID_PRICE", quantity

    est_notional = float(quantity) * float(price)
    max_notional = float(LIVE_TEST_MAX_NOTIONAL)
    if est_notional > max_notional:
        capped_qty = max_notional / float(price)
        qty_q, norm_reason, _ = normalize_amount(symbol, capped_qty, price, "BUY")
        if norm_reason != "ok":
            return False, f"LIVE_TEST_NOTIONAL_CAP_NORMALIZE_FAILED:{norm_reason}", quantity
        capped_notional = qty_q * price
        if capped_notional > max_notional + 1e-9:
            return False, f"LIVE_TEST_NOTIONAL_EXCEEDS_CAP:{capped_notional:.4f}>{max_notional:.4f}", quantity
        logger.warning(
            "LIVE_TEST_NOTIONAL_CAP: %s qty %.8f -> %.8f (notional %.4f -> %.4f cap=%.4f)",
            symbol,
            quantity,
            qty_q,
            est_notional,
            capped_notional,
            max_notional,
        )
        quantity = qty_q

    return True, "", quantity
