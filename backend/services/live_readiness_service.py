"""
Read-only live readiness report for operator dashboard and pre-flight checks.

Never logs or returns API secrets.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _api_key_present() -> tuple[bool, str]:
    key = (
        os.getenv("BINANCE_US_API_KEY", "")
        or os.getenv("BINANCEUS_API_KEY", "")
        or os.getenv("BINANCE_API_KEY", "")
    )
    secret = (
        os.getenv("BINANCE_US_SECRET_KEY", "")
        or os.getenv("BINANCE_US_API_SECRET", "")
        or os.getenv("BINANCEUS_API_SECRET", "")
        or os.getenv("BINANCE_SECRET", "")
        or os.getenv("BINANCE_SECRET_KEY", "")
    )
    if not key or len(key) < 10:
        return False, "Binance API key missing or too short (BINANCE_US_API_KEY / BINANCEUS_API_KEY / BINANCE_API_KEY)"
    if not secret or len(secret) < 10:
        return False, "Binance API secret missing or too short (BINANCE_US_SECRET_KEY / BINANCEUS_API_SECRET / BINANCE_SECRET)"
    return True, "configured"


async def _fetch_exchange_account_auth() -> dict[str, Any]:
    """Signed Binance.US account probe — permissions and USDT balance only."""
    import hashlib
    import hmac

    import httpx

    out: dict[str, Any] = {
        "binance_api_auth_status": "unknown",
        "can_trade": None,
        "can_withdraw": None,
        "usdt_free_balance": None,
        "open_binance_orders_count": None,
        "exchange_time_drift_ms": None,
        "errors": [],
    }
    key_ok, key_msg = _api_key_present()
    if not key_ok:
        out["binance_api_auth_status"] = "missing_keys"
        out["errors"].append(key_msg)
        return out

    api_key = (
        os.getenv("BINANCE_US_API_KEY", "")
        or os.getenv("BINANCEUS_API_KEY", "")
        or os.getenv("BINANCE_API_KEY", "")
    )
    api_secret = (
        os.getenv("BINANCE_US_SECRET_KEY", "")
        or os.getenv("BINANCE_US_API_SECRET", "")
        or os.getenv("BINANCEUS_API_SECRET", "")
        or os.getenv("BINANCE_SECRET", "")
        or os.getenv("BINANCE_SECRET_KEY", "")
    )

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            t_resp = await client.get("https://api.binance.us/api/v3/time")
            if t_resp.status_code == 200:
                server_ms = int(t_resp.json().get("serverTime", 0))
                local_ms = int(time.time() * 1000)
                out["exchange_time_drift_ms"] = abs(server_ms - local_ms)
            else:
                out["errors"].append(f"/time HTTP {t_resp.status_code}")

            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}&recvWindow=10000"
            signature = hmac.new(
                api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            url = f"https://api.binance.us/api/v3/account?{query_string}&signature={signature}"
            headers = {"X-MBX-APIKEY": api_key}
            a_resp = await client.get(url, headers=headers)
            if a_resp.status_code != 200:
                out["binance_api_auth_status"] = "auth_failed"
                out["errors"].append(f"/account HTTP {a_resp.status_code}")
                return out

            acct = a_resp.json()
            out["binance_api_auth_status"] = "ok"
            out["can_trade"] = bool(acct.get("canTrade", False))
            out["can_withdraw"] = bool(acct.get("canWithdraw", False))
            for bal in acct.get("balances", []):
                if str(bal.get("asset", "")).upper() == "USDT":
                    out["usdt_free_balance"] = float(bal.get("free", 0) or 0)
                    break

            oq = f"timestamp={int(time.time() * 1000)}&recvWindow=10000"
            osig = hmac.new(api_secret.encode("utf-8"), oq.encode("utf-8"), hashlib.sha256).hexdigest()
            ourl = f"https://api.binance.us/api/v3/openOrders?{oq}&signature={osig}"
            o_resp = await client.get(ourl, headers=headers)
            if o_resp.status_code == 200:
                orders = o_resp.json()
                out["open_binance_orders_count"] = len(orders) if isinstance(orders, list) else 0
            else:
                out["errors"].append(f"/openOrders HTTP {o_resp.status_code}")
    except Exception as exc:
        out["binance_api_auth_status"] = "error"
        out["errors"].append(str(exc)[:120])
        logger.debug("live readiness exchange probe failed: %s", exc)

    return out


def _tiny_live_checklist() -> list[dict[str, Any]]:
    from backend.config.live_test_mode import (
        LIVE_TEST_MAX_NOTIONAL,
        LIVE_TEST_MAX_OPEN_POSITIONS,
        LIVE_TEST_MODE,
        LIVE_TEST_REQUIRE_MANUAL_ARM,
        _runtime_live_test_manual_arm,
        _runtime_symbol_allowlist,
    )
    from backend.config.protected_execution import (
        PROTECTED_LIMIT_ALLOW_PARTIAL,
        USE_PROTECTED_LIMIT_EXECUTION,
    )

    allowlist = sorted(_runtime_symbol_allowlist())
    return [
        {"item": "USDT balance > 0 on exchange", "required": True},
        {"item": "Trade-only API key (canTrade=true)", "required": True},
        {"item": "Withdrawals disabled (canWithdraw=false)", "required": True},
        {"item": "LIVE_EXECUTION=true", "env": os.getenv("LIVE_EXECUTION", "false"), "required": True},
        {"item": "EXECUTION_MODE=live", "env": os.getenv("EXECUTION_MODE", "paper"), "required": True},
        {"item": "LIVE_TRADES_ALLOWED=true", "env": os.getenv("LIVE_TRADES_ALLOWED", "false"), "required": True},
        {"item": "LIVE_TEST_MODE=true", "env": str(LIVE_TEST_MODE).lower(), "required": True},
        {
            "item": "LIVE_TEST_MANUAL_ARM=true",
            "env": str(_runtime_live_test_manual_arm()).lower(),
            "required": LIVE_TEST_REQUIRE_MANUAL_ARM,
        },
        {"item": "LIVE_TEST_MAX_NOTIONAL", "value": LIVE_TEST_MAX_NOTIONAL, "required": True},
        {"item": "LIVE_TEST_MAX_OPEN_POSITIONS", "value": LIVE_TEST_MAX_OPEN_POSITIONS, "required": True},
        {"item": "LIVE_TEST_SYMBOL_ALLOWLIST", "value": allowlist, "required": True},
        {"item": "USE_PROTECTED_LIMIT_EXECUTION", "value": USE_PROTECTED_LIMIT_EXECUTION, "required": True},
        {
            "item": "PROTECTED_LIMIT_ALLOW_PARTIAL",
            "value": PROTECTED_LIMIT_ALLOW_PARTIAL,
            "required": False,
            "note": "Should remain false unless explicitly allowed",
        },
    ]


async def build_live_readiness_report() -> dict[str, Any]:
    """Single read-only live readiness packet."""
    from backend.config.live_test_mode import can_place_live_orders_sync, get_live_test_api_fields
    from backend.config.protected_execution import (
        PROTECTED_LIMIT_ALLOW_PARTIAL,
        USE_PROTECTED_LIMIT_EXECUTION,
        get_protected_execution_snapshot,
    )
    from backend.config.core_test_flags import ENABLE_SLEEVE_BLOCKING
    from backend.services.execution_mode_service import is_live_execution_allowed_sync

    execution_mode = (os.getenv("EXECUTION_MODE") or "paper").strip().lower()
    live_execution = _env_bool("LIVE_EXECUTION", False)
    live_trades_allowed = _env_bool("LIVE_TRADES_ALLOWED", False)
    live_fields = get_live_test_api_fields()
    permitted, block_reason = can_place_live_orders_sync()
    taker = float(os.getenv("TAKER_FEE", "0.0002") or 0.0002)
    prot = get_protected_execution_snapshot(taker_fee=taker)

    current_mode = "LIVE" if is_live_execution_allowed_sync() else "PAPER"

    exchange = await _fetch_exchange_account_auth()
    warnings: list[str] = []
    blockers: list[str] = []

    if exchange.get("can_withdraw") is True:
        warnings.append("canWithdraw=true — use trade-only API key without withdrawal permission")
    if exchange.get("can_trade") is False:
        blockers.append("canTrade=false on exchange account")
    usdt = exchange.get("usdt_free_balance")
    if usdt is not None and float(usdt) <= 0:
        blockers.append("USDT free balance is zero")
    if not live_fields.get("live_test_manual_arm") and live_fields.get("live_test_require_manual_arm"):
        blockers.append("LIVE_TEST_MANUAL_ARM not set")
    if not permitted:
        blockers.append(f"live_orders_blocked:{block_reason or 'unknown'}")
    if not live_execution:
        blockers.append("LIVE_EXECUTION not true")
    if execution_mode != "live":
        blockers.append("EXECUTION_MODE is not live")
    if not live_trades_allowed:
        blockers.append("LIVE_TRADES_ALLOWED not true")
    drift = exchange.get("exchange_time_drift_ms")
    if drift is not None and drift > 5000:
        blockers.append(f"exchange_time_drift_ms={drift} exceeds 5000")

    key_ok, key_msg = _api_key_present()
    if not key_ok:
        blockers.append(key_msg)

    ready_tiny = len(blockers) == 0 and (usdt is None or float(usdt) > 0)

    full_blockers = list(blockers)
    if not live_fields.get("full_live_confirmed"):
        full_blockers.append("FULL_LIVE_CONFIRMED not true")
    if live_fields.get("live_test_mode"):
        full_blockers.append("LIVE_TEST_MODE still true (full live expects test mode off)")
    ready_full = len(full_blockers) == 0

    return {
        "execution_mode": execution_mode,
        "LIVE_EXECUTION": live_execution,
        "LIVE_TRADES_ALLOWED": live_trades_allowed,
        "LIVE_TEST_MODE": live_fields.get("live_test_mode"),
        "LIVE_TEST_MANUAL_ARM": live_fields.get("live_test_manual_arm"),
        "FULL_LIVE_CONFIRMED": live_fields.get("full_live_confirmed"),
        "live_orders_permitted": permitted,
        "live_orders_block_reason": block_reason if not permitted else "",
        "protected_execution_enabled": bool(prot.use_protected_limit_execution),
        "protected_limit_allow_partial": bool(prot.protected_limit_allow_partial),
        "live_test_max_notional": live_fields.get("live_test_max_notional"),
        "live_test_max_open_positions": live_fields.get("live_test_max_open_positions"),
        "live_symbol_allowlist": live_fields.get("live_test_symbol_allowlist"),
        "binance_api_auth_status": exchange.get("binance_api_auth_status"),
        "can_trade": exchange.get("can_trade"),
        "can_withdraw": exchange.get("can_withdraw"),
        "usdt_free_balance": exchange.get("usdt_free_balance"),
        "open_binance_orders_count": exchange.get("open_binance_orders_count"),
        "exchange_time_drift_ms": exchange.get("exchange_time_drift_ms"),
        "current_local_mode": current_mode,
        "ready_for_tiny_live_test": ready_tiny,
        "ready_for_full_live": ready_full,
        "live_readiness_warnings": warnings,
        "live_readiness_blockers": blockers,
        "full_live_blockers": full_blockers,
        "tiny_live_checklist": _tiny_live_checklist(),
        "sleeve_blocking_enabled": ENABLE_SLEEVE_BLOCKING,
        "sleeve_telemetry_only": not ENABLE_SLEEVE_BLOCKING,
        "exchange_probe_errors": exchange.get("errors", []),
    }
