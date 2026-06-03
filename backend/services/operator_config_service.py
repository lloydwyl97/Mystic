"""
Dashboard-updatable operator limits and live-test caps.

Writes .env, updates os.environ, and applies hot values to the portfolio engine
module so changes take effect without restart.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from backend.services.execution_mode_service import _read_env_file, _write_env_file

logger = logging.getLogger(__name__)

_DEFAULT_MAX_OPEN = 4
_DEFAULT_LIVE_TEST_MAX_OPEN = 4
_DEFAULT_LIVE_TEST_NOTIONAL = 25.0
_DEFAULT_RISK_PCT = 0.04
_DEFAULT_MAX_CASH_PER_COIN = 0.25


def _env_float(name: str, default: float) -> float:
    file_env = _read_env_file()
    raw = os.getenv(name) or file_env.get(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    file_env = _read_env_file()
    raw = os.getenv(name) or file_env.get(name, "")
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    file_env = _read_env_file()
    raw = os.getenv(name) or file_env.get(name, "")
    if not raw:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def get_max_open_positions() -> int:
    file_env = _read_env_file()
    raw = (
        os.getenv("MAX_OPEN_POSITIONS")
        or os.getenv("MAX_POSITIONS")
        or file_env.get("MAX_OPEN_POSITIONS")
        or file_env.get("MAX_POSITIONS")
    )
    try:
        return max(1, int(raw)) if raw else _DEFAULT_MAX_OPEN
    except (TypeError, ValueError):
        return _DEFAULT_MAX_OPEN


def get_live_test_max_open_positions() -> int:
    return max(1, _env_int("LIVE_TEST_MAX_OPEN_POSITIONS", _DEFAULT_LIVE_TEST_MAX_OPEN))


def get_live_test_max_notional() -> float:
    return max(0.01, _env_float("LIVE_TEST_MAX_NOTIONAL", _DEFAULT_LIVE_TEST_NOTIONAL))


def get_live_test_manual_arm() -> bool:
    return _env_bool("LIVE_TEST_MANUAL_ARM", False)


def get_live_test_symbol_allowlist_raw() -> str:
    file_env = _read_env_file()
    return (os.getenv("LIVE_TEST_SYMBOL_ALLOWLIST") or file_env.get("LIVE_TEST_SYMBOL_ALLOWLIST") or "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT").strip()


def get_risk_per_trade_pct() -> float:
    # Accept either 4 (=4%) or 0.04 (=4%) for operator convenience
    raw = os.getenv("RISK_PER_TRADE_PCT") or _read_env_file().get("RISK_PER_TRADE_PCT", "")
    if raw:
        try:
            val = float(raw)
            return val / 100.0 if val > 1.0 else val
        except (TypeError, ValueError):
            pass
    return _DEFAULT_RISK_PCT


def get_max_cash_per_coin_pct() -> float:
    val = _env_float("MAX_CASH_PER_COIN_PCT", _DEFAULT_MAX_CASH_PER_COIN)
    return val / 100.0 if val > 1.0 else val


def apply_runtime_config() -> None:
    """Push current env-backed limits into the running portfolio engine module."""
    try:
        import backend.services.portfolio_engine as pe

        pe.MAX_OPEN_POSITIONS = get_max_open_positions()
        pe.RISK_PER_TRADE_PCT = get_risk_per_trade_pct()
        logger.info(
            "OPERATOR_CONFIG_APPLIED max_open_positions=%s risk_per_trade_pct=%.4f live_test_max_open=%s live_test_max_notional=%.2f manual_arm=%s",
            pe.MAX_OPEN_POSITIONS,
            pe.RISK_PER_TRADE_PCT,
            get_live_test_max_open_positions(),
            get_live_test_max_notional(),
            get_live_test_manual_arm(),
        )
    except Exception as exc:
        logger.warning("OPERATOR_CONFIG_APPLY_FAILED: %s", exc)


async def get_operator_config() -> dict[str, Any]:
    from backend.config.live_test_mode import get_live_test_api_fields
    from backend.services.portfolio_engine import get_portfolio_engine

    engine = get_portfolio_engine()
    kill = engine.get_kill_switch_status()
    live_fields = get_live_test_api_fields()
    return {
        "max_open_positions": get_max_open_positions(),
        "live_test_max_open_positions": get_live_test_max_open_positions(),
        "live_test_max_notional": get_live_test_max_notional(),
        "live_test_symbol_allowlist": get_live_test_symbol_allowlist_raw(),
        "live_test_manual_arm": get_live_test_manual_arm(),
        "risk_per_trade_pct": round(get_risk_per_trade_pct() * 100, 2),
        "max_cash_per_coin_pct": round(get_max_cash_per_coin_pct() * 100, 2),
        "kill_switch": kill.get("mode"),
        "kill_switch_reason": kill.get("reason"),
        **live_fields,
    }


async def set_operator_config(payload: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, str] = {}

    if "max_open_positions" in payload and payload["max_open_positions"] is not None:
        updates["MAX_OPEN_POSITIONS"] = str(max(1, int(payload["max_open_positions"])))

    if "live_test_max_open_positions" in payload and payload["live_test_max_open_positions"] is not None:
        updates["LIVE_TEST_MAX_OPEN_POSITIONS"] = str(max(1, int(payload["live_test_max_open_positions"])))

    if "live_test_max_notional" in payload and payload["live_test_max_notional"] is not None:
        updates["LIVE_TEST_MAX_NOTIONAL"] = str(max(0.01, float(payload["live_test_max_notional"])))

    if "live_test_symbol_allowlist" in payload and payload["live_test_symbol_allowlist"] is not None:
        allow = str(payload["live_test_symbol_allowlist"]).strip()
        updates["LIVE_TEST_SYMBOL_ALLOWLIST"] = allow.upper().replace(" ", "")

    if "live_test_manual_arm" in payload and payload["live_test_manual_arm"] is not None:
        updates["LIVE_TEST_MANUAL_ARM"] = "true" if bool(payload["live_test_manual_arm"]) else "false"

    if "risk_per_trade_pct" in payload and payload["risk_per_trade_pct"] is not None:
        val = float(payload["risk_per_trade_pct"])
        updates["RISK_PER_TRADE_PCT"] = str(val)

    if "max_cash_per_coin_pct" in payload and payload["max_cash_per_coin_pct"] is not None:
        val = float(payload["max_cash_per_coin_pct"])
        updates["MAX_CASH_PER_COIN_PCT"] = str(val)

    if "kill_switch" in payload and payload["kill_switch"] is not None:
        from backend.services.portfolio_engine import get_portfolio_engine

        engine = get_portfolio_engine()
        reason = str(payload.get("kill_switch_reason") or "dashboard")
        await engine.set_kill_switch(str(payload["kill_switch"]).strip().upper(), reason)

    if updates:
        _write_env_file(updates)
        for key, value in updates.items():
            os.environ[key] = value
        apply_runtime_config()
        logger.info("OPERATOR_CONFIG_UPDATED keys=%s", sorted(updates.keys()))

    return await get_operator_config()
