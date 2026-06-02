import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = _PROJECT_ROOT / ".env"


def _read_env_file() -> dict[str, str]:
    data: dict[str, str] = {}
    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def _write_env_file(values: dict[str, str]) -> None:
    existing_lines: list[str] = []
    if ENV_PATH.exists():
        existing_lines = ENV_PATH.read_text().splitlines()

    seen: set[str] = set()
    output: list[str] = []

    for raw_line in existing_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            output.append(raw_line)
            continue
        key, _ = raw_line.split("=", 1)
        k = key.strip()
        if k in values:
            output.append(f"{k}={values[k]}")
            seen.add(k)
        else:
            output.append(raw_line)

    for k, v in values.items():
        if k not in seen:
            output.append(f"{k}={v}")

    ENV_PATH.write_text("\n".join(output) + "\n")


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def get_execution_status() -> dict[str, Any]:
    from backend.config.live_test_mode import get_live_test_api_fields

    file_env = _read_env_file()
    mode = (os.getenv("EXECUTION_MODE") or file_env.get("EXECUTION_MODE") or "paper").strip().lower()
    allowed = _normalize_bool(
        os.getenv("LIVE_TRADES_ALLOWED", file_env.get("LIVE_TRADES_ALLOWED", "false")),
        default=False,
    )
    status = {
        "execution_mode": mode,
        "live_trades_allowed": allowed,
        "is_live_execution_allowed": mode == "live" and allowed,
    }
    status.update(get_live_test_api_fields())
    return status


async def set_execution_mode(mode: str) -> dict[str, Any]:
    normalized = (mode or "").strip().lower()
    if normalized not in {"paper", "live"}:
        raise ValueError("mode must be 'paper' or 'live'")
    os.environ["EXECUTION_MODE"] = normalized
    os.environ["TRADING_MODE"] = normalized
    _write_env_file({"EXECUTION_MODE": normalized, "TRADING_MODE": normalized})
    logger.info("Execution mode set to '%s' (TRADING_MODE synced)", normalized)
    return await get_execution_status()


async def set_live_trades_allowed(enabled: bool) -> dict[str, Any]:
    normalized = "true" if bool(enabled) else "false"
    os.environ["LIVE_TRADES_ALLOWED"] = normalized
    _write_env_file({"LIVE_TRADES_ALLOWED": normalized})
    return await get_execution_status()


def is_live_execution_allowed_sync() -> bool:
    file_env = _read_env_file()
    mode = (os.getenv("EXECUTION_MODE") or file_env.get("EXECUTION_MODE") or "paper").strip().lower()
    allowed = _normalize_bool(
        os.getenv("LIVE_TRADES_ALLOWED", file_env.get("LIVE_TRADES_ALLOWED", "false")),
        default=False,
    )
    return mode == "live" and allowed


async def check_live_readiness() -> dict[str, Any]:
    """Run pre-flight checks before allowing live mode switch.

    Checks:
    1. API key presence
    2. LIVE_EXECUTION env flag
    3. Portfolio engine cash > 0
    4. Exchange connectivity + time sync (Binance US)
    5. Kill switch not active
    """
    import asyncio

    failures: list[str] = []

    api_key = os.getenv("BINANCE_US_API_KEY", "") or os.getenv("BINANCEUS_API_KEY", "")
    api_secret = os.getenv("BINANCE_US_SECRET_KEY", "") or os.getenv("BINANCE_US_API_SECRET", "") or os.getenv("BINANCEUS_API_SECRET", "")
    if not api_key or len(api_key) < 10:
        failures.append("BINANCE_US_API_KEY missing or too short")
    if not api_secret or len(api_secret) < 10:
        failures.append("BINANCE_US_SECRET_KEY missing or too short")

    live_exec = _normalize_bool(os.getenv("LIVE_EXECUTION", "false"), default=False)
    if not live_exec:
        failures.append("LIVE_EXECUTION env var is not set to true")

    try:
        from backend.services.portfolio_engine import get_portfolio_engine

        engine = get_portfolio_engine()
        cash = getattr(engine, "cash_balance", getattr(engine, "_cash_balance", 0))
        if cash <= 0:
            failures.append(f"Cash balance is {cash} (must be > 0)")

        if getattr(engine, "_kill_switch_on", False):
            failures.append("Kill switch is active — disable before enabling live")
    except Exception as e:
        failures.append(f"Portfolio engine check failed: {str(e)[:80]}")

    if api_key and len(api_key) >= 10:
        try:
            import time as _time

            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://api.binance.us/api/v3/time")
                if resp.status_code == 200:
                    server_ms = resp.json().get("serverTime", 0)
                    local_ms = int(_time.time() * 1000)
                    drift_ms = abs(server_ms - local_ms)
                    if drift_ms > 5000:
                        failures.append(f"Exchange time drift {drift_ms}ms exceeds 5s tolerance")
                else:
                    failures.append(f"Binance US /time returned HTTP {resp.status_code}")
        except Exception as e:
            failures.append(f"Exchange connectivity failed: {str(e)[:80]}")

    try:
        from backend.config.live_test_mode import check_full_live_readiness_requirements

        failures.extend(check_full_live_readiness_requirements())
    except Exception as e:
        failures.append(f"Live test mode check failed: {str(e)[:80]}")

    ready = len(failures) == 0
    if not ready:
        logger.warning("Live readiness check FAILED: %s", failures)
    return {"ready": ready, "failures": failures}
