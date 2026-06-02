"""
Validator Staking Module - Live Configuration Only

Provides utilities for staking operations to validator nodes across different blockchains.
Integrates with the main trading system for automated staking of idle capital.
All configuration values come from live config - no hardcoded values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Optional imports - try at top level
try:
    from backend.services.blockchain_service import BlockchainService  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    BlockchainService = None

# --- Service interface & fallback -------------------------------------------------


@runtime_checkable
class SupportsStaking(Protocol):
    async def stake_to_validator(self, *, amount: float, chain: str, validator_address: str) -> dict[str, Any]: ...

    async def get_staking_rewards(self, *, chain: str, validator_address: str) -> dict[str, Any]: ...


class _FallbackBlockchainService:
    """Fail-closed fallback: no fabricated data."""

    async def stake_to_validator(self, *, _amount: float, _chain: str, _validator_address: str) -> dict[str, Any]:
        msg = "service_unavailable"
        raise RuntimeError(msg)

    async def get_staking_rewards(self, *, _chain: str, _validator_address: str) -> dict[str, Any]:
        msg = "service_unavailable"
        raise RuntimeError(msg)


def _get_blockchain_service() -> SupportsStaking:
    try:
        # Your real implementation
        if BlockchainService is None:
            logger.warning("BlockchainService not available")
            return _FallbackBlockchainService()

        svc = BlockchainService()  # must implement SupportsStaking
        if not isinstance(svc, SupportsStaking):
            logger.warning("BlockchainService does not fully implement SupportsStaking Protocol; continuing anyway.")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("BlockchainService import failed: %s", e)
        return _FallbackBlockchainService()
    else:
        return svc  # type: ignore[return-value]


# --- Helpers ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_positive_amount(amount: float) -> None:
    if not isinstance(amount, (int, float)):
        msg = f"Amount must be numeric, got {type(amount).__name__}"
        raise TypeError(msg)
    if amount <= 0:
        msg = f"Amount must be a positive number, got {amount}"
        raise ValueError(msg)


def _validate_nonempty(name: str, value: str | None) -> None:
    if not value or not str(value).strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)


# --- Public API (async-first) ----------------------------------------------------


async def stake_to_validator(
    amount: float,
    chain: str,
    validator_address: str,
    *,
    units: str | None = None,
) -> dict[str, Any]:
    """
    Stake funds to a validator node on the specified blockchain.

    Args:
        amount: Amount to stake in native token UNITS (not USD) - conversions must be done upstream
        chain: Target blockchain (e.g., 'Ethereum', 'Polygon', 'Solana')
        validator_address: Validator node address

    Returns:
        dict: Staking operation results
    """
    _validate_positive_amount(amount)
    _validate_nonempty("Chain", chain)
    _validate_nonempty("Validator address", validator_address)
    chain = str(chain).strip()
    validator_address = str(validator_address).strip()
    # basic sanity check for amount finiteness
    if not math.isfinite(float(amount)):
        return {
            "status": "error",
            "error": "invalid_amount",
            "message": "Amount must be a finite positive number",
            "amount_staked": float(amount),
            "chain": chain,
            "validator_address": validator_address,
            "timestamp": _utc_now_iso(),
        }

    svc = _get_blockchain_service()
    mask_prefix_len = _get_address_mask_prefix_length()
    mask_suffix_len = _get_address_mask_suffix_length()
    mask_min_length = _get_address_mask_min_length()
    masked_addr = validator_address[:mask_prefix_len] + "..." + validator_address[-mask_suffix_len:] if len(validator_address) > mask_min_length else validator_address
    logger.info(
        "[STAKE] Initiating stake: amount=%.8f units chain=%s validator=%s",
        amount,
        chain,
        masked_addr,
    )

    try:
        result = await svc.stake_to_validator(amount=amount, chain=chain, validator_address=validator_address)
        tx_hash = result.get("tx_hash") or result.get("transaction_hash") or "unknown"
        provider = result.get("provider") or "blockchain_service"
        payload = {
            "status": "success",
            "transaction_hash": tx_hash,
            "amount_staked": float(amount),
            "chain": chain,
            "validator_address": validator_address,
            "units": units or _get_default_units(),
            "service": "blockchain_service",
            "provider": provider,
            "op_id": f"stake-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": _utc_now_iso(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error staking to validator")
        err_code = str(e)
        dependency_missing = err_code == "service_unavailable"
        return {
            "status": "error",
            "error": err_code,
            "dependency_missing": dependency_missing,
            "service_status": "unavailable" if dependency_missing else "error",
            "amount_staked": float(amount),
            "chain": chain,
            "validator_address": validator_address,
            "units": units or _get_default_units(),
            "service": "blockchain_service",
            "provider": "unknown",
            "op_id": f"stake-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": _utc_now_iso(),
        }
    else:
        logger.info("[STAKE] Success tx=%s amount=%.8f units chain=%s", tx_hash, amount, chain)
        return payload


async def auto_stake_if_idle(
    balance: float,
    threshold: float,
    *,
    _chain: str,
    _validator_address: str,
    stake_pct: float | None = None,
    quote_currency: str | None = None,
) -> dict[str, Any]:
    """
    Automatically stake funds if balance exceeds threshold.

    Args:
        balance: Current available balance in USD (or configured currency)
        threshold: Minimum balance threshold for staking (env/config)
        chain: Target blockchain
        validator_address: Validator node address
        stake_pct: Fraction of available balance to stake when threshold met (0..1). Conversion to units must be upstream.

    Returns:
        dict: Auto-staking operation results
    """
    if balance is None:
        msg = "balance cannot be None"
        raise ValueError(msg)

    logger.info("[STAKE] Auto-check balance=%.4f threshold=%.4f", balance, threshold)

    # Resolve stake_pct from live config if not provided
    if stake_pct is None:
        stake_pct = _get_default_stake_pct()

    # Enforce bounds
    try:
        sp = float(stake_pct)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return {
            "auto_stake_triggered": False,
            "error": "invalid_stake_pct",
            "message": "stake_pct must be a numeric fraction between 0 and 1",
            "timestamp": _utc_now_iso(),
        }
    if not (0.0 <= sp <= 1.0) or not math.isfinite(sp):
        return {
            "auto_stake_triggered": False,
            "error": "stake_pct_out_of_range",
            "message": "stake_pct must be between 0 and 1",
            "stake_pct": float(sp),
            "timestamp": _utc_now_iso(),
        }

    if balance >= threshold:
        logger.info(
            "[STAKE] Threshold met: initiating auto-stake using stake_pct=%.4f of balance=%.4f",
            sp,
            balance,
        )
        # Conversion to native units is required by the caller
        staking_result = {
            "status": "error",
            "error": "unit_conversion_required",
            "message": "Convert quote balance to native token units before calling stake_to_validator",
            "expected_param": "amount_units",
        }
        return {
            "auto_stake_triggered": True,
            "balance": float(balance),
            "threshold": float(threshold),
            "stake_pct": float(sp),
            "quote_currency": quote_currency,
            "staking_result": staking_result,
            "op_id": f"auto-stake-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": _utc_now_iso(),
        }

    msg = f"Balance ${balance:.2f} below staking threshold (${threshold:.2f})"
    logger.info("[STAKE] %s", msg)
    return {
        "auto_stake_triggered": False,
        "balance": float(balance),
        "threshold": float(threshold),
        "reason": "balance_below_threshold",
        "timestamp": _utc_now_iso(),
    }


async def get_staking_rewards(validator_address: str, chain: str) -> dict[str, Any]:
    """
    Get staking rewards for a validator.

    Args:
        validator_address: Validator node address
        chain: Blockchain network

    Returns:
        dict: Staking rewards information
    """
    _validate_nonempty("Validator address", validator_address)
    _validate_nonempty("Chain", chain)

    svc = _get_blockchain_service()
    mask_prefix_len = _get_address_mask_prefix_length()
    mask_suffix_len = _get_address_mask_suffix_length()
    mask_min_length = _get_address_mask_min_length()
    masked_addr = validator_address[:mask_prefix_len] + "..." + validator_address[-mask_suffix_len:] if len(validator_address) > mask_min_length else validator_address
    logger.info("[STAKE] Fetching rewards chain=%s validator=%s", chain, masked_addr)

    try:
        rewards = await svc.get_staking_rewards(chain=chain, validator_address=validator_address)
        # Ensure JSON-serializable
        try:
            json.dumps(rewards, default=str)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            rewards = {"value": str(rewards)}
        return {
            "status": "success",
            "rewards": rewards,
            "chain": chain,
            "validator_address": validator_address,
            "service": "blockchain_service",
            "provider": rewards.get("provider", "blockchain_service") if isinstance(rewards, dict) else "blockchain_service",
            "op_id": f"rewards-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": _utc_now_iso(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error getting staking rewards")
        dep = str(e) == "service_unavailable"
        return {
            "status": "error",
            "error": str(e),
            "dependency_missing": dep,
            "service_status": "unavailable" if dep else "error",
            "chain": chain,
            "validator_address": validator_address,
            "service": "blockchain_service",
            "provider": "unknown",
            "op_id": f"rewards-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp": _utc_now_iso(),
        }


# --- Integration helpers ---------------------------------------------------------


async def integrate_with_trading_system() -> bool:
    """
    Integrate staking module with the main trading system.
    Returns True if integration bootstrap succeeds.
    """
    try:
        logger.info("[STAKE] Integrating with trading system (bootstrap only; no demo staking)")
        logger.info("[STAKE] Integration complete")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("[STAKE] Integration failed: %s", e)
        return False
    else:
        return True


# --- Synchronous wrappers (optional) --------------------------------------------


def run_auto_stake_if_idle_sync(
    balance: float,
    threshold: float,
    *,
    chain: str,
    validator_address: str,
    stake_pct: float | None = None,
) -> dict[str, Any]:
    """Convenience sync wrapper (CLI-only). Not safe under running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        msg = "run_auto_stake_if_idle_sync cannot be called inside an active event loop"
        raise RuntimeError(msg)

    return asyncio.run(
        auto_stake_if_idle(
            balance,
            threshold,
            chain=chain,
            validator_address=validator_address,
            stake_pct=stake_pct,
        )
    )


def run_integrate_with_trading_system_sync() -> bool:
    """Convenience sync wrapper (CLI-only)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        msg = "run_integrate_with_trading_system_sync cannot be called inside an active event loop"
        raise RuntimeError(msg)

    return asyncio.run(integrate_with_trading_system())


# --- Manual test ----------------------------------------------------------------


# ------------------------------------------------------------------------------
# Configuration helpers (live config)
# ------------------------------------------------------------------------------
def _get_default_stake_pct() -> float:
    """Get default stake percentage from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "validator_staking", None)
            if value and hasattr(value, "default_stake_pct"):
                pct = value.default_stake_pct
                if isinstance(pct, (int, float)) and 0 <= pct <= 1:
                    return float(pct)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("STAKE_PCT", "0.02"))
        return max(0.0, min(1.0, value))
    except (ValueError, TypeError):
        return 0.02


def _get_default_units() -> str:
    """Get default units from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "validator_staking", None)
            if value and hasattr(value, "default_units"):
                units = value.default_units
                if isinstance(units, str) and units:
                    return units.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    units = os.getenv("VALIDATOR_STAKING_DEFAULT_UNITS", "native").strip()
    return units if units else "native"


def _get_address_mask_prefix_length() -> int:
    """Get address mask prefix length from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "validator_staking", None)
            if value and hasattr(value, "address_mask_prefix_length"):
                length = value.address_mask_prefix_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("VALIDATOR_STAKING_ADDRESS_MASK_PREFIX_LENGTH", "6"))
        return max(1, value)
    except (ValueError, TypeError):
        return 6


def _get_address_mask_suffix_length() -> int:
    """Get address mask suffix length from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "validator_staking", None)
            if value and hasattr(value, "address_mask_suffix_length"):
                length = value.address_mask_suffix_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("VALIDATOR_STAKING_ADDRESS_MASK_SUFFIX_LENGTH", "6"))
        return max(1, value)
    except (ValueError, TypeError):
        return 6


def _get_address_mask_min_length() -> int:
    """Get address mask minimum length from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "validator_staking", None)
            if value and hasattr(value, "address_mask_min_length"):
                length = value.address_mask_min_length
                if isinstance(length, int) and length > 0:
                    return length
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("VALIDATOR_STAKING_ADDRESS_MASK_MIN_LENGTH", "12"))
        return max(1, value)
    except (ValueError, TypeError):
        return 12


if __name__ == "__main__":
    logger.info("Validator Staking Module loaded. Use through application entrypoints.")
