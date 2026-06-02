"""
Explicit feature flags for layered local testing (CORE_ONLY_MODE and ENABLE_*).
No code deletion: gates read these flags and either enforce or telemetry-only/log.

Resolution order for each ENABLE_*:
1. If the env var is set (true/false), that wins.
2. Else if CORE_ONLY_MODE is true → use the "phase-1 core" default (usually off).
3. Else → use production-style defaults (layers on unless historically optional).
"""

from __future__ import annotations

import os

# Mirror risk_governor.GOVERNANCE_SHADOW_ONLY without importing backend.services (heavy import chain).
GOVERNANCE_SHADOW_ONLY = os.getenv("GOVERNANCE_SHADOW_ONLY", "false").lower() == "true"


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None or raw.strip() == "":
        return None
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def _core_only_mode() -> bool:
    v = _parse_bool(os.getenv("CORE_ONLY_MODE"))
    return v is True


CORE_ONLY_MODE = _core_only_mode()


def _flag(name: str, *, when_core: bool, when_prod: bool) -> bool:
    v = _parse_bool(os.getenv(name))
    if v is not None:
        return v
    if CORE_ONLY_MODE:
        return when_core
    return when_prod


# --- Required public flags (env names match) ---
ENABLE_GOVERNANCE_ENFORCEMENT = _flag(
    "ENABLE_GOVERNANCE_ENFORCEMENT",
    when_core=False,
    when_prod=not GOVERNANCE_SHADOW_ONLY,
)
ENABLE_COOLDOWN_ENFORCEMENT = _flag("ENABLE_COOLDOWN_ENFORCEMENT", when_core=False, when_prod=True)
ENABLE_ATR_ENFORCEMENT = _flag("ENABLE_ATR_ENFORCEMENT", when_core=False, when_prod=True)
ENABLE_PROFITABILITY_ENFORCEMENT = _flag("ENABLE_PROFITABILITY_ENFORCEMENT", when_core=False, when_prod=True)
ENABLE_REGIME_ENFORCEMENT = _flag("ENABLE_REGIME_ENFORCEMENT", when_core=False, when_prod=True)
ENABLE_QUARANTINE_ENFORCEMENT = _flag("ENABLE_QUARANTINE_ENFORCEMENT", when_core=False, when_prod=True)
ENABLE_SLEEVE_BLOCKING = _flag("ENABLE_SLEEVE_BLOCKING", when_core=False, when_prod=True)
ENABLE_TRADE_STATE_ENTRY_BLOCKING = _flag(
    "ENABLE_TRADE_STATE_ENTRY_BLOCKING",
    when_core=False,
    when_prod=True,
)
# Alternate paths (e.g. hot bridge): off in core; off by default in prod unless explicitly enabled elsewhere
ENABLE_ALTERNATE_SIGNAL_PATHS = _flag("ENABLE_ALTERNATE_SIGNAL_PATHS", when_core=False, when_prod=False)
# Hourly overflow sizing / HOURLY_OVERFLOW path after quality filters
ENABLE_OVERFLOW_SIZING = _flag("ENABLE_OVERFLOW_SIZING", when_core=False, when_prod=True)


def governance_risk_governor_shadow_only() -> bool:
    """
    RiskGovernor.shadow_only: True => no buy blocking from governor.
    Enforce only when ENABLE_GOVERNANCE_ENFORCEMENT and not GOVERNANCE_SHADOW_ONLY.
    """
    if not ENABLE_GOVERNANCE_ENFORCEMENT:
        return True
    return GOVERNANCE_SHADOW_ONLY


def local_bar_signal_grace_seconds() -> float:
    """
    Seconds to wait after a bar boundary before running process_bar_candidates.

    Local/staged stacks only: set PORTFOLIO_LOCAL_BAR_SIGNAL_GRACE_SEC (e.g. in
    deploy/core_only_local.env) so the signal consumer can attach ML-backed candidates in the same
    wall-clock cycle. Default 0 (production: no delay).
    """
    try:
        raw = os.getenv("PORTFOLIO_LOCAL_BAR_SIGNAL_GRACE_SEC", "0") or "0"
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def log_effective_flags_once() -> None:
    """One-shot summary for audits (idempotent)."""
    import logging

    log = logging.getLogger("core_test_flags")
    if getattr(log_effective_flags_once, "_done", False):
        return
    log_effective_flags_once._done = True  # type: ignore[attr-defined]
    log.warning(
        "CORE_TEST_FLAGS: CORE_ONLY_MODE=%s GOV=%s COOLDOWN=%s ATR=%s PROF=%s REGIME=%s QUAR=%s SLEEVE=%s TRADE_STATE=%s ALT_PATH=%s OVERFLOW=%s",
        CORE_ONLY_MODE,
        ENABLE_GOVERNANCE_ENFORCEMENT,
        ENABLE_COOLDOWN_ENFORCEMENT,
        ENABLE_ATR_ENFORCEMENT,
        ENABLE_PROFITABILITY_ENFORCEMENT,
        ENABLE_REGIME_ENFORCEMENT,
        ENABLE_QUARANTINE_ENFORCEMENT,
        ENABLE_SLEEVE_BLOCKING,
        ENABLE_TRADE_STATE_ENTRY_BLOCKING,
        ENABLE_ALTERNATE_SIGNAL_PATHS,
        ENABLE_OVERFLOW_SIZING,
    )
