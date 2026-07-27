"""Versioned DAY gate registry — one owner and one behavior per gate.

Policy: day_aw_owner_v1 — AllWeather qualifies entries; ML ranks/sizes only.
Threshold ablations are frozen until shadow measurement evidence exists.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

Behavior = Literal["hard_block", "size", "rank", "telemetry", "exit"]
Layer = Literal["data_integrity", "strategy_signal", "portfolio_risk", "execution", "position_exits"]
Dependency = Literal["safety_critical", "strategy_critical", "optional", "telemetry"]
GateStatus = Literal["enabled", "disabled", "shadow_only"]

DECISION_POLICY_VERSION = "day_aw_owner_v1"
REGISTRY_VERSION = "1.1.0"
CONFIG_VERSION = os.getenv("DAY_CONFIG_VERSION", "day_cfg_v1")
MEASUREMENT_WINDOW_STARTED_UTC = "2026-07-26T00:00:00+00:00"
THRESHOLD_FREEZE_ACTIVE = True


def day_aw_owner_enabled() -> bool:
    """AllWeather is the sole DAY *entry qualifier* when True (default).

    DAY_AW_OWNER_ENABLED=true (default, production):
      - AllWeather must pass before a DAY buy can execute.
      - ML may only rank/size among AW-passed candidates.
      - Does NOT by itself enable/disable the portfolio engine loop.

    DAY_AW_OWNER_ENABLED=false (explicit rollback only):
      - Restores legacy fall-through where ML-enriched candidates may
        qualify without an AllWeather pass (requires operator intent).
      - AllWeather sleeve still follows ALLWEATHER_BREAKOUT_PULLBACK_ENABLED.
    """
    return os.getenv("DAY_AW_OWNER_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def day_ml_bypass_enabled() -> bool:
    """Legacy ML margin/EV/thesis bypass — default OFF. Rollback only; never leave on in prod."""
    return os.getenv("DAY_ML_BYPASS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def warn_or_fail_day_ml_bypass(*, fail_in_live: bool = True) -> None:
    """Surface DAY_ML_BYPASS_ENABLED loudly; fail-fast when live execution is allowed."""
    if not day_ml_bypass_enabled():
        return
    import logging

    log = logging.getLogger("backend.services.day_gate_registry")
    msg = (
        "DAY_ML_BYPASS_ENABLED=true — legacy ML margin/EV bypass is ACTIVE. "
        "This violates day_aw_owner_v1 and must not be used in production. "
        "Set DAY_ML_BYPASS_ENABLED=false immediately unless you are in a controlled rollback."
    )
    log.warning("DAY_ML_BYPASS_WARNING %s", msg)
    live = False
    try:
        from backend.services.execution_mode_service import is_live_execution_allowed_sync

        live = bool(is_live_execution_allowed_sync())
    except Exception:
        mode = (os.getenv("MYSTIC_TRADING_MODE") or os.getenv("TRADING_MODE") or "").strip().lower()
        live = mode == "live"
    if fail_in_live and live:
        raise RuntimeError(msg)


def allweather_qualifies_day_entries() -> bool:
    """True when DAY buys require an AllWeather pass (owner mode).

    Distinct from ``execution_enabled()`` on the sleeve adapter: owner mode
    controls *who may qualify entries*, not whether the portfolio loop runs.
    """
    return day_aw_owner_enabled()


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    layer: Layer
    behavior: Behavior
    dependency: Dependency
    purpose: str
    reason_code: str = ""
    threshold_refs: tuple[str, ...] = ()
    failure_fallback: str = "no_trade"
    status: GateStatus = "enabled"
    pending_ablation: bool = False

    def __post_init__(self) -> None:
        if not self.reason_code:
            object.__setattr__(self, "reason_code", self.gate_id)


DAY_GATES: dict[str, GateSpec] = {
    g.gate_id: g
    for g in (
        GateSpec(
            "AW_SETUP_PASS",
            "strategy_signal",
            "telemetry",
            "strategy_critical",
            "AllWeather setup evaluated and passed",
            reason_code="AW_SETUP_PASS",
        ),
        GateSpec(
            "AW_NO_SIGNAL",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "No AllWeather setup this bar",
            reason_code="AW_NO_SIGNAL",
            threshold_refs=("ALLWEATHER_BREAKOUT_RSI_MAX_TREND", "ALLWEATHER_BREAKOUT_RSI_MAX_NEUTRAL"),
            pending_ablation=True,
        ),
        GateSpec(
            "AW_EVAL_ERROR",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Kline/indicator evaluation failed — not a market no-trade",
            reason_code="AW_EVAL_ERROR",
        ),
        GateSpec(
            "AW_ROUTE_BLOCK",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "AllWeather production route/bucket mismatch",
            reason_code="AW_ROUTE_BLOCK",
        ),
        GateSpec(
            "THESIS_NO_CLEAR",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "New open requires a clear thesis / invalidation level",
            reason_code="THESIS_NO_CLEAR",
            threshold_refs=("DAY_REQUIRE_THESIS_FOR_ENTRY",),
        ),
        GateSpec(
            "NEGATIVE_EV",
            "portfolio_risk",
            "hard_block",
            "strategy_critical",
            "Selected candidate net EV <= 0 after costs model",
            reason_code="NEGATIVE_EV",
        ),
        GateSpec(
            "BUY_MARGIN_FLOOR",
            "portfolio_risk",
            "hard_block",
            "strategy_critical",
            "Bar buy_margin below sleeve/absolute floor",
            reason_code="BUY_MARGIN_FLOOR",
            threshold_refs=("BUY_MARGIN_THRESHOLD_ACTIVE", "BAR_EXEC_ABSOLUTE_MIN_BUY_MARGIN"),
            pending_ablation=True,
        ),
        GateSpec(
            "STALL_RISK_HARD",
            "portfolio_risk",
            "hard_block",
            "strategy_critical",
            "Severe stall-risk sizing multiplier collapsed to zero",
            reason_code="STALL_RISK_HARD",
            threshold_refs=("DAY_STALL_RISK_HARD_GATE_ENABLED",),
            pending_ablation=True,
        ),
        GateSpec(
            "KILL_SWITCH_BUY",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Kill switch blocks new entries",
            reason_code="KILL_SWITCH_BUY",
        ),
        GateSpec(
            "CASH_OR_SLOTS",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Insufficient cash, slots, or reserved exposure",
            reason_code="CASH_OR_SLOTS",
        ),
        GateSpec(
            "DUPLICATE_SYMBOL",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Symbol already open or reserved",
            reason_code="DUPLICATE_SYMBOL",
        ),
        GateSpec(
            "ORDERBOOK_PREFLIGHT",
            "execution",
            "hard_block",
            "safety_critical",
            "Spread/depth/impact fails protected preflight",
            reason_code="ORDERBOOK_PREFLIGHT",
        ),
        GateSpec(
            "ARTIFACT_CONTRACT",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Model artifact contract fail-closed",
            reason_code="ARTIFACT_CONTRACT",
        ),
        GateSpec(
            "ENTRY_CONTEXT",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Signal/context freshness fail-closed",
            reason_code="ENTRY_CONTEXT",
        ),
        GateSpec(
            "CLOSED_BAR",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Primary bar not closed or bar integrity failed",
            reason_code="CLOSED_BAR",
        ),
        GateSpec(
            "BAR_INTEGRITY",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Future/duplicate/OOO/missing/stale/wrong-interval candle",
            reason_code="BAR_INTEGRITY",
        ),
        GateSpec(
            "ML_RANK_SIZE",
            "strategy_signal",
            "rank",
            "optional",
            "ML buy_margin used only for ranking and size among AW-passed",
            reason_code="ML_RANK_SIZE",
            status="enabled",
        ),
        GateSpec(
            "SLEEVE_CAPACITY",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Strategy sleeve capacity exceeded including pending",
            reason_code="SLEEVE_CAPACITY",
        ),
    )
}


def get_gate(gate_id: str) -> GateSpec | None:
    return DAY_GATES.get(str(gate_id or "").strip().upper()) or DAY_GATES.get(str(gate_id or "").strip())


def registry_snapshot() -> dict[str, Any]:
    return {
        "registry_version": REGISTRY_VERSION,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "config_version": CONFIG_VERSION,
        "measurement_window_started_utc": MEASUREMENT_WINDOW_STARTED_UTC,
        "threshold_freeze_active": THRESHOLD_FREEZE_ACTIVE,
        "day_aw_owner_enabled": day_aw_owner_enabled(),
        "day_ml_bypass_enabled": day_ml_bypass_enabled(),
        "gates": {k: asdict(v) for k, v in DAY_GATES.items()},
        "notes": {
            "authority": "AllWeather qualifies DAY entries; ML ranks and sizes only",
            "threshold_freeze": "No RSI/strategy threshold changes until shadow ablation evidence",
            "rollback": "DAY_AW_OWNER_ENABLED=false restores legacy ML fallthrough; DAY_ML_BYPASS_ENABLED=true restores margin/EV bypasses",
            "phase_d_deferred": "Ablations wait for ≥N shadows + ≥M executes or 7d paper",
        },
    }


__all__ = [
    "CONFIG_VERSION",
    "DAY_GATES",
    "DECISION_POLICY_VERSION",
    "MEASUREMENT_WINDOW_STARTED_UTC",
    "REGISTRY_VERSION",
    "THRESHOLD_FREEZE_ACTIVE",
    "GateSpec",
    "allweather_qualifies_day_entries",
    "day_aw_owner_enabled",
    "day_ml_bypass_enabled",
    "get_gate",
    "registry_snapshot",
    "warn_or_fail_day_ml_bypass",
]
