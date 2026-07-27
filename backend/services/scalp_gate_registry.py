"""Versioned SCALP gate registry — strategy owns entry; ML/intel ranks only.

Policy: scalp_strategy_owner_v1 — genuine sig.passed qualifies; soft-rank never enters.
Threshold ablations are frozen until shadow measurement evidence exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Behavior = Literal["hard_block", "size", "rank", "telemetry", "exit"]
Layer = Literal["data_integrity", "strategy_signal", "portfolio_risk", "execution", "position_exits"]
Dependency = Literal["safety_critical", "strategy_critical", "optional", "telemetry"]

DECISION_POLICY_VERSION = "scalp_strategy_owner_v1"
REGISTRY_VERSION = "1.0.0"
MEASUREMENT_WINDOW_STARTED_UTC = "2026-07-26T00:00:00+00:00"
THRESHOLD_FREEZE_ACTIVE = True


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    layer: Layer
    behavior: Behavior
    dependency: Dependency
    purpose: str
    threshold_refs: tuple[str, ...] = ()
    failure_fallback: str = "no_trade"
    pending_ablation: bool = False


SCALP_GATES: dict[str, GateSpec] = {
    g.gate_id: g
    for g in (
        GateSpec(
            "STRATEGY_PASS",
            "strategy_signal",
            "telemetry",
            "strategy_critical",
            "Strategy pass_signal confirmed (genuine setup)",
        ),
        GateSpec(
            "STRATEGY_NO_SIGNAL",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "Strategy rejected setup (soft-rank may score for diagnostics only)",
            ("SCALP_MIN_TRADEABLE_SCORE",),
            pending_ablation=True,
        ),
        GateSpec(
            "SOFT_RANK_BLOCKED",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "Soft-rank promotion permanently refused — never executable",
        ),
        GateSpec(
            "RANK_BELOW_MIN",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "Genuine pass but rank_score below SCALP_MIN_TRADEABLE_SCORE",
            ("SCALP_MIN_TRADEABLE_SCORE",),
            pending_ablation=True,
        ),
        GateSpec(
            "REGIME_MISMATCH",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "Setup not regime-native when SCALP_REQUIRE_REGIME_NATIVE",
            ("SCALP_REQUIRE_REGIME_NATIVE",),
            pending_ablation=True,
        ),
        GateSpec(
            "MTF_NOT_ALIGNED",
            "strategy_signal",
            "hard_block",
            "strategy_critical",
            "5m/15m MTF confirmation failed",
            ("SCALP_MTF_CONFIRMATION_GATE",),
            pending_ablation=True,
        ),
        GateSpec(
            "NO_EXECUTABLE_NET_EDGE",
            "portfolio_risk",
            "hard_block",
            "strategy_critical",
            "Target unreachable after costs / no net edge",
            pending_ablation=True,
        ),
        GateSpec(
            "ENTRY_NOT_ARMED",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Redis entry arm required (unless auto-arm/calibration)",
        ),
        GateSpec(
            "SCALP_CIRCUIT_BREAKER",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Daily loss / consecutive-loss circuit open — blocks buys only",
        ),
        GateSpec(
            "CASH_OR_SLOTS",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Insufficient cash, max open positions, or reserved exposure",
            ("SCALP_MAX_OPEN_POSITIONS", "SCALP_MAX_NOTIONAL_PAPER"),
        ),
        GateSpec(
            "DUPLICATE_SYMBOL",
            "portfolio_risk",
            "hard_block",
            "safety_critical",
            "Symbol already open or reserved",
        ),
        GateSpec(
            "ORDERBOOK_PREFLIGHT",
            "execution",
            "hard_block",
            "safety_critical",
            "Spread/depth/impact / momentum preflight failed",
        ),
        GateSpec(
            "CLOSED_1M_BAR",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Primary 1m bar not closed — defer evaluation",
        ),
        GateSpec(
            "STALE_DATA",
            "data_integrity",
            "hard_block",
            "safety_critical",
            "Stale book or kline data",
        ),
        GateSpec(
            "SYMBOL_STALL_RISK",
            "portfolio_risk",
            "hard_block",
            "strategy_critical",
            "Symbol on stall-risk blocklist",
            pending_ablation=True,
        ),
        GateSpec(
            "ML_INTEL_RANK",
            "strategy_signal",
            "rank",
            "optional",
            "AI/intel adjusts rank_score among strategy-passed only",
        ),
    )
}


# Map common reject reason substrings → gate_id
REASON_TO_GATE: dict[str, str] = {
    "SOFT_RANK": "SOFT_RANK_BLOCKED",
    "RANKED_NOT_EXECUTABLE": "SOFT_RANK_BLOCKED",
    "WOULD_ENTER_NOT_ARMED": "ENTRY_NOT_ARMED",
    "SCALP_CIRCUIT_BREAKER": "SCALP_CIRCUIT_BREAKER",
    "INSUFFICIENT_CASH": "CASH_OR_SLOTS",
    "MAX_OPEN": "CASH_OR_SLOTS",
    "SYMBOL_ALREADY_OPEN": "DUPLICATE_SYMBOL",
    "ENTRY_RESERVED": "DUPLICATE_SYMBOL",
    "SPREAD_TOO_WIDE": "ORDERBOOK_PREFLIGHT",
    "DEPTH": "ORDERBOOK_PREFLIGHT",
    "IMPACT": "ORDERBOOK_PREFLIGHT",
    "MOMENTUM": "ORDERBOOK_PREFLIGHT",
    "STALE_DATA": "STALE_DATA",
    "REGIME_BLOCKED": "REGIME_MISMATCH",
    "REGIME_MISMATCH": "REGIME_MISMATCH",
    "MTF_5M": "MTF_NOT_ALIGNED",
    "MTF_15M": "MTF_NOT_ALIGNED",
    "NO_EXECUTABLE_NET_EDGE": "NO_EXECUTABLE_NET_EDGE",
    "TARGET_NOT_REACHABLE": "NO_EXECUTABLE_NET_EDGE",
    "RANK_BELOW_MIN": "RANK_BELOW_MIN",
    "BELOW_MIN": "RANK_BELOW_MIN",
    "SYMBOL_STALL_RISK": "SYMBOL_STALL_RISK",
    "CLOSED_1M": "CLOSED_1M_BAR",
    "NO_ENTRY_ELIGIBLE": "STRATEGY_NO_SIGNAL",
    "NOT_NEAR_SUPPORT": "STRATEGY_NO_SIGNAL",
    "NO_REJECTION_WICK": "STRATEGY_NO_SIGNAL",
}


def map_reason_to_gate(reason: str) -> str:
    u = str(reason or "").upper()
    for key, gid in REASON_TO_GATE.items():
        if key in u:
            return gid
    if u.startswith("RANK_BELOW"):
        return "RANK_BELOW_MIN"
    return "STRATEGY_NO_SIGNAL"


def get_gate(gate_id: str) -> GateSpec | None:
    return SCALP_GATES.get(str(gate_id or "").strip().upper()) or SCALP_GATES.get(str(gate_id or "").strip())


def registry_snapshot() -> dict[str, Any]:
    return {
        "registry_version": REGISTRY_VERSION,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "measurement_window_started_utc": MEASUREMENT_WINDOW_STARTED_UTC,
        "threshold_freeze_active": THRESHOLD_FREEZE_ACTIVE,
        "gates": {k: asdict(v) for k, v in SCALP_GATES.items()},
        "notes": {
            "authority": "Strategy pass_signal qualifies SCALP entries; ML/intel ranks only",
            "soft_rank": "Soft rejects may score for diagnostics but never enter",
            "threshold_freeze": "No strategy threshold changes until shadow ablation evidence",
            "phase_d_deferred": "Ablations wait for ≥N shadows + ≥M executes or 7d paper",
        },
    }


__all__ = [
    "DECISION_POLICY_VERSION",
    "SCALP_GATES",
    "GateSpec",
    "MEASUREMENT_WINDOW_STARTED_UTC",
    "REASON_TO_GATE",
    "REGISTRY_VERSION",
    "THRESHOLD_FREEZE_ACTIVE",
    "get_gate",
    "map_reason_to_gate",
    "registry_snapshot",
]
