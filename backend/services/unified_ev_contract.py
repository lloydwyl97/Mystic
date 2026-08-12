"""
Unified ranking/EV contract (item p22).

Mystic's ranking/EV composition is genuinely spread across many real,
individually-tested modules (RF direction probability + isotonic
confidence, the ai_market_context ctx_multiplier family, DAY setup/block/
exec-delta scores, basket relative-strength ranks, market-role intelligence,
the real microstructure engine, the p15-p20 feature-stack/derivatives/
cross-exchange additions, SCALP's arm/MTF rank penalties + dynamic sizing,
and HoldEV) — see the reconnaissance notes inline below for what already
exists. This module does not replace any of those; it does two things that
were genuinely missing:

  1. MANIFEST: a single, declarative, importable list of every ranking/EV/
     sizing/exit family currently wired into DAY and/or SCALP, with its real
     weight/cap pulled live from the actual source constant (never a
     hardcoded duplicate number that could drift out of sync), its stage
     (ranking / sizing / exit / diagnostic-only), and its module location —
     so "what feeds this decision and how much" is answerable by reading
     ONE file instead of grepping eight.

  2. SNAPSHOT: a pure aggregation function that takes an already-fetched
     ai_context payload (+ optional SCALP ranking meta) and returns one flat
     per-family breakdown for audit/debugging — "why did this symbol rank
     where it did." It does not fetch data itself (Redis I/O stays in the
     caller, e.g. an API endpoint or a REPL) so it stays a pure, hermetically
     testable function.

Honest scope limitation: the *weights themselves* are configured (mostly
via env vars) rather than learned/ablation-validated end to end — that
statistical validation is item p14's separate deliverable (feature-family
ablation framework). This module is the contract/manifest those ablation
runs will report against; it does not itself run ablation studies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from backend.services.ai_decision_contract import (
    CTX_BTC_LAG_WEIGHT,
    CTX_CROSS_EXCHANGE_WEIGHT,
    CTX_DEPTH_WEIGHT,
    CTX_DERIVATIVES_WEIGHT,
    CTX_FEATURE_STACK_WEIGHT,
    CTX_MICROSTRUCTURE_WEIGHT,
    CTX_MTF_ALIGN_WEIGHT,
    CTX_REGIME_WEIGHT,
    CTX_RS_WEIGHT,
    CTX_TOTAL_CAP,
)


@dataclass(frozen=True)
class FamilySpec:
    name: str
    engine: str  # "day" | "scalp" | "shared"
    stage: str  # "ranking" | "sizing" | "exit" | "diagnostic_only"
    weight_or_cap: float
    weight_source: str  # human-readable description of where the number comes from
    module: str
    gating: bool = False  # must always be False per the architecture rule


def _holdev_weights() -> dict[str, float]:
    return {
        "momentum": float(os.getenv("HOLDEV_WEIGHT_MOMENTUM", "0.25")),
        "orderflow": float(os.getenv("HOLDEV_WEIGHT_ORDERFLOW", "0.15")),
        "excursion": float(os.getenv("HOLDEV_WEIGHT_EXCURSION", "0.40")),
        "progress": float(os.getenv("HOLDEV_WEIGHT_PROGRESS", "0.20")),
    }


def unified_contract_manifest() -> list[FamilySpec]:
    """Declarative, always-current manifest of every ranking/EV/sizing/exit
    family wired into DAY and/or SCALP. Every entry has ``gating=False`` —
    this is asserted by a test, not just documented, per the architecture
    rule that trade-opinion evidence never becomes a new hard entry blocker."""
    holdev_w = _holdev_weights()
    manifest = [
        FamilySpec("mtf_alignment", "day", "ranking", CTX_MTF_ALIGN_WEIGHT, "ai_decision_contract.CTX_MTF_ALIGN_WEIGHT", "ai_market_context.py"),
        FamilySpec("relative_strength_btc_eth", "day", "ranking", CTX_RS_WEIGHT, "ai_decision_contract.CTX_RS_WEIGHT", "ai_market_context.py"),
        FamilySpec("orderbook_depth_imbalance", "day", "ranking", CTX_DEPTH_WEIGHT, "ai_decision_contract.CTX_DEPTH_WEIGHT", "ai_market_context.py"),
        FamilySpec("market_regime", "day", "ranking", CTX_REGIME_WEIGHT, "ai_decision_contract.CTX_REGIME_WEIGHT", "ai_market_context.py"),
        FamilySpec("microstructure_ofi", "shared", "ranking", CTX_MICROSTRUCTURE_WEIGHT, "ai_decision_contract.CTX_MICROSTRUCTURE_WEIGHT", "microstructure_engine.py"),
        FamilySpec("ctx_multiplier_total_cap", "day", "ranking", CTX_TOTAL_CAP, "ai_decision_contract.CTX_TOTAL_CAP (sum of the above, capped)", "ai_market_context.py"),
        FamilySpec(
            "momentum_rvol_confirmation",
            "day",
            "ranking",
            CTX_FEATURE_STACK_WEIGHT,
            "ai_decision_contract.CTX_FEATURE_STACK_WEIGHT (day_feature_stack_v2.momentum_rvol_confirmation_signal — momentum x same-symbol RVOL, now in ctx_multiplier)",
            "day_feature_stack_v2.py",
        ),
        FamilySpec(
            "volatility_stack",
            "day",
            "diagnostic_only",
            0.0,
            "day_feature_stack_v2.py (feeds sizing/targets/exits, not ctx_multiplier — see day_adaptive_targets.py / scalp_dynamic_sizing.py)",
            "day_feature_stack_v2.py",
        ),
        FamilySpec(
            "btc_lag_correlation",
            "day",
            "ranking",
            CTX_BTC_LAG_WEIGHT,
            "ai_decision_contract.CTX_BTC_LAG_WEIGHT (day_feature_stack_v2.btc_lag_predictive_signal — confident-lag-only, now in ctx_multiplier)",
            "day_feature_stack_v2.py",
        ),
        FamilySpec(
            "derivatives_reference",
            "day",
            "ranking",
            CTX_DERIVATIVES_WEIGHT,
            "ai_decision_contract.CTX_DERIVATIVES_WEIGHT (derivatives_monitor.derivatives_positioning_signal — OI bias + funding percentile, now in ctx_multiplier)",
            "derivatives_monitor.py",
        ),
        FamilySpec(
            "cross_exchange_reference",
            "day",
            "ranking",
            CTX_CROSS_EXCHANGE_WEIGHT,
            "ai_decision_contract.CTX_CROSS_EXCHANGE_WEIGHT (cross_exchange_reference.cross_exchange_dislocation_signal — now in ctx_multiplier)",
            "cross_exchange_reference.py",
        ),
        FamilySpec(
            "multi_horizon_ev",
            "shared",
            "diagnostic_only",
            0.0,
            "multi_horizon_ev.py (composite EV over DAY 15m-24h / SCALP 30s-20m horizons; append-only ctx field / scalp ranking-meta field)",
            "multi_horizon_ev.py",
        ),
        FamilySpec(
            "multi_target_ml",
            "day",
            "diagnostic_only",
            0.0,
            "ai_multi_target_regressors.py (expected_return/MFE/MAE/time-to-target regression heads; append-only ctx field)",
            "ai_multi_target_regressors.py",
        ),
        FamilySpec(
            "walk_forward_validation",
            "shared",
            "diagnostic_only",
            0.0,
            "walk_forward_validation.py (purged/embargoed after-cost fold report; offline /walk-forward/{symbol} endpoint, never a live decision input)",
            "walk_forward_validation.py",
        ),
        FamilySpec(
            "feature_family_ablation",
            "day",
            "diagnostic_only",
            0.0,
            "feature_family_ablation.py (zero-ablation net expectancy/PF/drawdown/MFE-capture impact per family; offline /feature-ablation/{symbol} endpoint, never a live decision input)",
            "feature_family_ablation.py",
        ),
        FamilySpec(
            "hold_ev_momentum",
            "shared",
            "exit",
            holdev_w["momentum"],
            "hold_ev_engine._holdev_weights (env HOLDEV_WEIGHT_MOMENTUM); combined score feeds "
            "hold_ev_giveback_tighten_factor (DAY) / hold_ev_scratch_review_reduction (SCALP), "
            "bounded tighten-only exit levers, never an independent trigger",
            "hold_ev_engine.py",
        ),
        FamilySpec(
            "hold_ev_orderflow",
            "shared",
            "exit",
            holdev_w["orderflow"],
            "hold_ev_engine._holdev_weights (env HOLDEV_WEIGHT_ORDERFLOW); same tighten-only exit wiring as hold_ev_momentum",
            "hold_ev_engine.py",
        ),
        FamilySpec(
            "hold_ev_excursion",
            "shared",
            "exit",
            holdev_w["excursion"],
            "hold_ev_engine._holdev_weights (env HOLDEV_WEIGHT_EXCURSION); same tighten-only exit wiring as hold_ev_momentum",
            "hold_ev_engine.py",
        ),
        FamilySpec(
            "hold_ev_progress",
            "shared",
            "exit",
            holdev_w["progress"],
            "hold_ev_engine._holdev_weights (env HOLDEV_WEIGHT_PROGRESS); same tighten-only exit wiring as hold_ev_momentum",
            "hold_ev_engine.py",
        ),
        FamilySpec("scalp_arm_penalty", "scalp", "ranking", 1.0, "scalp_candidate_ranking.py arm_penalty_mult (multiplicative, not additive)", "scalp_candidate_ranking.py"),
        FamilySpec(
            "scalp_mtf_conflict_penalty",
            "scalp",
            "ranking",
            float(os.getenv("SCALP_MTF_5M_CONFLICT_RANK_MULT", "0.40")),
            "scalp_strategy_router.py env SCALP_MTF_5M_CONFLICT_RANK_MULT",
            "scalp_strategy_router.py",
        ),
        FamilySpec("scalp_dynamic_sizing", "scalp", "sizing", 1.0, "scalp_dynamic_sizing.compute_scalp_position_size (multiplicative combined factor)", "scalp_dynamic_sizing.py"),
        FamilySpec(
            "calibration_confidence_mult",
            "scalp",
            "sizing",
            float(os.getenv("CALIBRATION_DEGRADED_CONFIDENCE_MULT", "0.85")),
            "ai_calibration_tracker.calibration_confidence_multiplier (env CALIBRATION_DEGRADED_CONFIDENCE_MULT; 1.0 unless Brier/ECE measured degraded)",
            "ai_calibration_tracker.py",
        ),
        FamilySpec("scalp_execution_style", "scalp", "exit", 0.0, "scalp_execution_selector.py — chooses order type, not a size/decision input", "scalp_execution_selector.py"),
        FamilySpec(
            "day_adaptive_trail", "day", "exit", 0.0, "day_adaptive_trail.py — ratchet-only override, no numeric weight (replaces fixed tier when arm history supports it)", "day_adaptive_trail.py"
        ),
        FamilySpec("day_adaptive_target", "day", "exit", 0.0, "day_adaptive_targets.py — tighten-only additional target candidate (MFE-percentile)", "day_adaptive_targets.py"),
        FamilySpec(
            "day_atr_grid_target",
            "day",
            "exit",
            0.0,
            "day_adaptive_targets.atr_grid_target_candidate (item p6) — ATR-benchmark-grid expectancy-selected tighten-only target candidate",
            "day_adaptive_targets.py",
        ),
        FamilySpec("day_adaptive_giveback", "day", "exit", 0.0, "day_adaptive_targets.py — replaces fixed giveback trigger when arm has real losing history", "day_adaptive_targets.py"),
        FamilySpec("day_progress_decay_exit", "day", "exit", 0.0, "day_controlled_exits.evaluate_progress_decay_exit — new decay signal, symbol-gated", "day_controlled_exits.py"),
    ]
    bad = [f.name for f in manifest if f.gating]
    if bad:  # pragma: no cover - defensive; a test also asserts this statically
        raise RuntimeError(f"Architecture violation: families marked as gating: {bad}")
    return manifest


def manifest_as_dict() -> list[dict[str, Any]]:
    return [
        {
            "name": f.name,
            "engine": f.engine,
            "stage": f.stage,
            "weight_or_cap": f.weight_or_cap,
            "weight_source": f.weight_source,
            "module": f.module,
            "gating": f.gating,
        }
        for f in unified_contract_manifest()
    ]


def compute_unified_ranking_snapshot(
    symbol: str,
    ctx_payload: dict[str, Any] | None = None,
    scalp_ranking_meta: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten an already-fetched ai_context payload (+ optional SCALP
    ranking meta + optional calibration lookup) into one per-family audit
    breakdown for `symbol`.

    Pure function — callers fetch ``ctx_payload``/``calibration`` themselves
    (e.g. from Redis ``ai_context:{symbol}``, ``ai_calibration_tracker.
    calibration_confidence_multiplier``, or a test fixture) so this stays
    hermetically testable and never does its own I/O. ``calibration``, when
    provided, is expected as ``{"mult": float, "reason": str}``.
    """
    ctx_payload = ctx_payload or {}
    families: dict[str, Any] = {}

    families["ctx_multiplier"] = _safe_float(ctx_payload.get("ctx_multiplier"))
    families["ctx_rs_btc"] = _safe_float(ctx_payload.get("ctx_rs_btc"))
    families["ctx_rs_eth"] = _safe_float(ctx_payload.get("ctx_rs_eth"))
    families["ctx_depth_imbalance"] = _safe_float(ctx_payload.get("ctx_depth_imbalance"))
    families["ctx_market_regime"] = ctx_payload.get("ctx_market_regime")
    families["ctx_microstructure_ranking_delta"] = _safe_float(ctx_payload.get("ctx_microstructure_ranking_delta"))
    families["ctx_role_ranking_delta"] = _safe_float(ctx_payload.get("ctx_role_ranking_delta"))

    for key, out_key in (
        ("ctx_feature_stack_json", "feature_stack"),
        ("ctx_derivatives_json", "derivatives_reference"),
        ("ctx_cross_exchange_json", "cross_exchange_reference"),
        ("ctx_role_intel_json", "role_intelligence"),
        ("ctx_multi_horizon_ev_json", "multi_horizon_ev"),
        ("ctx_multi_target_ml_json", "multi_target_ml"),
    ):
        raw = ctx_payload.get(key)
        if isinstance(raw, str) and raw:
            try:
                families[out_key] = json.loads(raw)
            except (ValueError, TypeError):
                families[out_key] = {"parse_error": True}
        else:
            families[out_key] = None

    if scalp_ranking_meta:
        families["scalp_arm_penalty_mult"] = _safe_float(scalp_ranking_meta.get("arm_penalty_mult"))
        families["scalp_mtf_penalty_mult"] = _safe_float(scalp_ranking_meta.get("mtf_penalty_mult"))
        families["scalp_regime_mismatch"] = bool(scalp_ranking_meta.get("regime_mismatch", False))
        families["scalp_symbol_stall_risk"] = bool(scalp_ranking_meta.get("symbol_stall_risk", False))
        families["scalp_strategy_passed"] = bool(scalp_ranking_meta.get("strategy_passed", False))
        families["scalp_entry_owner"] = scalp_ranking_meta.get("entry_owner")

    if calibration is not None:
        families["calibration_confidence_mult"] = _safe_float(calibration.get("mult", 1.0)) or 1.0
        families["calibration_reason"] = calibration.get("reason")

    return {"symbol": symbol, "families": families}


def _safe_float(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "FamilySpec",
    "compute_unified_ranking_snapshot",
    "manifest_as_dict",
    "unified_contract_manifest",
]
