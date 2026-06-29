"""
Unified replay promotion gate — research only.

No live promotion unless execution replay passes all gates.
Label-proxy / train-only metrics never satisfy promotion.
"""

from __future__ import annotations

from typing import Any

TARGET_MONTHLY_USD = 500.0
PRINCIPAL = 25_000.0
MAX_HOLD_HOURS_DAY = 72.0
MAX_HOLD_HOURS_SCALP = 0.5


def _metric(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    nested = metrics.get("metrics_90d") or {}
    for key in keys:
        if key in nested and nested[key] is not None:
            return float(nested[key])
    return default


def evaluate_day_promotion(
    metrics: dict[str, Any],
    *,
    stress_pass: bool = False,
    walk_forward_test_pass: bool = False,
    walk_forward_val_pass: bool = False,
    execution_replay_verified: bool = False,
    label_proxy_only: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if label_proxy_only:
        reasons.append("label_proxy_not_execution_replay")
    if not execution_replay_verified:
        reasons.append("execution_replay_not_verified")
    monthly = _metric(metrics, "monthly_pnl_usd_on_25k", "monthly_pnl_usd")
    if monthly < TARGET_MONTHLY_USD:
        reasons.append("below_500_mo")
    exp = _metric(metrics, "expectancy_per_trade_usd", "expectancy_per_trade")
    if exp <= 0:
        reasons.append("negative_expectancy")
    if not walk_forward_val_pass:
        reasons.append("walk_forward_val_fail")
    if not walk_forward_test_pass:
        reasons.append("walk_forward_test_fail")
    if not stress_pass:
        reasons.append("stress_fail")
    hold = _metric(metrics, "longest_hold_hours", "longest_hold_hours_90d")
    if hold > MAX_HOLD_HOURS_DAY:
        reasons.append("fat_tail_hold")
    dd = _metric(metrics, "max_drawdown_pct")
    if dd > 15.0:
        reasons.append("drawdown_limit")
    return len(reasons) == 0, reasons


def evaluate_scalp_promotion(metrics: dict[str, Any], *, execution_replay_verified: bool = False) -> tuple[bool, list[str]]:
    """Scalp never contributes to profit expectation unless execution replay + 30m hold."""
    reasons: list[str] = ["scalp_isolated_from_profit_expectation"]
    if not execution_replay_verified:
        reasons.append("execution_replay_not_verified")
    hold_h = float(metrics.get("longest_hold_hours") or metrics.get("longest_hold_sec", 0) / 3600.0 or 0)
    if hold_h > MAX_HOLD_HOURS_SCALP:
        reasons.append("scalp_hold_exceeds_30m")
    if float(metrics.get("monthly_pnl_usd_on_25k") or metrics.get("monthly_pnl_usd") or 0) < TARGET_MONTHLY_USD:
        reasons.append("below_500_mo")
    return False, reasons  # scalp never auto-promotes until explicitly re-enabled


RESEARCH_EXHAUSTED = frozenset(
    {
        "ltf_hand_coded_patterns",
        "controlled_risk_bracket_exits",
        "frequency_first_regime_remap",
        "ai_label_proxy_buckets",
        "universe_expansion_without_execution_replay",
        "scalp_profit_contribution",
    }
)

RESEARCH_EXCLUDED_SYMBOLS = frozenset(
    {
        "BNB/USD",
        "BNB/USDT",
        "BNB/USDC",
    }
)


def block_exhausted_branch(branch_name: str) -> None:
    import os
    import sys

    if os.environ.get("MYSTIC_FORCE_EXHAUSTED_RESEARCH") != "1":
        print(
            f"{branch_name} exhausted. Active path: python3 scripts/run_topfour_profit_rebuild.py",
            file=sys.stderr,
        )
        sys.exit(2)
