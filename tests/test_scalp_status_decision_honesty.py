"""
Regression: SCALP status "decision" must not label ordinary no-candidate ticks
as BLOCKED. BLOCKED is reserved for genuine operational/safety preflight
failures (stale data, excessive spread/impact, insufficient depth, no
executable net edge, fee model unverified, paper disabled). An ordinary tick
with no ranked candidate must report NO_SIGNAL, so the dashboard and the
paper engine's own entry decision never disagree about whether trading is
actually blocked.
"""

from __future__ import annotations

from backend.services.binance_scalp.status_snapshot import (
    _is_genuine_safety_block,
    _overall_decision,
    _symbol_decision,
)


def test_no_candidate_no_reject_reason_is_no_signal_not_blocked():
    row = {"would_enter_if_armed": False, "would_arm_high_quality_near_pass": False, "distance_to_pass": {"distance_to_pass_pct": 999.0}}
    assert _symbol_decision(row) == "NO_SIGNAL"


def test_soft_rank_reject_reason_is_no_signal_not_blocked():
    row = {
        "would_enter_if_armed": False,
        "would_arm_high_quality_near_pass": False,
        "distance_to_pass": {"distance_to_pass_pct": 999.0},
        "reject_reason": "RANK_BELOW_MIN:1.2",
    }
    assert _symbol_decision(row) == "NO_SIGNAL"


def test_genuine_safety_reject_reason_is_blocked():
    for reason in ("SPREAD_TOO_WIDE", "PRICE_IMPACT_TOO_HIGH", "DEPTH_INSUFFICIENT", "NET_EDGE_BELOW_MIN", "ORDERBOOK_MISSING"):
        row = {
            "would_enter_if_armed": False,
            "would_arm_high_quality_near_pass": False,
            "distance_to_pass": {"distance_to_pass_pct": 999.0},
            "reject_reason": reason,
        }
        assert _symbol_decision(row) == "BLOCKED", f"{reason} must be a genuine block"
        assert _is_genuine_safety_block(reason) is True


def test_would_enter_if_armed_is_pass():
    row = {"would_enter_if_armed": True}
    assert _symbol_decision(row) == "PASS"


def test_would_arm_near_pass_is_ready_to_watch():
    row = {"would_enter_if_armed": False, "would_arm_high_quality_near_pass": True}
    assert _symbol_decision(row) == "READY_TO_WATCH"


def test_error_row_is_blocked():
    assert _symbol_decision({"error": "NO_MARKET_DATA"}) == "BLOCKED"


def test_overall_decision_prefers_no_signal_over_blocked_absence():
    rows = [
        {"would_enter_if_armed": False, "would_arm_high_quality_near_pass": False, "distance_to_pass": {"distance_to_pass_pct": 999.0}},
        {"would_enter_if_armed": False, "would_arm_high_quality_near_pass": False, "distance_to_pass": {"distance_to_pass_pct": 999.0}},
    ]
    assert _overall_decision(rows) == "NO_SIGNAL"


def test_overall_decision_surfaces_genuine_block_over_no_signal():
    rows = [
        {"would_enter_if_armed": False, "would_arm_high_quality_near_pass": False, "distance_to_pass": {"distance_to_pass_pct": 999.0}},
        {"would_enter_if_armed": False, "would_arm_high_quality_near_pass": False, "distance_to_pass": {"distance_to_pass_pct": 999.0}, "reject_reason": "SPREAD_TOO_WIDE"},
    ]
    assert _overall_decision(rows) == "BLOCKED"
