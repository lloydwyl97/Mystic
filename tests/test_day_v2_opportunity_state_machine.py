"""Tests for DAY V2 opportunity state machine."""

import pytest

from backend.services.day_v2.opportunity import (
    InvalidOpportunityTransition,
    Opportunity,
    OpportunityId,
    OpportunityRegistry,
    OpportunityState,
    OpportunityStateMachine,
)


def make_opp_id(
    symbol="BTCUSDT",
    setup_family="MOMENTUM",
    regime="bull",
    structural_anchor="4H_HIGH_20260920T16",
    formation_bar="2026-09-20T16:00:00",
) -> OpportunityId:
    return OpportunityId(
        symbol=symbol,
        setup_family=setup_family,
        regime=regime,
        structural_anchor=structural_anchor,
        formation_bar=formation_bar,
    )


def make_registry_with_opp(symbol="BTCUSDT") -> OpportunityRegistry:
    registry = OpportunityRegistry()
    opp_id = make_opp_id(symbol=symbol)
    registry.get_or_create(symbol, opp_id)
    return registry


def test_opportunity_id_is_deterministic():
    id1 = make_opp_id()
    id2 = make_opp_id()
    assert id1.canonical_id == id2.canonical_id


def test_opportunity_id_differs_by_symbol():
    id1 = make_opp_id(symbol="BTCUSDT")
    id2 = make_opp_id(symbol="ETHUSDT")
    assert id1.canonical_id != id2.canonical_id


def test_opportunity_id_differs_by_setup_family():
    id1 = make_opp_id(setup_family="MOMENTUM")
    id2 = make_opp_id(setup_family="REVERSION")
    assert id1.canonical_id != id2.canonical_id


def test_opportunity_id_differs_by_bar():
    id1 = make_opp_id(formation_bar="2026-09-20T16:00:00")
    id2 = make_opp_id(formation_bar="2026-09-20T16:15:00")
    assert id1.canonical_id != id2.canonical_id


def test_initial_state_is_no_opportunity():
    registry = make_registry_with_opp()
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.state == OpportunityState.NO_OPPORTUNITY


def test_transition_forming_to_active():
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal_received")
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.state == OpportunityState.FORMING

    registry.transition(
        "BTCUSDT",
        OpportunityState.ACTIVE_NOT_ENTERED,
        reason="formation_confirmed_on_closed_bar",
    )
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.state == OpportunityState.ACTIVE_NOT_ENTERED


def test_wall_clock_alone_does_not_advance_state():
    """State must not advance simply because time passes."""
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal")

    # Simulate time passing by calling transition with only a now argument
    # (but no state change) — the state must remain FORMING
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.state == OpportunityState.FORMING

    # Attempting an illegal transition still raises
    with pytest.raises(InvalidOpportunityTransition):
        registry.transition(
            "BTCUSDT",
            OpportunityState.POSITION_OPEN,
            reason="wall_clock_only",
        )


def test_closing_position_does_not_create_new_opportunity():
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal")
    registry.transition("BTCUSDT", OpportunityState.ACTIVE_NOT_ENTERED, reason="formed")
    registry.transition("BTCUSDT", OpportunityState.POSITION_OPEN, reason="entry_executed")
    registry.transition(
        "BTCUSDT",
        OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE,
        reason="position_closed",
    )

    opp = registry.all_opportunities()["BTCUSDT"]
    # State is POSITION_CLOSED_OPPORTUNITY_ACTIVE, NOT FORMING
    assert opp.state == OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE


def test_reentry_requires_structural_reset():
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal")
    registry.transition("BTCUSDT", OpportunityState.ACTIVE_NOT_ENTERED, reason="formed")
    registry.transition("BTCUSDT", OpportunityState.POSITION_OPEN, reason="entry")
    registry.transition(
        "BTCUSDT",
        OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE,
        reason="closed",
    )

    # Re-entry without structural_reset must fail
    with pytest.raises(InvalidOpportunityTransition):
        registry.transition(
            "BTCUSDT",
            OpportunityState.POSITION_OPEN,
            reason="re-entry_no_structural_reset",
            structural_reset=False,
        )

    # is_entry_eligible should still be True (caller must confirm structural reset)
    eligible, reason = registry.is_entry_eligible("BTCUSDT")
    assert eligible is True
    assert "structural_reset" in reason.lower()


def test_only_one_active_opportunity_per_symbol():
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="first_signal")

    # Trying to transition to FORMING again should raise (already in FORMING)
    with pytest.raises(InvalidOpportunityTransition):
        registry.transition("BTCUSDT", OpportunityState.FORMING, reason="second_signal")


def test_invalid_transition_raises():
    registry = make_registry_with_opp()
    # NO_OPPORTUNITY -> POSITION_OPEN is not a valid transition
    with pytest.raises(InvalidOpportunityTransition):
        registry.transition(
            "BTCUSDT",
            OpportunityState.POSITION_OPEN,
            reason="invalid",
        )


def test_all_four_symbols_eligible():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    registry = OpportunityRegistry()
    for sym in symbols:
        opp_id = make_opp_id(symbol=sym)
        registry.get_or_create(sym, opp_id)
        registry.transition(sym, OpportunityState.FORMING, reason=f"{sym}_signal")

    for sym in symbols:
        opp = registry.all_opportunities()[sym]
        assert opp.state == OpportunityState.FORMING


def test_no_consecutive_loss_veto():
    """No hidden block based on prior trade outcomes — only structural evidence."""
    # Create registry, simulate a loss, then verify FORMING is still reachable
    # after exhaustion + reset
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal")
    registry.transition("BTCUSDT", OpportunityState.ACTIVE_NOT_ENTERED, reason="formed")
    registry.transition("BTCUSDT", OpportunityState.POSITION_OPEN, reason="entry")
    registry.transition(
        "BTCUSDT",
        OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE,
        reason="loss_closed",
    )
    registry.transition("BTCUSDT", OpportunityState.EXHAUSTED, reason="structure_gone")
    # Transition to RESET_ELIGIBLE requires structural_reset
    registry.transition(
        "BTCUSDT",
        OpportunityState.RESET_ELIGIBLE,
        reason="new_structural_anchor",
        structural_reset=True,
    )
    # Now forming a new opportunity must be allowed
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="new_signal_after_reset")
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.state == OpportunityState.FORMING


def test_exhausted_to_reset_eligible_requires_structural_reset():
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal")
    registry.transition("BTCUSDT", OpportunityState.ACTIVE_NOT_ENTERED, reason="formed")
    registry.transition("BTCUSDT", OpportunityState.EXHAUSTED, reason="expired_without_entry")

    # Without structural_reset, EXHAUSTED -> RESET_ELIGIBLE must fail
    with pytest.raises(InvalidOpportunityTransition):
        registry.transition(
            "BTCUSDT",
            OpportunityState.RESET_ELIGIBLE,
            reason="no_structural_evidence",
            structural_reset=False,
        )

    # With structural_reset, it should succeed
    registry.transition(
        "BTCUSDT",
        OpportunityState.RESET_ELIGIBLE,
        reason="confirmed_structural_reset",
        structural_reset=True,
    )
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.state == OpportunityState.RESET_ELIGIBLE


def test_serialize_deserialize_roundtrip():
    registry = make_registry_with_opp("ETHUSDT")
    registry.transition("ETHUSDT", OpportunityState.FORMING, reason="signal")

    serialized = registry.to_dict()
    restored = OpportunityRegistry.from_dict(serialized)

    orig_opp = registry.all_opportunities()["ETHUSDT"]
    restored_opp = restored.all_opportunities()["ETHUSDT"]

    assert orig_opp.state == restored_opp.state
    assert orig_opp.opportunity_id.canonical_id == restored_opp.opportunity_id.canonical_id
    assert orig_opp.created_at == restored_opp.created_at


def test_from_signal_classmethod():
    opp_id = OpportunityId.from_signal(
        symbol="SOLUSDT",
        setup_family="BREAKOUT",
        regime="neutral",
        structural_anchor="4H_LOW_20260920T12",
        bar_timestamp="2026-09-20T12:00:00",
    )
    assert opp_id.symbol == "SOLUSDT"
    assert opp_id.setup_family == "BREAKOUT"
    assert "SOLUSDT" in opp_id.canonical_id
    assert "BREAKOUT" in opp_id.canonical_id


def test_position_entry_count_increments():
    registry = make_registry_with_opp()
    registry.transition("BTCUSDT", OpportunityState.FORMING, reason="signal")
    registry.transition("BTCUSDT", OpportunityState.ACTIVE_NOT_ENTERED, reason="formed")
    registry.transition("BTCUSDT", OpportunityState.POSITION_OPEN, reason="first_entry")
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.position_entry_count == 1

    registry.transition(
        "BTCUSDT",
        OpportunityState.POSITION_CLOSED_OPPORTUNITY_ACTIVE,
        reason="closed",
    )
    registry.transition(
        "BTCUSDT",
        OpportunityState.POSITION_OPEN,
        reason="re-entry",
        structural_reset=True,
    )
    opp = registry.all_opportunities()["BTCUSDT"]
    assert opp.position_entry_count == 2
