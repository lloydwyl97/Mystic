"""Tests for DAY V2 engine identity and authority boundaries."""

import pytest

from backend.services.day_v2.engine_identity import (
    _AUTHORITY_TABLE,
    LIVE_ENGINE_IDS,
    SHADOW_ENGINE_IDS,
    AuthorityBoundary,
    AuthorityLevel,
    EngineId,
    assert_no_live_authority,
    create_authority_boundary,
    get_authority,
    has_live_authority,
    is_shadow_eligible,
)


def test_legacy_day_live_has_live_authority():
    assert has_live_authority(EngineId.LEGACY_DAY_LIVE) is True


def test_scalp_v2_candidate_has_no_live_authority():
    assert has_live_authority(EngineId.SCALP_V2_CANDIDATE) is False


def test_day_v2_shadow_has_no_live_authority():
    assert has_live_authority(EngineId.DAY_V2_SHADOW) is False


def test_assert_no_live_authority_raises_for_legacy():
    with pytest.raises(PermissionError):
        assert_no_live_authority(EngineId.LEGACY_DAY_LIVE)


def test_assert_no_live_authority_passes_for_candidates():
    # Must not raise
    assert_no_live_authority(EngineId.SCALP_V2_CANDIDATE)
    assert_no_live_authority(EngineId.DAY_V2_SHADOW)


def test_day_v2_live_and_legacy_in_live_engine_ids():
    """DAY_V2_LIVE was promoted to LIVE on 2026-09-21 after qualifying replay."""
    assert EngineId.LEGACY_DAY_LIVE in LIVE_ENGINE_IDS
    assert EngineId.DAY_V2_LIVE in LIVE_ENGINE_IDS


def test_scalp_v2_live_has_live_authority():
    """SCALP_V2_LIVE promoted to LIVE on 2026-09-22 after 24-trade paper proof."""
    assert has_live_authority(EngineId.SCALP_V2_LIVE) is True


def test_scalp_v2_live_in_live_engine_ids():
    """SCALP_V2_LIVE must appear in LIVE_ENGINE_IDS alongside DAY and LEGACY."""
    assert EngineId.SCALP_V2_LIVE in LIVE_ENGINE_IDS


def test_new_scalp_entry_uses_scalp_v2_engine_id():
    """String value stored on paper_trades.engine_id for SCALP V2 entries is 'SCALP_V2'."""
    assert EngineId.SCALP_V2_LIVE.value == "SCALP_V2"


def test_new_day_entry_uses_day_v2_engine_id():
    """String value stored on day_trailing_buy_intents.engine_id for DAY V2 is 'DAY_V2'."""
    assert EngineId.DAY_V2_LIVE.value == "DAY_V2"


def test_legacy_day_live_is_exit_only_not_in_shadow():
    """LEGACY_DAY_LIVE has LIVE authority (for exit management) but is NOT a shadow engine.

    No new entries must ever be created with engine_id='LEGACY_DAY_LIVE'.
    The authority is kept LIVE only so that accounting and exit management
    correctly identify these positions as live (exit-only).
    """
    assert has_live_authority(EngineId.LEGACY_DAY_LIVE) is True
    assert EngineId.LEGACY_DAY_LIVE not in SHADOW_ENGINE_IDS


def test_assert_no_live_authority_raises_for_scalp_v2_live():
    """SCALP_V2_LIVE must be blocked from shadow/research code paths."""
    with pytest.raises(PermissionError):
        assert_no_live_authority(EngineId.SCALP_V2_LIVE)


def test_candidates_in_shadow_engine_ids():
    assert EngineId.SCALP_V2_CANDIDATE in SHADOW_ENGINE_IDS
    assert EngineId.DAY_V2_SHADOW in SHADOW_ENGINE_IDS


def test_authority_table_is_exhaustive():
    """Every EngineId value must have an entry in the authority table."""
    for engine_id in EngineId:
        assert engine_id in _AUTHORITY_TABLE, f"{engine_id!r} missing from _AUTHORITY_TABLE"


def test_create_authority_boundary_raises_for_live_engine():
    with pytest.raises(PermissionError):
        create_authority_boundary(EngineId.LEGACY_DAY_LIVE, "test_caller")


def test_create_authority_boundary_succeeds_for_shadow():
    boundary = create_authority_boundary(EngineId.SCALP_V2_CANDIDATE, "unit_test")
    assert isinstance(boundary, AuthorityBoundary)
    assert boundary.engine_id == EngineId.SCALP_V2_CANDIDATE
    assert boundary.caller_context == "unit_test"
    assert boundary.checked_at > 0


def test_get_authority_returns_correct_levels():
    assert get_authority(EngineId.LEGACY_DAY_LIVE) == AuthorityLevel.LIVE
    assert get_authority(EngineId.SCALP_V2_CANDIDATE) == AuthorityLevel.SHADOW
    assert get_authority(EngineId.DAY_V2_SHADOW) == AuthorityLevel.SHADOW


def test_is_shadow_eligible():
    assert is_shadow_eligible(EngineId.SCALP_V2_CANDIDATE) is True
    assert is_shadow_eligible(EngineId.DAY_V2_SHADOW) is True
    assert is_shadow_eligible(EngineId.LEGACY_DAY_LIVE) is False


def test_live_engine_ids_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        LIVE_ENGINE_IDS.add(EngineId.DAY_V2_SHADOW)  # type: ignore[attr-defined]


def test_shadow_engine_ids_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        SHADOW_ENGINE_IDS.add(EngineId.LEGACY_DAY_LIVE)  # type: ignore[attr-defined]
