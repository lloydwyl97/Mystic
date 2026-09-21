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


def test_only_legacy_in_live_engine_ids():
    assert {EngineId.LEGACY_DAY_LIVE} == LIVE_ENGINE_IDS


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
