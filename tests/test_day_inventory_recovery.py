"""Legacy inventory metadata helpers (no trading blocks)."""

from __future__ import annotations

from backend.services.day_inventory_recovery import (
    apply_legacy_tags_from_thesis,
    mark_new_regime_entry,
    thesis_json_for_position,
)


def test_legacy_tag_from_empty_thesis():
    class P:
        symbol = "BTC/USDT"
        legacy_pre_regime_router = False
        opened_under_router = False

    pos = P()
    apply_legacy_tags_from_thesis(pos, {})
    assert pos.legacy_pre_regime_router is True
    assert pos.opened_under_router is False


def test_new_regime_entry_tags():
    class P:
        legacy_pre_regime_router = False
        opened_under_router = False

    pos = P()
    mark_new_regime_entry(pos)
    assert pos.opened_under_router is True
    assert pos.legacy_pre_regime_router is False


def test_thesis_json_includes_legacy_flags():
    class P:
        entry_thesis = "HTF_TREND_PULLBACK"
        thesis_score = 0.7
        thesis_invalid_level = 0.0
        thesis_target_level = 0.0
        entry_vwap = 0.0
        thesis_trend_tf = ""
        day_route_regime_at_entry = ""
        legacy_pre_regime_router = True
        opened_under_router = False

    payload = thesis_json_for_position(P())
    assert payload["legacy_pre_regime_router"] is True
