from backend.services.day_entry_blocker_taxonomy import (
    classify_same_4h_thesis_slot_cap,
    classify_thesis_invalid_components,
)
from backend.services.day_trade_thesis import SETUP_HTF_TREND_PULLBACK, SETUP_VWAP_REVERSION, intact_4h_slot_blocked


def test_same_4h_slot_cap_is_concurrency_not_reentry():
    info = classify_same_4h_thesis_slot_cap()
    assert info["class"] == "concurrency"
    assert info["limits_open_economic_positions"] is True
    assert info["blocks_reentry_after_close"] is False
    assert intact_4h_slot_blocked(open_intact=2, candidate_intact=True) is True
    # After the slot is economically free (one name still open, one closed), re-entry is allowed.
    assert intact_4h_slot_blocked(open_intact=1, candidate_intact=True) is False


def test_deterministic_price_through_invalid_stays_hard():
    out = classify_thesis_invalid_components(
        setup=SETUP_VWAP_REVERSION,
        mark=1.2000,
        invalid_level=1.3588,
        bundle={},
        entry_price=1.3597,
    )
    assert "price_through_invalidation_level" in out["deterministic"]
    assert out["keep_as_hard_block"] is True


def test_subjective_ema_becomes_rank_delta_not_blind_admit():
    out = classify_thesis_invalid_components(
        setup=SETUP_HTF_TREND_PULLBACK,
        mark=2.50,
        invalid_level=2.10,
        bundle={"1h": {"ema_align": 0.20}, "4h": {"ema_align": 0.20}},
        entry_price=2.50,
    )
    assert "ema_alignment_1h_4h" in out["subjective"]
    assert out["keep_as_hard_block"] is False
    assert out["subjective_rank_delta"] < 0.0
