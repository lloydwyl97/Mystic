"""Four-coin ranking uses microstructure as rank/size, never a hard block."""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.binance_scalp.scalp_candidate_ranking import HARD_REJECT_REASONS
from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size
from backend.services.binance_scalp.scalp_micro_replay import replay_four_coin_rank


def test_all_four_coins_remain_candidates():
    feats = {
        "BTCUSDT": {"ofi_5s": 1.0, "agg_flow_imbalance_5s": 0.1, "microprice_pressure": 0.0, "obi_l5": 0.1, "adverse_selection_score": 0.2, "spread_pct": 0.0001},
        "ETHUSDT": {"ofi_5s": 0.2, "agg_flow_imbalance_5s": 0.0, "microprice_pressure": 0.0, "obi_l5": 0.0, "adverse_selection_score": 0.3, "spread_pct": 0.0001},
        "SOLUSDT": {"ofi_5s": -0.5, "agg_flow_imbalance_5s": -0.1, "microprice_pressure": 0.0, "obi_l5": -0.1, "adverse_selection_score": 0.4, "spread_pct": 0.0002},
        "XRPUSDT": {"ofi_5s": -2.0, "agg_flow_imbalance_5s": -0.4, "microprice_pressure": -0.0002, "obi_l5": -0.3, "adverse_selection_score": 0.6, "spread_pct": 0.0003},
    }
    ranked = replay_four_coin_rank(feats)
    symbols = {s for s, _ in ranked}
    assert symbols == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert ranked[0][0] == "BTCUSDT"


def test_adverse_selection_lowers_rank_not_hard_block():
    good = {"ofi_5s": 2.0, "agg_flow_imbalance_5s": 0.3, "microprice_pressure": 0.0002, "obi_l5": 0.3, "adverse_selection_score": 0.05, "spread_pct": 0.0001}
    bad = dict(good)
    bad["adverse_selection_score"] = 0.9
    ranked = replay_four_coin_rank({"GOOD": good, "BAD": bad, "ETHUSDT": good, "XRPUSDT": bad})
    order = [s for s, _ in ranked]
    assert order.index("GOOD") < order.index("BAD")
    assert "ADVERSE_SELECTION" not in HARD_REJECT_REASONS


def test_positive_ofi_microprice_increases_rank():
    base = {"ofi_5s": 0.0, "agg_flow_imbalance_5s": 0.0, "microprice_pressure": 0.0, "obi_l5": 0.0, "adverse_selection_score": 0.2, "spread_pct": 0.0001}
    hot = dict(base)
    hot["ofi_5s"] = 6.0
    hot["microprice_pressure"] = 0.0005
    ranked = replay_four_coin_rank({"BASE": base, "HOT": hot, "ETHUSDT": base, "XRPUSDT": base})
    assert ranked[0][0] == "HOT"


def test_learning_does_not_become_eligibility():
    from backend.services.binance_scalp.scalp_micro_learning import micro_learning_adjustments

    adj = micro_learning_adjustments(":memory:", symbol="XRPUSDT", ofi_5s=-1.0, obi_l5=-0.4, adverse_selection_score=0.8)
    assert adj["eligibility"] is False
    assert "rank_delta" in adj
    assert adj["size_mult"] > 0


def test_micro_quality_sizes_inside_caps_not_to_zero():
    weak = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=1000.0,
        strategy_passed=True,
        micro_quality_mult=0.70,
    )
    strong = compute_scalp_position_size(
        base_cap=100.0,
        free_cash=1000.0,
        strategy_passed=True,
        micro_quality_mult=1.15,
    )
    assert weak.notional >= 5.0
    assert strong.notional <= 100.0
    assert strong.notional > weak.notional


def test_optional_venue_outage_does_not_disable_scalp():
    from backend.services.binance_scalp.scalp_cross_market import cross_market_features

    out = cross_market_features("SOLUSDT", own_mid=0.0)
    # Engine can still rank with remaining features.
    ranked = replay_four_coin_rank(
        {
            "BTCUSDT": {"ofi_5s": 1.0, "spread_pct": 0.0001, "adverse_selection_score": 0.1},
            "ETHUSDT": {"ofi_5s": 0.5, "spread_pct": 0.0001, "adverse_selection_score": 0.2},
            "SOLUSDT": {"ofi_5s": 0.2, "spread_pct": 0.0001, "adverse_selection_score": 0.2},
            "XRPUSDT": {"ofi_5s": 0.1, "spread_pct": 0.0001, "adverse_selection_score": 0.2},
        }
    )
    assert len(ranked) == 4
    assert out.get("eligibility") is None
