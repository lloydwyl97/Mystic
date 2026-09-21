from unittest.mock import patch

from backend.services.day_grouped_decision_ledger import CandidateRow, DecisionGroup
from backend.services.day_production_lifecycle_replay import ClosedTrade
from backend.services.day_profit_attribution import (
    champion_picker,
    hold_label,
    isolated_eligible,
    label_groups,
    oracle_picker,
    portfolio_replay_from_policy,
    simulate_isolated,
    summarize_attribution,
)


def _row(**kwargs):
    base = {
        "decision_group_id": "dg1",
        "ts_utc": "2026-09-01T12:00:00+00:00",
        "epoch": 1_788_000_000,
        "bar_epoch": 1_788_000_000,
        "symbol": "ETHUSDT",
        "selected_action": "HOLD",
        "live_selected": False,
        "path_ev": 0.0004,
        "hold_ev": 0.0,
        "p_buy": 0.3,
        "p_hold": 0.7,
        "p_sell": 0.0,
        "feature_schema_version": "5",
        "feature_dim": 145,
        "features": None,
        "model_version": "day_path_net_v1",
        "data_source": "redis_orderbook",
        "freshness_sec": 1.0,
        "best_bid": 100.0,
        "best_ask": 100.1,
        "mid": 100.05,
        "spread_bps": 10.0,
        "predicted_impact_bps": 1.0,
        "expected_slippage_bps": 1.4,
        "expected_commission_bps": 4.0,
        "slot_occupancy": 0,
        "cash": 10000.0,
        "symbol_already_open": False,
        "max_open": False,
        "proposed_notional": 70.0,
        "outcome_class": "ranking_loser",
        "eligibility_reason": "valid_unselected_alternative",
        "field_authority": {},
        "extras": {},
    }
    base.update(kwargs)
    return CandidateRow(**base)


def test_hold_is_zero_and_safety_not_eligible():
    hold = hold_label("dg1")
    assert hold.net_bps == 0.0
    assert hold.net_usd == 0.0
    assert hold.capital_hours == 0.0
    assert isolated_eligible(_row(symbol="HOLD", outcome_class="hold")) is True
    assert isolated_eligible(_row(outcome_class="stale_invalid")) is False
    assert isolated_eligible(_row(outcome_class="symbol_open")) is False
    assert isolated_eligible(_row(outcome_class="ranking_loser")) is True


def test_ranking_loser_is_counterfactual_not_fill():
    closed = ClosedTrade(
        symbol="ETHUSDT",
        entry_epoch=1_788_000_000,
        exit_epoch=1_788_003_600,
        entry_price=100.1,
        exit_price=100.0,
        notional=70.0,
        gross_pct=-0.001,
        commission_pct=0.0004,
        spread_pct=0.00007,
        slippage_pct=0.00014,
        net_pct=-0.00161,
        net_usd=-0.11,
        exit_reason="DAY_4H_STRUCTURE_BREAK_EXIT",
        mfe_pct=0.002,
        mae_pct=-0.003,
        p_buy=0.3,
        setup="HTF_TREND_PULLBACK",
        unlocked_band=False,
        time_to_mfe_sec=120.0,
        hold_sec=3600.0,
    )
    bars = {"ETHUSDT": [(1_788_000_000, 100.0, 100.2, 99.9, 100.05, 1.0)]}
    fourh = {"ETHUSDT": []}
    with (
        patch("backend.services.day_profit_attribution._advance_position", return_value=closed),
        patch("backend.services.day_profit_attribution.get_coin_profile", return_value={"sl": 0.01, "tp": 0.02, "trail": 0.004, "max_hold_min": 360}),
    ):
        lab = simulate_isolated(_row(), bars, fourh)
    assert lab.labeled is True
    assert lab.label_kind == "counterfactual_isolated"
    assert lab.fill_status == "counterfactual"
    assert lab.outcome_class == "ranking_loser"


def test_oracle_picks_best_eligible_and_champion_keeps_production():
    hold = _row(symbol="HOLD", outcome_class="hold", live_selected=True, selected_action="HOLD", path_ev=0.0)
    eth = _row(symbol="ETHUSDT", outcome_class="ranking_loser", live_selected=False)
    btc = _row(symbol="BTCUSDT", outcome_class="ranking_loser", live_selected=False, path_ev=0.0001)
    sol = _row(symbol="SOLUSDT", outcome_class="stale_invalid", live_selected=False)
    xrp = _row(symbol="XRPUSDT", outcome_class="ranking_loser", live_selected=False)
    group = DecisionGroup(
        decision_group_id="dg1",
        ts_utc=eth.ts_utc,
        epoch=eth.epoch,
        bar_epoch=eth.bar_epoch,
        selected_action="HOLD",
        selected_symbol="",
        path_ev_winner="HOLD",
        why_selected="HOLD_WINS",
        model_version="day_path_net_v1",
        candidates=[btc, eth, sol, xrp, hold],
    )

    def _closed(net):
        return ClosedTrade(
            symbol="ETHUSDT",
            entry_epoch=eth.epoch,
            exit_epoch=eth.epoch + 3600,
            entry_price=100.0,
            exit_price=100.0,
            notional=70.0,
            gross_pct=net / 1e4,
            commission_pct=0.0,
            spread_pct=0.0,
            slippage_pct=0.0,
            net_pct=net / 1e4,
            net_usd=net / 1e4 * 70.0,
            exit_reason="MAX_HOLD",
            mfe_pct=abs(net / 1e4) + 0.001,
            mae_pct=-0.001,
            p_buy=0.3,
            setup="HTF_TREND_PULLBACK",
            unlocked_band=False,
            time_to_mfe_sec=60.0,
            hold_sec=3600.0,
        )

    def fake_advance(pos, *_a, **_k):
        nets = {"ETHUSDT": 12.0, "BTCUSDT": 3.0, "XRPUSDT": -5.0}
        return _closed(nets[pos.symbol])

    bars = {s: [(eth.epoch, 100.0, 101.0, 99.0, 100.0, 1.0)] for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}
    with (
        patch("backend.services.day_profit_attribution._advance_position", side_effect=fake_advance),
        patch("backend.services.day_profit_attribution.get_coin_profile", return_value={"sl": 0.01, "tp": 0.02, "trail": 0.004, "max_hold_min": 360}),
        patch("backend.services.day_profit_attribution.resample_4h", return_value=[]),
    ):
        attrs = label_groups([group], bars)
    assert attrs[0].best_eligible_symbol == "ETHUSDT"
    assert attrs[0].opportunity is True
    assert champion_picker(group, attrs[0], set()) == "HOLD"
    assert oracle_picker(group, attrs[0], set()) == "ETHUSDT"
    summary = summarize_attribution(attrs)
    assert summary["hindsight_positive_paths"] is True
    port = portfolio_replay_from_policy([group], attrs, picker=oracle_picker)
    assert port["trades"] == 1
    assert port["net_usd"] > 0
    from backend.services.day_profit_attribution import exclusive_group_waterfall

    water = exclusive_group_waterfall(attrs)
    assert water["sum"] == 1
    assert water["mutually_exclusive"] is True


def test_null_oracle_does_not_claim_predictable_edge():
    from backend.services.day_profit_attribution import FROZEN_LOCKED, null_oracle_distribution

    assert FROZEN_LOCKED["promote"] is None
    hold = _row(symbol="HOLD", outcome_class="hold", live_selected=True, selected_action="HOLD", path_ev=0.0)
    eth = _row(symbol="ETHUSDT", outcome_class="ranking_loser")
    group = DecisionGroup(
        decision_group_id="dg1",
        ts_utc=eth.ts_utc,
        epoch=eth.epoch,
        bar_epoch=eth.bar_epoch,
        selected_action="HOLD",
        selected_symbol="",
        path_ev_winner="HOLD",
        why_selected="HOLD_WINS",
        model_version="day_path_net_v1",
        candidates=[eth, hold],
    )
    closed = ClosedTrade(
        symbol="ETHUSDT",
        entry_epoch=eth.epoch,
        exit_epoch=eth.epoch + 60,
        entry_price=100.0,
        exit_price=101.0,
        notional=70.0,
        gross_pct=0.01,
        commission_pct=0.0,
        spread_pct=0.0,
        slippage_pct=0.0,
        net_pct=0.01,
        net_usd=0.7,
        exit_reason="MAX_HOLD",
        mfe_pct=0.012,
        mae_pct=-0.001,
        p_buy=0.3,
        setup="HTF_TREND_PULLBACK",
        unlocked_band=False,
        time_to_mfe_sec=10.0,
        hold_sec=60.0,
    )
    bars = {"ETHUSDT": [(eth.epoch, 100.0, 101.0, 99.0, 100.0, 1.0)], "BTCUSDT": [], "SOLUSDT": [], "XRPUSDT": []}
    with (
        patch("backend.services.day_profit_attribution._advance_position", return_value=closed),
        patch("backend.services.day_profit_attribution.get_coin_profile", return_value={"sl": 0.01, "tp": 0.02, "trail": 0.004, "max_hold_min": 360}),
        patch("backend.services.day_profit_attribution.resample_4h", return_value=[]),
    ):
        attrs = label_groups([group], bars)
    null = null_oracle_distribution([group], attrs, n_perm=8, block=1)
    assert "predictable" not in (null.get("predecision_incremental_information") or "")
    assert null["observed_oracle"]["trades"] >= 0
