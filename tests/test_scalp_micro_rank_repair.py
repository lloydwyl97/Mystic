"""select_v2 ranking repair: EV_10s primary, no new gates, V1 still queryable."""

from __future__ import annotations

import inspect

from backend.services.binance_scalp.scalp_candidate_ranking import (
    HARD_REJECT_REASONS,
    pick_best_global_candidate,
    rank_setup_signal,
)
from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size
from backend.services.binance_scalp.scalp_micro_contract import (
    MICROSTRUCTURE_VERSION,
    MODEL_VERSION,
    SELECTION_VERSION,
    SELECTION_VERSION_V1,
    version_stamps,
)
from backend.services.binance_scalp.scalp_micro_ev import heuristic_horizon_ev
from backend.services.binance_scalp.scalp_micro_observability import build_peer_micro_snapshot
from backend.services.binance_scalp.scalp_micro_rank import (
    AUTHORITATIVE_SELECTION,
    EV_TIE_TOLERANCE,
    RANK_PRIMARY,
    TIEBREAK_SCALE,
    apply_repaired_rank,
    ev_scores_tied,
    repaired_primary_score,
)
from backend.services.binance_scalp.scalp_micro_replay import replay_four_coin_rank
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from types import SimpleNamespace


def _base(**overrides):
    feats = {
        "ofi_5s": 0.0,
        "agg_flow_imbalance_5s": 0.0,
        "microprice_pressure": 0.0,
        "obi_l5": 0.0,
        "adverse_selection_score": 0.2,
        "spread_pct": 0.0002,
        "bid_absorption_score": 0.0,
        "ask_absorption_score": 0.0,
        "depth_fragility": 0.1,
    }
    feats.update(overrides)
    return feats


def test_static_cannot_reverse_nontied_ev10s():
    weak = apply_repaired_rank(static_rank=2.6, feats=_base(), micro_ev={"EV_10s": -6.794e-05})[0]
    strong = apply_repaired_rank(static_rank=0.40, feats=_base(), micro_ev={"EV_10s": -6.54e-05})[0]
    assert strong > weak
    assert TIEBREAK_SCALE == 0.0
    assert not ev_scores_tied(-6.794e-05, -6.54e-05)


def test_exact_ev_tie_is_deterministic():
    a, _, ca = apply_repaired_rank(static_rank=1.28, feats=_base(), micro_ev={"EV_10s": -6.5e-05})
    b, _, cb = apply_repaired_rank(static_rank=0.98, feats=_base(), micro_ev={"EV_10s": -6.5e-05})
    assert a == b
    assert ev_scores_tied(a, b)
    assert ca["static_rank"] != cb["static_rank"]


def test_repaired_rank_orders_by_ev10s_not_static():
    weak_ev = _base(agg_flow_imbalance_5s=-0.6, ofi_5s=-2.0)
    strong_ev = _base(agg_flow_imbalance_5s=0.6, ofi_5s=2.0)
    low_static, _, _ = apply_repaired_rank(static_rank=2.4, feats=strong_ev)
    high_static, _, _ = apply_repaired_rank(static_rank=0.4, feats=weak_ev)
    assert low_static > high_static
    assert repaired_primary_score(strong_ev) > repaired_primary_score(weak_ev)


def test_repaired_score_is_deterministic():
    feats = _base(agg_flow_imbalance_5s=0.3, ofi_5s=1.0)
    a = apply_repaired_rank(static_rank=1.2, feats=feats)
    b = apply_repaired_rank(static_rank=1.2, feats=feats)
    assert a == b
    assert a[2]["selection_version"] == "scalp_micro_select_v2"
    assert a[2]["primary"] == RANK_PRIMARY == "EV_10s"


def test_all_four_symbols_remain_rankable():
    feats = {
        "BTCUSDT": _base(agg_flow_imbalance_5s=0.4),
        "ETHUSDT": _base(agg_flow_imbalance_5s=0.1),
        "SOLUSDT": _base(agg_flow_imbalance_5s=-0.1),
        "XRPUSDT": _base(agg_flow_imbalance_5s=-0.3),
    }
    ranked = replay_four_coin_rank(feats)
    assert {s for s, _ in ranked} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert ranked[0][0] == "BTCUSDT"
    assert ranked[-1][0] == "XRPUSDT"


def test_no_eligibility_or_permission_gate_from_rank():
    _, _, comps = apply_repaired_rank(static_rank=1.0, feats=_base(agg_flow_imbalance_5s=-0.9, ofi_5s=-4.0))
    assert comps["eligibility_effect"] is False
    assert "EV > 0" not in HARD_REJECT_REASONS
    assert "NEGATIVE_FLOW" not in HARD_REJECT_REASONS
    assert "OFI" not in HARD_REJECT_REASONS
    src = inspect.getsource(apply_repaired_rank)
    assert "entry_eligible" not in src
    assert "markout" not in src.lower()


def test_aggressive_flow_and_adverse_move_primary_score():
    pos_flow = repaired_primary_score(_base(agg_flow_imbalance_5s=0.5))
    neg_flow = repaired_primary_score(_base(agg_flow_imbalance_5s=-0.5))
    low_adv = repaired_primary_score(_base(adverse_selection_score=0.05))
    high_adv = repaired_primary_score(_base(adverse_selection_score=0.9))
    assert pos_flow > neg_flow
    assert low_adv > high_adv
    ev_src = inspect.getsource(heuristic_horizon_ev)
    assert "agg_flow_imbalance_5s" in ev_src
    assert "adverse_selection_score" in ev_src


def test_obi_absorption_fragility_standalone_neutralized():
    _, _, comps = apply_repaired_rank(static_rank=1.0, feats=_base(obi_l5=0.9, depth_fragility=0.9))
    assert comps["obi_standalone"] == "neutralized"
    assert comps["absorption_standalone"] == "neutralized"
    assert comps["fragility_standalone"] == "neutralized"
    src = inspect.getsource(apply_repaired_rank)
    assert "obi_l5" not in src
    assert "absorption" not in src
    assert "fragility" not in src


def test_no_forward_data_in_rank_formula():
    src = inspect.getsource(repaired_primary_score) + inspect.getsource(apply_repaired_rank)
    for banned in ("executable_markout", "fwd_mid", "realized_pnl", "mfe_bp", "mae_bp"):
        assert banned not in src
    assert "heuristic_horizon_ev" in src or "EV_10s" in src


def test_model_and_version_stamps():
    stamps = version_stamps()
    assert stamps["selection_version"] == SELECTION_VERSION == "scalp_micro_select_v2"
    assert stamps["microstructure_version"] == MICROSTRUCTURE_VERSION == "scalp_micro_v1"
    assert stamps["model_version"] == MODEL_VERSION == "scalp_micro_ev_v1"
    assert SELECTION_VERSION_V1 == "scalp_micro_select_v1"
    assert SELECTION_VERSION != SELECTION_VERSION_V1


def test_old_v1_baseline_remains_queryable():
    assert MICROSTRUCTURE_VERSION == "scalp_micro_v1"
    assert SELECTION_VERSION_V1 == "scalp_micro_select_v1"
    from backend.services.binance_scalp.scalp_micro_contract import feature_context_extra

    extra = feature_context_extra({"ofi_5s": 0.2, "symbol": "BTCUSDT"})
    assert extra["microstructure_version"] == "scalp_micro_v1"
    assert extra["selection_version"] == "scalp_micro_select_v2"


def test_peer_observability_preserves_four_coin_ev_and_components():
    rows = []
    for i, sym in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")):
        ev = -0.00010 + i * 0.00002
        score, _, comps = apply_repaired_rank(static_rank=1.0, feats=_base(), micro_ev={"EV_10s": ev})
        rows.append(
            {
                "symbol": sym,
                "rank_score": score,
                "entry_eligible": True,
                "strategy_passed": False,
                "EV_1s": ev * 0.4,
                "EV_5s": ev * 0.8,
                "EV_10s": ev,
                "EV_30s": ev * 0.9,
                "EV_60s": ev * 0.7,
                "rank_components": comps,
                "selection_version": "scalp_micro_select_v2",
            }
        )
    snap = build_peer_micro_snapshot(rows, selected_symbol="XRPUSDT", max_open=4, open_count=0)
    assert set(snap["peers"]) >= {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        rec = snap["peers"][sym]
        assert rec["EV_10s"] is not None
        assert rec["rank_position"] >= 1
        assert rec["rank_components"]["selection_version"] == "scalp_micro_select_v2"
    assert snap["selected_symbol"] == "XRPUSDT"
    assert snap["peers"]["XRPUSDT"]["rank_position"] == 1
    assert snap["selected_ev_position"] == 1


def test_sizing_formula_unchanged_by_rank_repair():
    kwargs = dict(
        base_cap=50.0,
        free_cash=1000.0,
        min_notional=5.0,
        strategy_passed=False,
        micro_quality_mult=1.0,
        calibration_mult=1.0,
        spread_pct=0.0002,
        impact_pct=0.0001,
    )
    a = compute_scalp_position_size(**kwargs)
    b = compute_scalp_position_size(**kwargs)
    assert a.notional == b.notional
    assert a.combined_multiplier == b.combined_multiplier
    size_src = inspect.getsource(compute_scalp_position_size)
    assert "apply_repaired_rank" not in size_src
    assert "repaired_primary_score" not in size_src


def test_xrp_ofi_depth_norm_still_comparable():
    from backend.services import microstructure_engine as m
    import time

    m._STATE.clear()
    t0 = time.time()

    def _book(bid_px, bid_sz, ask_px, ask_sz, depth=20, spacing=0.01):
        bids = [(bid_px - i * spacing, bid_sz) for i in range(depth)]
        asks = [(ask_px + i * spacing, ask_sz) for i in range(depth)]
        return bids, asks

    btc1, a1 = _book(80000.0, 1.0, 80001.0, 1.0)
    btc2, a2 = _book(80000.0, 1.6, 80001.0, 0.4)
    m.record_snapshot("BTCUSDT", btc1, a1, ts=t0)
    m.record_snapshot("BTCUSDT", btc2, a2, ts=t0 + 0.1)
    x1, xa1 = _book(1.50, 5000.0, 1.51, 5000.0)
    x2, xa2 = _book(1.50, 8000.0, 1.51, 2000.0)
    m.record_snapshot("XRPUSDT", x1, xa1, ts=t0)
    m.record_snapshot("XRPUSDT", x2, xa2, ts=t0 + 0.1)
    btc = m.compute_features("BTCUSDT")["ofi_1s"]
    xrp = m.compute_features("XRPUSDT")["ofi_1s"]
    m._STATE.clear()
    assert btc > 0 and xrp > 0
    assert abs(btc - xrp) / max(abs(btc), abs(xrp)) < 0.15


def test_pick_best_uses_repaired_rank_among_hold_survivors(monkeypatch):
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda row: None)
    def _row(sym, rank, ev_move):
        return {
            "symbol": sym,
            "rank_score": rank,
            "entry_eligible": True,
            "signal": SimpleNamespace(
                passed=True,
                spread_pct=0.0002,
                expected_move_pct=ev_move,
                impact_pct=0.0,
                confidence=0.7,
            ),
        }

    rows = [
        _row("BTCUSDT", -0.00040, 0.004),
        _row("ETHUSDT", -0.00010, 0.004),
        _row("SOLUSDT", -0.00025, 0.004),
        _row("XRPUSDT", -0.00050, 0.004),
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "ETHUSDT"


def test_pick_best_selects_select_v2_leader_even_when_path_net_hold_disagrees(monkeypatch):
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda row: None)

    def _row(sym, rank, expected):
        return {
            "symbol": sym,
            "rank_score": rank,
            "entry_eligible": True,
            "signal": SimpleNamespace(
                passed=False,
                spread_pct=0.0002,
                expected_move_pct=expected,
                impact_pct=0.0,
                confidence=0.4,
            ),
        }

    rows = [
        _row("ETHUSDT", 2.393e-05, 0.0),
        _row("BTCUSDT", -4.729e-05, 0.0),
        _row("SOLUSDT", -0.00011764, 0.0),
        _row("XRPUSDT", -0.00011823, 0.004),
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "ETHUSDT"
    assert float(best.get("expected_net_ev") or 0) <= 0


def test_pick_best_second_slot_is_next_select_v2_leader(monkeypatch):
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda row: None)
    rows = [
        {"symbol": "ETHUSDT", "rank_score": 3e-05, "entry_eligible": True, "already_open": True, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
        {"symbol": "BTCUSDT", "rank_score": -1e-05, "entry_eligible": True, "already_open": False, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
        {"symbol": "SOLUSDT", "rank_score": -5e-05, "entry_eligible": True, "already_open": False, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
        {"symbol": "XRPUSDT", "rank_score": -9e-05, "entry_eligible": True, "already_open": False, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "BTCUSDT"


def test_pick_best_skips_hard_block_and_already_open(monkeypatch):
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda row: None)
    rows = [
        {"symbol": "ETHUSDT", "rank_score": 1e-04, "entry_eligible": False, "hard_block": "SPREAD_TOO_WIDE", "signal": SimpleNamespace(passed=True, spread_pct=0.02, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
        {"symbol": "BTCUSDT", "rank_score": 5e-05, "entry_eligible": True, "already_open": True, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
        {"symbol": "SOLUSDT", "rank_score": -2e-05, "entry_eligible": True, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
        {"symbol": "XRPUSDT", "rank_score": -8e-05, "entry_eligible": True, "signal": SimpleNamespace(passed=True, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.7)},
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "SOLUSDT"


def test_negative_ev_leader_still_selected(monkeypatch):
    import backend.services.binance_scalp.forward_net_predictor as fnp

    monkeypatch.setattr(fnp, "predict_row_expected_net", lambda row: None)
    rows = [
        {"symbol": "XRPUSDT", "rank_score": -6.112e-05, "entry_eligible": True, "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.0, impact_pct=0.0, confidence=0.4)},
        {"symbol": "BTCUSDT", "rank_score": -8.004e-05, "entry_eligible": True, "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.4)},
        {"symbol": "ETHUSDT", "rank_score": -9.706e-05, "entry_eligible": True, "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.4)},
        {"symbol": "SOLUSDT", "rank_score": -0.00011865, "entry_eligible": True, "signal": SimpleNamespace(passed=False, spread_pct=0.0002, expected_move_pct=0.004, impact_pct=0.0, confidence=0.4)},
    ]
    best = pick_best_global_candidate(rows)
    assert best is not None
    assert best["symbol"] == "XRPUSDT"


def test_hard_safety_and_universe_unchanged():
    assert "SPREAD_TOO_WIDE" in HARD_REJECT_REASONS
    src = inspect.getsource(pick_best_global_candidate)
    assert "expected_net_ev" not in src
    assert "HOLD_ACTION_EV" not in src
    assert inspect.getsource(compute_scalp_position_size).count("apply_repaired_rank") == 0


def test_rank_setup_signal_stamps_select_v2(monkeypatch):
    monkeypatch.setattr(
        "backend.services.microstructure_engine.compute_features",
        lambda symbol: _base(agg_flow_imbalance_5s=0.4, ofi_5s=1.5),
    )
    ctx = StrategyMarketContext(
        symbol="BTCUSDT",
        snap=SimpleNamespace(symbol="BTCUSDT", spread_pct=0.0002, best_ask=100.0, best_bid=99.98, mid=100.0, asks=[[100.0, 1000.0]]),
        mom=SimpleNamespace(mid_change_15s=0.0, mid_change_30s=0.0, bid_change_15s=0.0, momentum_confirmed=False),
        bars_1m=[{"low": 99.5, "high": 100.5, "close": 100.0}] * 15,
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        notional_usd=25.0,
    )
    sig = ScalpSetupSignal(
        symbol="BTCUSDT",
        side="BUY",
        score=2.6,
        setup_name="range_bounce_scalp",
        confidence=0.7,
        entry_reason="support_bounce",
        invalidation_reason="support_break",
        required_target_pct=0.0025,
        expected_move_pct=0.0035,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=True,
        reject_reason=None,
    )
    ranked = rank_setup_signal(sig, regime="range", ctx=ctx)
    assert ranked.entry_eligible is True
    assert ranked.rank_components["selection_version"] == "scalp_micro_select_v2"
    assert ranked.rank_components["primary"] == "EV_10s"
    assert ranked.rank_components["authoritative_selection"] is True
    assert abs(ranked.rank_score - ranked.rank_components["EV_10s"]) < 1e-3


def test_negative_ev_and_negative_flow_still_eligible(monkeypatch):
    monkeypatch.setattr(
        "backend.services.microstructure_engine.compute_features",
        lambda symbol: _base(agg_flow_imbalance_5s=-0.8, ofi_5s=-3.0, adverse_selection_score=0.8),
    )
    ctx = StrategyMarketContext(
        symbol="ETHUSDT",
        snap=SimpleNamespace(symbol="ETHUSDT", spread_pct=0.0002, best_ask=100.0, best_bid=99.98, mid=100.0, asks=[[100.0, 1000.0]]),
        mom=SimpleNamespace(mid_change_15s=0.0, mid_change_30s=0.0, bid_change_15s=0.0, momentum_confirmed=False),
        bars_1m=[{"low": 99.5, "high": 100.5, "close": 100.0}] * 15,
        econ=ScalpEconomics.from_env(),
        config=get_scalp_config(),
        notional_usd=25.0,
    )
    sig = ScalpSetupSignal(
        symbol="ETHUSDT",
        side="BUY",
        score=0.0,
        setup_name="range_bounce_scalp",
        confidence=0.4,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=0.006,
        spread_pct=0.0002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.0,
        passed=False,
        reject_reason="NOT_NEAR_SUPPORT",
    )
    ranked = rank_setup_signal(sig, regime="range", ctx=ctx)
    assert ranked.entry_eligible is True
    assert ranked.hard_block is None
    assert ranked.rank_components["EV_10s"] < 0
    assert TIEBREAK_SCALE == 0.0
    assert AUTHORITATIVE_SELECTION is True
    assert EV_TIE_TOLERANCE <= 1e-12


def test_day_exits_universe_and_max_open_untouched():
    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.exit_manager import _path_max_adverse_net_pct
    from backend.services.microstructure_engine import get_microstructure_ranking_delta
    import backend.services.binance_scalp.paper_engine as pe

    day_src = inspect.getsource(get_microstructure_ranking_delta)
    assert "apply_repaired_rank" not in day_src
    assert "scalp_micro_select_v2" not in day_src
    assert "authoritative_selection" not in day_src
    exit_src = inspect.getsource(_path_max_adverse_net_pct)
    assert 'SCALP_PATH_MAX_ADVERSE_NET_PCT", "0.0015"' in exit_src
    paper_src = inspect.getsource(pe)
    assert "max_open_positions" in paper_src
    assert "SCALP_MAX_OPEN_POSITIONS" in inspect.getsource(ScalpConfig.from_env)
    products = get_scalp_config().products
    assert set(products) >= {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
