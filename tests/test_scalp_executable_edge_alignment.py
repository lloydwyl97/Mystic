"""SCALP edge alignment: one canonical executable net edge decides eligibility, rank and size."""

from __future__ import annotations

import inspect
import json
import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scalp_edge_contract import _bars, _ctx, _sig

import backend.services.adaptive_learning as al
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, canonical_roundtrip_cost_pct
from backend.services.binance_scalp import scalp_candidate_ranking as ranking
from backend.services.binance_scalp.scalp_candidate_ranking import HARD_REJECT_REASONS, rank_setup_signal
from backend.services.scalp_v2.decision_log import classify_scalp_candidate, record_scalp_decision
from backend.services.scalp_v2.executable_edge import decision_detail, scalp_executable_edge
from backend.services.scalp_v2.exit_evaluator import (
    SCALP_V2_EXIT_ADVERSE,
    evaluate_scalp_v2_exit,
    scalp_v2_adverse_net_threshold_pct,
    scalp_v2_max_adverse_net_pct,
)

# ETH trade 2740 (SCALP_V2, 2026-10-02): stamped learned edge after micro -0.8 bps,
# micro tilt -2.6 bps; the strategy's raw gross claim cleared the old ~40 bps hurdle.
ETH_2740_VIEW = {
    "expected_edge": -0.0000798,
    "micro_tilt": -0.000258,
    "edge_prior": 0.0015,
    "confidence": 0.62,
    "target_pct": 0.0031,
    "hold_min": 9.0,
    "risk_estimate": 0.000981,
    "size_mult": 0.6,
    "abstain": False,
    "abstain_reason": "",
}
ETH_2740_RAW_MOVE = 0.0045


def _patch_view(monkeypatch, **overrides):
    real = al.scalp_decision

    def fake(*args, **kwargs):
        view = real(*args, **kwargs)
        view.update(overrides)
        return view

    monkeypatch.setattr(al, "scalp_decision", fake)


def _row(rc) -> dict:
    return {
        "entry_eligible": rc.entry_eligible,
        "hard_block": rc.hard_block,
        "soft_reason": rc.soft_reason,
        "executable_edge": rc.executable_edge,
        "adaptive_regime": rc.adaptive_regime,
        "best_setup": rc.signal.setup_name,
        "rank_meta": {"expected_move_pct": rc.expected_move_pct},
    }


# ── one canonical function ────────────────────────────────────────────────────


def test_ranking_uses_the_one_canonical_edge_function():
    src = inspect.getsource(ranking)
    assert "scalp_executable_edge" in src
    assert "target_reachable(" not in src
    assert "entry_required_gross_edge_pct" not in src
    assert "TARGET_NOT_REACHABLE" not in HARD_REJECT_REASONS
    assert "NO_EXECUTABLE_NET_EDGE" in HARD_REJECT_REASONS


def test_ranked_edge_equals_canonical_function_output():
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    view = rc.adaptive_decision
    again = scalp_executable_edge(view, raw_expected_move_pct=rc.expected_move_pct, spread_pct=rc.executable_edge["spread_pct"], impact_pct=rc.executable_edge["impact_pct"])
    assert rc.net_edge_after_costs_pct == pytest.approx(again.edge_after_cost_pct)
    assert view["size_mult"] == pytest.approx(again.size_mult)
    assert view["executable_net_edge"] == pytest.approx(rc.net_edge_after_costs_pct)


def test_integration_loop_reads_the_router_edge_not_a_second_view():
    from backend.services import portfolio_engine_integration as pei

    src = inspect.getsource(pei)
    assert re.search(r"(?<!\w)scalp_decision\(", src) is None
    assert "detail=decision_detail(row" in src
    assert 'row.get("executable_edge")' in src


def test_canonical_edge_is_net_of_live_spread_and_impact():
    view = {"expected_edge": 0.0004, "micro_tilt": 0.0, "edge_prior": 0.0015}
    tight = scalp_executable_edge(view, raw_expected_move_pct=0.003, spread_pct=0.00002, impact_pct=0.0)
    wide = scalp_executable_edge(view, raw_expected_move_pct=0.003, spread_pct=0.0006, impact_pct=0.0002)
    assert tight.canonical_cost_pct == pytest.approx(canonical_roundtrip_cost_pct(spread_pct=0.00002))
    assert wide.canonical_cost_pct == pytest.approx(canonical_roundtrip_cost_pct(spread_pct=0.0006, buy_impact_pct=0.0002))
    assert wide.edge_after_cost_pct == pytest.approx(tight.edge_after_cost_pct - (wide.canonical_cost_pct - tight.canonical_cost_pct))
    assert wide.edge_after_cost_pct < tight.edge_after_cost_pct


# ── learned / micro edge reaches eligibility ──────────────────────────────────


def test_micro_tilt_moves_eligibility(monkeypatch):
    ctx = _ctx(_bars(30, 0.012))
    _patch_view(monkeypatch, expected_edge=0.0002 + 0.0001, micro_tilt=0.0001)
    up = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=ctx)
    assert up.entry_eligible
    assert up.executable_edge["micro_tilt_pct"] == pytest.approx(0.0001)
    _patch_view(monkeypatch, expected_edge=0.0002 - 0.0015, micro_tilt=-0.0015)
    down = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=ctx)
    assert down.hard_block == "NO_EXECUTABLE_NET_EDGE"
    assert not down.entry_eligible
    assert down.executable_edge["edge_before_micro_pct"] == pytest.approx(0.0002)


def test_eth_2740_negative_learned_edge_cannot_pass_on_the_old_gross_hurdle(monkeypatch):
    old_hurdle = 0.0025 + ESTIMATED_ROUNDTRIP_COST + 0.0005 + 0.0003
    assert old_hurdle < ETH_2740_RAW_MOVE
    _patch_view(monkeypatch, **ETH_2740_VIEW)
    sig = _sig(passed=True, reason=None, expected=ETH_2740_RAW_MOVE)
    rc = rank_setup_signal(sig, regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    assert rc.executable_edge["raw_expected_move_pct"] == pytest.approx(ETH_2740_RAW_MOVE)
    assert rc.executable_edge["edge_after_cost_pct"] < 0
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE"
    assert not rc.entry_eligible
    code, _ = classify_scalp_candidate(_row(rc))
    assert code == "REJECTED:NO_EXECUTABLE_NET_EDGE"


# ── negative net edge cannot fill; no legacy bypass ───────────────────────────


@pytest.mark.parametrize("edge", [-0.0004, 0.0])
def test_negative_or_zero_canonical_edge_cannot_arm_even_if_marked_eligible(edge):
    row = {"entry_eligible": True, "hard_block": None, "executable_edge": {"edge_after_cost_pct": edge}}
    assert classify_scalp_candidate(row)[0] == "REJECTED:NO_EXECUTABLE_NET_EDGE"


def test_eligible_row_without_canonical_edge_fails_closed():
    assert classify_scalp_candidate({"entry_eligible": True, "hard_block": None})[0] == "REJECTED:NO_EXECUTABLE_EDGE_ESTIMATE"


def test_strategy_target_not_reachable_is_not_a_bypass_or_a_veto(monkeypatch):
    ctx = _ctx(_bars(30, 0.012))
    sig = _sig(passed=False, reason="TARGET_NOT_REACHABLE", expected=0.001)
    _patch_view(monkeypatch, expected_edge=0.0006, micro_tilt=0.0)
    assert rank_setup_signal(sig, regime="RANGE", ctx=ctx).entry_eligible
    _patch_view(monkeypatch, expected_edge=-0.0006, micro_tilt=0.0)
    blocked = rank_setup_signal(sig, regime="RANGE", ctx=ctx)
    assert blocked.hard_block == "NO_EXECUTABLE_NET_EDGE"
    assert not blocked.entry_eligible


def test_edge_estimate_failure_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("adaptive store unavailable")

    monkeypatch.setattr(al, "scalp_decision", boom)
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    assert rc.hard_block == "NO_EXECUTABLE_EDGE_ESTIMATE"
    assert not rc.entry_eligible


def test_hard_safety_still_rejects_before_edge():
    from backend.services.scalp_v2.decision_log import classify_scalp_candidate as classify

    row = {"entry_eligible": False, "hard_block": "SPREAD_TOO_WIDE", "executable_edge": {"edge_after_cost_pct": 0.002}}
    assert classify(row)[0] == "REJECTED:SPREAD_TOO_WIDE"


# ── no sample-count gate; confidence moves size, not permission ───────────────


def test_cold_key_with_positive_edge_is_eligible_without_minimum_samples(tmp_path):
    db = str(tmp_path / "cold.db")
    view = al.scalp_decision(db, "BTCUSDT", "VWAP_RECLAIM_SCALP", "btcup_volhi", None)
    assert view["n_net"] == 0 and view["confidence"] == 0
    edge = scalp_executable_edge(view, raw_expected_move_pct=0.0, spread_pct=0.00002, impact_pct=0.0)
    assert edge.edge_after_cost_pct > 0
    assert edge.eligible


def test_confidence_changes_size_not_permission(tmp_path):
    db = str(tmp_path / "conf.db")
    version = al.current_strategy_version(al.SCALP_ENGINE)
    kw = {"engine": al.SCALP_ENGINE, "setup": "VWAP_RECLAIM_SCALP", "regime": "btcup_volhi", "metric": "trade_net", "strategy_version": version}
    assert al.observe(db, symbol="BTCUSDT", value=0.004, **kw)
    for _ in range(60):
        al.observe(db, symbol="ETHUSDT", value=0.004, **kw)
    thin = al.scalp_decision(db, "BTCUSDT", "VWAP_RECLAIM_SCALP", "btcup_volhi", None)
    deep = al.scalp_decision(db, "ETHUSDT", "VWAP_RECLAIM_SCALP", "btcup_volhi", None)
    assert deep["confidence"] > thin["confidence"]
    e_thin = scalp_executable_edge(thin, raw_expected_move_pct=0.0, spread_pct=0.00002, impact_pct=0.0)
    e_deep = scalp_executable_edge(deep, raw_expected_move_pct=0.0, spread_pct=0.00002, impact_pct=0.0)
    assert e_thin.eligible and e_deep.eligible
    assert e_deep.size_mult > e_thin.size_mult

    same = {"expected_edge": 0.0003, "micro_tilt": 0.0, "edge_prior": 0.0015}
    lo = scalp_executable_edge({**same, "confidence": 0.0}, raw_expected_move_pct=0.0, spread_pct=0.00002, impact_pct=0.0)
    hi = scalp_executable_edge({**same, "confidence": 0.99}, raw_expected_move_pct=0.0, spread_pct=0.00002, impact_pct=0.0)
    assert lo.eligible == hi.eligible is True
    assert lo.edge_after_cost_pct == hi.edge_after_cost_pct


# ── adverse stop reads adaptive risk in net units ─────────────────────────────


def _pos(risk: float | None) -> SimpleNamespace:
    adapt = {"target_pct": 0.004, "hold_min": 20.0}
    if risk is not None:
        adapt["risk_estimate"] = risk
    return SimpleNamespace(engine_id="SCALP_V2", cost_basis=100.0, entry_price=100.0, highest_price=100.0, lowest_price=99.9, adaptive_decision=adapt, symbol="ETHUSDT")


def _eval(risk, net):
    return evaluate_scalp_v2_exit(position=_pos(risk), current_price=100.0 * (1 + net + ESTIMATED_ROUNDTRIP_COST), net_pnl_pct=net, hold_minutes=2.0, bar_low=99.95, symbol="ETHUSDT")


def test_adverse_threshold_is_learned_mae_plus_cost_capped_at_contract():
    contract = scalp_v2_max_adverse_net_pct("ETHUSDT")
    tight = scalp_v2_adverse_net_threshold_pct("ETHUSDT", {"risk_estimate": 0.0003})
    assert tight == pytest.approx(0.0003 + ESTIMATED_ROUNDTRIP_COST)
    assert tight < contract
    assert scalp_v2_adverse_net_threshold_pct("ETHUSDT", {"risk_estimate": 0.01}) == contract
    assert scalp_v2_adverse_net_threshold_pct("ETHUSDT", {}) == contract
    assert scalp_v2_adverse_net_threshold_pct("ETHUSDT", None) == contract


def test_adverse_stop_uses_adaptive_risk_live():
    assert _eval(0.0003, -0.0009).get("reason") != SCALP_V2_EXIT_ADVERSE
    assert _eval(0.0003, -0.0010).get("reason") == SCALP_V2_EXIT_ADVERSE


def test_eth_2740_geometry_no_longer_stops_on_a_three_bp_gross_dip():
    # risk 9.81 bps gross used to fire at -9.81 bps net, i.e. a 3.2 bps gross dip.
    assert _eval(ETH_2740_VIEW["risk_estimate"], -0.0010).get("reason") != SCALP_V2_EXIT_ADVERSE
    assert _eval(ETH_2740_VIEW["risk_estimate"], -0.0016).get("reason") == SCALP_V2_EXIT_ADVERSE


# ── reject rows persist exact edge math ───────────────────────────────────────


def test_reject_rows_store_edge_deficit(monkeypatch, tmp_path):
    _patch_view(monkeypatch, **ETH_2740_VIEW)
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=ETH_2740_RAW_MOVE), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    row = _row(rc)
    code, reason = classify_scalp_candidate(row)
    db = tmp_path / "decisions.db"
    record_scalp_decision(db, "ETHUSDT", code, reason, cycle_ts=1.0, detail=decision_detail(row))
    with sqlite3.connect(db) as conn:
        (stored,) = conn.execute("SELECT detail FROM scalp_v2_decisions WHERE symbol='ETHUSDT'").fetchone()
    d = json.loads(stored)
    for key in (
        "raw_expected_move",
        "canonical_cost",
        "micro_tilt",
        "edge_before_micro",
        "edge_after_micro",
        "confidence",
        "target",
        "hold",
        "risk_estimate",
        "final_executable_net_edge",
        "reject_threshold",
        "reject_deficit",
        "edge_before_cost",
        "cost",
        "edge_after_cost",
        "deficit_to_zero",
    ):
        assert key in d, key
    assert d["edge_after_cost"] == pytest.approx(d["edge_before_cost"] - d["cost"], abs=1e-8)
    assert d["deficit_to_zero"] == pytest.approx(-d["edge_after_cost"], abs=1e-8)
    assert d["edge_after_micro"] == pytest.approx(ETH_2740_VIEW["expected_edge"], abs=1e-8)


def test_detail_is_never_empty_for_hard_blocks_before_edge():
    d = json.loads(decision_detail({"hard_block": "STALE_DATA", "rank_meta": {"expected_move_pct": 0.002}}))
    assert d["edge_computed"] is False and d["hard_block"] == "STALE_DATA"


# ── scope: DAY untouched, slots and coins intact ──────────────────────────────


def test_day_does_not_read_the_scalp_edge():
    import backend.services.day_v2.live_exit_evaluator as day_exit

    assert "executable_edge" not in inspect.getsource(day_exit)
    assert "scalp_executable_edge" not in inspect.getsource(al.day_decision)


def test_four_scalp_slots_and_four_coins():
    from backend.services import two_engine_claim
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.portfolio_engine import SCALP_MAX_OPEN_POSITIONS

    assert SCALP_MAX_OPEN_POSITIONS == 4
    assert two_engine_claim.engine_cap("SCALP_V2") == 4
    assert set(get_scalp_config().products) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
