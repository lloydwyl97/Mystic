"""SCALP canonical executable edge: the current candidate's calibrated move minus live
cost, plus bounded learned residuals. The raw projection is a calibration input only."""

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
import backend.services.scalp_v2.executable_edge as ee
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, canonical_roundtrip_cost_pct
from backend.services.binance_scalp import scalp_candidate_ranking as ranking
from backend.services.binance_scalp.scalp_candidate_ranking import HARD_REJECT_REASONS, rank_setup_signal
from backend.services.scalp_v2.decision_log import classify_scalp_candidate, record_scalp_decision
from backend.services.scalp_v2.edge_state_rebuild import rebuild_scalp_edge_state
from backend.services.scalp_v2.executable_edge import decision_detail, scalp_executable_edge
from backend.services.scalp_v2.exit_evaluator import (
    SCALP_V2_EXIT_ADVERSE,
    evaluate_scalp_v2_exit,
    scalp_v2_adverse_net_threshold_pct,
    scalp_v2_max_adverse_net_pct,
)

SPREAD = 0.00002
KEY = {"symbol": "ETHUSDT", "setup": "VWAP_EMA_RECLAIM", "regime": "btcup_volhi"}


def _cold_view(tmp_path, name="cold.db"):
    return al.scalp_decision(str(tmp_path / name), KEY["symbol"], KEY["setup"], KEY["regime"], None)


def _edge(view, raw, spread=SPREAD, impact=0.0, edge_source="STRATEGY_CLAIM"):
    return scalp_executable_edge(view, raw_expected_move_pct=raw, spread_pct=spread, impact_pct=impact, edge_source=edge_source)


def _observe(db, metric, value, n=1, symbol=KEY["symbol"], version=None):
    for _ in range(n):
        assert al.observe(
            db,
            engine=al.SCALP_ENGINE,
            symbol=symbol,
            setup=KEY["setup"],
            regime=KEY["regime"],
            metric=metric,
            value=value,
            strategy_version=version or al.current_strategy_version(al.SCALP_ENGINE),
        )


def _claims(db, *, raw, gross, n, symbol=KEY["symbol"]):
    for _ in range(n):
        assert al.learn_claim_label(db, symbol=symbol, setup=KEY["setup"], regime=KEY["regime"], strategy_version=al.current_strategy_version(al.SCALP_ENGINE), raw=raw, gross=gross)


def _view(setup_gross, key_gross=None, **extra):
    """A decision view with a learned claim calibration (capture 0 unless given)."""
    return {"claim_gross_setup": setup_gross, "claim_gross_mean": setup_gross if key_gross is None else key_gross, "micro_residual": 0.0, **extra}


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


# ── cold prior is neutral; no sample gate ─────────────────────────────────────


def test_cold_adaptive_residual_is_zero(tmp_path):
    view = _cold_view(tmp_path)
    assert al._PRIORS[al.SCALP_ENGINE]["edge_residual"] == 0.0
    assert view["adaptive_residual"] == 0.0
    assert view["n_residual"] == 0
    assert view["confidence"] == 0.0
    assert "expected_edge" not in view


@pytest.mark.parametrize("raw", [0.0, 0.0001, 0.0012, 0.006])
def test_cold_claim_is_priced_at_minus_cost_whatever_its_geometry(tmp_path, raw):
    view = _cold_view(tmp_path)
    cost = canonical_roundtrip_cost_pct(spread_pct=SPREAD)
    edge = _edge(view, raw)
    assert edge.n_residual == 0
    assert edge.expected_move_pct == 0.0
    assert edge.final_executable_edge_pct == pytest.approx(-cost)
    assert not edge.eligible


def test_post_deploy_fill_geometry_raw_below_cost_is_negative_when_cold(tmp_path):
    view = _cold_view(tmp_path)
    for raw in (0.00052, 0.00061):
        edge = _edge(view, raw, spread=0.00006)
        assert edge.base_executable_edge_pct < 0
        assert edge.final_executable_edge_pct == pytest.approx(edge.base_executable_edge_pct)
        assert not edge.eligible


def test_claim_below_cost_recovers_through_realized_gross_evidence(tmp_path):
    db = str(tmp_path / "evidence.db")
    cost = canonical_roundtrip_cost_pct(spread_pct=SPREAD)
    raw = cost - 0.0002
    cold = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    assert not _edge(cold, raw).eligible
    _claims(db, raw=raw, gross=3.0 * cost, n=12)
    learned = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    assert learned["claim_gross_setup"] > cost
    up = _edge(learned, raw)
    assert up.raw_expected_move_pct < cost < up.calibrated_move_pct
    assert up.final_executable_edge_pct > 0 and up.eligible


def test_negative_gross_evidence_prices_a_large_projection_below_cost(tmp_path):
    db = str(tmp_path / "neg.db")
    raw = canonical_roundtrip_cost_pct(spread_pct=SPREAD) + 0.0004
    _claims(db, raw=raw, gross=-0.0015, n=12)
    view = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    edge = _edge(view, raw)
    assert edge.raw_expected_move_pct > edge.live_cost_pct
    assert edge.calibrated_move_pct < 0
    assert edge.final_executable_edge_pct < 0
    assert not edge.eligible


def test_key_residual_is_bounded_by_the_residual_cap():
    assert al.SCALP_RESIDUAL_MAX == 0.006
    assert al.scalp_expected_gross(_view(0.0, -0.05), 0.002)["adaptive"] == pytest.approx(-0.006)
    assert al.scalp_expected_gross(_view(0.0, 0.05), 0.002)["adaptive"] == pytest.approx(0.006)


def test_atr_rows_never_reach_the_claim_calibration(tmp_path):
    db = str(tmp_path / "source.db")
    _observe(db, "edge_residual", 0.005, n=60)
    view = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    assert view["adaptive_residual"] > 0.004
    claim = _edge(view, 0.003)
    assert claim.raw_move_source == "STRATEGY_CLAIM"
    assert claim.expected_move_pct == 0.0 and view["n_claim"] == 0
    _claims(db, raw=0.003, gross=-0.005, n=60)
    view = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    assert _edge(view, 0.003).expected_move_pct < -0.004
    assert al.residual_metric("STRATEGY_CLAIM") == al.residual_metric("strategy") == "edge_residual_strategy"
    assert al.residual_metric("ATR_ESTIMATE") == al.residual_metric("atr_estimate") == "edge_residual"
    assert {"edge_residual", "edge_residual_strategy", *al.CLAIM_MOMENT_METRICS} <= al.MEAN_FORM_METRICS


@pytest.mark.parametrize("source", ["ATR_ESTIMATE", "atr_estimate", "NONE", "unavailable", ""])
def test_atr_or_missing_source_cannot_be_a_directional_raw_edge(tmp_path, source):
    with pytest.raises(ValueError):
        _edge(_cold_view(tmp_path), 0.004, edge_source=source)


def test_atr_markouts_do_not_train_the_micro_residual(tmp_path):
    db = str(tmp_path / "micro.db")
    t0 = 1_790_000_000.0
    for i, src in enumerate(("ATR_ESTIMATE", "NONE")):
        al.record_candidate(
            db,
            engine=al.SCALP_ENGINE,
            symbol=KEY["symbol"],
            setup=KEY["setup"],
            regime=KEY["regime"],
            ref_price=100.0,
            roundtrip_cost=ESTIMATED_ROUNDTRIP_COST,
            signaled=False,
            evaluated_at=t0 + i * 60,
            features={"obi_l5": 0.2, "spread_pct": 0.00004},
            raw_expected_move=0.0008 if src != "NONE" else None,
            raw_move_source=src,
        )
    al.resolve_markouts(db, lambda _s, _t: 100.2, now=t0 + 5000, path_low=lambda _s, _a, _b: 99.9)
    with sqlite3.connect(db) as conn:
        sources = {r[0] for r in conn.execute("SELECT raw_move_source FROM adaptive_candidate_markouts")}
        micro = conn.execute("SELECT COUNT(*) FROM adaptive_linear_model").fetchone()[0] if conn.execute("SELECT name FROM sqlite_master WHERE name='adaptive_linear_model'").fetchone() else 0
        strategy_n = conn.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric='edge_residual_strategy'").fetchone()[0]
    assert sources == {"ATR_ESTIMATE", "NONE"}
    assert micro == 0
    assert strategy_n == 0


def test_residual_is_a_running_mean_not_a_fast_ewma(tmp_path):
    db = str(tmp_path / "mean.db")
    for v in (0.002, 0.002, 0.002, -0.004):
        _observe(db, "edge_residual", v)
    with sqlite3.connect(db) as conn:
        (ewma,) = conn.execute("SELECT ewma FROM adaptive_metric_state WHERE metric='edge_residual'").fetchone()
    assert ewma == pytest.approx(0.0005, abs=1e-6)


def test_confidence_changes_size_not_permission():
    base = _view(0.004, risk_estimate=0.0015)
    lo = _edge({**base, "confidence": 0.0}, 0.0012)
    hi = _edge({**base, "confidence": 0.95}, 0.0012)
    assert lo.eligible and hi.eligible
    assert lo.final_executable_edge_pct == hi.final_executable_edge_pct
    assert lo.size_mult < hi.size_mult
    neg = _view(0.0003, risk_estimate=0.0015)
    lo_neg = _edge({**neg, "confidence": 0.0}, 0.0012)
    hi_neg = _edge({**neg, "confidence": 0.95}, 0.0012)
    assert not lo_neg.eligible and not hi_neg.eligible


def test_size_reads_final_edge_and_learned_risk_within_bounds():
    view = _view(0.003, confidence=0.5)
    tight = _edge({**view, "risk_estimate": 0.0005}, 0.0015)
    wide = _edge({**view, "risk_estimate": 0.004}, 0.0015)
    assert tight.size_mult > wide.size_mult
    lo, hi = al.SIZE_BOUNDS[al.SCALP_ENGINE]
    for e in (tight, wide, _edge({**_view(0.012, confidence=0.5), "risk_estimate": 0.0}, 0.006)):
        assert lo <= e.size_mult <= hi


# ── the formula: calibrated move of the current candidate, residuals separate ─


def test_final_edge_starts_from_the_current_candidates_calibrated_move():
    view = _view(0.0008, 0.0011, claim_capture=0.5, claim_raw_center=0.0010, micro_residual=-0.0001)
    a = _edge(view, 0.0010, spread=0.0001, impact=0.00005)
    b = _edge(view, 0.0014, spread=0.0001, impact=0.00005)
    cost = canonical_roundtrip_cost_pct(spread_pct=0.0001, buy_impact_pct=0.00005)
    assert a.live_cost_pct == pytest.approx(cost)
    assert a.calibrated_move_pct == pytest.approx(0.0008)
    assert a.base_executable_edge_pct == pytest.approx(0.0008 - cost)
    assert a.final_executable_edge_pct == pytest.approx(0.0008 - cost + 0.0003 - 0.0001)
    assert b.final_executable_edge_pct - a.final_executable_edge_pct == pytest.approx(0.5 * 0.0004)


def test_adaptive_and_micro_residuals_are_separate_from_base():
    e = _edge(_view(0.003, 0.0032, micro_residual=0.0001), 0.0012)
    d = e.as_dict()
    keys = ("raw_expected_move_pct", "calibrated_move_pct", "expected_move_pct", "live_cost_pct", "base_executable_edge_pct")
    keys += ("adaptive_residual_pct", "micro_residual_pct", "final_executable_edge_pct", "confidence")
    for key in keys:
        assert key in d
    assert d["base_executable_edge_pct"] == pytest.approx(d["calibrated_move_pct"] - d["live_cost_pct"])
    assert d["calibrated_move_pct"] == pytest.approx(0.003)
    assert d["adaptive_residual_pct"] == pytest.approx(0.0002)
    assert d["micro_residual_pct"] == pytest.approx(0.0001)


def test_micro_cannot_lift_a_non_positive_candidate():
    cost = canonical_roundtrip_cost_pct(spread_pct=SPREAD)
    e = _edge(_view(cost - 0.0002, micro_residual=0.0015), 0.002)
    assert e.micro_residual_model_pct == pytest.approx(0.0015)
    assert e.micro_residual_pct == 0.0
    assert not e.eligible
    down = _edge(_view(cost + 0.0002, micro_residual=-0.0015), 0.002)
    assert down.micro_residual_pct == pytest.approx(-0.0015)
    assert not down.eligible
    lifted = _edge(_view(cost - 0.0002, cost + 0.0002, micro_residual=0.0005), 0.002)
    assert lifted.micro_residual_pct == pytest.approx(0.0005)
    assert lifted.eligible


def test_micro_model_clamp_is_unchanged():
    assert al.MICRO_MODEL_TILT_MAX == 0.0015


# ── one authority ─────────────────────────────────────────────────────────────


def test_ranking_uses_the_one_canonical_edge_function():
    src = inspect.getsource(ranking)
    assert "scalp_executable_edge" in src
    assert "target_reachable(" not in src
    assert "entry_required_gross_edge_pct" not in src
    assert "TARGET_NOT_REACHABLE" not in HARD_REJECT_REASONS
    assert "NO_EXECUTABLE_NET_EDGE" in HARD_REJECT_REASONS


def test_eligibility_reads_final_edge_computed_exactly_once(monkeypatch):
    calls = []
    real = ee.scalp_executable_edge

    def counting(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(ee, "scalp_executable_edge", counting)
    _patch_view(monkeypatch, **_view(0.006))
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    assert len(calls) == 1
    assert rc.entry_eligible == calls[0].eligible
    assert rc.net_edge_after_costs_pct == calls[0].final_executable_edge_pct
    assert classify_scalp_candidate(_row(rc))[0] == "ARMED"
    stale_key_only = {"entry_eligible": True, "hard_block": None, "executable_edge": {"edge_after_cost_pct": 0.002}}
    assert classify_scalp_candidate(stale_key_only)[0] == "REJECTED:NO_EXECUTABLE_EDGE_ESTIMATE"


@pytest.mark.parametrize("edge", [-0.0004, 0.0])
def test_non_positive_final_edge_cannot_arm_even_if_marked_eligible(edge):
    row = {"entry_eligible": True, "hard_block": None, "executable_edge": {"final_executable_edge_pct": edge}}
    assert classify_scalp_candidate(row)[0] == "REJECTED:NO_EXECUTABLE_NET_EDGE"


def test_ranked_negative_final_edge_is_hard_blocked(monkeypatch):
    cost = canonical_roundtrip_cost_pct(spread_pct=SPREAD)
    _patch_view(monkeypatch, **_view(cost + 0.001, cost - 0.002))
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.0025), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    assert rc.executable_edge["raw_move_source"] == "STRATEGY_CLAIM"
    assert rc.executable_edge["base_executable_edge_pct"] > 0
    assert rc.executable_edge["final_executable_edge_pct"] < 0
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE"
    assert classify_scalp_candidate(_row(rc))[0] == "REJECTED:NO_EXECUTABLE_NET_EDGE"


def test_edge_estimate_failure_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("adaptive store unavailable")

    monkeypatch.setattr(al, "scalp_decision", boom)
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    assert rc.hard_block == "NO_EXECUTABLE_EDGE_ESTIMATE"
    assert not rc.entry_eligible


def test_hard_safety_still_rejects_before_edge():
    row = {"entry_eligible": False, "hard_block": "SPREAD_TOO_WIDE", "executable_edge": {"final_executable_edge_pct": 0.002}}
    assert classify_scalp_candidate(row)[0] == "REJECTED:SPREAD_TOO_WIDE"


def test_no_duplicate_negative_edge_live_gate_in_scalp_loop():
    from backend.services import portfolio_engine_integration as pei

    src = inspect.getsource(pei)
    assert src.count("REJECTED:LEARNED_NEGATIVE_EDGE") == 0  # learned flag is telemetry on both engines
    assert "SCALP_V2_ABSTAIN" not in src
    assert re.search(r"(?<!\w)scalp_decision\(", src) is None
    assert "detail=decision_detail(row" in src


def test_scalp_abstention_is_telemetry_only(tmp_path):
    db = str(tmp_path / "abst.db")
    _observe(db, "markout_forward", -0.003, n=40)
    _claims(db, raw=0.003, gross=0.006, n=20)
    view = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    assert view["abstain"] is True
    assert view["abstain_live_veto"] is False
    edge = _edge(view, 0.003)
    assert edge.eligible


# ── learning: residual and risk in consistent units ───────────────────────────


def _record(db, *, t0, raw=0.0008, version_ok=True, symbol=KEY["symbol"]):
    al.record_candidate(
        db,
        engine=al.SCALP_ENGINE,
        symbol=symbol,
        setup=KEY["setup"],
        regime=KEY["regime"],
        ref_price=100.0,
        roundtrip_cost=ESTIMATED_ROUNDTRIP_COST,
        signaled=False,
        evaluated_at=t0,
        features={"obi_l5": 0.2, "spread_pct": 0.00004},
        raw_expected_move=raw,
    )
    if not version_ok:
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE adaptive_candidate_markouts SET strategy_version='LEGACY' WHERE evaluated_at=?", (t0,))


def test_resolver_learns_residual_and_gross_path_mae(tmp_path):
    db = str(tmp_path / "resolve.db")
    t0 = 1_790_000_000.0
    _record(db, t0=t0)
    learned = al.resolve_markouts(db, lambda _s, _t: 100.1, now=t0 + 3000, path_low=lambda _s, _a, _b: 99.8)
    assert learned == 1
    res = al.estimate(db, al.SCALP_ENGINE, KEY["symbol"], KEY["setup"], KEY["regime"], "edge_residual")
    mae = al.estimate(db, al.SCALP_ENGINE, KEY["symbol"], KEY["setup"], KEY["regime"], "markout_mae")
    forward = 0.001 - ESTIMATED_ROUNDTRIP_COST
    with sqlite3.connect(db) as conn:
        rows = dict(conn.execute("SELECT metric, ewma FROM adaptive_metric_state").fetchall())
    assert rows["edge_residual"] == pytest.approx(forward - (0.0008 - ESTIMATED_ROUNDTRIP_COST))
    assert rows["markout_mae"] == pytest.approx(0.002)
    assert res["n"] == 1 and mae["n"] == 1


def test_risk_threshold_uses_consistent_units():
    contract = scalp_v2_max_adverse_net_pct("ETHUSDT")
    assert scalp_v2_adverse_net_threshold_pct("ETHUSDT", {"risk_estimate": 0.0004}) == pytest.approx(0.0004 + ESTIMATED_ROUNDTRIP_COST)
    assert scalp_v2_adverse_net_threshold_pct("ETHUSDT", {"risk_estimate": 0.002}) == contract
    assert scalp_v2_adverse_net_threshold_pct("ETHUSDT", {}) == contract


def test_view_risk_is_gross_path_mae_not_censored_trade_mae(tmp_path):
    db = str(tmp_path / "risk.db")
    _observe(db, "trade_mae", 0.0003, n=20)
    _observe(db, "markout_mae", 0.0021, n=20)
    view = al.scalp_decision(db, KEY["symbol"], KEY["setup"], KEY["regime"], None)
    assert view["risk_source"] == "markout_mae_gross_path"
    assert view["risk_estimate"] > 0.0019


def _pos(risk):
    adapt = {"target_pct": 0.004, "hold_min": 20.0, "risk_estimate": risk}
    return SimpleNamespace(engine_id="SCALP_V2", cost_basis=100.0, entry_price=100.0, highest_price=100.0, lowest_price=99.9, adaptive_decision=adapt, symbol="ETHUSDT")


def _eval(risk, net):
    return evaluate_scalp_v2_exit(position=_pos(risk), current_price=100.0 * (1 + net + ESTIMATED_ROUNDTRIP_COST), net_pnl_pct=net, hold_minutes=2.0, bar_low=99.95, symbol="ETHUSDT")


def test_adverse_stop_uses_adaptive_risk_live():
    assert _eval(0.0003, -0.0009).get("reason") != SCALP_V2_EXIT_ADVERSE
    assert _eval(0.0003, -0.0010).get("reason") == SCALP_V2_EXIT_ADVERSE
    assert _eval(0.003, -0.0014).get("reason") != SCALP_V2_EXIT_ADVERSE
    assert _eval(0.003, -0.0015).get("reason") == SCALP_V2_EXIT_ADVERSE


def test_rebuild_uses_only_current_version_evidence_and_keeps_other_state(tmp_path):
    db = str(tmp_path / "rebuild.db")
    t0 = 1_790_000_000.0
    for i in range(6):
        _record(db, t0=t0 + i * 60)
    _record(db, t0=t0 + 999, version_ok=False)
    al.resolve_markouts(db, lambda _s, _t: 100.1, now=t0 + 5000, path_low=lambda _s, _a, _b: 99.8)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_metric_state SET ewma=0.0000001 WHERE metric='markout_mae'")
        conn.execute("UPDATE adaptive_candidate_markouts SET learned=1 WHERE strategy_version='LEGACY'")
    _observe(db, "trade_net", 0.0005, n=3)
    assert al.observe(db, engine=al.DAY_ENGINE, symbol="BTCUSDT", setup="X", regime="r", metric="trade_mae", value=0.004, strategy_version=al.current_strategy_version(al.DAY_ENGINE))
    seen = []
    stats = rebuild_scalp_edge_state(
        db,
        raw_move_for=lambda _row: (None, "atr_estimate"),
        path_low=lambda _s, _a, _b: 99.7,
        backup_suffix="t1",
        on_decision=lambda row, view, _raw, _source: seen.append((row["id"], view["n_residual"])),
    )
    assert stats["rows"] == 6 and stats["legacy_rows_ignored"] == 1
    assert stats["residual_obs"] == 6 and stats["risk_obs"] == 6
    assert seen[0][1] == 0
    with sqlite3.connect(db) as conn:
        state = {(r[0], r[1]): (r[2], r[3]) for r in conn.execute("SELECT engine_id, metric, n, ewma FROM adaptive_metric_state")}
        backup_n = conn.execute("SELECT COUNT(*) FROM adaptive_metric_state_bak_t1").fetchone()[0]
    assert state[(al.SCALP_ENGINE, "markout_mae")][1] == pytest.approx(0.003)
    assert state[(al.SCALP_ENGINE, "trade_net")][0] == pytest.approx(3.0, abs=0.01)
    assert state[(al.DAY_ENGINE, "trade_mae")][1] == pytest.approx(0.004)
    assert backup_n == len(state)


def test_rebuild_walk_forward_is_causal(tmp_path):
    db = str(tmp_path / "causal.db")
    t0 = 1_790_000_000.0
    _record(db, t0=t0)
    _record(db, t0=t0 + 30)
    al.resolve_markouts(db, lambda _s, _t: 100.1, now=t0 + 5000, path_low=None)
    seen = []
    rebuild_scalp_edge_state(db, raw_move_for=lambda _row: (None, "atr_estimate"), path_low=None, backup_suffix=None, on_decision=lambda _row, view, _raw, _source: seen.append(view["n_residual"]))
    assert seen == [0, 0]


# ── persisted decision rows ───────────────────────────────────────────────────


def test_reject_rows_store_edge_deficit(tmp_path):
    rc = rank_setup_signal(_sig(passed=False, reason="TARGET_NOT_REACHABLE", directional=0.0004), regime="RANGE", ctx=_ctx(_bars(30, 0.0004)))
    row = _row(rc)
    code, reason = classify_scalp_candidate(row)
    assert code == "REJECTED:NO_EXECUTABLE_NET_EDGE"
    db = tmp_path / "decisions.db"
    record_scalp_decision(db, "ETHUSDT", code, reason, cycle_ts=1.0, detail=decision_detail(row))
    with sqlite3.connect(db) as conn:
        (stored,) = conn.execute("SELECT detail FROM scalp_v2_decisions WHERE symbol='ETHUSDT'").fetchone()
    d = json.loads(stored)
    for key in (
        "raw_expected_move",
        "calibrated_move",
        "expected_move",
        "live_cost",
        "base_executable_edge",
        "adaptive_residual",
        "micro_residual",
        "final_executable_edge",
        "confidence",
        "target",
        "hold",
        "risk_estimate",
        "reject_threshold",
        "reject_deficit",
        "edge_before_cost",
        "cost",
        "edge_after_cost",
        "deficit_to_zero",
    ):
        assert key in d, key
    assert d["base_executable_edge"] == pytest.approx(d["calibrated_move"] - d["live_cost"], abs=1e-8)
    assert d["final_executable_edge"] == pytest.approx(d["base_executable_edge"] + d["adaptive_residual"] + d["micro_residual"], abs=1e-8)
    assert d["deficit_to_zero"] == pytest.approx(-d["final_executable_edge"], abs=1e-8)


def test_detail_is_never_empty_for_hard_blocks_before_edge():
    d = json.loads(decision_detail({"hard_block": "STALE_DATA", "rank_meta": {"expected_move_pct": 0.002}}))
    assert d["edge_computed"] is False and d["hard_block"] == "STALE_DATA"


# ── scope: DAY untouched, slots and coins intact ──────────────────────────────


def test_day_priors_and_decision_unchanged():
    # Net quantities (realized trade net, lifecycle net, cost-adjusted forward markout) are neutral.
    assert al._PRIORS[al.DAY_ENGINE] == {
        "trade_mfe": 0.012,
        "trade_mae": 0.006,
        "trade_time_to_mfe_min": 90.0,
        "trade_continuation": 0.45,
        "trade_net": 0.0,
        "lifecycle_net": 0.0,
        "markout_forward": 0.0,
        "markout_mae": 0.006,
    }
    src = inspect.getsource(al.day_decision)
    assert "edge_residual" not in src and "executable_edge" not in src


def test_day_markout_mae_keeps_its_units(tmp_path):
    db = str(tmp_path / "day.db")
    t0 = 1_790_000_000.0
    al.record_candidate(db, engine=al.DAY_ENGINE, symbol="BTCUSDT", setup="DAY_X", regime="r", ref_price=100.0, roundtrip_cost=0.001, signaled=False, evaluated_at=t0)
    al.resolve_markouts(db, lambda _s, _t: 99.5, now=t0 + 30000, path_low=lambda _s, _a, _b: 90.0)
    with sqlite3.connect(db) as conn:
        rows = dict(conn.execute("SELECT metric, ewma FROM adaptive_metric_state WHERE engine_id='DAY_V2'").fetchall())
    assert rows["markout_mae"] == pytest.approx(0.006)
    assert "edge_residual" not in rows


def test_day_exit_does_not_read_the_scalp_edge():
    import backend.services.day_v2.live_exit_evaluator as day_exit

    assert "executable_edge" not in inspect.getsource(day_exit)


def test_four_scalp_slots_and_four_coins():
    from backend.services import two_engine_claim
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.portfolio_engine import SCALP_MAX_OPEN_POSITIONS

    assert SCALP_MAX_OPEN_POSITIONS == 4
    assert two_engine_claim.engine_cap("SCALP_V2") == 4
    assert set(get_scalp_config().products) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
