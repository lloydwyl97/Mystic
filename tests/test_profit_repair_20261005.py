"""Profit repair: DAY expected net from realized closes and candidate lifecycles,
pooled within the setup; SCALP claim calibration learned from every claim,
rejected or filled; a bounded micro weight; versioned economic state; causal
labels. No count gate, no slot or coin change, no exit change."""

from __future__ import annotations

import inspect
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

import backend.services.adaptive_learning as al
import backend.services.economic_replay as replay
import backend.services.economic_state_rebuild as rebuild_mod
import backend.services.portfolio_engine as pe
from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services.day_v2.config import DAY_V2_UNIVERSE
from backend.services.day_v2.lifecycle_sim import LifecycleParams, simulate_lifecycle
from backend.services.day_v2.live_signal import ENABLED_SETUPS
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from backend.services.scalp_v2.decision_log import classify_scalp_candidate
from backend.services.scalp_v2.executable_edge import REJECT_THRESHOLD_PCT, scalp_executable_edge

DAY, SCALP = al.DAY_ENGINE, al.SCALP_ENGINE
T0 = 1_791_200_000.0  # after both economic anchors
SPREAD = 0.00002
COST = canonical_roundtrip_cost_pct(spread_pct=SPREAD, buy_impact_pct=0.0, sell_impact_pct=0.0)
SKEY = ("ETHUSDT", "VWAP_EMA_RECLAIM", "btcup_vollo")
LIVE_DAY_FIELDS = ("expected_net", "size_mult", "objective_atr_mult", "structural_emphasis", "runner_activation_mult", "runner_trail_mult", "runner_tighten_mult")


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _life(db, value, *, symbol="BTCUSDT", setup="BREAKOUT_CONTINUATION", regime="btcup_volhi", at=T0):
    return al.observe(db, engine=DAY, symbol=symbol, setup=setup, regime=regime, metric="lifecycle_net", value=value, strategy_version=al.current_strategy_version(DAY), now=at)


def _day(db, symbol="BTCUSDT", setup="BREAKOUT_CONTINUATION", regime="btcup_volhi", at=T0):
    return al.day_decision(db, symbol, setup, regime, now=at)


def _cand(symbol, setup, adaptive):
    sig = SimpleNamespace(setup=setup, structural_anchor=99.0, target_price=103.0, atr_1h=1.0, objective_structural=103.0, regime="neutral", atr=0.4)
    return {"symbol": symbol, "signal": sig, "ask_price": 100.0, "adaptive": adaptive}


def _params(entry_time=T0):
    return LifecycleParams(setup="RANGE_BOUNCE", entry_price=100.0, entry_time=entry_time, atr_15m=0.4, structural_anchor=99.0, target_price=103.0, atr_1h=1.0, objective_structural=103.0)


def _rising_bars(start=T0, n=730):
    px = [100.0 + 0.01 * min(i, 300) for i in range(n)]
    return [(start + 60.0 * i, p, p + 0.05, p - 0.02, p + 0.01) for i, p in enumerate(px)]


def _scalp_claims(db, n, *, start, raw, exit_price, step=60.0):
    """``n`` rejected (never filled) strategy claims, all resolved at one moment."""
    for i in range(n):
        al.record_candidate(
            db,
            engine=SCALP,
            symbol=SKEY[0],
            setup=SKEY[1],
            regime=SKEY[2],
            ref_price=100.0,
            roundtrip_cost=COST,
            signaled=False,
            evaluated_at=start + step * i,
            raw_expected_move=raw,
            raw_move_source="STRATEGY_CLAIM",
        )
    end = start + step * n + 1300.0
    al.resolve_markouts(db, lambda _s, _t: exit_price, now=end)
    return end


def _edge(view, raw):
    return scalp_executable_edge(view, raw_expected_move_pct=raw, spread_pct=SPREAD, impact_pct=0.0, edge_source="STRATEGY_CLAIM")


# --- SCALP ---------------------------------------------------------------------


def test_scalp_overprediction_is_learned_downward_from_rejected_claims(tmp_path):
    db = str(tmp_path / "s.db")
    raw = 0.0012
    cold = _edge(al.scalp_decision(db, *SKEY, now=T0), raw)
    assert cold.final_executable_edge_pct > REJECT_THRESHOLD_PCT
    end = _scalp_claims(db, 30, start=T0, raw=raw, exit_price=100.0)
    view = al.scalp_decision(db, *SKEY, now=end)
    warm = _edge(view, raw)
    realized = -COST
    assert view["n_residual_strategy"] > 0 and view["n_claim"] > 0
    assert view["adaptive_residual_strategy"] < 0
    assert abs(warm.final_executable_edge_pct - realized) < 0.25 * abs(cold.final_executable_edge_pct - realized)
    assert warm.final_executable_edge_pct <= REJECT_THRESHOLD_PCT


def test_one_scalp_claim_already_moves_the_edge(tmp_path):
    db = str(tmp_path / "s.db")
    end = _scalp_claims(db, 1, start=T0, raw=0.0012, exit_price=100.0)
    view = al.scalp_decision(db, *SKEY, now=end)
    assert view["n_residual_strategy"] == pytest.approx(1.0, rel=1e-2)
    assert -0.0012 < view["adaptive_residual_strategy"] < 0.0


def test_negative_final_edge_cannot_arm_a_live_order(tmp_path):
    db = str(tmp_path / "s.db")
    end = _scalp_claims(db, 30, start=T0, raw=0.0012, exit_price=100.0)
    edge = _edge(al.scalp_decision(db, *SKEY, now=end), 0.0012)
    assert edge.final_executable_edge_pct <= REJECT_THRESHOLD_PCT and edge.eligible is False
    row = {"entry_eligible": True, "snap": {"bid": 1.0}, "executable_edge": edge.as_dict()}
    assert classify_scalp_candidate(row) == ("REJECTED:NO_EXECUTABLE_NET_EDGE", "NO_EXECUTABLE_NET_EDGE")


def test_scalp_positive_evidence_recovers_after_losses(tmp_path):
    db = str(tmp_path / "s.db")
    raw = 0.0012
    neg_end = _scalp_claims(db, 30, start=T0, raw=raw, exit_price=100.0)
    assert _edge(al.scalp_decision(db, *SKEY, now=neg_end), raw).final_executable_edge_pct <= REJECT_THRESHOLD_PCT
    pos_end = _scalp_claims(db, 30, start=T0 + 7 * 86400, raw=raw, exit_price=100.4)
    assert _edge(al.scalp_decision(db, *SKEY, now=pos_end), raw).final_executable_edge_pct > REJECT_THRESHOLD_PCT


def test_scalp_cost_units_are_counted_once(tmp_path):
    db = str(tmp_path / "s.db")
    raw = 0.0020
    end = _scalp_claims(db, 1, start=T0, raw=raw, exit_price=100.1)
    with sqlite3.connect(db) as conn:
        state = dict(conn.execute("SELECT metric, ewma FROM adaptive_metric_state WHERE engine_id='SCALP_V2' AND setup=?", (SKEY[1],)).fetchall())
    # realized net (0.10% gross - cost) minus the claimed base edge (raw - cost): the cost cancels.
    assert state["claim_residual"] == pytest.approx(0.001 - raw)
    assert state["claim_base_edge"] == pytest.approx(raw - COST)
    edge = _edge(al.scalp_decision(db, *SKEY, now=end), raw)
    econ = edge.economic()
    assert edge.base_executable_edge_pct == pytest.approx(raw - COST)
    assert econ["expected_cost"] == pytest.approx(COST)
    assert econ["expected_gross"] - econ["expected_cost"] == pytest.approx(econ["expected_net_edge"])
    assert econ["economic_version"] == al.current_economic_version(SCALP)


def test_micro_weight_is_bounded_and_falls_when_micro_misleads(tmp_path):
    version = al.current_strategy_version(SCALP)
    db = str(tmp_path / "m.db")
    assert al.micro_weight(db, now=T0)["weight"] == pytest.approx(1.0)
    for i in range(40):
        al.learn_micro_weight(db, strategy_version=version, tilt=0.0005, miss=-0.0005, now=T0 + i)
    assert 0.0 <= al.micro_weight(db, now=T0 + 40)["weight"] < 0.2
    helpful = str(tmp_path / "h.db")
    for i in range(40):
        al.learn_micro_weight(helpful, strategy_version=version, tilt=0.0005, miss=0.0015, now=T0 + i)
    assert al.micro_weight(helpful, now=T0 + 40)["weight"] == pytest.approx(1.0)


def test_micro_cannot_lift_a_non_positive_edge_and_its_tilt_is_capped(tmp_path):
    lifted = _edge({"adaptive_residual_strategy": -0.002, "micro_residual": 0.0015, "claim_capture": 1.0}, 0.0012)
    assert lifted.micro_residual_pct == 0.0
    assert lifted.final_executable_edge_pct <= REJECT_THRESHOLD_PCT
    damped = _edge({"adaptive_residual_strategy": 0.0, "micro_residual": -0.0003, "claim_capture": 1.0}, 0.0030)
    assert damped.micro_residual_pct == pytest.approx(-0.0003)
    db = str(tmp_path / "c.db")
    for i in range(200):
        al.update_linear_model(db, SCALP, "micro_edge", {"ofi": 1.0 + (i % 3)}, 0.05, now=T0 + i)
    view = al.scalp_decision(db, *SKEY, {"ofi": 50.0}, now=T0 + 200)
    assert abs(view["micro_residual"]) <= al.MICRO_MODEL_TILT_MAX + 1e-12


def test_scalp_tick_marks_are_recorded_but_carry_no_learning(tmp_path):
    db = str(tmp_path / "t.db")
    rid = al.record_candidate(
        db,
        engine=SCALP,
        symbol=SKEY[0],
        setup=SKEY[1],
        regime=SKEY[2],
        ref_price=100.0,
        roundtrip_cost=COST,
        signaled=False,
        evaluated_at=T0,
        raw_expected_move=0.0012,
        raw_move_source="STRATEGY_CLAIM",
    )
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=T0 + 1300, tick_quote=lambda _s, _a, _b: 100.05)
    with sqlite3.connect(db) as conn:
        marks = json.loads(conn.execute("SELECT markouts_json FROM adaptive_candidate_markouts WHERE id=?", (rid,)).fetchone()[0])
        metrics = {r[0] for r in conn.execute("SELECT metric FROM adaptive_metric_state")}
    assert marks["1s"] == pytest.approx(0.0005 - COST) and {"5s", "10s"} <= set(marks)
    assert not any(m.endswith("s") and m[:-1].isdigit() for m in metrics)


# --- DAY -----------------------------------------------------------------------


def test_day_lifecycle_label_is_causal_and_learned_once(tmp_path):
    db = str(tmp_path / "d.db")
    al.record_candidate(
        db,
        engine=DAY,
        symbol="XRPUSDT",
        setup="RANGE_BOUNCE",
        regime="neutral",
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=T0,
        lifecycle=_params(),
        candidate_state=al.CANDIDATE_QUALIFIED,
    )
    bars = _rising_bars()
    clock = [T0 + 120 * 60]
    windows: list[tuple[float, float]] = []

    def closed_bars(_s, a, b):
        windows.append((a, b))
        return [bar for bar in bars if a <= bar[0] < b and bar[0] + 60.0 <= clock[0]]

    al.resolve_markouts(db, lambda _s, _t: 101.0, now=clock[0], bars_1m=closed_bars)
    assert _day(db, "XRPUSDT", "RANGE_BOUNCE", "neutral", at=clock[0])["economic"]["n_lifecycle"] == 0.0
    assert windows and all(a >= T0 for a, _b in windows)
    clock[0] = T0 + 731 * 60
    al.resolve_markouts(db, lambda _s, _t: 103.0, now=clock[0], bars_1m=closed_bars)
    assert _day(db, "XRPUSDT", "RANGE_BOUNCE", "neutral", at=clock[0])["economic"]["n_lifecycle"] == pytest.approx(1.0)
    al.resolve_markouts(db, lambda _s, _t: 103.0, now=clock[0] + 3600, bars_1m=closed_bars)
    assert _day(db, "XRPUSDT", "RANGE_BOUNCE", "neutral", at=clock[0])["economic"]["n_lifecycle"] == pytest.approx(1.0, rel=1e-2)


def test_lifecycle_ignores_bars_before_the_decision_and_waits_for_the_exit():
    flat = [(T0 + 60.0 * i, 100.0, 100.02, 99.98, 100.0) for i in range(721)]
    crash_before = [(T0 - 60.0, 100.0, 100.0, 50.0, 100.0)]
    end = T0 + 722 * 60
    assert simulate_lifecycle(_params(), crash_before + flat, roundtrip_cost=0.00066, now=end) == simulate_lifecycle(_params(), flat, roundtrip_cost=0.00066, now=end)
    assert simulate_lifecycle(_params(), flat[:60], roundtrip_cost=0.00066, now=T0 + 61 * 60) == {"final": False}


def test_scalp_replay_exit_ignores_bars_before_the_decision():
    store = SimpleNamespace(minute_path=lambda _s, _a, _b: [(T0 - 60.0, 100.0, 100.0, 50.0, 100.0), *[(T0 + 60.0 * i, 100.0, 100.01, 99.99, 100.0) for i in range(30)]])
    out = replay.scalp_exit_sim({"symbol": "ETHUSDT", "ref": 100.0, "t": T0, "cost": COST}, {"target_pct": 0.004, "hold_min": 10.0}, store)
    assert out is not None and out["reason"] == "TIME_STOP"


def test_one_learnable_lifecycle_per_opportunity_and_filled_ones_excluded(tmp_path):
    db = str(tmp_path / "o.db")
    common = {"engine": DAY, "setup": "RANGE_BOUNCE", "regime": "neutral", "ref_price": 100.0, "roundtrip_cost": 0.00066, "signaled": True}
    al.record_candidate(db, symbol="BTCUSDT", evaluated_at=T0, lifecycle=_params(), opportunity_id="BTC-1", candidate_state=al.CANDIDATE_QUALIFIED, **common)
    al.record_candidate(db, symbol="BTCUSDT", evaluated_at=T0 + 900, lifecycle=_params(T0 + 900), opportunity_id="BTC-1", candidate_state=al.CANDIDATE_QUALIFIED, **common)
    filled = al.record_candidate(db, symbol="ETHUSDT", evaluated_at=T0, lifecycle=_params(), opportunity_id="ETH-1", candidate_state=al.CANDIDATE_QUALIFIED, **common)
    al.record_candidate(db, symbol="SOLUSDT", evaluated_at=T0, lifecycle=_params(), opportunity_id="SOL-1", candidate_state=al.CANDIDATE_QUALIFIED_BLOCKED, **common)
    al.record_candidate(db, symbol="XRPUSDT", evaluated_at=T0, lifecycle=_params(), opportunity_id="XRP-1", candidate_state=al.CANDIDATE_NEAR_QUALIFIED, **common)
    assert al.mark_candidate_filled(db, filled) is True
    bars = _rising_bars(n=760)
    now = T0 + 760 * 60
    al.resolve_markouts(db, lambda _s, _t: 103.0, now=now, bars_1m=lambda _s, a, b: [bar for bar in bars if a <= bar[0] < b])
    n_life = {sym: _day(db, sym, "RANGE_BOUNCE", "neutral", at=now)["economic"]["n_lifecycle"] for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}
    assert n_life == {"BTCUSDT": pytest.approx(1.0), "ETHUSDT": 0.0, "SOLUSDT": pytest.approx(1.0), "XRPUSDT": 0.0}


def test_day_hierarchy_pools_key_related_then_setup(tmp_path):
    db = str(tmp_path / "h.db")
    for _ in range(6):
        _life(db, -0.01, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="r1")
    own = _day(db, "BTCUSDT", "RANGE_BOUNCE", "r1")
    related = _day(db, "BTCUSDT", "RANGE_BOUNCE", "r2")
    setup_only = _day(db, "ETHUSDT", "RANGE_BOUNCE", "r3")
    other_setup = _day(db, "BTCUSDT", "HTF_TREND_PULLBACK", "r1")
    assert own["expected_net"] < related["expected_net"] < setup_only["expected_net"] < 0.0
    assert (other_setup["expected_net"], other_setup["size_mult"]) == (0.0, 1.0)
    assert set(own["economic"]["levels"]) == {"engine", "setup", "related", "key"}


def test_breakout_losses_lower_rank_and_size_then_later_gains_restore_it(tmp_path):
    db = str(tmp_path / "b.db")
    for i in range(6):
        _life(db, -0.008, at=T0 + 3600 * i)
    now = T0 + 6 * 3600
    breakout = _day(db, at=now)
    rng = _day(db, "ETHUSDT", "RANGE_BOUNCE", at=now)
    assert breakout["size_mult"] < 1.0 == rng["size_mult"]
    ranked = rank_day_candidates([_cand("BTCUSDT", "BREAKOUT_CONTINUATION", breakout), _cand("ETHUSDT", "RANGE_BOUNCE", rng)], ["BTCUSDT", "ETHUSDT"], 0.00066)
    assert [c["symbol"] for c in ranked] == ["ETHUSDT", "BTCUSDT"]
    later = T0 + 20 * 86400
    for i in range(12):
        _life(db, 0.012, at=later + 3600 * i)
    restored = _day(db, at=later + 12 * 3600)
    assert restored["expected_net"] > 0.0 and restored["size_mult"] > 1.0


def test_range_and_trend_move_only_with_their_own_evidence(tmp_path):
    db = str(tmp_path / "i.db")
    keys = [(sym, setup, "btcup_volhi") for sym in ("BTCUSDT", "ETHUSDT") for setup in ("RANGE_BOUNCE", "HTF_TREND_PULLBACK")]

    def snapshot():
        return {k: {f: al.day_decision(db, *k, now=T0)[f] for f in LIVE_DAY_FIELDS} for k in keys}

    before = snapshot()
    for _ in range(10):
        _life(db, -0.02)
    assert al.learn_from_close(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_volhi",
        strategy_version=al.current_strategy_version(DAY),
        net_pct=-0.02,
        mfe_pct=0.001,
        mae_pct=0.025,
        hold_min=30.0,
        continuation=0.1,
        version_current=True,
        is_dust=False,
        entered_at=T0,
        now=T0,
    )
    assert snapshot() == before
    _life(db, 0.01, symbol="ETHUSDT", setup="RANGE_BOUNCE")
    assert al.day_decision(db, "ETHUSDT", "RANGE_BOUNCE", "btcup_volhi", now=T0)["expected_net"] > 0.0


def test_one_day_label_moves_the_decision_with_no_count_gate(tmp_path):
    db = str(tmp_path / "n.db")
    _life(db, 0.01, symbol="SOLUSDT", setup="VWAP_REVERSION", regime="r")
    d = _day(db, "SOLUSDT", "VWAP_REVERSION", "r")
    assert d["expected_net"] > 0.0 and d["size_mult"] > 1.0
    for fn in (al.day_decision, al.day_net_expectancy, al._lattice, al.scalp_decision, al.scalp_claim_calibration, al.micro_weight, al.resolve_markouts):
        src = inspect.getsource(fn)
        assert not any(gate in src for gate in ("MIN_TRADES", "min_trades", "MIN_SAMPLES", "n >= 8", "n < 8", "min_n"))


def test_legacy_and_other_version_state_never_move_decisions(tmp_path):
    db = str(tmp_path / "l.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE adaptive_metric_state (engine_id TEXT, symbol TEXT, setup TEXT, regime TEXT, metric TEXT, "
            "n REAL, ewma REAL, m2 REAL, updated_at TEXT, PRIMARY KEY (engine_id, symbol, setup, regime, metric))"
        )
        conn.execute("INSERT INTO adaptive_metric_state VALUES ('DAY_V2','BTCUSDT','RANGE_BOUNCE','r','trade_net',50,-0.05,0,?)", (_iso(T0),))
    cold = _day(db, "BTCUSDT", "RANGE_BOUNCE", "r")
    assert (cold["expected_net"], cold["size_mult"]) == (0.0, 1.0)
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {al.LEGACY_STATE_TABLE}").fetchone()[0] == 1
        conn.execute(
            "INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at) "
            "VALUES ('DAY_V2','OLD','BTCUSDT','RANGE_BOUNCE','r','lifecycle_net',50,-0.05,0,?)",
            (_iso(T0),),
        )
    assert _day(db, "BTCUSDT", "RANGE_BOUNCE", "r")["expected_net"] == 0.0
    close = {
        "engine": DAY,
        "symbol": "BTCUSDT",
        "setup": "RANGE_BOUNCE",
        "regime": "r",
        "strategy_version": al.current_strategy_version(DAY),
        "net_pct": -0.02,
        "mfe_pct": 0.001,
        "mae_pct": 0.02,
        "hold_min": 60.0,
        "continuation": 0.2,
        "version_current": True,
        "is_dust": False,
        "now": T0,
    }
    assert al.learn_from_close(db, entered_at=al.anchor_epoch(DAY) - 60.0, **close) is False
    rid = al.record_candidate(db, engine=DAY, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="r", ref_price=100.0, roundtrip_cost=0.00066, signaled=True, evaluated_at=T0)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET economic_version='' WHERE id=?", (rid,))
    al.resolve_markouts(db, lambda _s, _t: 90.0, now=T0 + 400 * 60)
    stale = _day(db, "BTCUSDT", "RANGE_BOUNCE", "r", at=T0 + 400 * 60)
    assert (stale["expected_net"], stale["n_forward"]) == (0.0, 0.0)


def test_adaptive_values_enter_live_ranking_and_sizing(tmp_path):
    db = str(tmp_path / "r.db")
    for _ in range(5):
        _life(db, -0.01, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="r")
        _life(db, 0.01, symbol="ETHUSDT", setup="RANGE_BOUNCE", regime="r")
    htf = _day(db, "BTCUSDT", "HTF_TREND_PULLBACK", "r")
    rng = _day(db, "ETHUSDT", "RANGE_BOUNCE", "r")
    ranked = rank_day_candidates([_cand("BTCUSDT", "HTF_TREND_PULLBACK", htf), _cand("ETHUSDT", "RANGE_BOUNCE", rng)], ["BTCUSDT", "ETHUSDT"], 0.00066)
    assert [c["symbol"] for c in ranked] == ["ETHUSDT", "BTCUSDT"]
    for cand, adaptive in zip(ranked, (rng, htf), strict=True):
        assert cand["rank"]["score"] == pytest.approx(adaptive["expected_net"])
        assert cand["rank"]["size_mult"] == pytest.approx(adaptive["size_mult"])
    assert htf["size_mult"] < 1.0 < rng["size_mult"]
    fund = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert 'size_mult = float(adapt.get("size_mult") or 1.0)' in fund and "sized_qty * size_mult" in fund
    params = LifecycleParams.from_signal(_cand("ETHUSDT", "RANGE_BOUNCE", rng)["signal"], entry_price=100.0, entry_time=T0, adaptive=rng)
    assert params.objective_atr_mult == pytest.approx(rng["objective_atr_mult"])
    assert params.runner_trail_mult == pytest.approx(rng["runner_trail_mult"])


def test_loss_memory_decays_so_later_evidence_can_recover(tmp_path):
    decayed, fresh = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    later = T0 + 42 * 86400
    for _ in range(10):
        _life(decayed, -0.01, symbol="XRPUSDT", setup="RANGE_BOUNCE", regime="r", at=T0)
        _life(fresh, -0.01, symbol="XRPUSDT", setup="RANGE_BOUNCE", regime="r", at=T0)
    for _ in range(3):
        _life(decayed, 0.01, symbol="XRPUSDT", setup="RANGE_BOUNCE", regime="r", at=later)
        _life(fresh, 0.01, symbol="XRPUSDT", setup="RANGE_BOUNCE", regime="r", at=T0)
    assert _day(decayed, "XRPUSDT", "RANGE_BOUNCE", "r", at=later)["expected_net"] > 0.0
    assert _day(fresh, "XRPUSDT", "RANGE_BOUNCE", "r", at=T0)["expected_net"] < 0.0


def test_decisions_read_the_data_clock_not_the_wall_clock(tmp_path):
    db = str(tmp_path / "w.db")
    for _ in range(4):
        _life(db, -0.01, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="r", at=T0)
    assert al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "r") == al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "r", now=T0)


def test_state_rebuild_refuses_once_live_rows_exist(tmp_path):
    db = str(tmp_path / "live.db")
    al.record_candidate(db, engine=DAY, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="r", ref_price=100.0, roundtrip_cost=0.00066, signaled=True, evaluated_at=T0)
    out = rebuild_mod.rebuild(db, now=T0 + 60, apply=True, workdir=str(tmp_path))
    assert out["applied"] is False and out["refused"] == "LIVE_CURRENT_VERSION_ROWS=1"


# --- scope ---------------------------------------------------------------------


def test_scope_slots_coins_setups_and_no_order_paths():
    assert DAY_V2_UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert (pe.DAY_MAX_OPEN_POSITIONS, pe.SCALP_MAX_OPEN_POSITIONS) == (4, 4)
    assert {"HTF_TREND_PULLBACK", "RANGE_BOUNCE", "BREAKOUT_CONTINUATION", "VWAP_REVERSION", "EXHAUSTION_MR"} == ENABLED_SETUPS
    assert al.SIZE_BOUNDS == {"DAY_V2": (0.55, 1.35), "SCALP_V2": (0.50, 1.25)}
    for module in (al, replay, rebuild_mod):
        src = inspect.getsource(module)
        assert not any(call in src for call in ("execute_buy", "execute_sell", "place_order", "create_order"))
