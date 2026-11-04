"""SCALP learned continuation value: continuous state, uncensored labels, one function for entry and exit."""

from __future__ import annotations

import numpy as np
import pytest

from backend.services import adaptive_learning
from backend.services.scalp_v2 import continuation_value as cv
from backend.services.scalp_v2.executable_edge import scalp_executable_edge

T0 = 1_790_000_000.0
COST = 0.0006
REGIME = "btcdown_vollo"


def _bars(closes: list[float], lows: list[float] | None = None) -> cv.Bars:
    lows = lows or closes
    rows = [(T0 + 60.0 * i, c, max(c, lo), lo, c) for i, (c, lo) in enumerate(zip(closes, lows, strict=True))]
    return cv.bars_from(rows)


def _cand(cid: int = 1, t0: float = T0 + 1.0, ref: float = 100.0) -> cv.Candidate:
    return cv.Candidate(cid, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, t0, ref, COST, {"calibrated": 0.0004, "cost": COST})


def _state(net: float, age: float = 10.0) -> dict[str, float]:
    return {"net": net, "mfe": 0.0, "mae": max(0.0, -net - COST), "giveback": 0.0, "age_min": age}


def test_labels_uncensored_by_stop_and_cover_states_no_position_held():
    closes = [100.0, 100.3, 100.5, 100.4, 100.6, 99.0, 98.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    labels = cv.candidate_labels(_cand(), _bars(closes), -np.inf, np.inf)
    assert labels
    by_age = {}
    for horizon, x, advantage, end, weight in labels:
        assert weight > 0
        by_age.setdefault(round((end - horizon - T0 - 1.0) / 60.0), []).append((horizon, x, advantage))
    # A profitable state, which the old policy sold at once, is labeled with what followed.
    assert any(x[1] > 0 for items in by_age.values() for _, x, _ in items)
    # Forward regression labels reflect the true observed gross path (uncensored by the -150 bps execution stop).
    age2 = {h: a for h, _, a in by_age[2]}
    gross_at_2 = 100.3 / 100.0 - 1.0
    true_gross_at_7 = 98.0 / 100.0 - 1.0
    assert age2[300] == pytest.approx(true_gross_at_7 - gross_at_2)
    # No state exists once a closed bar has reached the stop.
    assert max(by_age) == 6


def test_deep_and_shallow_losses_are_different_states_and_monotonic_non_increasing():
    deep = cv.vector(_state(-0.0150), {}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
    shallow = cv.vector(_state(-0.0008), {}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
    assert not np.allclose(deep, shallow)
    model = cv.ValueModel()
    rng = np.random.default_rng(3)
    labels = []
    for _ in range(4000):
        net = -rng.uniform(0.0, 0.016)
        x = cv.vector(_state(net), {}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
        # Even if raw labels have a perverse negative slope where deeper losses appear higher,
        # the monotonic adverse constraints prevent deeper losses from exceeding shallow losses.
        labels.append((120, x, 0.3 * (net + 0.008) + rng.normal(0, 0.0002), T0, 1.0))
    model.fold(labels, T0)
    deep_v = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, _state(-0.0150), {}, model=model)
    shallow_v = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, _state(-0.0008), {}, model=model)
    assert shallow_v["advantage"] > 0 > deep_v["advantage"]


def test_pure_noise_state_terms_shrink_to_the_mean_advantage():
    model = cv.ValueModel()
    rng = np.random.default_rng(5)
    labels = []
    for _ in range(3000):
        net = -rng.uniform(0.0, 0.016)
        x = cv.vector(_state(net), {"calibrated": rng.normal(0, 0.001)}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
        labels.append((120, x, -0.0002 + rng.normal(0, 0.002), T0, 1.0))
    model.fold(labels, T0)
    beta = model.beta(120)
    assert beta[0] == pytest.approx(-0.0002, abs=0.0001)
    assert np.linalg.norm(beta[1:]) * 1e4 < 0.5


def test_entry_value_replaces_claim_gap_and_calibration():
    view = {"claim_capture": 0.5, "claim_gross_setup": 0.001, "policy_gap": 0.002, "policy_calibration": 0.001}
    legacy = scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.0001, impact_pct=0.0, edge_source="STRATEGY_CLAIM")
    unset = scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.0001, impact_pct=0.0, edge_source="STRATEGY_CLAIM", value_of=lambda _entry: None)
    assert unset.final_executable_edge_pct == legacy.final_executable_edge_pct
    assert legacy.entry_value_pct is None
    seen = {}
    valued = scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.0001, impact_pct=0.0, edge_source="STRATEGY_CLAIM", value_of=lambda entry: seen.update(entry) or -0.0001)
    assert valued.final_executable_edge_pct == pytest.approx(-0.0001 - valued.live_cost_pct)
    assert valued.policy_gap_pct == 0.0 and valued.policy_calibration_pct == 0.0
    assert valued.market_edge_pct == pytest.approx(legacy.market_edge_pct)
    assert not valued.eligible
    assert seen["cost"] == pytest.approx(valued.live_cost_pct)
    assert valued.economic()["entry_value"] == pytest.approx(-0.0001)


def test_admitted_entry_is_kept_at_its_first_check():
    model = cv.ValueModel()
    x = cv.vector(cv.entry_position_state(COST), {"cost": COST}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
    model.fold([(120, x, 0.002, T0, 1.0)] * 50, T0)
    entry = cv.entry_value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, {"cost": COST}, model=model)
    assert entry > COST
    hold = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, cv.entry_position_state(COST), {"cost": COST}, model=model)
    terminal = -COST + hold["advantage"]
    assert adaptive_learning.learned_hold_or_exit(expected_terminal_net=terminal, unrealized_net=-COST) == "hold"


def test_continuation_terminal_reads_value_model_for_scalp(monkeypatch, tmp_path):
    db = str(tmp_path / "s.db")
    seen = {}

    def fake_value(db_path, symbol, setup, regime, state, entry, **_):
        seen.update(state=state, entry=entry)
        return {"advantage": -0.0007, "horizon": 120, "horizons": {120: -0.0007}}

    monkeypatch.setattr(cv, "value", fake_value)
    features = {**_state(-0.012), "entry_economic": {"calibrated_move": 0.0003, "expected_cost": COST}}
    terminal = adaptive_learning.continuation_terminal(db, "SCALP_V2", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, -0.012, features=features)
    assert terminal == pytest.approx(-0.0127)
    assert seen["entry"]["calibrated"] == pytest.approx(0.0003) and seen["entry"]["cost"] == pytest.approx(COST)
    assert seen["state"]["net"] == pytest.approx(-0.012)


def _install_loaders(monkeypatch, candidates, bars):
    monkeypatch.setattr(cv, "load_candidates", lambda _db, start, end: [c for c in candidates if start <= c.t0 < end])
    monkeypatch.setattr(cv, "load_bars", lambda *_args: bars)


def test_build_installs_and_fold_due_folds_each_label_once(monkeypatch, tmp_path):
    db = str(tmp_path / "s.db")
    adaptive_learning._connect(db).close()
    closes = [100.0 + 0.05 * ((i * 7) % 5) for i in range(240)]
    bars = _bars(closes)
    candidates = [_cand(i, T0 + 1.0 + 60.0 * i) for i in range(0, 200, 3)]
    _install_loaders(monkeypatch, candidates, bars)
    assert cv.load_model(db, fresh=True) is None
    summary = cv.build(db, as_of=T0 + 3600.0)
    assert cv.authority(db)
    first = cv.load_model(db, fresh=True)
    assert first.clock == pytest.approx(T0 + 3600.0 - cv.LABEL_LAG_SEC)
    assert sum(summary["labels"].values()) > 0
    cv._LAST_FOLD.clear()
    now = T0 + 3600.0 + 600.0
    folded = cv.fold_due(db, now=now)
    assert folded > 0
    after = cv.load_model(db, fresh=True)
    assert after.clock == pytest.approx(now - cv.LABEL_LAG_SEC)
    cv._LAST_FOLD.clear()
    assert cv.fold_due(db, now=now) == 0
    full = cv.ValueModel()
    full.fold(cv._labels_for(db, candidates, -np.inf, now - cv.LABEL_LAG_SEC, {"BTCUSDT": bars}), now - cv.LABEL_LAG_SEC)
    for horizon in cv.HORIZONS_SEC:
        assert after.stats[horizon]["n"] == pytest.approx(full.stats[horizon]["n"], rel=1e-6)


def test_entry_microstructure_is_encoded_stored_and_fades_with_age():
    feats = {"obi_l5": 0.4, "ofi_5s": 2.0, "agg_flow_imbalance_5s": -1.0, "adverse_selection_score": 0.1, "microprice_pressure": 5e-5, "spread_pct": 1e-4, "data_age_sec": 0.2}
    raw = cv.micro_raw(feats)
    assert "data_age_sec" not in raw and raw["obi_l5"] == 0.4
    entry = cv.entry_state({"calibrated_move": 0.0002, "entry_micro": raw})
    assert entry["m_obi"] == pytest.approx(0.4) and entry["m_ofi"] == pytest.approx(np.tanh(1.0)) and entry["m_spread"] == pytest.approx(1.0)
    col = cv.FEATURES.index("micro_obi_fresh")
    fresh = cv.vector(_state(-COST, age=0.0), entry, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)[col]
    later = cv.vector(_state(-COST, age=9.0), entry, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)[col]
    assert fresh == pytest.approx(0.4) and later == pytest.approx(0.04)
    seen = {}
    edge = scalp_executable_edge(
        {"claim_capture": 0.5},
        raw_expected_move_pct=0.002,
        spread_pct=0.0001,
        impact_pct=0.0,
        edge_source="STRATEGY_CLAIM",
        value_of=lambda e: seen.update(e) or 0.0,
        micro_features=feats,
    )
    assert seen["m_obi"] == pytest.approx(0.4)
    assert edge.economic()["entry_micro"] == raw
    assert cv.entry_state(edge.economic())["m_flow"] == pytest.approx(-1.0)


def test_without_a_model_scalp_learns_nothing_and_prices_entry_at_minus_cost(tmp_path):
    from backend.services.binance_scalp.scalp_candidate_ranking import _scalp_entry_value

    db = str(tmp_path / "s.db")
    adaptive_learning._connect(db).close()
    assert cv.value(db, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, _state(-0.01), {}) is None
    assert _scalp_entry_value(db, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, {"cost": COST}) == 0.0
    edge = scalp_executable_edge(
        {"claim_capture": 1.0, "claim_gross_setup": 0.01},
        raw_expected_move_pct=0.01,
        spread_pct=0.0001,
        impact_pct=0.0,
        edge_source="STRATEGY_CLAIM",
        value_of=lambda e: _scalp_entry_value(db, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, e),
    )
    assert edge.market_edge_pct > 0
    assert edge.final_executable_edge_pct == pytest.approx(-edge.live_cost_pct)
    assert adaptive_learning.continuation_terminal(db, "SCALP_V2", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, -0.01, features=_state(-0.01)) == pytest.approx(-0.01)


class _FakeProc:
    started: list = []

    def __init__(self, cmd, **_kwargs):
        self.cmd = cmd
        self.returncode = None
        _FakeProc.started.append(self)

    def poll(self):
        return self.returncode


def test_fold_due_bootstraps_one_build_when_no_model(monkeypatch, tmp_path):
    db = str(tmp_path / "s.db")
    adaptive_learning._connect(db).close()
    _FakeProc.started = []
    monkeypatch.setattr(cv.subprocess, "Popen", _FakeProc)
    cv._LAST_FOLD.clear()
    cv._BOOTSTRAP.clear()
    assert cv.fold_due(db, now=T0) == 0
    assert len(_FakeProc.started) == 1
    assert _FakeProc.started[0].cmd[-2:] == ["--db", db]
    assert _FakeProc.started[0].cmd[1].endswith("scripts/scalp_value_build.py")
    cv._LAST_FOLD.clear()
    assert cv.fold_due(db, now=T0 + 120) == 0
    assert len(_FakeProc.started) == 1
    _FakeProc.started[0].returncode = 0
    cv._LAST_FOLD.clear()
    cv.fold_due(db, now=T0 + 240, start_build=False)
    assert db not in cv._BOOTSTRAP
    cv._BOOTSTRAP.clear()


def test_status_reports_installed_clock_and_stall(monkeypatch, tmp_path):
    db = str(tmp_path / "s.db")
    adaptive_learning._connect(db).close()
    assert cv.status(db)["installed"] is False
    closes = [100.0 + 0.05 * ((i * 7) % 5) for i in range(240)]
    _install_loaders(monkeypatch, [_cand(i, T0 + 1.0 + 60.0 * i) for i in range(0, 200, 3)], _bars(closes))
    cv.build(db, as_of=T0 + 3600.0)
    fresh = cv.status(db, now=T0 + 3600.0)
    assert fresh["installed"] and not fresh["stalled"]
    assert set(fresh["horizons"]) == {str(h) for h in cv.HORIZONS_SEC}
    assert fresh["build_candidates"] > 0
    assert cv.status(db, now=T0 + 3600.0 + 3600.0)["stalled"]


def test_scalp_value_endpoint_reads_status(monkeypatch, tmp_path):
    import asyncio

    from backend.endpoints import portfolio_engine_endpoints as ep

    db = str(tmp_path / "s.db")
    adaptive_learning._connect(db).close()
    monkeypatch.setenv("TRADING_DB_PATH", db)
    out = asyncio.run(ep.get_scalp_value_status())
    assert out["success"] and out["data"]["installed"] is False


def test_live_wiring_passes_entry_economics_and_folds_every_cycle():
    import inspect

    from backend.services import portfolio_engine, portfolio_engine_integration

    engine_src = inspect.getsource(portfolio_engine)
    assert '_scalp_state["entry_economic"] = ((_scalp_adapt or {}).get("executable_edge") or {}).get("economic")' in engine_src
    integration_src = inspect.getsource(portfolio_engine_integration)
    assert "await _asyncio.to_thread(fold_due, markout_db)" in integration_src
    assert "SCALP_VALUE_FOLD_FAILED" in integration_src


def test_load_candidates_preserves_decision_time_entry_micro_over_later_features(tmp_path):
    db = str(tmp_path / "m.db")
    adaptive_learning.record_candidate(
        db,
        engine="SCALP_V2",
        symbol="BTCUSDT",
        setup="FAILED_BREAKOUT_REVERSAL",
        regime="btcflat_volhi",
        ref_price=100.0,
        roundtrip_cost=COST,
        signaled=False,
        evaluated_at=T0,
        features={"obi_l5": -0.5, "ofi_5s": -4.0},
        economic={"calibrated_move": 0.0008, "entry_micro": {"obi_l5": 0.6, "ofi_5s": 2.0}},
    )
    adaptive_learning.record_candidate(
        db,
        engine="SCALP_V2",
        symbol="ETHUSDT",
        setup="FAILED_BREAKOUT_REVERSAL",
        regime="btcflat_volhi",
        ref_price=100.0,
        roundtrip_cost=COST,
        signaled=False,
        evaluated_at=T0 + 1.0,
        features={"obi_l5": 0.35, "ofi_5s": 2.0},
        economic={"calibrated_move": 0.0008},
    )
    c1, c2 = cv.load_candidates(db, T0 - 1.0, T0 + 2.0)
    assert c1.entry["m_obi"] == pytest.approx(0.6)
    assert c1.entry["m_ofi"] == pytest.approx(np.tanh(1.0))
    assert c2.entry["m_obi"] == pytest.approx(0.35)
    assert c2.entry["m_ofi"] == pytest.approx(np.tanh(1.0))


def test_scalp_claim_regime_isolation_and_value_fading(tmp_path):
    db = str(tmp_path / "iso.db")
    ver = adaptive_learning.current_strategy_version("SCALP_V2")
    for _ in range(30):
        adaptive_learning.learn_claim_label(db, symbol="ETHUSDT", setup="FAILED_BREAKOUT_REVERSAL", regime="btcdown_vollo", strategy_version=ver, raw=0.002, gross=-0.004, now=T0)
    for _ in range(5):
        adaptive_learning.learn_claim_label(db, symbol="XRPUSDT", setup="FAILED_BREAKOUT_REVERSAL", regime="btcflat_volhi", strategy_version=ver, raw=0.002, gross=0.0012, now=T0)
    cal = adaptive_learning.scalp_claim_calibration(db, "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", now=T0)
    assert cal["claim_gross_mean"] > COST
    cold_cal = adaptive_learning.scalp_claim_calibration(db, "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "unseen_regime", now=T0)
    assert cold_cal["claim_gross_mean"] < 0.0
    model = cv.ValueModel()
    x = cv.vector(cv.entry_position_state(COST), {"cost": COST}, "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi")
    model.fold([(120, x, 0.0, T0, 1.0)] * 20, T0)
    ent = {"calibrated": cal["claim_gross_setup"], "adaptive": cal["claim_gross_mean"] - cal["claim_gross_setup"], "cost": COST}
    claim_total = ent["calibrated"] + ent["adaptive"]
    ev0 = cv.entry_value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", ent, model=model)
    hold0 = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", cv.entry_position_state(COST), ent, model=model)["advantage"]
    hold2 = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", _state(0.0, age=2.0), ent, model=model)["advantage"]
    hold5 = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", _state(0.0, age=5.0), ent, model=model)["advantage"]
    hold10 = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", _state(0.0, age=10.0), ent, model=model)["advantage"]
    assert ev0 == pytest.approx(hold0)
    assert hold2 == pytest.approx(0.8 * claim_total, abs=1e-6)
    assert hold5 == pytest.approx(0.5 * claim_total, abs=1e-6)
    assert ev0 > COST
    assert abs(hold10) < 1e-6


def test_claim_expectation_cannot_double_count_even_with_nonzero_betas():
    model = cv.ValueModel()
    x0 = cv.vector(cv.entry_position_state(COST), {"calibrated": 0.0008, "adaptive": 0.0002, "cost": COST}, "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi")
    for idx in cv._DIRECT_CLAIM_INDICES:
        assert x0[idx] == 0.0
    model.fold([(120, x0, 0.0, T0, 1.0)] * 20, T0)
    ent = {"calibrated": 0.0008, "adaptive": 0.0, "cost": COST}
    baseline = cv.entry_value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", ent, model=model)
    cf = cv.ValueModel.from_payload(model.to_payload())
    cf._beta = {h: model.beta(h).copy() for h in cv.HORIZONS_SEC if model.beta(h) is not None}
    for idx in cv._DIRECT_CLAIM_INDICES:
        cf._beta[120][idx] = 0.5
    counterfactual = cv.entry_value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", ent, model=cf)
    assert baseline == pytest.approx(0.0008, abs=1e-6)
    assert counterfactual == pytest.approx(baseline, abs=1e-12)


def test_no_minimum_hold_and_no_fixed_10m_exit():
    model = cv.ValueModel()
    x0 = cv.vector(cv.entry_position_state(COST), {"cost": COST}, "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi")
    model.fold([(120, x0, -0.0005, T0, 1.0)] * 40, T0)
    ent_small = {"calibrated": 0.0003, "adaptive": 0.0, "cost": COST}
    early = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", _state(-COST, age=5.0), ent_small, model=model)
    assert adaptive_learning.learned_hold_or_exit(expected_terminal_net=-COST + early["advantage"], unrealized_net=-COST) == "exit"

    pos_model = cv.ValueModel()
    pos_model.fold([(120, x0, 0.0004, T0, 1.0)] * 40, T0)
    post10 = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", _state(0.0, age=10.0), {"cost": COST}, model=pos_model)
    assert post10["advantage"] > 0.0
    assert adaptive_learning.learned_hold_or_exit(expected_terminal_net=post10["advantage"], unrealized_net=0.0) == "hold"

    # A profitable position at age=2m is NOT dumped merely because net > 0 while claim_rem remains positive.
    prof = cv.value("unused", "XRPUSDT", "FAILED_BREAKOUT_REVERSAL", "btcflat_volhi", _state(0.0050, age=2.0), {"calibrated": 0.0008, "adaptive": 0.0, "cost": COST}, model=model)
    assert prof["advantage"] == pytest.approx(0.8 * 0.0008 - 0.0005, abs=1e-6)
    assert adaptive_learning.learned_hold_or_exit(expected_terminal_net=0.0050 + prof["advantage"], unrealized_net=0.0050) == "hold"


def test_continuation_learning_defect_repairs_20261104(tmp_path):
    from types import SimpleNamespace

    from backend.services import continuation_surface as cs
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT, SCALP_V2_EXIT_CATASTROPHIC, evaluate_scalp_v2_exit

    # 1. Horizon labels are uncensored by catastrophic execution protection.
    cand = _cand(ref=100.0)
    bars = _bars([100.0, 99.0, 98.8, 97.5, 97.0, 96.5])
    last_i, _gross_1, _ = cv.path_state(cand, bars, 1.0)
    later_3 = cv.horizon_gross(cand, bars, last_i, T0 + 1.0 + 240.0)
    assert later_3 == pytest.approx(97.5 / 100.0 - 1.0)
    assert later_3 < -SCALP_V2_CATASTROPHIC_PCT

    # 2. Catastrophic execution protection itself remains unchanged at -150 bps (0.015).
    assert pytest.approx(0.015) == SCALP_V2_CATASTROPHIC_PCT
    pos = SimpleNamespace(engine_id="SCALP_V2", cost_basis=100.0, entry_price=100.0, highest_price=100.0, lowest_price=98.4, symbol="XRPUSDT")
    cat_dec = evaluate_scalp_v2_exit(position=pos, current_price=98.5, net_pnl_pct=-0.0156, hold_minutes=5.0, bar_low=98.4, symbol="XRPUSDT", expected_terminal_net=0.01)
    assert cat_dec["action"] == "sell" and cat_dec["reason"] == SCALP_V2_EXIT_CATASTROPHIC

    # 3 & 4. Increasing MAE alone cannot raise continuation value, and worsening current net below zero cannot mechanically raise continuation value.
    model = cv.ValueModel()
    rng = np.random.default_rng(42)
    synth_labels = []
    for _ in range(3000):
        net_v = -rng.uniform(0.0005, 0.0145)
        mae_v = abs(net_v) + rng.uniform(0.0, 0.003)
        st_s = {"net": net_v, "mfe": 0.0001, "mae": mae_v, "giveback": 0.0001, "age_min": rng.uniform(1.0, 40.0)}
        x_s = cv.vector(st_s, {"cost": COST}, "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi")
        # Feed perverse labels where deeper loss and higher MAE have artificially higher labels
        y_perverse = -0.25 * net_v + 0.40 * mae_v - 0.0010 + rng.normal(0.0, 0.0002)
        for h in cv.HORIZONS_SEC:
            synth_labels.append((h, x_s, y_perverse, T0, 1.0))
    model.fold(synth_labels, T0)

    for h in cv.HORIZONS_SEC:
        b = model.beta(h)
        assert float(b[cv.FEATURES.index("mae")]) <= 0.0
        assert float(b[cv.FEATURES.index("giveback")]) <= 0.0

    # Monotonicity across MAE = 10..145 bps with net fixed at -50 bps
    prev_adv = None
    for mae_bps in (10, 25, 50, 75, 100, 125, 145):
        st_m = {"net": -0.0050, "mfe": 0.0001, "mae": mae_bps / 1e4, "giveback": 0.0001, "age_min": 15.0}
        adv_m = cv.value("unused", "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", st_m, {"cost": COST}, model=model)["advantage"]
        if prev_adv is not None:
            assert adv_m <= prev_adv + 1e-12
        prev_adv = adv_m

    # Monotonicity across net = 0, -10, -25, -50, -75, -100, -125, -145 bps with other state fixed
    prev_net_adv = None
    for net_bps in (0, -10, -25, -50, -75, -100, -125, -145):
        st_n = {"net": net_bps / 1e4, "mfe": 0.0001, "mae": 0.0150, "giveback": 0.0001, "age_min": 15.0}
        adv_n = cv.value("unused", "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", st_n, {"cost": COST}, model=model)["advantage"]
        if prev_net_adv is not None:
            assert adv_n <= prev_net_adv + 1e-12
        prev_net_adv = adv_n

    # 5 & 6. Healthy profitable low-MAE positions still receive positive continuation when evidence supports it,
    # while immediate-failure states naturally produce negative continuation without an MFE or time gate.
    mixed_model = cv.ValueModel()
    mixed_labels = []
    for _ in range(2000):
        # Healthy developing states (mfe > cost, low mae) continue positively
        st_good = {"net": rng.uniform(0.0002, 0.0030), "mfe": rng.uniform(0.0010, 0.0040), "mae": rng.uniform(0.0, 0.0010), "giveback": 0.0001, "age_min": rng.uniform(1.0, 10.0)}
        x_good = cv.vector(st_good, {"cost": COST}, "BTCUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi")
        # Immediate-failure states (mfe < cost, underwater) continue negatively
        st_bad = {"net": -rng.uniform(0.0020, 0.0120), "mfe": 0.0001, "mae": rng.uniform(0.0025, 0.0120), "giveback": 0.0001, "age_min": rng.uniform(10.0, 30.0)}
        x_bad = cv.vector(st_bad, {"cost": COST}, "BTCUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi")
        for h in cv.HORIZONS_SEC:
            mixed_labels.append((h, x_good, +0.0012, T0, 1.0))
            mixed_labels.append((h, x_bad, -0.0015, T0, 1.0))
    mixed_model.fold(mixed_labels, T0)

    ent_good = {"calibrated": 0.0006, "adaptive": 0.0002, "cost": COST}
    st_good_eval = {"net": 0.0010, "mfe": 0.0020, "mae": 0.0005, "giveback": 0.0001, "age_min": 4.0}
    st_bad_eval = {"net": -0.0060, "mfe": 0.0001, "mae": 0.0070, "giveback": 0.0001, "age_min": 15.0}
    good_read = cv.value("unused", "BTCUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", st_good_eval, ent_good, model=mixed_model)
    bad_read = cv.value("unused", "BTCUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", st_bad_eval, ent_good, model=mixed_model)
    assert good_read["advantage"] > 0.0
    assert bad_read["advantage"] < 0.0

    # 7, 8, 9, 10. Entry decisions, DAY_V2, b90d5e8 claim damping & policy hierarchy, and no future-label leakage.
    db = str(tmp_path / "entry_parity.db")
    ver = adaptive_learning.current_strategy_version("SCALP_V2")
    adaptive_learning.learn_claim_label(db, symbol="XRPUSDT", setup="FAILED_BREAKDOWN_REVERSAL", regime="btcup_volhi", strategy_version=ver, raw=0.0028, gross=0.0010, now=T0)
    view = adaptive_learning.scalp_claim_calibration(db, "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", now=T0 + 10)
    move = adaptive_learning.scalp_expected_gross(view, 0.0028)
    assert "key_policy_calibration" in view and view["key_policy_calibration"] == 0.0
    assert move["expected"] == pytest.approx(move["calibrated"] + move["adaptive"])

    # Preserved entry_beta guarantees byte/value identical entry_value and executable_edge
    entry_b = {str(h): mixed_model.beta(h).tolist() for h in cv.HORIZONS_SEC}
    mixed_model.meta["entry_beta"] = entry_b
    ent_dict = {"calibrated": move["calibrated"], "adaptive": move["adaptive"], "cost": COST}
    ev_before = cv.entry_value("unused", "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", ent_dict, model=mixed_model)
    # Perturb hold _beta while keeping entry_beta intact
    mixed_model._beta = {h: mixed_model.beta(h) - 0.0005 for h in cv.HORIZONS_SEC}
    ev_after = cv.entry_value("unused", "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", ent_dict, model=mixed_model)
    assert ev_before == ev_after

    # DAY_V2 surface_trace remains byte/value identical
    for h_d, adv_d in ((900, 0.0010), (1800, 0.0014), (3600, 0.0020), (7200, 0.0028), (14400, 0.0034), (21600, 0.0038), (43200, -0.0020)):
        cs.record_advantage(
            db,
            engine="DAY_V2",
            symbol="BTCUSDT",
            setup="HTF_TREND_PULLBACK__MARKET",
            regime="btcup_volhi",
            horizon=h_d,
            features={"net": -0.00066, "age_min": 0.0},
            advantage=adv_d,
            weight=8.0,
            now=T0,
        )
    day_tr = cs.surface_trace(db, "DAY_V2", "BTCUSDT", "HTF_TREND_PULLBACK__MARKET", "btcup_volhi", {"age_min": 0.0}, T0)
    assert day_tr is not None and day_tr["advantage"] > 0.0015

    # No future-label leakage: candidate_labels strictly excludes horizons ending after hi
    leaked = cv.candidate_labels(cand, bars, T0, T0 + 100.0)
    assert leaked == []
