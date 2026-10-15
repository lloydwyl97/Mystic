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


def test_labels_clamp_at_stop_and_cover_states_no_position_held():
    closes = [100.0, 100.3, 100.5, 100.4, 100.6, 99.0, 98.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0]
    labels = cv.candidate_labels(_cand(), _bars(closes), -np.inf, np.inf)
    assert labels
    by_age = {}
    for horizon, x, advantage, end, weight in labels:
        assert weight > 0
        by_age.setdefault(round((end - horizon - T0 - 1.0) / 60.0), []).append((horizon, x, advantage))
    # A profitable state, which the old policy sold at once, is labeled with what followed.
    assert any(x[1] > 0 for items in by_age.values() for _, x, _ in items)
    # A window that crosses the 1.5% stop ends at the stop price, not the later low.
    age2 = {h: a for h, _, a in by_age[2]}
    gross_at_2 = 100.3 / 100.0 - 1.0
    assert age2[300] == pytest.approx(-cv.SCALP_V2_CATASTROPHIC_PCT - gross_at_2)
    # No state exists once a closed bar has reached the stop.
    assert max(by_age) == 6


def test_deep_and_shallow_losses_are_different_states_and_values():
    deep = cv.vector(_state(-0.0150), {}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
    shallow = cv.vector(_state(-0.0008), {}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
    assert not np.allclose(deep, shallow)
    model = cv.ValueModel()
    rng = np.random.default_rng(3)
    labels = []
    for _ in range(4000):
        net = -rng.uniform(0.0, 0.016)
        x = cv.vector(_state(net), {}, "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME)
        labels.append((120, x, -0.3 * (net + 0.004) + rng.normal(0, 0.0002), T0, 1.0))
    model.fold(labels, T0)
    deep_v = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, _state(-0.0150), {}, model=model)
    shallow_v = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", REGIME, _state(-0.0008), {}, model=model)
    assert deep_v["advantage"] > 0 > shallow_v["advantage"]


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
