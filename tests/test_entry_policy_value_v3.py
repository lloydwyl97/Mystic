"""Comprehensive semantic and regression tests for ENTRY_POLICY_VALUE_V3.

Proves all Section 37 invariants:
  - ENTRY_POLICY_VALUE_V1 unchanged
  - ENTRY_POLICY_VALUE_V2 unchanged
  - V3 strictly causal fitting & zero future leakage
  - Economic cold-start prior sourced dynamically from canonical economics (-6.60 bps)
  - Zero sample-count gate (fits from observation 1)
  - Strict DETECTOR_FIRED vs MARKET_FALLBACK population isolation
  - Strict directional BTC regime isolation (btcdown / btcflat / btcup)
  - Continuous volatility & state conditioning (distinguishing calm oversold vs crash)
  - Same-(symbol, 15m_bar, population) cluster weight bounded <= 1.0
  - Same-state duplicated candidates cannot inflate effective training weight
  - Cross-coin same-bar Kish effective sample size deflation
  - All 4 coins (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT) & all 5 setups remain eligible
  - Crash target tails (-242 bps) preserved without clipping/winsorization
  - Deterministic SHA-256 model_digest & restart determinism
  - Canonical 6.60 bps cost unchanged
  - Global economic ranking & tie-breaks preserved
  - Uncertainty adjusts bounded size only and is NEVER an entry gate
  - DAY_V2 and SCALP_V2 remain concurrent and isolated
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from types import SimpleNamespace

import pytest

import backend.config.trading_economics as te
import backend.services.adaptive_learning as al
from backend.config.trading_economics import (
    ESTIMATED_ROUNDTRIP_COST,
    canonical_roundtrip_cost_pct,
)
from backend.services.adaptive_learning import executable_bid_net
from backend.services.day_v2.config import DAY_V2_UNIVERSE
from backend.services.day_v2.entry_policy_value import (
    ENTRY_POLICY_NET_V1,
    ENTRY_POLICY_NET_V2,
    ENTRY_POLICY_VALUE_V1,
    ENTRY_POLICY_VALUE_V2,
    ENTRY_POLICY_VALUE_V3,
    LABEL_COUNTERFACTUAL_POLICY,
    LABEL_REAL_FUNDED_EXECUTION,
    apply_resolved_entry_policy_labels,
    causal_replay,
    entry_funding_authorized,
    entry_policy_value,
    rebuild_entry_policy_value_v2,
    record_entry_policy_observation,
)
from backend.services.day_v2.entry_policy_value_v3 import (
    ENTRY_POLICY_V3_FEATURE_VERSION,
    POPULATION_DETECTOR_FIRED,
    POPULATION_MARKET_FALLBACK,
    V3_CONTINUOUS_FEATURE_NAMES,
    apply_resolved_entry_policy_v3_labels,
    causal_replay_v3,
    entry_policy_value_v3,
    invalidate_v3_cache,
    rebuild_entry_policy_value_v3,
    record_entry_policy_v3_observation,
    v3_feature_contract,
    v3_prior_net,
)
from backend.services.day_v2.live_signal import ENABLED_SETUPS, day_state_features
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.exact_policy_capture import ensure_capture_schema
from backend.services.policy_episode import UNIVERSE
from backend.services.portfolio_engine import DAY_MAX_OPEN_POSITIONS, SCALP_MAX_OPEN_POSITIONS
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

T0 = 1_795_000_200.0  # aligned 15m boundary


def _insert_episode(
    conn: sqlite3.Connection,
    *,
    cid: int | None,
    symbol: str = "ETHUSDT",
    setup: str = "VWAP_REVERSION__MARKET",
    regime: str = "btcdown_vollo",
    decided_at: float = T0,
    label_available_at: float = T0 + 900.0,
    entry_ask: float = 1600.0,
    exit_bid: float = 1612.0,
    entry_spread: float = 0.00012,
    kind: str = "COUNTERFACTUAL_POLICY_OUTCOME",
    funded: int = 0,
    realized_net: float | None = None,
    features: dict | None = None,
) -> int:
    ensure_capture_schema(conn)
    cost = canonical_roundtrip_cost_pct()
    p_net = executable_bid_net(entry_ask, exit_bid, cost)
    snap = json.dumps(
        {
            "capture": "EXACT_POLICY_V1",
            "decision_timestamp": decided_at,
            "spread": entry_spread,
            "features": features or {},
        }
    )
    cur = conn.execute(
        """
        INSERT INTO policy_episodes (
            candidate_id, kind, engine_id, symbol, setup, regime, decided_at,
            entry_ask, exit_bid, entry_spread, roundtrip_cost, snapshot_json,
            status, capture_version, planning_net, net, funded,
            exit_at, label_available_at, train_eligible, exit_reason, parity_status
        ) VALUES (?, ?, 'DAY_V2', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED', 'EXACT_POLICY_V1', ?, ?, ?, ?, ?, 1, 'LEARNED_CONTINUATION_EXIT', 'OK')
        """,
        (
            cid,
            kind,
            symbol,
            setup,
            regime,
            decided_at,
            entry_ask,
            exit_bid,
            entry_spread,
            cost,
            snap,
            p_net,
            realized_net if realized_net is not None else p_net,
            funded,
            label_available_at,
            label_available_at,
        ),
    )
    return int(cur.lastrowid)


def _cand(symbol: str, setup: str, adaptive: dict, ask: float = 100.0) -> dict:
    sig = SimpleNamespace(
        setup=setup,
        atr_1h=1.0,
        objective_structural=102.0,
        structural_anchor=98.5,
        target_price=102.0,
        atr=0.4,
    )
    return {"symbol": symbol, "signal": sig, "ask_price": ask, "adaptive": adaptive}


def test_v1_and_v2_remain_completely_unchanged_and_reproducible(tmp_path):
    db = str(tmp_path / "v1_v2_v3.db")
    conn = sqlite3.connect(db)
    ep_id = _insert_episode(
        conn,
        cid=1,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        decided_at=T0,
        label_available_at=T0 + 600.0,
        entry_ask=1600.0,
        exit_bid=1616.0,
        features={"atr15_pct": 0.004, "atr1h_pct": 0.008, "sma20_dist": -0.01, "h1_ret": -0.015, "bar_return": 0.002, "rsi": 0.32, "bb_pct": 0.15, "range_location": 0.2},
    )
    conn.commit()
    conn.close()

    # Seed V1 and V2 explicitly
    assert record_entry_policy_observation(
        db,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        value=0.0094,
        version=ENTRY_POLICY_VALUE_V1,
        now=T0 + 600.0,
    )
    r_v2 = rebuild_entry_policy_value_v2(db, now=T0 + 700.0)
    assert r_v2["version"] == ENTRY_POLICY_VALUE_V2 and r_v2["applied"] == 1

    with sqlite3.connect(db) as c0:
        v1_before = c0.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=?",
            (ENTRY_POLICY_VALUE_V1,),
        ).fetchall()
        v2_before = c0.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=? ORDER BY metric, symbol, setup, regime",
            (ENTRY_POLICY_VALUE_V2,),
        ).fetchall()
        ep_v2_before = c0.execute(
            "SELECT entry_policy_v2_learned, entry_policy_v2_target, entry_policy_v2_target_type FROM policy_episodes WHERE id=?",
            (ep_id,),
        ).fetchone()

    v1_view_before = entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 700.0, version=ENTRY_POLICY_VALUE_V1)
    v2_view_before = entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 700.0, version=ENTRY_POLICY_VALUE_V2)

    # Rebuild V3 twice and verify zero mutation to V1 or V2
    r_v3_1 = rebuild_entry_policy_value_v3(db, now=T0 + 700.0)
    r_v3_2 = rebuild_entry_policy_value_v3(db, now=T0 + 700.0)
    assert r_v3_1 == r_v3_2
    assert r_v3_1["version"] == ENTRY_POLICY_VALUE_V3
    assert r_v3_1["applied"] == 1

    with sqlite3.connect(db) as c1:
        v1_after = c1.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=?",
            (ENTRY_POLICY_VALUE_V1,),
        ).fetchall()
        v2_after = c1.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=? ORDER BY metric, symbol, setup, regime",
            (ENTRY_POLICY_VALUE_V2,),
        ).fetchall()
        ep_v2_after = c1.execute(
            "SELECT entry_policy_v2_learned, entry_policy_v2_target, entry_policy_v2_target_type FROM policy_episodes WHERE id=?",
            (ep_id,),
        ).fetchone()

    assert v1_after == v1_before
    assert v2_after == v2_before
    assert ep_v2_after == ep_v2_before
    assert entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 700.0, version=ENTRY_POLICY_VALUE_V1) == v1_view_before
    assert entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 700.0, version=ENTRY_POLICY_VALUE_V2) == v2_view_before
    assert v1_view_before["metric"] == ENTRY_POLICY_NET_V1
    assert v2_view_before["metric"] == ENTRY_POLICY_NET_V2


def test_v3_economic_prior_is_sourced_dynamically_from_canonical_economics_with_no_sample_gate(tmp_path, monkeypatch):
    db = str(tmp_path / "prior.db")
    cost = canonical_roundtrip_cost_pct()
    assert cost == pytest.approx(0.00066)
    assert v3_prior_net() == pytest.approx(-cost)

    cold = al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "btcdown_vollo", now=T0)
    assert cold["entry_expected_policy_net"] == pytest.approx(-cost)
    assert cold["expected_net"] == pytest.approx(-cost)
    assert cold["economic"]["policy_base"] == ENTRY_POLICY_VALUE_V3
    assert cold["economic"]["entry_policy_version"] == ENTRY_POLICY_VALUE_V3
    assert cold["abstain"] is False
    assert entry_funding_authorized(cold["entry_expected_policy_net"]) is False

    # Prove prior is NOT hard-coded: changing canonical_roundtrip_cost_pct dynamically shifts V3 cold prior
    monkeypatch.setattr(te, "canonical_roundtrip_cost_pct", lambda **_kw: 0.00084)
    monkeypatch.setattr("backend.services.day_v2.entry_policy_value_v3.canonical_roundtrip_cost_pct", lambda **_kw: 0.00084)
    invalidate_v3_cache()
    assert v3_prior_net() == pytest.approx(-0.00084)
    shifted = entry_policy_value_v3(db, "BTCUSDT", "RANGE_BOUNCE", "btcdown_vollo", now=T0)
    assert shifted["mean"] == pytest.approx(-0.00084)

    # Restore canonical cost and prove 1 valid observation immediately moves V3 (no sample gate)
    monkeypatch.undo()
    invalidate_v3_cache()
    assert record_entry_policy_v3_observation(
        db,
        symbol="BTCUSDT",
        setup="RANGE_BOUNCE",
        regime="btcdown_vollo",
        value=0.0080,
        decided_at=T0,
        now=T0 + 60.0,
    )
    one_obs = al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "btcdown_vollo", now=T0 + 60.0)
    assert -cost < one_obs["entry_expected_policy_net"] < 0.0080
    assert one_obs["entry_expected_policy_net"] > 0.0
    assert one_obs["n_entry_policy"] == pytest.approx(1.0)
    for mod_fn in (al.day_decision, entry_policy_value_v3):
        src = inspect.getsource(mod_fn)
        assert not any(tok in src for tok in ("MIN_TRADES", "MIN_SAMPLES", "min_samples"))


def test_v3_strict_causal_fitting_and_zero_future_leakage(tmp_path):
    db = str(tmp_path / "causal.db")
    conn = sqlite3.connect(db)
    # Episode 1 decided at T0, closes at T0 + 1800 (+94 bps)
    _insert_episode(
        conn,
        cid=1,
        symbol="SOLUSDT",
        setup="VWAP_REVERSION",
        regime="btcdown_vollo",
        decided_at=T0,
        label_available_at=T0 + 1800.0,
        entry_ask=100.0,
        exit_bid=101.0,
        features={"atr15_pct": 0.003, "atr1h_pct": 0.006, "sma20_dist": -0.008},
    )
    # Episode 2 decided at T0 + 900 (while Ep 1 is still open!), closes at T0 + 3600 (-200 bps crash)
    _insert_episode(
        conn,
        cid=2,
        symbol="SOLUSDT",
        setup="VWAP_REVERSION",
        regime="btcdown_vollo",
        decided_at=T0 + 900.0,
        label_available_at=T0 + 3600.0,
        entry_ask=100.0,
        exit_bid=98.06,
        features={"atr15_pct": 0.015, "atr1h_pct": 0.012, "sma20_dist": -0.030},
    )
    conn.commit()
    conn.close()

    # Before T0 + 1800: zero labels available -> prediction equals cold economic prior
    assert apply_resolved_entry_policy_v3_labels(db, now=T0 + 900.0) == 0
    at_900 = entry_policy_value_v3(db, "SOLUSDT", "VWAP_REVERSION", "btcdown_vollo", now=T0 + 900.0)
    assert at_900["mean"] == pytest.approx(v3_prior_net())

    # At T0 + 1800: only Ep 1 is available -> prediction lifts above prior
    assert apply_resolved_entry_policy_v3_labels(db, now=T0 + 1800.0) == 1
    at_1800 = entry_policy_value_v3(db, "SOLUSDT", "VWAP_REVERSION", "btcdown_vollo", now=T0 + 1800.0)
    assert at_1800["mean"] > v3_prior_net()

    # Walk-forward replay invariance: mutating future episode outcome/features cannot alter earlier predictions or digests
    ep_a = [
        {
            "id": 1,
            "symbol": "SOLUSDT",
            "setup": "VWAP_REVERSION",
            "regime": "btcdown_vollo",
            "decided_at": T0,
            "label_available_at": T0 + 900.0,
            "entry_ask": 100.0,
            "exit_bid": 101.0,
            "features": {"atr15_pct": 0.003, "atr1h_pct": 0.006},
        },
        {
            "id": 2,
            "symbol": "SOLUSDT",
            "setup": "VWAP_REVERSION",
            "regime": "btcdown_vollo",
            "decided_at": T0 + 900.0,
            "label_available_at": T0 + 1800.0,
            "entry_ask": 100.0,
            "exit_bid": 100.5,
            "features": {"atr15_pct": 0.004, "atr1h_pct": 0.007},
        },
        {
            "id": 3,
            "symbol": "SOLUSDT",
            "setup": "VWAP_REVERSION",
            "regime": "btcdown_vollo",
            "decided_at": T0 + 1800.0,
            "label_available_at": T0 + 2700.0,
            "entry_ask": 100.0,
            "exit_bid": 97.5,
            "features": {"atr15_pct": 0.020, "atr1h_pct": 0.015},
        },
    ]
    ep_b = [
        ep_a[0],
        ep_a[1],
        {**ep_a[2], "exit_bid": 125.0, "features": {"atr15_pct": 0.50, "atr1h_pct": 0.40}},
    ]
    replay_a = {r["id"]: r for r in causal_replay_v3(ep_a)}
    replay_b = {r["id"]: r for r in causal_replay_v3(ep_b)}
    for eid in (1, 2, 3):
        assert replay_a[eid]["prediction"] == pytest.approx(replay_b[eid]["prediction"], abs=1e-15)
        assert replay_a[eid]["model_digest"] == replay_b[eid]["model_digest"]


def test_detector_and_market_fallback_populations_are_strictly_isolated(tmp_path):
    db = str(tmp_path / "pop_iso.db")
    det_before = entry_policy_value_v3(db, "ETHUSDT", "VWAP_REVERSION", "btcdown_vollo", now=T0)
    assert det_before["population"] == POPULATION_DETECTOR_FIRED

    # Flood *__MARKET fallback with 40 severe losses across all setups and symbols
    for step in range(40):
        sym = UNIVERSE[step % len(UNIVERSE)]
        stp = f"{sorted(ENABLED_SETUPS)[step % 5]}__MARKET"
        record_entry_policy_v3_observation(
            db,
            symbol=sym,
            setup=stp,
            regime="btcdown_vollo",
            value=-0.0220,
            decided_at=T0 + step * 900.0,
            now=T0 + (step + 1) * 900.0,
            features={"atr15_pct": 0.010 + 0.0002 * step, "atr1h_pct": 0.015, "sma20_dist": -0.02},
        )

    mkt_after = entry_policy_value_v3(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 50_000.0)
    det_after = entry_policy_value_v3(db, "ETHUSDT", "VWAP_REVERSION", "btcdown_vollo", now=T0 + 50_000.0)
    assert mkt_after["population"] == POPULATION_MARKET_FALLBACK
    assert mkt_after["mean"] < -0.0150
    assert det_after["mean"] == pytest.approx(det_before["mean"], abs=1e-15)
    assert det_after["model_digest"] == det_before["model_digest"]


def test_directional_btc_regimes_are_strictly_isolated(tmp_path):
    db = str(tmp_path / "dir_iso.db")
    up_before = entry_policy_value_v3(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", now=T0)
    flat_before = entry_policy_value_v3(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcflat_vollo", now=T0)

    for step in range(15):
        record_entry_policy_v3_observation(
            db,
            symbol="BTCUSDT",
            setup="BREAKOUT_CONTINUATION",
            regime="btcdown_vollo",
            value=-0.0180,
            decided_at=T0 + step * 900.0,
            now=T0 + (step + 1) * 900.0,
        )

    down_after = entry_policy_value_v3(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcdown_vollo", now=T0 + 20_000.0)
    up_after = entry_policy_value_v3(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", now=T0 + 20_000.0)
    flat_after = entry_policy_value_v3(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcflat_vollo", now=T0 + 20_000.0)

    assert down_after["directional_regime"] == "btcdown"
    assert down_after["mean"] < -0.0100
    assert up_after["mean"] == pytest.approx(up_before["mean"], abs=1e-15)
    assert flat_after["mean"] == pytest.approx(flat_before["mean"], abs=1e-15)
    assert up_after["model_digest"] == up_before["model_digest"]
    assert flat_after["model_digest"] == flat_before["model_digest"]


def test_continuous_state_distinguishes_calm_oversold_reversion_from_liquidation_cascade(tmp_path):
    db = str(tmp_path / "state_cond.db")
    # Train inside the EXACT SAME (symbol, setup, regime) cell ('ETHUSDT', 'VWAP_REVERSION__MARKET', 'btcdown_vollo')
    # where calm oversold dips pay (+42 bps, matching Nov 20 calm counterfactuals) while high-ATR/high-spread cascades lose (-85 bps).
    for step in range(40):
        is_cascade = step % 2 == 1
        feats = (
            {"atr15_pct": 0.0140, "atr1h_pct": 0.0085, "h1_ret": -0.075, "spread_pct": 0.00045, "sma20_dist": -0.035}
            if is_cascade
            else {"atr15_pct": 0.0028, "atr1h_pct": 0.0065, "h1_ret": -0.008, "spread_pct": 0.00010, "sma20_dist": -0.009}
        )
        target = -0.0085 if is_cascade else 0.0042
        record_entry_policy_v3_observation(
            db,
            symbol="ETHUSDT",
            setup="VWAP_REVERSION__MARKET",
            regime="btcdown_vollo",
            value=target,
            decided_at=T0 + step * 900.0,
            now=T0 + (step + 1) * 900.0,
            features=feats,
        )

    t_eval = T0 + 45 * 900.0
    calm_dec = al.day_decision(
        db,
        "ETHUSDT",
        "VWAP_REVERSION__MARKET",
        "btcdown_vollo",
        features={"atr15_pct": 0.0028, "atr1h_pct": 0.0065, "h1_ret": -0.008, "spread_pct": 0.00010, "sma20_dist": -0.009, "decision_timestamp": t_eval},
        now=t_eval,
    )
    cascade_dec = al.day_decision(
        db,
        "ETHUSDT",
        "VWAP_REVERSION__MARKET",
        "btcdown_vollo",
        features={"atr15_pct": 0.0140, "atr1h_pct": 0.0085, "h1_ret": -0.075, "spread_pct": 0.00045, "sma20_dist": -0.035, "decision_timestamp": t_eval},
        now=t_eval,
    )
    assert calm_dec["entry_expected_policy_net"] > 0.0
    assert cascade_dec["entry_expected_policy_net"] < -0.0030
    assert calm_dec["entry_expected_policy_net"] > cascade_dec["entry_expected_policy_net"] + 0.0040


def test_cluster_weights_bounded_and_duplicates_cannot_inflate_effective_sample_size(tmp_path):
    # 1. Five unfunded setups on same (symbol, 15m_bar, population) must have total weight <= 1.0
    db_multi = str(tmp_path / "multi_setup.db")
    conn = sqlite3.connect(db_multi)
    for idx, stp in enumerate(sorted(ENABLED_SETUPS), start=1):
        _insert_episode(
            conn,
            cid=idx,
            symbol="ETHUSDT",
            setup=f"{stp}__MARKET",
            regime="btcdown_vollo",
            decided_at=T0,
            label_available_at=T0 + 900.0,
            entry_ask=1600.0,
            exit_bid=1616.0,
        )
    conn.commit()
    conn.close()
    assert apply_resolved_entry_policy_v3_labels(db_multi, now=T0 + 900.0) == 5
    with sqlite3.connect(db_multi) as chk:
        weights = [float(r[0]) for r in chk.execute("SELECT entry_policy_v3_weight FROM policy_episodes").fetchall()]
    assert sum(weights) <= 1.0 + 1e-12
    assert all(w == pytest.approx(0.20) for w in weights)
    view_multi = entry_policy_value_v3(db_multi, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 900.0)
    assert view_multi["nominal_weight"] == pytest.approx(1.0)
    assert view_multi["effective_sample_size"] == pytest.approx(1.0)

    # 2. Duplicating an observation 10x on the same (symbol, bar, population) cannot inflate effective weight or prediction
    db_single = str(tmp_path / "single.db")
    db_dup = str(tmp_path / "dup.db")
    record_entry_policy_v3_observation(
        db_single,
        symbol="BTCUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        value=0.0080,
        decided_at=T0,
        now=T0 + 900.0,
        obs_id="single:1",
    )
    for k in range(10):
        record_entry_policy_v3_observation(
            db_dup,
            symbol="BTCUSDT",
            setup="VWAP_REVERSION__MARKET",
            regime="btcdown_vollo",
            value=0.0080,
            decided_at=T0,
            now=T0 + 900.0,
            obs_id=f"dup:{k}",
        )
    s_view = entry_policy_value_v3(db_single, "BTCUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 900.0)
    d_view = entry_policy_value_v3(db_dup, "BTCUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 900.0)
    assert d_view["nominal_weight"] == pytest.approx(s_view["nominal_weight"], abs=1e-12)
    assert d_view["effective_sample_size"] == pytest.approx(s_view["effective_sample_size"], abs=1e-12)
    assert d_view["mean"] == pytest.approx(s_view["mean"], abs=1e-12)

    # 3. Four correlated coins on the same 15m bar: nominal_weight == 4.0, effective_sample_size == 1.0
    db_4c = str(tmp_path / "four_coins.db")
    for sy in UNIVERSE:
        record_entry_policy_v3_observation(
            db_4c,
            symbol=sy,
            setup="RANGE_BOUNCE__MARKET",
            regime="btcdown_vollo",
            value=-0.0050,
            decided_at=T0,
            now=T0 + 900.0,
        )
    c4_view = entry_policy_value_v3(db_4c, "BTCUSDT", "RANGE_BOUNCE__MARKET", "btcdown_vollo", now=T0 + 900.0)
    assert c4_view["nominal_weight"] == pytest.approx(4.0)
    assert c4_view["effective_sample_size"] == pytest.approx(1.0)


def test_real_funded_execution_vs_counterfactual_provenance_preserved(tmp_path):
    db = str(tmp_path / "prov.db")
    cid_real = al.record_candidate(
        db,
        engine="DAY_V2",
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        ref_price=1625.17,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=T0,
    )
    conn = sqlite3.connect(db)
    conn.execute("UPDATE adaptive_candidate_markouts SET filled=1, realized_net=? WHERE id=?", (-0.007252, cid_real))
    # Real funded execution with actual accounting net -72.52 bps and an unfunded same-bar shadow
    real_id = _insert_episode(
        conn,
        cid=cid_real,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        decided_at=T0,
        label_available_at=T0 + 1200.0,
        entry_ask=1625.17,
        exit_bid=1615.78,
        kind="REAL_POLICY_OUTCOME",
        funded=1,
        realized_net=-0.007252,
    )
    shadow_id = _insert_episode(
        conn,
        cid=10,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        decided_at=T0,
        label_available_at=T0 + 1200.0,
        entry_ask=1625.17,
        exit_bid=1635.00,
        kind="COUNTERFACTUAL_POLICY_OUTCOME",
        funded=0,
    )
    cf_id = _insert_episode(
        conn,
        cid=11,
        symbol="SOLUSDT",
        setup="RANGE_BOUNCE__MARKET",
        regime="btcdown_vollo",
        decided_at=T0 + 1800.0,
        label_available_at=T0 + 2700.0,
        entry_ask=70.00,
        exit_bid=70.70,
        kind="COUNTERFACTUAL_POLICY_OUTCOME",
        funded=0,
    )
    conn.commit()
    conn.close()

    assert apply_resolved_entry_policy_v3_labels(db, now=T0 + 3000.0) == 2
    with sqlite3.connect(db) as chk:
        chk.row_factory = sqlite3.Row
        r_real = chk.execute("SELECT * FROM policy_episodes WHERE id=?", (real_id,)).fetchone()
        r_shad = chk.execute("SELECT * FROM policy_episodes WHERE id=?", (shadow_id,)).fetchone()
        r_cf = chk.execute("SELECT * FROM policy_episodes WHERE id=?", (cf_id,)).fetchone()

    assert r_real["entry_policy_v3_target_type"] == LABEL_REAL_FUNDED_EXECUTION
    assert r_real["entry_policy_v3_target"] == pytest.approx(-0.007252)
    assert r_real["entry_policy_v3_weight"] == pytest.approx(1.0)
    assert r_shad["entry_policy_v3_weight"] == pytest.approx(0.0)
    assert r_cf["entry_policy_v3_target_type"] == LABEL_COUNTERFACTUAL_POLICY
    assert r_cf["entry_policy_v3_weight"] == pytest.approx(1.0)


def test_crash_target_tails_are_never_clipped_or_winsorized(tmp_path):
    db_crash = str(tmp_path / "tail_crash.db")
    db_mild = str(tmp_path / "tail_mild.db")

    # Record -242.45 bps crash vs -50 bps mild loss
    record_entry_policy_v3_observation(
        db_crash,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        value=-0.024245,
        decided_at=T0,
        now=T0 + 900.0,
    )
    record_entry_policy_v3_observation(
        db_mild,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        value=-0.005000,
        decided_at=T0,
        now=T0 + 900.0,
    )
    with sqlite3.connect(db_crash) as chk:
        stored_target = float(chk.execute("SELECT target FROM entry_policy_v3_observations").fetchone()[0])
    assert stored_target == pytest.approx(-0.024245, abs=1e-12)

    v_crash = entry_policy_value_v3(db_crash, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 900.0)
    v_mild = entry_policy_value_v3(db_mild, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 900.0)
    assert v_crash["mean"] < v_mild["mean"] - 0.0050
    assert v_crash["residual_sd"] > v_mild["residual_sd"]


def test_model_digest_and_restart_reload_are_strictly_deterministic(tmp_path):
    db = str(tmp_path / "digest.db")
    for step in range(6):
        record_entry_policy_v3_observation(
            db,
            symbol=UNIVERSE[step % 4],
            setup="VWAP_REVERSION",
            regime="btcdown_vollo",
            value=0.0040 if step % 2 == 0 else -0.0030,
            decided_at=T0 + step * 900.0,
            now=T0 + (step + 1) * 900.0,
            features={"atr15_pct": 0.0035 + 0.0003 * step, "atr1h_pct": 0.007, "sma20_dist": -0.006},
        )
    eval_t = T0 + 10_000.0
    before = al.day_decision(
        db,
        "ETHUSDT",
        "VWAP_REVERSION",
        "btcdown_vollo",
        features={"atr15_pct": 0.004, "atr1h_pct": 0.007, "sma20_dist": -0.006, "decision_timestamp": eval_t},
        now=eval_t,
    )
    digest_before = before["economic"]["entry_policy_model_digest"]
    assert len(digest_before) == 64

    # Simulate process restart: clear all in-memory caches and re-query from SQLite
    al._invalidate_adaptive_caches(db)
    invalidate_v3_cache()

    after = al.day_decision(
        db,
        "ETHUSDT",
        "VWAP_REVERSION",
        "btcdown_vollo",
        features={"atr15_pct": 0.004, "atr1h_pct": 0.007, "sma20_dist": -0.006, "decision_timestamp": eval_t},
        now=eval_t,
    )
    assert after["economic"]["entry_policy_model_digest"] == digest_before
    assert after["entry_expected_policy_net"] == pytest.approx(before["entry_expected_policy_net"], abs=1e-15)
    assert after["uncertainty"] == pytest.approx(before["uncertainty"], abs=1e-15)
    assert after["size_mult"] == pytest.approx(before["size_mult"], abs=1e-15)


def test_all_four_coins_and_setups_eligible_ranking_economic_uncertainty_not_gate_and_scalp_isolated(tmp_path):
    db = str(tmp_path / "full_contract.db")
    assert UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert DAY_V2_UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert DAY_MAX_OPEN_POSITIONS == 4
    assert SCALP_MAX_OPEN_POSITIONS == 4
    assert len(v3_feature_contract()) == len(V3_CONTINUOUS_FEATURE_NAMES) == 14

    # Verify live day_state_features populates V3 state inputs deterministically
    bars = [{"ts": T0 - (40 - i) * 900, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + 0.05 * i, "volume": 25.0} for i in range(40)]
    sf = day_state_features(bars, bars[-16:], "VWAP_REVERSION")
    for req_key in ("sma20_dist", "vwap20_dist", "bar_return", "ret4_15m", "h1_ret", "atr15_pct", "atr1h_pct", "vol_expansion", "rsi", "bb_pct", "range_location"):
        assert req_key in sf

    # Snapshot SCALP before DAY V3 learning
    scalp_before = al.scalp_decision(db, "BTCUSDT", "VWAP_EMA_RECLAIM", "btcdown_vollo", now=T0)

    # Teach distinct positive edges across all 4 coins & verify all 4 authorize funding and rank economically
    edges = {"XRPUSDT": 0.012, "SOLUSDT": 0.009, "ETHUSDT": 0.006, "BTCUSDT": 0.004}
    for idx, (sym, val) in enumerate(edges.items()):
        for rep in range(3):
            record_entry_policy_v3_observation(
                db,
                symbol=sym,
                setup="VWAP_REVERSION",
                regime="btcdown_vollo",
                value=val,
                decided_at=T0 + (idx * 10 + rep) * 900.0,
                now=T0 + (idx * 10 + rep + 1) * 900.0,
            )

    eval_t = T0 + 60 * 900.0
    decisions = {sym: al.day_decision(db, sym, "VWAP_REVERSION", "btcdown_vollo", now=eval_t) for sym in UNIVERSE}
    for sym, dec in decisions.items():
        assert dec["entry_expected_policy_net"] > 0.0, sym
        assert entry_funding_authorized(dec["entry_expected_policy_net"]) is True
        assert dec["economic"]["entry_policy_feature_version"] == ENTRY_POLICY_V3_FEATURE_VERSION
        lo, hi = al.SIZE_BOUNDS["DAY_V2"]
        assert lo <= dec["size_mult"] <= hi
        assert dec["abstain_live_veto"] is False

    ranked = rank_day_candidates(
        [_cand(sym, "VWAP_REVERSION", decisions[sym]) for sym in UNIVERSE],
        UNIVERSE,
        canonical_roundtrip_cost_pct(),
    )
    assert [c["symbol"] for c in ranked] == ["XRPUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT"]

    # High residual dispersion (volatile wins + losses with positive mean) still funds & stays within SIZE_BOUNDS
    db_hi_unc = str(tmp_path / "hi_unc.db")
    for step, val in enumerate((0.035, -0.020, 0.032, -0.018, 0.030)):
        record_entry_policy_v3_observation(
            db_hi_unc,
            symbol="SOLUSDT",
            setup="EXHAUSTION_MR",
            regime="btcdown_volhi",
            value=val,
            decided_at=T0 + step * 900.0,
            now=T0 + (step + 1) * 900.0,
        )
    hi_unc_dec = al.day_decision(db_hi_unc, "SOLUSDT", "EXHAUSTION_MR", "btcdown_volhi", now=T0 + 10_000.0)
    assert hi_unc_dec["uncertainty"] > 0.010
    assert hi_unc_dec["entry_expected_policy_net"] > 0.0
    assert entry_funding_authorized(hi_unc_dec["entry_expected_policy_net"]) is True
    assert al.SIZE_BOUNDS["DAY_V2"][0] <= hi_unc_dec["size_mult"] <= al.SIZE_BOUNDS["DAY_V2"][1]

    # Verify _fund_day_v2_candidate has no uncertainty/confidence/setup/regime gates and canonical cost is 6.60 bps
    fund_src = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    for forbidden in ("uncertainty", "confidence", "MIN_SAMPLES", "VWAP_REVERSION", "btcdown"):
        assert forbidden not in fund_src
    assert pytest.approx(0.00066) == ESTIMATED_ROUNDTRIP_COST

    # Verify SCALP state is untouched
    scalp_after = al.scalp_decision(db, "BTCUSDT", "VWAP_EMA_RECLAIM", "btcdown_vollo", now=T0)
    assert scalp_after == scalp_before
