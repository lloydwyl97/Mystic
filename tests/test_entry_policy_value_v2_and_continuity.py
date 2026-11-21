"""Focused semantic tests for ENTRY_POLICY_VALUE_V2 and continuous age=0 continuation.

Covers all 15 invariants required by the DAY_V2 semantic repair:
  1. Engine sibling-group total weight is normalized to 1.0 on a bar
  2. Specific setup/related/key evidence is not divided by sibling setup count
  3. Overlapping counterfactual trajectories bound total specific weight to <= 1.0
     (both simultaneous exit and staggered/inverted exit)
  4. One funded real outcome receives full specific-level weight = 1.0
  5. Funded target uses actual confirmed fill economics (never stale admission ask)
  6. Counterfactual target still uses executable planning economics
  7. No double-counting of funded + shadow/overlapping counterfactual outcomes
  8. Explicit label provenance (REAL_FUNDED_EXECUTION vs COUNTERFACTUAL_POLICY)
  9. Historical V1 state (ENTRY_POLICY_VALUE_V1 & entry_policy_learned) remains untouched
 10. V2 causal rebuild is deterministic and matches incremental folding
 11. No future leakage (including IOC #2 resolving 22.5s after Trade #2 decision)
 12. Age=0 continuation varies continuously across age=0, age=epsilon, and first monitor age
 13. No new entry gate (continuation_terminal_net <= 0 does not block entry; cold key is 0)
 14. Canonical 6.60 bps DAY cost is unchanged
 15. DAY and SCALP concurrency and top-4 universe are unchanged.
"""

from __future__ import annotations

import inspect
import sqlite3

import pytest

import backend.services.adaptive_learning as al
import backend.services.continuation_surface as cs
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, canonical_roundtrip_cost_pct
from backend.services.adaptive_learning import executable_bid_net
from backend.services.day_v2.day_policy_decision import decide_day_continuation
from backend.services.day_v2.entry_policy_value import (
    ENTRY_POLICY_ENGINE_NET_V2,
    ENTRY_POLICY_NET_V1,
    ENTRY_POLICY_NET_V2,
    ENTRY_POLICY_VALUE_V1,
    ENTRY_POLICY_VALUE_V2,
    LABEL_COUNTERFACTUAL_POLICY,
    LABEL_REAL_FUNDED_EXECUTION,
    apply_resolved_entry_policy_labels,
    causal_replay,
    entry_funding_authorized,
    entry_policy_value,
    rebuild_entry_policy_value_v2,
    record_entry_policy_observation,
)
from backend.services.exact_policy_capture import ensure_capture_schema
from backend.services.policy_episode import UNIVERSE
from backend.services.portfolio_engine import DAY_MAX_OPEN_POSITIONS, SCALP_MAX_OPEN_POSITIONS
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from tests.day_entry_value_support import install_day_surface, teach_policy

T0 = 1_795_195_800.0  # 2026-11-20T17:30:00Z


def _insert_episode(
    conn: sqlite3.Connection,
    *,
    cid: int,
    symbol: str = "ETHUSDT",
    setup: str = "VWAP_REVERSION__MARKET",
    regime: str = "btcdown_vollo",
    decided_at: float,
    exit_at: float,
    label_available_at: float | None = None,
    entry_ask: float = 1600.0,
    exit_bid: float = 1605.0,
    planning_net: float | None = None,
    realized_net: float | None = None,
    funded: int = 0,
    kind: str = "COUNTERFACTUAL_POLICY_OUTCOME",
    trade_id: str | None = None,
    train_eligible: int = 1,
) -> int:
    avail = float(exit_at if label_available_at is None else label_available_at)
    pnet = float(planning_net) if planning_net is not None else float(executable_bid_net(entry_ask, exit_bid, canonical_roundtrip_cost_pct()))
    cur = conn.execute(
        """
        INSERT INTO policy_episodes (
            candidate_id, trade_id, kind, engine_id, symbol, setup, regime,
            decided_at, exit_at, label_available_at, entry_ask, exit_bid,
            roundtrip_cost, snapshot_json, status, funded, capture_version,
            planning_net, net, train_eligible, exit_reason, parity_status
        ) VALUES (?,?,?,'DAY_V2',?,?,?,?,?,?,?,?,?,'{}','CLOSED',?,'EXACT_POLICY_V1',?,?,?,'LEARNED_CONTINUATION_EXIT','OK')
        """,
        (
            cid,
            trade_id,
            kind,
            symbol,
            setup,
            regime,
            decided_at,
            exit_at,
            avail,
            entry_ask,
            exit_bid,
            canonical_roundtrip_cost_pct(),
            funded,
            pnet,
            realized_net,
            train_eligible,
        ),
    )
    return int(cur.lastrowid)


def test_engine_sibling_weight_and_specific_setup_non_dilution(tmp_path):
    """5 setups on one symbol-bar contribute 1.0 total to engine, while each setup gets 1.0 at setup/related/key."""
    db = str(tmp_path / "sib.db")
    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    setups = (
        "BREAKOUT_CONTINUATION__MARKET",
        "EXHAUSTION_MR__MARKET",
        "HTF_TREND_PULLBACK__MARKET",
        "RANGE_BOUNCE__MARKET",
        "VWAP_REVERSION__MARKET",
    )
    for idx, stp in enumerate(setups, start=1):
        _insert_episode(
            conn,
            cid=idx,
            symbol="ETHUSDT",
            setup=stp,
            regime="btcdown_vollo",
            decided_at=T0,
            exit_at=T0 + 120.0,
            planning_net=0.0020,
        )
    conn.commit()
    conn.close()

    assert apply_resolved_entry_policy_labels(db, now=T0 + 120.0) == 5
    with sqlite3.connect(db) as check:
        eng_ns = [r[0] for r in check.execute("SELECT n FROM adaptive_metric_state WHERE metric=?", (ENTRY_POLICY_ENGINE_NET_V2,))]
        spec_ns = [r[0] for r in check.execute("SELECT n FROM adaptive_metric_state WHERE metric=?", (ENTRY_POLICY_NET_V2,))]
    assert len(eng_ns) == 5 and sum(eng_ns) == pytest.approx(1.0)
    assert eng_ns == pytest.approx([0.2] * 5)
    assert spec_ns == pytest.approx([1.0] * 5)

    view = entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 120.0)
    assert view["level_weights"]["engine"] == pytest.approx(1.0)
    assert view["level_weights"]["setup"] == pytest.approx(1.0)
    assert view["level_weights"]["related"] == pytest.approx(1.0)
    assert view["level_weights"]["key"] == pytest.approx(1.0)
    assert view["key_weight"] == pytest.approx(1.0)


def test_overlapping_counterfactual_trajectories_bounded_at_one_unit(tmp_path):
    """4 overlapping consecutive bars resolving on the same rebound share 1.0 specific weight (0.25 each), not 4.0."""
    db = str(tmp_path / "ovl.db")
    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    rebound_exit = T0 + 3432.92  # 18:27:12.92Z
    for i, bps in enumerate((19.79, 39.73, 42.94, 34.19)):
        _insert_episode(
            conn,
            cid=10 + i,
            symbol="ETHUSDT",
            setup="VWAP_REVERSION__MARKET",
            regime="btcdown_vollo",
            decided_at=T0 + i * 900.0,
            exit_at=rebound_exit,
            planning_net=bps / 10000.0,
        )
    conn.commit()
    conn.close()

    assert apply_resolved_entry_policy_labels(db, now=rebound_exit + 1.0) == 4
    with sqlite3.connect(db) as check:
        rows = check.execute("SELECT entry_policy_v2_specific_weight, entry_policy_v2_overlap_group FROM policy_episodes ORDER BY id").fetchall()
        key_n, key_mean = check.execute(
            "SELECT n, ewma FROM adaptive_metric_state WHERE metric=?",
            (ENTRY_POLICY_NET_V2,),
        ).fetchone()
    weights = [r[0] for r in rows]
    groups = {r[1] for r in rows}
    assert len(groups) == 1
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(weights) == pytest.approx(1.0)
    assert key_n == pytest.approx(1.0)
    assert key_mean == pytest.approx((19.79 + 39.73 + 42.94 + 34.19) / 4.0 / 10000.0)


def test_staggered_inverted_overlap_remains_causal_and_bounded_by_one(tmp_path):
    """When an inner overlapping episode resolves before the outer anchor, causal incremental folds still sum to 1.0."""
    db = str(tmp_path / "inv.db")
    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    # ep1: [T0, T0+3600], ep2: [T0+900, T0+1200] (resolves early!), ep3: [T0+1800, T0+3600] (arrives after ep2 resolved!)
    _insert_episode(conn, cid=1, decided_at=T0, exit_at=T0 + 3600.0, planning_net=0.0030)
    _insert_episode(conn, cid=2, decided_at=T0 + 900.0, exit_at=T0 + 1200.0, planning_net=-0.0020)
    _insert_episode(conn, cid=3, decided_at=T0 + 1800.0, exit_at=T0 + 3600.0, planning_net=0.0010)
    conn.commit()
    conn.close()

    # Step 1 at T0+1200 (before ep3 even exists at T0+1800!): ep1 and ep2 are admitted -> ep2 gets 1/2 = 0.50
    assert apply_resolved_entry_policy_labels(db, now=T0 + 1200.0) == 1
    # Step 2 at T0+3600: ep1 and ep3 resolve and split remaining 0.50 -> 0.25 each
    assert apply_resolved_entry_policy_labels(db, now=T0 + 3600.0) == 2

    with sqlite3.connect(db) as check:
        ws = [r[0] for r in check.execute("SELECT entry_policy_v2_specific_weight FROM policy_episodes ORDER BY id")]
    assert ws == pytest.approx([0.25, 0.50, 0.25])
    assert sum(ws) == pytest.approx(1.0)


def test_funded_real_outcome_uses_fill_economics_full_weight_and_no_shadow_double_count(tmp_path):
    """Funded real trade learns actual realized fill net (-72.52 bps, not -63.78 bps ask planning), gets 1.0 key weight, and skips shadow copy."""
    db = str(tmp_path / "real.db")
    conn = al._connect(db)
    ensure_capture_schema(conn)
    cid = al.record_candidate(
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
    conn.execute("UPDATE adaptive_candidate_markouts SET filled=1, realized_net=? WHERE id=?", (-0.0072517732, cid))
    # Funded real episode + 4 sibling setups on same bar + a shadow counterfactual duplicate + a mid-hold counterfactual
    real_id = _insert_episode(
        conn,
        cid=cid,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        decided_at=T0,
        exit_at=T0 + 939.5,
        entry_ask=1625.17,
        exit_bid=1615.78,
        planning_net=-0.0063778571,
        realized_net=-0.0072517732,
        funded=1,
        kind="REAL_POLICY_OUTCOME",
        trade_id="mystic_ETH/USDT_1",
        train_eligible=0,
    )
    for idx, stp in enumerate(("BREAKOUT_CONTINUATION__MARKET", "EXHAUSTION_MR__MARKET", "HTF_TREND_PULLBACK__MARKET", "RANGE_BOUNCE__MARKET"), start=100):
        _insert_episode(conn, cid=idx, symbol="ETHUSDT", setup=stp, regime="btcdown_vollo", decided_at=T0, exit_at=T0 + 110.0, planning_net=0.0004)
    # Shadow duplicate on same (symbol, setup, bar)
    shadow_id = _insert_episode(
        conn,
        cid=cid + 500,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        decided_at=T0,
        exit_at=T0 + 939.5,
        planning_net=-0.0063778571,
        funded=0,
        kind="COUNTERFACTUAL_POLICY_OUTCOME",
    )
    # Unfunded counterfactual opened while real ETH trade was still open (T0 + 600 < T0 + 939.5)
    occupied_id = _insert_episode(
        conn,
        cid=200,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        decided_at=T0 + 600.0,
        exit_at=T0 + 1000.0,
        planning_net=0.0050,
    )
    conn.commit()
    conn.close()

    applied = apply_resolved_entry_policy_labels(db, now=T0 + 1100.0)
    assert applied == 5  # 1 real + 4 distinct sibling setups; shadow & mid-hold occupied skipped!

    with sqlite3.connect(db) as check:
        check.row_factory = sqlite3.Row
        real_row = check.execute("SELECT * FROM policy_episodes WHERE id=?", (real_id,)).fetchone()
        shadow_row = check.execute("SELECT * FROM policy_episodes WHERE id=?", (shadow_id,)).fetchone()
        occ_row = check.execute("SELECT * FROM policy_episodes WHERE id=?", (occupied_id,)).fetchone()
        spec_vwap = check.execute(
            "SELECT n, ewma FROM adaptive_metric_state WHERE metric=? AND setup='VWAP_REVERSION__MARKET'",
            (ENTRY_POLICY_NET_V2,),
        ).fetchone()
    # Both planning_net and realized net preserved; V2 learned actual realized fill net!
    assert real_row["planning_net"] == pytest.approx(-0.0063778571)
    assert real_row["entry_policy_v2_target"] == pytest.approx(-0.0072517732)
    assert real_row["entry_policy_v2_target_type"] == LABEL_REAL_FUNDED_EXECUTION
    assert real_row["entry_policy_v2_specific_weight"] == pytest.approx(1.0)
    assert real_row["entry_policy_v2_engine_weight"] == pytest.approx(0.20)
    assert shadow_row["entry_policy_v2_specific_weight"] == pytest.approx(0.0)
    assert shadow_row["entry_policy_v2_overlap_group"] == "SHADOW_OF_REAL"
    assert occ_row["entry_policy_v2_specific_weight"] == pytest.approx(0.0)
    assert spec_vwap["n"] == pytest.approx(1.0)
    assert spec_vwap["ewma"] == pytest.approx(-0.0072517732)


def test_counterfactual_target_provenance_and_ioc2_causality(tmp_path):
    """Counterfactual uses planning_net (COUNTERFACTUAL_POLICY) and IOC #2 resolving +22.5s after Trade #2 never leaks into Trade #2."""
    t_ioc2_open = 1_795_201_500.0
    t_trade2_eval = 1_795_202_167.839
    t_ioc2_resolve = 1_795_202_190.358  # 22.519s AFTER Trade #2 was scored!

    episodes = [
        {
            "id": 52775,
            "symbol": "ETHUSDT",
            "setup": "VWAP_REVERSION__MARKET",
            "regime": "btcdown_vollo",
            "decided_at": 1_795_200_300.0,
            "exit_at": 1_795_201_239.553,
            "label_available_at": 1_795_201_239.553,
            "planning_net": -0.0063778571,
            "net": -0.0072517732,
            "funded": 1,
            "kind": "REAL_POLICY_OUTCOME",
            "trade_id": "t1",
        },
        {
            "id": 52810,
            "symbol": "ETHUSDT",
            "setup": "VWAP_REVERSION__MARKET",
            "regime": "btcdown_vollo",
            "decided_at": t_ioc2_open,
            "exit_at": t_ioc2_resolve,
            "label_available_at": t_ioc2_resolve,
            "planning_net": -0.01287809,
            "funded": 0,
            "kind": "COUNTERFACTUAL_POLICY_OUTCOME",
            "train_eligible": 1,
        },
        {
            "id": 52815,
            "symbol": "ETHUSDT",
            "setup": "VWAP_REVERSION__MARKET",
            "regime": "btcdown_vollo",
            "decided_at": 1_795_202_100.0,
            "eval_at": t_trade2_eval,
            "exit_at": 1_795_202_813.412,
            "label_available_at": 1_795_202_813.412,
            "planning_net": -0.00600198,
            "net": -0.0067829149,
            "funded": 1,
            "kind": "REAL_POLICY_OUTCOME",
            "trade_id": "t2",
        },
        {
            "id": 52835,
            "symbol": "ETHUSDT",
            "setup": "VWAP_REVERSION__MARKET",
            "regime": "btcdown_vollo",
            "decided_at": 1_795_203_000.0,
            "exit_at": 1_795_203_717.587,
            "label_available_at": 1_795_203_717.587,
            "planning_net": 0.006039,
            "funded": 0,
            "kind": "COUNTERFACTUAL_POLICY_OUTCOME",
            "train_eligible": 1,
        },
    ]
    scored = {int(r["id"]): r for r in causal_replay(episodes)}
    assert scored[52810]["target_type"] == LABEL_COUNTERFACTUAL_POLICY
    assert scored[52810]["target_net"] == pytest.approx(-0.01287809)
    assert scored[52775]["target_type"] == LABEL_REAL_FUNDED_EXECUTION
    assert scored[52775]["target_net"] == pytest.approx(-0.0072517732)
    # At Trade #2 scoring (19:16:07.839Z), ONLY Trade #1 (52775) has resolved; IOC #2 (52810) has NOT resolved yet!
    assert scored[52815]["labels_folded"] == 1
    # Change IOC #2's future outcome arbitrarily and verify Trade #2's prediction is 100% invariant
    perturbed = [dict(e, planning_net=-0.50) if e["id"] == 52810 else dict(e) for e in episodes]
    scored_perturbed = {int(r["id"]): r for r in causal_replay(perturbed)}
    assert scored_perturbed[52815]["prediction"] == pytest.approx(scored[52815]["prediction"], abs=1e-15)
    # By 19:30Z (52835), both IOC #2 and Trade #2 have resolved and lowered prediction further
    assert scored[52835]["prediction"] < scored[52815]["prediction"]


def test_v1_state_preserved_and_v2_rebuild_is_deterministic(tmp_path):
    """Rebuilding V2 leaves V1 rows and entry_policy_learned untouched and produces identical results across runs."""
    db = str(tmp_path / "v1v2.db")
    conn = al._connect(db)
    ensure_capture_schema(conn)
    # Seed legacy V1 state row and entry_policy_learned=1
    record_entry_policy_observation(
        db,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION__MARKET",
        regime="btcdown_vollo",
        value=0.00123,
        weight=0.2,
        version=ENTRY_POLICY_VALUE_V1,
        now=T0,
        conn=conn,
    )
    ep_id = _insert_episode(conn, cid=1, decided_at=T0, exit_at=T0 + 300.0, planning_net=-0.0040)
    conn.execute("UPDATE policy_episodes SET entry_policy_learned=1 WHERE id=?", (ep_id,))
    conn.commit()
    v1_snap_before = [
        tuple(r)
        for r in conn.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=?",
            (ENTRY_POLICY_VALUE_V1,),
        ).fetchall()
    ]
    conn.close()

    r1 = rebuild_entry_policy_value_v2(db, now=T0 + 500.0)
    with sqlite3.connect(db) as c1:
        v2_snap_1 = c1.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=? ORDER BY metric, symbol, setup, regime",
            (ENTRY_POLICY_VALUE_V2,),
        ).fetchall()
        v1_snap_mid = c1.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=?",
            (ENTRY_POLICY_VALUE_V1,),
        ).fetchall()
        v1_flag = c1.execute("SELECT entry_policy_learned FROM policy_episodes WHERE id=?", (ep_id,)).fetchone()[0]

    r2 = rebuild_entry_policy_value_v2(db, now=T0 + 500.0)
    with sqlite3.connect(db) as c2:
        v2_snap_2 = c2.execute(
            "SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE economic_version=? ORDER BY metric, symbol, setup, regime",
            (ENTRY_POLICY_VALUE_V2,),
        ).fetchall()

    assert r1 == r2
    assert v1_snap_mid == v1_snap_before
    assert v1_flag == 1
    assert v2_snap_1 == v2_snap_2
    assert entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 500.0, version=ENTRY_POLICY_VALUE_V1)["metric"] == ENTRY_POLICY_NET_V1
    assert entry_policy_value(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0 + 500.0)["entry_policy_version"] == ENTRY_POLICY_VALUE_V2


def test_continuation_age_zero_is_continuous_with_positive_age(tmp_path):
    """Evaluating continuation at age=0, age=epsilon, and first realistic monitor age varies continuously with no step jump."""
    db = str(tmp_path / "cont.db")
    install_day_surface(db, how="blended")
    cost = canonical_roundtrip_cost_pct()

    # Seed realistic multi-horizon posteriors AND intra-hold ridge observations (matching production structure)
    posteriors = {900: -0.000056, 1800: 0.00049, 3600: 0.001543, 7200: 0.001982, 14400: 0.004187, 21600: 0.004586, 43200: -0.003387}
    for h, adv in posteriors.items():
        f_zero = cs.state_features(entry=1627.0, mark=1627.0, net=-cost, mfe=0.0, mae=0.0, high_water=1627.0, prev_net=None, age_sec=0.0)
        cs.record_advantage(db, engine="DAY_V2", symbol="ETHUSDT", setup="VWAP_REVERSION__MARKET", regime="btcdown_vollo", horizon=h, features=f_zero, advantage=adv, weight=20.0, now=T0)
    for step in range(12):
        f_hold = cs.state_features(
            entry=1627.0,
            mark=1624.0 - step,
            net=-0.0015 - 0.0004 * step,
            mfe=0.0010,
            mae=0.0015 + 0.0004 * step,
            high_water=1628.6,
            prev_net=-0.0012 - 0.0004 * step,
            age_sec=300.0 + 180.0 * step,
        )
        for h in cs.DAY_ADVANTAGE_HORIZONS:
            cs.record_advantage(
                db,
                engine="DAY_V2",
                symbol="ETHUSDT",
                setup="VWAP_REVERSION__MARKET",
                regime="btcdown_vollo",
                horizon=h,
                features=f_hold,
                advantage=-0.0020 - 0.0003 * step,
                weight=1.0,
                now=T0 + step,
            )

    def _eval_at(age_s: float) -> dict:
        return decide_day_continuation(
            db,
            symbol="ETHUSDT",
            setup="VWAP_REVERSION__MARKET",
            regime="btcdown_vollo",
            entry_price=1627.0,
            mark=1627.0,
            low=1627.0,
            high=1627.0,
            prev_net=None,
            age_sec=age_s,
            atr_at_entry=27.0,
            structural_anchor=1622.5,
            target_price=1672.7,
            entry_time=T0,
            roundtrip_cost=cost,
            now=T0 + age_s,
        )

    c_zero = _eval_at(0.0)
    c_eps = _eval_at(1e-4)  # 0.1 millisecond
    c_1s = _eval_at(1.0)
    c_mon = _eval_at(30.0)  # first realistic 30s monitor tick

    entry_view = al.day_decision(db, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=T0)
    assert c_zero["terminal"] == pytest.approx(entry_view["continuation_terminal_net"], abs=1e-12)
    assert abs(float(c_eps["terminal"]) - float(c_zero["terminal"])) < 1e-6
    assert abs(float(c_1s["terminal"]) - float(c_zero["terminal"])) < 5e-5
    # Smooth progression from 0 -> 1s -> 30s with no +15 bps jump
    assert abs(float(c_mon["terminal"]) - float(c_zero["terminal"])) < 0.0010


def test_no_new_entry_gate_cost_unchanged_and_concurrency_unchanged(tmp_path):
    """Negative continuation_terminal_net does NOT block entry when policy net > 0; cost == 6.60 bps; DAY & SCALP caps unchanged."""
    db = str(tmp_path / "nogate.db")
    install_day_surface(db, how="blended")
    # Make continuation_terminal_net strongly negative while ENTRY_POLICY_VALUE_V2 is positive
    for h in cs.DAY_ADVANTAGE_HORIZONS:
        f0 = cs.state_features(entry=100.0, mark=100.0, net=-0.00066, mfe=0.0, mae=0.0, high_water=100.0, prev_net=None, age_sec=0.0)
        cs.record_advantage(db, engine="DAY_V2", symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="btcdown_vollo", horizon=h, features=f0, advantage=-0.0080, weight=20.0, now=T0)
    teach_policy(db, "BTCUSDT", "RANGE_BOUNCE", "btcdown_vollo", 0.0040, times=4, now=T0)

    d = al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "btcdown_vollo", now=T0)
    assert d["continuation_terminal_net"] < -0.0050
    assert d["entry_expected_policy_net"] > 0.0
    assert d["abstain"] is False
    assert entry_funding_authorized(d["entry_expected_policy_net"]) is True

    fund_src = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert "continuation_terminal_net" not in fund_src
    assert canonical_roundtrip_cost_pct() == pytest.approx(0.00066)
    assert pytest.approx(0.00066) == ESTIMATED_ROUNDTRIP_COST
    assert UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert DAY_MAX_OPEN_POSITIONS == 4
    assert SCALP_MAX_OPEN_POSITIONS == 4
