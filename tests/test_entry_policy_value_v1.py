"""ENTRY_POLICY_VALUE_V1 is the only DAY number that can release capital.

The continuation terminal stays the open-position hold value. It does not
enter the entry estimate, the rank score, or the funding check.
"""

from __future__ import annotations

import inspect
import sqlite3

import pytest

import backend.services.adaptive_learning as al
from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services.adaptive_learning import executable_bid_net
from backend.services.day_v2.day_policy_decision import decide_day_continuation
from backend.services.day_v2.entry_policy_value import (
    ENTRY_POLICY_ENGINE_NET_V2,
    ENTRY_POLICY_NET_V2,
    ENTRY_POLICY_VALUE_V2,
    ENTRY_POLICY_VALUE_V3,
    apply_resolved_entry_policy_labels,
    causal_replay,
    entry_funding_authorized,
    entry_policy_value,
    record_entry_policy_observation,
    shrinkage_strength,
)
from backend.services.day_v2.live_signal import ENABLED_SETUPS
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.exact_policy_capture import ensure_capture_schema
from backend.services.policy_episode import UNIVERSE
from backend.services.portfolio_engine import DAY_MAX_OPEN_POSITIONS, SCALP_MAX_OPEN_POSITIONS
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from tests.day_entry_value_support import teach_entry, teach_policy

KEY = ("BTCUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo")
T0 = 1_790_000_000.0


def _signal(setup: str):
    from types import SimpleNamespace

    return SimpleNamespace(setup=setup, atr_1h=1.0, objective_structural=0.0)


def _candidate(symbol: str, policy: float, continuation: float, setup: str = "RANGE_BOUNCE") -> dict:
    return {
        "symbol": symbol,
        "signal": _signal(setup),
        "ask_price": 100.0,
        "adaptive": {
            "entry_expected_policy_net": policy,
            "expected_net": continuation,
            "continuation_terminal_net": continuation,
            "size_mult": 1.0,
        },
    }


def test_continuation_terminal_cannot_enter_the_policy_estimate(tmp_path):
    db = str(tmp_path / "t.db")
    teach_entry(db, *KEY, 0.02, times=4, now=T0)
    decision = al.day_decision(db, *KEY, now=T0)
    assert decision["continuation_terminal_net"] > 0.0
    assert decision["entry_expected_policy_net"] == pytest.approx(-canonical_roundtrip_cost_pct())
    assert decision["expected_net"] == decision["entry_expected_policy_net"]
    assert decision["economic"]["policy_value"] == decision["entry_expected_policy_net"]
    assert decision["economic"]["policy_base"] == ENTRY_POLICY_VALUE_V3


def test_continuation_terminal_cannot_release_capital():
    source = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert "day_entry_value" not in source
    assert "continuation_terminal" not in source
    assert "entry_funding_authorized" in source
    assert "__MARKET" not in source
    assert "detector_fired" not in source
    assert entry_funding_authorized(0.0) is False
    assert entry_funding_authorized(-0.001) is False
    assert entry_funding_authorized(0.001) is True


def test_strict_and_market_keys_both_remain_eligible(tmp_path):
    db = str(tmp_path / "t.db")
    strict = al.day_learned_setup("VWAP_REVERSION", detector_fired=True)
    market = al.day_learned_setup("VWAP_REVERSION", detector_fired=False)
    assert strict == "VWAP_REVERSION"
    assert market.endswith("__MARKET")
    teach_policy(db, "ETHUSDT", strict, "btcdown_vollo", 0.01, times=3, now=T0)
    teach_policy(db, "ETHUSDT", market, "btcdown_vollo", 0.01, times=3, now=T0)
    fired = al.day_decision(db, "ETHUSDT", strict, "btcdown_vollo", now=T0)
    quiet = al.day_decision(db, "ETHUSDT", market, "btcdown_vollo", now=T0)
    assert fired["entry_expected_policy_net"] > 0.0
    assert quiet["entry_expected_policy_net"] > 0.0
    assert frozenset({"HTF_TREND_PULLBACK", "BREAKOUT_CONTINUATION", "RANGE_BOUNCE", "VWAP_REVERSION", "EXHAUSTION_MR"}) == ENABLED_SETUPS


def test_low_count_shrinks_and_does_not_reject(tmp_path):
    db = str(tmp_path / "t.db")
    teach_policy(db, *KEY, 0.01, times=1, now=T0)
    view = entry_policy_value(db, *KEY, now=T0)
    view_v3 = entry_policy_value(db, *KEY, now=T0, version=ENTRY_POLICY_VALUE_V3)
    assert shrinkage_strength() > 1.0
    assert 0.0 < view["mean"] < 0.01
    assert 0.0 < view_v3["mean"] < 0.01
    assert view["n"] > 0.0
    decision = al.day_decision(db, *KEY, now=T0)
    assert decision["entry_expected_policy_net"] == pytest.approx(view_v3["mean"])
    assert "INSUFFICIENT" not in decision["abstain_reason"]
    source = inspect.getsource(al.day_decision)
    assert "n <" not in source and "MIN_TRADES" not in source


def test_rank_and_size_use_policy_value_not_the_continuation_terminal(tmp_path):
    db = str(tmp_path / "t.db")
    teach_entry(db, "BTCUSDT", "RANGE_BOUNCE", "btcup_vollo", 0.05, times=6, now=T0)
    teach_policy(db, "ETHUSDT", "VWAP_REVERSION", "btcup_vollo", 0.02, times=4, now=T0)
    btc = al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0)
    eth = al.day_decision(db, "ETHUSDT", "VWAP_REVERSION", "btcup_vollo", now=T0)
    assert btc["continuation_terminal_net"] > eth["continuation_terminal_net"]
    assert eth["entry_expected_policy_net"] > btc["entry_expected_policy_net"]
    assert eth["size_mult"] > btc["size_mult"]
    ranked = rank_day_candidates(
        [
            _candidate("BTCUSDT", btc["entry_expected_policy_net"], btc["continuation_terminal_net"], "RANGE_BOUNCE"),
            _candidate("ETHUSDT", eth["entry_expected_policy_net"], eth["continuation_terminal_net"], "VWAP_REVERSION"),
        ],
        ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
        canonical_roundtrip_cost_pct(),
    )
    assert [c["symbol"] for c in ranked] == ["ETHUSDT", "BTCUSDT"]
    assert ranked[0]["rank"]["score"] == pytest.approx(eth["entry_expected_policy_net"])


def test_open_position_continuation_still_uses_the_continuation_terminal(tmp_path):
    db = str(tmp_path / "t.db")
    cost = canonical_roundtrip_cost_pct()
    teach_policy(db, *KEY, 0.02, times=5, now=T0)
    teach_entry(db, *KEY, -0.01, times=4, now=T0)
    decision = decide_day_continuation(
        db,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        entry_price=100.0,
        mark=100.0,
        low=100.0,
        high=100.0,
        prev_net=None,
        age_sec=0.0,
        atr_at_entry=0.0,
        structural_anchor=0.0,
        target_price=0.0,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0,
    )
    assert decision["terminal"] == pytest.approx(al.day_decision(db, *KEY, now=T0)["continuation_terminal_net"])
    assert decision["terminal"] < decision["net"]
    assert decision["action"] == "exit"
    exit_source = inspect.getsource(__import__("backend.services.day_v2.live_exit_evaluator", fromlist=["evaluate_day_v2_exit"]).evaluate_day_v2_exit)
    assert "learned_hold_or_exit" in exit_source
    assert "entry_expected_policy_net" not in exit_source


def test_policy_labels_are_causal_and_cost_is_applied_once(tmp_path):
    db = str(tmp_path / "t.db")
    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    cost = canonical_roundtrip_cost_pct()
    label = executable_bid_net(100.0, 101.0, cost)
    assert label == pytest.approx(0.01 - (cost - 0.00006))
    conn.execute(
        """
        INSERT INTO policy_episodes (
            candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask, exit_bid,
            roundtrip_cost, snapshot_json, status, capture_version, planning_net, label_available_at,
            train_eligible, exit_reason, parity_status
        ) VALUES (1,'COUNTERFACTUAL_POLICY_OUTCOME','DAY_V2','BTCUSDT','RANGE_BOUNCE','btcup_vollo',?,?,?,?,?,'CLOSED','EXACT_POLICY_V1',?,? ,1,'LEARNED_CONTINUATION_EXIT','OK')
        """,
        (T0, 100.0, 101.0, cost, "{}", label, T0 + 100.0),
    )
    conn.commit()
    conn.close()
    assert apply_resolved_entry_policy_labels(db, now=T0 + 50.0) == 0
    assert entry_policy_value(db, "BTCUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0 + 50.0)["mean"] == pytest.approx(0.0)
    assert apply_resolved_entry_policy_labels(db, now=T0 + 100.0) == 1
    learned = entry_policy_value(db, "BTCUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0 + 100.0)
    with sqlite3.connect(db) as check:
        stored = check.execute(
            "SELECT ewma FROM adaptive_metric_state WHERE metric=?",
            (ENTRY_POLICY_NET_V2,),
        ).fetchone()[0]
    assert stored == pytest.approx(label)
    assert 0.0 < learned["mean"] < stored
    assert learned["mean"] != pytest.approx(stored - cost)
    assert apply_resolved_entry_policy_labels(db, now=T0 + 100.0) == 0
    future = {
        "id": 2,
        "symbol": "BTCUSDT",
        "setup": "RANGE_BOUNCE",
        "regime": "btcup_vollo",
        "decided_at": T0 + 50.0,
        "label_available_at": T0 + 100.0,
        "planning_net": label,
        "eligible": True,
        "weight": 1.0,
    }
    scored = causal_replay([future])
    assert scored[0]["prediction"] == pytest.approx(0.0)


def test_same_bar_setups_share_one_path_weight(tmp_path):
    db = str(tmp_path / "t.db")
    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    for cid, setup in ((1, "RANGE_BOUNCE__MARKET"), (2, "VWAP_REVERSION__MARKET")):
        conn.execute(
            """
            INSERT INTO policy_episodes (
                candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask, exit_bid,
                roundtrip_cost, snapshot_json, status, capture_version, planning_net, label_available_at,
                train_eligible, exit_reason, parity_status
            ) VALUES (?,'COUNTERFACTUAL_POLICY_OUTCOME','DAY_V2','SOLUSDT',?,'btcup_vollo',?,?,?,?,?,'CLOSED','EXACT_POLICY_V1',?,?,1,'LEARNED_CONTINUATION_EXIT','OK')
            """,
            (cid, setup, T0, 100.0, 100.0, 0.00066, "{}", 0.002, T0 + 10.0),
        )
    conn.commit()
    conn.close()
    assert apply_resolved_entry_policy_labels(db, now=T0 + 10.0) == 2
    with sqlite3.connect(db) as check:
        eng_weights = [row[0] for row in check.execute("SELECT n FROM adaptive_metric_state WHERE metric=?", (ENTRY_POLICY_ENGINE_NET_V2,))]
        spec_weights = [row[0] for row in check.execute("SELECT n FROM adaptive_metric_state WHERE metric=?", (ENTRY_POLICY_NET_V2,))]
    assert eng_weights == pytest.approx([0.5, 0.5])
    assert spec_weights == pytest.approx([1.0, 1.0])


def test_universe_and_slot_caps_stay():
    assert UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert DAY_MAX_OPEN_POSITIONS == 4
    assert SCALP_MAX_OPEN_POSITIONS == 4


def test_a_future_episode_does_not_enter_an_earlier_decision():
    early = {
        "id": 1,
        "symbol": "XRPUSDT",
        "setup": "RANGE_BOUNCE",
        "regime": "btcdown_volhi",
        "decided_at": 1_000.0,
        "label_available_at": 2_000.0,
        "planning_net": -0.01,
        "eligible": True,
        "weight": 1.0,
    }
    later = {**early, "id": 2, "decided_at": 3_000.0, "label_available_at": 4_000.0, "planning_net": -0.02}
    scored = {row["id"]: row for row in causal_replay([later, early])}
    assert scored[1]["prediction"] == pytest.approx(0.0)
    assert scored[2]["prediction"] < 0.0
