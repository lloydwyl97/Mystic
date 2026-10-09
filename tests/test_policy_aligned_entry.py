"""Entry learns expected realized result under Mystic's own exit policy.

Market labels stay diagnostics. SCALP still adds its policy gap to the
executable edge. DAY entry reads realized trade net, not that gap.
"""

from __future__ import annotations

import inspect
import json
import re
import sqlite3

import backend.services.adaptive_learning as al
from backend.config.day_entry_execution import learned_exit_contract_buy
from backend.services.scalp_v2 import exit_evaluator
from backend.services.scalp_v2.executable_edge import scalp_executable_edge
from backend.services.strategy_version import DAY_STRATEGY_VERSION, exit_policy_anchor

DAY = al.DAY_ENGINE
SCALP = al.SCALP_ENGINE
DAY_KEY = ("BTCUSDT", "RANGE_BOUNCE", "btcdown_vollo")
SCALP_KEY = ("ETHUSDT", "RANGE_BOUNCE_SCALP", "btcflat_vollo")
COST = 0.0006
# Entered under the current exit policy, closed by its learned continuation.
CURRENT_ENTRY = float(exit_policy_anchor(DAY)["epoch"]) + 60.0


def _day_obs(db, metric, value, key=DAY_KEY):
    return al.observe(db, engine=DAY, symbol=key[0], setup=key[1], regime=key[2], metric=metric, value=value, strategy_version=DAY_STRATEGY_VERSION)


def _scalp_claim_row(db, *, evaluated_at=1_000.0, symbol=SCALP_KEY[0]):
    return al.record_candidate(
        db,
        engine=SCALP,
        symbol=symbol,
        setup=SCALP_KEY[1],
        regime=SCALP_KEY[2],
        ref_price=100.0,
        roundtrip_cost=COST,
        signaled=True,
        evaluated_at=evaluated_at,
        raw_expected_move=0.002,
        raw_move_source="STRATEGY_CLAIM",
    )


def _resolve(db, price=100.0, now=10_000.0):
    return al.resolve_markouts(db, lambda _s, _t: price, now=now)


def _close(db, engine, opp, net, key):
    return al.learn_from_close(
        db,
        engine=engine,
        symbol=key[0],
        setup=key[1],
        regime=key[2],
        strategy_version=al.current_strategy_version(engine),
        net_pct=net,
        mfe_pct=None,
        mae_pct=None,
        hold_min=0.0,
        continuation=None,
        version_current=True,
        is_dust=False,
        opportunity_id=opp,
        entered_at=CURRENT_ENTRY,
        exit_reason="LEARNED_CONTINUATION_EXIT",
    )


def _gap(db, engine, key):
    return al.policy_gap(db, engine, *key)


def _filled_day_row(db, opp, lifecycle_net):
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol=DAY_KEY[0],
        setup=DAY_KEY[1],
        regime=DAY_KEY[2],
        ref_price=100.0,
        roundtrip_cost=COST,
        signaled=True,
        evaluated_at=1_000.0,
        candidate_state=al.CANDIDATE_QUALIFIED,
        opportunity_id=opp,
    )
    assert al.mark_candidate_filled(db, row_id)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?",
            (json.dumps({"lifecycle": {"net": lifecycle_net, "reason": "HORIZON_MARK", "minutes": 720}}), row_id),
        )
    return row_id


def _edge(view):
    return scalp_executable_edge(view, raw_expected_move_pct=0.002, spread_pct=0.00002, impact_pct=0.0, edge_source="STRATEGY_CLAIM")


# --- DAY: entry target is policy value, market alpha is kept separately -------


def test_day_entry_target_is_realized_policy_net_not_the_lifecycle_path(tmp_path):
    db = str(tmp_path / "t.db")
    for _ in range(6):
        _day_obs(db, "lifecycle_net", 0.0050)
        _day_obs(db, "policy_gap", 0.0200)
    untouched = al.day_net_expectancy(db, *DAY_KEY)
    assert untouched["mean"] == 0.0
    assert untouched["market_alpha"] > 0.0
    assert untouched["policy_gap"] > 0.0

    for _ in range(4):
        _day_obs(db, "trade_net", -0.0006)
    after = al.day_net_expectancy(db, *DAY_KEY)
    assert after["mean"] < 0.0
    assert abs(after["market_alpha"] - untouched["market_alpha"]) < 1e-12
    assert abs(after["policy_gap"] - untouched["policy_gap"]) < 1e-12
    decision = al.day_decision(db, *DAY_KEY)
    econ = decision["economic"]
    assert econ["policy_base"] == "trade_net"
    assert econ["policy_value"] == decision["expected_net"]
    assert abs(econ["policy_value"] - (econ["uncalibrated_policy_value"] + econ["policy_calibration"])) < 1e-12


def test_day_realized_policy_fill_raises_future_entry_value(tmp_path):
    db = str(tmp_path / "t.db")
    for _ in range(6):
        _day_obs(db, "lifecycle_net", 0.0010)
    before = al.day_net_expectancy(db, *DAY_KEY)["mean"]
    assert before == 0.0
    for _ in range(4):
        _day_obs(db, "trade_net", 0.0060)
    assert al.day_net_expectancy(db, *DAY_KEY)["mean"] > before


# --- SCALP: immediate fee-loss fills lower the next similar opportunity -------


def test_immediate_scalp_fee_loss_lowers_next_entry_then_a_winner_raises_it(tmp_path):
    db = str(tmp_path / "t.db")
    row = _scalp_claim_row(db)
    assert al.link_candidate_fill(db, row, "SOPP1")
    _resolve(db)
    cold_view = al.scalp_decision(db, *SCALP_KEY)
    cold = _edge(cold_view)
    assert cold.policy_gap_pct == 0.0

    # Exit on the next tick at about minus the round-trip cost. The forward
    # markout at the label horizon was flat (net -cost); the policy paid more.
    assert _close(db, SCALP, "SOPP1", -0.0011, SCALP_KEY)
    lowered = _edge(al.scalp_decision(db, *SCALP_KEY))
    assert lowered.policy_gap_pct < 0
    assert lowered.final_executable_edge_pct < cold.final_executable_edge_pct
    assert abs(lowered.market_edge_pct - cold.market_edge_pct) < 1e-12

    row2 = _scalp_claim_row(db, evaluated_at=20_000.0)
    assert al.link_candidate_fill(db, row2, "SOPP2")
    _resolve(db, now=40_000.0)
    assert _close(db, SCALP, "SOPP2", 0.0040, SCALP_KEY)
    raised = _edge(al.scalp_decision(db, *SCALP_KEY))
    assert raised.final_executable_edge_pct > lowered.final_executable_edge_pct


def test_close_before_the_market_label_waits_and_learns_once(tmp_path):
    db = str(tmp_path / "t.db")
    row = _scalp_claim_row(db)
    assert al.link_candidate_fill(db, row, "SOPP3")
    # The close lands before any forward mark is due: nothing is learned yet.
    _close(db, SCALP, "SOPP3", -0.0010, SCALP_KEY)
    assert _gap(db, SCALP, SCALP_KEY)["level_weights"]["engine"] == 0.0
    _resolve(db, now=1_001.0)
    assert _gap(db, SCALP, SCALP_KEY)["level_weights"]["engine"] == 0.0
    _resolve(db)
    learned = _gap(db, SCALP, SCALP_KEY)
    assert learned["level_weights"]["engine"] > 0 and learned["mean"] < 0
    _resolve(db, now=20_000.0)
    assert al.record_policy_outcome(db, engine=SCALP, opportunity_id="SOPP3", net_pct=-0.0010) is False
    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT n FROM adaptive_metric_state WHERE metric='policy_gap'").fetchone()[0]
    assert abs(n - 1.0) < 1e-6


def test_markouts_and_realized_outcomes_are_separate_channels(tmp_path):
    db = str(tmp_path / "t.db")
    row = _scalp_claim_row(db)
    assert al.link_candidate_fill(db, row, "SOPP4")
    _resolve(db)
    market = al.estimate(db, SCALP, *SCALP_KEY, "markout_forward")
    claim = al.scalp_claim_calibration(db, *SCALP_KEY)
    _close(db, SCALP, "SOPP4", -0.0030, SCALP_KEY)
    assert al.estimate(db, SCALP, *SCALP_KEY, "markout_forward")["mean"] == market["mean"]
    assert al.scalp_claim_calibration(db, *SCALP_KEY)["claim_gross_mean"] == claim["claim_gross_mean"]
    assert al.estimate(db, SCALP, *SCALP_KEY, "trade_net")["n"] > 0
    assert _gap(db, SCALP, SCALP_KEY)["mean"] < 0


def test_micro_label_is_trained_against_market_edge_not_policy(tmp_path):
    view = {"claim_gross_setup": 0.001, "claim_gross_mean": 0.001, "claim_capture": 0.0, "micro_residual": 0.0001, "policy_gap": -0.0005}
    edge = _edge(view)
    assert abs(edge.final_executable_edge_pct - (edge.market_edge_pct - 0.0005)) < 1e-12
    assert abs(edge.pre_micro_edge_pct - (edge.market_edge_pct - 0.0001)) < 1e-12
    assert edge.economic()["pre_micro_edge"] == edge.pre_micro_edge_pct


# --- learning shape: two-sided, no floors, all four coins ----------------------


def test_policy_gap_has_no_sample_floor_and_no_one_way_clamp(tmp_path):
    db = str(tmp_path / "t.db")
    sv = al.current_strategy_version(SCALP)
    assert al.observe(db, engine=SCALP, symbol="BTCUSDT", setup="X", regime="r", metric="policy_gap", value=-0.05, strategy_version=sv)
    down = _gap(db, SCALP, ("BTCUSDT", "X", "r"))["mean"]
    assert down < -0.001
    for _ in range(3):
        al.observe(db, engine=SCALP, symbol="BTCUSDT", setup="X", regime="r", metric="policy_gap", value=0.05, strategy_version=sv)
    assert _gap(db, SCALP, ("BTCUSDT", "X", "r"))["mean"] > 0.001


def test_policy_gap_reaches_all_four_coins_and_other_setups(tmp_path):
    db = str(tmp_path / "t.db")
    sv = al.current_strategy_version(DAY)
    al.observe(db, engine=DAY, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="r", metric="policy_gap", value=-0.004, strategy_version=sv)
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        assert _gap(db, DAY, (symbol, "EXHAUSTION_MR__MARKET", "q"))["mean"] < 0


def test_entry_alignment_adds_no_hold_floor_target_or_frequency_rule():
    src = inspect.getsource(al.policy_gap) + inspect.getsource(al.record_policy_outcome) + inspect.getsource(al._learn_policy_gap)
    for banned in ("hold_min", "min_hold", "cooldown", "min_trades", "max_trades", "target_pct", "rsi", "atr", "regime_permission"):
        assert not re.search(rf"\b{banned}\b", src, re.IGNORECASE)
    assert "policy_gap" not in inspect.getsource(exit_evaluator)
    assert "policy_gap" not in inspect.getsource(al.learned_hold_or_exit)
    assert "policy_gap" not in inspect.getsource(al.continuation_terminal)


def test_thesis_gate_and_cooldown_stay_out_of_learned_entries():
    assert learned_exit_contract_buy("", "DAY_V2") and learned_exit_contract_buy("", "SCALP_V2")


# --- seed: causal replay of existing fills, refuses twice --------------------


def _seed_tables(db):
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, engine_id TEXT, symbol TEXT, entry_timestamp REAL, exit_timestamp REAL, net_profit_pct REAL, close_reason TEXT)")
        conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, engine_id TEXT, symbol TEXT, side TEXT, timestamp TEXT, scalp_opportunity_id TEXT)")


def test_seed_policy_gap_learns_existing_fills_once(tmp_path):
    db = str(tmp_path / "t.db")
    _filled_day_row(db, "OPP9", lifecycle_net=0.0100)
    _seed_tables(db)
    anchor = CURRENT_ENTRY
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO trade_learning_outcomes (engine_id, symbol, entry_timestamp, exit_timestamp, net_profit_pct, close_reason) VALUES ('DAY_V2','BTC/USDT',?,?,?, 'LEARNED_CONTINUATION_EXIT')",
            (anchor + 10, anchor + 70, -0.0007),
        )
        conn.execute(
            "INSERT INTO trade_learning_outcomes (engine_id, symbol, entry_timestamp, exit_timestamp, net_profit_pct, close_reason) VALUES ('DAY_V2','BTC/USDT',?,?,?, 'DUST_WRITEOFF')",
            (anchor + 10, anchor + 80, -0.5),
        )
        conn.execute(
            "INSERT INTO paper_trades (engine_id, symbol, side, timestamp, scalp_opportunity_id) VALUES ('DAY_V2','BTC/USDT','SELL', strftime('%Y-%m-%dT%H:%M:%S', ?, 'unixepoch'), 'OPP9')",
            (anchor + 66,),
        )
    dry = al.seed_policy_gap(db, DAY)
    assert dry["refused"] == "" and len(dry["pairs"]) == 1 and not dry["applied"]
    assert abs(dry["pairs"][0]["gap"] - (-0.0007 - 0.0100)) < 1e-12
    assert _gap(db, DAY, DAY_KEY)["level_weights"]["engine"] == 0.0
    assert al.seed_policy_gap(db, DAY, apply=True)["applied"]
    assert _gap(db, DAY, DAY_KEY)["mean"] < 0
    assert al.seed_policy_gap(db, DAY)["refused"] == "POLICY_GAP_STATE_EXISTS"
