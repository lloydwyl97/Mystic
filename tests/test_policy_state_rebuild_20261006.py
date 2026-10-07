"""Policy-aligned learning state: exit-contract isolation, rebuild equivalence, close lineage.

A close teaches entry and continuation state only under the exit policy that
is live now. A rebuild from authoritative rows reproduces the live learner's
state, ``policy_gap`` included. Each DAY / SCALP SELL carries its entry decision
and the continuation reading that closed it, by stable ids.
"""

from __future__ import annotations

import inspect
import json
import math
import sqlite3

import backend.services.adaptive_learning as al
import backend.services.continuation_backfill as cb
import backend.services.economic_state_rebuild as esr
from backend.services.strategy_version import CURRENT_EXIT_AUTHORITIES, current_lifecycle_label, exit_contract_of, exit_policy_anchor

DAY = al.DAY_ENGINE
SCALP = al.SCALP_ENGINE
COST = 0.0006
ANCHOR = float(exit_policy_anchor(SCALP)["epoch"])
KEY = ("ETHUSDT", "RANGE_BOUNCE_SCALP", "btcflat_vollo")


def _close(db, engine, net, *, entered_at, exit_reason, opp="", candidate_id=None, key=KEY, now=None, marks=None):
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
        hold_min=1.0,
        continuation=None,
        version_current=True,
        is_dust=False,
        entered_at=entered_at,
        now=now if now is not None else entered_at + 60.0,
        unrealized_marks=marks,
        opportunity_id=opp,
        exit_reason=exit_reason,
        candidate_id=candidate_id,
    )


def _n(db, metric):
    with sqlite3.connect(db) as conn:
        try:
            row = conn.execute("SELECT COALESCE(SUM(n), 0) FROM adaptive_metric_state WHERE metric LIKE ?", (metric,)).fetchone()
        except sqlite3.Error:
            return 0.0
    return float(row[0] or 0.0)


# --- exit contract ---------------------------------------------------------------


def test_exit_contract_classifies_current_retired_and_unknown():
    for engine in (DAY, SCALP):
        anchor = float(exit_policy_anchor(engine)["epoch"])
        for reason in CURRENT_EXIT_AUTHORITIES[engine]:
            assert exit_contract_of(engine, entered_at=anchor + 1, exit_reason=reason) == "CURRENT"
            assert exit_contract_of(engine, entered_at=anchor - 1, exit_reason=reason) == "RETIRED"
        for retired in ("TIME_STOP_EXIT", "NET_PROFIT_EXIT", "TRAILING_STOP_EXIT", "WINNER_PROTECTION", "STRUCTURAL_INVALIDATION"):
            assert exit_contract_of(engine, entered_at=anchor + 1, exit_reason=retired) == "RETIRED"
        assert exit_contract_of(engine, entered_at=None, exit_reason="LEARNED_CONTINUATION_EXIT") == "UNKNOWN"
        assert exit_contract_of(engine, entered_at=anchor + 1, exit_reason="") == "UNKNOWN"
    assert current_lifecycle_label("DAY_V2_CATASTROPHIC_PROTECTION") and current_lifecycle_label("HORIZON_MARK")
    assert not current_lifecycle_label("OBJECTIVE_COMPLETE") and not current_lifecycle_label("WINNER_PROTECTION")


def test_retired_contract_close_teaches_nothing_current_close_teaches(tmp_path):
    db = str(tmp_path / "t.db")
    for reason, entered in (("TIME_STOP_EXIT", ANCHOR + 60), ("LEARNED_CONTINUATION_EXIT", ANCHOR - 60), ("STOP_LOSS_EXIT", ANCHOR - 600)):
        assert _close(db, SCALP, -0.003, entered_at=entered, exit_reason=reason, marks=[-0.001, -0.002]) is False
    assert _n(db, "trade_%") == 0.0 and _n(db, "hold_remaining_%") == 0.0 and _n(db, "policy_gap") == 0.0
    assert _close(db, SCALP, -0.003, entered_at=ANCHOR + 60, exit_reason="LEARNED_CONTINUATION_EXIT", marks=[-0.001]) is True
    assert _n(db, "trade_net") > 0 and _n(db, "hold_remaining_%") > 0


def test_retired_day_lifecycle_label_never_reaches_lifecycle_net(tmp_path):
    db = str(tmp_path / "t.db")
    row = al.record_candidate(db, engine=DAY, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="r", ref_price=100.0, roundtrip_cost=COST, signaled=True, evaluated_at=ANCHOR + 60)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?",
            (json.dumps({"lifecycle": {"net": 0.02, "reason": "WINNER_PROTECTION", "minutes": 90}}), row),
        )
    assert al._market_label(DAY, {"lifecycle_json": ""}, {"lifecycle": {"net": 0.02, "reason": "WINNER_PROTECTION"}}) is None
    assert al._market_label(DAY, {"lifecycle_json": ""}, {"lifecycle": {"net": 0.02, "reason": "HORIZON_MARK"}}) == 0.02
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=ANCHOR + 86400)
    assert _n(db, "lifecycle_net") == 0.0


# --- policy gap: own candidate, repeated opportunity, two-sided, no floor ---------


def _claim_row(db, evaluated_at, opp):
    row = al.record_candidate(
        db,
        engine=SCALP,
        symbol=KEY[0],
        setup=KEY[1],
        regime=KEY[2],
        ref_price=100.0,
        roundtrip_cost=COST,
        signaled=True,
        evaluated_at=evaluated_at,
        raw_expected_move=0.002,
        raw_move_source="STRATEGY_CLAIM",
    )
    assert al.link_candidate_fill(db, row, opp)
    return row


def test_repeated_opportunity_close_learns_its_own_fill(tmp_path):
    db = str(tmp_path / "t.db")
    t0 = ANCHOR + 3600
    first = _claim_row(db, t0, "SAME_OPP")
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=t0 + 3000)
    assert _close(db, SCALP, -0.002, entered_at=t0 + 1, exit_reason="LEARNED_CONTINUATION_EXIT", opp="SAME_OPP", now=t0 + 3100)
    after_first = _n(db, "policy_gap")
    second = _claim_row(db, t0 + 10_000, "SAME_OPP")
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=t0 + 13_000)
    # Without the candidate id the newest unresolved fill of the opportunity is used.
    assert _close(db, SCALP, -0.004, entered_at=t0 + 10_001, exit_reason="LEARNED_CONTINUATION_EXIT", opp="SAME_OPP", now=t0 + 13_100)
    assert _n(db, "policy_gap") > after_first
    with sqlite3.connect(db) as conn:
        nets = dict(conn.execute("SELECT id, realized_net FROM adaptive_candidate_markouts WHERE id IN (?, ?)", (first, second)).fetchall())
    assert nets == {first: -0.002, second: -0.004}


def test_candidate_id_links_the_close_to_its_row(tmp_path):
    db = str(tmp_path / "t.db")
    t0 = ANCHOR + 3600
    rows = [_claim_row(db, t0 + i, "OPP_X") for i in range(3)]
    al.resolve_markouts(db, lambda _s, _t: 100.0, now=t0 + 3000)
    assert _close(db, SCALP, 0.003, entered_at=t0 + 2, exit_reason="LEARNED_CONTINUATION_EXIT", opp="OPP_X", candidate_id=rows[1], now=t0 + 3100)
    with sqlite3.connect(db) as conn:
        learned = [r[0] for r in conn.execute("SELECT id FROM adaptive_candidate_markouts WHERE realized_net IS NOT NULL")]
    assert learned == [rows[1]]


def test_policy_gap_recovers_both_ways_without_code_change(tmp_path):
    db = str(tmp_path / "t.db")
    t = ANCHOR + 3600
    gaps = []
    for net in (-0.006, -0.006, 0.008, 0.008, 0.008):
        row = _claim_row(db, t, f"O{t}")
        al.resolve_markouts(db, lambda _s, _t: 100.0, now=t + 3000)
        _close(db, SCALP, net, entered_at=t + 1, exit_reason="LEARNED_CONTINUATION_EXIT", opp=f"O{t}", candidate_id=row, now=t + 3100)
        gaps.append(al.policy_gap(db, SCALP, *KEY)["mean"])
        t += 4000
    assert gaps[0] < 0 and gaps[1] < gaps[0]
    assert gaps[-1] > 0 and gaps[-1] > gaps[1]
    src = inspect.getsource(al._learn_policy_gap) + inspect.getsource(al.record_policy_outcome)
    assert "max(0" not in src.replace(" ", "") and "min_n" not in src


# --- rebuild equivalence ------------------------------------------------------------


def _tables(db):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, engine_id TEXT, symbol TEXT, setup TEXT, strategy_version TEXT, entry_timestamp REAL, "
            "exit_timestamp REAL, net_profit_pct REAL, close_reason TEXT, hold_seconds REAL, extra_json TEXT, indicators_while_holding_json TEXT)"
        )
        conn.execute(
            "CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, trade_id TEXT, decision_id TEXT, engine_id TEXT, symbol TEXT, side TEXT, timestamp TEXT, "
            "scalp_opportunity_id TEXT, adaptive_decision_json TEXT)"
        )


def _iso(epoch):
    import time as _t

    return _t.strftime("%Y-%m-%dT%H:%M:%S", _t.gmtime(epoch))


def _price(_sym, t):
    return 100.0 * (1.0 + 0.002 * math.sin(float(t) / 700.0))


def _live_scalp_history(db):
    """Live order of events: claim, fill, close, then each label at its horizon."""
    _tables(db)
    t = ANCHOR + 3600
    plan = [
        (-0.0011, "LEARNED_CONTINUATION_EXIT", "A"),
        (0.004, "LEARNED_CONTINUATION_EXIT", "A"),
        (-0.003, "TIME_STOP_EXIT", "B"),
        (0.002, "LEARNED_CONTINUATION_EXIT", "C"),
        (-0.0012, "LEARNED_CONTINUATION_EXIT", "D"),
    ]
    decision = json.dumps({"setup": KEY[1], "regime": KEY[2]})
    for i, (net, reason, opp) in enumerate(plan):
        row = _claim_row(db, t, opp)
        with sqlite3.connect(db) as conn:
            horizon = float(conn.execute("SELECT label_horizon FROM adaptive_candidate_markouts WHERE id=?", (row,)).fetchone()[0] or 600.0)
        closed = t + 90.0
        _close(db, SCALP, net, entered_at=t + 1, exit_reason=reason, opp=opp, candidate_id=row, now=closed)
        al.resolve_markouts(db, _price, now=t + horizon)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO trade_learning_outcomes (engine_id, symbol, setup, strategy_version, entry_timestamp, exit_timestamp, net_profit_pct, close_reason, "
                "hold_seconds, extra_json, indicators_while_holding_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (SCALP, KEY[0], KEY[1], al.current_strategy_version(SCALP), t + 1, closed, net, reason, 60.0, json.dumps({"version_current": True}), "{}"),
            )
            conn.execute(
                "INSERT INTO paper_trades (trade_id, decision_id, engine_id, symbol, side, timestamp, scalp_opportunity_id, adaptive_decision_json) VALUES (?,?,?,?,?,?,?,?)",
                (f"b{i}", f"d{i}", SCALP, KEY[0], "BUY", _iso(t + 1), opp, decision),
            )
            conn.execute(
                "INSERT INTO paper_trades (trade_id, decision_id, engine_id, symbol, side, timestamp, scalp_opportunity_id, adaptive_decision_json) VALUES (?,?,?,?,?,?,?,?)",
                (f"s{i}", f"d{i}", SCALP, KEY[0], "SELL", _iso(closed), opp, decision),
            )
        t += 5000
    al.resolve_markouts(db, _price, now=t + 5000)
    return t + 5000


def test_scalp_regeneration_reproduces_live_state_including_policy_gap(tmp_path):
    db = str(tmp_path / "t.db")
    now = _live_scalp_history(db)
    assert _n(db, "policy_gap") > 0
    out = esr.regenerate(db, SCALP, now=now, workdir=str(tmp_path))
    assert out["replay"]["closes_current"] == 4 and out["replay"]["closes_retired"] == 1
    assert out["decisions"]["keys"] >= 1
    for field, diff in out["decisions"]["max_abs_diff"].items():
        assert diff < 1e-6, (field, diff)
    for metric in ("policy_gap", "trade_net", "markout_forward"):
        assert abs(out["metrics"][metric]["before_n"] - out["metrics"][metric]["after_n"]) < 1e-3, metric


def test_applied_regeneration_keeps_decisions_and_continuation(tmp_path):
    db = str(tmp_path / "t.db")
    now = _live_scalp_history(db)
    sv = al.current_strategy_version(SCALP)
    al.observe(db, engine=SCALP, symbol=KEY[0], setup=KEY[1], regime=KEY[2], metric="hold_remaining_up", value=0.001, strategy_version=sv, now=now)
    before = esr.decision_snapshot(db, SCALP, esr._candidate_rows(db, SCALP), now=now)
    hold_before = _n(db, "hold_remaining_%")
    assert esr.regenerate(db, SCALP, now=now, apply=True, workdir=str(tmp_path))["applied"]
    after = esr.decision_snapshot(db, SCALP, esr._candidate_rows(db, SCALP), now=now)
    assert esr.compare_snapshots(before, after)["max_abs_diff"]["policy_gap"] < 1e-6
    assert _n(db, "hold_remaining_%") == hold_before
    assert esr.preserved_metric("hold_adv_300") and esr.preserved_metric("trade_continuation") and not esr.preserved_metric("policy_gap")


def test_rebuilds_learn_policy_gap():
    assert "learn_gap" in inspect.getsource(esr.rebuild_day)
    assert "_scalp_policy_gaps(" in inspect.getsource(esr._scalp_scratch)
    assert "record_policy_outcome" in inspect.getsource(al.learn_from_close)


# --- continuation: retired exit fills do not teach ------------------------------------


def _position(reason, entered):
    pos = cb.Position(
        engine=SCALP,
        trade_id="t1",
        symbol="ETHUSDT",
        setup=KEY[1],
        regime=KEY[2],
        entry_price=100.0,
        entry_time=entered,
        exit_time=entered + 300.0,
        exit_price=99.0,
        exit_reason=reason,
        atr=0.0,
        anchor=0.0,
    )
    pos.snapshots = [cb.Snapshot(time=entered + 60.0, net=-0.002, mark=99.8, mfe=0.0, mae=0.002, hold_sec=60.0, high_water=100.0)]
    pos.closed = True
    return pos


def test_retired_exit_fill_is_dropped_from_continuation_labels():
    retired = _position("TIME_STOP_EXIT", ANCHOR - 3600)
    current = _position("LEARNED_CONTINUATION_EXIT", ANCHOR + 3600)
    assert cb.position_contract(retired) == "RETIRED" and cb.position_contract(current) == "CURRENT"
    old, _ = cb.observations_for(retired, [], as_of=ANCHOR + 86400)
    assert [o.source for o in old] == ["exit_fill"]
    clean, skipped = cb.observations_for(retired, [], as_of=ANCHOR + 86400, current_contract_only=True)
    assert clean == [] and skipped[(SCALP, "retired_exit_contract_fill")] == 1
    kept, _ = cb.observations_for(current, [], as_of=ANCHOR + 86400, current_contract_only=True)
    assert [o.source for o in kept] == ["exit_fill"]


def test_hold_remaining_rebuild_leaves_advantage_engines_alone(tmp_path):
    db = str(tmp_path / "t.db")
    al._connect(db).close()
    assert "skipped" in cb.rebuild_hold_remaining(db, DAY, now=ANCHOR + 1)


# --- close lineage ---------------------------------------------------------------------


def _lineage(engine, decision, raw_reason, entered):
    cont = {
        "learner": {"learning_version": "CONTINUATION_REMAINING_V1"},
        "at": entered + 200,
        "mark_net": -0.001,
        "terminal_net": -0.002,
        "hold_advantage": -0.001,
        "action": "exit",
        "reason": raw_reason,
    }
    return al.close_lineage(
        decision,
        engine=engine,
        symbol="ETH/USDT",
        position_trade_id="BUY1",
        sell_trade_id="SELL1",
        opportunity_id="",
        entry_price=100.0,
        exit_price=100.2,
        fees_usd=0.12,
        net_usd=0.08,
        net_pct=0.0008,
        hold_seconds=240.0,
        raw_exit_reason=raw_reason,
        exit_reason="LEARNED_CONTINUATION_EXIT",
        entered_at=entered,
        closed_at=entered + 240,
        continuation=cont,
        versions={"strategy_version": al.current_strategy_version(engine), "economic_version": al.current_economic_version(engine)},
    )


def test_day_close_lineage_links_entry_continuation_and_result():
    entry = al.entry_lineage(engine=DAY, candidate_id=41, opportunity_id="DOPP", rank_position=1, rank_of=3, rank_score=0.004, size_mult=1.2)
    decision = {
        "setup": "RANGE_BOUNCE",
        "regime": "r",
        "economic_version": al.current_economic_version(DAY),
        "economic": {"market_alpha": 0.003, "policy_gap": -0.005, "policy_value": -0.002},
        "lineage": entry,
    }
    lin = _lineage(DAY, decision, "DAY_V2_LEARNED_CONTINUATION", ANCHOR + 60)
    assert lin["entry"]["candidate_id"] == 41 and lin["opportunity_id"] == "DOPP" and lin["position_trade_id"] == "BUY1"
    e = lin["entry"]
    assert e["market_alpha"] > 0 > e["policy_value"]
    assert abs(e["market_alpha"] + e["policy_gap"] - e["policy_value"]) < 1e-12
    assert (e["rank_position"], e["rank_of"], e["size_mult"]) == (1, 3, 1.2)
    assert lin["continuation"]["action"] == "exit" and lin["continuation"]["learner"]["learning_version"]
    assert lin["exit"]["contract"] == "CURRENT" and lin["exit"]["raw_reason"] == "DAY_V2_LEARNED_CONTINUATION"
    assert abs(lin["realized"]["gross_pct"] - 0.002) < 1e-12 and lin["realized"]["fees_usd"] == 0.12 and lin["realized"]["net_pct"] == 0.0008
    assert lin["versions"]["economic_version"] == al.current_economic_version(DAY) and lin["lineage_version"] == al.CLOSE_LINEAGE_VERSION
    assert "features" not in json.dumps(lin)


def test_scalp_close_lineage_reads_executable_edge_economics():
    entry = al.entry_lineage(engine=SCALP, candidate_id=7, opportunity_id="SOPP", rank_position=2, rank_of=2, rank_score=-0.0004, size_mult=0.8)
    decision = {"setup": KEY[1], "regime": KEY[2], "executable_edge": {"economic": {"market_edge": 0.0009, "policy_gap": -0.0013, "policy_value": -0.0004}}, "lineage": entry}
    lin = _lineage(SCALP, decision, "SCALP_V2_LEARNED_CONTINUATION", ANCHOR - 60)
    assert lin["entry"]["candidate_id"] == 7 and lin["opportunity_id"] == "SOPP"
    assert abs(lin["entry"]["market_alpha"] + lin["entry"]["policy_gap"] - lin["entry"]["policy_value"]) < 1e-12
    assert lin["exit"]["contract"] == "RETIRED"
    stamped = al.with_close_lineage(decision, lin)
    assert stamped["close_lineage"] is lin and stamped["lineage"] == entry


def test_live_sell_rows_persist_close_lineage_and_entries_stamp_it():
    import backend.services.portfolio_engine as pe
    import backend.services.portfolio_engine_integration as pei

    src = inspect.getsource(pe)
    assert "with_close_lineage(" in src and "close_lineage(" in src and "continuation_snapshot(" in src
    assert 'candidate_id=(_adapt_dec.get("lineage")' in src
    integ = inspect.getsource(pei)
    assert integ.count("entry_lineage(") >= 2
    assert "SYNTHETIC" not in inspect.getsource(al.close_lineage).upper()
