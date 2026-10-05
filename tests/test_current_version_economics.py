"""Current-version economics: learned evidence moves DAY and SCALP decisions
continuously, with no sample-count gate, no legacy contamination, causal
markouts and unchanged hard safety."""

from __future__ import annotations

import inspect
import json
import sqlite3
from types import SimpleNamespace

import pytest

import backend.services.adaptive_learning as al
from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services.day_v2.live_signal import ENABLED_SETUPS
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from backend.services.scalp_v2.executable_edge import REJECT_THRESHOLD_PCT, scalp_executable_edge
from backend.services.strategy_version import DAY_STRATEGY_VERSION

DAY = al.DAY_ENGINE
SCALP = al.SCALP_ENGINE
SCALP_KEY = ("ETHUSDT", "VWAP_EMA_RECLAIM", "btcup_volhi")
SPREAD = 0.00002


def _day_obs(db, metric, value, *, symbol="BTCUSDT", setup="BREAKOUT_CONTINUATION", regime="btcup_volhi", version=DAY_STRATEGY_VERSION):
    return al.observe(db, engine=DAY, symbol=symbol, setup=setup, regime=regime, metric=metric, value=value, strategy_version=version)


def _day(db, symbol="BTCUSDT", setup="BREAKOUT_CONTINUATION", regime="btcup_volhi"):
    return al.day_decision(db, symbol, setup, regime)


def _record_day(db, symbol, setup, state, evaluated_at=1_000.0):
    return al.record_candidate(
        db,
        engine=DAY,
        symbol=symbol,
        setup=setup,
        regime="r",
        ref_price=100.0,
        roundtrip_cost=0.0006,
        signaled=state == al.CANDIDATE_QUALIFIED,
        evaluated_at=evaluated_at,
        candidate_state=state,
    )


def _cand(symbol, setup, adaptive):
    sig = SimpleNamespace(setup=setup, structural_anchor=99.0, target_price=103.0, atr_1h=1.0, objective_structural=103.0, regime="neutral", atr=0.4)
    return {"symbol": symbol, "signal": sig, "ask_price": 100.0, "adaptive": adaptive}


def _claim(db, base, residual, version=None):
    return al.learn_claim_label(
        db,
        symbol=SCALP_KEY[0],
        setup=SCALP_KEY[1],
        regime=SCALP_KEY[2],
        strategy_version=version or al.current_strategy_version(SCALP),
        base_edge=base,
        residual=residual,
    )


def _edge_for_base(view, base):
    cost = canonical_roundtrip_cost_pct(spread_pct=SPREAD, buy_impact_pct=0.0, sell_impact_pct=0.0)
    return scalp_executable_edge(view, raw_expected_move_pct=base + cost, spread_pct=SPREAD, impact_pct=0.0, edge_source="STRATEGY_CLAIM")


# --- DAY: continuous learned effects, no sample-count gate --------------------


def test_one_day_outcome_already_moves_size_with_no_count_floor(tmp_path):
    db = str(tmp_path / "t.db")
    assert not hasattr(al, "ABSTAIN_CONFIDENCE_FLOOR")
    cold = _day(db)
    assert cold["expected_net"] == 0.0 and cold["size_mult"] == 1.0 and cold["abstain"] is False
    _day_obs(db, "trade_net", -0.004)
    one = _day(db)
    assert one["n_trade_net"] == pytest.approx(1.0)
    assert one["expected_net"] < 0.0
    assert one["size_mult"] < 1.0
    # Evidence strength, not a trade count, sets the telemetry flag.
    strong = str(tmp_path / "strong.db")
    _day_obs(strong, "trade_net", -0.08)
    assert _day(strong)["abstain"] is True


def test_negative_expectancy_lowers_rank_and_size_then_recovers(tmp_path):
    db = str(tmp_path / "t.db")
    for value in (-0.01, -0.008, -0.012):
        _day_obs(db, "trade_net", value, setup="RANGE_BOUNCE")
    weak = _day(db, setup="RANGE_BOUNCE")
    neutral = _day(db, symbol="ETHUSDT", setup="HTF_TREND_PULLBACK")
    assert weak["size_mult"] < neutral["size_mult"] == 1.0
    ranked = rank_day_candidates(
        [_cand("BTCUSDT", "RANGE_BOUNCE", weak), _cand("ETHUSDT", "RANGE_BOUNCE", neutral)],
        ["BTCUSDT", "ETHUSDT"],
        0.0006,
    )
    assert [c["symbol"] for c in ranked] == ["ETHUSDT", "BTCUSDT"]
    assert ranked[1]["rank"]["expected_net"] == pytest.approx(weak["expected_net"])
    for _ in range(12):
        _day_obs(db, "trade_net", 0.012, setup="RANGE_BOUNCE")
    recovered = _day(db, setup="RANGE_BOUNCE")
    assert recovered["expected_net"] > 0.0
    assert recovered["size_mult"] > 1.0


def test_day_setups_stay_available_under_negative_evidence(tmp_path):
    db = str(tmp_path / "t.db")
    lo, _hi = al.SIZE_BOUNDS[DAY]
    candidates = []
    for setup in sorted(ENABLED_SETUPS):
        for _ in range(20):
            _day_obs(db, "trade_net", -0.03, setup=setup)
        decision = _day(db, setup=setup)
        assert decision["abstain_live_veto"] is False
        assert decision["size_mult"] >= lo
        candidates.append(_cand("BTCUSDT", setup, decision))
    assert len(rank_day_candidates(candidates, ["BTCUSDT"], 0.0006)) == len(ENABLED_SETUPS)


def test_realized_and_counterfactual_learning_stay_separate(tmp_path):
    db = str(tmp_path / "t.db")
    assert al.learn_from_close(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="RANGE_BOUNCE",
        regime="r",
        strategy_version=DAY_STRATEGY_VERSION,
        net_pct=-0.005,
        mfe_pct=0.002,
        mae_pct=-0.006,
        hold_min=50.0,
        continuation=0.1,
        version_current=True,
        is_dust=False,
    )
    _record_day(db, "ETHUSDT", "RANGE_BOUNCE", al.CANDIDATE_QUALIFIED)
    assert al.resolve_markouts(db, lambda _s, _t: 101.0, now=1_000.0 + 400 * 60) == 1
    with sqlite3.connect(db) as conn:
        by_symbol: dict[str, set[str]] = {}
        for symbol, metric in conn.execute("SELECT symbol, metric FROM adaptive_metric_state WHERE engine_id='DAY_V2'"):
            by_symbol.setdefault(symbol, set()).add(metric)
    assert "trade_net" in by_symbol["BTCUSDT"] and "markout_forward" not in by_symbol["BTCUSDT"]
    assert "markout_forward" in by_symbol["ETHUSDT"] and "trade_net" not in by_symbol["ETHUSDT"]


def _paper_trades(db, rows):
    from backend.services.strategy_version import engine_versions

    current = engine_versions(DAY)
    cols = ("trade_id", "decision_id", "engine_id", "symbol", "side", "timestamp", "entry_timestamp", "entry_price", "price", "exit_reason", "adaptive_decision_json", *current)
    with sqlite3.connect(db) as conn:
        conn.execute(f"CREATE TABLE paper_trades ({', '.join(cols)})")
        for row in rows:
            values = {**dict.fromkeys(cols, ""), "engine_id": DAY, **current, **row}
            conn.execute(f"INSERT INTO paper_trades ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(values[c] for c in cols))


def test_trade_net_seed_replays_only_current_post_anchor_closes_once(tmp_path):
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST

    db = str(tmp_path / "t.db")
    keys = json.dumps({"setup": "BREAKOUT_CONTINUATION", "regime": "btcup_volhi"})
    sell = {"side": "SELL", "symbol": "BTC/USDT", "entry_price": 100.0, "price": 99.0, "exit_reason": "STOP_LOSS_EXIT", "timestamp": "2026-10-05T14:34:01"}
    _paper_trades(
        db,
        [
            {"trade_id": "b1", "decision_id": "d1", "side": "BUY", "symbol": "BTC/USDT", "timestamp": "2026-10-05T14:16:08", "adaptive_decision_json": keys},
            {**sell, "trade_id": "s1", "decision_id": "d1", "entry_timestamp": "2026-10-05T14:16:08"},
            {"trade_id": "b0", "decision_id": "d0", "side": "BUY", "symbol": "BTC/USDT", "timestamp": "2026-10-05T01:30:10", "adaptive_decision_json": keys},
            {**sell, "trade_id": "s0", "decision_id": "d0", "entry_timestamp": "2026-10-05T01:30:10"},
            {**sell, "trade_id": "s2", "decision_id": "d1", "entry_timestamp": "2026-10-05T14:16:08", "strategy_version": "legacy"},
            {**sell, "trade_id": "s3", "decision_id": "d1", "entry_timestamp": "2026-10-05T14:16:08", "exit_reason": "DUST_WRITEOFF"},
        ],
    )
    dry = al.seed_day_trade_net(db, "2026-10-05T02:02:29")
    assert [t["trade_id"] for t in dry["trades"]] == ["s1"] and dry["applied"] is False
    assert _day(db)["n_trade_net"] == 0.0
    applied = al.seed_day_trade_net(db, "2026-10-05T02:02:29", apply=True)
    assert applied["applied"] is True
    with sqlite3.connect(db) as conn:
        n, ewma = conn.execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()
    assert (n, ewma) == (1.0, pytest.approx(-0.01 - ESTIMATED_ROUNDTRIP_COST))
    again = al.seed_day_trade_net(db, "2026-10-05T02:02:29", apply=True)
    assert again["refused"] == "DAY_TRADE_NET_STATE_EXISTS"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT n FROM adaptive_metric_state WHERE metric='trade_net'").fetchone()[0] == 1.0
    learned = _day(db)
    assert learned["expected_net"] < 0.0 and learned["size_mult"] < 1.0


# --- DAY candidate markouts -----------------------------------------------------


def test_near_qualified_markouts_persist_but_never_train(tmp_path):
    db = str(tmp_path / "t.db")
    rid = _record_day(db, "SOLUSDT", "RANGE_BOUNCE", al.CANDIDATE_NEAR_QUALIFIED)
    assert isinstance(rid, int)
    assert al.resolve_markouts(db, lambda _s, _t: 102.0, now=1_000.0 + 400 * 60) == 0
    with sqlite3.connect(db) as conn:
        resolved, learned, marks, state = conn.execute("SELECT resolved, learned, markouts_json, candidate_state FROM adaptive_candidate_markouts WHERE id=?", (rid,)).fetchone()
        n_state = conn.execute("SELECT COUNT(*) FROM adaptive_metric_state").fetchone()[0]
    assert (resolved, learned, state) == (1, 0, al.CANDIDATE_NEAR_QUALIFIED)
    assert json.loads(marks)["360"] == pytest.approx(0.02 - 0.0006)
    assert n_state == 0


def test_candidate_markout_report_separates_selected_rejected_and_near(tmp_path):
    db = str(tmp_path / "t.db")
    t0 = 1_000.0
    for symbol, state in (("BTCUSDT", al.CANDIDATE_QUALIFIED), ("ETHUSDT", al.CANDIDATE_QUALIFIED), ("SOLUSDT", al.CANDIDATE_NEAR_QUALIFIED)):
        _record_day(db, symbol, "HTF_TREND_PULLBACK", state, evaluated_at=t0)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE day_v2_decisions (symbol TEXT, cycle_ts REAL, result TEXT)")
        conn.execute("INSERT INTO day_v2_decisions VALUES ('BTC/USDT', ?, 'FILLED')", (t0,))
        conn.execute("INSERT INTO day_v2_decisions VALUES ('ETHUSDT', ?, 'REJECTED:SLEEVE_FULL')", (t0,))
    al.resolve_markouts(db, lambda _s, _t: 101.0, now=t0 + 400 * 60)
    report = al.day_candidate_markout_report(db, window_days=1e6)
    states = report["states"]
    assert set(states) == {"SELECTED", "REJECTED", "NEAR_QUALIFIED"}
    for state in states.values():
        assert state["HTF_TREND_PULLBACK"]["60"]["n"] == 1
        assert state["HTF_TREND_PULLBACK"]["60"]["mean"] == pytest.approx(0.01 - 0.0006)


def test_candidate_markouts_are_causal(tmp_path):
    db = str(tmp_path / "t.db")
    asked: list[float] = []

    def quote(_symbol, ts):
        asked.append(ts)
        return 101.0

    t0 = 10_000.0
    _record_day(db, "BTCUSDT", "HTF_TREND_PULLBACK", al.CANDIDATE_QUALIFIED, evaluated_at=t0)
    assert al.resolve_markouts(db, quote, now=t0 + 20 * 60) == 0
    assert asked == [t0 + 15 * 60]
    with sqlite3.connect(db) as conn:
        marks = json.loads(conn.execute("SELECT markouts_json FROM adaptive_candidate_markouts").fetchone()[0])
        n_state = conn.execute("SELECT COUNT(*) FROM adaptive_metric_state").fetchone()[0]
    assert set(marks) == {"15"}
    assert n_state == 0


def test_near_qualified_recording_is_wired_and_bounded():
    src = inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)
    assert "CANDIDATE_NEAR_QUALIFIED" in src and "len(unmet) == 1" in src and "closest in ENABLED_SETUPS" in src
    assert "candidate_state=CANDIDATE_QUALIFIED" in src


# --- SCALP: claim calibration, learning without fills ---------------------------


def test_cold_claim_capture_changes_nothing(tmp_path):
    view = al.scalp_decision(str(tmp_path / "c.db"), *SCALP_KEY)
    assert view["claim_capture"] == 1.0 and view["n_claim"] == 0.0
    edge = _edge_for_base(view, 0.003)
    assert edge.claim_capture == 1.0 and edge.claim_residual_pct == 0.0
    assert edge.final_executable_edge_pct == pytest.approx(edge.base_executable_edge_pct + edge.adaptive_residual_pct + edge.micro_residual_pct)


def test_uninformative_claims_lose_their_size_advantage(tmp_path):
    db = str(tmp_path / "t.db")
    version = al.current_strategy_version(SCALP)
    for i in range(60):
        base = 0.0005 + 0.0001 * (i % 10)
        residual = -0.0005 - base
        assert _claim(db, base, residual)
        al.observe(db, engine=SCALP, symbol=SCALP_KEY[0], setup=SCALP_KEY[1], regime=SCALP_KEY[2], metric="edge_residual_strategy", value=residual, strategy_version=version)
    view = al.scalp_decision(db, *SCALP_KEY)
    assert view["claim_capture"] < 0.2
    assert view["claim_base_mean"] == pytest.approx(0.00095)
    outsized = _edge_for_base(view, 0.0030)
    assert outsized.base_executable_edge_pct + outsized.adaptive_residual_pct > 0.0
    assert outsized.claim_residual_pct < 0.0
    assert outsized.final_executable_edge_pct <= REJECT_THRESHOLD_PCT
    small = _edge_for_base(view, 0.0005)
    assert small.claim_residual_pct == 0.0
    assert small.final_executable_edge_pct <= REJECT_THRESHOLD_PCT


def test_informative_claims_keep_capture_and_recover(tmp_path):
    db = str(tmp_path / "t.db")
    for i in range(60):
        assert _claim(db, 0.0005 + 0.0001 * (i % 10), -0.0002)
    assert al.scalp_decision(db, *SCALP_KEY)["claim_capture"] == pytest.approx(1.0)
    noisy = str(tmp_path / "noisy.db")
    for i in range(30):
        base = 0.0005 + 0.0001 * (i % 10)
        _claim(noisy, base, -0.0005 - base)
    low = al.scalp_decision(noisy, *SCALP_KEY)["claim_capture"]
    for i in range(300):
        _claim(noisy, 0.0005 + 0.0001 * (i % 10), -0.0002)
    assert al.scalp_decision(noisy, *SCALP_KEY)["claim_capture"] > low


def test_scalp_learns_from_unfilled_claims_with_decision_time_inputs_only(tmp_path):
    db = str(tmp_path / "t.db")
    t0 = 1_000.0
    rid = al.record_candidate(
        db,
        engine=SCALP,
        symbol=SCALP_KEY[0],
        setup=SCALP_KEY[1],
        regime=SCALP_KEY[2],
        ref_price=100.0,
        roundtrip_cost=0.0007,
        signaled=False,
        evaluated_at=t0,
        raw_expected_move=0.003,
        raw_move_source="STRATEGY_CLAIM",
    )
    query = "SELECT features_json, raw_expected_move, roundtrip_cost, ref_price, evaluated_at FROM adaptive_candidate_markouts WHERE id=?"
    with sqlite3.connect(db) as conn:
        before = conn.execute(query, (rid,)).fetchone()
    assert al.resolve_markouts(db, lambda _s, _t: 100.1, now=t0 + 2000) == 1
    with sqlite3.connect(db) as conn:
        after = conn.execute(query, (rid,)).fetchone()
        state = dict(conn.execute("SELECT metric, ewma FROM adaptive_metric_state WHERE engine_id='SCALP_V2'").fetchall())
    assert after == before
    base = 0.003 - 0.0007
    realized = 0.001 - 0.0007
    assert state["claim_base_edge"] == pytest.approx(base)
    assert state["claim_residual"] == pytest.approx(realized - base)
    assert state["edge_residual_strategy"] == pytest.approx(realized - base)
    assert al.scalp_decision(db, *SCALP_KEY)["n_claim"] == pytest.approx(1.0)


def test_scalp_breaker_blocks_entries_without_stopping_learning():
    src = inspect.getsource(PortfolioEngineIntegration._process_scalp_v2_signals)
    warning = src.index("SCALP_V2_BREAKER halt=True")
    assert "return" not in src[warning : src.index("from backend.services.portfolio_engine import SCALP_MAX_OPEN_POSITIONS")]
    loop = src[src.index("for sym_raw in sorted(products") :]
    halt = loop.index("if breaker.halt:")
    assert loop.index('if result_code != "ARMED":') < halt < loop.index("existing_pos = self.engine._find_position(")
    assert halt < loop.index("arm_opportunity(") < loop.index("execute_scalp_v2_buy_live(")
    halted_branch = loop[halt : loop.index("existing_pos = self.engine._find_position(")]
    assert 'f"REJECTED:{halt_reason}"' in halted_branch and "continue" in halted_branch
    before_loop = src[: src.index("for sym_raw in sorted(products")]
    assert before_loop.index("check_scalp_loss_breaker(") < before_loop.index("resolve_markouts,") < before_loop.index("record_candidate(")


def test_scalp_stays_enabled_and_admission_is_economic_only():
    assert REJECT_THRESHOLD_PCT == 0.0
    import backend.services.scalp_v2.executable_edge as ee

    for module in (al, ee):
        assert "SCALP_LIVE" not in inspect.getsource(module)
    scalp_loop = inspect.getsource(PortfolioEngineIntegration._process_scalp_v2_signals)
    assert "record_candidate(" in scalp_loop and "LEARNED_NEGATIVE_EDGE" not in scalp_loop


# --- Version isolation, obsolete tables, hard safety ----------------------------


def test_legacy_versions_never_reach_current_state_or_reports(tmp_path):
    db = str(tmp_path / "t.db")
    assert not al.learn_from_close(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_volhi",
        strategy_version="legacy",
        net_pct=-0.05,
        mfe_pct=0.0,
        mae_pct=-0.05,
        hold_min=30.0,
        continuation=0.0,
        version_current=False,
        is_dust=False,
    )
    assert not _day_obs(db, "trade_net", -0.05, version="legacy")
    assert not _claim(db, 0.001, -0.003, version="legacy")
    assert _day(db)["expected_net"] == 0.0
    _record_day(db, "BTCUSDT", "BREAKOUT_CONTINUATION", al.CANDIDATE_QUALIFIED)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET strategy_version='legacy'")
    assert al.resolve_markouts(db, lambda _s, _t: 99.0, now=1_000.0 + 400 * 60) == 0
    assert al.day_candidate_markout_report(db, window_days=1e6)["states"] == {}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM adaptive_metric_state").fetchone()[0] == 0


def test_obsolete_day_clock_labels_stay_non_authoritative(tmp_path):
    db = str(tmp_path / "t.db")
    _day_obs(db, "trade_net", -0.003, setup="RANGE_BOUNCE")
    clean = _day(db, setup="RANGE_BOUNCE")
    with sqlite3.connect(db) as conn:
        for table in ("day_decision_outcome_labels", "day_clock_v2_outcome_labels"):
            conn.execute(f"CREATE TABLE {table} (symbol TEXT, setup TEXT, net REAL, label INTEGER)")
            conn.execute(f"INSERT INTO {table} VALUES ('BTCUSDT', 'RANGE_BOUNCE', 0.5, 1)")
    assert _day(db, setup="RANGE_BOUNCE") == clean
    sources = (
        inspect.getsource(al),
        inspect.getsource(rank_day_candidates),
        inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals),
        inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate),
    )
    for src in sources:
        assert "day_decision_outcome_labels" not in src and "day_clock_v2_outcome_labels" not in src


def test_hard_safety_is_unchanged():
    from backend.services.day_v2 import config as day_cfg
    from backend.services.scalp_v2 import exit_evaluator as scalp_exit

    assert day_cfg.DAY_V2_CATASTROPHIC_ATR_MULTIPLIER == 3.0
    assert day_cfg.DAY_V2_CATASTROPHIC_ANCHOR_BUFFER_ATR == 1.0
    assert day_cfg.DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED == 3
    assert scalp_exit.SCALP_V2_CATASTROPHIC_PCT == 0.015
    assert al.SIZE_BOUNDS == {DAY: (0.55, 1.35), SCALP: (0.50, 1.25)}
