"""Current-version production contract: DAY five setups, SCALP short horizon, version isolation."""

from __future__ import annotations

import inspect
import sqlite3
from types import SimpleNamespace

import pytest

from backend.services.day_v2 import live_signal
from backend.services.day_v2.live_signal import (
    BREAKOUT_BUFFER,
    ENABLED_SETUPS,
    evaluate_entry_signal,
    prior_structure_high,
)
from backend.services.day_v2.ranking import clamp_to_sleeve, rank_day_candidates
from backend.services.strategy_version import (
    DAY_ENGINE,
    DAY_EXIT_CONTRACT_VERSION,
    DAY_STRATEGY_VERSION,
    SCALP_ENGINE,
    SCALP_STRATEGY_VERSION,
    engine_versions,
    is_current_version,
    performance_by_version,
    register_version_boundaries,
    stamp_buy_version,
    stamp_sell_version,
)

T0 = 1_780_000_000


def _bars15(closes, lows=None, wick=0.05):
    out = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        low = (lows or {}).get(i - len(closes), min(o, c) - wick)
        out.append({"open": o, "high": max(o, c) + wick, "low": low, "close": c, "ts_epoch": T0 + i * 900})
    return out


def _bars4h(regime, n=15):
    last = {"bull": 103.0, "bear": 97.0, "neutral": 100.0}[regime]
    closes = [100.0] * (n - 1) + [last]
    return [{"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "ts_epoch": T0 - (n - i) * 14400} for i, c in enumerate(closes)]


def _bars1h(bullish, n=20):
    closes = [100 + (0.2 * i if bullish else -0.2 * i) for i in range(n)]
    return [{"open": c, "high": c + 0.3, "low": c - 0.3, "close": c, "ts_epoch": T0 + 59 * 900 - (n - i) * 3600} for i, c in enumerate(closes)]


def _zigzag(n, start, up, down):
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] + (up if i % 2 == 0 else -down))
    return closes


def _scenario(setup):
    if setup == "HTF_TREND_PULLBACK":
        c = _zigzag(60, 110, 0.1, 0.15)
        c = [*c[:-1], c[-2] + 0.1]
        return _bars15(c), _bars1h(True), _bars4h("bull")
    if setup == "RANGE_BOUNCE":
        return _bars15(_zigzag(61, 110, 0.1, 0.25)), _bars1h(False), _bars4h("bear")
    if setup == "BREAKOUT_CONTINUATION":
        c = [100.0 if i % 2 == 0 else 100.3 for i in range(56)] + [100.0, 100.2, 100.4, 101.0]
        return _bars15(c), _bars1h(False), _bars4h("neutral")
    if setup == "VWAP_REVERSION":
        c = _zigzag(61, 110, 0.1, 0.25)
        return _bars15(c, lows={-2: c[-2] * 0.99}), _bars1h(False), _bars4h("neutral")
    if setup == "EXHAUSTION_MR":
        c = [100.0 if i % 2 == 0 else 100.2 for i in range(57)] + [100.0]
        c = [*c, c[-1] * 0.978, c[-1] * 0.978 * 1.008]
        return _bars15(c), _bars1h(False), _bars4h("bear")
    raise AssertionError(setup)


ALL_SETUPS = ("HTF_TREND_PULLBACK", "RANGE_BOUNCE", "BREAKOUT_CONTINUATION", "VWAP_REVERSION", "EXHAUSTION_MR")


# --- A. BREAKOUT uses prior structure and can fire ------------------------------


def test_a_breakout_uses_prior_twenty_closed_bars_and_fires():
    b15, b1h, b4h = _scenario("BREAKOUT_CONTINUATION")
    highs = [b["high"] for b in b15]
    assert prior_structure_high(highs) == max(highs[-21:-1])
    sig = evaluate_entry_signal("BTCUSDT", b15, b1h, b4h)
    assert sig is not None and sig.setup == "BREAKOUT_CONTINUATION"
    assert sig.structural_anchor == pytest.approx(prior_structure_high(highs) * 0.995)


def test_a_breakout_including_signal_bar_was_unreachable_and_prior_level_is_not():
    b15, _, _ = _scenario("BREAKOUT_CONTINUATION")
    c0 = b15[-1]["close"]
    window_with_signal = max(b["high"] for b in b15[-20:])
    assert not c0 > window_with_signal * BREAKOUT_BUFFER
    assert c0 > prior_structure_high([b["high"] for b in b15]) * BREAKOUT_BUFFER


def test_a_breakout_needs_the_buffer_above_prior_high():
    b15, b1h, b4h = _scenario("BREAKOUT_CONTINUATION")
    prior = prior_structure_high([b["high"] for b in b15])
    b15[-1]["close"] = prior * 1.0005
    b15[-1]["high"] = b15[-1]["close"] + 0.05
    sig = evaluate_entry_signal("BTCUSDT", b15, b1h, b4h)
    assert sig is None or sig.setup != "BREAKOUT_CONTINUATION"


# --- B. EXHAUSTION matches its intended definition -------------------------------


def test_b_exhaustion_is_long_reversal_after_downside_spike():
    sig = evaluate_entry_signal("BTCUSDT", *_scenario("EXHAUSTION_MR"))
    assert sig is not None and sig.setup == "EXHAUSTION_MR"
    assert sig.target_price > sig.structural_anchor
    src = inspect.getsource(live_signal._detect_setup)
    assert "downside momentum spike" in src
    assert "b1c < b2c * (1.0 - EXHAUSTION_SPIKE_PCT)" in src and "c0 > b1c" in src


def test_b_up_spike_then_retrace_is_not_exhaustion():
    c = [100.0 if i % 2 == 0 else 100.2 for i in range(57)] + [100.0]
    c = [*c, c[-1] * 1.022, c[-1] * 1.022 * 0.995]
    sig = evaluate_entry_signal("BTCUSDT", _bars15(c), _bars1h(False), _bars4h("bear"))
    assert sig is None or sig.setup != "EXHAUSTION_MR"


# --- C. All five DAY setups participate live -------------------------------------


@pytest.mark.parametrize("setup", ALL_SETUPS)
def test_c_every_setup_fires_through_the_live_signal(setup):
    assert setup in ENABLED_SETUPS
    sig = evaluate_entry_signal("BTCUSDT", *_scenario(setup))
    assert sig is not None and sig.setup == setup
    assert sig.atr_1h > 0


# --- D/E. No trade-count or legacy-PF gate ---------------------------------------


def test_d_e_day_entry_reads_no_trade_history():
    from backend.services import portfolio_engine_integration as pei
    from backend.services.day_v2 import ranking

    assert "db_path" not in inspect.signature(evaluate_entry_signal).parameters
    sources = [
        inspect.getsource(live_signal),
        inspect.getsource(ranking),
        inspect.getsource(pei.PortfolioEngineIntegration._process_day_v2_signals),
        inspect.getsource(pei.PortfolioEngineIntegration._fund_day_v2_candidate),
    ]
    for src in sources:
        for banned in ("profit_factor", "win_rate", "trade_learning_outcomes", "min_trades", "sample_count", "paper_trades"):
            assert banned not in src


def test_d_e_ranking_never_removes_a_candidate():
    sigs = []
    for setup in ALL_SETUPS:
        sig = evaluate_entry_signal("BTCUSDT", *_scenario(setup))
        sigs.append({"symbol": f"C{len(sigs)}USDT", "signal": sig, "ask_price": sig.target_price * 0.99})
    ranked = rank_day_candidates(sigs, [s["symbol"] for s in sigs], 0.0006)
    assert len(ranked) == len(ALL_SETUPS)
    edges = [c["rank"]["executable_objective_edge"] for c in ranked]
    assert edges == sorted(edges, reverse=True)
    assert [c["rank"]["position"] for c in ranked] == list(range(1, len(ALL_SETUPS) + 1))


def test_d_e_legacy_scalp_history_never_penalises_an_arm(tmp_path, monkeypatch):
    from backend.services.binance_scalp import scalp_arm_blocker

    db = str(tmp_path / "t.db")
    _scalp_outcomes(db, [("", -1.0)] * 10)
    stats = scalp_arm_blocker._query_arm_stats("BTCUSDT", "range_bounce_scalp", db_path=db)
    assert stats["n"] == 0


# --- F/G. DAY runner unchanged, small target gone --------------------------------


def test_f_day_runner_contract_unchanged():
    from backend.services.day_v2 import winner_contract as wc

    assert DAY_EXIT_CONTRACT_VERSION == wc.DAY_EXIT_CONTRACT_RUNNER == "DAY_V2_STRUCTURE_RUNNER_V1"
    assert (wc.RUNNER_ACTIVATION_ATR_1H, wc.RUNNER_TRAIL_ATR_1H, wc.RUNNER_TIGHT_TRAIL_ATR_1H) == (1.0, 1.5, 0.75)
    state = wc.runner_stop(entry_price=100.0, highest_price=103.0, atr_1h=1.0, objective=105.0, estimated_roundtrip_cost=0.0006)
    assert state["activated"] and state["stop"] == pytest.approx(101.5)


def test_g_day_has_no_small_scalp_target():
    import time

    from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit

    dec = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=100.9,
        bar_low=100.0,
        highest_price=100.9,
        atr_at_entry=0.4,
        structural_anchor=98.5,
        target_price=100.9,
        entry_time=time.time() - 3600,
        estimated_roundtrip_cost=0.0006,
        setup="HTF_TREND_PULLBACK",
        atr_1h_at_entry=1.0,
        objective_structural=103.0,
    )
    assert dec is None


# --- H/I. SCALP short-horizon lifecycle and hard net-edge safety -----------------


def _scalp_pos(**kw):
    base = {"engine_id": "SCALP_V2", "entry_price": 100.0, "highest_price": 100.0, "lowest_price": 100.0, "symbol": "BTCUSDT"}
    base.update(kw)
    return SimpleNamespace(**base)


def _scalp_exit(net, hold):
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    price = 100.0 * (1 + net + 0.0006)
    return evaluate_scalp_v2_exit(position=_scalp_pos(), current_price=price, net_pnl_pct=net, hold_minutes=hold, bar_low=min(price, 100.0))


def test_h_scalp_exit_is_target_stop_horizon():
    from backend.services.scalp_v2 import exit_calibration as cal
    from backend.services.scalp_v2 import exit_evaluator as ev

    assert cal.SELECTED_EXIT_POLICY == "target_stop_horizon"
    assert ev.SCALP_V2_TIME_STOP_MIN <= 30
    target = cal.scalp_v2_min_net_profit_pct()
    stop = cal.scalp_v2_max_adverse_net_pct()
    assert 0 < stop < target < 0.004
    assert _scalp_exit(target, 3)["reason"] == ev.SCALP_V2_EXIT_NET_PROFIT
    assert _scalp_exit(-stop, 3)["reason"] == ev.SCALP_V2_EXIT_ADVERSE
    assert _scalp_exit(0.001, 5)["action"] == "hold"
    assert _scalp_exit(0.001, ev.SCALP_V2_TIME_STOP_MIN)["reason"] == ev.SCALP_V2_EXIT_TIME_STOP
    assert _scalp_exit(-0.0005, ev.SCALP_V2_TIME_STOP_MIN)["reason"] == ev.SCALP_V2_EXIT_TIME_STOP


def test_h_scalp_target_is_the_entry_economics_target(monkeypatch):
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.scalp_v2.exit_calibration import scalp_v2_min_net_profit_pct

    monkeypatch.delenv("SCALP_V2_MIN_NET_PROFIT_PCT", raising=False)
    monkeypatch.setenv("SCALP_NET_PROFIT_TARGET_PCT", "0.0025")
    assert scalp_v2_min_net_profit_pct() == pytest.approx(ScalpEconomics.from_env().net_profit_target_pct)


def test_h_scalp_exits_are_strategy_exits_not_manual():
    from backend.services.portfolio_engine import ExitType, strategy_exit_type
    from backend.services.scalp_v2 import exit_evaluator as ev

    for trig in (ev.SCALP_V2_EXIT_CATASTROPHIC, ev.SCALP_V2_EXIT_NET_PROFIT, ev.SCALP_V2_EXIT_ADVERSE, ev.SCALP_V2_EXIT_TIME_STOP):
        assert strategy_exit_type(trig) == ExitType.STRATEGY


def test_i_scalp_positive_net_edge_is_a_hard_block():
    from backend.services.binance_scalp import protected_preflight
    from backend.services.binance_scalp.strategies import range_bounce_scalp

    assert "expected_net <= 0 or expected_net < econ.min_net_edge_pct" in inspect.getsource(protected_preflight)
    assert "TARGET_NOT_REACHABLE" in inspect.getsource(range_bounce_scalp)


# --- J/K/L/M. Coins and slots ----------------------------------------------------


def test_j_k_l_m_coins_and_slots():
    from backend.services import portfolio_engine as pe
    from backend.services import two_engine_claim
    from backend.services.day_v2.config import DAY_V2_UNIVERSE

    assert set(DAY_V2_UNIVERSE) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert (pe.DAY_MAX_OPEN_POSITIONS, pe.SCALP_MAX_OPEN_POSITIONS, pe.COMBINED_ENGINE_MAX_POSITIONS) == (4, 4, 8)
    assert two_engine_claim.engine_cap("DAY_V2") == 4 and two_engine_claim.engine_cap("SCALP_V2") == 4


# --- N. Current-version fields persist on new trades -----------------------------


def _paper_trades(db):
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (trade_id TEXT, side TEXT, mode TEXT, pnl REAL, exit_reason TEXT, engine_id TEXT)")


def test_n_buy_and_sell_rows_carry_versions(tmp_path):
    db = str(tmp_path / "t.db")
    _paper_trades(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO paper_trades (trade_id, side, engine_id) VALUES ('b1', 'BUY', 'DAY_V2')")
        conn.execute("INSERT INTO paper_trades (trade_id, side, engine_id) VALUES ('s1', 'SELL', 'DAY_V2')")
        stamp_buy_version(conn, "b1", DAY_ENGINE)
        sold = stamp_sell_version(conn, "s1", "b1", DAY_ENGINE)
        conn.row_factory = sqlite3.Row
        buy = conn.execute("SELECT * FROM paper_trades WHERE trade_id='b1'").fetchone()
        sell = conn.execute("SELECT * FROM paper_trades WHERE trade_id='s1'").fetchone()
    assert buy["strategy_version"] == DAY_STRATEGY_VERSION and buy["code_sha"]
    assert buy["exit_contract_version"] == DAY_EXIT_CONTRACT_VERSION and buy["accounting_contract_version"]
    assert sell["strategy_version"] == DAY_STRATEGY_VERSION and is_current_version(DAY_ENGINE, sell)
    assert sold["version_current"] == "1"


def test_n_learning_row_carries_versions(tmp_path):
    from backend.config.trading_mode import TradingMode
    from backend.services.learning_provenance import learning_provenance
    from backend.services.trade_learning_writer import TradeLearningRecord, record_trade_outcome

    db = str(tmp_path / "t.db")
    _paper_trades(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO paper_trades (trade_id, side, engine_id) VALUES ('b9', 'BUY', 'SCALP_V2')")
        stamp_buy_version(conn, "b9", SCALP_ENGINE)
    prov = learning_provenance(db, SimpleNamespace(engine_id="SCALP_V2", trade_id="b9", entry_thesis="range_bounce_scalp", status="ACTIVE"), "SCALP_V2_NET_PROFIT")
    assert prov["strategy_version"] == SCALP_STRATEGY_VERSION and prov["version_current"] is True
    assert record_trade_outcome(TradeLearningRecord(symbol="BTCUSDT", net_profit_usd=0.1, extra=prov), db_path=db, mode_override=TradingMode.LIVE)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT engine_id, setup, strategy_version, exit_contract_version FROM trade_learning_outcomes").fetchone()
    assert row == ("SCALP_V2", "range_bounce_scalp", SCALP_STRATEGY_VERSION, engine_versions(SCALP_ENGINE)["exit_contract_version"])


def test_n_unversioned_lot_closes_as_legacy(tmp_path):
    db = str(tmp_path / "t.db")
    _paper_trades(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO paper_trades (trade_id, side, engine_id) VALUES ('old', 'BUY', 'DAY_V2')")
        conn.execute("INSERT INTO paper_trades (trade_id, side, engine_id) VALUES ('s2', 'SELL', 'DAY_V2')")
        sold = stamp_sell_version(conn, "s2", "old", DAY_ENGINE)
    assert sold["strategy_version"] == "" and sold["version_current"] == "0"


# --- O. Legacy rows cannot enter current-version learning ------------------------


def _scalp_outcomes(db, rows):
    from backend.services.ai_learning_ingestion import _ensure_scalp_outcomes_table
    from backend.services.strategy_version import SCALP_ENTRY_CONTRACT_VERSION, SCALP_EXIT_CONTRACT_VERSION

    _ensure_scalp_outcomes_table(db)
    with sqlite3.connect(db) as conn:
        for i, (version, pnl) in enumerate(rows):
            current = version == SCALP_STRATEGY_VERSION
            conn.execute(
                """
                INSERT INTO scalp_learning_outcomes (
                    ingested_at, source_id, symbol, setup_name, net_pnl_usd,
                    strategy_version, entry_contract_version, exit_contract_version
                ) VALUES (datetime('now'), ?, 'BTCUSDT', 'range_bounce_scalp', ?, ?, ?, ?)
                """,
                (i + 1, pnl, version, SCALP_ENTRY_CONTRACT_VERSION if current else "", SCALP_EXIT_CONTRACT_VERSION if current else ""),
            )


def test_o_consume_for_ranking_reads_only_current_version(tmp_path):
    import json

    from backend.config.trading_mode import TradingMode
    from backend.services.trade_learning_writer import TradeLearningRecord, consume_setup_outcomes_for_ranking, record_trade_outcome

    db = str(tmp_path / "t.db")
    legacy = {"setup": "range_bounce_scalp", "engine_id": "SCALP_V2"}
    for _ in range(5):
        record_trade_outcome(TradeLearningRecord(symbol="BTCUSDT", entry_price=100.0, exit_price=99.0, quantity=1.0, net_profit_usd=-1.0, extra=legacy), db_path=db, mode_override=TradingMode.LIVE)
    assert consume_setup_outcomes_for_ranking(db, "range_bounce_scalp", engine_id="SCALP_V2")["n"] == 0
    current = {**legacy, **engine_versions(SCALP_ENGINE)}
    record_trade_outcome(TradeLearningRecord(symbol="BTCUSDT", entry_price=100.0, exit_price=100.5, quantity=1.0, net_profit_usd=0.5, extra=current), db_path=db, mode_override=TradingMode.LIVE)
    out = consume_setup_outcomes_for_ranking(db, "range_bounce_scalp", engine_id="SCALP_V2")
    assert out["n"] == 1 and out["wins"] == 1
    assert json.dumps(out)


def test_o_arm_stats_read_only_current_version(tmp_path):
    from backend.services.binance_scalp import scalp_arm_blocker

    db = str(tmp_path / "t.db")
    _scalp_outcomes(db, [("", -1.0)] * 6 + [(SCALP_STRATEGY_VERSION, 0.4)])
    stats = scalp_arm_blocker._query_arm_stats("BTCUSDT", "range_bounce_scalp", db_path=db)
    assert stats["n"] == 1 and stats["wins"] == 1


# --- P. Current-version performance excludes legacy -----------------------------


def test_p_performance_splits_current_and_legacy(tmp_path):
    db = str(tmp_path / "t.db")
    _paper_trades(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO paper_trades (trade_id, side, engine_id) VALUES ('b1', 'BUY', 'DAY_V2')")
        stamp_buy_version(conn, "b1", DAY_ENGINE)
        for tid, pnl, buy in (("s1", 2.0, "b1"), ("s2", -1.0, "b1"), ("s3", -5.0, "none"), ("s4", 3.0, "none")):
            conn.execute("INSERT INTO paper_trades (trade_id, side, mode, pnl, exit_reason, engine_id) VALUES (?, 'SELL', 'live', ?, 'X', 'DAY_V2')", (tid, pnl))
            stamp_sell_version(conn, tid, buy, DAY_ENGINE)
        conn.execute("INSERT INTO paper_trades (trade_id, side, mode, pnl, exit_reason, engine_id) VALUES ('d', 'SELL', 'live', -9, 'DUST_WRITEOFF', 'DAY_V2')")
    register_version_boundaries(db)
    perf = performance_by_version(db)["engines"]["DAY_V2"]
    assert perf["current"]["round_trips"] == 2 and perf["current"]["net_usd"] == pytest.approx(1.0)
    assert perf["current"]["profit_factor"] == pytest.approx(2.0)
    assert perf["legacy"]["round_trips"] == 2 and perf["legacy"]["net_usd"] == pytest.approx(-2.0)
    assert perf["current_version_start_utc"]


def test_p_boundary_is_set_once(tmp_path):
    db = str(tmp_path / "t.db")
    first = register_version_boundaries(db)
    assert register_version_boundaries(db) == first


# --- Q. Hard safety blocks while mode stays LIVE --------------------------------


def test_q_budget_hard_block_does_not_change_mode(tmp_path):
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.two_engine_capital import ENGINE_BUDGET_EXCEEDED, check_engine_budget

    mode_before = get_scalp_config().resolved_structural_mode()
    ok, reason, snap = check_engine_budget(str(tmp_path / "t.db"), "DAY_V2", 500.0, 200.0, 200.0, {})
    assert not ok and reason == ENGINE_BUDGET_EXCEEDED
    assert get_scalp_config().resolved_structural_mode() == mode_before
    qty = clamp_to_sleeve(10.0, 50.0, snap.day.remaining_budget)
    assert qty * 50.0 <= snap.day.remaining_budget
    assert check_engine_budget(str(tmp_path / "t.db"), "DAY_V2", qty * 50.0, 200.0, 200.0, {})[0]


def test_q_sleeve_clamp_never_exceeds_budget_or_request():
    assert clamp_to_sleeve(2.0, 10.0, 100.0) == 2.0
    assert clamp_to_sleeve(2.0, 10.0, 5.0) * 10.0 <= 5.0
    assert clamp_to_sleeve(2.0, 10.0, 0.0) == 0.0


def test_no_shadow_or_checkpoint_runtime():
    from pathlib import Path

    from backend.services.day_v2 import migrations

    start = (Path(__file__).resolve().parents[1] / "start_mystic.sh").read_text()
    assert "scalp_v2_checkpoint_monitor" not in start
    assert "shadow" not in inspect.getsource(migrations).lower()
    from backend.services import portfolio_engine_integration as pei

    assert "process_bar_candidates(" not in inspect.getsource(pei)
