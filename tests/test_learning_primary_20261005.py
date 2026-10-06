"""The learner that actually runs: a realized loss changes the next DAY rank
and size, a rejected lifecycle does too, and a setup name is not an edge.

State features are recorded with the candidate. They do not change the
expected net. A walk-forward that withheld capital whenever that net was
non-positive made the book worse, so it is not a live gate.
"""

from __future__ import annotations

import inspect

import pytest

import backend.services.adaptive_learning as al
import backend.services.portfolio_engine as pe
from backend.services.day_v2.config import DAY_V2_UNIVERSE
from backend.services.day_v2.live_exit_evaluator import DAY_V2_CATASTROPHIC_ATR_MULTIPLIER
from backend.services.day_v2.live_signal import ENABLED_SETUPS, day_state_features
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from backend.services.scalp_v2 import exit_evaluator as scalp_exits

DAY = al.DAY_ENGINE
T0 = 1_791_200_000.0


def _cand(symbol: str, setup: str, adaptive: dict) -> dict:
    from types import SimpleNamespace

    sig = SimpleNamespace(setup=setup, structural_anchor=99.0, target_price=103.0, atr_1h=1.0, objective_structural=103.0, regime="neutral", atr=0.4)
    return {"symbol": symbol, "signal": sig, "ask_price": 100.0, "adaptive": adaptive}


def test_cold_setups_are_neutral_and_the_name_is_not_an_edge(tmp_path):
    db = str(tmp_path / "c.db")
    nets = [al.day_decision(db, "BTCUSDT", setup, "", now=T0)["expected_net"] for setup in sorted(ENABLED_SETUPS)]
    assert nets == [0.0] * len(nets)
    assert "if n <" not in inspect.getsource(al.day_decision)
    for banned in ("HTF bad", "RANGE good"):
        assert banned not in inspect.getsource(al.day_decision)


def test_one_loss_moves_the_next_decision(tmp_path):
    db = str(tmp_path / "o.db")
    before = al.day_decision(db, "SOLUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    al.observe(db, engine=DAY, symbol="SOLUSDT", setup="HTF_TREND_PULLBACK", regime="btcup_vollo", metric="trade_net", value=-0.01, strategy_version=al.current_strategy_version(DAY), now=T0)
    after = al.day_decision(db, "SOLUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    assert after["expected_net"] < before["expected_net"] == 0.0
    assert after["size_mult"] < before["size_mult"]
    assert after["economic"]["final_learned_net_edge"] == pytest.approx(after["expected_net"])


def test_repeated_losses_lower_rank_and_size_and_later_wins_recover(tmp_path):
    db = str(tmp_path / "r.db")
    for _ in range(8):
        al.observe(db, engine=DAY, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="btcup_vollo", metric="lifecycle_net", value=-0.008, strategy_version=al.current_strategy_version(DAY), now=T0)
    weak = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    other = al.day_decision(db, "ETHUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0)
    ranked = rank_day_candidates([_cand("BTCUSDT", "HTF_TREND_PULLBACK", weak), _cand("ETHUSDT", "RANGE_BOUNCE", other)], DAY_V2_UNIVERSE, 0.00066)
    assert [c["symbol"] for c in ranked] == ["ETHUSDT", "BTCUSDT"]
    assert weak["size_mult"] < 1.0
    assert weak["expected_net"] < 0
    for _ in range(24):
        al.observe(db, engine=DAY, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="btcup_vollo", metric="lifecycle_net", value=0.012, strategy_version=al.current_strategy_version(DAY), now=T0)
    recovered = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    assert recovered["expected_net"] > 0.0 > weak["expected_net"]
    assert recovered["size_mult"] > weak["size_mult"]


def test_a_rejected_lifecycle_changes_the_next_ranking(tmp_path):
    db = str(tmp_path / "a.db")
    al.observe(db, engine=DAY, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="btcup_vollo", metric="trade_net", value=-0.004, strategy_version=al.current_strategy_version(DAY), now=T0)
    al.observe(db, engine=DAY, symbol="SOLUSDT", setup="RANGE_BOUNCE", regime="btcup_vollo", metric="lifecycle_net", value=0.003, strategy_version=al.current_strategy_version(DAY), now=T0)
    btc = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    sol = al.day_decision(db, "SOLUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0)
    ranked = rank_day_candidates([_cand("BTCUSDT", "HTF_TREND_PULLBACK", btc), _cand("SOLUSDT", "RANGE_BOUNCE", sol)], DAY_V2_UNIVERSE, 0.00066)
    assert ranked[0]["symbol"] == "SOLUSDT"
    assert btc["expected_net"] < 0 < sol["expected_net"]


def test_recorded_state_features_do_not_change_the_expected_net(tmp_path):
    bars = [{"ts": 1_700_000_000 + i * 900, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + (0.1 if i == 40 else 0.0), "volume": 10.0} for i in range(40)]
    feats = day_state_features(bars, bars[:8], "HTF_TREND_PULLBACK")
    assert feats["setup_htf"] == 1.0 and feats["setup_range"] == 0.0
    db = str(tmp_path / "f.db")
    plain = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    with_feats = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", features=feats, now=T0)
    assert with_feats["expected_net"] == plain["expected_net"]
    src = inspect.getsource(PortfolioEngineIntegration._record_day_v2_candidate)
    assert "state_features" in src and "alternatives" in src


def test_context_candidates_exist_without_an_opinion_pass():
    from backend.services.day_v2.live_signal import context_entry_signals, evaluate_entry_signal

    bars = []
    for i in range(40):
        close = 100.0 + (0.05 if i % 2 == 0 else -0.05)
        bars.append({"ts": 1_700_000_000 + i * 900, "open": close, "high": close + 0.4, "low": close - 0.4, "close": close, "volume": 10.0})
    # Flat, mixed candles: the opinion detector does not fire.
    assert evaluate_entry_signal("BTCUSDT", bars, bars[-8:], bars[-8:]) is None
    names = {sig.setup for sig in context_entry_signals("BTCUSDT", bars, bars[-8:], bars[-8:])}
    assert "HTF_TREND_PULLBACK" in names
    assert "RANGE_BOUNCE" in names
    assert "BREAKOUT_CONTINUATION" in names


def test_opinion_thresholds_are_not_live_permission():
    import os

    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.strategies import STRATEGY_NAMES
    from backend.services.day_v2.live_signal import context_entry_signals

    src = inspect.getsource(context_entry_signals)
    for gate in ("rsi", "c0 > b1c", "regime ==", "30 <", "confidence"):
        assert gate not in src
    os.environ.pop("SCALP_DISABLED_STRATEGIES", None)
    cfg = ScalpConfig.from_env()
    assert cfg.disabled_strategies == frozenset()
    assert len(STRATEGY_NAMES) == 9


def test_scalp_losses_lower_edge_and_size_and_wins_restore_them(tmp_path):
    from backend.services.scalp_v2.executable_edge import scalp_executable_edge

    db = str(tmp_path / "s.db")
    reg = "btcup_vollo"
    ver = al.current_strategy_version(al.SCALP_ENGINE)

    def edge():
        view = al.scalp_decision(db, "ETHUSDT", "VWAP_EMA_RECLAIM", reg, now=T0)
        return scalp_executable_edge(view, raw_expected_move_pct=0.0, spread_pct=0.0002, impact_pct=0.0, edge_source="STRATEGY_CLAIM")

    cold = edge()
    for _ in range(6):
        al.learn_claim_label(db, symbol="ETHUSDT", setup="VWAP_EMA_RECLAIM", regime=reg, strategy_version=ver, raw=0.001, gross=-0.004, now=T0)
    down = edge()
    assert down.final_executable_edge_pct < cold.final_executable_edge_pct
    assert down.size_mult <= cold.size_mult
    for _ in range(18):
        al.learn_claim_label(db, symbol="ETHUSDT", setup="VWAP_EMA_RECLAIM", regime=reg, strategy_version=ver, raw=0.001, gross=0.006, now=T0)
    back = edge()
    assert back.final_executable_edge_pct > down.final_executable_edge_pct
    assert back.size_mult >= down.size_mult


def test_continuation_exits_when_holding_is_worse_and_holds_when_it_is_better():
    from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    assert al.learned_hold_or_exit(expected_terminal_net=None, unrealized_net=0.01) == "hold"
    assert al.learned_hold_or_exit(expected_terminal_net=0.0, unrealized_net=0.0) == "hold"
    worse = al.learned_hold_or_exit(expected_terminal_net=-0.01, unrealized_net=-0.001)
    better = al.learned_hold_or_exit(expected_terminal_net=0.02, unrealized_net=-0.001)
    assert worse == "exit" and better == "hold"
    day = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=100.2,
        bar_low=100.0,
        highest_price=100.2,
        atr_at_entry=1.0,
        structural_anchor=99.0,
        target_price=103.0,
        entry_time=T0,
        estimated_roundtrip_cost=0.0006,
        now=T0 + 3600,
        expected_terminal_net=-0.002,
    )
    assert day["reason"] == "DAY_V2_LEARNED_CONTINUATION"
    held = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=99.5,
        bar_low=99.5,
        highest_price=100.0,
        atr_at_entry=1.0,
        structural_anchor=90.0,
        target_price=103.0,
        entry_time=T0,
        estimated_roundtrip_cost=0.0006,
        now=T0 + 50 * 60,
        expected_terminal_net=0.01,
    )
    assert held is None
    from types import SimpleNamespace

    pos = SimpleNamespace(engine_id="SCALP_V2", cost_basis=100.0, entry_price=100.0, highest_price=100.0, lowest_price=100.0, symbol="ETHUSDT", adaptive_decision={})
    early = evaluate_scalp_v2_exit(position=pos, current_price=100.0, net_pnl_pct=-0.001, hold_minutes=3.0, bar_low=100.0, expected_terminal_net=-0.01)
    assert early["reason"] == "SCALP_V2_LEARNED_CONTINUATION"
    stayed = evaluate_scalp_v2_exit(position=pos, current_price=100.0, net_pnl_pct=0.002, hold_minutes=40.0, bar_low=100.0, expected_terminal_net=0.01)
    assert stayed["action"] == "hold"


def test_later_positive_terminal_stops_the_early_exit():
    assert al.learned_hold_or_exit(expected_terminal_net=-0.02, unrealized_net=-0.004) == "exit"
    assert al.learned_hold_or_exit(expected_terminal_net=0.01, unrealized_net=-0.004) == "hold"


def test_continuation_learns_both_directions_without_a_config_change(tmp_path):
    db = str(tmp_path / "cont.db")
    day_ver = al.current_strategy_version(al.DAY_ENGINE)
    scalp_ver = al.current_strategy_version(al.SCALP_ENGINE)

    def day_term(mark: float) -> float:
        return float(al.continuation_terminal(db, al.DAY_ENGINE, "SOLUSDT", "RANGE_BOUNCE", "neutral", mark, now=T0))

    def scalp_term(mark: float) -> float:
        return float(al.continuation_terminal(db, al.SCALP_ENGINE, "ETHUSDT", "VWAP_EMA_RECLAIM", "neutral", mark, now=T0))

    assert day_term(0.008) == pytest.approx(0.008)
    assert scalp_term(-0.003) == pytest.approx(-0.003)
    assert al.learned_hold_or_exit(expected_terminal_net=day_term(0.008), unrealized_net=0.008) == "hold"
    assert al.learned_hold_or_exit(expected_terminal_net=scalp_term(-0.003), unrealized_net=-0.003) == "hold"

    common = {"version_current": True, "is_dust": False, "continuation": None}
    assert al.learn_from_close(
        db,
        engine=al.DAY_ENGINE,
        symbol="SOLUSDT",
        setup="RANGE_BOUNCE",
        regime="neutral",
        strategy_version=day_ver,
        net_pct=-0.002,
        mfe_pct=0.01,
        mae_pct=0.004,
        hold_min=30,
        now=T0,
        unrealized_marks=[0.008],
        **common,
    )
    green_after_loss = day_term(0.008)
    assert green_after_loss < 0.008
    assert al.learned_hold_or_exit(expected_terminal_net=green_after_loss, unrealized_net=0.008) == "exit"
    for _ in range(6):
        al.learn_from_close(
            db,
            engine=al.DAY_ENGINE,
            symbol="SOLUSDT",
            setup="RANGE_BOUNCE",
            regime="neutral",
            strategy_version=day_ver,
            net_pct=0.02,
            mfe_pct=0.03,
            mae_pct=0.002,
            hold_min=40,
            now=T0 + 100,
            unrealized_marks=[0.008],
            **common,
        )
    green_recovered = day_term(0.008)
    assert green_recovered > green_after_loss
    assert al.learned_hold_or_exit(expected_terminal_net=green_recovered, unrealized_net=0.008) == "hold"

    al.learn_from_close(
        db,
        engine=al.SCALP_ENGINE,
        symbol="ETHUSDT",
        setup="VWAP_EMA_RECLAIM",
        regime="neutral",
        strategy_version=scalp_ver,
        net_pct=-0.012,
        mfe_pct=0.001,
        mae_pct=0.012,
        hold_min=5,
        now=T0,
        unrealized_marks=[-0.003],
        **common,
    )
    red_worse = scalp_term(-0.003)
    assert red_worse < -0.003
    assert al.learned_hold_or_exit(expected_terminal_net=red_worse, unrealized_net=-0.003) == "exit"
    for _ in range(6):
        al.learn_from_close(
            db,
            engine=al.SCALP_ENGINE,
            symbol="ETHUSDT",
            setup="VWAP_EMA_RECLAIM",
            regime="neutral",
            strategy_version=scalp_ver,
            net_pct=0.004,
            mfe_pct=0.006,
            mae_pct=0.003,
            hold_min=8,
            now=T0 + 200,
            unrealized_marks=[-0.003],
            **common,
        )
    red_recovered = scalp_term(-0.003)
    assert red_recovered > red_worse
    assert al.learned_hold_or_exit(expected_terminal_net=red_recovered, unrealized_net=-0.003) == "hold"


def test_universe_slots_and_exits_are_unchanged():
    assert DAY_V2_UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert (pe.DAY_MAX_OPEN_POSITIONS, pe.SCALP_MAX_OPEN_POSITIONS) == (4, 4)
    assert frozenset({"HTF_TREND_PULLBACK", "RANGE_BOUNCE", "BREAKOUT_CONTINUATION", "VWAP_REVERSION", "EXHAUSTION_MR"}) == ENABLED_SETUPS
    assert DAY_V2_CATASTROPHIC_ATR_MULTIPLIER == 3.0
    assert pytest.approx(0.015) == scalp_exits.SCALP_V2_CATASTROPHIC_PCT
    src = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert "NO_EXECUTABLE_NET_EDGE" in src
    assert "check_frequency_limit" not in inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)


def test_unfired_market_state_is_priced_and_does_not_inherit_the_fired_setup(tmp_path):
    db = str(tmp_path / "pop.db")
    fired_key = al.day_learned_setup("BREAKOUT_CONTINUATION", detector_fired=True)
    market_key = al.day_learned_setup("BREAKOUT_CONTINUATION", detector_fired=False)
    assert fired_key == "BREAKOUT_CONTINUATION"
    assert market_key != fired_key
    assert al.day_geometry_setup(market_key) == "BREAKOUT_CONTINUATION"
    for _ in range(12):
        al.observe(db, engine=DAY, symbol="XRPUSDT", setup=fired_key, regime="btcup_vollo", metric="trade_net", value=0.02, strategy_version=al.current_strategy_version(DAY), now=T0)
    fired = al.day_decision(db, "XRPUSDT", fired_key, "btcup_vollo", now=T0)
    market = al.day_decision(db, "XRPUSDT", market_key, "btcup_vollo", now=T0)
    assert fired["expected_net"] > 0.0
    assert market["expected_net"] == 0.0
    src = inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)
    assert "if signal is not fired:" not in src
    assert "detector_fired=signal is fired" in src


def test_market_state_can_outrank_a_deteriorating_setup_without_a_code_change(tmp_path):
    db = str(tmp_path / "rank.db")
    ver = al.current_strategy_version(DAY)
    fired_key = al.day_learned_setup("RANGE_BOUNCE", detector_fired=True)
    market_key = al.day_learned_setup("BREAKOUT_CONTINUATION", detector_fired=False)
    for _ in range(10):
        al.observe(db, engine=DAY, symbol="BTCUSDT", setup=fired_key, regime="btcup_vollo", metric="trade_net", value=0.01, strategy_version=ver, now=T0)
    winner = al.day_decision(db, "BTCUSDT", fired_key, "btcup_vollo", now=T0)
    quiet = al.day_decision(db, "ETHUSDT", market_key, "btcup_vollo", now=T0)
    ranked = rank_day_candidates([_cand("BTCUSDT", "RANGE_BOUNCE", winner), _cand("ETHUSDT", "BREAKOUT_CONTINUATION", quiet)], DAY_V2_UNIVERSE, 0.00066)
    assert ranked[0]["symbol"] == "BTCUSDT"
    for _ in range(16):
        al.observe(db, engine=DAY, symbol="BTCUSDT", setup=fired_key, regime="btcup_vollo", metric="trade_net", value=-0.02, strategy_version=ver, now=T0)
        al.observe(db, engine=DAY, symbol="ETHUSDT", setup=market_key, regime="btcup_vollo", metric="lifecycle_net", value=0.01, strategy_version=ver, now=T0)
    faded = al.day_decision(db, "BTCUSDT", fired_key, "btcup_vollo", now=T0)
    emerged = al.day_decision(db, "ETHUSDT", market_key, "btcup_vollo", now=T0)
    assert emerged["expected_net"] > 0.0 > faded["expected_net"]
    assert emerged["size_mult"] > faded["size_mult"]
    ranked = rank_day_candidates([_cand("BTCUSDT", "RANGE_BOUNCE", faded), _cand("ETHUSDT", "BREAKOUT_CONTINUATION", emerged)], DAY_V2_UNIVERSE, 0.00066)
    assert ranked[0]["symbol"] == "ETHUSDT"
    assert ranked[0]["adaptive"]["setup"] == market_key
