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


def test_universe_slots_and_exits_are_unchanged():
    assert DAY_V2_UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert (pe.DAY_MAX_OPEN_POSITIONS, pe.SCALP_MAX_OPEN_POSITIONS) == (4, 4)
    assert frozenset({"HTF_TREND_PULLBACK", "RANGE_BOUNCE", "BREAKOUT_CONTINUATION", "VWAP_REVERSION", "EXHAUSTION_MR"}) == ENABLED_SETUPS
    assert DAY_V2_CATASTROPHIC_ATR_MULTIPLIER == 3.0
    assert pytest.approx(0.015) == scalp_exits.SCALP_V2_CATASTROPHIC_PCT
    src = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert "NO_EXECUTABLE_NET_EDGE" in src
    assert "check_frequency_limit" not in inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)


def test_unfired_day_contexts_are_unlearnable_shadow_evidence(tmp_path):
    import sqlite3

    db = str(tmp_path / "shadow.db")
    PortfolioEngineIntegration._record_day_context_shadow(db, symbol="XRPUSDT", setup="BREAKOUT_CONTINUATION", ask_price=1.5, as_of=T0)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT signaled, candidate_state, filled FROM adaptive_candidate_markouts WHERE engine_id='DAY_V2'").fetchall()
    assert rows == [(0, al.CANDIDATE_NEAR_QUALIFIED, 0)]
    assert al.CANDIDATE_NEAR_QUALIFIED not in al.LEARNABLE_DAY_STATES
    src = inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)
    assert "if signal is not fired:" in src
    assert src.index("_record_day_context_shadow") < src.index("self._admit_day_context_candidate(")
