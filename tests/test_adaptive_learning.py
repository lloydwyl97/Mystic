"""Adaptive learning changes the next DAY and SCALP decision. No sample-size gate."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from backend.services.adaptive_learning import (
    day_decision,
    learn_from_close,
    observe,
    record_candidate,
    resolve_markouts,
    scalp_decision,
)
from backend.services.day_v2.ranking import clamp_to_sleeve, rank_day_candidates
from backend.services.day_v2.winner_contract import objective_level, runner_stop
from backend.services.strategy_version import DAY_STRATEGY_VERSION, SCALP_STRATEGY_VERSION

DAY = "DAY_V2"
SCALP = "SCALP_V2"


def _sig(setup="RANGE_BOUNCE", anchor=99.0, target=103.0, atr_1h=1.0, structural=103.0):
    return SimpleNamespace(setup=setup, structural_anchor=anchor, target_price=target, atr_1h=atr_1h, objective_structural=structural, regime="neutral", atr=0.4)


def _cand(symbol, adaptive=None):
    return {"symbol": symbol, "signal": _sig(), "ask_price": 100.0, "adaptive": adaptive}


# --- A / I -------------------------------------------------------------------


def test_a_i_day_markout_updates_state_and_next_candidate_reads_it(tmp_path):
    db = str(tmp_path / "t.db")
    before = day_decision(db, "XRPUSDT", "RANGE_BOUNCE", "neutral")
    record_candidate(db, engine=DAY, symbol="XRPUSDT", setup="RANGE_BOUNCE", regime="neutral", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=1_000.0)
    learned = resolve_markouts(db, lambda _s, _t: 103.0, now=1_000.0 + 400 * 60)
    after = day_decision(db, "XRPUSDT", "RANGE_BOUNCE", "neutral")
    assert learned == 1
    assert after["n_forward"] >= 1
    assert after["expected_move"] != before["expected_move"]
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

    assert "day_decision" in inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)


# --- B / C / D / E / F -------------------------------------------------------


def test_b_c_d_e_f_day_outcome_moves_rank_size_objective_and_runner(tmp_path):
    db = str(tmp_path / "t.db")
    assert observe(db, engine=DAY, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="bull", metric="trade_mfe", value=0.04, strategy_version=DAY_STRATEGY_VERSION)
    assert observe(db, engine=DAY, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="bull", metric="trade_continuation", value=0.9, strategy_version=DAY_STRATEGY_VERSION)
    assert observe(db, engine=DAY, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="bull", metric="trade_time_to_mfe_min", value=40.0, strategy_version=DAY_STRATEGY_VERSION)
    assert observe(db, engine=DAY, symbol="SOLUSDT", setup="RANGE_BOUNCE", regime="bear", metric="trade_mae", value=0.02, strategy_version=DAY_STRATEGY_VERSION)
    btc = day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "bull")
    sol = day_decision(db, "SOLUSDT", "RANGE_BOUNCE", "bear")
    assert btc["mfe"] > btc["expected_move_prior"]
    assert btc["size_mult"] > 1.0
    assert 0.75 <= btc["objective_atr_mult"] <= 1.35
    assert objective_level("HTF_TREND_PULLBACK", 100.0, 1.0, 101.0, atr_mult=btc["objective_atr_mult"]) > objective_level("HTF_TREND_PULLBACK", 100.0, 1.0, 101.0)
    assert 0.80 <= btc["runner_activation_mult"] <= 1.25
    assert sol["mae"] > 0.006
    plain = _cand("ETHUSDT")
    boosted = _cand("BTCUSDT", btc)
    ranked = rank_day_candidates([plain, boosted], ["ETHUSDT", "BTCUSDT"], 0.0006)
    assert ranked[0]["symbol"] == "BTCUSDT"
    assert ranked[0]["rank"]["size_mult"] == pytest.approx(btc["size_mult"])
    sized = clamp_to_sleeve(2.0 * btc["size_mult"], 10.0, 100.0)
    assert 2.0 < sized <= 2.0 * 1.35
    assert clamp_to_sleeve(sized, 10.0, 15.0) * 10.0 <= 15.0


def test_g_h_ratchet_never_loosens_and_hard_stops_stay():
    stops = []
    for high in (101.2, 102.0, 103.5):
        state = runner_stop(entry_price=100.0, highest_price=high, atr_1h=1.0, objective=104.0, estimated_roundtrip_cost=0.0006, trail_mult=1.1, activation_mult=0.9)
        if state["activated"]:
            stops.append(state["stop"])
    assert stops == sorted(stops)
    assert stops[-1] >= stops[0]
    from backend.services.day_v2.live_exit_evaluator import DAY_V2_CATASTROPHIC_ATR_MULTIPLIER, evaluate_day_v2_exit

    assert DAY_V2_CATASTROPHIC_ATR_MULTIPLIER == 3.0
    dec = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=98.0,
        bar_low=98.0,
        highest_price=100.0,
        atr_at_entry=0.4,
        structural_anchor=97.0,
        target_price=103.0,
        entry_time=1.0,
        estimated_roundtrip_cost=0.0006,
        setup="HTF_TREND_PULLBACK",
        atr_1h_at_entry=1.0,
        objective_structural=103.0,
        objective_atr_mult=1.2,
    )
    assert dec["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"


# --- J / K / L / M / N / O / P ------------------------------------------------


def test_j_p_scalp_markout_and_next_candidate(tmp_path):
    db = str(tmp_path / "t.db")
    before = scalp_decision(db, "ETHUSDT", "VWAP_EMA_RECLAIM", "")
    record_candidate(db, engine=SCALP, symbol="ETHUSDT", setup="VWAP_EMA_RECLAIM", regime="", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=1_000.0)
    assert resolve_markouts(db, lambda _s, _t: 100.4, now=1_000.0 + 2000) == 1
    after = scalp_decision(db, "ETHUSDT", "VWAP_EMA_RECLAIM", "")
    assert after["n_forward"] >= 1
    assert after["expected_edge"] != before["expected_edge"]
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

    src = inspect.getsource(PortfolioEngineIntegration._process_scalp_v2_signals)
    assert "scalp_decision" in src and "sorted(" in src


def test_k_l_m_n_o_scalp_loss_win_hold_size_and_rank(tmp_path):
    db = str(tmp_path / "t.db")
    observe(db, engine=SCALP, symbol="XRPUSDT", setup="RANGE_BOUNCE_SCALP", regime="", metric="trade_mae", value=0.004, strategy_version=SCALP_STRATEGY_VERSION)
    observe(db, engine=SCALP, symbol="BTCUSDT", setup="VWAP_EMA_RECLAIM", regime="", metric="trade_mfe", value=0.005, strategy_version=SCALP_STRATEGY_VERSION)
    observe(db, engine=SCALP, symbol="SOLUSDT", setup="RANGE_BOUNCE_SCALP", regime="", metric="trade_time_to_mfe_min", value=5.0, strategy_version=SCALP_STRATEGY_VERSION)
    observe(db, engine=SCALP, symbol="ETHUSDT", setup="VWAP_EMA_RECLAIM", regime="", metric="trade_net", value=0.004, strategy_version=SCALP_STRATEGY_VERSION)
    xrp = scalp_decision(db, "XRPUSDT", "RANGE_BOUNCE_SCALP", "")
    btc = scalp_decision(db, "BTCUSDT", "VWAP_EMA_RECLAIM", "")
    sol = scalp_decision(db, "SOLUSDT", "RANGE_BOUNCE_SCALP", "")
    eth = scalp_decision(db, "ETHUSDT", "VWAP_EMA_RECLAIM", "")
    assert xrp["risk_estimate"] > xrp["mae"] * 0 + 0.0015
    assert btc["target_pct"] > 0.0025
    assert btc["target_pct"] <= 0.006
    assert sol["hold_min"] < sol["hold_hard_max_min"]
    assert sol["hold_min"] >= 4.0
    assert eth["size_mult"] > 1.0
    assert eth["size_mult"] <= 1.25
    order = sorted(
        [("BTCUSDT", btc), ("ETHUSDT", eth), ("SOLUSDT", sol), ("XRPUSDT", xrp)],
        key=lambda item: -(item[1]["expected_edge"] + 0.001 * item[1]["confidence"]),
    )
    assert order[0][0] == "ETHUSDT"
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    pos = SimpleNamespace(engine_id="SCALP_V2", entry_price=100.0, highest_price=100.0, lowest_price=100.0, symbol="SOLUSDT", adaptive_decision=sol)
    dec = evaluate_scalp_v2_exit(position=pos, current_price=100.05, net_pnl_pct=0.0002, hold_minutes=sol["hold_min"], bar_low=100.0)
    assert dec["reason"] == "SCALP_V2_TIME_STOP"


# --- Q / R -------------------------------------------------------------------


def test_q_r_engines_do_not_contaminate(tmp_path):
    db = str(tmp_path / "t.db")
    observe(db, engine=DAY, symbol="BTCUSDT", setup="BREAKOUT_CONTINUATION", regime="bull", metric="trade_mfe", value=0.05, strategy_version=DAY_STRATEGY_VERSION)
    scalp_before = scalp_decision(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "bull")
    observe(db, engine=SCALP, symbol="BTCUSDT", setup="BREAKOUT_CONTINUATION", regime="bull", metric="trade_net", value=-0.01, strategy_version=SCALP_STRATEGY_VERSION)
    day_after = day_decision(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "bull")
    scalp_after = scalp_decision(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "bull")
    assert day_after["mfe"] > 0.012
    assert scalp_before["expected_edge"] == pytest.approx(scalp_decision(db, "ETHUSDT", "OTHER", "")["edge_prior"]) or scalp_before["n_net"] == 0
    assert scalp_after["n_net"] == 1
    assert day_after["n_mfe"] == 1
    assert scalp_after["expected_edge"] < scalp_before["expected_edge"]


# --- S / T / U ---------------------------------------------------------------


def test_s_t_u_no_count_gate_no_legacy(tmp_path):
    db = str(tmp_path / "t.db")
    cold = day_decision(db, "SOLUSDT", "EXHAUSTION_MR", "bear")
    assert cold["confidence"] == 0.0 and cold["size_mult"] == 1.0
    ranked = rank_day_candidates([_cand("SOLUSDT", cold), _cand("XRPUSDT")], ["SOLUSDT", "XRPUSDT"], 0.0006)
    assert len(ranked) == 2
    assert observe(db, engine=DAY, symbol="SOLUSDT", setup="EXHAUSTION_MR", regime="bear", metric="trade_mfe", value=-0.05, strategy_version="LEGACY_MIXED") is False
    assert day_decision(db, "SOLUSDT", "EXHAUSTION_MR", "bear")["n_mfe"] == 0
    assert (
        learn_from_close(
            db,
            engine=DAY,
            symbol="SOLUSDT",
            setup="EXHAUSTION_MR",
            regime="bear",
            strategy_version="",
            net_pct=-1.0,
            mfe_pct=0.0,
            mae_pct=0.02,
            hold_min=10,
            continuation=0.0,
            version_current=False,
            is_dust=False,
        )
        is False
    )
    src = inspect.getsource(day_decision) + inspect.getsource(scalp_decision)
    assert "profit_factor" not in src
    for banned in ("if n <", "if spec_n <", "min_trades"):
        assert banned not in src


# --- V / W / X / Y / Z / AA / AB ---------------------------------------------


def test_v_through_ab_universe_slots_and_hard_safety():
    from backend.services import portfolio_engine as pe
    from backend.services import two_engine_claim
    from backend.services.binance_scalp import protected_preflight
    from backend.services.day_v2.config import DAY_V2_UNIVERSE
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT

    assert set(DAY_V2_UNIVERSE) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert (pe.DAY_MAX_OPEN_POSITIONS, pe.SCALP_MAX_OPEN_POSITIONS, pe.COMBINED_ENGINE_MAX_POSITIONS) == (4, 4, 8)
    assert two_engine_claim.engine_cap("DAY_V2") == 4 and two_engine_claim.engine_cap("SCALP_V2") == 4
    assert "expected_net <= 0 or expected_net < econ.min_net_edge_pct" in inspect.getsource(protected_preflight)
    scalp_src = inspect.getsource(PortfolioEngineIntegration._process_scalp_v2_signals)
    assert "SYMBOL_OCCUPIED" in scalp_src
    assert pytest.approx(0.015) == SCALP_V2_CATASTROPHIC_PCT
    assert "adverse_move >= SCALP_V2_CATASTROPHIC_PCT" in inspect.getsource(__import__("backend.services.scalp_v2.exit_evaluator", fromlist=["evaluate_scalp_v2_exit"]).evaluate_scalp_v2_exit)


# --- close-path key alignment -------------------------------------------------


def test_close_learns_under_entry_stamped_key_reaches_next_candidate(tmp_path):
    """A closed SCALP outcome must update the SAME setup/regime the entry read.

    The entry reads best_setup (e.g. SCALP_STRUCTURAL); the closed lot's
    provenance can name the setup differently (e.g. range_bounce_scalp). The
    close-path must learn under the entry-stamped key so the next candidate,
    which reads best_setup, actually sees the update.
    """
    db = str(tmp_path / "t.db")
    entry_setup, entry_regime = "SCALP_STRUCTURAL", ""
    before = scalp_decision(db, "ETHUSDT", entry_setup, entry_regime)

    # Provenance names the closed lot with a different setup string.
    assert learn_from_close(
        db,
        engine=SCALP,
        symbol="ETHUSDT",
        setup=entry_setup,  # the key the entry stamped, not the provenance name
        regime=entry_regime,
        strategy_version=SCALP_STRATEGY_VERSION,
        net_pct=0.006,
        mfe_pct=0.007,
        mae_pct=0.002,
        hold_min=6.0,
        continuation=1.0,
        version_current=True,
        is_dust=False,
    )

    after = scalp_decision(db, "ETHUSDT", entry_setup, entry_regime)
    assert after["expected_edge"] != before["expected_edge"]
    assert after["n_net"] == 1

    # A read under the provenance name would NOT have seen it (proves the risk).
    stranded = scalp_decision(db, "ETHUSDT", "RANGE_BOUNCE_SCALP", entry_regime)
    assert stranded["n_net"] == 0

    # The live close-path keys the learn call on the entry-stamped decision.
    import pathlib

    pe_src = pathlib.Path(__import__("backend.services.portfolio_engine", fromlist=["__file__"]).__file__).read_text()
    assert '_adapt_dec.get("setup")' in pe_src
    assert '_adapt_dec.get("regime")' in pe_src
