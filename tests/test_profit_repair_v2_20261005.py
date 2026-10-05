"""Profit repair v2: a SCALP claim is the strategy's bare structural projection
(no reachability floor), priced only through a calibration learned from realized
gross moves of every claim; DAY setup evidence moves rank and size continuously
and never removes a candidate; correlated same-bar DAY candidates are all
ranked; blank regimes stay out of regime-specific state; universe, slots and
exits are unchanged."""

from __future__ import annotations

import inspect
import sqlite3
from types import SimpleNamespace

import pytest

import backend.services.adaptive_learning as al
import backend.services.portfolio_engine as pe
from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_candidate_ranking import rank_setup_signal
from backend.services.binance_scalp.strategies import range_bounce_scalp, vwap_ema_reclaim
from backend.services.binance_scalp.strategies.base import StrategyMarketContext
from backend.services.binance_scalp.strategies.common import reject_signal
from backend.services.day_v2.config import DAY_V2_UNIVERSE
from backend.services.day_v2.live_exit_evaluator import DAY_V2_CATASTROPHIC_ATR_MULTIPLIER
from backend.services.day_v2.live_signal import ENABLED_SETUPS
from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.economic_state_rebuild import rebuild_scalp
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
from backend.services.scalp_v2 import exit_evaluator as scalp_exits
from backend.services.scalp_v2.executable_edge import REJECT_THRESHOLD_PCT, scalp_executable_edge

DAY, SCALP = al.DAY_ENGINE, al.SCALP_ENGINE
T0 = 1_791_200_000.0  # after both economic anchors
SPREAD = 0.00002
COST = canonical_roundtrip_cost_pct(spread_pct=SPREAD, buy_impact_pct=0.0, sell_impact_pct=0.0)
SKEY = ("ETHUSDT", "VWAP_EMA_RECLAIM", "btcup_vollo")


def _snap(px: float = 100.0) -> MarketSnapshot:
    bid, ask = px - 0.001, px + 0.001
    return MarketSnapshot(
        symbol="BTC/USDT",
        symbol_bus="BTCUSDT",
        best_bid=bid,
        best_ask=ask,
        mid=px,
        spread_pct=(ask - bid) / px,
        bids=[[bid, 1000.0]],
        asks=[[ask, 1000.0]],
        redis_spread_pct=None,
        order_book_imbalance=None,
        book_source="ws",
        orderbook_age_sec=0.1,
    )


def _rising_mom(now: float = 1000.0):
    tr = MomentumTracker()
    for t in range(0, 65, 5):
        px = 99.97 + 0.0005 * t
        tr.record("BTCUSDT", now - 60 + t, px - 0.001, px)
    return tr.diagnostics("BTCUSDT", now, 99.999, 100.0)


def _ctx(bars: list[dict]) -> StrategyMarketContext:
    return StrategyMarketContext(
        symbol="BTC/USDT",
        snap=_snap(),
        mom=_rising_mom(),
        bars_1m=bars,
        econ=ScalpEconomics.from_env(),
        config=SimpleNamespace(scalp_paper_enabled=False, scalp_live=True, calibration_mode=False, database_path=":memory:"),
        notional_usd=25.0,
    )


def _bar(high: float, low: float, close: float = 99.99) -> dict:
    return {"open": close, "high": high, "low": low, "close": close, "volume": 10.0}


def _edge(view: dict, raw: float):
    return scalp_executable_edge(view, raw_expected_move_pct=raw, spread_pct=SPREAD, impact_pct=0.0, edge_source="STRATEGY_CLAIM")


def _claim_rows(db: str, raws: list[float], *, gross_of, start: float = T0, step: float = 1300.0, signaled: bool = False) -> float:
    """Strategy claims, each resolved by the live resolver once its horizons pass;
    ``gross_of(raw)`` is the claim's realized gross move."""
    end = start
    for i, raw in enumerate(raws):
        at = start + step * i
        al.record_candidate(
            db,
            engine=SCALP,
            symbol=SKEY[0],
            setup=SKEY[1],
            regime=SKEY[2],
            ref_price=100.0,
            roundtrip_cost=COST,
            signaled=signaled,
            evaluated_at=at,
            raw_expected_move=raw,
            raw_move_source="STRATEGY_CLAIM",
        )
        end = at + 1300.0
        price = 100.0 * (1.0 + gross_of(raw))
        al.resolve_markouts(db, lambda _s, _t, p=price: p, now=end)
    return end


def _day_obs(db: str, value: float, *, symbol="BTCUSDT", setup="HTF_TREND_PULLBACK", regime="btcup_vollo", metric="trade_net") -> bool:
    return al.observe(db, engine=DAY, symbol=symbol, setup=setup, regime=regime, metric=metric, value=value, strategy_version=al.current_strategy_version(DAY), now=T0)


def _cand(symbol: str, setup: str, adaptive: dict | None) -> dict:
    sig = SimpleNamespace(setup=setup, structural_anchor=99.0, target_price=103.0, atr_1h=1.0, objective_structural=103.0, regime="neutral", atr=0.4)
    return {"symbol": symbol, "signal": sig, "ask_price": 100.0, "adaptive": adaptive}


# --- SCALP: the claim is the projection, never a floor -------------------------


def test_vwap_claim_is_the_bare_reclaim_projection_below_the_reachability_minimum():
    bars = [_bar(100.0, 99.97) for _ in range(14)] + [_bar(100.0, 99.98)]
    sig = vwap_ema_reclaim.VwapEmaReclaimStrategy().evaluate(_ctx(bars))
    projection = vwap_ema_reclaim.reclaim_projection(bars, 100.0)
    assert sig.passed and sig.claim_available
    assert 0.0 < projection < vwap_ema_reclaim._REACH_MIN_PCT
    assert sig.directional_move_pct == pytest.approx(projection)
    # The reachability minimum shapes only the setup's target label.
    assert sig.expected_move_pct >= vwap_ema_reclaim._REACH_MIN_PCT > sig.directional_move_pct


def test_range_bounce_claim_is_the_bare_distance_to_the_range_high():
    bars = [_bar(100.03, 99.985) for _ in range(15)]
    sig = range_bounce_scalp.RangeBounceScalpStrategy().evaluate(_ctx(bars))
    assert sig.passed and sig.claim_available
    assert sig.directional_move_pct == pytest.approx(range_bounce_scalp.bounce_projection(bars, 100.0)) == pytest.approx(0.0003)
    assert sig.expected_move_pct > sig.directional_move_pct


def test_a_reject_with_a_zero_projection_is_still_a_zero_claim():
    ctx = _ctx([_bar(100.0, 99.97) for _ in range(15)])
    zero = reject_signal(ctx, "vwap_ema_reclaim", "TARGET_NOT_REACHABLE", expected_move=0.0012, directional_move=-0.0004)
    assert zero.claim_available and zero.directional_move_pct == 0.0
    none = reject_signal(ctx, "vwap_ema_reclaim", "NO_VWAP_EMA_RECLAIM")
    assert not none.claim_available


@pytest.mark.parametrize("raw", [0.0, 0.0001, 0.0012, 0.006])
def test_cold_geometry_or_volatility_never_manufactures_edge(tmp_path, raw):
    view = al.scalp_decision(str(tmp_path / "c.db"), *SKEY, now=T0)
    edge = _edge(view, raw)
    assert edge.expected_move_pct == 0.0
    assert edge.final_executable_edge_pct == pytest.approx(-COST)
    assert not edge.eligible


def test_wide_atr_and_a_large_projection_rank_at_minus_cost_cold(tmp_path, monkeypatch):
    real = al.scalp_decision
    monkeypatch.setattr(al, "scalp_decision", lambda *a, **k: real(str(tmp_path / "w.db"), *a[1:], **k))
    wide = [_bar(100.6, 99.4, close=100.0) for _ in range(29)] + [_bar(100.6, 99.5, close=100.0)]
    sig = vwap_ema_reclaim.VwapEmaReclaimStrategy().evaluate(_ctx(wide))
    assert sig.passed and sig.directional_move_pct == pytest.approx(0.006)
    rc = rank_setup_signal(sig, regime="RANGE", ctx=_ctx(wide))
    assert rc.volatility_move_pct > 0.004
    assert rc.expected_move_pct == pytest.approx(sig.directional_move_pct)
    assert rc.executable_edge["expected_move_pct"] == 0.0
    assert rc.executable_edge["final_executable_edge_pct"] == pytest.approx(-rc.roundtrip_cost_pct)
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE" and not rc.entry_eligible


def test_zero_and_small_claims_are_recorded_and_learned(tmp_path):
    db = str(tmp_path / "z.db")
    end = _claim_rows(db, [0.0, 0.0001], gross_of=lambda _raw: -0.0004)
    view = al.scalp_decision(db, *SKEY, now=end)
    assert view["n_claim"] == pytest.approx(2.0, rel=1e-2)
    assert view["claim_gross_setup"] < 0.0
    with sqlite3.connect(db) as conn:
        learned = conn.execute("SELECT COUNT(*) FROM adaptive_candidate_markouts WHERE learned=1").fetchone()[0]
    assert learned == 2


def test_rejected_claims_update_the_calibration(tmp_path):
    db = str(tmp_path / "r.db")
    raws = [0.0012] * 12
    cold = _edge(al.scalp_decision(db, *SKEY, now=T0), 0.0012)
    end = _claim_rows(db, raws, gross_of=lambda _raw: -0.0010, signaled=False)
    warm = _edge(al.scalp_decision(db, *SKEY, now=end), 0.0012)
    assert warm.final_executable_edge_pct < cold.final_executable_edge_pct == pytest.approx(-COST)
    assert warm.expected_move_pct < 0.0


def test_causal_evidence_recovers_a_supported_claim_and_a_1bp_claim_stays_negative(tmp_path):
    db = str(tmp_path / "p.db")
    raws = [0.0005 + 0.0001 * (i % 21) for i in range(63)]
    end = _claim_rows(db, raws, gross_of=lambda raw: raw - 0.0001)
    view = al.scalp_decision(db, *SKEY, now=end)
    assert view["claim_capture"] > 0.5
    supported = _edge(view, 0.0015)
    tiny = _edge(view, 0.0001)
    assert supported.final_executable_edge_pct > REJECT_THRESHOLD_PCT
    assert tiny.final_executable_edge_pct <= REJECT_THRESHOLD_PCT
    assert supported.size_mult > tiny.size_mult


def test_overlapping_claims_count_as_shared_windows(tmp_path):
    spaced = str(tmp_path / "s.db")
    packed = str(tmp_path / "o.db")
    end_s = _claim_rows(spaced, [0.001] * 6, gross_of=lambda _raw: 0.002, step=1300.0)
    end_o = _claim_rows(packed, [0.001] * 6, gross_of=lambda _raw: 0.002, step=60.0)
    n_spaced = al.scalp_decision(spaced, *SKEY, now=end_s)["n_claim"]
    n_packed = al.scalp_decision(packed, *SKEY, now=end_o)["n_claim"]
    assert n_spaced == pytest.approx(6.0, rel=0.05)
    assert 1.0 <= n_packed < 0.5 * n_spaced


# --- DAY: setup evidence is continuous; correlation never removes ---------------


def test_htf_negative_evidence_lowers_rank_and_size_continuously_and_never_removes(tmp_path):
    lo, _hi = al.SIZE_BOUNDS[DAY]
    assert "HTF_TREND_PULLBACK" in ENABLED_SETUPS
    sizes, nets = [], []
    for k, loss in enumerate((-0.001, -0.003, -0.006, -0.012, -0.03)):
        db = str(tmp_path / f"h{k}.db")
        for _ in range(6):
            _day_obs(db, loss)
        d = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
        assert d["abstain_live_veto"] is False
        sizes.append(d["size_mult"])
        nets.append(d["expected_net"])
        neutral = al.day_decision(db, "ETHUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0)
        ranked = rank_day_candidates([_cand("BTCUSDT", "HTF_TREND_PULLBACK", d), _cand("ETHUSDT", "RANGE_BOUNCE", neutral)], DAY_V2_UNIVERSE, COST)
        assert [c["symbol"] for c in ranked] == ["ETHUSDT", "BTCUSDT"]
    assert nets == sorted(nets, reverse=True) and len(set(nets)) == len(nets)
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1] >= lo


def test_htf_positive_evidence_lifts_it_back(tmp_path):
    db = str(tmp_path / "u.db")
    for _ in range(6):
        _day_obs(db, -0.01)
    weak = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    for _ in range(30):
        _day_obs(db, 0.012)
    strong = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    assert strong["expected_net"] > 0.0 > weak["expected_net"]
    assert strong["size_mult"] > 1.0 > weak["size_mult"]


def test_correlated_same_bar_day_candidates_are_all_ranked_and_keep_their_size(tmp_path):
    db = str(tmp_path / "c.db")
    for sym in DAY_V2_UNIVERSE:
        _day_obs(db, 0.004, symbol=sym, setup="BREAKOUT_CONTINUATION")
    decisions = {sym: al.day_decision(db, sym, "BREAKOUT_CONTINUATION", "btcup_vollo", now=T0) for sym in DAY_V2_UNIVERSE}
    ranked = rank_day_candidates([_cand(sym, "BREAKOUT_CONTINUATION", d) for sym, d in decisions.items()], DAY_V2_UNIVERSE, COST)
    assert sorted(c["symbol"] for c in ranked) == sorted(DAY_V2_UNIVERSE)
    for cand in ranked:
        d = decisions[cand["symbol"]]
        assert cand["rank"]["size_mult"] == pytest.approx(d["size_mult"]) and d["size_mult"] > 1.0
        assert d["economic"]["correlation_adjustment"] == {"applied": False, "edge": 0.0, "size_mult": 1.0}
    src = inspect.getsource(PortfolioEngineIntegration._process_day_v2_signals)
    for gate in ("max_one", "one_per_bar", "MAX_DAY_ENTRIES_PER_BAR", "CORRELATED_BLOCK"):
        assert gate not in src


def test_day_admitted_opportunity_persists_its_full_economic_score(tmp_path):
    db = str(tmp_path / "e.db")
    _day_obs(db, -0.002, setup="RANGE_BOUNCE")
    econ = al.day_decision(db, "BTCUSDT", "RANGE_BOUNCE", "btcup_vollo", now=T0)["economic"]
    for field in ("raw_expected_move", "calibration_adjustment", "expected_cost", "uncertainty", "setup_expected_edge", "correlation_adjustment", "expected_net_edge", "size_effect"):
        assert field in econ
    assert econ["raw_expected_move"] == 0.0
    assert econ["expected_gross"] - econ["expected_cost"] == pytest.approx(econ["expected_net_edge"])
    helper = inspect.getsource(PortfolioEngineIntegration._record_day_v2_candidate)
    assert '"rank_effect"' in helper and "economic=economic" in helper
    scalp = _edge(al.scalp_decision(db, *SKEY, now=T0), 0.001).economic()
    for field in ("raw_claim", "calibration_adjustment", "expected_cost", "uncertainty", "setup_expected_edge", "expected_net_edge", "size_effect"):
        assert field in scalp


# --- blank regimes ---------------------------------------------------------------


def test_blank_regime_rows_never_enter_regime_specific_state(tmp_path):
    db = str(tmp_path / "b.db")
    _day_obs(db, -0.01, regime="")
    known = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "btcup_vollo", now=T0)
    weights = known["economic"]["level_weights"]
    assert weights["key"] == 0.0 and weights["related"] > 0.0
    assert known["n_trade_net"] == 0.0 and known["expected_net"] < 0.0
    blank = al.day_decision(db, "BTCUSDT", "HTF_TREND_PULLBACK", "", now=T0)
    assert blank["economic"]["level_weights"]["key"] == 0.0
    other = al.day_decision(db, "ETHUSDT", "HTF_TREND_PULLBACK", "", now=T0)["economic"]["level_weights"]
    assert other["related"] == 0.0 and other["setup"] > 0.0
    al.learn_claim_label(db, symbol=SKEY[0], setup=SKEY[1], regime="", strategy_version=al.current_strategy_version(SCALP), raw=0.001, gross=-0.002, now=T0)
    view = al.scalp_decision(db, SKEY[0], SKEY[1], "", now=T0)
    assert view["claim_key_n"] == 0.0 and view["n_claim"] > 0.0 and view["claim_gross_setup"] < 0.0
    src = inspect.getsource(PortfolioEngineIntegration)
    assert "or signal.regime" not in src and "or sig.regime" not in src
    assert 'market_regime_tag(db_path, symbol) or ""' in src
    assert '_learn_regime = str(_adapt_dec.get("regime") or "")' in inspect.getsource(pe)


# --- state rebuild ---------------------------------------------------------------


def test_scalp_rebuild_refuses_once_live_rows_of_the_version_exist(tmp_path):
    db = str(tmp_path / "live.db")
    claim = {"raw_expected_move": 0.0002, "raw_move_source": "STRATEGY_CLAIM"}
    al.record_candidate(db, engine=SCALP, symbol=SKEY[0], setup=SKEY[1], regime=SKEY[2], ref_price=100.0, roundtrip_cost=COST, signaled=False, evaluated_at=T0, **claim)
    out = rebuild_scalp(db, now=T0 + 60, apply=True, workdir=str(tmp_path))
    assert out["applied"] is False and out["refused"] == "LIVE_SCALP_VERSION_ROWS=1"


def test_scalp_rebuild_leaves_day_state_untouched(tmp_path):
    db = str(tmp_path / "d.db")
    _day_obs(db, -0.004, setup="RANGE_BOUNCE")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE feature_ohlcv (id INTEGER PRIMARY KEY, symbol TEXT, interval TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, ts TEXT)")
        conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, side TEXT, engine_id TEXT, timestamp TEXT)")
    query = "SELECT engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE engine_id=? ORDER BY metric"
    with sqlite3.connect(db) as conn:
        before = conn.execute(query, (DAY,)).fetchall()
    out = rebuild_scalp(db, now=T0 + 60, apply=True, workdir=str(tmp_path))
    assert out["refused"] == "" and out["applied"] is True
    with sqlite3.connect(db) as conn:
        assert conn.execute(query, (DAY,)).fetchall() == before


# --- scope -----------------------------------------------------------------------


def test_universe_slots_and_exits_are_unchanged():
    assert DAY_V2_UNIVERSE == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert (pe.DAY_MAX_OPEN_POSITIONS, pe.SCALP_MAX_OPEN_POSITIONS) == (4, 4)
    assert DAY_V2_CATASTROPHIC_ATR_MULTIPLIER == 3.0
    assert pytest.approx(0.015) == scalp_exits.SCALP_V2_CATASTROPHIC_PCT
    exit_src = inspect.getsource(scalp_exits)
    for learned in ("claim_gross", "claim_capture", "correlation_adjustment"):
        assert learned not in exit_src
