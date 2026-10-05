"""SCALP momentum history + executable-edge contract applied to every candidate."""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_candidate_ranking import HARD_REJECT_REASONS, pick_best_ranked, rank_setup_signal
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext


def _bars(n: int, range_frac: float, px: float = 100.0) -> list[dict]:
    half = px * range_frac / 2.0
    return [{"open": px, "high": px + half, "low": px - half, "close": px, "volume": 10.0} for _ in range(n)]


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


def _mom(now: float = 1000.0) -> object:
    tr = MomentumTracker()
    for t in range(0, 65, 5):
        tr.record("BTCUSDT", now - 60 + t, 100.0, 100.0)
    return tr.diagnostics("BTCUSDT", now, 100.0, 100.0)


def _ctx(bars: list[dict], mom=None) -> StrategyMarketContext:
    return StrategyMarketContext(
        symbol="BTC/USDT",
        snap=_snap(),
        mom=mom if mom is not None else _mom(),
        bars_1m=bars,
        econ=ScalpEconomics.from_env(),
        config=SimpleNamespace(scalp_paper_enabled=False, scalp_live=True, calibration_mode=False, database_path=":memory:"),
        notional_usd=25.0,
    )


def _sig(*, passed: bool, reason: str | None, expected: float = 0.0, setup: str = "range_bounce_scalp", directional: float | None = None) -> ScalpSetupSignal:
    return ScalpSetupSignal(
        symbol="BTC/USDT",
        side="BUY",
        score=2.0 if passed else 0.0,
        setup_name=setup,
        confidence=0.6,
        entry_reason="",
        invalidation_reason=None,
        required_target_pct=0.0025,
        expected_move_pct=expected,
        spread_pct=0.00002,
        impact_pct=0.0,
        depth_sufficient=True,
        limit_buy_price=100.001,
        passed=passed,
        reject_reason=reason,
        setup_context={"reject_class": "opinion"} if not passed else {},
        directional_move_pct=expected if directional is None else directional,
    )


def test_momentum_reports_real_sample_age_errors():
    d = _mom()
    assert d.insufficient_windows == ()
    assert not d.insufficient_history
    assert set(d.sample_age_error_sec) == {"15s", "30s", "60s"}
    assert all(v <= 2.5 for v in d.sample_age_error_sec.values())


def test_momentum_without_history_is_insufficient_not_zero_flat():
    tr = MomentumTracker()
    tr.record("BTCUSDT", 1000.0, 100.0, 100.0)
    d = tr.diagnostics("BTCUSDT", 1000.0, 100.0, 100.0)
    assert d.insufficient_history
    assert set(d.insufficient_windows) == {"15s", "30s", "60s"}
    assert d.momentum_confirmed is False
    assert d.flat_regime is False


def test_insufficient_history_is_hard_block():
    tr = MomentumTracker()
    tr.record("BTCUSDT", 1000.0, 100.0, 100.0)
    rc = rank_setup_signal(_sig(passed=False, reason="NOT_NEAR_SUPPORT"), regime="RANGE", ctx=_ctx(_bars(30, 0.012), mom=tr.diagnostics("BTCUSDT", 1000.0, 100.0, 100.0)))
    assert rc.hard_block == "INSUFFICIENT_HISTORY"
    assert "INSUFFICIENT_HISTORY" in HARD_REJECT_REASONS


def test_opinion_reject_is_priced_and_cold_net_does_not_trade():
    rc = rank_setup_signal(_sig(passed=False, reason="NOT_NEAR_SUPPORT"), regime="RANGE", ctx=_ctx(_bars(30, 0.012)))
    assert rc.volatility_move_pct > 0.001
    assert rc.edge_source == "STRATEGY_CLAIM"
    assert rc.expected_move_pct == 0.0
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE"
    assert not rc.entry_eligible


def _learned_gross(monkeypatch, gross: float) -> None:
    """Every view carries a calibration that learned ``gross`` as the claims' realized move."""
    import backend.services.adaptive_learning as al

    real = al.scalp_decision

    def learned(*args, **kwargs):
        view = real(*args, **kwargs)
        return {**view, "claim_gross_setup": gross, "claim_gross_mean": gross, "micro_residual": 0.0}

    monkeypatch.setattr(al, "scalp_decision", learned)


def test_soft_reject_with_strategy_claim_keeps_its_directional_edge(monkeypatch):
    ctx = _ctx(_bars(30, 0.012))
    rc = rank_setup_signal(_sig(passed=False, reason="TARGET_NOT_REACHABLE", directional=0.004), regime="RANGE", ctx=ctx)
    assert rc.edge_source == "STRATEGY_CLAIM"
    assert rc.executable_edge["raw_move_source"] == "STRATEGY_CLAIM"
    assert rc.executable_edge["raw_expected_move_pct"] == rc.expected_move_pct == 0.004
    assert rc.net_edge_after_costs_pct == rc.executable_edge["final_executable_edge_pct"]
    assert rc.roundtrip_cost_pct == rc.executable_edge["live_cost_pct"]
    # Priced on the economic contract, not refused as unestimable; cold, the claim earns nothing.
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE"
    _learned_gross(monkeypatch, 0.004)
    rc = rank_setup_signal(_sig(passed=False, reason="TARGET_NOT_REACHABLE", directional=0.004), regime="RANGE", ctx=ctx)
    assert rc.hard_block is None
    assert rc.entry_eligible


def test_raw_move_below_cost_needs_learned_evidence(monkeypatch):
    ctx = _ctx(_bars(30, 0.0004))
    rc = rank_setup_signal(_sig(passed=False, reason="TARGET_NOT_REACHABLE", directional=0.0004), regime="RANGE", ctx=ctx)
    assert rc.executable_edge["base_executable_edge_pct"] < 0
    assert rc.executable_edge["adaptive_residual_pct"] == 0.0
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE"
    assert not rc.entry_eligible
    _learned_gross(monkeypatch, 0.002)
    rc = rank_setup_signal(_sig(passed=False, reason="TARGET_NOT_REACHABLE", directional=0.0004), regime="RANGE", ctx=ctx)
    assert rc.executable_edge["final_executable_edge_pct"] > 0
    assert rc.entry_eligible


def test_opinion_reject_without_projection_is_still_an_economic_decision():
    rc = rank_setup_signal(_sig(passed=False, reason="NOT_NEAR_SUPPORT"), regime="RANGE", ctx=_ctx(_bars(5, 0.012)))
    assert rc.edge_source == "STRATEGY_CLAIM"
    assert rc.hard_block == "NO_EXECUTABLE_NET_EDGE"


def test_passed_signal_uses_strategy_projection(monkeypatch):
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=_ctx(_bars(30, 0.0004)))
    assert rc.edge_source == "STRATEGY_CLAIM"
    assert rc.expected_move_pct == 0.006
    assert rc.executable_edge["expected_move_pct"] == 0.0
    assert not rc.entry_eligible
    _learned_gross(monkeypatch, 0.003)
    rc = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=_ctx(_bars(30, 0.0004)))
    assert rc.entry_eligible


def test_equal_learned_edge_does_not_prefer_a_geometric_pass():
    ctx = _ctx(_bars(30, 0.012))
    soft = rank_setup_signal(_sig(passed=False, reason="NOT_NEAR_SUPPORT", setup="vwap_ema_reclaim"), regime="RANGE", ctx=ctx)
    good = rank_setup_signal(_sig(passed=True, reason=None, expected=0.006), regime="RANGE", ctx=ctx)
    from dataclasses import replace

    soft = replace(soft, rank_score=0.1, reachability_surplus=0.01)
    good = replace(good, rank_score=0.1, reachability_surplus=0.0)
    chosen = pick_best_ranked([soft, good])
    assert chosen.signal.passed is False
    assert chosen.signal.setup_name == "vwap_ema_reclaim"
