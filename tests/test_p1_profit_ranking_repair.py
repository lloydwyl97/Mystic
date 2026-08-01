"""P1 profit/ranking repair regressions — selection, penalties, SCALP score band."""

from __future__ import annotations

from pathlib import Path

from backend.services.portfolio_engine import BuyCandidate


REPO = Path(__file__).resolve().parents[1]


def test_buy_candidate_rank_applies_signal_side_penalty():
    buy = BuyCandidate(
        symbol="BTC/USDT",
        confidence=0.70,
        trend_score=0.5,
        chop_score=0.5,
        coin_edge_score=0.5,
        volatility_penalty=0.0,
        spread_penalty=0.0,
        atr=100.0,
        current_price=50000.0,
        decision_data={"side": "buy", "quality_opinion_penalty": 0.0},
    )
    hold = BuyCandidate(
        symbol="BTC/USDT",
        confidence=0.70,
        trend_score=0.5,
        chop_score=0.5,
        coin_edge_score=0.5,
        volatility_penalty=0.0,
        spread_penalty=0.0,
        atr=100.0,
        current_price=50000.0,
        decision_data={"side": "hold", "quality_opinion_penalty": 4.0, "signal_side_penalty": 4.0},
    )
    assert hold.rank_score() < buy.rank_score()
    assert buy.rank_score() - hold.rank_score() >= 0.08


def test_buy_candidate_rank_applies_confidence_floor_soft_penalty():
    strong = BuyCandidate(
        symbol="ETH/USDT",
        confidence=0.72,
        trend_score=0.5,
        chop_score=0.5,
        coin_edge_score=0.5,
        volatility_penalty=0.0,
        spread_penalty=0.0,
        atr=10.0,
        current_price=3000.0,
        decision_data={},
    )
    weak = BuyCandidate(
        symbol="ETH/USDT",
        confidence=0.72,
        trend_score=0.5,
        chop_score=0.5,
        coin_edge_score=0.5,
        volatility_penalty=0.0,
        spread_penalty=0.0,
        atr=10.0,
        current_price=3000.0,
        decision_data={"confidence_floor_penalty": 0.12},
    )
    assert weak.rank_score() < strong.rank_score()


def test_decision_data_parsed_whitelist_includes_probs_and_penalties():
    src = (REPO / "backend/services/portfolio_engine_integration.py").read_text()
    assert '"prob_buy"' in src
    assert '"prob_hold"' in src
    assert '"prob_sell"' in src
    assert '"signal_side_penalty"' in src
    assert '"side"' in src
    # Must not wipe accumulated dd penalties with empty q_det alone.
    assert "_q_pen = float(dd.get(\"quality_opinion_penalty\")" in src
    # SELL must take the same non-BUY penalty path (not a pass-through).
    assert 'if not is_buy:' in src
    assert 'side_penalty = 8.0 if str(side).strip().lower() == "sell" else 4.0' in src


def test_candidate_replace_keeps_higher_score():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    assert "reason=higher_score_retained" in src
    assert "reason=latest_signal_wins" not in src


def test_position_persist_includes_status_and_dust():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    assert '("status", "TEXT DEFAULT \'ACTIVE\'")' in src
    assert "dust_detected_at" in src
    assert "dust_qty_canonical" in src
    assert "status, dust_detected_at, dust_qty_canonical, last_updated" in src.replace("\n", " ")


def test_paper_dust_persists_before_return():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    idx = src.find("PAPER_SELL_SKIP")
    assert idx > 0
    window = src[idx : idx + 500]
    assert "_persist_position_to_sqlite(position)" in window


def test_scalp_dead_strategies_score_in_tradeable_band():
    """Pass scores must clear SCALP_MIN_TRADEABLE_SCORE (1.45), matching working strategies."""
    modules = [
        "backend/services/binance_scalp/strategies/failed_breakout_reversal.py",
        "backend/services/binance_scalp/strategies/failed_breakdown_reversal.py",
        "backend/services/binance_scalp/strategies/compression_breakout.py",
        "backend/services/binance_scalp/strategies/trend_pullback_micro.py",
        "backend/services/binance_scalp/strategies/volume_impulse_continuation.py",
    ]
    for rel in modules:
        body = (REPO / rel).read_text()
        assert "score=0.5" not in body
        assert "score=0.6" not in body
        assert any(tok in body for tok in ("2.25", "2.35", "2.40", "2.45", "2.50"))


def test_failed_breakout_requires_up_momentum_reclaim():
    src = (REPO / "backend/services/binance_scalp/strategies/failed_breakout_reversal.py").read_text()
    assert "NO_FAILED_BREAKOUT_RECLAIM" in src
    assert "up_mom" in src
    assert "down_mom" not in src
