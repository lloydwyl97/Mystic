"""DAY bucket quality — kill list and fat-tail gates."""

from __future__ import annotations

from backend.services.day_bucket_quality import (
    BucketMetrics,
    evaluate_bucket_entry,
    record_bucket_outcome,
)
from backend.services.day_regime_router import DAY_REGIME_NEUTRAL, evaluate_day_entry_route
from backend.services.day_trade_thesis import SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK, SETUP_VWAP_REVERSION


def test_neutral_blocks_breakout_and_pullback():
    for setup in (SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK):
        route = evaluate_day_entry_route(
            setup_type=setup,
            day_regime=DAY_REGIME_NEUTRAL,
            decision_data={"vwap": 100, "bb_position": 0.2, "rsi": 35, "mtf_json": '{"1h":{"ema_align":0.55},"4h":{"ema_align":0.52}}'},
            current_price=99.0,
            thesis_score=0.72,
        )
        assert route["allowed"] is False
        assert route["block_reason"] == "REGIME_ROUTE_NEUTRAL_MR_ONLY"


def test_neutral_allows_vwap_reclaim():
    route = evaluate_day_entry_route(
        setup_type=SETUP_VWAP_REVERSION,
        day_regime=DAY_REGIME_NEUTRAL,
        decision_data={
            "vwap": 100,
            "bb_position": 0.2,
            "rsi": 35,
            "thesis_target_level": 101.0,
            "mtf_json": '{"1h":{"ema_align":0.55},"4h":{"ema_align":0.52}}',
        },
        current_price=99.0,
        thesis_score=0.65,
    )
    assert route["allowed"] is True


def test_replay_killed_range_vwap_btc_is_advisory_not_blocking():
    """
    Historical bucket-quality kill lists are trade-opinion, not a real-time
    safety fact (final pre-push audit item 3) — they must shrink size/rank,
    never remove an otherwise-executable candidate outright.
    """
    from backend.services.day_bucket_quality import BUCKET_SIZE_FLOOR, REPLAY_KILLED_BUCKETS
    from backend.services.day_regime_router import DAY_REGIME_RANGE

    r = evaluate_bucket_entry(symbol="BTC/USDT", regime=DAY_REGIME_RANGE, setup=SETUP_VWAP_REVERSION)
    assert r["allowed"] is True
    assert r["block_reason"] == "BUCKET_KILL_REPLAY_RANGE_VWAP"
    assert r["bucket_size_factor"] == BUCKET_SIZE_FLOOR
    assert r["bucket_rank_delta"] < 0
    assert ("BTC/USDT", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION) in REPLAY_KILLED_BUCKETS


def test_global_kill_neutral_breakout_bucket_is_advisory_not_blocking():
    from backend.services.day_bucket_quality import BUCKET_SIZE_FLOOR

    r = evaluate_bucket_entry(symbol="BTC/USDT", regime="neutral", setup=SETUP_BREAKOUT_CONTINUATION)
    assert r["allowed"] is True
    assert r["block_reason"] == "BUCKET_KILL_REGIME_THESIS"
    assert r["bucket_size_factor"] == BUCKET_SIZE_FLOOR
    assert r["bucket_rank_delta"] < 0


def test_fat_tail_bucket_shrinks_size_but_still_allowed():
    from backend.services.day_bucket_quality import BUCKET_SIZE_FLOOR

    stats = {}
    for _ in range(5):
        record_bucket_outcome(
            stats,
            symbol="SOL/USDT",
            regime="neutral",
            setup=SETUP_VWAP_REVERSION,
            pnl_usd=-80,
            hold_sec=200 * 3600,
            mae_pct=-0.06,
            exit_reason="REPLAY_MARK",
        )
    r = evaluate_bucket_entry(
        symbol="SOL/USDT",
        regime="neutral",
        setup=SETUP_VWAP_REVERSION,
        bucket_stats=stats,
    )
    assert r["allowed"] is True
    assert "BUCKET_KILL" in r["block_reason"]
    assert r["bucket_size_factor"] == BUCKET_SIZE_FLOOR
    assert r["bucket_size_factor"] > 0.0, "size must shrink, never zero out to a de facto hard block"


def test_portfolio_engine_bucket_gate_never_disallows_a_candidate():
    """
    Engine-level check: _apply_bucket_quality_gate must never return
    allowed=False, even for a globally-killed regime/thesis bucket —
    otherwise BUCKET_QUALITY_BLOCKED would still function as a hidden
    permission gate at the call site in process_bar (which now only logs a
    BUCKET_QUALITY_ADVISORY and keeps the candidate routed).
    """
    import tempfile
    from pathlib import Path

    from backend.services.portfolio_engine import BuyCandidate, PortfolioEngine

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bucket_gate.db"
        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        engine._ensure_db_schema()

        candidate = BuyCandidate(
            symbol="BTC/USDT",
            confidence=0.6,
            trend_score=0.0,
            chop_score=0.0,
            coin_edge_score=0.0,
            volatility_penalty=0.0,
            spread_penalty=0.0,
            atr=100.0,
            current_price=64000.0,
            decision_data={
                "day_route_regime": "neutral",
                "setup_type": SETUP_BREAKOUT_CONTINUATION,
            },
        )
        result = engine._apply_bucket_quality_gate(candidate)
        assert result["allowed"] is True
        assert result["block_reason"] == "BUCKET_KILL_REGIME_THESIS"
        assert candidate.decision_data["bucket_allowed"] is True
