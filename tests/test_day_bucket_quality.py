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


def test_replay_killed_range_vwap_btc():
    from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS
    from backend.services.day_regime_router import DAY_REGIME_RANGE

    r = evaluate_bucket_entry(symbol="BTC/USDT", regime=DAY_REGIME_RANGE, setup=SETUP_VWAP_REVERSION)
    assert r["allowed"] is False
    assert r["block_reason"] == "BUCKET_KILL_REPLAY_RANGE_VWAP"
    assert ("BTC/USDT", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION) in REPLAY_KILLED_BUCKETS


def test_global_kill_blocks_neutral_breakout_bucket():
    r = evaluate_bucket_entry(symbol="BTC/USDT", regime="neutral", setup=SETUP_BREAKOUT_CONTINUATION)
    assert r["allowed"] is False
    assert r["block_reason"] == "BUCKET_KILL_REGIME_THESIS"


def test_fat_tail_bucket_killed_after_stats():
    stats = {}
    for _ in range(5):
        record_bucket_outcome(
            stats, symbol="SOL/USDT", regime="neutral", setup=SETUP_VWAP_REVERSION,
            pnl_usd=-80, hold_sec=200 * 3600, mae_pct=-0.06, exit_reason="REPLAY_MARK",
        )
    r = evaluate_bucket_entry(
        symbol="SOL/USDT", regime="neutral", setup=SETUP_VWAP_REVERSION, bucket_stats=stats,
    )
    assert r["allowed"] is False
    assert "BUCKET_KILL" in r["block_reason"]
