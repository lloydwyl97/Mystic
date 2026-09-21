from backend.services.day_asof_4h import FOURH_SEC, FourHAsOfTracker, merge_seed_and_1m_completed
from backend.services.day_controlled_exits import evaluate_pre_buy_exit_consistency
from backend.services.portfolio_engine import get_coin_profile


def _t0() -> int:
    return (1_700_000_000 // FOURH_SEC) * FOURH_SEC


def test_tracker_never_uses_future_1m_in_forming_high():
    t0 = _t0()
    seed = [[float(t0 - FOURH_SEC), 100.0, 101.0, 99.0, 100.5, 0.0]]
    bars = [
        (t0, 100.5, 101.0, 100.0, 100.8),
        (t0 + 60, 100.8, 102.0, 100.7, 101.5),
        (t0 + 120, 101.5, 110.0, 101.0, 109.0),  # future vs mid eval
    ]
    tr = FourHAsOfTracker(bars_1m=bars, keep=80)
    tr.seed_completed(seed)
    mid = t0 + 90
    bundle = tr.advance(float(mid))
    tr.assert_as_of(float(mid), bundle)
    forming = bundle["4h"][-1]
    assert forming[0] == float(t0)
    assert forming[2] == 102.0
    assert forming[4] == 101.5
    assert forming[2] < 110.0


def test_completed_close_not_after_eval():
    t0 = _t0()
    seed = [[float(t0 - i * FOURH_SEC), 90.0 + i, 91.0 + i, 89.0 + i, 90.5 + i, 0.0] for i in range(60, 0, -1)]
    bars = [(t0 + i * 60, 100.0, 100.2, 99.8, 100.1) for i in range(10)]
    tr = FourHAsOfTracker(bars_1m=bars, keep=80)
    tr.seed_completed(seed)
    now = t0 + 300
    bundle = tr.advance(float(now))
    tr.assert_as_of(float(now), bundle)
    for r in bundle["4h"]:
        close_ts = r[0] + FOURH_SEC
        if close_ts > now + 1e-9:
            assert int(r[0]) == t0


def test_align_ready_after_80_seed_bars():
    t0 = _t0()
    seed = []
    px = 100.0
    for i in range(80):
        o = px
        c = px + 0.2
        seed.append([float(t0 - (80 - i) * FOURH_SEC), o, c + 0.05, o - 0.05, c, 0.0])
        px = c
    tr = FourHAsOfTracker(bars_1m=[], keep=80)
    tr.seed_completed(seed)
    bundle = tr.advance(float(t0 - 1))
    assert bundle["_asof"]["align_ready"] is True
    assert bundle["_asof"]["n_completed"] >= 50


def test_merge_seed_drops_unclosed_at_first_1m():
    t0 = _t0()
    seed = [
        [float(t0 - FOURH_SEC), 100.0, 101.0, 99.0, 100.5, 0.0],
        [float(t0), 100.5, 120.0, 100.0, 119.0, 0.0],
    ]
    bars = [(t0 + 60, 100.6, 100.7, 100.5, 100.65)]
    merged = merge_seed_and_1m_completed(seed, bars)
    assert len(merged) == 1
    assert merged[0][0] == float(t0 - FOURH_SEC)


def test_forming_volume_excludes_future_1m():
    t0 = _t0()
    seed = [[float(t0 - FOURH_SEC), 100.0, 101.0, 99.0, 100.5, 10.0]]
    bars = [
        (t0, 100.5, 101.0, 100.0, 100.8, 2.0),
        (t0 + 60, 100.8, 102.0, 100.7, 101.5, 3.0),
        (t0 + 120, 101.5, 110.0, 101.0, 109.0, 50.0),
    ]
    tr = FourHAsOfTracker(bars_1m=bars, keep=80)
    tr.seed_completed(seed)
    bundle = tr.advance(float(t0 + 90))
    forming = bundle["4h"][-1]
    assert forming[5] == 5.0


def test_prebuy_no_longer_blocks_on_broken_4h_with_seeded_align():
    """4H removed from trading authority (2026-09-17). Pre-buy no longer
    blocks on a broken 4H structure even with seeded alignment."""
    t0 = _t0()
    seed = []
    px = 90.0
    for i in range(60):
        o = px
        c = px + 1.0
        seed.append([float(t0 - (60 - i) * FOURH_SEC), o, c + 0.2, o - 0.5, c, 0.0])
        px = c
    prior_low = seed[-1][3]
    bars = [(t0, prior_low * 0.99, prior_low * 0.995, prior_low * 0.98, prior_low * 0.99)]
    tr = FourHAsOfTracker(bars_1m=bars, keep=80)
    tr.seed_completed(seed)
    now = float(t0 + 30)
    bundle = tr.advance(now)
    tr.assert_as_of(now, bundle)
    fill = prior_low * 0.99
    pre = evaluate_pre_buy_exit_consistency(
        setup="HTF_TREND_PULLBACK",
        entry_price=fill,
        stop_price=fill * 0.98,
        thesis_invalid_level=0.0,
        thesis_target_level=fill * 1.02,
        entry_vwap=fill,
        entry_ts=now,
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle=bundle,
        bar_ts=now,
    )
    assert pre.get("allowed") is True
    assert "STRUCTURE" not in str(pre.get("immediate_exit_reason") or pre.get("block_reason") or "")
