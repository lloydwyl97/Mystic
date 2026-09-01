"""MFE/MAE identities: exit fill is a path observation; no double-counted costs."""

import pytest

from backend.services.day_fill_path_metrics import (
    PathMetricInvariantError,
    assert_long_fill_path_invariants,
    executable_net_mfe_bps,
    mark_mfe_mae_bps,
    realized_capture_pct,
)


def test_mark_mfe_includes_exit_fill_when_bars_miss_it():
    """NET_PROFIT contradiction: 1m snapshots can be below the actual exit print."""
    entry = 100.0
    exit_px = 101.401  # +140.1 bps
    mfe, mae = mark_mfe_mae_bps(
        entry_px=entry,
        path_highs=[101.168],  # +116.8 bps snapshot high
        path_lows=[99.8],
        exit_px=exit_px,
    )
    realized = (exit_px - entry) / entry * 10_000
    assert mfe + 1e-9 >= realized
    assert mfe == pytest.approx(140.1, abs=0.05)
    assert mae < 0


def test_mark_mfe_rejects_below_realized_gross():
    with pytest.raises(PathMetricInvariantError):
        assert_long_fill_path_invariants(entry_px=100.0, exit_px=101.4, mark_mfe_bps=116.8)


def test_executable_net_mfe_compared_to_net_not_gross():
    entry = 100.0
    exit_px = 101.401
    exit_comm_bps = 2.0
    exe = executable_net_mfe_bps(
        entry_px=entry,
        path_bids=[101.168],
        exit_fill_px=exit_px,
        exit_commission_bps=exit_comm_bps,
    )
    actual_net = (exit_px - entry) / entry * 10_000 - 4.0  # 2 in + 2 out
    assert_long_fill_path_invariants(
        entry_px=entry,
        exit_px=exit_px,
        mark_mfe_bps=140.1,
        executable_net_mfe_bps_value=exe,
        actual_net_exchange_bps=actual_net,
    )
    assert exe + 1e-9 >= actual_net


def test_capture_uses_same_mfe_as_realized():
    cap = realized_capture_pct(realized_bps=140.1, mfe_bps=140.1)
    assert cap == pytest.approx(100.0)
