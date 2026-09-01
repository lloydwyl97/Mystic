"""Fill-path MFE/MAE identities for live DAY trades.

Mark MFE/MAE is evaluation-only. Executable MFE is net of a one-time exit
commission and must be compared to actual net exchange bps, not gross bps.

The actual exit fill is always a path observation. For every long:

    mark_mfe_bps >= actual_gross_fill_bps
"""

from __future__ import annotations


class PathMetricInvariantError(AssertionError):
    """Raised when a path metric contradicts the realized fill."""


def _bps(delta_px: float, entry_px: float) -> float:
    if entry_px <= 0:
        return 0.0
    return float(delta_px) / float(entry_px) * 10_000.0


def mark_mfe_mae_bps(
    *,
    entry_px: float,
    path_highs: list[float],
    path_lows: list[float],
    exit_px: float,
) -> tuple[float, float]:
    """Mark-price MFE/MAE. Includes the actual exit fill as a path print."""
    highs = [float(x) for x in path_highs if x is not None] + [float(exit_px)]
    lows = [float(x) for x in path_lows if x is not None] + [float(exit_px)]
    mfe = _bps(max(highs) - entry_px, entry_px)
    mae = _bps(min(lows) - entry_px, entry_px)
    realized_gross = _bps(float(exit_px) - entry_px, entry_px)
    if mfe + 1e-9 < realized_gross:
        raise PathMetricInvariantError(f"mark_mfe_bps {mfe} < realized_gross_bps {realized_gross}")
    return mfe, mae


def executable_net_mfe_bps(
    *,
    entry_px: float,
    path_bids: list[float],
    exit_fill_px: float,
    exit_commission_bps: float,
) -> float:
    """Best executable bid along the path, net of a one-time exit commission.

    The actual exit fill is included. Compare this to actual net exchange bps.
    Do not compare to gross fill bps.
    """
    bids = [float(x) for x in path_bids if x is not None] + [float(exit_fill_px)]
    best = max(bids)
    gross = _bps(best - entry_px, entry_px)
    return gross - float(exit_commission_bps)


def assert_long_fill_path_invariants(
    *,
    entry_px: float,
    exit_px: float,
    mark_mfe_bps: float,
    executable_net_mfe_bps_value: float | None = None,
    actual_net_exchange_bps: float | None = None,
) -> None:
    realized_gross = _bps(float(exit_px) - float(entry_px), float(entry_px))
    if mark_mfe_bps + 1e-9 < realized_gross:
        raise PathMetricInvariantError(f"mark_mfe_bps {mark_mfe_bps} < realized_gross_bps {realized_gross}")
    if executable_net_mfe_bps_value is not None and actual_net_exchange_bps is not None:
        if executable_net_mfe_bps_value + 1e-9 < float(actual_net_exchange_bps):
            raise PathMetricInvariantError(f"executable_net_mfe_bps {executable_net_mfe_bps_value} < actual_net_exchange_bps {actual_net_exchange_bps}")


def realized_capture_pct(*, realized_bps: float, mfe_bps: float) -> float:
    if abs(mfe_bps) < 1e-9:
        return 0.0
    return float(realized_bps) / float(mfe_bps) * 100.0
