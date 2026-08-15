"""Forward path labels. Targets only — never input features.

Answers: was there a tradable profitable excursion, even if the horizon closed red?
"""

from __future__ import annotations

from typing import Any

DEFAULT_COST = 0.0006
TARGET_PCT = 0.0025
HORIZONS_MIN = (1, 3, 5, 10, 20)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if out == out else default
    except (TypeError, ValueError):
        return default


def path_labels_for_horizon(
    mid0: float,
    future: list[dict[str, Any]],
    *,
    horizon_min: int,
    cost_pct: float = DEFAULT_COST,
    target_pct: float = TARGET_PCT,
) -> dict[str, Any]:
    """Labels for one horizon. Uses only bars inside that horizon."""
    empty = {
        "horizon_min": horizon_min,
        "terminal_gross": None,
        "terminal_net": None,
        "mfe": 0.0,
        "mae": 0.0,
        "executable_mfe_net": -cost_pct,
        "max_executable_net": -cost_pct,
        "time_to_mfe": None,
        "time_to_mae": None,
        "executable_profit_occurred": False,
        "profit_before_adverse": False,
        "target_reached": False,
        "time_to_target": None,
        "worst_dd_before_target": 0.0,
        "path_order": "NONE",
        "target_c": False,
        "target_d_net": None,
    }
    if mid0 <= 0 or not future:
        return empty
    window = future[: max(1, int(horizon_min))]
    mfe = 0.0
    mae = 0.0
    t_mfe = None
    t_mae = None
    t_target = None
    t_exec = None
    t_adverse = None
    worst_before_target = 0.0
    last_close = mid0
    for i, bar in enumerate(window, start=1):
        high = _f(bar.get("high"))
        low = _f(bar.get("low"))
        close = _f(bar.get("close"))
        if high > 0:
            fav = (high - mid0) / mid0
            if fav > mfe:
                mfe = fav
                t_mfe = i
            if t_target is None and fav >= target_pct:
                t_target = i
            if t_exec is None and fav > cost_pct:
                t_exec = i
        if low > 0:
            adv = (low - mid0) / mid0
            if adv < mae:
                mae = adv
                t_mae = i
            if t_adverse is None and adv <= -cost_pct:
                t_adverse = i
            if t_target is None:
                worst_before_target = min(worst_before_target, adv)
        if close > 0:
            last_close = close
    terminal_gross = (last_close - mid0) / mid0
    terminal_net = terminal_gross - cost_pct
    exec_mfe = mfe - cost_pct
    profit = exec_mfe > 0
    if t_mfe is not None and t_mae is not None:
        order = "MFE_FIRST" if t_mfe < t_mae else ("MAE_FIRST" if t_mae < t_mfe else "TIE")
    elif t_mfe is not None:
        order = "MFE_FIRST"
    elif t_mae is not None:
        order = "MAE_FIRST"
    else:
        order = "NONE"
    profit_before = bool(profit and t_exec is not None and (t_adverse is None or t_exec < t_adverse))
    # Target D: take first executable profit if it prints before adverse; else terminal net.
    if profit_before:
        target_d = exec_mfe
    else:
        target_d = terminal_net
    return {
        "horizon_min": horizon_min,
        "terminal_gross": terminal_gross,
        "terminal_net": terminal_net,
        "mfe": mfe,
        "mae": mae,
        "executable_mfe_net": exec_mfe,
        "max_executable_net": exec_mfe,
        "time_to_mfe": t_mfe,
        "time_to_mae": t_mae,
        "executable_profit_occurred": profit,
        "profit_before_adverse": profit_before,
        "target_reached": bool(t_target is not None),
        "time_to_target": t_target,
        "worst_dd_before_target": worst_before_target,
        "path_order": order,
        "target_c": bool(t_target is not None),
        "target_d_net": target_d,
        "mfe_first": order == "MFE_FIRST",
    }


def all_horizon_path_labels(
    mid0: float,
    future: list[dict[str, Any]],
    *,
    cost_pct: float = DEFAULT_COST,
    target_pct: float = TARGET_PCT,
    horizons: tuple[int, ...] | None = None,
) -> dict[int, dict[str, Any]]:
    use = horizons if horizons else HORIZONS_MIN
    return {h: path_labels_for_horizon(mid0, future, horizon_min=h, cost_pct=cost_pct, target_pct=target_pct) for h in use}
