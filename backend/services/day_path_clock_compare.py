"""Legacy row-count vs clock-consistent feature comparison.

Uses already-inspected history only. Does not score locked outcomes.
Does not run day_path_net_v1 coefficients on clock-resampled inputs.
"""

from __future__ import annotations

import math
from typing import Any

from backend.services.binance_scalp.reconstructable_features import reconstructable_features
from backend.services.day_path_clock_features import build_clock_features, clip_asof, normalize_bars, parse_as_of
from backend.services.day_path_clock_v2 import SCHEMA_VERSION
from backend.services.day_path_input_validity import five_bar_return


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy)


def legacy_row_features(bars: Any, *, btc_ret_5: float = 0.0) -> dict[str, float]:
    norm = [{"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume, "ts": b.ts} for b in normalize_bars(bars)]
    return reconstructable_features(norm, btc_ret_5=btc_ret_5, ts=norm[-1]["ts"] if norm else None)


def compare_legacy_vs_clock(
    samples: list[dict[str, Any]],
    *,
    as_of_key: str = "as_of",
) -> dict[str, Any]:
    """Compare row-count ret_5 / realized_vol_10 to clock ret_5m / realized_vol_10m."""
    paired_ret: list[tuple[float, float]] = []
    paired_vol: list[tuple[float, float]] = []
    paired_btc: list[tuple[float, float]] = []
    legacy_available = 0
    clock_available = 0
    both = 0
    for sample in samples:
        as_of = parse_as_of(sample.get(as_of_key))
        if as_of is None:
            continue
        bars = clip_asof(normalize_bars(sample.get("bars")), as_of)
        if len(bars) < 8:
            continue
        legacy = legacy_row_features(bars, btc_ret_5=float(sample.get("legacy_btc_ret_5") or 0.0))
        clock = build_clock_features(
            bars,
            as_of=as_of,
            symbol=str(sample.get("symbol") or "BTCUSDT"),
            btc_bars=sample.get("btc_bars"),
        )
        legacy_available += 1
        if clock.get("feature_available"):
            clock_available += 1
        ret_legacy = five_bar_return([{"close": b.close, "ts": b.ts} for b in bars])
        if ret_legacy is None:
            ret_legacy = float(legacy.get("ret_5") or 0.0)
        ret_clock = clock.get("ret_5m")
        if ret_legacy is not None and ret_clock is not None:
            paired_ret.append((float(ret_legacy), float(ret_clock)))
            both += 1
        if clock.get("realized_vol_10m") is not None:
            paired_vol.append((float(legacy.get("realized_vol_10") or 0.0), float(clock["realized_vol_10m"])))
        if clock.get("btc_rel_ret_5m") is not None:
            paired_btc.append((float(legacy.get("rel_vs_btc_5") or 0.0), float(clock["btc_rel_ret_5m"])))

    def _shift(pairs: list[tuple[float, float]]) -> dict[str, float | None]:
        if not pairs:
            return {"n": 0, "mean_legacy": None, "mean_clock": None, "mean_abs_diff": None, "correlation": None}
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        return {
            "n": len(pairs),
            "mean_legacy": sum(xs) / len(xs),
            "mean_clock": sum(ys) / len(ys),
            "mean_abs_diff": sum(abs(a - b) for a, b in pairs) / len(pairs),
            "correlation": _corr(xs, ys),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "legacy_semantics": "row_count_slices_btc_ret_5_default_0",
        "clock_semantics": "point_in_time_clock_lookbacks_correct_btc_relative",
        "do_not_reuse_legacy_coefficients_on_clock_inputs": True,
        "samples": len(samples),
        "legacy_model_input_available": legacy_available,
        "clock_feature_available": clock_available,
        "paired_ret_coverage": both,
        "ret_5_vs_ret_5m": _shift(paired_ret),
        "realized_vol_10_vs_realized_vol_10m": _shift(paired_vol),
        "legacy_rel_vs_btc_5_vs_btc_rel_ret_5m": _shift(paired_btc),
    }


def refuse_legacy_coefficients_on_clock_features() -> dict[str, Any]:
    return {
        "allowed": False,
        "reason": "day_path_net_v1 coefficients belong to reconstructable_features_v1 row-count semantics",
        "schema_version": SCHEMA_VERSION,
    }
