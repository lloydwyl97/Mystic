"""Rebuild DAY training-cache vectors point-in-time at each 4h label anchor.

Samples collected before the as-of fix kept forming 8h/12h/1d/1w bars whose
closes came from fetch time, which can sit inside the label horizon. Each sample
is rebuilt from completed exchange bars as of its anchor close with the same
builder and inputs the historical collector uses (no live order book, volume
profile, sentiment or ai_context). Labels and anchors are never changed.
"""

from __future__ import annotations

from typing import Any

from backend.config.canonical_candle_intervals import interval_ms
from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, fetch_limit_for_day_tf
from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145
from backend.services.day_active_market_bundle import completed_rows_asof, validate_day_active_bundle

CACHE_REBUILD_VERSION = "day_cache_asof_v1"
ANCHOR_WIDTH_MS = interval_ms("4h")
# Native-1m technical block and the sub-4h slopes never see a forming HTF bar.
LOWER_TF_DIMS: tuple[int, ...] = (124, 125, 126, 127, 128)


def anchor_asof_ms(anchor_open_ms: int) -> int:
    """Last millisecond of the 4h anchor bar (the historical collector's ``endTime``)."""
    return int(anchor_open_ms) + ANCHOR_WIDTH_MS - 1


def asof_bundle_from_history(history: dict[str, list[list]], end_ms: int) -> dict[str, list[list]]:
    """What ``async_fetch_day_active_ohlcv_bundle_asof`` returns at ``end_ms`` (ascending history)."""
    out: dict[str, list[list]] = {}
    for tf in DAY_ACTIVE_TIMEFRAMES:
        rows = [r for r in history.get(tf) or [] if int(r[0]) <= int(end_ms)]
        out[tf] = completed_rows_asof(tf, rows[-fetch_limit_for_day_tf(tf) :], end_ms)
    return out


def rebuild_cache_samples(samples: list[dict[str, Any]], history: dict[str, list[list]], symbol_ccxt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return rebuilt copies of ``samples`` plus a report; unbuildable samples are kept unchanged."""
    out: list[dict[str, Any]] = []
    report: dict[str, Any] = {"samples": len(samples), "rebuilt": 0, "kept": 0, "already": 0, "changed_dims": {}, "lower_tf_match": 0, "anchor_close_mismatch": 0, "failures": {}}
    changed: dict[int, int] = {}
    closes_4h = {int(r[0]): float(r[4]) for r in history.get("4h") or []}
    for sample in samples:
        row = dict(sample)
        if isinstance(row.get("htf_rebuild"), dict):
            report["already"] += 1
            out.append(row)
            continue
        anchor = row.get("label_anchor_4h_open_ms")
        if anchor is None:
            report["failures"]["no_anchor"] = report["failures"].get("no_anchor", 0) + 1
            report["kept"] += 1
            out.append(row)
            continue
        end_ms = anchor_asof_ms(int(anchor))
        bundle = asof_bundle_from_history(history, end_ms)
        ok, miss = validate_day_active_bundle(bundle)
        if not ok:
            key = (miss[0] if miss else "invalid").split("_bars_")[0]
            report["failures"][key] = report["failures"].get(key, 0) + 1
            report["kept"] += 1
            out.append(row)
            continue
        feats = build_day_htf_feature_vector_145(symbol_ccxt=symbol_ccxt, day_bundle=bundle, volume_profile=None, orderbook=None, sentiment=None, ai_context={})
        old = [float(x) for x in row.get("features") or []]
        if len(old) == len(feats):
            for i, (a, b) in enumerate(zip(old, feats, strict=True)):
                if abs(a - b) > 1e-12:
                    changed[i] = changed.get(i, 0) + 1
            if all(abs(old[i] - feats[i]) <= 1e-12 for i in LOWER_TF_DIMS):
                report["lower_tf_match"] += 1
        ref_close = closes_4h.get(int(anchor))
        if ref_close is not None and abs(ref_close - float(row.get("label_anchor_close") or 0.0)) > 1e-9 * max(1.0, ref_close):
            report["anchor_close_mismatch"] += 1
        row["features"] = [float(x) for x in feats]
        row["feature_count"] = len(feats)
        row["day_htf_bar_counts"] = {tf: len(bundle.get(tf) or []) for tf in DAY_ACTIVE_TIMEFRAMES}
        row["htf_rebuild"] = {"v": CACHE_REBUILD_VERSION, "asof_ms": end_ms}
        report["rebuilt"] += 1
        out.append(row)
    report["changed_dims"] = dict(sorted(changed.items()))
    return out, report


__all__ = [
    "ANCHOR_WIDTH_MS",
    "CACHE_REBUILD_VERSION",
    "LOWER_TF_DIMS",
    "anchor_asof_ms",
    "asof_bundle_from_history",
    "rebuild_cache_samples",
]
