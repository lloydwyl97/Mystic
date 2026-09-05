"""DAY path-net 1m input contract. Timing safety only.

Live authority may reject sparse/stale/gappy/unordered history.
It does not change feature semantics, coefficients, or OOD policy.

Contract day_path_input_contract_v1 is justified from the declared 1m
interval and inspected training-source spacing (median gap ~30s,
40-row span ~19.5 minutes, max gap < 80s). Bounds are engineering
envelopes, not P&L-tuned.

ROW COUNT is not CLOCK TIME. last-40 / ret_N are row counts that are
only meaningful on a coherent short-interval series.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PATH_INPUT_CONTRACT_VERSION = "day_path_input_contract_v1"
PATH_FEATURE_SCHEMA_VERSION = "reconstructable_features_v1"
DECLARED_INTERVAL_SEC = 60
LOOKBACK_ROWS = 40
MIN_LOOKBACK_ROWS = 40
MAX_LAST_BAR_AGE_SEC = 180
MAX_GAP_SEC = 180
MIN_LOOKBACK_SPAN_SEC = 12 * 60
MAX_LOOKBACK_SPAN_SEC = 50 * 60

PATH_INPUT_VALID = "PATH_INPUT_VALID"
PATH_INPUT_INVALID_STALE = "PATH_INPUT_INVALID_STALE"
PATH_INPUT_INVALID_SPARSE = "PATH_INPUT_INVALID_SPARSE"
PATH_INPUT_INVALID_GAP = "PATH_INPUT_INVALID_GAP"
PATH_INPUT_INVALID_SCHEMA = "PATH_INPUT_INVALID_SCHEMA"


def parse_bar_ts(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        dt = raw
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if val > 1e12:
            val /= 1000.0
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(raw).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _empty_telemetry(*, reason: str, row_count: int = 0) -> dict[str, Any]:
    return {
        "path_input_valid": False,
        "path_invalid_reason": reason,
        "path_input_contract_version": PATH_INPUT_CONTRACT_VERSION,
        "path_expected_bar_interval": DECLARED_INTERVAL_SEC,
        "path_row_count": row_count,
        "path_first_bar_ts": None,
        "path_last_bar_ts": None,
        "path_actual_lookback_seconds": None,
        "path_max_gap_seconds": None,
        "path_median_gap_seconds": None,
        "path_latest_bar_age_seconds": None,
        "path_missing_interval_count": None,
        "path_feature_schema_version": PATH_FEATURE_SCHEMA_VERSION,
    }


def validate_path_bars(
    bars: Any,
    *,
    as_of: datetime | None = None,
    min_rows: int = MIN_LOOKBACK_ROWS,
) -> dict[str, Any]:
    """Timing/schema validity. Does not invent prices or change features."""
    if not isinstance(bars, list) or len(bars) < min_rows:
        return _empty_telemetry(
            reason=PATH_INPUT_INVALID_SCHEMA,
            row_count=len(bars) if isinstance(bars, list) else 0,
        )
    parsed: list[datetime] = []
    for bar in bars:
        if not isinstance(bar, dict):
            return _empty_telemetry(reason=PATH_INPUT_INVALID_SCHEMA, row_count=len(bars))
        ts = parse_bar_ts(bar.get("ts"))
        if ts is None:
            return _empty_telemetry(reason=PATH_INPUT_INVALID_SCHEMA, row_count=len(bars))
        try:
            close = float(bar.get("close") or 0.0)
        except (TypeError, ValueError):
            return _empty_telemetry(reason=PATH_INPUT_INVALID_SCHEMA, row_count=len(bars))
        if close <= 0:
            return _empty_telemetry(reason=PATH_INPUT_INVALID_SCHEMA, row_count=len(bars))
        parsed.append(ts)
    for i in range(1, len(parsed)):
        if parsed[i] < parsed[i - 1]:
            return _empty_telemetry(reason=PATH_INPUT_INVALID_SCHEMA, row_count=len(parsed))
        if parsed[i] == parsed[i - 1]:
            return _empty_telemetry(reason=PATH_INPUT_INVALID_SCHEMA, row_count=len(parsed))
    gaps = [(parsed[i] - parsed[i - 1]).total_seconds() for i in range(1, len(parsed))]
    span_sec = (parsed[-1] - parsed[0]).total_seconds()
    as_of_dt = as_of or parsed[-1]
    if as_of_dt.tzinfo is None:
        as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
    as_of_dt = as_of_dt.astimezone(timezone.utc)
    last_age = (as_of_dt - parsed[-1]).total_seconds()
    unique_minutes = {t.replace(second=0, microsecond=0) for t in parsed}
    clock_minutes = int(span_sec // 60) + 1 if span_sec >= 0 else 0
    missing = max(0, clock_minutes - len(unique_minutes))
    tel = {
        "path_input_valid": False,
        "path_invalid_reason": None,
        "path_input_contract_version": PATH_INPUT_CONTRACT_VERSION,
        "path_expected_bar_interval": DECLARED_INTERVAL_SEC,
        "path_row_count": len(parsed),
        "path_first_bar_ts": parsed[0].isoformat(),
        "path_last_bar_ts": parsed[-1].isoformat(),
        "path_actual_lookback_seconds": span_sec,
        "path_max_gap_seconds": max(gaps) if gaps else 0.0,
        "path_median_gap_seconds": sorted(gaps)[len(gaps) // 2] if gaps else 0.0,
        "path_latest_bar_age_seconds": last_age,
        "path_missing_interval_count": missing,
        "path_feature_schema_version": PATH_FEATURE_SCHEMA_VERSION,
    }
    if last_age < 0:
        tel["path_invalid_reason"] = PATH_INPUT_INVALID_SCHEMA
        return tel
    if last_age > MAX_LAST_BAR_AGE_SEC:
        tel["path_invalid_reason"] = PATH_INPUT_INVALID_STALE
        return tel
    if gaps and max(gaps) > MAX_GAP_SEC:
        tel["path_invalid_reason"] = PATH_INPUT_INVALID_GAP
        return tel
    if span_sec > MAX_LOOKBACK_SPAN_SEC or span_sec < MIN_LOOKBACK_SPAN_SEC or missing > LOOKBACK_ROWS:
        tel["path_invalid_reason"] = PATH_INPUT_INVALID_SPARSE
        return tel
    tel["path_input_valid"] = True
    return tel


def five_bar_return(bars: list[dict[str, Any]]) -> float | None:
    if not isinstance(bars, list) or len(bars) < 6:
        return None
    try:
        cur = float(bars[-1].get("close") or 0.0)
        prev = float(bars[-6].get("close") or 0.0)
    except (TypeError, ValueError, AttributeError):
        return None
    if cur <= 0 or prev <= 0:
        return None
    return (cur - prev) / prev


def ood_telemetry(feats: dict[str, float], art: Any) -> dict[str, Any]:
    """Diagnostic only. Never used for live reject/rank."""
    names = list(getattr(art, "feature_names", None) or [])
    mean = list(getattr(art, "mean", None) or [])
    scale = list(getattr(art, "scale", None) or [])
    if not names or len(mean) != len(names) or len(scale) != len(names):
        return {
            "path_max_abs_z": None,
            "path_ood_feature_count_at_4": None,
            "path_ood_feature_count_at_6": None,
            "path_ood_feature_count_at_8": None,
            "path_outside_training_minmax_count": None,
        }
    zs: list[float] = []
    for i, name in enumerate(names):
        raw = float(feats.get(name) or 0.0)
        sc = float(scale[i])
        z = 0.0 if abs(sc) < 1e-12 else (raw - float(mean[i])) / sc
        zs.append(z)
    return {
        "path_max_abs_z": max(abs(z) for z in zs),
        "path_ood_feature_count_at_4": sum(1 for z in zs if abs(z) > 4.0),
        "path_ood_feature_count_at_6": sum(1 for z in zs if abs(z) > 6.0),
        "path_ood_feature_count_at_8": sum(1 for z in zs if abs(z) > 8.0),
        "path_outside_training_minmax_count": None,
    }
