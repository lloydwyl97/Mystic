"""
Model calibration validation (item p12).

Mystic's live per-decision predicted probability (``prob_buy`` /
``confidence``, stored at entry time inside
``ai_outcome_training_rows.score_components_json``) is never checked against
what actually happened. This module closes that loop:

- Reads closed-trade rows per symbol, pairing each row's entry-time predicted
  probability against its own realized ``outcome_label`` (already denormalized
  onto the same row by ``ai_outcome_training_writer`` — no join required).
- Computes a real Brier score and Expected Calibration Error (ECE) over
  reliability buckets (50-55, 55-60, ..., 95-100 predicted-confidence %).
- Populates the ``calibration_brier`` / ``calibration_ece`` Prometheus gauges
  in ``backend/metrics.py`` — previously defined but never set anywhere in
  the repo (dead instrumentation).
- Persists a snapshot per symbol per cycle into ``ai_calibration_snapshots``
  for history/API access.
- Exposes ``calibration_confidence_multiplier()``: a continuous, honest
  degraded-state signal for how much a symbol's model confidence should be
  trusted right now. Per the core architecture rule this NEVER hard-blocks a
  trade — it only ever dampens (or leaves neutral) a confidence/sizing
  multiplier that an existing caller already applies. Degraded calibration is
  never converted into a trade veto; low-sample symbols are never punished
  (returned as neutral, not degraded).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

_BUCKET_EDGES: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01)
MIN_SAMPLES_FOR_CALIBRATION = int(os.getenv("CALIBRATION_MIN_SAMPLES", "20") or "20")


def _tracking_enabled() -> bool:
    return str(os.getenv("CALIBRATION_TRACKING_ENABLED", "true")).strip().lower() in ("1", "true", "yes", "on")


def _degraded_ece_threshold() -> float:
    return float(os.getenv("CALIBRATION_ECE_DEGRADED_THRESHOLD", "0.15") or "0.15")


def _degraded_brier_threshold() -> float:
    return float(os.getenv("CALIBRATION_BRIER_DEGRADED_THRESHOLD", "0.28") or "0.28")


def _degraded_confidence_multiplier() -> float:
    return float(os.getenv("CALIBRATION_DEGRADED_CONFIDENCE_MULT", "0.85") or "0.85")


@dataclass(frozen=True)
class ReliabilityBucket:
    low: float
    high: float
    count: int
    avg_predicted: float
    actual_win_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "count": self.count,
            "avg_predicted": round(self.avg_predicted, 4),
            "actual_win_rate": round(self.actual_win_rate, 4),
        }


@dataclass(frozen=True)
class CalibrationResult:
    symbol: str
    available: bool
    sample_count: int
    brier_score: float | None = None
    ece: float | None = None
    buckets: tuple[ReliabilityBucket, ...] = field(default_factory=tuple)
    degraded: bool = False
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "available": self.available,
            "sample_count": self.sample_count,
            "brier_score": round(self.brier_score, 6) if self.brier_score is not None else None,
            "ece": round(self.ece, 6) if self.ece is not None else None,
            "buckets": [b.to_dict() for b in self.buckets],
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }


def _extract_predicted_and_outcome(row: dict[str, Any]) -> tuple[float, float] | None:
    """From one ai_outcome_training_rows row, extract (predicted_prob, actual_win)
    if both are present and predicted_prob looks like a real probability."""
    raw_components = row.get("score_components_json")
    predicted: float | None = None
    if raw_components:
        try:
            components = json.loads(raw_components)
        except (TypeError, ValueError):
            components = {}
        for key in ("prob_buy", "confidence", "ai_confidence"):
            v = components.get(key) if isinstance(components, dict) else None
            if v is not None:
                try:
                    predicted = float(v)
                    break
                except (TypeError, ValueError):
                    continue
    if predicted is None:
        return None
    # Confidence is sometimes stored as 0-100; normalize to a 0-1 probability.
    if predicted > 1.0:
        predicted = predicted / 100.0
    if not (0.0 <= predicted <= 1.0):
        return None

    outcome_label = row.get("outcome_label")
    actual: float | None = None
    if outcome_label is not None:
        try:
            actual = 1.0 if float(outcome_label) > 0 else 0.0
        except (TypeError, ValueError):
            actual = None
    if actual is None:
        net_pnl_pct = row.get("net_pnl_pct")
        if net_pnl_pct is not None:
            try:
                actual = 1.0 if float(net_pnl_pct) > 0 else 0.0
            except (TypeError, ValueError):
                actual = None
    if actual is None:
        return None
    return predicted, actual


def compute_calibration_for_symbol(symbol: str, rows: list[dict[str, Any]]) -> CalibrationResult:
    """Pure aggregation — no I/O. `rows` are raw ai_outcome_training_rows dicts."""
    pairs: list[tuple[float, float]] = []
    for row in rows:
        extracted = _extract_predicted_and_outcome(row)
        if extracted is not None:
            pairs.append(extracted)

    if len(pairs) < MIN_SAMPLES_FOR_CALIBRATION:
        return CalibrationResult(
            symbol=symbol,
            available=False,
            sample_count=len(pairs),
            degraded=False,
            degraded_reason="insufficient_samples",
        )

    n = len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / n

    buckets: list[ReliabilityBucket] = []
    ece = 0.0
    for i in range(len(_BUCKET_EDGES) - 1):
        low, high = _BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]
        bucket_pairs = [(p, y) for p, y in pairs if low <= p < high]
        if not bucket_pairs:
            continue
        bcount = len(bucket_pairs)
        avg_pred = sum(p for p, _ in bucket_pairs) / bcount
        win_rate = sum(y for _, y in bucket_pairs) / bcount
        buckets.append(ReliabilityBucket(low=low, high=min(high, 1.0), count=bcount, avg_predicted=avg_pred, actual_win_rate=win_rate))
        ece += (bcount / n) * abs(avg_pred - win_rate)

    degraded = brier > _degraded_brier_threshold() or ece > _degraded_ece_threshold()
    degraded_reason = None
    if degraded:
        reasons = []
        if brier > _degraded_brier_threshold():
            reasons.append(f"brier={brier:.4f}>{_degraded_brier_threshold():.4f}")
        if ece > _degraded_ece_threshold():
            reasons.append(f"ece={ece:.4f}>{_degraded_ece_threshold():.4f}")
        degraded_reason = ",".join(reasons)

    return CalibrationResult(
        symbol=symbol,
        available=True,
        sample_count=n,
        brier_score=brier,
        ece=ece,
        buckets=tuple(buckets),
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


def _fetch_outcome_rows(symbol: str, db_path: str, limit: int) -> list[dict[str, Any]]:
    try:
        from backend.services.ai_canonical_storage import _symbol_variants_for_lookup

        sym_variants = _symbol_variants_for_lookup(symbol)
    except Exception:
        sym_variants = [symbol]
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ", ".join("?" for _ in sym_variants)
            cur = conn.execute(
                f"""
                SELECT outcome_label, net_pnl_pct, score_components_json
                FROM ai_outcome_training_rows
                WHERE symbol IN ({placeholders})
                ORDER BY closed_at_utc DESC
                LIMIT ?
                """,
                (*sym_variants, limit),
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        logger.debug("CALIBRATION_FETCH_FAILED symbol=%s: %s", symbol, exc)
        return []


def _persist_snapshot(result: CalibrationResult, db_path: str) -> None:
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.execute(
                """
                INSERT INTO ai_calibration_snapshots
                    (symbol, computed_at_utc, sample_count, brier_score, ece, available, degraded, degraded_reason, buckets_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.symbol,
                    datetime.now(timezone.utc).isoformat(),
                    result.sample_count,
                    result.brier_score,
                    result.ece,
                    1 if result.available else 0,
                    1 if result.degraded else 0,
                    result.degraded_reason,
                    json.dumps([b.to_dict() for b in result.buckets], separators=(",", ":")),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("CALIBRATION_PERSIST_FAILED symbol=%s: %s", result.symbol, exc)


def _set_prometheus_gauges(result: CalibrationResult) -> None:
    if not result.available:
        return
    try:
        from backend.metrics import calibration_brier, calibration_ece

        if result.brier_score is not None:
            calibration_brier.labels(symbol=result.symbol).set(result.brier_score)
        if result.ece is not None:
            calibration_ece.labels(symbol=result.symbol).set(result.ece)
    except Exception as exc:
        logger.debug("CALIBRATION_METRICS_SKIPPED symbol=%s: %s", result.symbol, exc)


def run_calibration_tracking_cycle(
    *,
    db_path: str = DATABASE_PATH,
    symbols: list[str] | None = None,
    lookback_rows: int = 300,
) -> dict[str, dict[str, Any]]:
    """Compute + persist + publish calibration for every symbol. Called
    periodically from start_ai_learning.py's core learning loop."""
    if not _tracking_enabled():
        return {}
    if symbols is None:
        try:
            from backend.config.trading_universe import TRADING_SYMBOLS

            symbols = list(TRADING_SYMBOLS)
        except Exception:
            symbols = []

    out: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        rows = _fetch_outcome_rows(sym, db_path, lookback_rows)
        result = compute_calibration_for_symbol(sym, rows)
        _persist_snapshot(result, db_path)
        _set_prometheus_gauges(result)
        out[sym] = result.to_dict()
        if result.available and result.degraded:
            logger.info("CALIBRATION_DEGRADED symbol=%s reason=%s n=%d", sym, result.degraded_reason, result.sample_count)
    return out


_MULT_CACHE: dict[str, tuple[float, float, str]] = {}  # symbol -> (mult, expiry_epoch, reason)
_MULT_CACHE_TTL_SEC = 300.0


def calibration_confidence_multiplier(symbol: str, *, db_path: str = DATABASE_PATH) -> tuple[float, str]:
    """Continuous confidence dampener — never a hard block. Reads the most
    recent persisted snapshot (cheap indexed lookup, TTL-cached in-process).
    Returns (1.0, reason) whenever data is unavailable/insufficient/disabled —
    the neutral, honest default — and only < 1.0 when calibration is
    genuinely measured as poor."""
    if not _tracking_enabled():
        return 1.0, "calibration_tracking_disabled"

    now = time.time()
    cached = _MULT_CACHE.get(symbol)
    if cached is not None and cached[1] > now:
        return cached[0], cached[2]

    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT available, degraded, degraded_reason, sample_count
                FROM ai_calibration_snapshots
                WHERE symbol = ?
                ORDER BY computed_at_utc DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
    except Exception as exc:
        logger.debug("CALIBRATION_MULT_LOOKUP_FAILED symbol=%s: %s", symbol, exc)
        row = None

    if row is None or not row["available"]:
        mult, reason = 1.0, "no_calibration_snapshot"
    elif row["degraded"]:
        mult, reason = _degraded_confidence_multiplier(), f"calibration_degraded:{row['degraded_reason']}"
    else:
        mult, reason = 1.0, "calibration_ok"

    _MULT_CACHE[symbol] = (mult, now + _MULT_CACHE_TTL_SEC, reason)
    return mult, reason


__all__ = [
    "CalibrationResult",
    "ReliabilityBucket",
    "calibration_confidence_multiplier",
    "compute_calibration_for_symbol",
    "run_calibration_tracking_cycle",
]
