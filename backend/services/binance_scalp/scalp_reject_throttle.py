"""Throttle high-frequency scalp_rejects inserts — diagnostics only, not a trade gate."""

from __future__ import annotations

import os
import time
from typing import Any


def reject_log_interval_sec() -> float:
    raw = os.getenv("SCALP_REJECT_LOG_INTERVAL_SEC", "300")
    try:
        return max(30.0, float(raw))
    except (TypeError, ValueError):
        return 300.0


class ScalpRejectThrottle:
    """Skip duplicate (symbol, side, reason) reject rows within a TTL window."""

    def __init__(self, *, interval_sec: float | None = None) -> None:
        self.interval_sec = interval_sec if interval_sec is not None else reject_log_interval_sec()
        self._last: dict[tuple[str, str, str], float] = {}

    def should_log(self, symbol: str, side: str, reason: str, *, now: float | None = None) -> bool:
        key = (symbol.upper(), side.upper(), str(reason or ""))
        ts = now if now is not None else time.time()
        prev = self._last.get(key)
        if prev is not None and (ts - prev) < self.interval_sec:
            return False
        self._last[key] = ts
        if len(self._last) > 5000:
            cutoff = ts - self.interval_sec * 2
            self._last = {k: v for k, v in self._last.items() if v >= cutoff}
        return True

    def reset(self) -> None:
        self._last.clear()


def maybe_run_scalp_reject_retention(db_path: str, *, min_interval_sec: float = 3600.0) -> dict[str, Any] | None:
    """Run one bounded retention pass for scalp_rejects (at most once per hour per process)."""
    global _LAST_RETENTION_AT
    now = time.time()
    if now - _LAST_RETENTION_AT < min_interval_sec:
        return None
    try:
        from backend.services.sqlite_large_table_retention import run_large_table_retention

        out = run_large_table_retention(
            db_path,
            batch_size=2000,
            max_batches_per_table=200,
            max_run_seconds=30.0,
        )
        tables = out.get("tables") or {}
        tables.get("scalp_rejects") or {}
        _LAST_RETENTION_AT = now
        return out
    except Exception:
        return None


_LAST_RETENTION_AT = 0.0

__all__ = [
    "ScalpRejectThrottle",
    "maybe_run_scalp_reject_retention",
    "reject_log_interval_sec",
]
