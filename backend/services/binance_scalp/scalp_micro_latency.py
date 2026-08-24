"""Bounded latency histograms for SCALP event-engine diagnostics."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

_SAMPLES: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2000))


def record_latency(name: str, seconds: float) -> None:
    if seconds < 0:
        return
    _SAMPLES[name].append(float(seconds))


def timed(name: str):
    t0 = time.perf_counter()

    def done() -> float:
        dt = time.perf_counter() - t0
        record_latency(name, dt)
        return dt

    return done


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def latency_report() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, buf in _SAMPLES.items():
        xs = list(buf)
        out[name] = {
            "n": len(xs),
            "p50": round(percentile(xs, 50), 6),
            "p95": round(percentile(xs, 95), 6),
            "p99": round(percentile(xs, 99), 6),
        }
    return out


def reset_latency() -> None:
    _SAMPLES.clear()


__all__ = ["latency_report", "percentile", "record_latency", "reset_latency", "timed"]
